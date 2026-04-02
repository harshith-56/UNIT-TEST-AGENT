from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

from context.event_context import EventContext
from context.token_budget import MAX_CONTEXT_TOKENS, estimate_tokens, trim_rules_to_budget


_SENTENCE_SPLIT_PATTERN = re.compile(r"[\r\n]+|(?<=[.!?])\s+")
_BULLET_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*")
_PR_SIGNAL_PATTERN = re.compile(
    r"\b(must|should|required|return|returns|error|errors|invalid|valid|reject|rejects|accept|accepts|raise|raises|throw|throws|non-empty|empty|null|none|boundary|fallback|default|only|never|always)\b|>=|<=|==|!=|\bat\s+least\b|\bat\s+most\b",
    re.IGNORECASE,
)
_PR_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(fixed bug|fix bug|updated code|minor changes?|misc(?:ellaneous)? changes?)\b",
        r"\b(refactor(?:ed|ing)?|cleanup|clean up|restructure(?:d)?|rename(?:d|ing)?|format(?:ting)?|lint(?:ing)?)\b",
        r"\b(improv(?:e|ed|ing) performance|optimization|optimized)\b",
        r"\b(documentation|docs?|comments?)\b",
        r"\b(tests? only|ci only|build only|dependency update|version bump)\b",
    )
]
_VAGUE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^be careful$",
        r"^best practices$",
        r"^handle edge cases$",
        r"^improve quality$",
        r"^etc\.?$",
        r"^do the right thing$",
        r"^fixed bug$",
        r"^minor changes?$",
        r"^updated code$",
    )
]
_PREFIX_FLUFF_PATTERN = re.compile(
    r"^(?:now|please|note that|ensure that|ensures that|this pr|this change|this update|for this change)\s+",
    re.IGNORECASE,
)
_MAX_PR_RULES = 8


@dataclass(frozen=True)
class StructuredContext:
    rules: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def combined_rules(self) -> list[str]:
        base_rules = [*self.constraints, *self.rules]

        file_filter_rule = (
                "Only generate tests for functions that contain meaningful logic. "
                "Skip functions that are trivial, such as those with only constants, "
                "simple returns, configuration, routing, or no branching. "
                "Focus only on functions with business logic, validation, "
                "transformations, or multiple execution paths."

        )

        return [file_filter_rule, *base_rules]


def parse_project_context(raw_context: str) -> StructuredContext:
    if not raw_context or not raw_context.strip():
        return StructuredContext(rules=[], constraints=[])

    try:
        parsed = yaml.safe_load(raw_context)
    except Exception:
        # Not valid YAML — treat entire string as a single rule
        return StructuredContext(rules=[raw_context.strip()], constraints=[])

    if not isinstance(parsed, dict):
        # Plain string or list — treat as rules
        if isinstance(parsed, list):
            return StructuredContext(rules=[str(r) for r in parsed], constraints=[])
        return StructuredContext(rules=[str(parsed)], constraints=[])

    rules = parsed.get("rules") or parsed.get("constraints") or []
    constraints = parsed.get("constraints") or []

    if isinstance(rules, str):
        rules = [rules]
    if isinstance(constraints, str):
        constraints = [constraints]

    return StructuredContext(
        rules=[str(r) for r in rules],
        constraints=[str(r) for r in constraints],
    )


def extract_pr_context(event_context: EventContext) -> StructuredContext:
    extracted_rules = extract_pr_rules(event_context.pull_request_title, event_context.pull_request_body)
    if not extracted_rules:
        return StructuredContext()

    constraints: list[str] = []
    rules: list[str] = []
    for rule in extracted_rules:
        target = constraints if _is_constraint_rule(rule) else rules
        if rule not in target:
            target.append(rule)
    return StructuredContext(rules=rules, constraints=constraints)


def extract_pr_rules(pr_title: str, pr_body: str) -> list[str]:
    raw_text = "\n".join(part for part in [pr_title, pr_body] if part.strip())
    if not raw_text.strip():
        return []

    extracted: list[str] = []
    seen: set[str] = set()
    for fragment in _SENTENCE_SPLIT_PATTERN.split(raw_text):
        cleaned_fragment = _clean_fragment(fragment)
        if not cleaned_fragment:
            continue
        for candidate in _expand_rule_candidates(cleaned_fragment):
            normalized = _normalize_pr_rule(candidate)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            extracted.append(normalized)
            if len(extracted) >= _MAX_PR_RULES:
                break
        if len(extracted) >= _MAX_PR_RULES:
            break

    return trim_rules_to_budget(extracted, MAX_CONTEXT_TOKENS)


def _validate_entries(entries: object, field_name: str) -> list[str]:
    if not isinstance(entries, list):
        raise RuntimeError(f"{field_name} must be a list of strings")
    cleaned: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise RuntimeError(f"{field_name} must contain only strings")
        normalized = _normalize_generic_rule(entry)
        if not normalized:
            continue
        if _is_vague(normalized):
            raise RuntimeError(f"{field_name} contains vague guidance: {entry}")
        cleaned.append(normalized)
    return cleaned


def _clean_fragment(text: str) -> str:
    normalized = _BULLET_PREFIX_PATTERN.sub("", text.strip())
    normalized = normalized.replace("\u2022", " ")
    normalized = re.sub(r"^\s*>\s*", "", normalized)
    normalized = re.sub(r"[`*_#]", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized[:240]


def _expand_rule_candidates(fragment: str) -> list[str]:
    if _is_pr_noise(fragment):
        return []
    if not _has_pr_signal(fragment):
        return []

    candidates = [fragment]
    if " and " in fragment.lower():
        parts = [part.strip(" ,;:-") for part in re.split(r"\s+and\s+", fragment, flags=re.IGNORECASE)]
        if len(parts) > 1 and all(_has_pr_signal(part) and re.search(r"[a-zA-Z]{3}", part) for part in parts):
            candidates = parts

    filtered: list[str] = []
    for candidate in candidates:
        if _is_pr_noise(candidate):
            continue
        if not _has_pr_signal(candidate):
            continue
        filtered.append(candidate)
    return filtered


def _normalize_pr_rule(text: str) -> str:
    normalized = _clean_fragment(text).lower()
    normalized = _PREFIX_FLUFF_PATTERN.sub("", normalized)
    normalized = normalized.rstrip(". ;:")
    normalized = re.sub(r"\breturns\b", "return", normalized)
    normalized = re.sub(r"\braises\b", "raise", normalized)
    normalized = re.sub(r"\bthrows\b", "throw", normalized)
    normalized = re.sub(r"\brejects\b", "reject", normalized)
    normalized = re.sub(r"\baccepts\b", "accept", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or _is_pr_noise(normalized) or _is_vague(normalized):
        return ""
    if not _has_pr_signal(normalized):
        return ""
    return normalized[:180]


def _normalize_generic_rule(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized[:240]


def _has_pr_signal(text: str) -> bool:
    return bool(_PR_SIGNAL_PATTERN.search(text))


def _is_pr_noise(text: str) -> bool:
    lowered = text.strip().lower()
    if len(lowered.split()) < 3 and not _has_pr_signal(lowered):
        return True
    return any(pattern.search(lowered) for pattern in _PR_NOISE_PATTERNS)


def _is_constraint_rule(text: str) -> bool:
    return bool(
        re.search(r"\b(must|should|required|only|never|always)\b|>=|<=|==|!=|\bat\s+least\b|\bat\s+most\b", text, re.IGNORECASE)
    )


def _is_vague(text: str) -> bool:
    if len(text.split()) < 3:
        return True
    return any(pattern.match(text) for pattern in _VAGUE_PATTERNS)

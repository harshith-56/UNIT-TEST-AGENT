from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

from context.token_budget import MAX_CONTEXT_TOKENS, estimate_tokens, trim_rules_to_budget
from context.event_context import EventContext


_SENTENCE_SPLIT_PATTERN = re.compile(r"[\r\n]+|(?<=[.!?])\s+")
_RELEVANT_PATTERN = re.compile(
    r"\b(must|should|require|constraint|validate|validation|invalid|return|returns|error|failure|non-empty|empty|null|none|reject|accept|only|never|always)\b",
    re.IGNORECASE,
)
_BEHAVIOR_PATTERN = re.compile(r"\b(return|returns|reject|accept|raise|throw|fail|fallback|default)\b", re.IGNORECASE)
_VAGUE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^be careful$",
        r"^best practices$",
        r"^handle edge cases$",
        r"^improve quality$",
        r"^etc\.?$",
        r"^do the right thing$",
    )
]


@dataclass(frozen=True)
class StructuredContext:
    rules: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def combined_rules(self) -> list[str]:
        return [*self.constraints, *self.rules]


def parse_project_context(raw_context: str) -> StructuredContext:
    if not raw_context.strip():
        return StructuredContext()

    loaded = yaml.safe_load(raw_context)
    if not isinstance(loaded, dict):
        raise RuntimeError("project_context must be structured YAML or JSON with rules/constraints lists")

    unsupported = set(loaded) - {"rules", "constraints"}
    if unsupported:
        raise RuntimeError(f"project_context contains unsupported keys: {sorted(unsupported)}")

    rules = _validate_entries(loaded.get("rules") or [], "project_context.rules")
    constraints = _validate_entries(loaded.get("constraints") or [], "project_context.constraints")

    total_tokens = estimate_tokens("\n".join([*rules, *constraints]))
    if total_tokens > MAX_CONTEXT_TOKENS:
        raise RuntimeError("project_context exceeds the 300 token limit")

    return StructuredContext(rules=rules, constraints=constraints)


def extract_pr_context(event_context: EventContext) -> StructuredContext:
    raw_text = "\n".join(part for part in [event_context.pull_request_title, event_context.pull_request_body] if part.strip())
    if not raw_text.strip():
        return StructuredContext()

    rules: list[str] = []
    constraints: list[str] = []
    for fragment in _SENTENCE_SPLIT_PATTERN.split(raw_text):
        normalized = _normalize_rule(fragment)
        if not normalized or not _RELEVANT_PATTERN.search(normalized):
            continue
        target = constraints if not _BEHAVIOR_PATTERN.search(normalized) else rules
        if normalized not in target:
            target.append(normalized)

    remaining_tokens = MAX_CONTEXT_TOKENS
    trimmed_constraints = trim_rules_to_budget(constraints, remaining_tokens)
    remaining_tokens -= estimate_tokens("\n".join(trimmed_constraints))
    trimmed_rules = trim_rules_to_budget(rules, max(0, remaining_tokens))
    return StructuredContext(rules=trimmed_rules, constraints=trimmed_constraints)


def _validate_entries(entries: object, field_name: str) -> list[str]:
    if not isinstance(entries, list):
        raise RuntimeError(f"{field_name} must be a list of strings")
    cleaned: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise RuntimeError(f"{field_name} must contain only strings")
        normalized = _normalize_rule(entry)
        if not normalized:
            continue
        if _is_vague(normalized):
            raise RuntimeError(f"{field_name} contains vague guidance: {entry}")
        cleaned.append(normalized)
    return cleaned


def _normalize_rule(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized[:240]


def _is_vague(text: str) -> bool:
    if len(text.split()) < 3:
        return True
    return any(pattern.match(text) for pattern in _VAGUE_PATTERNS)

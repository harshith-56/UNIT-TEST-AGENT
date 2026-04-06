from __future__ import annotations


MAX_INPUT_TOKENS = 20000
MAX_DEPENDENCIES = 20
MAX_CONTEXT_TOKENS = 4500


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def trim_rules_to_budget(rules: list[str], max_tokens: int) -> list[str]:
    kept: list[str] = []
    used_tokens = 0
    for rule in rules:
        rule_tokens = estimate_tokens(rule)
        if used_tokens + rule_tokens > max_tokens:
            break
        kept.append(rule)
        used_tokens += rule_tokens
    return kept

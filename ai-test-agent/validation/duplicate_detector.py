from __future__ import annotations

import re

from context.repo_context import GenerationContext
from generation.test_generator import GeneratedTest
from utils.logger import get_logger
from validation.test_naming import extract_test_names


LOGGER = get_logger(__name__)


def filter_duplicate_tests(generated_tests: list[GeneratedTest], generation_context: GenerationContext) -> list[GeneratedTest]:
    target_lookup = {
        (target.source_file, target.function_change.function_name): target
        for target in generation_context.targets
    }
    deduplicated: list[GeneratedTest] = []
    seen_normalized: set[str] = set()
    for generated_test in generated_tests:
        normalized = _normalize(generated_test.content)
        if normalized in seen_normalized:
            continue
        target = target_lookup.get((generated_test.source_file, generated_test.function_name))
        if target is None:
            continue
        if _is_duplicate(generated_test, target.existing_tests_text, target.existing_test_names):
            continue

        ok, reason = _check_generator_usage(generated_test.content)
        if not ok:
            LOGGER.warning(f"[SKIP][{generated_test.test_id}] {reason}")
            continue

        ok, reason = _check_function_call_args(
            generated_test.content,
            generated_test.function_name,
        )
        if not ok:
            LOGGER.warning(f"[SKIP][{generated_test.test_id}] {reason}")
            continue

        seen_normalized.add(normalized)
        deduplicated.append(generated_test)
    return deduplicated


def _is_duplicate(generated_test: GeneratedTest, existing_content: str, existing_test_names: list[str]) -> bool:
    normalized_generated = _normalize(generated_test.content)
    normalized_existing = _normalize(existing_content)
    if generated_test.generation_mode != "repair" and normalized_generated and normalized_generated in normalized_existing:
        return True

    generated_names = set(extract_test_names(generated_test.language, generated_test.content))
    if generated_test.generation_mode == "repair":
        return False
    return bool(generated_names & set(existing_test_names))


def _normalize(content: str) -> str:
    return "".join(content.split())


def _check_generator_usage(content: str) -> tuple[bool, str]:
    """
    Detects the pattern: result = some_func() followed immediately by result.method()
    without any next() or list() call in between.
    This catches tests that treat generator return values as plain objects.
    Returns (True, '') if safe, (False, reason) if generator misuse detected.
    """
    lines = [l for l in content.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        # Find: var = func() assignment
        assign_match = re.match(r"\s*(\w+)\s*=\s*(\w+)\s*\(", line)
        if not assign_match:
            continue
        var_name = assign_match.group(1)
        func_name = assign_match.group(2)

        # Skip if next() or list() is already wrapping the call
        if re.search(r"\b(next|list|tuple|set)\s*\(", line):
            continue

        # Look at the next 3 lines for var_name.method() usage
        for j in range(i + 1, min(i + 4, len(lines))):
            next_line = lines[j]
            if re.search(rf"\b{re.escape(var_name)}\.\w+\s*\(", next_line):
                # Check if there was a next()/list() call between i and j
                between = lines[i+1:j]
                if not any(re.search(r"\b(next|list|tuple)\s*\(", l) for l in between):
                    LOGGER.warning(
                        f"[generator_misuse] {func_name}() result used as object on line {j+1}"
                    )
                    return False, "generator_not_unwrapped"
    return True, ""


def _check_function_call_args(content: str, function_name: str) -> tuple[bool, str]:
    """
    High-confidence check: if a function comment/docstring near the call says
    'no arguments', 'takes no args', 'no parameters', but the test calls it
    with positional args, flag it.
    Also catches the common case: get_db("something") when function is get_db()
    by checking if the call appears with args but the function definition nearby
    shows zero parameters.
    """
    # Look for the function definition in the content (it may be in a comment or the test setup)
    # Pattern: function called with args
    calls_with_args = re.findall(
        rf"\b{re.escape(function_name)}\s*\(\s*([^)]+)\s*\)",
        content,
    )
    if not calls_with_args:
        return True, ""

    # Check if there's a nearby 'no arguments' hint
    no_arg_hints = re.search(
        r"(takes\s+no\s+(arg|param)|no\s+(arg|param)|zero\s+(arg|param)|\(\s*self\s*\)|\(\s*\))",
        content,
        re.IGNORECASE,
    )

    if calls_with_args and no_arg_hints:
        # Filter out mock setup lines — those are expected to reference the function name
        real_calls = [c for c in calls_with_args if c.strip() not in ("", "self")]
        if real_calls:
            LOGGER.warning(
                f"[wrong_args] {function_name}() called with args {real_calls} but hints suggest zero-arg"
            )
            return False, "wrong_arg_count"

    return True, ""

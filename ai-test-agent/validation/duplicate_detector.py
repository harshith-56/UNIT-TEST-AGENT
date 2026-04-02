from __future__ import annotations

from context.repo_context import GenerationContext
from generation.test_models import GeneratedTest
from validation.test_naming import extract_test_names


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
    return False


def _normalize(content: str) -> str:
    return "".join(content.split())




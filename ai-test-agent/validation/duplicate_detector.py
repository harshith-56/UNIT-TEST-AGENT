from __future__ import annotations

import ast
import re

from context.repo_context import GenerationContext
from generation.test_generator import GeneratedTest


JS_TEST_NAME_PATTERN = re.compile(r"""(?:it|test)\(\s*['"`](.*?)['"`]""")


def filter_duplicate_tests(generated_tests: list[GeneratedTest], generation_context: GenerationContext) -> list[GeneratedTest]:
    target_lookup = {target.source_file: target for target in generation_context.targets}
    deduplicated: list[GeneratedTest] = []
    seen_normalized: set[str] = set()
    for generated_test in generated_tests:
        normalized = _normalize(generated_test.content)
        if normalized in seen_normalized:
            continue
        existing_tests = target_lookup[generated_test.source_file].existing_tests
        if _is_duplicate(generated_test.language, generated_test.content, existing_tests):
            continue
        seen_normalized.add(normalized)
        deduplicated.append(generated_test)
    return deduplicated


def _is_duplicate(language: str, generated_content: str, existing_content: str) -> bool:
    normalized_generated = _normalize(generated_content)
    normalized_existing = _normalize(existing_content)
    if normalized_generated and normalized_generated in normalized_existing:
        return True
    generated_names = _extract_test_names(language, generated_content)
    existing_names = _extract_test_names(language, existing_content)
    return bool(generated_names & existing_names)


def _extract_test_names(language: str, content: str) -> set[str]:
    if language == "python":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return set()
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        }
    return {match.group(1).strip() for match in JS_TEST_NAME_PATTERN.finditer(content)}


def _normalize(content: str) -> str:
    return "".join(content.split())

from __future__ import annotations

import ast

import esprima
from tree_sitter_languages import get_parser

from generation.test_generator import GeneratedTest
from validation.test_naming import extract_test_names, has_duplicate_test_names


def validate_generated_tests(generated_tests: list[GeneratedTest]) -> tuple[list[GeneratedTest], list[GeneratedTest]]:
    valid: list[GeneratedTest] = []
    invalid: list[GeneratedTest] = []
    for generated_test in generated_tests:
        if _is_valid_generated_test(generated_test):
            valid.append(generated_test)
        else:
            invalid.append(generated_test)
    return valid, invalid


def _is_valid_generated_test(generated_test: GeneratedTest) -> bool:
    if not generated_test.content.strip():
        return False
    if has_duplicate_test_names(generated_test.language, generated_test.content):
        return False
    test_names = extract_test_names(generated_test.language, generated_test.content)
    if generated_test.generation_mode == "repair":
        if not test_names:
            return False
    elif not 3 <= len(test_names) <= 8:
        return False
    if any(not name.startswith(f"test_{generated_test.test_id}_") for name in test_names):
        return False
    return is_syntax_valid(generated_test.language, generated_test.content, generated_test.source_file)


def is_syntax_valid(language: str, content: str, source_file: str = "generated") -> bool:
    try:
        if language == "python":
            ast.parse(content)
            return True
        if language == "javascript":
            esprima.parseModule(content, {"jsx": source_file.endswith(".jsx")})
            return True
        parser = get_parser("tsx" if source_file.endswith(".tsx") else "typescript")
        tree = parser.parse(content.encode("utf-8"))
        return not _contains_error(tree.root_node)
    except Exception:
        return False


def _contains_error(node) -> bool:
    if node.type == "ERROR" or node.has_error:
        return True
    return any(_contains_error(child) for child in node.children)

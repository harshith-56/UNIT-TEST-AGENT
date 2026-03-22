from __future__ import annotations

import ast
import re

import esprima
from tree_sitter_languages import get_parser

from generation.test_generator import GeneratedTest
from validation.test_naming import extract_test_names, has_duplicate_test_names


BANNED_OUTPUT_PATTERNS = (
    re.compile(r"(?<![\w*])\*{3,}(?![\w*])"),
    re.compile(r"(?<![\w?])\?{3,}(?![\w?])"),
    re.compile(r"(?<!\.)\.\.\.(?!\.)"),
    re.compile(r"\b(?:todo|tbd)\b", re.IGNORECASE),
    re.compile(r"\byour_module\b"),
    re.compile(r"\bcreate_engine\s*\("),
    re.compile(r"\b(?:requests|httpx)\.(?:get|post|put|delete|patch)\s*\("),
    re.compile(r"\b(?:sqlite3|psycopg|psycopg2)\.connect\s*\("),
    re.compile(r"\bopen\s*\([^\n,]+,\s*['\"](?:w|a|x)"),
    re.compile(r"\b(?:Path|pathlib\.Path)\([^\n]*\)\.(?:write_text|write_bytes|open)\s*\("),
)


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
    if _contains_banned_output(generated_test.content):
        return False
    if has_duplicate_test_names(generated_test.language, generated_test.content):
        return False

    test_names = extract_test_names(generated_test.language, generated_test.content)
    if generated_test.generation_mode == "repair":
        if generated_test.repair_test_names:
            if set(test_names) != set(generated_test.repair_test_names):
                return False
        elif not 1 <= len(test_names) <= 8:
            return False
    elif not 3 <= len(test_names) <= 8:
        return False

    if any(not name.startswith(f"test_{generated_test.test_id}_") for name in test_names):
        return False
    return is_syntax_valid(generated_test.language, generated_test.content, generated_test.source_file)


def _contains_banned_output(content: str) -> bool:
    return any(pattern.search(content) for pattern in BANNED_OUTPUT_PATTERNS)


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

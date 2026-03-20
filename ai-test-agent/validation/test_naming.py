from __future__ import annotations

import ast
import re


JS_TEST_NAME_PATTERN = re.compile(r"(?:it|test)\(\s*['\"`](.*?)['\"`]")


def sanitize_test_identifier(function_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", function_name)
    return sanitized.strip("_") or "target"


def build_test_prefix(function_name: str) -> str:
    return f"test_{sanitize_test_identifier(function_name)}_"


def extract_test_names(language: str, content: str) -> list[str]:
    if language == "python":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        ]
        return sorted(dict.fromkeys(names))
    names = [match.group(1).strip() for match in JS_TEST_NAME_PATTERN.finditer(content)]
    return list(dict.fromkeys(name for name in names if name.startswith("test_")))


def has_duplicate_test_names(language: str, content: str) -> bool:
    names = extract_test_names(language, content)
    return len(names) != len(set(names))

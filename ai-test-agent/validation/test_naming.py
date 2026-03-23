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
    if language in ("javascript", "typescript"):
        names: list[str] = []

        # Pattern 1: test_/it_ prefixed function names
        for match in re.finditer(
            r"(?:^|\s)(?:async\s+)?(?:function\s+)?(test_\w+|it_\w+)\s*[=(]",
            content, re.MULTILINE,
        ):
            name = match.group(1)
            if name and name not in names:
                names.append(name)

        # Pattern 2: Jest/RTL style — it('desc', ...) or test('desc', ...)
        for match in re.finditer(
            r"^\s*(?:it|test)\s*\(\s*['\"]([^'\"]+)['\"]",
            content, re.MULTILINE,
        ):
            raw = match.group(1)
            normalized = "it_" + re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")
            if normalized not in names:
                names.append(normalized)

        # Pattern 3: describe block — only if no it/test found
        if not names:
            for match in re.finditer(
                r"^\s*describe\s*\(\s*['\"]([^'\"]+)['\"]",
                content, re.MULTILINE,
            ):
                raw = match.group(1)
                normalized = "describe_" + re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")
                if normalized not in names:
                    names.append(normalized)

        return names

    # Fallback for unknown language
    names_raw = [match.group(1).strip() for match in JS_TEST_NAME_PATTERN.finditer(content)]
    return list(dict.fromkeys(name for name in names_raw if name.startswith("test_")))


def has_duplicate_test_names(language: str, content: str) -> bool:
    names = extract_test_names(language, content)
    return len(names) != len(set(names))

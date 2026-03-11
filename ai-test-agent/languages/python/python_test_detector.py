from __future__ import annotations

import ast
from pathlib import Path


def extract_test_names(file_path: Path) -> set[str]:
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    test_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            test_names.add(node.name)
    return test_names

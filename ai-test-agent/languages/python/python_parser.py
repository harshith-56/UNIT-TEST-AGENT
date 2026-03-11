from __future__ import annotations

import ast
from pathlib import Path

from diff.diff_models import ChangedFunction


def parse_functions(file_path: Path) -> list[ChangedFunction]:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    functions: list[ChangedFunction] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append(_build_function(node, lines, self.class_stack))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            functions.append(_build_function(node, lines, self.class_stack))
            self.generic_visit(node)

    Visitor().visit(tree)
    return functions


def _build_function(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str], class_stack: list[str]) -> ChangedFunction:
    start_line = node.lineno
    end_line = node.end_lineno or node.lineno
    source_code = "\n".join(lines[start_line - 1 : end_line])
    function_name = ".".join([*class_stack, node.name]) if class_stack else node.name
    return ChangedFunction(
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
    )

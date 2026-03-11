from __future__ import annotations

from pathlib import Path
from typing import Any

import esprima

from diff.diff_models import ChangedFunction


def parse_functions(file_path: Path) -> list[ChangedFunction]:
    source = file_path.read_text(encoding="utf-8")
    program = esprima.parseModule(source, {"loc": True, "jsx": True, "tolerant": True})
    functions: list[ChangedFunction] = []
    _walk(program, source, functions, None)
    return functions


def _walk(node: Any, source: str, functions: list[ChangedFunction], current_class: str | None) -> None:
    if node is None:
        return

    node_type = getattr(node, "type", None)
    if node_type == "ClassDeclaration":
        class_name = getattr(getattr(node, "id", None), "name", None)
        for child in getattr(getattr(node, "body", None), "body", []) or []:
            _walk(child, source, functions, class_name)
        return

    if node_type == "FunctionDeclaration":
        name = getattr(getattr(node, "id", None), "name", None) or "anonymous"
        functions.append(_build_changed_function(name, node, source))
    elif node_type == "VariableDeclarator":
        init = getattr(node, "init", None)
        init_type = getattr(init, "type", None)
        if init_type in {"ArrowFunctionExpression", "FunctionExpression"}:
            identifier = getattr(getattr(node, "id", None), "name", None) or "anonymous"
            functions.append(_build_changed_function(identifier, init, source))
    elif node_type == "MethodDefinition":
        key = getattr(node, "key", None)
        method_name = getattr(key, "name", None) or getattr(key, "value", None) or "method"
        qualified_name = f"{current_class}.{method_name}" if current_class else method_name
        value = getattr(node, "value", None)
        if value is not None:
            functions.append(_build_changed_function(qualified_name, value, source))

    for child in _iter_children(node):
        _walk(child, source, functions, current_class)


def _iter_children(node: Any) -> list[Any]:
    children: list[Any] = []
    for value in vars(node).values():
        if isinstance(value, list):
            children.extend(item for item in value if hasattr(item, "type"))
        elif hasattr(value, "type"):
            children.append(value)
    return children


def _build_changed_function(name: str, node: Any, source: str) -> ChangedFunction:
    loc = getattr(node, "loc", None)
    if loc is None:
        return ChangedFunction(function_name=name, start_line=1, end_line=1, source_code=source)
    start_line = loc.start.line
    end_line = loc.end.line
    snippet = "\n".join(source.splitlines()[start_line - 1 : end_line])
    return ChangedFunction(
        function_name=name,
        start_line=start_line,
        end_line=end_line,
        source_code=snippet,
    )

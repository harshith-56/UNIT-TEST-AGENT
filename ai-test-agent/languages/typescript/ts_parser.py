from __future__ import annotations

from pathlib import Path

from tree_sitter_languages import get_parser

from diff.diff_models import ChangedFunction


def parse_functions(file_path: Path) -> list[ChangedFunction]:
    source_bytes = file_path.read_bytes()
    parser = get_parser("tsx" if file_path.suffix.lower() == ".tsx" else "typescript")
    tree = parser.parse(source_bytes)
    functions: list[ChangedFunction] = []
    _walk(tree.root_node, source_bytes, functions, None)
    return functions


def _walk(node, source_bytes: bytes, functions: list[ChangedFunction], current_class: str | None) -> None:
    node_type = node.type
    if node_type == "class_declaration":
        class_name = _field_text(node, "name", source_bytes) or current_class
        for child in node.children:
            _walk(child, source_bytes, functions, class_name)
        return

    if node_type == "function_declaration":
        name = _field_text(node, "name", source_bytes) or "anonymous"
        functions.append(_build_changed_function(name, node, source_bytes))
    elif node_type == "method_definition":
        name = _field_text(node, "name", source_bytes) or "method"
        qualified_name = f"{current_class}.{name}" if current_class else name
        functions.append(_build_changed_function(qualified_name, node, source_bytes))
    elif node_type == "variable_declarator":
        value = node.child_by_field_name("value")
        if value and value.type in {"arrow_function", "function_expression"}:
            name = _field_text(node, "name", source_bytes) or "anonymous"
            functions.append(_build_changed_function(name, node, source_bytes))

    for child in node.children:
        _walk(child, source_bytes, functions, current_class)


def _field_text(node, field_name: str, source_bytes: bytes) -> str | None:
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return source_bytes[child.start_byte : child.end_byte].decode("utf-8")


def _build_changed_function(name: str, node, source_bytes: bytes) -> ChangedFunction:
    snippet = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
    return ChangedFunction(
        function_name=name,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        source_code=snippet,
    )

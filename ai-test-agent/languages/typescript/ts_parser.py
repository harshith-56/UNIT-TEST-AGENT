from __future__ import annotations

import re
from pathlib import Path

from tree_sitter_languages import get_parser

from diff.diff_models import ParsedFunction


_CALL_PATTERN = re.compile(r"\b([A-Za-z_][\w\.]*)\s*\(")
_BRANCH_PATTERN = re.compile(r"\b(if|for|while|switch|catch|case)\b|\?")
_VALIDATION_PATTERN = re.compile(r"\b(validate|check|parse|assert|throw|error|required|invalid|empty|null|undefined)\b", re.IGNORECASE)


_FUNCTION_CONTAINER_TYPES = {
    "function_declaration",
    "method_definition",
    "arrow_function",
    "function_expression",
}


def parse_functions(file_path: Path, source_text: str | None = None) -> list[ParsedFunction]:
    suffix = file_path.suffix.lower()
    source = source_text if source_text is not None else file_path.read_text(encoding="utf-8")
    source_bytes = source.encode("utf-8")
    parser = get_parser("tsx" if suffix == ".tsx" else "typescript")
    tree = parser.parse(source_bytes)
    functions: list[ParsedFunction] = []

    for child in tree.root_node.children:
        node_type = child.type
        if node_type == "class_declaration":
            class_name = _field_text(child, "name", source_bytes) or "Class"
            class_source = _node_text(child, source_bytes)
            body = child.child_by_field_name("body")
            if body is None:
                continue
            for member in body.children:
                if member.type != "method_definition":
                    continue
                name = _field_text(member, "name", source_bytes) or "method"
                functions.append(
                    _build_function(
                        function_name=f"{class_name}.{name}",
                        node=member,
                        source_bytes=source_bytes,
                        context_code=class_source,
                        signature=_build_signature(name, member, source_bytes),
                        enclosing_class_name=class_name,
                    )
                )
            continue

        if node_type == "function_declaration":
            name = _field_text(child, "name", source_bytes) or "anonymous"
            functions.append(
                _build_function(
                    function_name=name,
                    node=child,
                    source_bytes=source_bytes,
                    context_code=_node_text(child, source_bytes),
                    signature=_build_signature(name, child, source_bytes),
                )
            )
            continue

        for declarator in _find_top_level_declarators(child):
            value = declarator.child_by_field_name("value")
            if value is None or value.type not in {"arrow_function", "function_expression"}:
                continue
            name = _field_text(declarator, "name", source_bytes) or "anonymous"
            functions.append(
                _build_function(
                    function_name=name,
                    node=declarator,
                    source_bytes=source_bytes,
                    context_code=_node_text(declarator, source_bytes),
                    signature=_build_signature(name, declarator, source_bytes),
                )
            )

    return functions


def _find_top_level_declarators(node) -> list:
    if node.type == "variable_declarator":
        return [node]
    if node.type in _FUNCTION_CONTAINER_TYPES:
        return []
    declarators = []
    for child in node.children:
        declarators.extend(_find_top_level_declarators(child))
    return declarators


def _build_function(
    function_name: str,
    node,
    source_bytes: bytes,
    context_code: str,
    signature: str,
    enclosing_class_name: str | None = None,
) -> ParsedFunction:
    source_code = _node_text(node, source_bytes)
    called_functions = _extract_called_functions(source_code)
    return ParsedFunction(
        function_name=function_name,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        source_code=source_code,
        context_code=context_code,
        signature=signature,
        enclosing_class_name=enclosing_class_name,
        called_functions=called_functions,
        branch_count=len(_BRANCH_PATTERN.findall(source_code)),
        has_validation=bool(_VALIDATION_PATTERN.search(source_code)) or any(
            any(token in call.lower() for token in ("validate", "check", "parse")) for call in called_functions
        ),
    )


def _build_signature(function_name: str, node, source_bytes: bytes) -> str:
    target = node.child_by_field_name("value") if node.type == "variable_declarator" else node
    parameters = target.child_by_field_name("parameters") if target is not None else None
    rendered_params = _node_text(parameters, source_bytes) if parameters is not None else "()"
    prefix = "async function" if target is not None and "async" in target.type else "function"
    return f"{prefix} {function_name}{rendered_params}"


def _field_text(node, field_name: str, source_bytes: bytes) -> str | None:
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return _node_text(child, source_bytes)


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _extract_called_functions(source_code: str) -> list[str]:
    seen: list[str] = []
    for match in _CALL_PATTERN.finditer(source_code):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "catch", "function"}:
            continue
        if name not in seen:
            seen.append(name)
    return seen

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import esprima

from diff.diff_models import ParsedFunction


_CALL_PATTERN = re.compile(r"\b([A-Za-z_][\w\.]*)\s*\(")
_BRANCH_PATTERN = re.compile(r"\b(if|for|while|switch|catch|case)\b|\?")
_VALIDATION_PATTERN = re.compile(r"\b(validate|check|parse|assert|throw|error|required|invalid|empty|null|undefined)\b", re.IGNORECASE)


def parse_functions(file_path: Path, source_text: str | None = None) -> list[ParsedFunction]:
    source = source_text if source_text is not None else file_path.read_text(encoding="utf-8")
    program = esprima.parseModule(source, {"loc": True, "jsx": True, "tolerant": True})
    functions: list[ParsedFunction] = []
    lines = source.splitlines()

    for statement in getattr(program, "body", []) or []:
        node_type = getattr(statement, "type", None)
        if node_type == "ClassDeclaration":
            class_name = getattr(getattr(statement, "id", None), "name", None)
            class_source = _slice_lines(lines, statement.loc.start.line, statement.loc.end.line)
            for member in getattr(getattr(statement, "body", None), "body", []) or []:
                if getattr(member, "type", None) != "MethodDefinition":
                    continue
                method_name = getattr(getattr(member, "key", None), "name", None) or getattr(getattr(member, "key", None), "value", None) or "method"
                value = getattr(member, "value", None)
                if value is None or getattr(value, "loc", None) is None:
                    continue
                source_code = _slice_lines(lines, value.loc.start.line, value.loc.end.line)
                functions.append(
                    _build_function(
                        function_name=f"{class_name}.{method_name}" if class_name else method_name,
                        source_code=source_code,
                        context_code=class_source,
                        start_line=value.loc.start.line,
                        end_line=value.loc.end.line,
                        params=getattr(value, "params", []) or [],
                        is_async=bool(getattr(value, "isAsync", False) or getattr(value, "async", False)),
                        enclosing_class_name=class_name,
                    )
                )
            continue

        if node_type == "FunctionDeclaration":
            name = getattr(getattr(statement, "id", None), "name", None) or "anonymous"
            functions.append(_build_from_node(statement, lines, name))
            continue

        if node_type == "VariableDeclaration":
            for declarator in getattr(statement, "declarations", []) or []:
                init = getattr(declarator, "init", None)
                init_type = getattr(init, "type", None)
                if init_type not in {"ArrowFunctionExpression", "FunctionExpression"} or getattr(init, "loc", None) is None:
                    continue
                name = getattr(getattr(declarator, "id", None), "name", None) or "anonymous"
                functions.append(_build_from_node(init, lines, name))

    return functions


def _build_from_node(node: Any, lines: list[str], name: str) -> ParsedFunction:
    source_code = _slice_lines(lines, node.loc.start.line, node.loc.end.line)
    return _build_function(
        function_name=name,
        source_code=source_code,
        context_code=source_code,
        start_line=node.loc.start.line,
        end_line=node.loc.end.line,
        params=getattr(node, "params", []) or [],
        is_async=bool(getattr(node, "isAsync", False) or getattr(node, "async", False)),
    )


def _build_function(
    function_name: str,
    source_code: str,
    context_code: str,
    start_line: int,
    end_line: int,
    params: list[Any],
    is_async: bool,
    enclosing_class_name: str | None = None,
) -> ParsedFunction:
    called_functions = _extract_called_functions(source_code)
    return ParsedFunction(
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        context_code=context_code,
        signature=_build_signature(function_name, params, is_async),
        enclosing_class_name=enclosing_class_name,
        called_functions=called_functions,
        branch_count=len(_BRANCH_PATTERN.findall(source_code)),
        has_validation=bool(_VALIDATION_PATTERN.search(source_code)) or any(
            any(token in call.lower() for token in ("validate", "check", "parse")) for call in called_functions
        ),
    )


def _build_signature(function_name: str, params: list[Any], is_async: bool) -> str:
    rendered_params = ", ".join(_param_text(parameter) for parameter in params)
    prefix = "async function" if is_async else "function"
    return f"{prefix} {function_name}({rendered_params})"


def _param_text(parameter: Any) -> str:
    node_type = getattr(parameter, "type", None)
    if node_type == "Identifier":
        return getattr(parameter, "name", "arg")
    if node_type == "AssignmentPattern":
        left = _param_text(getattr(parameter, "left", None))
        return f"{left}=..."
    if node_type == "RestElement":
        argument = _param_text(getattr(parameter, "argument", None))
        return f"...{argument}"
    if node_type == "ObjectPattern":
        return "{...}"
    if node_type == "ArrayPattern":
        return "[...]"
    return "arg"


def _extract_called_functions(source_code: str) -> list[str]:
    seen: list[str] = []
    for match in _CALL_PATTERN.finditer(source_code):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "catch", "function"}:
            continue
        if name not in seen:
            seen.append(name)
    return seen


def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])

from __future__ import annotations

import ast
import re
from pathlib import Path

from diff.diff_models import ParsedFunction


_VALIDATION_PATTERN = re.compile(r"\b(validate|check|parse|assert|raise|required|invalid|empty|none)\b", re.IGNORECASE)


class _FunctionMetadataVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.called_functions: list[str] = []
        self.branch_count = 0
        self.has_validation = False
        self._function_depth = 0

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self.called_functions.append(name)
            if any(token in name.lower() for token in ("validate", "check", "parse")):
                self.has_validation = True
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.has_validation = True
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.has_validation = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(
            isinstance(operand, ast.Constant) and operand.value is None
            for operand in [node.left, *node.comparators]
        ):
            self.has_validation = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth > 0:
            return
        self._function_depth += 1
        for decorator in node.decorator_list:
            self.visit(decorator)
        for statement in node.body:
            self.visit(statement)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth > 0:
            return
        self._function_depth += 1
        for decorator in node.decorator_list:
            self.visit(decorator)
        for statement in node.body:
            self.visit(statement)
        self._function_depth -= 1


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, source: str, lines: list[str]) -> None:
        self.source = source
        self.lines = lines
        self.functions: list[ParsedFunction] = []
        self.class_stack: list[tuple[str, ast.ClassDef]] = []
        self.function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append((node.name, node))
        for child in node.body:
            self.visit(child)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth > 0:
            return
        self.function_depth += 1
        metadata = _collect_metadata(node, self.source)
        class_name = self.class_stack[-1][0] if self.class_stack else None
        class_node = self.class_stack[-1][1] if self.class_stack else None
        source_code = _slice_source(self.lines, node.lineno, node.end_lineno or node.lineno)
        if class_node is not None:
            context_code = _slice_source(self.lines, class_node.lineno, class_node.end_lineno or class_node.lineno)
            function_name = f"{class_name}.{node.name}"
        else:
            context_code = source_code
            function_name = node.name
        self.functions.append(
            ParsedFunction(
                function_name=function_name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                source_code=source_code,
                context_code=context_code,
                signature=_build_signature(node),
                enclosing_class_name=class_name,
                called_functions=metadata.called_functions,
                branch_count=metadata.branch_count,
                has_validation=metadata.has_validation or bool(_VALIDATION_PATTERN.search(source_code)),
            )
        )
        self.function_depth -= 1


def parse_functions(file_path: Path, source_text: str | None = None) -> list[ParsedFunction]:
    source = source_text if source_text is not None else file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    collector = _FunctionCollector(source, lines)
    collector.visit(tree)
    return collector.functions


def _collect_metadata(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> _FunctionMetadataVisitor:
    visitor = _FunctionMetadataVisitor()
    visitor.visit(node)
    return visitor


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [_format_arg(argument) for argument in node.args.posonlyargs]
    if node.args.posonlyargs:
        args.append("/")
    args.extend(_format_arg(argument) for argument in node.args.args)
    if node.args.vararg is not None:
        args.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        args.append("*")
    args.extend(_format_arg(argument) for argument in node.args.kwonlyargs)
    if node.args.kwarg is not None:
        args.append(f"**{node.args.kwarg.arg}")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


def _format_arg(argument: ast.arg) -> str:
    if argument.annotation is None:
        return argument.arg
    annotation = ast.unparse(argument.annotation)
    return f"{argument.arg}: {annotation}"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _slice_source(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])

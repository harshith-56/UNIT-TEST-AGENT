from __future__ import annotations

import ast
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import esprima
from tree_sitter_languages import get_parser

from agent.config import detect_language
from diff.diff_models import (
    CHANGE_TYPE_DELETION,
    CHANGE_TYPE_LOGIC,
    CHANGE_TYPE_RENAME,
    CHANGE_TYPE_SIGNATURE,
    ChangedFile,
    FunctionChange,
    ParsedFunction,
)
from languages.javascript.js_parser import parse_functions as parse_js_functions
from languages.python.python_parser import parse_functions as parse_python_functions
from languages.typescript.ts_parser import parse_functions as parse_ts_functions
from utils.git_utils import get_changed_files, get_diff, get_file_content_at_ref


HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_VALIDATION_HINT_PATTERN = re.compile(r"\b(validate|check|parse|required|invalid|empty|none|null|raise|throw)\b", re.IGNORECASE)
_CONFIG_FILE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|/)dockerfile$",
        r"(^|/)\.env(?:\.[^/]+)?$",
        r"(^|/)(?:requirements|constraints)\.txt$",
        r"(^|/)package(?:-lock)?\.json$",
        r"(^|/)pnpm-lock\.yaml$",
        r"(^|/)yarn\.lock$",
        r"(^|/).*\.mdx?$",
        r"(^|/).*\.config\.(?:py|js|jsx|ts|tsx)$",
        r"(^|/)(?:jest|webpack|vite|rollup|babel|eslint|prettier|tsup|vitest|tailwind|postcss|next|nuxt)\.config\.(?:js|ts|mjs|cjs)$",
    )
]
_CONFIG_NAME_KEYWORDS = ("config", "settings", "constants", "env", "setup", "build")
_TRIVIAL_CALL_WHITELIST = {
    "bool",
    "dict",
    "float",
    "frozenset",
    "int",
    "len",
    "list",
    "set",
    "sorted",
    "str",
    "tuple",
    "Object",
    "Array",
    "String",
    "Number",
    "Boolean",
}
_COMPLEX_LINE_PATTERN = re.compile(r"\b(await|yield|raise|throw|try|except|catch|switch|case)\b|&&|\|\||\?")
_SIMPLE_STATEMENT_PATTERN = re.compile(
    r"^(?:return\b.*|pass$|(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=.*|[A-Za-z_$][\w$\.\[\]]*\s*=.*|[{}]$)$"
)
_JS_LINE_COMMENT_PATTERN = re.compile(r"//.*?(?=\r?$)", re.MULTILINE)
_JS_BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
_JS_STRING_PATTERN = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`", re.DOTALL)

def analyze_diff(repo_root: Path, base_branch: str) -> list[ChangedFile]:
    diff_text = get_diff(repo_root, base_branch)
    changed_file_paths = get_changed_files(repo_root, base_branch)
    diff_by_file = _split_diff_by_file(diff_text)
    changed_files: list[ChangedFile] = []

    for file_path in changed_file_paths:
        language = detect_language(file_path)
        if not language or _is_test_or_generated_path(file_path) or _is_ignored_non_source_path(file_path):
            continue

        current_path = repo_root / file_path
        current_source = current_path.read_text(encoding="utf-8") if current_path.exists() else ""
        previous_source = get_file_content_at_ref(repo_root, f"origin/{base_branch}", file_path)
        changed_lines, removed_lines = _extract_changed_line_sets(diff_by_file.get(file_path, ""))

        current_functions = _parse_functions(language, current_path, current_source) if current_source else []
        previous_functions = _parse_functions(language, current_path, previous_source) if previous_source else []
        candidate_functions = current_functions or previous_functions
        if not candidate_functions:
            continue
        if is_config_like_file(file_path, candidate_functions):
            continue

        function_changes = _classify_function_changes(
            language,
            current_functions,
            previous_functions,
            changed_lines,
            removed_lines,
        )
        if not function_changes:
            continue

        changed_files.append(
            ChangedFile(
                file_path=file_path,
                language=language,
                current_source=current_source,
                previous_source=previous_source,
                changed_lines=changed_lines,
                removed_lines=removed_lines,
                function_changes=function_changes,
            )
        )

    return changed_files


def should_generate_tests(function_change: FunctionChange) -> bool:
    return (
        bool(function_change.function_name)
        and function_change.change_type not in {CHANGE_TYPE_RENAME, CHANGE_TYPE_DELETION}
        and function_change.has_behavioral_change
        and not function_change.should_skip_generation
    )


def has_behavioral_change(language: str, current_function: ParsedFunction, previous_function: ParsedFunction) -> bool:
    current_normalized = _normalize_function_source(language, current_function)
    previous_normalized = _normalize_function_source(language, previous_function)
    return current_normalized != previous_normalized


def is_config_like_file(file_path: str, functions: list[ParsedFunction]) -> bool:
    normalized_name = Path(file_path).stem.lower()
    if any(keyword in normalized_name for keyword in _CONFIG_NAME_KEYWORDS):
        return True
    return bool(functions) and all(_function_is_trivial(function) for function in functions)


def _split_diff_by_file(diff_text: str) -> dict[str, str]:
    chunks: dict[str, list[str]] = defaultdict(list)
    current_file: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_file = None
        elif line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/").strip()
            chunks[current_file]
        elif current_file:
            chunks[current_file].append(line)
    return {file_path: "\n".join(lines) for file_path, lines in chunks.items()}


def _extract_changed_line_sets(file_diff: str) -> tuple[list[int], list[int]]:
    added_lines: list[int] = []
    removed_lines: list[int] = []
    for old_start_text, old_length_text, new_start_text, new_length_text in HUNK_PATTERN.findall(file_diff):
        old_start = int(old_start_text)
        old_length = int(old_length_text or "1")
        new_start = int(new_start_text)
        new_length = int(new_length_text or "1")
        if new_length > 0:
            added_lines.extend(range(new_start, new_start + new_length))
        if old_length > 0:
            removed_lines.extend(range(old_start, old_start + old_length))
    return sorted(set(added_lines)), sorted(set(removed_lines))


def _classify_function_changes(
    language: str,
    current_functions: list[ParsedFunction],
    previous_functions: list[ParsedFunction],
    changed_lines: list[int],
    removed_lines: list[int],
) -> list[FunctionChange]:
    impacted_current = _impacted_functions(current_functions, changed_lines)
    impacted_previous = _impacted_functions(previous_functions, removed_lines)
    previous_by_name = {function.function_name: function for function in impacted_previous}
    matched_previous: set[str] = set()
    function_changes: list[FunctionChange] = []

    for current_function in impacted_current:
        previous_function = previous_by_name.get(current_function.function_name)
        if previous_function is not None:
            matched_previous.add(previous_function.function_name)
            change_type = CHANGE_TYPE_SIGNATURE if current_function.signature != previous_function.signature else CHANGE_TYPE_LOGIC
            function_changes.append(_build_function_change(language, current_function, change_type, previous_function))
            continue

        renamed_from = _match_renamed_function(current_function, impacted_previous, matched_previous)
        if renamed_from is not None:
            matched_previous.add(renamed_from.function_name)
            function_changes.append(_build_function_change(language, current_function, CHANGE_TYPE_RENAME, renamed_from))
            continue

        function_changes.append(_build_function_change(language, current_function, CHANGE_TYPE_LOGIC, None))

    for previous_function in impacted_previous:
        if previous_function.function_name in matched_previous:
            continue
        function_changes.append(
            FunctionChange(
                function_name=previous_function.function_name,
                start_line=previous_function.start_line,
                end_line=previous_function.end_line,
                source_code=previous_function.source_code,
                context_code=previous_function.context_code,
                signature=previous_function.signature,
                change_type=CHANGE_TYPE_DELETION,
                previous_name=previous_function.function_name,
                previous_signature=previous_function.signature,
                previous_source_code=previous_function.source_code,
                previous_context_code=previous_function.context_code,
                enclosing_class_name=previous_function.enclosing_class_name,
                called_functions=previous_function.called_functions,
                branch_count=previous_function.branch_count,
                has_validation=previous_function.has_validation,
                has_behavioral_change=True,
            )
        )

    return function_changes


def _impacted_functions(functions: list[ParsedFunction], changed_lines: list[int]) -> list[ParsedFunction]:
    if not changed_lines:
        return functions
    return [
        function
        for function in functions
        if any(function.start_line <= line <= function.end_line for line in changed_lines)
    ]


def _match_renamed_function(
    current_function: ParsedFunction,
    previous_functions: list[ParsedFunction],
    matched_previous: set[str],
) -> ParsedFunction | None:
    best_match: tuple[float, ParsedFunction] | None = None
    for candidate in previous_functions:
        if candidate.function_name in matched_previous:
            continue
        if candidate.enclosing_class_name != current_function.enclosing_class_name:
            continue
        similarity = _similarity_score(current_function, candidate)
        if similarity < 0.58:
            continue
        if best_match is None or similarity > best_match[0]:
            best_match = (similarity, candidate)
    return best_match[1] if best_match is not None else None


def _similarity_score(current_function: ParsedFunction, previous_function: ParsedFunction) -> float:
    current_body = _normalized_source(current_function.source_code, current_function.function_name)
    previous_body = _normalized_source(previous_function.source_code, previous_function.function_name)
    body_score = difflib.SequenceMatcher(a=current_body, b=previous_body).ratio()
    signature_score = difflib.SequenceMatcher(a=current_function.signature, b=previous_function.signature).ratio()
    return (body_score * 0.75) + (signature_score * 0.25)


def _normalized_source(source_code: str, function_name: str) -> str:
    normalized = "".join(source_code.split())
    bare_name = function_name.split(".")[-1]
    return normalized.replace(bare_name, "")


def _build_function_change(
    language: str,
    current_function: ParsedFunction,
    change_type: str,
    previous_function: ParsedFunction | None,
) -> FunctionChange:
    behavioral_change = True if previous_function is None else has_behavioral_change(language, current_function, previous_function)
    should_skip = False
    skip_reason = None
    if not behavioral_change and change_type != CHANGE_TYPE_RENAME:
        should_skip = True
        skip_reason = "cosmetic_change"
    elif change_type in {CHANGE_TYPE_LOGIC, CHANGE_TYPE_SIGNATURE} and _should_skip_generation(current_function):
        should_skip = True
        skip_reason = "short_function_without_branches_or_validation"

    return FunctionChange(
        function_name=current_function.function_name,
        start_line=current_function.start_line,
        end_line=current_function.end_line,
        source_code=current_function.source_code,
        context_code=current_function.context_code,
        signature=current_function.signature,
        change_type=change_type,
        previous_name=previous_function.function_name if previous_function else None,
        previous_signature=previous_function.signature if previous_function else None,
        previous_source_code=previous_function.source_code if previous_function else None,
        previous_context_code=previous_function.context_code if previous_function else None,
        enclosing_class_name=current_function.enclosing_class_name,
        called_functions=current_function.called_functions,
        branch_count=current_function.branch_count,
        has_validation=current_function.has_validation,
        has_behavioral_change=behavioral_change,
        should_skip_generation=should_skip,
        skip_reason=skip_reason,
    )


def _should_skip_generation(function: ParsedFunction) -> bool:
    meaningful_lines = [
        line for line in function.source_code.splitlines()
        if line.strip()
        and not line.strip().startswith("@")
        and not re.match(r"^\s*(async\s+)?def\s+\w+", line)
        and not re.match(r"^\s*(async\s+)?function\s+\w+", line)
        and not re.match(r"^\s*class\s+\w+", line)
    ]
    has_validation = (
        function.has_validation
        or bool(_VALIDATION_HINT_PATTERN.search(function.source_code))
    )
    return (
        len(meaningful_lines) < 10
        and function.branch_count == 0
        and not has_validation
    )


def _function_is_trivial(function: ParsedFunction) -> bool:
    if function.branch_count > 0 or function.has_validation:
        return False
    nontrivial_calls = [call for call in function.called_functions if call.split(".")[-1] not in _TRIVIAL_CALL_WHITELIST]
    if nontrivial_calls:
        return False
    body_lines = _function_body_lines(function.source_code)
    if not body_lines or len(body_lines) > 6:
        return False
    return all(_is_trivial_body_line(line) for line in body_lines)


def _function_body_lines(source_code: str) -> list[str]:
    meaningful = [line.strip().rstrip(";") for line in source_code.splitlines() if line.strip() and not line.strip().startswith("@")]
    if not meaningful:
        return []
    body_lines = meaningful[1:]
    return [line for line in body_lines if line not in {"{", "}"}]


def _is_trivial_body_line(line: str) -> bool:
    if _COMPLEX_LINE_PATTERN.search(line):
        return False
    return bool(_SIMPLE_STATEMENT_PATTERN.match(line))


def _normalize_function_source(language: str, function: ParsedFunction) -> str:
    if language == "python":
        return _normalize_python_function(function.source_code)
    if language == "javascript":
        return _normalize_javascript_function(function)
    return _normalize_typescript_function(function)


def _normalize_python_function(source_code: str) -> str:
    try:
        module = ast.parse(source_code)
    except SyntaxError:
        return "".join(source_code.split())

    function_node = next(
        (node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function_node is None:
        return "".join(source_code.split())

    local_names = _collect_python_local_names(function_node)
    canonical = _PythonCanonicalizer(local_names).visit(function_node)
    ast.fix_missing_locations(canonical)
    return ast.dump(canonical, annotate_fields=True, include_attributes=False)


def _normalize_javascript_function(function: ParsedFunction) -> str:
    for candidate in _candidate_js_like_sources(function):
        try:
            program = esprima.parseModule(candidate, {"jsx": True, "tolerant": True})
        except Exception:
            continue
        return json.dumps(_serialize_js_ast(program), sort_keys=True, separators=(",", ":"))
    return "".join(_strip_js_comments(function.source_code).split())


def _normalize_typescript_function(function: ParsedFunction) -> str:
    for candidate in _candidate_js_like_sources(function):
        for parser_name in ("typescript", "tsx"):
            parser = get_parser(parser_name)
            candidate_bytes = candidate.encode("utf-8")
            tree = parser.parse(candidate_bytes)
            if tree.root_node.has_error:
                continue
            return json.dumps(_serialize_ts_ast(tree.root_node, candidate_bytes), sort_keys=True, separators=(",", ":"))
    return "".join(_strip_js_comments(function.source_code).split())


def _candidate_js_like_sources(function: ParsedFunction) -> list[str]:
    stripped = function.source_code.strip()
    candidates = [stripped]
    if function.enclosing_class_name or re.match(r"^(?:async\s+)?[A-Za-z_$][\w$]*\s*\(", stripped):
        candidates.insert(0, f"class Temp {{\n{stripped}\n}}")
    if "=>" in stripped and not re.match(r"^(?:export\s+)?(?:const|let|var|async\s+function|function)\b", stripped):
        candidates.append(f"const __wrapped = {stripped};")
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _serialize_js_ast(node):
    if node is None:
        return None
    if isinstance(node, list):
        return [_serialize_js_ast(item) for item in node]
    if isinstance(node, (str, int, float, bool)):
        return node
    if hasattr(node, "type"):
        payload = {"type": getattr(node, "type")}
        for key, value in sorted(vars(node).items()):
            if key in {"type", "loc", "range", "raw", "comments", "tokens"}:
                continue
            if value is None:
                continue
            payload[key] = _serialize_js_ast(value)
        return payload
    if isinstance(node, dict):
        return {key: _serialize_js_ast(value) for key, value in sorted(node.items())}
    return str(node)


def _serialize_ts_ast(node, source_bytes: bytes):
    children = [
        _serialize_ts_ast(child, source_bytes)
        for child in node.named_children
        if child.type != "comment"
    ]
    if not children:
        return {
            "type": node.type,
            "text": source_bytes[node.start_byte : node.end_byte].decode("utf-8"),
        }
    return {"type": node.type, "children": children}


def _collect_python_local_names(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    collector = _PythonLocalNameCollector()
    collector.visit(function_node)
    return collector.local_names


class _PythonLocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.local_names: set[str] = set()

    def visit_arg(self, node: ast.arg) -> None:
        self.local_names.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.local_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self.visit(argument)
        if node.args.vararg is not None:
            self.visit(node.args.vararg)
        if node.args.kwarg is not None:
            self.visit(node.args.kwarg)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.local_names.add(node.name)
        for statement in node.body:
            self.visit(statement)


class _PythonCanonicalizer(ast.NodeTransformer):
    def __init__(self, local_names: set[str]) -> None:
        self.local_names = set(local_names)
        self.name_map: dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "function"
        node.body = _strip_python_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "function"
        node.body = _strip_python_docstring(node.body)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.local_names:
            node.arg = self._canonical_name(node.arg)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.local_names:
            node.id = self._canonical_name(node.id)
        return node

    def _canonical_name(self, name: str) -> str:
        return self.name_map.setdefault(name, f"v{len(self.name_map) + 1}")


def _strip_python_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def _strip_js_comments(source_code: str) -> str:
    protected_strings: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected_strings.append(match.group(0))
        return f"__STRING_{len(protected_strings) - 1}__"

    stripped = _JS_STRING_PATTERN.sub(_protect, source_code)
    stripped = _JS_BLOCK_COMMENT_PATTERN.sub("", stripped)
    stripped = _JS_LINE_COMMENT_PATTERN.sub("", stripped)

    for index, value in enumerate(protected_strings):
        stripped = stripped.replace(f"__STRING_{index}__", value)
    return stripped


def _parse_functions(language: str, file_path: Path, source_text: str) -> list[ParsedFunction]:
    parser = _select_parser(language)
    return parser(file_path, source_text)


def _select_parser(language: str):
    if language == "python":
        return parse_python_functions
    if language == "javascript":
        return parse_js_functions
    return parse_ts_functions


def _is_test_or_generated_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return "/tests/" in normalized or normalized.startswith("tests/") or "/ai_generated/" in normalized


def _is_ignored_non_source_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return any(pattern.search(normalized) for pattern in _CONFIG_FILE_PATTERNS)


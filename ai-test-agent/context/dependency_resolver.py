from __future__ import annotations

import re
import ast
from dataclasses import dataclass
from pathlib import Path

from context.token_budget import MAX_DEPENDENCIES, estimate_tokens
from diff.diff_models import FunctionChange, ParsedFunction
from languages.javascript.js_parser import parse_functions as parse_js_functions
from languages.python.python_parser import parse_functions as parse_python_functions
from languages.typescript.ts_parser import parse_functions as parse_ts_functions


_CALL_PATTERN = re.compile(r"\b([A-Za-z_][\w\.]*)\s*\(")
_IMPORT_FROM_PATTERN = re.compile(r"^from\s+([\.\w]+)\s+import\s+(.+)$", re.MULTILINE)
_IMPORT_PATTERN = re.compile(r"^import\s+([\w\.]+)(?:\s+as\s+(\w+))?$", re.MULTILINE)
_ESM_IMPORT_PATTERN = re.compile(r"^import\s+(.*?)\s+from\s+['\"](.+?)['\"];?$", re.MULTILINE)
_REQUIRE_PATTERN = re.compile(r"^const\s+(.*?)\s*=\s*require\(['\"](.+?)['\"]\);?$", re.MULTILINE)
_BRANCH_USAGE_PATTERN = re.compile(r"\b(if|while)\b.*\b{call}\b")
_RETURN_USAGE_PATTERN = re.compile(r"\breturn\b.*\b{call}\b")


@dataclass(frozen=True)
class DependencyContext:
    name: str
    source_file: str | None
    score: int
    mode: str
    content: str
    summary: str


@dataclass(frozen=True)
class ImportReference:
    local_name: str
    imported_name: str | None
    module_path: str


@dataclass(frozen=True)
class ResolvedDependency:
    name: str
    source_file: str | None
    function_name: str
    source_code: str
    score: int

def extract_dependencies(
    repo_root: Path,
    source_file: str,
    language: str,
    source_text: str,
    function_change: FunctionChange,
    max_dependencies: int = MAX_DEPENDENCIES,
) -> list[DependencyContext]:
    source_path = repo_root / source_file
    same_file_functions = _parse_functions(language, source_path, source_text)
    import_references = _extract_import_references(language, source_text, source_path, repo_root)
    candidates: dict[str, ResolvedDependency] = {}
    bare_function_name = function_change.function_name.split(".")[-1]

    for call_name in function_change.called_functions or _extract_called_functions(function_change.source_code):
        bare_call = call_name.split(".")[-1]
        if bare_call == bare_function_name:
            continue
        resolved = _resolve_dependency(
            repo_root=repo_root,
            source_path=source_path,
            language=language,
            call_name=call_name,
            same_file_functions=same_file_functions,
            import_references=import_references,
            enclosing_class_name=function_change.enclosing_class_name,
        )
        if resolved is None:
            continue
        scored = ResolvedDependency(
            name=call_name,
            source_file=resolved.source_file,
            function_name=resolved.function_name,
            source_code=resolved.source_code,
            score=_score_dependency(call_name, function_change.source_code, resolved.source_code),
        )
        existing = candidates.get(scored.function_name)
        if existing is None or scored.score > existing.score:
            candidates[scored.function_name] = scored

    selected = sorted(candidates.values(), key=lambda item: (-item.score, item.function_name))[:max_dependencies]
    return [_to_dependency_context(item, language) for item in selected if item.score > -4]


def _to_dependency_context(resolved: ResolvedDependency, language: str) -> DependencyContext:
    summary = _summarize_dependency(language, resolved.source_code)
    if _should_inline_dependency(resolved.source_code):
        return DependencyContext(
            name=resolved.function_name,
            source_file=resolved.source_file,
            score=resolved.score,
            mode="code",
            content=resolved.source_code,
            summary=summary,
        )
    return DependencyContext(
        name=resolved.function_name,
        source_file=resolved.source_file,
        score=resolved.score,
        mode="summary",
        content=summary,
        summary=summary,
    )


def _resolve_dependency(
    repo_root: Path,
    source_path: Path,
    language: str,
    call_name: str,
    same_file_functions: list[ParsedFunction],
    import_references: list[ImportReference],
    enclosing_class_name: str | None,
) -> ResolvedDependency | None:
    bare_call = call_name.split(".")[-1]

    same_file_match = _match_same_file_dependency(call_name, bare_call, same_file_functions, enclosing_class_name, source_path, repo_root)
    if same_file_match is not None:
        return same_file_match

    import_reference = _match_import_reference(call_name, bare_call, import_references)
    if import_reference is None:
        return None
    dependency_path = repo_root / import_reference.module_path
    if not dependency_path.exists():
        return None
    dependency_source = dependency_path.read_text(encoding="utf-8")
    dependency_functions = _parse_functions(_language_for_path(dependency_path, language), dependency_path, dependency_source)
    target_name = import_reference.imported_name or bare_call
    for function in dependency_functions:
        if function.function_name.split(".")[-1] == target_name:
            return ResolvedDependency(
                name=call_name,
                source_file=str(import_reference.module_path).replace("\\", "/"),
                function_name=function.function_name,
                source_code=function.source_code,
                score=0,
            )
    return None


def _match_same_file_dependency(
    call_name: str,
    bare_call: str,
    same_file_functions: list[ParsedFunction],
    enclosing_class_name: str | None,
    source_path: Path,
    repo_root: Path,
) -> ResolvedDependency | None:
    preferred_names = {bare_call, call_name}
    if enclosing_class_name and call_name.startswith(("self.", "cls.")):
        preferred_names.add(f"{enclosing_class_name}.{bare_call}")
    for function in same_file_functions:
        if function.function_name in preferred_names or function.function_name.split(".")[-1] == bare_call:
            return ResolvedDependency(
                name=call_name,
                source_file=str(source_path.relative_to(repo_root)).replace("\\", "/"),
                function_name=function.function_name,
                source_code=function.source_code,
                score=0,
            )
    return None


def _match_import_reference(call_name: str, bare_call: str, import_references: list[ImportReference]) -> ImportReference | None:
    parts = call_name.split(".")
    alias = parts[0]
    for reference in import_references:
        if reference.local_name == alias:
            if len(parts) > 1:
                return ImportReference(local_name=reference.local_name, imported_name=parts[-1], module_path=reference.module_path)
            return reference
        if reference.local_name == bare_call:
            return reference
    return None


def _score_dependency(call_name: str, function_source: str, dependency_source: str) -> int:
    score = 0
    escaped = re.escape(call_name)
    if re.search(_BRANCH_USAGE_PATTERN.pattern.format(call=escaped), function_source):
        score += 3
    if re.search(_RETURN_USAGE_PATTERN.pattern.format(call=escaped), function_source):
        score += 3
    lowered = call_name.lower()
    if any(token in lowered for token in ("validate", "check", "parse")):
        score += 2
    if any(token in lowered for token in ("log", "print", "debug", "trace")):
        score -= 5
    if _is_simple_wrapper(dependency_source):
        score -= 3
    return score


def _is_simple_wrapper(source_code: str) -> bool:
    meaningful_lines = [line.strip() for line in source_code.splitlines() if line.strip() and not line.strip().startswith(("#", "//", "@"))]
    if len(meaningful_lines) > 4:
        return False
    joined = " ".join(meaningful_lines).lower()
    return joined.startswith("def ") or joined.startswith("function ") or joined.startswith("async")


def _should_inline_dependency(source_code: str) -> bool:
    return estimate_tokens(source_code) <= 180 and len(source_code.splitlines()) <= 25

def _summarize_python(source_code: str) -> str:
    _BUILTIN_NOISE = {
        "len", "str", "int", "float", "bool", "list", "dict", "set",
        "tuple", "range", "enumerate", "zip", "map", "filter", "sorted",
        "round", "abs", "min", "max", "sum", "any", "all", "print",
        "isinstance", "hasattr", "getattr", "setattr", "type",
    }

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.rules: list[str] = []
            self._seen: set[str] = set()

        def _add(self, rule: str) -> None:
            if rule not in self._seen:
                self._seen.add(rule)
                self.rules.append(rule)

        def visit_If(self, node):
            self._add(f"condition: if {ast.unparse(node.test)}")
            # capture elif/else existence without full detail
            if node.orelse:
                if isinstance(node.orelse[0], ast.If):
                    self._add(f"condition: elif {ast.unparse(node.orelse[0].test)}")
                else:
                    self._add("condition: else branch exists")
            self.generic_visit(node)

        def visit_For(self, node):
            target = ast.unparse(node.target)
            iter_ = ast.unparse(node.iter)
            self._add(f"loop: for {target} in {iter_}")
            self.generic_visit(node)

        def visit_While(self, node):
            self._add(f"loop: while {ast.unparse(node.test)}")
            self.generic_visit(node)

        def visit_Try(self, node):
            self._add("error handling: try/except block")
            for handler in node.handlers:
                if handler.type:
                    self._add(f"catches: {ast.unparse(handler.type)}")
            self.generic_visit(node)

        def visit_Return(self, node):
            if node.value:
                self._add(f"returns: {ast.unparse(node.value)}")

        def visit_Raise(self, node):
            if node.exc:
                self._add(f"raises: {ast.unparse(node.exc)}")

        def visit_Assign(self, node):
            # only capture assignments to module-level or self. attributes
            # local variable churn adds noise
            for target in node.targets:
                t = ast.unparse(target)
                if t.startswith("self.") or t.startswith("cls."):
                    self._add(f"mutates: {t}")
            self.generic_visit(node)

        def visit_Call(self, node):
            func = ast.unparse(node.func)
            if func.split(".")[0] not in _BUILTIN_NOISE:
                self._add(f"calls: {func}()")
            self.generic_visit(node)

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return f"- behavior: {source_code.splitlines()[0].strip()}"

    visitor = _Visitor()
    visitor.visit(tree)

    rules = visitor.rules[:15]
    if not rules:
        return f"- behavior: {source_code.splitlines()[0].strip()}"
    return "\n".join(f"- {r}" for r in rules)


def _summarize_js(source_code: str) -> str:
    _BUILTIN_NOISE = {
        "console", "Math", "JSON", "Object", "Array", "String",
        "Number", "Boolean", "Promise", "Error", "Date",
        "parseInt", "parseFloat", "isNaN", "isFinite",
    }

    rules: list[str] = []
    seen: set[str] = set()

    def add(rule: str) -> None:
        if rule not in seen:
            seen.add(rule)
            rules.append(rule)

    for line in source_code.splitlines():
        s = line.strip().rstrip(";")
        if not s or s.startswith(("//", "/*", "*")):
            continue

        # if condition
        m = re.match(r"if\s*\((.+?)\)\s*\{?$", s)
        if m:
            add(f"condition: if {m.group(1).strip()}")
            continue

        # else if
        m = re.match(r"else\s+if\s*\((.+?)\)\s*\{?$", s)
        if m:
            add(f"condition: else if {m.group(1).strip()}")
            continue

        # else branch
        if re.match(r"^else\s*\{?$", s):
            add("condition: else branch exists")
            continue

        # for loop — classic and for...of / for...in
        m = re.match(r"for\s*\((.+?)\)\s*\{?$", s)
        if m:
            add(f"loop: for ({m.group(1).strip()})")
            continue

        # while loop
        m = re.match(r"while\s*\((.+?)\)\s*\{?$", s)
        if m:
            add(f"loop: while {m.group(1).strip()}")
            continue

        # try/catch/finally
        if re.match(r"^try\s*\{?$", s):
            add("error handling: try/catch block")
            continue

        m = re.match(r"catch\s*\((\w+)\)\s*\{?$", s)
        if m:
            add(f"catches: {m.group(1)}")
            continue

        if re.match(r"^finally\s*\{?$", s):
            add("error handling: finally block")
            continue

        # throw
        m = re.match(r"throw\s+new\s+(\w+)\s*\((.*?)?\)", s)
        if m:
            msg = m.group(2).strip().strip("'\"`") if m.group(2) else ""
            add(f"raises: {m.group(1)}({msg})" if msg else f"raises: {m.group(1)}")
            continue

        # return
        m = re.match(r"return\s+(.+?)$", s)
        if m:
            val = m.group(1).strip()
            if val not in ("null", "undefined", "true", "false", "void 0"):
                add(f"returns: {val[:60]}")
            continue

        # external calls — method chains and standalone
        m = re.search(r"(?:await\s+)?(\w+(?:\.\w+)+)\s*\(", s)
        if m:
            func = m.group(1)
            if func.split(".")[0] not in _BUILTIN_NOISE:
                add(f"calls: {func}()")

    if not rules:
        for line in source_code.splitlines():
            s = line.strip()
            if s and not re.match(
                r"^(export\s+)?(async\s+)?function|^const\s+\w+=|^//|^\*", s
            ):
                return f"- behavior: {s[:80]}"
        return f"- behavior: {source_code.splitlines()[0].strip()}"

    return "\n".join(f"- {r}" for r in rules[:15])

def _summarize_dependency(language: str, source_code: str) -> str:
    if language == "python":
        return _summarize_python(source_code)
    return _summarize_js(source_code)



def _extract_called_functions(source_code: str) -> list[str]:
    seen: list[str] = []
    for match in _CALL_PATTERN.finditer(source_code):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "catch", "function"}:
            continue
        if name not in seen:
            seen.append(name)
    return seen


def _extract_import_references(language: str, source_text: str, source_path: Path, repo_root: Path) -> list[ImportReference]:
    if language == "python":
        return _extract_python_imports(source_text, source_path, repo_root)
    return _extract_js_imports(source_text, source_path, repo_root)


def _extract_python_imports(source_text: str, source_path: Path, repo_root: Path) -> list[ImportReference]:
    references: list[ImportReference] = []
    for module_name, names_text in _IMPORT_FROM_PATTERN.findall(source_text):
        for imported_name, alias in _split_import_targets(names_text):
            module_path = _resolve_python_module_path(module_name, source_path, repo_root)
            if module_path is None:
                continue
            references.append(ImportReference(local_name=alias or imported_name, imported_name=imported_name, module_path=module_path))
    for module_name, alias in _IMPORT_PATTERN.findall(source_text):
        module_path = _resolve_python_module_path(module_name, source_path, repo_root)
        if module_path is None:
            continue
        local_name = alias or module_name.split(".")[-1]
        references.append(ImportReference(local_name=local_name, imported_name=None, module_path=module_path))
    return references


def _extract_js_imports(source_text: str, source_path: Path, repo_root: Path) -> list[ImportReference]:
    references: list[ImportReference] = []
    for names_text, module_name in _ESM_IMPORT_PATTERN.findall(source_text):
        module_path = _resolve_js_module_path(module_name, source_path, repo_root)
        if module_path is None:
            continue
        references.extend(_build_js_import_references(names_text, module_path))
    for names_text, module_name in _REQUIRE_PATTERN.findall(source_text):
        module_path = _resolve_js_module_path(module_name, source_path, repo_root)
        if module_path is None:
            continue
        references.extend(_build_js_import_references(names_text, module_path))
    return references


def _build_js_import_references(names_text: str, module_path: str) -> list[ImportReference]:
    names_text = names_text.strip()
    references: list[ImportReference] = []
    if names_text.startswith("{") and names_text.endswith("}"):
        for item in names_text[1:-1].split(","):
            item = item.strip()
            if not item:
                continue
            if " as " in item:
                imported_name, local_name = [part.strip() for part in item.split(" as ", 1)]
            elif ":" in item:
                imported_name, local_name = [part.strip() for part in item.split(":", 1)]
            else:
                imported_name = local_name = item
            references.append(ImportReference(local_name=local_name, imported_name=imported_name, module_path=module_path))
        return references
    if names_text.startswith("*") and " as " in names_text:
        references.append(ImportReference(local_name=names_text.split(" as ", 1)[1].strip(), imported_name=None, module_path=module_path))
        return references
    cleaned = names_text.split(",", 1)[0].strip()
    if cleaned:
        references.append(ImportReference(local_name=cleaned, imported_name=cleaned, module_path=module_path))
    return references


def _split_import_targets(names_text: str) -> list[tuple[str, str | None]]:
    targets: list[tuple[str, str | None]] = []
    for raw_target in names_text.split(","):
        target = raw_target.strip()
        if not target:
            continue
        if " as " in target:
            imported_name, alias = [part.strip() for part in target.split(" as ", 1)]
            targets.append((imported_name, alias))
        else:
            targets.append((target, None))
    return targets


def _resolve_python_module_path(module_name: str, source_path: Path, repo_root: Path) -> str | None:
    search_paths: list[Path] = []
    if module_name.startswith("."):
        relative_level = len(module_name) - len(module_name.lstrip("."))
        module_suffix = module_name.lstrip(".")
        base_path = source_path.parent
        for _ in range(max(relative_level - 1, 0)):
            base_path = base_path.parent
        search_base = base_path / module_suffix.replace(".", "/") if module_suffix else base_path
        search_paths.extend([search_base.with_suffix(".py"), search_base / "__init__.py"])
    else:
        module_path = repo_root / module_name.replace(".", "/")
        search_paths.extend([module_path.with_suffix(".py"), module_path / "__init__.py"])
    for candidate in search_paths:
        if candidate.exists():
            return str(candidate.relative_to(repo_root)).replace("\\", "/")
    return None


def _resolve_js_module_path(module_name: str, source_path: Path, repo_root: Path) -> str | None:
    if not module_name.startswith("."):
        return None
    base_path = (source_path.parent / module_name).resolve()
    candidates = [
        base_path,
        base_path.with_suffix(".js"),
        base_path.with_suffix(".jsx"),
        base_path.with_suffix(".ts"),
        base_path.with_suffix(".tsx"),
        base_path / "index.js",
        base_path / "index.ts",
        base_path / "index.tsx",
    ]
    for candidate in candidates:
        if candidate.exists() and repo_root in candidate.parents:
            return str(candidate.relative_to(repo_root)).replace("\\", "/")
    return None


def _language_for_path(file_path: Path, fallback_language: str) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".js", ".jsx"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    return fallback_language


def _parse_functions(language: str, file_path: Path, source_text: str) -> list[ParsedFunction]:
    if language == "python":
        return parse_python_functions(file_path, source_text)
    if language == "javascript":
        return parse_js_functions(file_path, source_text)
    return parse_ts_functions(file_path, source_text)


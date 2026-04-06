from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from context.token_budget import MAX_DEPENDENCIES
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
    return [_to_dependency_context(item, language, str(repo_root)) for item in selected if item.score > -4]


def _to_dependency_context(
    resolved: ResolvedDependency,
    language: str,
    repo_root: str,
) -> DependencyContext:
    """
    Local deps (same-file or cross-file project modules):
        send full raw source code — no summarization.
    Third-party deps (installed packages, no local file):
        send import line only.
    """
    if _is_third_party_dep(resolved.source_file, repo_root):
        import_line = _build_import_line(
            resolved.function_name,
            resolved.source_file,
            language,
        )
        return DependencyContext(
            name=resolved.function_name,
            source_file=resolved.source_file,
            score=resolved.score,
            mode="import_only",
            content=import_line,
            summary=import_line,
        )

    # Local dependency — send full source, no summarization
    return DependencyContext(
        name=resolved.function_name,
        source_file=resolved.source_file,
        score=resolved.score,
        mode="code",
        content=resolved.source_code or "",
        summary=resolved.source_code or "",
    )


def _is_third_party_dep(source_file: str | None, repo_root: str) -> bool:
    import os
    if not source_file:
        return True
    if "site-packages" in source_file:
        return True
    full_path = os.path.join(repo_root, source_file)
    return not os.path.exists(full_path)


def _build_import_line(name: str, source_file: str | None, language: str) -> str:
    if not source_file:
        return f"# {name} (third-party)"
    if language == "python":
        module = (
            source_file
            .replace("\\", "/")
            .removesuffix(".py")
            .replace("/", ".")
        )
        return f"from {module} import {name}"
    path = (
        source_file
        .replace("\\", "/")
        .removesuffix(".tsx")
        .removesuffix(".ts")
        .removesuffix(".jsx")
        .removesuffix(".js")
    )
    return f"import {{ {name} }} from '{path}'"


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


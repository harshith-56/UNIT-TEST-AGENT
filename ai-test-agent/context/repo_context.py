from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import test_framework_for_language
from diff.diff_models import ChangedFile, ChangedFunction
from languages.javascript.js_parser import parse_functions as parse_js_functions
from languages.python.python_parser import parse_functions as parse_python_functions
from languages.typescript.ts_parser import parse_functions as parse_ts_functions
from test_discovery.test_scanner import find_related_tests


IMPORT_PATTERNS = {
    "python": re.compile(r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$", re.MULTILINE),
    "javascript": re.compile(r"^(?:import\s+.+|const\s+.+\s+=\s+require\(.+\))$", re.MULTILINE),
    "typescript": re.compile(r"^(?:import\s+.+|const\s+.+\s+=\s+require\(.+\))$", re.MULTILINE),
}


@dataclass(frozen=True)
class GenerationTarget:
    source_file: str
    language: str
    framework: str
    imports: str
    helper_functions: str
    existing_tests: str
    changed_functions: list[ChangedFunction] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationContext:
    targets: list[GenerationTarget]


def build_generation_context(repo_root: Path, changed_files: list[ChangedFile], discovered_tests: list[Path]) -> GenerationContext:
    targets: list[GenerationTarget] = []
    for changed_file in changed_files:
        source_path = repo_root / changed_file.file_path
        source_text = source_path.read_text(encoding="utf-8")
        imports = _extract_imports(changed_file.language, source_text)
        helper_functions = _extract_helpers(source_path, changed_file)
        related_tests = find_related_tests(source_path, discovered_tests)
        existing_tests = _collect_existing_tests(related_tests, changed_file.changed_functions)
        targets.append(
            GenerationTarget(
                source_file=changed_file.file_path,
                language=changed_file.language,
                framework=test_framework_for_language(changed_file.language),
                imports=imports,
                helper_functions=helper_functions,
                existing_tests=existing_tests,
                changed_functions=changed_file.changed_functions,
            )
        )
    return GenerationContext(targets=targets)


def _extract_imports(language: str, source_text: str) -> str:
    matches = IMPORT_PATTERNS[language].findall(source_text)
    return "\n".join(matches[:20])


def _extract_helpers(source_path: Path, changed_file: ChangedFile) -> str:
    parser = _select_parser(changed_file.language)
    functions = parser(source_path)
    changed_names = {function.function_name for function in changed_file.changed_functions}
    helper_snippets: list[str] = []
    for function in functions:
        if function.function_name in changed_names:
            continue
        if _is_referenced(function.function_name, changed_file.changed_functions):
            helper_snippets.append(function.source_code)
        if len(helper_snippets) >= 3:
            break
    return "\n\n".join(helper_snippets)


def _is_referenced(helper_name: str, changed_functions: list[ChangedFunction]) -> bool:
    bare_name = helper_name.split(".")[-1]
    for changed_function in changed_functions:
        if bare_name in changed_function.source_code and bare_name != changed_function.function_name.split(".")[-1]:
            return True
    return False


def _collect_existing_tests(test_paths: list[Path], changed_functions: list[ChangedFunction]) -> str:
    snippets: list[str] = []
    function_names = {function.function_name.split(".")[-1] for function in changed_functions}
    for path in test_paths:
        source = path.read_text(encoding="utf-8")
        if not function_names or any(name in source for name in function_names):
            snippets.append(source[:4000])
        if sum(len(snippet) for snippet in snippets) >= 6000:
            break
    return "\n\n".join(snippets)


def _select_parser(language: str):
    if language == "python":
        return parse_python_functions
    if language == "javascript":
        return parse_js_functions
    return parse_ts_functions

from __future__ import annotations

import ast as _ast
from dataclasses import dataclass, field, replace
from pathlib import Path
import re

from context.dependency_resolver import DependencyContext, extract_dependencies
from context.project_context import StructuredContext
from diff.diff_analyzer import should_generate_tests
from diff.diff_models import CHANGE_TYPE_DELETION, CHANGE_TYPE_RENAME, ChangedFile, FunctionChange
from test_discovery.test_scanner import find_related_tests
from validation.test_naming import build_test_prefix, extract_test_names, sanitize_test_identifier


MAX_EXISTING_TEST_BLOCKS = 4
MAX_EXISTING_TEST_CHARS = 4000


@dataclass(frozen=True)
class GenerationTarget:
    source_file: str
    language: str
    function_change: FunctionChange
    test_id: str
    generation_mode: str
    project_rules: list[str]
    pr_rules: list[str]
    dependencies: list[DependencyContext]
    existing_tests_text: str
    existing_test_names: list[str]
    repair_test_names: list[str] = field(default_factory=list)
    repair_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MaintenanceAction:
    source_file: str
    language: str
    function_name: str
    test_id: str
    action_type: str
    previous_name: str | None = None
    previous_test_id: str | None = None
    existing_test_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationContext:
    targets: list[GenerationTarget]
    maintenance_actions: list[MaintenanceAction]


def build_generation_context(
    repo_root: Path,
    changed_files: list[ChangedFile],
    discovered_tests: list[Path],
    project_context: StructuredContext,
    pr_context: StructuredContext,
) -> GenerationContext:
    targets: list[GenerationTarget] = []
    maintenance_actions: list[MaintenanceAction] = []

    for changed_file in changed_files:
        source_path = repo_root / changed_file.file_path
        related_tests = find_related_tests(source_path, discovered_tests)
        source_text = changed_file.current_source or changed_file.previous_source or ""

        for function_change in changed_file.function_changes:
            test_id = sanitize_test_identifier(function_change.function_name)
            existing_tests_text, existing_test_names = _collect_existing_tests(
                changed_file.language,
                related_tests,
                function_change,
            )

            if function_change.change_type in {CHANGE_TYPE_RENAME, CHANGE_TYPE_DELETION}:
                maintenance_actions.append(
                    MaintenanceAction(
                        source_file=changed_file.file_path,
                        language=changed_file.language,
                        function_name=function_change.function_name,
                        test_id=test_id,
                        action_type=function_change.change_type,
                        previous_name=function_change.previous_name,
                        previous_test_id=sanitize_test_identifier(
                            function_change.previous_name or function_change.function_name
                        ),
                        existing_test_names=existing_test_names,
                    )
                )
                # Deletions never need new tests
                if function_change.change_type == CHANGE_TYPE_DELETION:
                    continue
                # Pure rename with no logic change: maintenance only, no generation
                if not function_change.has_behavioral_change:
                    continue
                # Rename + logic change: fall through to generate new tests below
                # (do NOT continue — let GenerationTarget creation run)

            if not should_generate_tests(function_change):
                continue

            if function_change.enclosing_class_name:
                class_src = _extract_class_source(
                    source_text,
                    function_change.enclosing_class_name,
                    changed_file.language,
                )
                if class_src:
                    function_change = replace(function_change, source_code=class_src)

            targets.append(
                GenerationTarget(
                    source_file=changed_file.file_path,
                    language=changed_file.language,
                    function_change=function_change,
                    test_id=test_id,
                    generation_mode="replace",
                    project_rules=project_context.combined_rules(),
                    pr_rules=pr_context.combined_rules(),
                    dependencies=extract_dependencies(
                        repo_root=repo_root,
                        source_file=changed_file.file_path,
                        language=changed_file.language,
                        source_text=source_text,
                        function_change=function_change,
                    ),
                    existing_tests_text=existing_tests_text,
                    existing_test_names=existing_test_names,
                )
            )

    return GenerationContext(targets=targets, maintenance_actions=maintenance_actions)


def _collect_existing_tests(
    language: str,
    test_paths: list[Path],
    function_change: FunctionChange,
) -> tuple[str, list[str]]:
    target_prefixes = {
        build_test_prefix(function_change.function_name),
        build_test_prefix(function_change.previous_name or function_change.function_name),
    }
    snippets: list[str] = []
    snippet_chars = 0
    test_names: list[str] = []

    for path in test_paths:
        source = path.read_text(encoding="utf-8")
        names = extract_test_names(language, source)
        relevant_names = [name for name in names if any(name.startswith(prefix) for prefix in target_prefixes)]
        if not relevant_names:
            continue

        test_names.extend(relevant_names)
        for block in _extract_named_test_blocks(language, source, relevant_names):
            block = block.strip()
            if not block:
                continue
            projected_chars = snippet_chars + len(block)
            if snippets and projected_chars > MAX_EXISTING_TEST_CHARS:
                break
            snippets.append(block)
            snippet_chars = projected_chars
            if len(snippets) >= MAX_EXISTING_TEST_BLOCKS:
                break
        if len(snippets) >= MAX_EXISTING_TEST_BLOCKS or snippet_chars >= MAX_EXISTING_TEST_CHARS:
            break

    return "\n\n".join(snippets), sorted(dict.fromkeys(test_names))


def _extract_named_test_blocks(language: str, source: str, test_names: list[str]) -> list[str]:
    blocks: list[str] = []
    for name in test_names:
        pattern = re.compile(_test_pattern(language, re.escape(name)), flags=re.MULTILINE | re.DOTALL)
        match = pattern.search(source)
        if match is None:
            continue
        block = match.group(0).strip()
        if block and block not in blocks:
            blocks.append(block)
    return blocks


def _test_pattern(language: str, name_pattern: str) -> str:
    if language == "python":
        return rf"^def\s+{name_pattern}\(.*?(?=^def\s+test_|\Z)"
    return rf"^\s*(?:it|test)\(\s*['\"`]{name_pattern}['\"`].*?(?=^\s*(?:it|test)\(\s*['\"`]test_|\Z)"


def _extract_class_source(file_source: str, class_name: str, language: str) -> str:
    """
    Extract the complete class definition from file source.
    Returns empty string if not found — caller falls back to
    the original function source.
    """
    if language == "python":
        return _extract_python_class(file_source, class_name)
    return _extract_js_class(file_source, class_name)


def _extract_python_class(source: str, class_name: str) -> str:
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return ""
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == class_name:
            lines = source.splitlines()
            start = node.lineno - 1       # ast is 1-based
            end = node.end_lineno         # inclusive
            return "\n".join(lines[start:end])
    return ""


def _extract_js_class(source: str, class_name: str) -> str:
    # Matches: [export] [default] [abstract] class ClassName
    #          [extends X] [implements Y, Z] {
    # Works for .js .ts .jsx .tsx
    pattern = re.compile(
        rf"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
        rf"class\s+{re.escape(class_name)}"
        rf"(?:\s+extends\s+[\w.<>, ]+?)?"
        rf"(?:\s+implements\s+[\w.<>, ]+?)?"
        rf"\s*\{{",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    start = m.start()
    # Walk forward counting braces to find matching closing }
    depth = 0
    for j in range(m.end() - 1, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    return source[start:]  # fallback: rest of file

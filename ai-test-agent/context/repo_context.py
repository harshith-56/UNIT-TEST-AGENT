from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from context.dependency_resolver import DependencyContext, extract_dependencies
from context.project_context import StructuredContext
from diff.diff_analyzer import should_generate_tests
from diff.diff_models import CHANGE_TYPE_DELETION, CHANGE_TYPE_RENAME, ChangedFile, FunctionChange
from test_discovery.test_scanner import find_related_tests
from validation.test_naming import build_test_prefix, extract_test_names, sanitize_test_identifier


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
                        previous_test_id=sanitize_test_identifier(function_change.previous_name or function_change.function_name),
                        existing_test_names=existing_test_names,
                    )
                )
                continue

            if not should_generate_tests(function_change):
                continue

            targets.append(
                GenerationTarget(
                    source_file=changed_file.file_path,
                    language=changed_file.language,
                    function_change=function_change,
                    test_id=test_id,
                    generation_mode="replace" if function_change.change_type == "signature_change" else "append",
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
    test_names: list[str] = []
    for path in test_paths:
        source = path.read_text(encoding="utf-8")
        names = extract_test_names(language, source)
        relevant_names = [name for name in names if any(name.startswith(prefix) for prefix in target_prefixes)]
        if not relevant_names:
            continue
        test_names.extend(relevant_names)
        snippets.append(source[:4000])
        if sum(len(snippet) for snippet in snippets) >= 6000:
            break
    return "\n\n".join(snippets), sorted(dict.fromkeys(test_names))

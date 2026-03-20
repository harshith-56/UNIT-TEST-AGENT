from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from agent.config import detect_languages, load_config
from context.event_context import load_event_context
from context.project_context import extract_pr_context, parse_project_context
from context.repo_context import GenerationContext, GenerationTarget, build_generation_context
from diff.diff_analyzer import analyze_diff
from execution.failure_parser import collect_failed_generated_files, collect_failed_test_names, collect_failure_notes
from execution.test_runner import execute_tests, has_failures
from generation.test_generator import generate_tests
from integration.test_mapping import mapping_key
from integration.test_writer import WriteResult, write_generated_tests
from reporting.pr_commenter import post_pr_comment
from test_discovery.test_scanner import discover_existing_tests
from utils.file_utils import sanitize_module_name
from utils.logger import get_logger
from validation.duplicate_detector import filter_duplicate_tests
from validation.syntax_validator import validate_generated_tests


LOGGER = get_logger(__name__)


def main() -> int:
    config = load_config()
    event_context = load_event_context()
    project_context = parse_project_context(config.project_context_raw)
    pr_context = extract_pr_context(event_context)

    changed_files = analyze_diff(config.repo_root, event_context.base_branch)
    detected_languages = detect_languages(changed_file.file_path for changed_file in changed_files)
    existing_tests = discover_existing_tests(config.repo_root)
    generation_context = build_generation_context(
        config.repo_root,
        changed_files,
        existing_tests,
        project_context,
        pr_context,
    )

    if not generation_context.targets and not generation_context.maintenance_actions:
        LOGGER.info("no_generation_needed")
        return 0

    generated_tests = generate_tests(generation_context.targets, config)
    if generation_context.targets and not generated_tests:
        LOGGER.error("no_tests_generated")
        return 1

    valid_tests, invalid_tests = validate_generated_tests(generated_tests)
    if invalid_tests:
        LOGGER.warning("invalid_tests_detected count=%s", len(invalid_tests))
    if generation_context.targets and not valid_tests:
        LOGGER.error("no_valid_tests_after_validation")
        return 1

    final_tests = filter_duplicate_tests(valid_tests, generation_context)
    write_result = write_generated_tests(
        config.repo_root,
        final_tests,
        generation_context.maintenance_actions,
        config,
    )
    if generation_context.targets and not write_result.written_paths and write_result.maintenance_changes == 0:
        LOGGER.error("no_files_written")
        return 1

    test_results = execute_tests(config.repo_root, detected_languages)
    repair_targets = _build_repair_targets(
        generation_context.targets,
        write_result.test_mapping,
        test_results,
        write_result.written_paths,
    )

    repaired_tests: list = []
    if repair_targets:
        LOGGER.info("repair_targets_detected count=%s", len(repair_targets))
        repaired_generated_tests = generate_tests(repair_targets, config)
        valid_repairs, invalid_repairs = validate_generated_tests(repaired_generated_tests)
        if invalid_repairs:
            LOGGER.warning("invalid_repaired_tests_detected count=%s", len(invalid_repairs))
        repair_context = GenerationContext(targets=repair_targets, maintenance_actions=[])
        repaired_tests = filter_duplicate_tests(valid_repairs, repair_context)
        repair_write_result = write_generated_tests(config.repo_root, repaired_tests, [], config)
        write_result = _merge_write_results(write_result, repair_write_result)
        test_results = execute_tests(config.repo_root, detected_languages)

    LOGGER.info(
        "agent_run_summary changed_files=%s targets=%s generated=%s repaired=%s invalid=%s written=%s maintenance=%s",
        len(changed_files),
        len(generation_context.targets),
        len(final_tests),
        len(repaired_tests),
        len(invalid_tests),
        len(write_result.written_paths),
        write_result.maintenance_changes,
    )

    if config.comment_on_pr:
        post_pr_comment(
            event_context,
            len(final_tests) + len(repaired_tests),
            len(write_result.written_paths),
            test_results,
        )

    if has_failures(test_results):
        LOGGER.warning("generated_tests_have_failures")
        if config.fail_on_test_failure:
            return 1

    return 0


def _build_repair_targets(
    targets: list[GenerationTarget],
    test_mapping: dict[str, dict],
    test_results,
    written_paths: list[Path],
) -> list[GenerationTarget]:
    failed_test_names = collect_failed_test_names(test_results)
    failed_generated_files = collect_failed_generated_files(test_results)
    if not failed_test_names and not failed_generated_files:
        return []

    written_file_names = {Path(path).name for path in written_paths}
    repair_targets: list[GenerationTarget] = []
    for target in targets:
        entry = test_mapping.get(mapping_key(target.source_file, target.function_change.function_name), {})
        mapped_names = list(entry.get("test_names") or [])
        target_file = str(entry.get("target_file") or "")
        if not target_file:
            expected_target_file = _expected_generated_file_name(target)
            if expected_target_file in written_file_names or expected_target_file in failed_generated_files:
                target_file = expected_target_file
        failing_names = [name for name in mapped_names if name in failed_test_names]
        file_failed = bool(target_file and target_file in failed_generated_files)
        if not failing_names and not file_failed:
            continue
        repair_names = failing_names or mapped_names
        repair_targets.append(
            replace(
                target,
                generation_mode="repair",
                repair_test_names=repair_names,
                repair_notes=collect_failure_notes(test_results, repair_names, [target_file] if target_file else []),
            )
        )
    return repair_targets


def _expected_generated_file_name(target: GenerationTarget) -> str:
    module_name = sanitize_module_name(target.source_file)
    if target.language == "python":
        return f"test_ai_generated_{module_name}.py"
    if target.language == "javascript":
        return f"ai_generated_{module_name}.test.js"
    return f"ai_generated_{module_name}.test.ts"


def _merge_write_results(left: WriteResult, right: WriteResult) -> WriteResult:
    return WriteResult(
        written_paths=sorted(dict.fromkeys([*left.written_paths, *right.written_paths])),
        test_mapping=right.test_mapping,
        maintenance_changes=left.maintenance_changes + right.maintenance_changes,
    )


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import sys

from agent.config import detect_languages, load_config
from context.event_context import load_event_context
from context.repo_context import build_generation_context
from diff.diff_analyzer import analyze_diff
from execution.test_runner import execute_tests, has_failures
from generation.test_generator import generate_tests
from integration.test_writer import write_generated_tests
from reporting.pr_commenter import post_pr_comment
from test_discovery.test_scanner import discover_existing_tests
from utils.logger import get_logger
from validation.duplicate_detector import filter_duplicate_tests
from validation.syntax_validator import validate_generated_tests


LOGGER = get_logger(__name__)


def main() -> int:
    config = load_config()

    event_context = load_event_context()

    changed_files = analyze_diff(config.repo_root, event_context.base_branch)

    detected_languages = detect_languages(
        changed_file.file_path for changed_file in changed_files
    )

    existing_tests = discover_existing_tests(config.repo_root)

    generation_context = build_generation_context(
        config.repo_root,
        changed_files,
        existing_tests,
    )

    generated_tests = (
        generate_tests(generation_context, config)
        if generation_context.targets
        else []
    )

    valid_tests, invalid_tests = validate_generated_tests(generated_tests)

    if invalid_tests:
        LOGGER.warning("invalid_tests_detected count=%s", len(invalid_tests))

    final_tests = filter_duplicate_tests(valid_tests, generation_context)

    written_files = (
        write_generated_tests(config.repo_root, final_tests, config)
        if final_tests
        else []
    )

    test_results = execute_tests(config.repo_root, detected_languages)

    LOGGER.info(
        "agent_run_summary changed_files=%s generated=%s valid=%s invalid=%s written=%s",
        len(changed_files),
        len(generated_tests),
        len(valid_tests),
        len(invalid_tests),
        len(written_files),
    )

    if config.comment_on_pr:
        post_pr_comment(
            event_context,
            len(final_tests),
            len(written_files),
            test_results,
        )

    # ---------- new exit policy ----------

    # Fail only if no tests were generated
    if len(written_files) == 0:
        LOGGER.error("no_tests_written_generation_failed")
        return 1

    # Log failing tests but do not fail CI
    if has_failures(test_results):
        LOGGER.warning("generated_tests_have_failures")

    return 0


if __name__ == "__main__":
    sys.exit(main())
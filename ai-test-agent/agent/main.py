from __future__ import annotations

import sys

from agent.config import load_config
from context.event_context import load_event_context
from context.project_context import extract_pr_context, parse_project_context
from context.repo_context import build_generation_context
from diff.diff_analyzer import analyze_diff
from generation.test_generator import generate_tests
from integration.test_writer import write_generated_tests
from test_discovery.test_scanner import discover_existing_tests
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
    if not changed_files:
        # LOGGER.info("No relevant changes detected — skipping generation")
        return 0

    existing_tests = discover_existing_tests(config.repo_root)
    generation_context = build_generation_context(
        config.repo_root,
        changed_files,
        existing_tests,
        project_context,
        pr_context,
    )

    if not generation_context.targets and not generation_context.maintenance_actions:
        # LOGGER.info("No generation targets — skipping")
        return 0

    # LOGGER.info(
    #     "generation_targets_detected count=%s maintenance_actions=%s",
    #     len(generation_context.targets),
    #     len(generation_context.maintenance_actions),
    # )

    # Generate tests for all targets
    generation_result = generate_tests(generation_context.targets, config)

    # Validate — drop invalid, keep valid
    valid_tests, invalid_tests = validate_generated_tests(
        generation_result.generated_tests
    )

    if invalid_tests:
        # LOGGER.warning(
        #     "invalid_tests_skipped count=%s — run agent again to retry",
        #     len(invalid_tests),
        # )
        for inv in invalid_tests:
            # LOGGER.warning("  skipped: %s", inv.test_id)
            pass

    # Deduplicate
    final_tests = filter_duplicate_tests(valid_tests, generation_context)

    # Write everything — tests + maintenance actions (renames, deletes)
    if final_tests or generation_context.maintenance_actions:
        write_result = write_generated_tests(
            config.repo_root,
            final_tests,
            generation_context.maintenance_actions,
            config,
        )
        # LOGGER.info(
        #     "agent_run_summary "
        #     "targets=%s generated=%s invalid=%s written=%s maintenance=%s",
        #     len(generation_context.targets),
        #     len(final_tests),
        #     len(invalid_tests),
        #     len(write_result.written_paths),
        #     len(generation_context.maintenance_actions),
        # )
    else:
        pass  # LOGGER.info("No valid tests to write")

    return 0


if __name__ == "__main__":
    sys.exit(main())

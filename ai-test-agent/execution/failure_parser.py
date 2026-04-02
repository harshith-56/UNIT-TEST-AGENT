from __future__ import annotations

import re
from pathlib import Path

from execution.test_runner import TestRunResult


_PYTEST_FAILURE_PATTERNS = [
    re.compile(r"FAILED\s+[^\s]+::(?P<name>test_[A-Za-z0-9_]+)"),
    re.compile(r"::(?P<name>test_[A-Za-z0-9_]+)\s+FAILED"),
]
_JEST_FAILURE_PATTERNS = [
    re.compile(r"^[\s\u2715\u00d7]*[\u2715\u00d7]\s+(?P<name>test_[A-Za-z0-9_]+)", re.MULTILINE),
    re.compile(r"\u203a\s+(?P<name>test_[A-Za-z0-9_]+)"),
]
_GENERATED_FILE_PATTERN = re.compile(
    r"(?P<file>(?:test_ai_generated_[A-Za-z0-9_./\\-]+\.py|ai_generated_[A-Za-z0-9_./\\-]+\.test\.(?:js|ts)))"
)


def collect_failed_test_names(results: list[TestRunResult]) -> set[str]:
    failed: set[str] = set()
    for result in results:
        text = _joined_output(result)
        patterns = _PYTEST_FAILURE_PATTERNS if result.language == "python" else _JEST_FAILURE_PATTERNS
        for pattern in patterns:
            failed.update(match.group("name") for match in pattern.finditer(text))
    return failed


def collect_collection_errors(test_results: list) -> set[str]:
    """
    Returns source file stems that had pytest collection errors
    (ImportError, ModuleNotFoundError, SyntaxError during collection).
    These are different from individual test failures — the entire
    file failed to import.
    """
    collection_error_files: set[str] = set()
    _COLLECTION_ERROR = re.compile(
        r"ERROR\s+collecting\s+.*?(?P<file>test_ai_generated_\w+\.py)"
    )
    for result in test_results:
        output = (result.stdout or "") + (result.stderr or "")
        for m in _COLLECTION_ERROR.finditer(output):
            collection_error_files.add(m.group("file"))
    return collection_error_files


def collect_failed_generated_files(results: list[TestRunResult]) -> set[str]:
    failed_files: set[str] = set()
    for result in results:
        text = _joined_output(result)
        for match in _GENERATED_FILE_PATTERN.finditer(text):
            failed_files.add(Path(match.group("file")).name)
    return failed_files


def collect_failure_notes(
    results: list[TestRunResult],
    test_names: list[str] | None = None,
    file_names: list[str] | None = None,
) -> list[str]:
    tracked_tests = set(test_names or [])
    tracked_files = set(file_names or [])
    if not tracked_tests and not tracked_files:
        return []

    notes: list[str] = []
    for result in results:
        lines = [line.rstrip() for line in _joined_output(result).splitlines()]
        for index, line in enumerate(lines):
            if not _line_matches_failure(line, tracked_tests, tracked_files):
                continue
            snippet = lines[index : index + 4]
            note = " | ".join(part.strip() for part in snippet if part.strip())
            if note and note not in notes:
                notes.append(note[:400])
    return notes[:6]


def _joined_output(result: TestRunResult) -> str:
    return "\n".join(part for part in [result.stdout, result.stderr] if part)


def _line_matches_failure(line: str, tracked_tests: set[str], tracked_files: set[str]) -> bool:
    if any(test_name in line for test_name in tracked_tests):
        return True
    return any(file_name in line for file_name in tracked_files)

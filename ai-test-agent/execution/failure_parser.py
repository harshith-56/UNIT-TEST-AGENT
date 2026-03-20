from __future__ import annotations

import re

from execution.test_runner import TestRunResult


_PYTEST_FAILURE_PATTERNS = [
    re.compile(r"FAILED\s+[^\s]+::(?P<name>test_[A-Za-z0-9_]+)"),
    re.compile(r"::(?P<name>test_[A-Za-z0-9_]+)\s+FAILED"),
]
_JEST_FAILURE_PATTERNS = [
    re.compile(r"^[\s\u2715\u00d7]*[\u2715\u00d7]\s+(?P<name>test_[A-Za-z0-9_]+)", re.MULTILINE),
    re.compile(r"\u203a\s+(?P<name>test_[A-Za-z0-9_]+)"),
]


def collect_failed_test_names(results: list[TestRunResult]) -> set[str]:
    failed: set[str] = set()
    for result in results:
        text = "\n".join(part for part in [result.stdout, result.stderr] if part)
        patterns = _PYTEST_FAILURE_PATTERNS if result.language == "python" else _JEST_FAILURE_PATTERNS
        for pattern in patterns:
            failed.update(match.group("name") for match in pattern.finditer(text))
    return failed


def collect_failure_notes(results: list[TestRunResult], test_names: list[str]) -> list[str]:
    if not test_names:
        return []
    notes: list[str] = []
    for result in results:
        lines = [line.rstrip() for line in (result.stdout + "\n" + result.stderr).splitlines()]
        for index, line in enumerate(lines):
            if not any(test_name in line for test_name in test_names):
                continue
            snippet = lines[index : index + 4]
            note = " | ".join(part.strip() for part in snippet if part.strip())
            if note and note not in notes:
                notes.append(note[:400])
    return notes[:6]

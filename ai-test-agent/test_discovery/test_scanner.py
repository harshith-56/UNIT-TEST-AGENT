from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path


TEST_PATTERNS = [
    "test_*.py",
    "*_test.py",
    "*.test.js",
    "*.spec.js",
    "*.test.ts",
    "*.spec.ts",
]

SEARCH_DIR_NAMES = {"tests", "test"}


def discover_existing_tests(repo_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if not _is_test_directory(path):
            continue
        if any(fnmatch(path.name, pattern) for pattern in TEST_PATTERNS):
            discovered.append(path)
    return sorted(discovered)


def find_related_tests(source_file: Path, discovered_tests: list[Path]) -> list[Path]:
    source_stem = source_file.stem.lower()
    source_parts = {part.lower() for part in source_file.parts}
    related: list[Path] = []
    for test_path in discovered_tests:
        test_name = test_path.name.lower()
        if source_stem in test_name:
            related.append(test_path)
            continue
        if any(part.lower() in source_parts for part in test_path.parts):
            related.append(test_path)
    return sorted(set(related))


def _is_test_directory(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts[:-1]}
    return any(part in SEARCH_DIR_NAMES or "tests" in part for part in lower_parts)

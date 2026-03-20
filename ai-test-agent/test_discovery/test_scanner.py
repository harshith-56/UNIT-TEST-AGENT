from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path


TEST_PATTERNS = [
    "test_*.py",
    "*_test.py",
    "*.test.js",
    "*.spec.js",
    "*.test.jsx",
    "*.spec.jsx",
    "*.test.ts",
    "*.spec.ts",
    "*.test.tsx",
    "*.spec.tsx",
]

SEARCH_DIR_NAMES = {"tests", "test", "spec", "specs", "__tests__"}
MAX_DISCOVERED_TESTS = 500
ROOT_PATTERNS = [
    "tests",
    "test",
    "spec",
    "specs",
    "__tests__",
    "*/tests",
    "*/test",
    "*/spec",
    "*/specs",
    "*/__tests__",
    "*/*/tests",
    "*/*/test",
    "*/*/spec",
    "*/*/specs",
    "*/*/__tests__",
]


def discover_existing_tests(repo_root: Path) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in _candidate_test_roots(repo_root):
        for path in root.rglob("*"):
            if len(discovered) >= MAX_DISCOVERED_TESTS:
                return sorted(discovered)
            if not path.is_file():
                continue
            if path in seen:
                continue
            if any(fnmatch(path.name, pattern) for pattern in TEST_PATTERNS):
                discovered.append(path)
                seen.add(path)
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


def _candidate_test_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for pattern in ROOT_PATTERNS:
        for candidate in repo_root.glob(pattern):
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(candidate)
    return sorted(roots)

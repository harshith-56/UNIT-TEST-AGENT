from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from agent.config import detect_language
from diff.diff_models import ChangedFile, ChangedFunction
from languages.javascript.js_parser import parse_functions as parse_js_functions
from languages.python.python_parser import parse_functions as parse_python_functions
from languages.typescript.ts_parser import parse_functions as parse_ts_functions
from utils.git_utils import get_changed_files, get_diff


HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def analyze_diff(repo_root: Path, base_branch: str) -> list[ChangedFile]:
    diff_text = get_diff(repo_root, base_branch)
    changed_file_paths = get_changed_files(repo_root, base_branch)
    diff_by_file = _split_diff_by_file(diff_text)
    changed_files: list[ChangedFile] = []

    for file_path in changed_file_paths:
        language = detect_language(file_path)
        if not language:
            continue
        absolute_path = repo_root / file_path
        if not absolute_path.exists() or _is_test_or_generated_path(file_path):
            continue
        changed_lines = _extract_changed_lines(diff_by_file.get(file_path, ""))
        changed_functions = _extract_changed_functions(absolute_path, language, changed_lines)
        if not changed_functions:
            continue
        changed_files.append(
            ChangedFile(
                file_path=file_path,
                language=language,
                changed_lines=changed_lines,
                changed_functions=changed_functions,
            )
        )
    return changed_files


def _split_diff_by_file(diff_text: str) -> dict[str, str]:
    chunks: dict[str, list[str]] = defaultdict(list)
    current_file: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_file = None
        elif line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/").strip()
            chunks[current_file]
        elif current_file:
            chunks[current_file].append(line)
    return {file_path: "\n".join(lines) for file_path, lines in chunks.items()}


def _extract_changed_lines(file_diff: str) -> list[int]:
    lines: list[int] = []
    for start_text, length_text in HUNK_PATTERN.findall(file_diff):
        start = int(start_text)
        length = int(length_text or "1")
        if length == 0:
            continue
        lines.extend(range(start, start + length))
    return sorted(set(lines))


def _extract_changed_functions(file_path: Path, language: str, changed_lines: list[int]) -> list[ChangedFunction]:
    parser = _select_parser(language)
    functions = parser(file_path)
    impacted: list[ChangedFunction] = []
    for function in functions:
        if not changed_lines or any(function.start_line <= line <= function.end_line for line in changed_lines):
            impacted.append(function)
    return impacted


def _select_parser(language: str):
    if language == "python":
        return parse_python_functions
    if language == "javascript":
        return parse_js_functions
    return parse_ts_functions


def _is_test_or_generated_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return "/tests/" in normalized or normalized.startswith("tests/") or "/ai_generated/" in normalized

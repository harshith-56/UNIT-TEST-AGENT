from __future__ import annotations

import difflib
import re
from collections import defaultdict
from pathlib import Path

from agent.config import detect_language
from diff.diff_models import (
    CHANGE_TYPE_DELETION,
    CHANGE_TYPE_LOGIC,
    CHANGE_TYPE_RENAME,
    CHANGE_TYPE_SIGNATURE,
    ChangedFile,
    FunctionChange,
    ParsedFunction,
)
from languages.javascript.js_parser import parse_functions as parse_js_functions
from languages.python.python_parser import parse_functions as parse_python_functions
from languages.typescript.ts_parser import parse_functions as parse_ts_functions
from utils.git_utils import get_changed_files, get_diff, get_file_content_at_ref


HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_VALIDATION_HINT_PATTERN = re.compile(r"\b(validate|check|parse|required|invalid|empty|none|null|raise|throw)\b", re.IGNORECASE)


def analyze_diff(repo_root: Path, base_branch: str) -> list[ChangedFile]:
    diff_text = get_diff(repo_root, base_branch)
    changed_file_paths = get_changed_files(repo_root, base_branch)
    diff_by_file = _split_diff_by_file(diff_text)
    changed_files: list[ChangedFile] = []

    for file_path in changed_file_paths:
        language = detect_language(file_path)
        if not language or _is_test_or_generated_path(file_path):
            continue

        current_path = repo_root / file_path
        current_source = current_path.read_text(encoding="utf-8") if current_path.exists() else ""
        previous_source = get_file_content_at_ref(repo_root, f"origin/{base_branch}", file_path)
        changed_lines, removed_lines = _extract_changed_line_sets(diff_by_file.get(file_path, ""))

        current_functions = _parse_functions(language, current_path, current_source) if current_source else []
        previous_functions = _parse_functions(language, current_path, previous_source) if previous_source else []
        function_changes = _classify_function_changes(current_functions, previous_functions, changed_lines, removed_lines)
        if not function_changes:
            continue

        changed_files.append(
            ChangedFile(
                file_path=file_path,
                language=language,
                current_source=current_source,
                previous_source=previous_source,
                changed_lines=changed_lines,
                removed_lines=removed_lines,
                function_changes=function_changes,
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


def _extract_changed_line_sets(file_diff: str) -> tuple[list[int], list[int]]:
    added_lines: list[int] = []
    removed_lines: list[int] = []
    for old_start_text, old_length_text, new_start_text, new_length_text in HUNK_PATTERN.findall(file_diff):
        old_start = int(old_start_text)
        old_length = int(old_length_text or "1")
        new_start = int(new_start_text)
        new_length = int(new_length_text or "1")
        if new_length > 0:
            added_lines.extend(range(new_start, new_start + new_length))
        if old_length > 0:
            removed_lines.extend(range(old_start, old_start + old_length))
    return sorted(set(added_lines)), sorted(set(removed_lines))


def _classify_function_changes(
    current_functions: list[ParsedFunction],
    previous_functions: list[ParsedFunction],
    changed_lines: list[int],
    removed_lines: list[int],
) -> list[FunctionChange]:
    impacted_current = _impacted_functions(current_functions, changed_lines)
    impacted_previous = _impacted_functions(previous_functions, removed_lines)
    previous_by_name = {function.function_name: function for function in impacted_previous}
    matched_previous: set[str] = set()
    function_changes: list[FunctionChange] = []

    for current_function in impacted_current:
        previous_function = previous_by_name.get(current_function.function_name)
        if previous_function is not None:
            matched_previous.add(previous_function.function_name)
            change_type = CHANGE_TYPE_SIGNATURE if current_function.signature != previous_function.signature else CHANGE_TYPE_LOGIC
            function_changes.append(_build_function_change(current_function, change_type, previous_function))
            continue

        renamed_from = _match_renamed_function(current_function, impacted_previous, matched_previous)
        if renamed_from is not None:
            matched_previous.add(renamed_from.function_name)
            function_changes.append(_build_function_change(current_function, CHANGE_TYPE_RENAME, renamed_from))
            continue

        function_changes.append(_build_function_change(current_function, CHANGE_TYPE_LOGIC, None))

    for previous_function in impacted_previous:
        if previous_function.function_name in matched_previous:
            continue
        function_changes.append(
            FunctionChange(
                function_name=previous_function.function_name,
                start_line=previous_function.start_line,
                end_line=previous_function.end_line,
                source_code=previous_function.source_code,
                context_code=previous_function.context_code,
                signature=previous_function.signature,
                change_type=CHANGE_TYPE_DELETION,
                previous_name=previous_function.function_name,
                previous_signature=previous_function.signature,
                previous_source_code=previous_function.source_code,
                previous_context_code=previous_function.context_code,
                enclosing_class_name=previous_function.enclosing_class_name,
                called_functions=previous_function.called_functions,
                branch_count=previous_function.branch_count,
                has_validation=previous_function.has_validation,
            )
        )

    return function_changes


def _impacted_functions(functions: list[ParsedFunction], changed_lines: list[int]) -> list[ParsedFunction]:
    if not changed_lines:
        return functions
    return [
        function
        for function in functions
        if any(function.start_line <= line <= function.end_line for line in changed_lines)
    ]


def _match_renamed_function(
    current_function: ParsedFunction,
    previous_functions: list[ParsedFunction],
    matched_previous: set[str],
) -> ParsedFunction | None:
    best_match: tuple[float, ParsedFunction] | None = None
    for candidate in previous_functions:
        if candidate.function_name in matched_previous:
            continue
        if candidate.enclosing_class_name != current_function.enclosing_class_name:
            continue
        similarity = _similarity_score(current_function, candidate)
        if similarity < 0.58:
            continue
        if best_match is None or similarity > best_match[0]:
            best_match = (similarity, candidate)
    return best_match[1] if best_match is not None else None


def _similarity_score(current_function: ParsedFunction, previous_function: ParsedFunction) -> float:
    current_body = _normalized_source(current_function.source_code, current_function.function_name)
    previous_body = _normalized_source(previous_function.source_code, previous_function.function_name)
    body_score = difflib.SequenceMatcher(a=current_body, b=previous_body).ratio()
    signature_score = difflib.SequenceMatcher(a=current_function.signature, b=previous_function.signature).ratio()
    return (body_score * 0.75) + (signature_score * 0.25)


def _normalized_source(source_code: str, function_name: str) -> str:
    normalized = "".join(source_code.split())
    bare_name = function_name.split(".")[-1]
    return normalized.replace(bare_name, "")


def _build_function_change(
    current_function: ParsedFunction,
    change_type: str,
    previous_function: ParsedFunction | None,
) -> FunctionChange:
    should_skip = False
    skip_reason = None
    if change_type == CHANGE_TYPE_LOGIC and _should_skip_generation(current_function):
        should_skip = True
        skip_reason = "short_function_without_branches_or_validation"
    return FunctionChange(
        function_name=current_function.function_name,
        start_line=current_function.start_line,
        end_line=current_function.end_line,
        source_code=current_function.source_code,
        context_code=current_function.context_code,
        signature=current_function.signature,
        change_type=change_type,
        previous_name=previous_function.function_name if previous_function else None,
        previous_signature=previous_function.signature if previous_function else None,
        previous_source_code=previous_function.source_code if previous_function else None,
        previous_context_code=previous_function.context_code if previous_function else None,
        enclosing_class_name=current_function.enclosing_class_name,
        called_functions=current_function.called_functions,
        branch_count=current_function.branch_count,
        has_validation=current_function.has_validation,
        should_skip_generation=should_skip,
        skip_reason=skip_reason,
    )


def _should_skip_generation(function: ParsedFunction) -> bool:
    meaningful_lines = [line for line in function.source_code.splitlines() if line.strip() and not line.strip().startswith("@")]
    has_validation = function.has_validation or bool(_VALIDATION_HINT_PATTERN.search(function.source_code))
    return len(meaningful_lines) < 10 and function.branch_count == 0 and not has_validation


def _parse_functions(language: str, file_path: Path, source_text: str) -> list[ParsedFunction]:
    parser = _select_parser(language)
    return parser(file_path, source_text)


def _select_parser(language: str):
    if language == "python":
        return parse_python_functions
    if language == "javascript":
        return parse_js_functions
    return parse_ts_functions


def _is_test_or_generated_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return "/tests/" in normalized or normalized.startswith("tests/") or "/ai_generated/" in normalized

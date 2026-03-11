from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChangedFunction:
    function_name: str
    start_line: int
    end_line: int
    source_code: str


@dataclass(frozen=True)
class ChangedFile:
    file_path: str
    language: str
    changed_lines: list[int] = field(default_factory=list)
    changed_functions: list[ChangedFunction] = field(default_factory=list)

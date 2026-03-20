from __future__ import annotations

from dataclasses import dataclass, field


CHANGE_TYPE_LOGIC = "logic_change"
CHANGE_TYPE_SIGNATURE = "signature_change"
CHANGE_TYPE_RENAME = "rename"
CHANGE_TYPE_DELETION = "deletion"


@dataclass(frozen=True)
class ParsedFunction:
    function_name: str
    start_line: int
    end_line: int
    source_code: str
    context_code: str
    signature: str
    enclosing_class_name: str | None = None
    called_functions: list[str] = field(default_factory=list)
    branch_count: int = 0
    has_validation: bool = False


@dataclass(frozen=True)
class FunctionChange:
    function_name: str
    start_line: int
    end_line: int
    source_code: str
    context_code: str
    signature: str
    change_type: str
    previous_name: str | None = None
    previous_signature: str | None = None
    previous_source_code: str | None = None
    previous_context_code: str | None = None
    enclosing_class_name: str | None = None
    called_functions: list[str] = field(default_factory=list)
    branch_count: int = 0
    has_validation: bool = False
    should_skip_generation: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class ChangedFile:
    file_path: str
    language: str
    current_source: str
    previous_source: str | None = None
    changed_lines: list[int] = field(default_factory=list)
    removed_lines: list[int] = field(default_factory=list)
    function_changes: list[FunctionChange] = field(default_factory=list)

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneratedTest:
    source_file: str
    language: str
    function_name: str
    test_id: str
    generation_mode: str
    content: str
    test_names: list[str] = field(default_factory=list)
    repair_test_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationFailure:
    source_file: str
    function_name: str
    test_id: str
    generation_mode: str
    reason: str
    attempts: int


@dataclass(frozen=True)
class GenerationResult:
    generated_tests: list[GeneratedTest]
    failures: list[GenerationFailure] = field(default_factory=list)

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path


if "esprima" not in sys.modules:
    sys.modules["esprima"] = types.SimpleNamespace(parseModule=lambda *args, **kwargs: types.SimpleNamespace(body=[]))

if "tree_sitter_languages" not in sys.modules:
    class _FakeParser:
        def parse(self, _content: bytes):
            return types.SimpleNamespace(root_node=types.SimpleNamespace(children=[]))

    sys.modules["tree_sitter_languages"] = types.SimpleNamespace(get_parser=lambda _name: _FakeParser())


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from context.project_context import StructuredContext
from context.repo_context import GenerationTarget, build_generation_context
from diff.diff_analyzer import has_behavioral_change
from diff.diff_models import CHANGE_TYPE_LOGIC, ChangedFile, FunctionChange, ParsedFunction
from execution.failure_parser import collect_failed_generated_files
from execution.test_runner import TestRunResult
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt


def test_generation_pipeline_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".smoke-test-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
        generation_target = _build_sample_target(repo_root)
        llm_input = build_llm_input(generation_target)
        prompt = build_prompt(llm_input)

        assert llm_input.function_name == "normalize_username"
        assert llm_input.language == "python"
        assert llm_input.test_framework == "pytest"
        assert "LANGUAGE: python" in prompt
        assert "TEST FRAMEWORK: pytest" in prompt
        assert "PROJECT CONTEXT:" in prompt
        assert "PR CONTEXT:" in prompt
        assert "FUNCTION:" in prompt
        assert "DEPENDENCIES:" in prompt
        assert "helper" in prompt
        assert "test_normalize_username_<scenario>" in prompt
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_cosmetic_python_change_is_not_behavioral() -> None:
    previous = ParsedFunction(
        function_name="normalize_username",
        start_line=1,
        end_line=6,
        source_code=(
            "def normalize_username(username):\n"
            "    \"\"\"Normalize a username.\"\"\"\n"
            "    cleaned = username.strip()\n"
            "    return cleaned.lower()\n"
        ),
        context_code="",
        signature="def normalize_username(username)",
        called_functions=[],
    )
    current = ParsedFunction(
        function_name="normalize_username",
        start_line=1,
        end_line=6,
        source_code=(
            "def normalize_username(value):\n"
            "    \"\"\"Normalize user login text.\"\"\"\n"
            "    normalized = value.strip()  # keep lowercase\n"
            "    return normalized.lower()\n"
        ),
        context_code="",
        signature="def normalize_username(value)",
        called_functions=[],
    )

    assert has_behavioral_change("python", current, previous) is False


def test_prompt_budget_overflow_raises_skip_generation() -> None:
    huge_body = "\n".join(f"    if value == {index}: return {index}" for index in range(3000))
    target = GenerationTarget(
        source_file="src/huge.py",
        language="python",
        function_change=FunctionChange(
            function_name="huge_function",
            start_line=1,
            end_line=3002,
            source_code=f"def huge_function(value):\n{huge_body}\n    return -1\n",
            context_code="",
            signature="def huge_function(value)",
            change_type=CHANGE_TYPE_LOGIC,
            called_functions=[],
            branch_count=3000,
            has_validation=False,
        ),
        test_id="huge_function",
        generation_mode="append",
        project_rules=[],
        pr_rules=[],
        dependencies=[],
        existing_tests_text="",
        existing_test_names=[],
    )

    try:
        build_llm_input(target)
    except SkipGeneration:
        return
    raise AssertionError("Expected SkipGeneration for oversized prompt")


def test_collect_failed_generated_files_from_import_error() -> None:
    results = [
        TestRunResult(
            language="python",
            command=["python", "-m", "pytest"],
            returncode=2,
            stdout="",
            stderr="ImportError while importing test module '/tmp/tests/ai_generated/test_ai_generated_src_sample.py'",
        )
    ]

    assert collect_failed_generated_files(results) == {"test_ai_generated_src_sample.py"}


def _build_sample_target(repo_root: Path) -> GenerationTarget:
    source_path = repo_root / "src" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = (
        "def helper(value):\n"
        "    return value.strip().lower()\n\n"
        "def normalize_username(username):\n"
        "    if username is None:\n"
        "        raise ValueError('username required')\n"
        "    cleaned = helper(username)\n"
        "    if not cleaned:\n"
        "        return None\n"
        "    return cleaned\n"
    )
    source_path.write_text(source_text, encoding="utf-8")

    changed_file = ChangedFile(
        file_path="src/sample.py",
        language="python",
        current_source=source_text,
        previous_source=None,
        changed_lines=[4, 5, 6, 7, 8],
        function_changes=[
            FunctionChange(
                function_name="normalize_username",
                start_line=4,
                end_line=9,
                source_code=(
                    "def normalize_username(username):\n"
                    "    if username is None:\n"
                    "        raise ValueError('username required')\n"
                    "    cleaned = helper(username)\n"
                    "    if not cleaned:\n"
                    "        return None\n"
                    "    return cleaned\n"
                ),
                context_code=(
                    "def normalize_username(username):\n"
                    "    if username is None:\n"
                    "        raise ValueError('username required')\n"
                    "    cleaned = helper(username)\n"
                    "    if not cleaned:\n"
                    "        return None\n"
                    "    return cleaned\n"
                ),
                signature="def normalize_username(username)",
                change_type=CHANGE_TYPE_LOGIC,
                called_functions=["helper"],
                branch_count=2,
                has_validation=True,
            )
        ],
    )

    generation_context = build_generation_context(
        repo_root,
        [changed_file],
        [],
        StructuredContext(rules=["usernames must be normalized before storage"]),
        StructuredContext(rules=["returns None when cleaned username is empty"]),
    )

    assert len(generation_context.targets) == 1
    return generation_context.targets[0]

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
from context.repo_context import build_generation_context
from diff.diff_models import CHANGE_TYPE_LOGIC, ChangedFile, FunctionChange
from llm.prompt_builder import build_llm_input, build_prompt


def test_generation_pipeline_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".smoke-test-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
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

        llm_input = build_llm_input(generation_context.targets[0])
        prompt = build_prompt(llm_input)

        assert llm_input.function_name == "normalize_username"
        assert "PROJECT CONTEXT:" in prompt
        assert "PR CONTEXT:" in prompt
        assert "FUNCTION:" in prompt
        assert "DEPENDENCIES:" in prompt
        assert "helper" in prompt
        assert "test_normalize_username_<scenario>" in prompt
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path


if "esprima" not in sys.modules:
    sys.modules["esprima"] = types.SimpleNamespace(parseModule=lambda *args, **kwargs: types.SimpleNamespace(body=[]))

if "tree_sitter_languages" not in sys.modules:
    class _FakeNode:
        def __init__(self) -> None:
            self.type = "program"
            self.named_children = []
            self.start_byte = 0
            self.end_byte = 0
            self.has_error = False

    class _FakeParser:
        def parse(self, content: bytes):
            root = _FakeNode()
            root.end_byte = len(content)
            return types.SimpleNamespace(root_node=root)

    sys.modules["tree_sitter_languages"] = types.SimpleNamespace(get_parser=lambda _name: _FakeParser())


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from agent import main as agent_main
from agent.config import AgentConfig
from context.event_context import EventContext
from context.project_context import StructuredContext, extract_pr_context, extract_pr_rules
from context.repo_context import GenerationContext, GenerationTarget, build_generation_context
from diff.diff_analyzer import has_behavioral_change, is_config_like_file
from diff.diff_models import CHANGE_TYPE_LOGIC, ChangedFile, FunctionChange, ParsedFunction
from execution.test_runner import TestRunResult
from generation.test_generator import GenerationFailure, GenerationResult, _invalid_output_reason, generate_tests
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
        assert "IMPORT HINTS:" in prompt
        assert "DEPENDENCIES:" in prompt
        assert "helper" in prompt
        assert "test_normalize_username_<scenario>" in prompt
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_generate_tests_uses_retry_prompt_after_invalid_output(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".retry-prompt-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
        target = _build_sample_target(repo_root)
        config = _build_test_config(repo_root)
        invalid_output = (
            "from missing_local_module import normalize_username\n\n"
            "def test_normalize_username_valid():\n"
            "    assert normalize_username(' User ') == 'user'\n\n"
            "def test_normalize_username_none():\n"
            "    try:\n"
            "        normalize_username(None)\n"
            "    except ValueError:\n"
            "        assert True\n"
            "    else:\n"
            "        assert False\n\n"
            "def test_normalize_username_empty():\n"
            "    assert normalize_username('   ') is None\n"
        )
        valid_output = (
            "from src.sample import normalize_username\n\n"
            "def test_normalize_username_valid():\n"
            "    assert normalize_username(' User ') == 'user'\n\n"
            "def test_normalize_username_none():\n"
            "    try:\n"
            "        normalize_username(None)\n"
            "    except ValueError:\n"
            "        assert True\n"
            "    else:\n"
            "        assert False\n\n"
            "def test_normalize_username_empty():\n"
            "    assert normalize_username('   ') is None\n"
        )
        stub_client = _StubLLMClient([invalid_output, valid_output])

        monkeypatch.setattr("generation.test_generator.LLMClient", lambda _config: stub_client)
        monkeypatch.setattr("generation.test_generator.time.sleep", lambda _seconds: None)

        result = generate_tests([target], config)

        assert len(result.generated_tests) == 1
        assert result.failures == []
        assert len(stub_client.prompts) == 2
        assert stub_client.prompts[0] != stub_client.prompts[1]
        assert "CORRECTIONS (STRICT):" in stub_client.prompts[1]
        assert "Previous failure reason: unknown_import." in stub_client.prompts[1]
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_generate_tests_stops_early_on_duplicate_invalid_output(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".duplicate-output-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
        target = _build_sample_target(repo_root)
        config = _build_test_config(repo_root)
        invalid_output = (
            "from src.sample import normalize_username\n\n"
            "def test_normalize_username_placeholder():\n"
            "    value = SignupRequest(..., ***)\n"
        )
        stub_client = _StubLLMClient([invalid_output, invalid_output, "unused third response"])

        monkeypatch.setattr("generation.test_generator.LLMClient", lambda _config: stub_client)
        monkeypatch.setattr("generation.test_generator.time.sleep", lambda _seconds: None)

        result = generate_tests([target], config)

        assert result.generated_tests == []
        assert len(result.failures) == 1
        assert result.failures[0].reason == "duplicate_output"
        assert stub_client.calls == 2
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_invalid_output_reason_detects_unknown_python_import() -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".unknown-import-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
        target = _build_sample_target(repo_root)
        config = _build_test_config(repo_root)
        content = (
            "from made.up.module import normalize_username\n\n"
            "def test_normalize_username_valid():\n"
            "    assert normalize_username(' User ') == 'user'\n\n"
            "def test_normalize_username_none():\n"
            "    try:\n"
            "        normalize_username(None)\n"
            "    except ValueError:\n"
            "        assert True\n"
            "    else:\n"
            "        assert False\n\n"
            "def test_normalize_username_empty():\n"
            "    assert normalize_username('   ') is None\n"
        )

        assert _invalid_output_reason(target, config, content) == "unknown_import"
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_main_continues_when_generation_fails_and_flag_is_disabled(monkeypatch) -> None:
    target = _build_memory_target()
    _mock_main_generation_failure(monkeypatch, _build_test_config(Path.cwd(), fail_on_generation_failure=False), target)

    assert agent_main.main() == 0


def test_main_fails_when_generation_fails_and_flag_is_enabled(monkeypatch) -> None:
    target = _build_memory_target()
    _mock_main_generation_failure(monkeypatch, _build_test_config(Path.cwd(), fail_on_generation_failure=True), target)

    assert agent_main.main() == 1


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

    repair_targets = agent_main._build_repair_targets(
        [_build_memory_target()],
        {},
        results,
        [Path("tests/ai_generated/test_ai_generated_src_sample.py")],
    )

    assert len(repair_targets) == 1
    assert repair_targets[0].generation_mode == "repair"


def test_config_like_file_is_skipped_for_trivial_functions() -> None:
    functions = [
        ParsedFunction(
            function_name="load_settings",
            start_line=1,
            end_line=3,
            source_code=(
                "def load_settings():\n"
                "    value = DEFAULT_TIMEOUT\n"
                "    return value\n"
            ),
            context_code="",
            signature="def load_settings()",
            called_functions=[],
            branch_count=0,
            has_validation=False,
        )
    ]

    assert is_config_like_file("src/app_settings.py", functions) is True


def test_existing_test_context_only_keeps_matching_tests() -> None:
    repo_root = Path(__file__).resolve().parents[2] / ".existing-tests-workdir"
    if repo_root.exists():
        shutil.rmtree(repo_root)

    try:
        generation_target = _build_sample_target(repo_root)
        test_path = repo_root / "tests" / "test_sample.py"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(
            "def test_normalize_username_valid():\n"
            "    assert normalize_username(' User ') == 'user'\n\n"
            "def test_other_function_behavior():\n"
            "    assert helper('X') == 'x'\n",
            encoding="utf-8",
        )

        changed_file = ChangedFile(
            file_path="src/sample.py",
            language="python",
            current_source=(repo_root / "src" / "sample.py").read_text(encoding="utf-8"),
            previous_source=None,
            changed_lines=[4, 5, 6, 7, 8],
            function_changes=[generation_target.function_change],
        )
        generation_context = build_generation_context(
            repo_root,
            [changed_file],
            [test_path],
            StructuredContext(rules=[]),
            StructuredContext(rules=[]),
        )

        assert len(generation_context.targets) == 1
        existing_tests_text = generation_context.targets[0].existing_tests_text
        assert "test_normalize_username_valid" in existing_tests_text
        assert "test_other_function_behavior" not in existing_tests_text
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def _build_test_config(repo_root: Path, fail_on_generation_failure: bool = False) -> AgentConfig:
    return AgentConfig(
        repo_root=repo_root,
        generated_tests_dir=repo_root / "tests" / "ai_generated",
        llm_api_url="https://example.invalid/v1/chat/completions",
        llm_api_key="test-key",
        llm_model="test-model",
        fail_on_generation_failure=fail_on_generation_failure,
    )


class _StubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        self.calls += 1
        content = self._responses.pop(0)
        return types.SimpleNamespace(content=content, model="test-model")


def _mock_main_generation_failure(monkeypatch, config: AgentConfig, target: GenerationTarget) -> None:
    event_context = EventContext(
        repository="owner/repo",
        pull_request_number=1,
        base_branch="main",
        head_branch="feature",
        commit_sha="abc123",
        pull_request_title="Test",
        pull_request_body="Body",
    )
    failure = GenerationFailure(
        source_file=target.source_file,
        function_name=target.function_change.function_name,
        test_id=target.test_id,
        generation_mode=target.generation_mode,
        reason="unknown_import",
        attempts=2,
    )

    monkeypatch.setattr(agent_main, "load_config", lambda: config)
    monkeypatch.setattr(agent_main, "load_event_context", lambda: event_context)
    monkeypatch.setattr(agent_main, "parse_project_context", lambda _raw: StructuredContext(rules=[]))
    monkeypatch.setattr(agent_main, "extract_pr_context", lambda _event: StructuredContext(rules=[]))
    monkeypatch.setattr(agent_main, "analyze_diff", lambda _repo_root, _base_branch: [])
    monkeypatch.setattr(agent_main, "detect_languages", lambda _paths: {"python"})
    monkeypatch.setattr(agent_main, "discover_existing_tests", lambda _repo_root: [])
    monkeypatch.setattr(agent_main, "build_generation_context", lambda *args, **kwargs: GenerationContext(targets=[target], maintenance_actions=[]))
    monkeypatch.setattr(agent_main, "generate_tests", lambda _targets, _config: GenerationResult(generated_tests=[], failures=[failure]))
    monkeypatch.setattr(agent_main, "post_pr_comment", lambda *args, **kwargs: None)


def _build_memory_target() -> GenerationTarget:
    return GenerationTarget(
        source_file="src/sample.py",
        language="python",
        function_change=FunctionChange(
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
            context_code="",
            signature="def normalize_username(username)",
            change_type=CHANGE_TYPE_LOGIC,
            called_functions=["helper"],
            branch_count=2,
            has_validation=True,
        ),
        test_id="normalize_username",
        generation_mode="append",
        project_rules=[],
        pr_rules=[],
        dependencies=[],
        existing_tests_text="",
        existing_test_names=[],
    )


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


def test_extract_pr_rules_keeps_behavioral_constraints_only() -> None:
    rules = extract_pr_rules(
        "Fix signup validation",
        "Fixed bug. Now email must contain '@' and age must be >= 18. Refactored validator. Return None on failure.",
    )

    assert "email must contain '@'" in rules
    assert "age must be >= 18" in rules
    assert "return none on failure" in rules
    assert all("fixed bug" not in rule for rule in rules)
    assert all("refactor" not in rule for rule in rules)


def test_extract_pr_context_returns_empty_for_vague_pr() -> None:
    context = extract_pr_context(
        EventContext(
            repository="owner/repo",
            pull_request_number=1,
            base_branch="main",
            head_branch="feature",
            commit_sha="abc123",
            pull_request_title="Minor changes",
            pull_request_body="Refactored logic and improved performance.",
        )
    )

    assert context.combined_rules() == []

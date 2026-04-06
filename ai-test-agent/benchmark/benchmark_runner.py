"""
Benchmark runner for TestGenEval.
Generates tests for a source file without needing a GitHub PR or git diff.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure agent modules are importable
_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

from diff.diff_analyzer import extract_functions_from_source
from generation.test_generator import generate_tests
from validation.syntax_validator import validate_generated_tests
from context.repo_context import GenerationTarget, FunctionChange
from context.project_context import parse_project_context


def _make_config(llm_api_url: str, llm_api_key: str, llm_model: str):
    """Build a minimal AgentConfig for benchmark use."""
    from agent.config import AgentConfig
    return AgentConfig(
        llm_api_url=llm_api_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        repo_root=Path(tempfile.mkdtemp()),
        project_context_raw="",
        pr_title="",
        pr_body="",
        log_level="WARNING",
    )


def generate_tests_for_file(
    source_code: str,
    source_file_path: str,
    language: str,
    llm_api_url: str,
    llm_api_key: str,
    llm_model: str,
) -> str:
    """
    Main entry point for benchmark.
    Takes source file content, returns generated test file as string.
    """
    # Step 1: Extract all meaningful functions
    functions = extract_functions_from_source(
        source_code=source_code,
        file_path=source_file_path,
        language=language,
    )

    if not functions:
        return ""

    # Step 2: Build config
    config = _make_config(llm_api_url, llm_api_key, llm_model)

    # Step 3: Build GenerationTargets
    targets = []
    for func in functions:
        target = GenerationTarget(
            test_id=func.function_name,
            function_name=func.function_name,
            source_file=source_file_path,
            language=language,
            function_change=func,
            dependencies=[],
            project_rules=[],
            pr_rules=[],
            existing_tests_text="",
            existing_test_names=[],
            generation_mode="replace",
            repair_test_names=[],
            repair_notes=[],
        )
        targets.append(target)

    # Step 4: Generate
    result = generate_tests(targets, config)

    # Step 5: Validate
    valid_tests, invalid = validate_generated_tests(result.generated_tests)

    if not valid_tests:
        return ""

    # Step 6: Merge into one file
    return _merge_test_files(valid_tests, language)


def _merge_test_files(valid_tests, language: str) -> str:
    """Merge multiple generated test objects into one test file."""
    imports = set()
    bodies = []

    for test in valid_tests:
        content = test.content.strip()
        if not content:
            continue

        lines = content.splitlines()
        body_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()
            if in_header and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped == ""
            ):
                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.add(line.strip())
            else:
                in_header = False
                body_lines.append(line)

        if body_lines:
            bodies.append("\n".join(body_lines).strip())

    if not imports and not bodies:
        return ""

    result = "\n".join(sorted(imports))
    result += "\n\n"
    result += "\n\n\n".join(bodies)
    return result.strip() + "\n"

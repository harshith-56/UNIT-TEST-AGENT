from __future__ import annotations

from context.repo_context import GenerationTarget


def build_prompt(target: GenerationTarget) -> str:
    function_sections = "\n\n".join(
        f"Function: {changed_function.function_name}\n"
        f"Lines: {changed_function.start_line}-{changed_function.end_line}\n"
        f"{changed_function.source_code}"
        for changed_function in target.changed_functions
    )

    return (
        "You are an automated unit test generation system.\n\n"

        "STRICT OUTPUT FORMAT RULES:\n"
        "1. Output ONLY executable test code.\n"
        "2. Do NOT output markdown.\n"
        "3. Do NOT output explanations.\n"
        "4. Do NOT output prose.\n"
        "5. Do NOT include code fences.\n\n"

        "TEST STRUCTURE RULES (VERY IMPORTANT):\n"
        "For EACH source function you MUST generate EXACTLY ONE test function.\n\n"

        "Naming rule:\n"
        "test_<function_name>\n\n"

        "Example:\n"
        "If the function name is fibonacci, the test must be:\n"
        "def test_fibonacci():\n\n"

        "Inside that single test function you must include multiple assertions\n"
        "to test different cases.\n\n"

        "Example structure:\n"
        "def test_fibonacci():\n"
        "    assert fibonacci(0) == 0\n"
        "    assert fibonacci(1) == 1\n"
        "    assert fibonacci(5) == 5\n"
        "    with pytest.raises(ValueError):\n"
        "        fibonacci(-1)\n\n"

        "Never create multiple test functions for the same source function.\n\n"

        f"Programming language: {target.language}\n"
        f"Test framework: {target.framework}\n\n"

        f"Source file:\n{target.source_file}\n\n"

        f"Changed functions:\n{function_sections}\n\n"

        f"Relevant imports from source:\n{target.imports or 'None'}\n\n"

        f"Helper functions available:\n{target.helper_functions or 'None'}\n\n"

        f"Existing tests (do not duplicate):\n{target.existing_tests or 'None'}\n\n"

        "Generate the unit tests now."
    )


def _language_fence(language: str) -> str:
    if language == "python":
        return "python"
    if language == "javascript":
        return "javascript"
    return "typescript"
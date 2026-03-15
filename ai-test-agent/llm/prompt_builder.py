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
        "You are an automated test generation system.\n"
        "Your task is to generate unit tests for modified functions in a pull request.\n\n"

        "STRICT OUTPUT RULES:\n"
        "1. Output ONLY valid executable test code.\n"
        "2. Do NOT output explanations.\n"
        "3. Do NOT output markdown.\n"
        "4. Do NOT output comments outside code.\n"
        "5. Do NOT output prose.\n"
        "6. The output MUST be syntactically valid.\n"
        "7. The output MUST be directly runnable by the specified framework.\n"
        "8. Do NOT include backticks or code fences.\n\n"

        f"Programming language: {target.language}\n"
        f"Test framework: {target.framework}\n\n"

        "TEST GENERATION RULES:\n"
        "- Generate deterministic unit tests only.\n"
        "- Avoid randomness and external dependencies.\n"
        "- Mock external systems when needed.\n"
        "- Do not test unrelated functionality.\n"
        "- Each test should focus on a single behavior.\n"
        "- Use clear and descriptive test function names.\n"
        "- Import the source module correctly.\n"
        "- Do not modify application code.\n"
        "- Do not duplicate existing tests.\n\n"

        f"Source file:\n{target.source_file}\n\n"

        f"Changed functions:\n{function_sections}\n\n"

        f"Relevant imports from source:\n{target.imports or 'None'}\n\n"

        f"Helper functions available:\n{target.helper_functions or 'None'}\n\n"

        f"Existing tests (do not duplicate):\n{target.existing_tests or 'None'}\n\n"

        "Generate unit tests now."
    )


def _language_fence(language: str) -> str:
    if language == "python":
        return "python"
    if language == "javascript":
        return "javascript"
    return "typescript"

from __future__ import annotations

from context.repo_context import GenerationTarget


def build_prompt(target: GenerationTarget) -> str:
    function_sections = "\n\n".join(
        f"Function: {changed_function.function_name}\n"
        f"Lines: {changed_function.start_line}-{changed_function.end_line}\n"
        f"```{_language_fence(target.language)}\n{changed_function.source_code}\n```"
        for changed_function in target.changed_functions
    )

    return (
        "You are generating unit tests for a pull request change.\n"
        f"Programming language: {target.language}\n"
        f"Test framework: {target.framework}\n"
        "Return only valid test code with no markdown fences and no explanation.\n"
        "Do not modify application code.\n"
        "Do not duplicate existing tests.\n"
        "Only generate deterministic unit tests.\n"
        "Prefer focused mocks over integration behavior.\n"
        "Use imports that match the source module path.\n\n"
        f"Source file: {target.source_file}\n\n"
        f"Changed functions:\n{function_sections}\n\n"
        f"Relevant imports from source:\n{target.imports or 'None'}\n\n"
        f"Relevant helper functions:\n{target.helper_functions or 'None'}\n\n"
        f"Existing tests to avoid duplicating:\n{target.existing_tests or 'None'}\n"
    )


def _language_fence(language: str) -> str:
    if language == "python":
        return "python"
    if language == "javascript":
        return "javascript"
    return "typescript"

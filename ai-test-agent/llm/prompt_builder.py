from __future__ import annotations

from context.repo_context import GenerationTarget


def build_prompt(target: GenerationTarget) -> str:
    function_sections = "\n\n".join(
        f"Function: {changed_function.function_name}\n"
        f"Lines: {changed_function.start_line}-{changed_function.end_line}\n"
        f"{changed_function.source_code}"
        for changed_function in target.changed_functions
    )
    return(
        "You are an automated unit test generation system designed for production-grade codebases.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Output ONLY executable test code.\n"
        "2. Do NOT output markdown.\n"
        "3. Do NOT output explanations or prose.\n"
        "4. Do NOT include code fences.\n"
        "5. The output must be directly runnable.\n\n"

        "TEST STRUCTURE RULES (CRITICAL):\n"
        "1. For EACH source function, generate EXACTLY ONE test function.\n"
        "2. NEVER create multiple test functions for the same source function.\n"
        "3. Naming rule:\n"
        "   test_<function_name>\n\n"

        "4. Inside each test function, include MULTIPLE assertions covering:\n"
        "   - Normal cases\n"
        "   - Edge cases\n"
        "   - Boundary conditions\n"
        "   - Invalid inputs (if applicable)\n"
        "   - Type variations (if dynamically typed language)\n"
        "   - Error handling using pytest.raises where appropriate\n\n"

        "5. Ensure HIGH COVERAGE:\n"
        "   - Cover all branches (if/else paths)\n"
        "   - Cover all return conditions\n"
        "   - Cover failure paths\n"
        "   - Cover minimum, maximum, and empty inputs\n\n"

        "6. Use deterministic inputs ONLY (no randomness, no time-based values).\n\n"

        "7. Do NOT duplicate existing tests.\n\n"

        "ASSERTION RULES:\n"
        "1. Use direct assertions (assert ...).\n"
        "2. For floating point comparisons, use approximate comparison if needed.\n"
        "3. For exceptions:\n"
        "   with pytest.raises(ExpectedException):\n\n"

        "4. Avoid redundant assertions.\n"
        "5. Each assertion must test a distinct behavior.\n\n"

        "ASSERTION STRENGTH RULE:\n"
        "Do NOT write weak or meaningless assertions.\n"
        "Avoid assertions like:\n"
        "- assert result is not None\n"
        "- assert isinstance(result, ... ) unless critical\n"
        "Each assertion must strictly validate correctness of behavior.\n\n"

        "IMPORT RULES:\n"
        "1. Include required imports (e.g., pytest) ONLY if necessary.\n"
        "2. Do NOT re-import the source module unless required.\n\n"

        "QUALITY RULES:\n"
        "1. Tests must fail if the implementation is incorrect.\n"
        "2. Avoid trivial assertions that always pass.\n"
        "3. Do NOT assume behavior not present in the code.\n"
        "4. Infer expected behavior strictly from the given source code.\n\n"

        "BEHAVIORAL VALIDATION REQUIREMENT:\n"
        "Tests must validate the logical correctness of the function, not just match hardcoded outputs.\n"
        "Where applicable, verify properties, invariants, and relationships between inputs and outputs.\n\n"

        "INPUT DIVERSITY REQUIREMENT:\n"
        "Each test function must use a diverse set of inputs.\n"
        "Do NOT reuse similar values.\n"
        "Ensure variation across:\n"
        "- Different magnitudes\n"
        "- Different formats\n"
        "- Different categories of inputs\n\n"

        "SKEPTICISM REQUIREMENT:\n"
        "Do NOT assume the implementation is correct.\n"
        "Design tests as if the function may contain bugs.\n"
        "Your goal is to expose incorrect behavior, not confirm correctness.\n\n"

        "IMPLICIT CONTRACT TESTING:\n"
        "If the function implies constraints (e.g., age >= 18, valid email format),\n"
        "you MUST test violations of those constraints explicitly.\n\n"

        "ADVERSARIAL TESTING REQUIREMENT (MANDATORY):\n"
        "The tests MUST attempt to break the function by providing invalid, extreme, and unexpected inputs.\n"
        "You MUST actively look for inputs that can cause:\n"
        "- Exceptions or crashes\n"
        "- Incorrect return values\n"
        "- Type errors\n"
        "- Boundary failures\n"
        "- Logical inconsistencies\n\n"

        "You MUST include test cases with:\n"
        "- None inputs (if applicable)\n"
        "- Empty strings, empty collections\n"
        "- Extremely large or small values\n"
        "- Wrong data types (e.g., string instead of int)\n"
        "- Malformed inputs (e.g., invalid email formats)\n"
        "- Negative values where not expected\n\n"

        "If the function does not explicitly handle these cases, tests should assert the actual behavior (for example: including failures using pytest.raises if needed).\n\n"

        "EDGE CASE EXPECTATIONS:\n"
        "You MUST consider:\n"
        "- Empty inputs\n"
        "- Null/None inputs (if applicable)\n"
        "- Boundary numeric values\n"
        "- Invalid formats (e.g., malformed strings)\n"
        "- Large inputs if relevant\n"
        "- Case sensitivity (for strings)\n\n"

        "REDUNDANCY CONSTRAINT:\n"
        "Avoid duplicate or overlapping test cases.\n"
        "Each test input must cover a unique scenario or edge case.\n\n"

        "CONSTRAINTS:\n"
        "- Follow the exact function signatures.\n"
        "- Do not modify source functions.\n"
        "- Do not introduce external dependencies.\n"
        "- Keep tests isolated and independent.\n\n"

        f"Programming language: {target.language}\n"
        f"Test framework: {target.framework}\n\n"

        f"Source file:\n{target.source_file}\n\n"

        f"Changed functions:\n{function_sections}\n\n"

        f"Relevant imports from source:\n{target.imports or 'None'}\n\n"

        f"Helper functions available:\n{target.helper_functions or 'None'}\n\n"

        f"Existing tests (DO NOT DUPLICATE):\n{target.existing_tests or 'None'}\n\n"

        "Generate the unit tests now."


    )
    '''return (
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
        "to test different cases. make sure to cover every case and inclucde all edge cases as well\n\n"

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
    ) '''

def _language_fence(language: str) -> str:
    if language == "python":
        return "python"
    if language == "javascript":
        return "javascript"
    return "typescript"
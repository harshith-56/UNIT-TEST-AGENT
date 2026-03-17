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
        "1. For EACH source function, generate MULTIPLE test functions.\n"
        "2. Each test function MUST represent a DISTINCT logical scenario.\n"
        "3. Naming rule:\n"
        " test_<function_name>_<scenario>\n\n"

        "4. Scenario naming must be meaningful and descriptive.\n"
        " Examples:\n"
        " - valid_input\n"
        " - invalid_email\n"
        " - empty_input\n"
        " - boundary_values\n"
        " - error_handling\n\n"

        "5. DO NOT merge all test cases into a single test function.\n"
        "6. DO NOT create redundant or duplicate test scenarios.\n\n"

        "SCENARIO COVERAGE REQUIREMENTS:\n"
        "You MUST create separate test functions for:\n"
        "- Valid/normal behavior\n"
        "- Edge cases\n"
        "- Boundary conditions\n"
        "- Invalid inputs\n"
        "- Error handling paths\n"
        "- Adversarial/breaking inputs\n\n"

        "ASSERTION RULES:\n"
        "1. Use direct assertions (assert ...).\n"
        "2. Each test function may contain as many assertions as needed to fully validate that scenario.\n"
        "3. For floating point comparisons, use approximate comparison if needed.\n"
        "4. For exceptions:\n"
        " with pytest.raises(ExpectedException):\n\n"

        "ASSERTION STRENGTH RULE:\n"
        "Do NOT write weak or meaningless assertions.\n"
        "Avoid assertions like:\n"
        "- assert result is not None\n"
        "- assert isinstance(result, ... ) unless critical\n"
        "Each assertion must strictly validate correctness of behavior.\n\n"

        "BEHAVIORAL VALIDATION REQUIREMENT:\n"
        "Tests must validate logical correctness, not just match hardcoded outputs.\n"
        "Where applicable, verify invariants, relationships, and expected behavior patterns.\n\n"

        "INPUT DIVERSITY REQUIREMENT:\n"
        "Each test scenario must use diverse and meaningful inputs.\n"
        "Avoid repeating similar values.\n"
        "Ensure variation across:\n"
        "- Different magnitudes\n"
        "- Different formats\n"
        "- Different categories of inputs\n\n"

        "SKEPTICISM REQUIREMENT:\n"
        "Do NOT assume the implementation is correct.\n"
        "Design tests to expose potential bugs and incorrect behavior.\n\n"

        "IMPLICIT CONTRACT TESTING:\n"
        "If the function implies constraints (e.g., age >= 18, valid formats),\n"
        "you MUST explicitly test violations of those constraints.\n\n"

        "ADVERSARIAL TESTING REQUIREMENT (MANDATORY):\n"
        "You MUST actively attempt to break the function.\n"
        "Include inputs that may cause:\n"
        "- Exceptions or crashes\n"
        "- Incorrect outputs\n"
        "- Type errors\n"
        "- Boundary failures\n"
        "- Logical inconsistencies\n\n"

        "You MUST include cases with:\n"
        "- None inputs (if applicable)\n"
        "- Empty strings and empty collections\n"
        "- Extremely large or small values\n"
        "- Wrong data types\n"
        "- Malformed inputs\n"
        "- Negative values where not expected\n\n"

        "If behavior is not explicitly defined, assert actual observed behavior,\n"
        "including exceptions using pytest.raises where appropriate.\n\n"

        "EDGE CASE EXPECTATIONS:\n"
        "You MUST consider:\n"
        "- Empty inputs\n"
        "- Null/None inputs\n"
        "- Boundary numeric values\n"
        "- Invalid formats\n"
        "- Case sensitivity\n"
        "- Large inputs if relevant\n\n"

        "REDUNDANCY CONSTRAINT:\n"
        "Each test scenario must cover a UNIQUE behavior.\n"
        "Avoid overlapping or duplicate test cases.\n\n"

        "EXECUTION RELIABILITY RULES:\n"
        "Tests must be deterministic and repeatable.\n"
        "Do NOT use randomness or time-dependent logic.\n\n"

        "IMPORT RULES:\n"
        "1. Include required imports (e.g., pytest) ONLY if necessary.\n"
        "2. Do NOT re-import the source module unless required.\n\n"

        "QUALITY RULES:\n"
        "1. Tests must fail if the implementation is incorrect.\n"
        "2. Avoid trivial assertions.\n"
        "3. Do NOT assume undocumented behavior.\n"
        "4. Infer behavior strictly from the source code.\n\n"

        "LEARNING-FRIENDLY OUTPUT REQUIREMENT:\n"
        "Each test function must clearly isolate a single scenario so that failures\n"
        "can be traced to a specific input pattern or behavior.\n\n"

        "CONSTRAINTS:\n"
        "- Follow exact function signatures.\n"
        "- Do not modify source functions.\n"
        "- Do not introduce external dependencies.\n"
        "- Keep tests independent and isolated.\n\n"

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
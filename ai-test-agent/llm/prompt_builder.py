from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from agent.config import test_framework_for_language
from context.dependency_resolver import DependencyContext
from context.repo_context import GenerationTarget
from context.token_budget import MAX_CONTEXT_TOKENS, MAX_INPUT_TOKENS, estimate_tokens, trim_rules_to_budget


class SkipGeneration(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMInput:
    function_name: str
    language: str
    test_framework: str
    primary_code_block: str
    dependencies: list[str]
    project_rules: list[str]
    pr_rules: list[str]
    existing_tests: str
    import_hints: list[str]


def build_llm_input(target: GenerationTarget, generated_tests_dir: Path | str = Path("tests/ai_generated")) -> LLMInput:
    dependency_entries = [_dependency_entry(dependency) for dependency in target.dependencies]
    project_rules, pr_rules = _fit_context_rules(target.project_rules, target.pr_rules)
    existing_tests = _fit_existing_tests(_build_existing_tests_reference(target))

    llm_input = LLMInput(
        function_name=target.test_id,
        language=target.language,
        test_framework=test_framework_for_language(target.language),
        primary_code_block=target.function_change.source_code.strip(),
        dependencies=[entry["active"] for entry in dependency_entries],
        project_rules=project_rules,
        pr_rules=pr_rules,
        existing_tests=existing_tests,
        import_hints=_build_import_hints(target, Path(generated_tests_dir)),
    )
    return _fit_llm_input_to_budget(llm_input, dependency_entries)


def build_prompt(llm_input: LLMInput) -> str:
    dependencies = "\n\n".join(llm_input.dependencies) or "- None"
    import_hints = "\n".join(f"- {hint}" for hint in llm_input.import_hints) or "- None"
    existing_tests = llm_input.existing_tests.strip() or "None"

    project_context = "\n".join(f"- {r}" for r in llm_input.project_rules) or "- None"
    pr_context = "\n".join(f"- {r}" for r in llm_input.pr_rules) or "- None"

    is_repair = "Repair only these failing tests:" in existing_tests

    if not is_repair:
        instructions = f"Generate 3 to 6 production-grade unit tests for {llm_input.function_name}."
    else:
        instructions = "Fix ONLY failing tests. Keep EXACT same names."

    return (
        "You are an EXPERT production-grade UNIT TEST generator.\n"
        "You write STRICT, HIGH-COVERAGE, FAILURE-DRIVEN tests.\n"
        "You do NOT guess. You do NOT improvise. You follow code strictly.\n\n"

        f"LANGUAGE: {llm_input.language}\n"
        f"TEST FRAMEWORK: {llm_input.test_framework}\n\n"

        "====================\n"
        "ABSOLUTE RULES (NON-NEGOTIABLE)\n"
        "====================\n"
        "- Output ONLY executable test code\n"
        "- NO markdown\n"
        "- NO explanations\n"
        "- NO comments\n"
        "- NO placeholders (***, ..., ???, TODO, TBD)\n"
        "- NEVER use 'your_module'\n"
        "- ALWAYS provide COMPLETE arguments\n"
        "- NEVER invent fields or parameters\n"
        "- ONLY use real imports from code, dependencies, or hints\n"
        "- NEVER cut off output mid-function — finish every function you start\n"
        "- NEVER generate more tests than you can complete — fewer complete tests beats more broken ones\n\n"

        "====================\n"
        "TEST TYPE (STRICT)\n"
        "====================\n"
        "- Generate ONLY UNIT tests\n"
        "- DO NOT use TestClient\n"
        "- DO NOT make HTTP calls\n"
        "- DO NOT create real DB engines\n\n"

        "====================\n"
        "PROJECT CONTEXT\n"
        "====================\n"
        f"{project_context}\n\n"

        "====================\n"
        "PR CONTEXT\n"
        "====================\n"
        f"{pr_context}\n\n"

        "====================\n"
        "FUNCTION UNDER TEST\n"
        "====================\n"
        f"{llm_input.primary_code_block}\n\n"

        "====================\n"
        "IMPORT HINTS (MANDATORY — USE ONLY THESE)\n"
        "====================\n"
        f"{import_hints}\n\n"
        "IMPORT RULES:\n"
        "- You MUST import ONLY from the module paths listed above\n"
        "- NEVER invent module paths not listed above\n"
        "- NEVER use the repository or project name as a package prefix\n"
        "constants, config unless explicitly listed above\n"
        "- If a symbol is not importable from the listed paths, do NOT import it\n"
        "- For Python: use exactly the module path shown (e.g. 'backend.main')\n"
        "- For JS/TS: use exactly the relative path shown (e.g. './Signup')\n\n"

        "====================\n"
        "DEPENDENCIES\n"
        "====================\n"
        f"{dependencies}\n\n"

        "====================\n"
        "EXISTING TESTS (DO NOT DUPLICATE)\n"
        "====================\n"
        f"{existing_tests}\n\n"

        "====================\n"
        "TEST GENERATION RULES\n"
        "====================\n"
        f"{instructions}\n"
        "- Each test must cover a UNIQUE scenario\n"
        "- Cover ALL branches and return paths\n"
        "- Cover valid, edge, boundary, invalid, and error cases\n\n"

        "====================\n"
        "ASSERTION STRENGTH (CRITICAL)\n"
        "====================\n"
        "- DO NOT write weak assertions\n"
        "- DO NOT use 'assert result is not None'\n"
        "- Assertions must validate REAL behavior\n\n"

        "====================\n"
        "INPUT RULES\n"
        "====================\n"
        "- Use ONLY visible schema/classes\n"
        "- NEVER leave arguments incomplete\n"
        "- If unsure, use VALID dummy values\n"
        "- Example:\n"
        "  SignupRequest(username='user', email='a@b.com', password='validpass123')\n\n"

        "====================\n"
        "BEHAVIOR RULES\n"
        "====================\n"
        "- Infer behavior ONLY from given code\n"
        "- DO NOT assume hidden logic\n"
        "- DO NOT invent outputs\n\n"

        "====================\n"
        "ADVERSARIAL TESTING (MANDATORY)\n"
        "====================\n"
        "- Try to BREAK the function\n"
        "- Include invalid inputs\n"
        "- Include wrong types\n"
        "- Include edge cases\n"
        "- Include boundary values\n\n"

        "====================\n"
        "MOCKING RULES\n"
        "====================\n"
        "- PURE functions → NO mocking\n"
        "- SIDE EFFECT → mock external dependencies ONLY\n"
        "- DO NOT mock internal logic\n\n"

        "====================\n"
        "GENERATOR FUNCTIONS\n"
        "====================\n"
        "- If the function under test contains 'yield', it is a generator\n"
        "- Generators MUST be tested with next() or list():\n"
        "  CORRECT:   db = next(get_db())\n"
        "  CORRECT:   results = list(get_items())\n"
        "  WRONG:     db = get_db(); db.query(...)  ← this will crash\n"
        "  WRONG:     result = get_db(); assert result.execute()\n"
        "- NEVER call methods directly on the return value of a generator function\n\n"

        "====================\n"
        "COMPLETENESS (NON-NEGOTIABLE)\n"
        "====================\n"
        "- Your output must be 100% complete with no cut-off lines\n"
        "- The last line must be a complete statement — closing paren, bracket, or a pass\n"
        "- Never end output with: =, ==, ,, (, and, or, not\n"
        "- If running out of space: stop BEFORE starting a new test, not in the middle of one\n\n"

        "====================\n"
        "FINAL OUTPUT\n"
        "====================\n"
        "- ONLY executable test code\n"
    )


def build_retry_prompt(llm_input: LLMInput, failure_reason: str, attempt_number: int) -> str:
    base = build_prompt(llm_input)

    correction = (
        "\n\n====================\n"
        "PREVIOUS OUTPUT REJECTED\n"
        "====================\n"
        f"SPECIFIC FAILURE REASON: {failure_reason}\n\n"
        "You MUST fix exactly this issue. Do not change anything else.\n\n"
        "General rules that must also hold:\n"
        "- Remove ALL placeholders (***, ..., ???)\n"
        "- Provide COMPLETE arguments\n"
        "- DO NOT use your_module\n"
        "- Use valid imports from hints only\n"
        "- Generate ONLY UNIT tests\n"
        "- DO NOT use DB, TestClient, or create_engine\n"
        "- Ensure every function is complete before starting the next\n"
    )

    if failure_reason == "truncated":
        correction += (
            "\nYOUR OUTPUT WAS CUT OFF. Rules:\n"
            "- Generate FEWER tests this time — 2 or 3 maximum\n"
            "- Make absolutely sure the last test is fully closed\n"
            "- The final line must be a complete statement\n"
        )

    if failure_reason in ("fake_import_your_module", "placeholder_import", "fake_import_js"):
        correction += (
            "\nYOUR IMPORTS WERE INVALID. Rules:\n"
            "- Only import from paths listed in IMPORT HINTS above\n"
            "- Never use your_module, example_module, or any placeholder path\n"
        )

    if "generator" in failure_reason or failure_reason == "generator_not_unwrapped":
        correction += (
            "\nGENERATOR USAGE WAS WRONG. Rules:\n"
            "- Use next(function()) to get a single value\n"
            "- Use list(function()) to get all values\n"
            "- Never call methods on the raw generator object\n"
        )

    if attempt_number >= 7:
        correction += (
            "\n\nCRITICAL — attempt {attempt_number} of 12:\n"
            "- Generate ONLY 2 simple tests\n"
            "- Use only basic assert statements\n"
            "- No complex mocking — patch only what is absolutely necessary\n"
            "- The last line of your output must close all open blocks\n"
        ).format(attempt_number=attempt_number)

    return base + correction


def _prompt_rules(llm_input: LLMInput) -> str:
    project_rules = "\n".join(f"- {rule}" for rule in llm_input.project_rules[:3]) or "- None"
    pr_rules = "\n".join(f"- {rule}" for rule in llm_input.pr_rules[:3]) or "- None"
    return (
        "- Generate UNIT tests only; mock side effects instead of using real DB engines, network calls, or file writes.\n"
        "- Ban placeholders such as ***, ..., ???, TODO, and TBD.\n"
        "- Never use 'your_module'.\n"
        "- Use complete constructor and function arguments.\n"
        "- Use real imports from the code, dependencies, or import hints.\n"
        "- If the function uses yield, treat it as a generator and test it via iteration, for example list(function_call()).\n"
        f"{project_rules}\n"
        f"{pr_rules}"
    )


def _build_existing_tests_reference(target: GenerationTarget) -> str:
    if target.generation_mode != "repair":
        return target.existing_tests_text

    lines: list[str] = []
    if target.repair_test_names:
        lines.append("Repair only these failing tests:")
        lines.extend(f"- {name}" for name in target.repair_test_names)
    if target.repair_notes:
        lines.append("Failure notes:")
        lines.extend(f"- {note}" for note in target.repair_notes)
    if target.existing_tests_text.strip():
        lines.append("Existing tests:")
        lines.append(target.existing_tests_text.strip())
    return "\n".join(lines).strip()


def _build_import_hints(target: GenerationTarget, generated_tests_dir: Path) -> list[str]:
    if target.language == "python":
        module_path = _python_module_path(target.source_file)
        func_name = target.function_change.function_name.split(".")[-1]
        class_name = target.function_change.enclosing_class_name
        hints = [f"Source file: {target.source_file}"]
        if module_path:
            hints.append(f"Module path: {module_path}")
            if class_name:
                hints.append(f"Import the class: from {module_path} import {class_name}")
            else:
                hints.append(f"Import the function: from {module_path} import {func_name}")
            hints.append(f"Only import from '{module_path}' — do NOT invent other module paths")
        return hints

    # Normalize source_file to a relative path (strip absolute prefix / drive letter)
    source_file_str = target.source_file.replace("\\", "/")
    source_file_str = re.sub(r"^[A-Za-z]:[/\\]", "", source_file_str)
    source_file_str = source_file_str.lstrip("/")
    source_no_suffix = PurePosixPath(source_file_str).with_suffix("")
    generated_dir = PurePosixPath(generated_tests_dir.as_posix())
    relative_import = posixpath.relpath(source_no_suffix.as_posix(), generated_dir.as_posix())
    if not relative_import.startswith("."):
        relative_import = f"./{relative_import}"
    symbol_name = target.function_change.enclosing_class_name or target.function_change.function_name.split(".")[0]
    return [
        f"Source file: {target.source_file}",
        f"Import path (relative from test file): '{relative_import}'",
        f"Example: import {symbol_name} from '{relative_import}'",
        f"ONLY use this exact path — never use absolute or repo-name paths",
    ]


def _python_module_path(source_file: str) -> str:
    path = PurePosixPath(source_file)
    if path.suffix != ".py":
        return ""
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _dependency_entry(dependency: DependencyContext) -> dict[str, str]:
    location = f" ({dependency.source_file})" if dependency.source_file else ""
    full = f"- {dependency.name}{location}\n{dependency.content.strip()}"
    summary = f"- {dependency.name}{location}\n{dependency.summary.strip()}"
    return {"active": full, "summary": summary}


def _fit_context_rules(project_rules: list[str], pr_rules: list[str]) -> tuple[list[str], list[str]]:
    if not project_rules and not pr_rules:
        return [], []

    project_budget = MAX_CONTEXT_TOKENS // 2
    pr_budget = MAX_CONTEXT_TOKENS - project_budget

    trimmed_project = trim_rules_to_budget(project_rules, project_budget)
    trimmed_pr = trim_rules_to_budget(pr_rules, pr_budget)

    return trimmed_project, trimmed_pr


def _fit_existing_tests(existing_tests: str) -> str:
    if not existing_tests.strip():
        return ""

    kept_lines: list[str] = []
    used_tokens = 0
    for line in existing_tests.splitlines():
        line_tokens = estimate_tokens(line)
        if kept_lines and used_tokens + line_tokens > 250:
            break
        kept_lines.append(line)
        used_tokens += line_tokens
    return "\n".join(kept_lines).strip()


def _fit_llm_input_to_budget(llm_input: LLMInput, dependency_entries: list[dict[str, str]]) -> LLMInput:
    fitted_input = llm_input

    while estimate_tokens(build_prompt(fitted_input)) > MAX_INPUT_TOKENS:
        trimmed_input = _trim_context(fitted_input)
        if trimmed_input != fitted_input:
            fitted_input = trimmed_input
            continue

        summarized_input = _summarize_dependencies(fitted_input, dependency_entries)
        if summarized_input != fitted_input:
            fitted_input = summarized_input
            continue

        reduced_existing_tests = _trim_existing_tests(fitted_input)
        if reduced_existing_tests != fitted_input:
            fitted_input = reduced_existing_tests
            continue

        break

    if estimate_tokens(build_prompt(fitted_input)) > MAX_INPUT_TOKENS:
        raise SkipGeneration(f"Prompt exceeds MAX_INPUT_TOKENS for {llm_input.function_name}")

    return fitted_input


def _trim_context(llm_input: LLMInput) -> LLMInput:
    if llm_input.pr_rules:
        return replace(llm_input, pr_rules=llm_input.pr_rules[:-1])
    if llm_input.project_rules:
        return replace(llm_input, project_rules=llm_input.project_rules[:-1])
    return llm_input


def _summarize_dependencies(llm_input: LLMInput, dependency_entries: list[dict[str, str]]) -> LLMInput:
    active_dependencies = list(llm_input.dependencies)
    for index, current_dependency in enumerate(active_dependencies):
        summary_dependency = dependency_entries[index]["summary"]
        if current_dependency != summary_dependency:
            active_dependencies[index] = summary_dependency
            return replace(llm_input, dependencies=active_dependencies)
    return llm_input


def _trim_existing_tests(llm_input: LLMInput) -> LLMInput:
    if not llm_input.existing_tests.strip():
        return llm_input

    lines = llm_input.existing_tests.splitlines()
    if len(lines) <= 1:
        return replace(llm_input, existing_tests="")
    return replace(llm_input, existing_tests="\n".join(lines[:-1]).strip())

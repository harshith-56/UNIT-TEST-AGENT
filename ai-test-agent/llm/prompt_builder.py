from __future__ import annotations

import posixpath
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
    project_context = "\n".join(f"- {r}" for r in llm_input.project_rules) or "- None"
    pr_context = "\n".join(f"- {r}" for r in llm_input.pr_rules) or "- None"
    dependencies = "\n\n".join(llm_input.dependencies) or "- None"
    import_hints = "\n".join(f"- {h}" for h in llm_input.import_hints) or "- None"
    existing_tests = llm_input.existing_tests.strip() or "None"

    is_repair = "Repair only these failing tests:" in existing_tests

    if not is_repair:
        instructions = f"Generate 3 to 7 high quality unit tests for {llm_input.function_name}."
    else:
        instructions = "Fix ONLY failing tests. Keep same names."

    return (
        f"LANGUAGE: {llm_input.language}\n"
        f"TEST FRAMEWORK: {llm_input.test_framework}\n\n"

        "CRITICAL RULES (MUST FOLLOW):\n"
        "- Use ONLY real imports from given code or dependencies\n"
        "- NEVER use 'your_module' while importing\n"
        "- NEVER use placeholders like ***, ..., ???,etc\n"
        "- ALWAYS provide complete function/class arguments\n"
        "- If unsure, use valid dummy values (e.g. strings, numbers)\n"
        "- DO NOT redefine classes already provided\n\n"

        "PROJECT CONTEXT:\n"
        f"{project_context}\n\n"

        "PR CONTEXT:\n"
        f"{pr_context}\n\n"

        "FUNCTION:\n"
        f"{llm_input.primary_code_block}\n\n"

        "IMPORT HINTS:\n"
        f"{import_hints}\n\n"

        "DEPENDENCIES:\n"
        f"{dependencies}\n\n"

        "EXISTING TESTS:\n"
        f"{existing_tests}\n\n"

        "INSTRUCTIONS:\n"
        f"{instructions}\n"
        "- Cover valid, edge, and error cases\n"
        "- Use real class constructors from schema\n"
        "- Example valid object:\n"
        "  SignupRequest(username='user', email='a@b.com', password='validpass123')\n\n"

        "MOCKING:\n"
        "- Mock external dependencies only (DB, API)\n"
        "- Do NOT mock internal logic\n\n"

        "OUTPUT:\n"
        "- Only executable test code\n"
        "- No markdown\n"
        "- No explanation\n"
    )


def build_retry_prompt(llm_input: LLMInput, failure_reason: str, attempt_number: int) -> str:
    dependencies = "\n\n".join(llm_input.dependencies) or "- None"
    import_hints = "\n".join(f"- {hint}" for hint in llm_input.import_hints) or "- None"
    existing_tests = llm_input.existing_tests.strip() or "None"
    condensed_rules = _retry_rules(llm_input)
    correction_lines = [
        "- Previous output was invalid or incomplete.",
        f"- Previous failure reason: {failure_reason}.",
        "- NEVER use 'your_module' or any made-up local module.",
        "- ONLY import from visible code, dependency snippets, import hints, or known test framework modules.",
        "- NEVER use placeholders such as ***, ..., ???, TODO, or TBD.",
        "- Every constructor and function call must have complete arguments.",
        "- If a value is required, use a realistic dummy value instead of omitting it.",
        "- Output complete executable tests only.",
    ]
    if attempt_number >= 5:
        correction_lines.extend(
            [
                "- If unsure about imports, infer them from the source file path or import hints.",
                "- Prefer smaller, correct tests over broad but invalid coverage.",
                "- Do not repeat a previous invalid structure.",
            ]
        )

    corrections = "\n".join(correction_lines)
    return (
        f"LANGUAGE: {llm_input.language}\n"
        f"TEST FRAMEWORK: {llm_input.test_framework}\n\n"
        "FUNCTION:\n"
        f"{llm_input.primary_code_block}\n\n"
        "IMPORT HINTS:\n"
        f"{import_hints}\n\n"
        "DEPENDENCIES:\n"
        f"{dependencies}\n\n"
        "EXISTING TESTS:\n"
        f"{existing_tests}\n\n"
        "KEEP THESE RULES:\n"
        f"{condensed_rules}\n\n"
        "TEST NAMING:\n"
        f"- Use names like test_{llm_input.function_name}_<scenario>\n"
        "- In repair mode, keep the same failing test names\n\n"
        "CORRECTIONS (STRICT):\n"
        f"{corrections}\n\n"
        "OUTPUT:\n"
        "- ONLY executable code\n"
        "- NO markdown\n"
        "- NO comments\n"
        "- NO explanation\n"
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
        symbol_name = target.function_change.enclosing_class_name or target.function_change.function_name.split(".")[0]
        hints = [f"Source file: {target.source_file}"]
        if module_path:
            hints.append(f"Module path for imports: {module_path}")
            hints.append(f"Prefer imports like: from {module_path} import {symbol_name}")
        return hints

    source_no_suffix = PurePosixPath(target.source_file).with_suffix("")
    generated_dir = PurePosixPath(generated_tests_dir.as_posix())
    relative_import = posixpath.relpath(source_no_suffix.as_posix(), generated_dir.as_posix())
    if not relative_import.startswith("."):
        relative_import = f"./{relative_import}"
    symbol_name = target.function_change.enclosing_class_name or target.function_change.function_name.split(".")[0]
    return [
        f"Source file: {target.source_file}",
        f"Relative import from generated tests: {relative_import}",
        f"Prefer imports from '{relative_import}' for {symbol_name}",
    ]


def _python_module_path(source_file: str) -> str:
    path = PurePosixPath(source_file)
    if path.suffix != ".py":
        return ""
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _retry_rules(llm_input: LLMInput) -> str:
    rules: list[str] = []
    rules.extend(f"- {rule}" for rule in llm_input.project_rules[:3])
    rules.extend(f"- {rule}" for rule in llm_input.pr_rules[:3])
    return "\n".join(rules) or "- None"


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

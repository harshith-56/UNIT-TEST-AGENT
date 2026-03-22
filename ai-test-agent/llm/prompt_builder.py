from __future__ import annotations

from dataclasses import dataclass, replace

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


def build_llm_input(target: GenerationTarget) -> LLMInput:
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
    )
    return _fit_llm_input_to_budget(llm_input, dependency_entries)


def build_prompt(llm_input: LLMInput) -> str:
    project_context = "\n".join(f"- {rule}" for rule in llm_input.project_rules) or "- None"
    pr_context = "\n".join(f"- {rule}" for rule in llm_input.pr_rules) or "- None"
    dependencies = "\n\n".join(llm_input.dependencies) or "- None"
    existing_tests = llm_input.existing_tests.strip() or "None"

    is_repair = "Repair only these failing tests:" in existing_tests

    if not is_repair:
        instructions = (
            f"- Generate between 3 and 8 test functions for {llm_input.function_name}.\n"
            "- Each test must cover a UNIQUE scenario.\n"
            f"- Naming format: test_{llm_input.function_name}_<scenario>\n"
            "- MUST cover: valid case, edge case, boundary, invalid input, error handling.\n"
        )
    else:
        instructions = (
            "- Regenerate ONLY the failing tests listed below.\n"
            "- Keep EXACT same test names.\n"
        )

    return (
        f"LANGUAGE: {llm_input.language}\n"
        f"TEST FRAMEWORK: {llm_input.test_framework}\n\n"

        "PROJECT CONTEXT:\n"
        f"{project_context}\n\n"

        "PR CONTEXT:\n"
        f"{pr_context}\n\n"

        "FUNCTION UNDER TEST:\n"
        f"{llm_input.primary_code_block}\n\n"

        "DEPENDENCIES (ONLY USE IF NECESSARY):\n"
        f"{dependencies}\n\n"

        "EXISTING TESTS (REFERENCE ONLY):\n"
        f"{existing_tests}\n\n"

        "STRICT INSTRUCTIONS:\n"
        f"{instructions}"
        "- ONLY use information present in the given function and dependencies.\n"
        "- DO NOT assume behavior that is not explicitly visible.\n"
        "- DO NOT invent return values, exceptions, or side effects.\n"
        "- DO NOT redefine the source function or its dependencies.\n"
        "- DO NOT use 'your_module' or fake imports.\n"
        "- If required inputs are unclear, SKIP that test case.\n"
        "- Prefer simple, realistic inputs.\n"
        "- Assertions must reflect actual observable behavior.\n"
        "- Avoid comparing complex objects unless explicitly defined.\n"
        "- Avoid testing internal implementation details.\n"
        "- Tests must be deterministic and independent.\n"
        "- Use direct assertions only.\n\n"

        "OUTPUT RULES:\n"
        "- ONLY executable code\n"
        "- NO markdown\n"
        "- NO explanations\n"
        "- NO comments\n"
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

    leftover = MAX_CONTEXT_TOKENS - estimate_tokens("\n".join([*trimmed_project, *trimmed_pr]))

    if leftover > 0:
        trimmed_project = [*trimmed_project, *trim_rules_to_budget(project_rules[len(trimmed_project):], leftover)]
        leftover = MAX_CONTEXT_TOKENS - estimate_tokens("\n".join([*trimmed_project, *trimmed_pr]))

    if leftover > 0:
        trimmed_pr = [*trimmed_pr, *trim_rules_to_budget(pr_rules[len(trimmed_pr):], leftover)]

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
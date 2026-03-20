from __future__ import annotations

from context.repo_context import GenerationTarget
from context.token_budget import MAX_INPUT_TOKENS, estimate_tokens


def build_prompt(target: GenerationTarget) -> str:
    project_rules = list(target.project_rules)
    pr_rules = list(target.pr_rules)
    dependency_entries = [
        {
            "name": dependency.name,
            "source_file": dependency.source_file,
            "mode": dependency.mode,
            "content": dependency.content,
            "summary": dependency.summary,
        }
        for dependency in target.dependencies
    ]

    prompt = _render_prompt(target, project_rules, pr_rules, dependency_entries)

    while estimate_tokens(prompt) > MAX_INPUT_TOKENS and (pr_rules or project_rules):
        if pr_rules:
            pr_rules.pop()
        elif project_rules:
            project_rules.pop()
        prompt = _render_prompt(target, project_rules, pr_rules, dependency_entries)

    while estimate_tokens(prompt) > MAX_INPUT_TOKENS:
        converted = False
        for entry in sorted(dependency_entries, key=lambda item: -estimate_tokens(item["content"])):
            if entry["mode"] == "code":
                entry["mode"] = "summary"
                entry["content"] = entry["summary"]
                converted = True
                break
        if not converted:
            break
        prompt = _render_prompt(target, project_rules, pr_rules, dependency_entries)

    return prompt


def _render_prompt(target: GenerationTarget, project_rules: list[str], pr_rules: list[str], dependency_entries: list[dict[str, str | None]]) -> str:
    rules_block = "\n".join(f"- {rule}" for rule in project_rules) or "- None"
    pr_block = "\n".join(f"- {rule}" for rule in pr_rules) or "- None"
    dependencies_block = _render_dependencies(dependency_entries)
    existing_tests_block = "\n".join(f"- {name}" for name in target.existing_test_names) or "- None"
    repair_block = _render_repair_block(target)

    if target.repair_test_names:
        generation_rule = (
            f"Return only corrected implementations for these existing tests: {', '.join(target.repair_test_names)}.\n"
            "Keep the same test names unless the failure requires a direct rename to match the current function symbol."
        )
    else:
        generation_rule = (
            f"Generate 3 to 8 runnable tests for the target function using names test_{target.test_id}_<scenario>.\n"
            "Each test must cover unique behavior only and avoid redundant scenarios."
        )

    return (
        "Generate runnable unit test code only. No markdown. No explanations. No code fences.\n"
        f"{generation_rule}\n"
        "Coverage requirements: valid case, edge case, boundary, invalid input, error handling, adversarial input.\n"
        "Keep tests deterministic and isolated. Use direct assertions. For Python exceptions use pytest.raises.\n"
        "Do not regenerate unrelated tests.\n\n"
        "PROJECT CONTEXT:\n"
        f"{rules_block}\n\n"
        "PR CONTEXT:\n"
        f"{pr_block}\n\n"
        "FUNCTION CODE:\n"
        f"{target.function_change.context_code}\n\n"
        "DEPENDENCIES:\n"
        f"{dependencies_block}\n\n"
        "EXISTING TEST NAMES:\n"
        f"{existing_tests_block}\n\n"
        f"{repair_block}"
        "Return the test code now."
    )


def _render_dependencies(dependency_entries: list[dict[str, str | None]]) -> str:
    if not dependency_entries:
        return "- None"
    sections = []
    for entry in dependency_entries:
        source_file = f" ({entry['source_file']})" if entry["source_file"] else ""
        label = f"- {entry['name']}{source_file} [{entry['mode']}]"
        sections.append(f"{label}\n{entry['content']}")
    return "\n\n".join(sections)


def _render_repair_block(target: GenerationTarget) -> str:
    if not target.repair_test_names:
        return ""
    failure_notes = "\n".join(f"- {note}" for note in target.repair_notes) or "- None"
    failing_tests = "\n".join(f"- {name}" for name in target.repair_test_names)
    return (
        "REPAIR CONTEXT:\n"
        f"Failing test names:\n{failing_tests}\n"
        f"Failure notes:\n{failure_notes}\n\n"
    )
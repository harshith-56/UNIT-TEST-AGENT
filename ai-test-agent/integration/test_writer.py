from __future__ import annotations

from pathlib import Path
import re

from agent.config import AgentConfig
from generation.test_generator import GeneratedTest
from utils.file_utils import ensure_directory, sanitize_module_name


def write_generated_tests(
    repo_root: Path,
    generated_tests: list[GeneratedTest],
    config: AgentConfig
) -> list[Path]:

    ensure_directory(config.generated_tests_dir)

    written_paths: list[Path] = []

    for generated_test in generated_tests:

        file_name = _build_file_name(generated_test)
        destination = config.generated_tests_dir / file_name

        new_content = generated_test.content.strip() + "\n"

        if destination.exists():

            old_content = destination.read_text(encoding="utf-8")

            existing_imports, existing_body = _extract_imports_and_body(old_content)
            new_imports, new_body = _extract_imports_and_body(new_content)

            cleaned_body = _remove_existing_tests(
                existing_body,
                generated_test.function_names
            )

            merged_imports = _merge_imports(existing_imports, new_imports)

            final_content = (
                "\n".join(merged_imports)
                + "\n\n"
                + cleaned_body.rstrip()
                + "\n\n"
                + new_body
                + "\n"
            )

            destination.write_text(final_content, encoding="utf-8")

        else:

            imports, body = _extract_imports_and_body(new_content)

            final_content = "\n".join(imports) + "\n\n" + body + "\n"

            destination.write_text(final_content, encoding="utf-8")

        written_paths.append(destination.relative_to(repo_root))

    return written_paths


def _remove_existing_tests(content: str, function_names: list[str]) -> str:
    """
    Removes test_<function>() blocks for the functions being regenerated.
    """

    for name in function_names:

        pattern = rf"def\s+test_{name}\(.*?\):.*?(?=\ndef\s+test_|\Z)"

        content = re.sub(
            pattern,
            "",
            content,
            flags=re.DOTALL
        )

    return content


def _extract_imports_and_body(content: str) -> tuple[list[str], str]:
    """
    Splits imports and test body.
    """

    imports = []
    body_lines = []

    for line in content.splitlines():

        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        else:
            body_lines.append(line)

    return imports, "\n".join(body_lines).strip()


def _merge_imports(existing: list[str], new: list[str]) -> list[str]:
    """
    Deduplicate and sort imports.
    """

    merged = list(dict.fromkeys(existing + new))

    merged.sort()

    return merged


def _build_file_name(generated_test: GeneratedTest) -> str:

    module_name = sanitize_module_name(generated_test.source_file)

    if generated_test.language == "python":
        return f"test_ai_generated_{module_name}.py"

    if generated_test.language == "javascript":
        return f"ai_generated_{module_name}.test.js"

    return f"ai_generated_{module_name}.test.ts"
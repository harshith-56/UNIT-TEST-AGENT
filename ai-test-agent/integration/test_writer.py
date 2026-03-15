from __future__ import annotations

from pathlib import Path

from agent.config import AgentConfig
from generation.test_generator import GeneratedTest
from utils.file_utils import ensure_directory, sanitize_module_name


def write_generated_tests(repo_root: Path, generated_tests: list[GeneratedTest], config: AgentConfig) -> list[Path]:
    ensure_directory(config.generated_tests_dir)

    written_paths: list[Path] = []

    for generated_test in generated_tests:

        file_name = _build_file_name(generated_test)
        destination = config.generated_tests_dir / file_name

        # append tests into the same file instead of creating new numbered files
        if destination.exists():
            with destination.open("a", encoding="utf-8") as f:
                f.write("\n\n")
                f.write(generated_test.content)
        else:
            destination.write_text(generated_test.content, encoding="utf-8")

        written_paths.append(destination.relative_to(repo_root))

    return written_paths


def _build_file_name(generated_test: GeneratedTest) -> str:
    module_name = sanitize_module_name(generated_test.source_file)

    if generated_test.language == "python":
        return f"test_ai_generated_{module_name}.py"

    if generated_test.language == "javascript":
        return f"ai_generated_{module_name}.test.js"

    return f"ai_generated_{module_name}.test.ts"
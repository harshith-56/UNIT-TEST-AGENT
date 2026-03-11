from __future__ import annotations

from pathlib import Path

from agent.config import AgentConfig
from generation.test_generator import GeneratedTest
from utils.file_utils import ensure_directory, safe_write_text, sanitize_module_name


def write_generated_tests(repo_root: Path, generated_tests: list[GeneratedTest], config: AgentConfig) -> list[Path]:
    ensure_directory(config.generated_tests_dir)
    written_paths: list[Path] = []
    for generated_test in generated_tests:
        file_name = _build_file_name(generated_test)
        destination = _next_available_path(config.generated_tests_dir, file_name)
        safe_write_text(destination, generated_test.content, config.generated_tests_dir)
        written_paths.append(destination.relative_to(repo_root))
    return written_paths


def _build_file_name(generated_test: GeneratedTest) -> str:
    module_name = sanitize_module_name(generated_test.source_file)
    if generated_test.language == "python":
        return f"test_ai_generated_{module_name}.py"
    if generated_test.language == "javascript":
        return f"ai_generated_{module_name}.test.js"
    return f"ai_generated_{module_name}.test.ts"


def _next_available_path(base_dir: Path, file_name: str) -> Path:
    candidate = base_dir / file_name
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        next_candidate = base_dir / f"{candidate.stem}_{counter}{candidate.suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1

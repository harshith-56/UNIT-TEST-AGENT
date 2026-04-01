from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestRunResult:
    language: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def execute_tests(repo_root: Path, languages: set[str]) -> list[TestRunResult]:
    results: list[TestRunResult] = []
    if "python" in languages and shutil.which("python"):
        results.append(_run(
            ["python", "-m", "pytest", "tests/ai_generated/",
             "--tb=short", "-q", "--no-header"],
            repo_root, "python"
        ))
    if {"javascript", "typescript"} & languages and (repo_root / "package.json").exists() and shutil.which("npm"):
        results.append(_run(
            ["npm", "test", "--", "--runInBand",
             "--testPathPattern=tests/ai_generated",
             "--passWithNoTests"],
            repo_root, "javascript"
        ))
    return results


def has_failures(results: list[TestRunResult]) -> bool:
    return any(result.returncode != 0 for result in results)


def _run(command: list[str], repo_root: Path, language: str) -> TestRunResult:
    process = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return TestRunResult(
        language=language,
        command=command,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )

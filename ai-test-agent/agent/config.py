from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

TEST_FRAMEWORK_BY_LANGUAGE = {
    "python": "pytest",
    "javascript": "jest",
    "typescript": "jest",
}


@dataclass(frozen=True)
class AgentConfig:
    repo_root: Path
    generated_tests_dir: Path
    llm_api_url: str
    llm_api_key: str
    llm_model: str
    project_context_raw: str = ""
    llm_timeout_seconds: int = 90
    comment_on_pr: bool = False
    fail_on_test_failure: bool = False


def load_config() -> AgentConfig:
    repo_root = Path(os.getenv("AI_TEST_AGENT_REPO_ROOT", os.getcwd())).resolve()
    generated_tests_dir = repo_root / "tests" / "ai_generated"
    return AgentConfig(
        repo_root=repo_root,
        generated_tests_dir=generated_tests_dir,
        llm_api_url=os.getenv("LLM_API_URL", "").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        project_context_raw=os.getenv("AI_TEST_AGENT_PROJECT_CONTEXT", "").strip(),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
        comment_on_pr=os.getenv("AI_TEST_AGENT_COMMENT_ON_PR", "false").lower() == "true",
        fail_on_test_failure=os.getenv("AI_TEST_AGENT_FAIL_ON_TEST_FAILURE", "false").lower() == "true",
    )


def detect_language(file_path: str) -> str | None:
    return LANGUAGE_BY_EXTENSION.get(Path(file_path).suffix.lower())


def detect_languages(file_paths: Iterable[str]) -> set[str]:
    return {language for path in file_paths if (language := detect_language(path))}


def test_framework_for_language(language: str) -> str:
    return TEST_FRAMEWORK_BY_LANGUAGE[language]

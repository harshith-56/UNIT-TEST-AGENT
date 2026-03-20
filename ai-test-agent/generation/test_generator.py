from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import build_prompt
from utils.logger import get_logger
from validation.test_naming import extract_test_names


LOGGER = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 5
LLM_CALL_DELAY_SECONDS = 5


@dataclass(frozen=True)
class GeneratedTest:
    source_file: str
    language: str
    function_name: str
    test_id: str
    generation_mode: str
    content: str
    test_names: list[str] = field(default_factory=list)
    repair_test_names: list[str] = field(default_factory=list)


def generate_tests(targets: list[GenerationTarget], config: AgentConfig) -> list[GeneratedTest]:
    if not targets:
        return []

    client = LLMClient(config)
    generated_tests: list[GeneratedTest] = []

    for target in targets:
        prompt = build_prompt(target)
        content: str | None = None

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            LOGGER.info(
                "test_generation_function source=%s function=%s mode=%s attempt=%s",
                target.source_file,
                target.function_change.function_name,
                target.generation_mode,
                attempt + 1,
            )
            try:
                response = client.generate(prompt)
            except Exception as error:
                LOGGER.warning("llm_generation_failed error=%s", str(error))
                time.sleep(LLM_CALL_DELAY_SECONDS)
                continue

            LOGGER.info("LLM_RAW_OUTPUT_START")
            LOGGER.info(response.content)
            LOGGER.info("LLM_RAW_OUTPUT_END")

            cleaned = _strip_code_fences(response.content)
            if cleaned.strip():
                content = cleaned.strip() + "\n"
                break

            LOGGER.warning("empty_llm_output_retry")
            time.sleep(LLM_CALL_DELAY_SECONDS)

        if not content:
            LOGGER.error(
                "generation_failed_after_retries source=%s function=%s",
                target.source_file,
                target.function_change.function_name,
            )
            continue

        generated_tests.append(
            GeneratedTest(
                source_file=target.source_file,
                language=target.language,
                function_name=target.function_change.function_name,
                test_id=target.test_id,
                generation_mode=target.generation_mode,
                content=content,
                test_names=extract_test_names(target.language, content),
                repair_test_names=target.repair_test_names,
            )
        )
        time.sleep(LLM_CALL_DELAY_SECONDS)

    return generated_tests


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
    for language in ("python\n", "javascript\n", "typescript\n", "tsx\n", "jsx\n"):
        stripped = stripped.replace(language, "")
    return stripped.replace("```", "").strip()

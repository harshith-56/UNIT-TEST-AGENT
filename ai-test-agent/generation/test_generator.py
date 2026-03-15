from __future__ import annotations

from dataclasses import dataclass

from agent.config import AgentConfig
from context.repo_context import GenerationContext, GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import build_prompt
from utils.logger import get_logger


LOGGER = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 3


@dataclass(frozen=True)
class GeneratedTest:
    source_file: str
    language: str
    function_names: list[str]
    content: str


def generate_tests(generation_context: GenerationContext, config: AgentConfig) -> list[GeneratedTest]:
    if not generation_context.targets:
        return []

    client = LLMClient(config)
    generated_tests: list[GeneratedTest] = []

    for target in generation_context.targets:
        prompt = build_prompt(target)

        content = None

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            LOGGER.info(
                "test_generation_attempt source=%s attempt=%s",
                target.source_file,
                attempt + 1,
            )

            response = client.generate(prompt)

            LOGGER.info("LLM_RAW_OUTPUT_START")
            LOGGER.info(response.content)
            LOGGER.info("LLM_RAW_OUTPUT_END")

            cleaned = _strip_code_fences(response.content)

            if cleaned.strip():
                content = cleaned
                break

            LOGGER.warning("empty_llm_output_retry")

        if not content:
            LOGGER.error(
                "generation_failed_after_retries source=%s",
                target.source_file,
            )
            continue

        generated_tests.append(_build_generated_test(target, content))

    return generated_tests


def _build_generated_test(target: GenerationTarget, content: str) -> GeneratedTest:
    return GeneratedTest(
        source_file=target.source_file,
        language=target.language,
        function_names=[changed_function.function_name for changed_function in target.changed_functions],
        content=content.strip() + "\n",
    )


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()

    # Remove fenced markdown blocks
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]

    # Remove language identifiers sometimes returned by LLM
    stripped = stripped.replace("python\n", "")
    stripped = stripped.replace("javascript\n", "")
    stripped = stripped.replace("typescript\n", "")

    # Remove stray fences
    stripped = stripped.replace("```", "")

    return stripped.strip()
from __future__ import annotations

from dataclasses import dataclass
import time

from agent.config import AgentConfig
from context.repo_context import GenerationContext, GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import build_prompt
from utils.logger import get_logger


LOGGER = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 5

# delay between LLM calls to avoid rate limits
LLM_CALL_DELAY_SECONDS = 5


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

        for changed_function in target.changed_functions:

            # create a temporary target containing only one function
            single_function_target = GenerationTarget(
                source_file=target.source_file,
                language=target.language,
                framework=target.framework,
                imports=target.imports,
                helper_functions=target.helper_functions,
                existing_tests=target.existing_tests,
                changed_functions=[changed_function],
            )

            prompt = build_prompt(single_function_target)

            content = None

            for attempt in range(MAX_GENERATION_ATTEMPTS):

                LOGGER.info(
                    "test_generation_function source=%s function=%s attempt=%s",
                    target.source_file,
                    changed_function.function_name,
                    attempt + 1,
                )

                try:
                    response = client.generate(prompt)
                except Exception as e:
                    LOGGER.warning("llm_generation_failed error=%s", str(e))
                    time.sleep(LLM_CALL_DELAY_SECONDS)
                    continue

                LOGGER.info("LLM_RAW_OUTPUT_START")
                LOGGER.info(response.content)
                LOGGER.info("LLM_RAW_OUTPUT_END")

                cleaned = _strip_code_fences(response.content)

                if cleaned.strip():
                    content = cleaned
                    break

                LOGGER.warning("empty_llm_output_retry")

                time.sleep(LLM_CALL_DELAY_SECONDS)

            if not content:
                LOGGER.error(
                    "generation_failed_after_retries source=%s function=%s",
                    target.source_file,
                    changed_function.function_name,
                )
                continue

            generated_tests.append(
                GeneratedTest(
                    source_file=target.source_file,
                    language=target.language,
                    function_names=[changed_function.function_name],
                    content=content.strip() + "\n",
                )
            )

            # delay between calls to avoid API rate limits
            time.sleep(LLM_CALL_DELAY_SECONDS)

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

    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]

    stripped = stripped.replace("python\n", "")
    stripped = stripped.replace("javascript\n", "")
    stripped = stripped.replace("typescript\n", "")

    stripped = stripped.replace("```", "")

    return stripped.strip()
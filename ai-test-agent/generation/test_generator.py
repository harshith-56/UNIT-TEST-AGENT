from __future__ import annotations

from dataclasses import dataclass

from agent.config import AgentConfig
from context.repo_context import GenerationContext, GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import build_prompt


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
        response = client.generate(prompt)
        content = _strip_code_fences(response.content)
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
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped

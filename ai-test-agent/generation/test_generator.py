from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field

import esprima
from tree_sitter_languages import get_parser

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt
from utils.logger import get_logger
from validation.test_naming import extract_test_names, has_duplicate_test_names


LOGGER = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 10
RETRY_DELAY_SECONDS = 5
MIN_OUTPUT_CHARACTERS = 24
PLACEHOLDER_PATTERNS = (
    re.compile(r"(?<![\w*])\*{3,}(?![\w*])"),
    re.compile(r"(?<![\w?])\?{2,}(?![\w?])"),
    re.compile(r"\b(?:todo|tbd|fixme|placeholder)\b", re.IGNORECASE),
    re.compile(r"(?:(?:^|[=\(:,\[])\s*|\b(?:assert|return|raise)\s+)\.\.\.(?=\s*(?:$|[,)\]}]))", re.MULTILINE),
)
RETRY_PROMPT_SUFFIX = (
    "\n\n"
    "Previous output was invalid or incomplete.\n"
    "Do NOT use placeholders.\n"
    "Ensure all function arguments are fully specified.\n"
    "Do NOT use fake imports like 'your_module'.\n"
    "If a value is required, use a reasonable valid dummy value instead of omitting it.\n"
)


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
        try:
            llm_input = build_llm_input(target)
            prompt = build_prompt(llm_input)
        except SkipGeneration as error:
            LOGGER.warning(
                "skip_generation source=%s function=%s reason=%s",
                target.source_file,
                target.function_change.function_name,
                str(error),
            )
            continue

        content: str | None = None
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1
            attempt_prompt = _prompt_for_attempt(prompt, attempt)
            LOGGER.info(
                "test_generation_function source=%s function=%s mode=%s attempt=%s",
                target.source_file,
                target.function_change.function_name,
                target.generation_mode,
                attempt_number,
            )
            try:
                response = client.generate(attempt_prompt)
            except Exception as error:
                LOGGER.warning(
                    "llm_generation_failed source=%s function=%s mode=%s attempt=%s error=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number,
                    str(error),
                )
                if attempt < MAX_GENERATION_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            LOGGER.info("LLM_RAW_OUTPUT_START")
            LOGGER.info(response.content)
            LOGGER.info("LLM_RAW_OUTPUT_END")

            cleaned = _strip_code_fences(response.content)
            invalid_reason = _invalid_output_reason(target, cleaned)
            if invalid_reason is None:
                content = cleaned.strip() + "\n"
                break

            LOGGER.warning(
                "invalid_llm_output_detected source=%s function=%s mode=%s attempt=%s reason=%s",
                target.source_file,
                target.function_change.function_name,
                target.generation_mode,
                attempt_number,
                invalid_reason,
            )
            if attempt < MAX_GENERATION_ATTEMPTS - 1:
                LOGGER.info(
                    "retrying_test_generation source=%s function=%s mode=%s next_attempt=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number + 1,
                )
                time.sleep(RETRY_DELAY_SECONDS)

        if not content:
            LOGGER.error(
                "generation_failed_after_retries source=%s function=%s mode=%s attempts=%s",
                target.source_file,
                target.function_change.function_name,
                target.generation_mode,
                MAX_GENERATION_ATTEMPTS,
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


def _prompt_for_attempt(prompt: str, attempt: int) -> str:
    if attempt == 0:
        return prompt
    return prompt + RETRY_PROMPT_SUFFIX


def _invalid_output_reason(target: GenerationTarget, content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return "empty_output"
    if len(stripped) < MIN_OUTPUT_CHARACTERS:
        return "near_empty_output"
    if "your_module" in stripped:
        return "fake_import"

    placeholder_reason = _placeholder_reason(stripped)
    if placeholder_reason:
        return placeholder_reason

    if not _is_syntax_valid(target.language, stripped, target.source_file):
        return "syntax_invalid"
    if has_duplicate_test_names(target.language, stripped):
        return "duplicate_test_names"

    test_names = extract_test_names(target.language, stripped)
    if not test_names:
        return "no_tests_detected"
    if target.generation_mode == "repair":
        if target.repair_test_names and set(test_names) != set(target.repair_test_names):
            return "repair_test_name_mismatch"
        if not 1 <= len(test_names) <= 8:
            return "invalid_repair_test_count"
    elif not 3 <= len(test_names) <= 8:
        return "invalid_test_count"

    expected_prefix = f"test_{target.test_id}_"
    if any(not name.startswith(expected_prefix) for name in test_names):
        return "unexpected_test_name_prefix"

    return None


def _placeholder_reason(content: str) -> str | None:
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(content):
            return "placeholder_content"

    for line in content.splitlines():
        normalized_line = line.strip()
        if not normalized_line:
            continue
        if re.fullmatch(r"[\*\?\.]{3,}", normalized_line):
            return "placeholder_line"
        if re.search(r"\bfrom\s+\S+\s+import\s*$", normalized_line):
            return "incomplete_import"
        if re.search(r"\bimport\s*$", normalized_line):
            return "incomplete_import"

    return None


def _is_syntax_valid(language: str, content: str, source_file: str) -> bool:
    try:
        if language == "python":
            ast.parse(content)
            return True
        if language == "javascript":
            esprima.parseModule(content, {"jsx": source_file.endswith(".jsx")})
            return True
        parser = get_parser("tsx" if source_file.endswith(".tsx") else "typescript")
        tree = parser.parse(content.encode("utf-8"))
        return not _contains_parse_error(tree.root_node)
    except Exception:
        return False


def _contains_parse_error(node) -> bool:
    if getattr(node, "type", None) == "ERROR" or getattr(node, "has_error", False):
        return True
    children = getattr(node, "children", None)
    if children is None:
        children = getattr(node, "named_children", ())
    return any(_contains_parse_error(child) for child in children)

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import esprima
from tree_sitter_languages import get_parser

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt, build_retry_prompt
from utils.logger import get_logger
from validation.test_naming import extract_test_names


LOGGER = get_logger(__name__)


MAX_GENERATION_ATTEMPTS = 12
RETRY_DELAY_SECONDS = 5
RATE_LIMIT_SLEEP_SECONDS = 10
POST_SUCCESS_DELAY_SECONDS = 2
MIN_OUTPUT_CHARACTERS = 24


PLACEHOLDER_PATTERNS = (
    re.compile(r"(?<![\w*])\*{3,}(?![\w*])"),
    re.compile(r"(?<![\w?])\?{3,}(?![\w?])"),
    re.compile(r"(?<!\.)\.\.\.(?!\.)"),
    re.compile(r"\b(?:todo|tbd)\b", re.IGNORECASE),
)

INCOMPLETE_CONSTRUCT_PATTERNS = (
    re.compile(r"=\s*(?=[,\)\]\}])"),
    re.compile(r":\s*(?=[,\}])"),
    re.compile(r"\(\s*,"),
    re.compile(r",\s*,"),
    re.compile(r"\[\s*,"),
    re.compile(r"\{\s*,"),
)

INTEGRATION_TEST_PATTERNS = (
    re.compile(r"\bcreate_engine\s*\("),
    re.compile(r"\b(?:requests|httpx)\.(?:get|post|put|delete|patch)\s*\("),
    re.compile(r"\b(?:sqlite3|psycopg|psycopg2)\.connect\s*\("),
)

RATE_LIMIT_PATTERNS = ("429", "rate limit", "too many requests")


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


@dataclass(frozen=True)
class GenerationFailure:
    source_file: str
    function_name: str
    test_id: str
    generation_mode: str
    reason: str
    attempts: int


@dataclass(frozen=True)
class GenerationResult:
    generated_tests: list[GeneratedTest]
    failures: list[GenerationFailure] = field(default_factory=list)


def generate_tests(targets: list[GenerationTarget], config: AgentConfig) -> GenerationResult:
    client = LLMClient(config)
    generated_tests = []
    failures = []

    for target in targets:
        try:
            llm_input = build_llm_input(target)
            base_prompt = build_prompt(llm_input)
        except SkipGeneration:
            continue

        content = None
        last_failure_reason = "unknown"

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1

            prompt = _prompt_for_attempt(llm_input, base_prompt, attempt_number, last_failure_reason)

            try:
                response = client.generate(prompt)

                LOGGER.info(f"[RAW][{target.test_id}][Attempt {attempt_number}]:\n{response.content[:800]}")

            except Exception as e:
                err = str(e).lower()

                if any(x in err for x in RATE_LIMIT_PATTERNS):
                    LOGGER.warning("Rate limit hit. Sleeping...")
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    continue

                LOGGER.warning(f"[ERROR][{target.test_id}] {e}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            cleaned = _strip_code_fences(response.content)

            LOGGER.info(f"[CLEANED][{target.test_id}]:\n{cleaned[:800]}")

            reason = _invalid_output_reason(target, cleaned)

            if reason is None:
                content = cleaned
                LOGGER.info(f"[SUCCESS][{target.test_id}]")
                time.sleep(POST_SUCCESS_DELAY_SECONDS)
                break

            LOGGER.warning(f"[INVALID][{target.test_id}] {reason} (attempt {attempt_number})")

            last_failure_reason = reason

            time.sleep(RETRY_DELAY_SECONDS)

        if not content:
            failures.append(
                GenerationFailure(
                    source_file=target.source_file,
                    function_name=target.function_change.function_name,
                    test_id=target.test_id,
                    generation_mode=target.generation_mode,
                    reason=last_failure_reason,
                    attempts=MAX_GENERATION_ATTEMPTS,
                )
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
            )
        )

        time.sleep(3)

    return GenerationResult(generated_tests=generated_tests, failures=failures)


def _prompt_for_attempt(llm_input, base_prompt, attempt_number, failure_reason):
    if attempt_number == 1:
        return base_prompt

    retry_prompt = build_retry_prompt(llm_input, failure_reason, attempt_number)

    return (
        retry_prompt
        + f"\n\nFIX PREVIOUS ERROR: {failure_reason}\n"
        + "Do not repeat same output.\n"
    )


def _strip_code_fences(content: str) -> str:
    return content.replace("```", "").strip()


def _invalid_output_reason(target, content):
    if not content.strip():
        return "empty"

    if "your_module" in content:
        return "fake_import"

    if not _is_valid_syntax(target.language, content, target.source_file):
        return "syntax_error"

    names = extract_test_names(target.language, content)

    if not names:
        return "no_tests"

    return None


def _is_valid_syntax(lang, code, file):
    try:
        if lang == "python":
            ast.parse(code)
        elif lang == "javascript":
            esprima.parseModule(code)
        else:
            parser = get_parser("tsx" if file.endswith(".tsx") else "typescript")
            tree = parser.parse(code.encode())
            return not tree.root_node.has_error
        return True
    except:
        return False
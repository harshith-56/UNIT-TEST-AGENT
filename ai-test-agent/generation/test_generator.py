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
from validation.test_naming import extract_test_names, has_duplicate_test_names


LOGGER = get_logger(__name__)


MAX_GENERATION_ATTEMPTS = 7
RETRY_DELAY_SECONDS = 5
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
    re.compile(r"\bopen\s*\([^\n,]+,\s*['\"](?:w|a|x)"),
    re.compile(r"\b(?:Path|pathlib\.Path)\([^\n]*\)\.(?:write_text|write_bytes|open)\s*\("),
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
    if not targets:
        return GenerationResult(generated_tests=[], failures=[])

    client = LLMClient(config)
    generated_tests: list[GeneratedTest] = []
    failures: list[GenerationFailure] = []
    generated_tests_dir = _relative_generated_tests_dir(config)

    for target in targets:
        try:
            llm_input = build_llm_input(target, generated_tests_dir)
            base_prompt = build_prompt(llm_input)
        except SkipGeneration:
            failures.append(
                GenerationFailure(
                    source_file=target.source_file,
                    function_name=target.function_change.function_name,
                    test_id=target.test_id,
                    generation_mode=target.generation_mode,
                    reason="skip_generation",
                    attempts=0,
                )
            )
            continue

        content = None
        previous_signature = None
        previous_failure_reason = None
        repeated_failure_count = 0
        last_failure_reason = "unknown_failure"
        attempts_used = 0

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1
            attempts_used = attempt_number

            attempt_prompt = _prompt_for_attempt(
                llm_input, base_prompt, attempt_number, last_failure_reason
            )

            try:
                response = client.generate(attempt_prompt)
            except Exception as error:
                error_text = str(error).lower()
                last_failure_reason = "llm_request_failed"

                if any(pattern in error_text for pattern in RATE_LIMIT_PATTERNS):
                    last_failure_reason = "rate_limit_risk"
                    break

                if attempt < MAX_GENERATION_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            cleaned = _strip_code_fences(response.content)
            signature = _normalized_output_signature(cleaned)

           
            if previous_signature and signature == previous_signature:
                last_failure_reason = "duplicate_output"
                break
            previous_signature = signature

            invalid_reason = _invalid_output_reason(target, cleaned)

            
            if invalid_reason is None:
                content = cleaned.strip() + "\n"
                break

            last_failure_reason = invalid_reason

            repeated_failure_count = (
                repeated_failure_count + 1
                if invalid_reason == previous_failure_reason
                else 1
            )
            previous_failure_reason = invalid_reason

            
            if invalid_reason in ("placeholder_detected", "integration_test_detected"):
                break

         
            if repeated_failure_count >= 2 or attempt_number >= 3:
                break

            if attempt < MAX_GENERATION_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS)

        if not content:
            failures.append(
                GenerationFailure(
                    source_file=target.source_file,
                    function_name=target.function_change.function_name,
                    test_id=target.test_id,
                    generation_mode=target.generation_mode,
                    reason=last_failure_reason,
                    attempts=attempts_used,
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
                repair_test_names=target.repair_test_names,
            )
        )

    return GenerationResult(generated_tests=generated_tests, failures=failures)


def _prompt_for_attempt(llm_input, base_prompt: str, attempt_number: int, failure_reason: str) -> str:
    if attempt_number == 1:
        return base_prompt

    retry_prompt = build_retry_prompt(llm_input, failure_reason, attempt_number)

  
    return (
        retry_prompt
        + "\n\nSTRICT CORRECTION:\n"
        "- Previous outputs were invalid\n"
        "- DO NOT repeat same mistakes\n"
        "- Provide complete valid tests only\n"
    )


def _relative_generated_tests_dir(config: AgentConfig) -> Path:
    try:
        return config.generated_tests_dir.relative_to(config.repo_root)
    except ValueError:
        return config.generated_tests_dir


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
    for language in ("python\n", "javascript\n", "typescript\n", "tsx\n", "jsx\n"):
        stripped = stripped.replace(language, "")
    return stripped.replace("```", "").strip()


def _normalized_output_signature(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def _invalid_output_reason(target: GenerationTarget, content: str) -> str | None:
    stripped = content.strip()

    if not stripped:
        return "empty_output"

    if len(stripped) < MIN_OUTPUT_CHARACTERS:
        return "near_empty_output"

    if "your_module" in stripped:
        return "fake_import"

    if _placeholder_reason(stripped):
        return "placeholder_detected"

    if _incomplete_construct_reason(stripped):
        return "incomplete_construct"

    if _integration_test_reason(stripped):
        return "integration_test_detected"

    if not _is_syntax_valid(target.language, stripped, target.source_file):
        return "syntax_error"

    test_names = extract_test_names(target.language, stripped)

    if not test_names:
        return "no_tests_detected"

    if target.generation_mode == "repair":
        if target.repair_test_names and set(test_names) != set(target.repair_test_names):
            return "repair_test_name_mismatch"
    elif not 3 <= len(test_names) <= 8:
        return "invalid_test_count"

    expected_prefix = f"test_{target.test_id}_"
    if any(not name.startswith(expected_prefix) for name in test_names):
        return "unexpected_test_name_prefix"

    return None


def _placeholder_reason(content: str) -> str | None:
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(content):
            return "placeholder_detected"
    return None


def _incomplete_construct_reason(content: str) -> str | None:
    for pattern in INCOMPLETE_CONSTRUCT_PATTERNS:
        if pattern.search(content):
            return "incomplete_construct"
    return None


def _integration_test_reason(content: str) -> str | None:
    for pattern in INTEGRATION_TEST_PATTERNS:
        if pattern.search(content):
            return "integration_test_detected"
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
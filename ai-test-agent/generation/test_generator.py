from __future__ import annotations

import re
import time

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from generation.test_models import GeneratedTest, GenerationFailure, GenerationResult
from integration.test_writer import get_test_file_path
from llm.llm_client import LLMClient
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt, build_retry_prompt
from utils.logger import get_logger
from validation.syntax_validator import validate_content
from validation.test_naming import extract_test_names


LOGGER = get_logger(__name__)


MAX_GENERATION_ATTEMPTS = 12
SKIP_AFTER_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
RATE_LIMIT_SLEEP_SECONDS = 1
POST_SUCCESS_DELAY_SECONDS = 1
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



def generate_tests(targets: list[GenerationTarget], config: AgentConfig) -> GenerationResult:
    client = LLMClient(config)
    generated_tests = []
    failures = []

    all_targets_by_file: dict[str, list[str]] = {}
    for target in targets:
        all_targets_by_file.setdefault(target.source_file, []).append(target.function_change.function_name)

    attempted_by_file: dict[str, list[str]] = {}
    succeeded_by_file: dict[str, list[str]] = {}

    for target in targets:
        try:
            specific_test_dir = get_test_file_path(
                config.generated_tests_dir,
                target.source_file,
                target.test_id,
                target.language,
            ).parent
            llm_input = build_llm_input(target, generated_tests_dir=specific_test_dir)
            # LOGGER.info(f"[DEBUG] import_hints for {target.test_id}: {llm_input.import_hints}")
            base_prompt = build_prompt(llm_input)
        except SkipGeneration:
            continue

        attempted_by_file.setdefault(target.source_file, []).append(target.function_change.function_name)
        content = None
        last_failure_reason = "unknown"

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1

            prompt = _prompt_for_attempt(llm_input, base_prompt, attempt_number, last_failure_reason)

            try:
                retry_temperature = None if attempt_number == 1 else min(0.2 + (attempt_number * 0.05), 0.8)
                response = client.generate(prompt, temperature=retry_temperature)

                # LOGGER.info(f"[RAW][{target.test_id}][Attempt {attempt_number}]:\n{response.content}")

            except Exception as e:
                err = str(e).lower()

                if any(x in err for x in RATE_LIMIT_PATTERNS):
                    # LOGGER.warning("Rate limit hit. Sleeping...")
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    continue

                # LOGGER.warning(f"[ERROR][{target.test_id}] {e}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            cleaned = _strip_code_fences(response.content)

            # LOGGER.debug(f"[CLEANED_FULL][{target.test_id}]:\n{cleaned}")
            # LOGGER.info(f"[CLEANED][{target.test_id}]:\n{cleaned}")

            valid, reason = validate_content(
                target.language,
                cleaned,
                target.source_file,
            )
            if not valid:
                if reason == "truncated":
                    last_lines = "\n".join(cleaned.splitlines()[-3:])
                    # LOGGER.warning(f"[TRUNCATED][{target.test_id}] Last 3 lines:\n{last_lines}")
                # LOGGER.warning(f"[INVALID][{target.test_id}] {reason} (attempt {attempt_number})")
                last_failure_reason = reason
                if attempt_number >= SKIP_AFTER_ATTEMPTS:
                    # LOGGER.warning(
                    #     f"[SKIP][{target.test_id}] Skipping after "
                    #     f"{SKIP_AFTER_ATTEMPTS} failed attempts — "
                    #     f"last failure: {reason}"
                    # )
                    last_failure_reason = f"skipped_after_{SKIP_AFTER_ATTEMPTS}_attempts"
                    break
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            content = cleaned
            # LOGGER.info(f"[SUCCESS][{target.test_id}]")
            time.sleep(POST_SUCCESS_DELAY_SECONDS)
            break

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
        succeeded_by_file.setdefault(target.source_file, []).append(target.function_change.function_name)

        time.sleep(3)

    for source_file, all_funcs in all_targets_by_file.items():
        attempted = attempted_by_file.get(source_file, [])
        succeeded = succeeded_by_file.get(source_file, [])
        failed_names = [f for f in attempted if f not in succeeded]
        # LOGGER.info(
        #     f"[COVERAGE][{source_file}] "
        #     f"{len(succeeded)}/{len(attempted)} functions covered. "
        #     f"Skipped/failed: {failed_names or 'none'}"
        # )
        untested_names = [f for f in all_funcs if f not in attempted]
        if untested_names:
            # LOGGER.warning(
            #     f"[COVERAGE][{source_file}] "
            #     f"No tests generated for: {untested_names}"
            # )
            pass 

    return GenerationResult(generated_tests=generated_tests, failures=failures)


def _prompt_for_attempt(llm_input, base_prompt, attempt_number, failure_reason):
    if attempt_number == 1:
        return base_prompt
    retry_prompt = build_retry_prompt(llm_input, failure_reason, attempt_number)
    return retry_prompt + f"\n\nPrevious attempt {attempt_number - 1} failed with: {failure_reason}. Fix exactly this issue and nothing else.\n"


def _strip_code_fences(content: str) -> str:
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue  # skip opening ```python, ```js, closing ``` etc
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _is_truncated(content: str, language: str) -> bool:
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return True
    last = lines[-1].rstrip()

    # Detect mid-word line breaks (streaming cut-off artifact)
    # A line ending with only word characters and no punctuation after an
    # identifier is a strong signal of truncation
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        # Line ends with a bare word character (letter/digit/underscore)
        # AND the line is not a complete statement (no =, :, ), ], } at end)
        if re.match(r".*\w$", stripped) and not re.search(
            r"""[=:,\)\]\}'"]\s*$""", stripped
        ):
            # Exclude lines that are complete Python keywords or identifiers
            # on their own (like "pass", "return None", "continue")
            if not re.match(
                r"^\s*(pass|return|continue|break|raise|import\s+\w+|from\s+\w+)\s*$",
                stripped,
            ):
                return True

    # A line that is a single letter alone is always a truncation artifact
    for line in lines[:-1]:  # check all lines except last (last might be valid)
        if re.match(r"^\s*[a-zA-Z]\s*$", line.rstrip()):
            return True

    # Incomplete trailing operator (critical — catches "assert result ==")
    if re.search(r"(==|!=|<=|>=|=|,|\(|and|or|not)\s*$", last):
        return True

    # Python-specific incomplete block starters
    if language == "python":
        if last.endswith(":"):
            return True
        if re.match(r"^\s*(def|class|async\s+def|async\s+for|async\s+with)\s*$", last):
            return True

    # JS/TS incomplete declarations
    if language in ("javascript", "typescript"):
        if re.match(r"^\s*(function|=>|async\s+function)\s*$", last):
            return True

    # Universal: unbalanced open parens/brackets across full content
    opens = content.count("(") + content.count("[") + content.count("{")
    closes = content.count(")") + content.count("]") + content.count("}")
    if opens - closes > 1:
        return True

    return False

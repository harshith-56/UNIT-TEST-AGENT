from __future__ import annotations

import re
import time

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from generation.test_models import GeneratedTest, GenerationFailure, GenerationResult
from llm.llm_client import LLMClient
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt, build_retry_prompt
from utils.logger import get_logger
from validation.syntax_validator import validate_content
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
                retry_temperature = None if attempt_number == 1 else min(0.2 + (attempt_number * 0.05), 0.8)
                response = client.generate(prompt, temperature=retry_temperature)

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

            fixed = _fix_import_paths(cleaned, target)
            if fixed != cleaned:
                LOGGER.info(f"[IMPORT_FIXED][{target.test_id}] Fixed hallucinated import paths")
            cleaned = fixed

            LOGGER.debug(f"[CLEANED_FULL][{target.test_id}]:\n{cleaned}")
            LOGGER.info(f"[CLEANED][{target.test_id}]:\n{cleaned[:800]}")

            valid, reason = validate_content(
                target.language,
                cleaned,
                target.source_file,
            )
            if not valid:
                if reason == "truncated":
                    last_lines = "\n".join(cleaned.splitlines()[-3:])
                    LOGGER.warning(f"[TRUNCATED][{target.test_id}] Last 3 lines:\n{last_lines}")
                LOGGER.warning(f"[INVALID][{target.test_id}] {reason} (attempt {attempt_number})")
                last_failure_reason = reason
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            content = cleaned
            LOGGER.info(f"[SUCCESS][{target.test_id}]")
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

        time.sleep(3)

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


def _fix_import_paths(content: str, target) -> str:
    """
    Post-process generated code to fix hallucinated import paths.

    The LLM often constructs import paths using the GitHub repo name
    as a top-level package (e.g. ECOMMERCE_UNIT_TEST_AGENT_TESTING.backend.main)
    instead of the real importable path (e.g. backend.main).

    This function detects and removes such prefixes.
    """
    import posixpath
    from pathlib import PurePosixPath

    source_file = target.source_file.replace("\\", "/")

    if target.language == "python":
        parts = PurePosixPath(source_file).with_suffix("").parts
        if not parts:
            return content

        real_top = parts[0]

        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
            match = re.match(
                rf"^(from|import)\s+([A-Za-z][A-Za-z0-9_]{{4,}})\.({re.escape(real_top)}(?:\.\S+)?)(.*)",
                line,
            )
            if match:
                keyword = match.group(1)
                prefix = match.group(2)
                real_path = match.group(3)
                rest = match.group(4)
                if "_" in prefix or prefix.isupper() or prefix[0].isupper():
                    line = f"{keyword} {real_path}{rest}"
            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    if target.language in ("javascript", "typescript"):
        source_no_ext = PurePosixPath(source_file).with_suffix("")
        generated_dir = "tests/ai_generated"
        real_relative = posixpath.relpath(source_no_ext.as_posix(), generated_dir)
        if not real_relative.startswith("."):
            real_relative = "./" + real_relative

        real_name = PurePosixPath(source_file).stem

        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
            match = re.search(
                r"""(['"])([^'"]*[A-Z][A-Z0-9_]{9,}[^'"]*)(['"])""",
                line,
            )
            if match:
                full_path = match.group(2)
                after_caps = re.sub(r"[A-Z][A-Z0-9_]{9,}/", "", full_path)
                if real_name.lower() in after_caps.lower():
                    line = line[: match.start(2)] + real_relative + line[match.end(2) :]
                else:
                    after_caps = after_caps.lstrip("/")
                    rel = posixpath.relpath(after_caps, generated_dir)
                    if not rel.startswith("."):
                        rel = "./" + rel
                    line = line[: match.start(2)] + rel + line[match.end(2) :]
            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    return content


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

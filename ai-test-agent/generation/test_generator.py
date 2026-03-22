from __future__ import annotations

import ast
import json
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import esprima
from tree_sitter_languages import get_parser

from agent.config import AgentConfig
from context.repo_context import GenerationTarget
from llm.llm_client import LLMClient
from llm.prompt_builder import SkipGeneration, build_llm_input, build_prompt, build_retry_prompt
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
    re.compile(r"(?:(?:^|[=\(:,\[]\s*|(?:return|assert)\s+)\.\.\.(?=\s*(?:$|[,)\]}])))", re.MULTILINE),
)
INCOMPLETE_CONSTRUCT_PATTERNS = (
    re.compile(r"=\s*(?=[,\)\]\}])"),
    re.compile(r":\s*(?=[,\}])"),
    re.compile(r"\(\s*,"),
    re.compile(r",\s*,"),
    re.compile(r"\[\s*,"),
    re.compile(r"\{\s*,"),
)
JS_IMPORT_PATTERN = re.compile(
    r"^\s*(?:import(?:.+?\sfrom\s+)?['\"]([^'\"]+)['\"]|(?:const|let|var)\s+.+?=\s*require\(['\"]([^'\"]+)['\"]\))",
    re.MULTILINE,
)
COMMON_PYTHON_TEST_MODULES = {
    "pytest",
    "unittest",
    "mock",
    "pytest_mock",
    "freezegun",
}
COMMON_JS_TEST_MODULES = {
    "jest",
    "vitest",
    "@jest/globals",
}


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
        except SkipGeneration as error:
            LOGGER.warning(
                "skip_generation source=%s function=%s reason=%s",
                target.source_file,
                target.function_change.function_name,
                str(error),
            )
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

        content: str | None = None
        previous_signature: str | None = None
        last_failure_reason = "unknown_failure"
        attempts_used = 0

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1
            attempts_used = attempt_number
            attempt_prompt = _prompt_for_attempt(llm_input, base_prompt, attempt_number, last_failure_reason)
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
                last_failure_reason = "llm_request_failed"
                LOGGER.warning(
                    "llm_generation_failed source=%s function=%s mode=%s attempt=%s reason=%s error=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number,
                    last_failure_reason,
                    str(error),
                )
                if attempt < MAX_GENERATION_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            LOGGER.info("LLM_RAW_OUTPUT_START")
            LOGGER.info(response.content)
            LOGGER.info("LLM_RAW_OUTPUT_END")

            cleaned = _strip_code_fences(response.content)
            signature = _normalized_output_signature(cleaned)
            if previous_signature is not None and signature and signature == previous_signature:
                last_failure_reason = "duplicate_output"
                LOGGER.warning(
                    "duplicate_llm_output_detected source=%s function=%s mode=%s attempt=%s reason=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number,
                    last_failure_reason,
                )
                LOGGER.info(
                    "retry_terminated_early source=%s function=%s mode=%s attempt=%s reason=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number,
                    last_failure_reason,
                )
                break
            previous_signature = signature

            invalid_reason = _invalid_output_reason(target, config, cleaned)
            if invalid_reason is None:
                content = cleaned.strip() + "\n"
                break

            last_failure_reason = invalid_reason
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
                    "retrying_test_generation source=%s function=%s mode=%s next_attempt=%s previous_reason=%s",
                    target.source_file,
                    target.function_change.function_name,
                    target.generation_mode,
                    attempt_number + 1,
                    invalid_reason,
                )
                time.sleep(RETRY_DELAY_SECONDS)

        if not content:
            LOGGER.error(
                "generation_failed_after_retries source=%s function=%s mode=%s attempts=%s reason=%s",
                target.source_file,
                target.function_change.function_name,
                target.generation_mode,
                attempts_used,
                last_failure_reason,
            )
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


def _prompt_for_attempt(llm_input, base_prompt: str, attempt_number: int, failure_reason: str) -> str:
    if attempt_number == 1:
        return base_prompt
    return build_retry_prompt(llm_input, failure_reason, attempt_number)


def _normalized_output_signature(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def _invalid_output_reason(target: GenerationTarget, config: AgentConfig, content: str) -> str | None:
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

    incomplete_reason = _incomplete_construct_reason(stripped)
    if incomplete_reason:
        return incomplete_reason

    if not _is_syntax_valid(target.language, stripped, target.source_file):
        return "syntax_error"

    import_reason = _invalid_import_reason(target, config, stripped)
    if import_reason:
        return import_reason

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
            return "placeholder_detected"

    for line in content.splitlines():
        normalized_line = line.strip()
        if not normalized_line:
            continue
        if re.fullmatch(r"[\*\?\.]{3,}", normalized_line):
            return "placeholder_detected"
        if re.search(r"\bfrom\s+\S+\s+import\s*$", normalized_line):
            return "incomplete_import"
        if re.search(r"\bimport\s*$", normalized_line):
            return "incomplete_import"

    return None


def _incomplete_construct_reason(content: str) -> str | None:
    for pattern in INCOMPLETE_CONSTRUCT_PATTERNS:
        if pattern.search(content):
            return "incomplete_construct"
    return None


def _invalid_import_reason(target: GenerationTarget, config: AgentConfig, content: str) -> str | None:
    if target.language == "python":
        return _invalid_python_import_reason(config, content)
    if target.language in {"javascript", "typescript"}:
        return _invalid_js_import_reason(config, content)
    return None


def _invalid_python_import_reason(config: AgentConfig, content: str) -> str | None:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    repo_modules = _known_python_repo_modules(str(config.repo_root))
    generated_dir = config.generated_tests_dir
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if not _is_allowed_python_import(module_name, repo_modules):
                    return "unknown_import"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level:
                if not _resolve_python_relative_import(generated_dir, config.repo_root, module_name, node.level):
                    return "unknown_import"
                continue
            if module_name and not _is_allowed_python_import(module_name, repo_modules):
                return "unknown_import"
    return None


def _is_allowed_python_import(module_name: str, repo_modules: frozenset[str]) -> bool:
    top_level = module_name.split(".")[0]
    if module_name in COMMON_PYTHON_TEST_MODULES or top_level in COMMON_PYTHON_TEST_MODULES:
        return True
    stdlib_modules = getattr(sys, "stdlib_module_names", set())
    if top_level in stdlib_modules:
        return True
    if module_name in repo_modules or top_level in repo_modules:
        return True
    return any(
        module_name.startswith(f"{known}.") or known.startswith(f"{module_name}.")
        for known in repo_modules
    )


def _resolve_python_relative_import(base_dir: Path, repo_root: Path, module_name: str, level: int) -> bool:
    current = base_dir
    for _ in range(max(level - 1, 0)):
        current = current.parent
    target = current / module_name.replace(".", "/") if module_name else current
    candidates = [target.with_suffix(".py"), target / "__init__.py"]
    return any(candidate.exists() and repo_root in candidate.resolve().parents for candidate in candidates)


@lru_cache(maxsize=8)
def _known_python_repo_modules(repo_root_str: str) -> frozenset[str]:
    repo_root = Path(repo_root_str)
    modules: set[str] = set()
    for path in repo_root.rglob("*.py"):
        try:
            relative_parts = list(path.relative_to(repo_root).with_suffix("").parts)
        except ValueError:
            continue
        if relative_parts and relative_parts[-1] == "__init__":
            relative_parts = relative_parts[:-1]
        if not relative_parts:
            continue
        for index in range(len(relative_parts)):
            modules.add(".".join(relative_parts[index:]))
    return frozenset(modules)


def _invalid_js_import_reason(config: AgentConfig, content: str) -> str | None:
    known_packages = _known_js_packages(str(config.repo_root))
    generated_dir = config.generated_tests_dir
    for match in JS_IMPORT_PATTERN.finditer(content):
        module_spec = match.group(1) or match.group(2)
        if not module_spec:
            continue
        if module_spec.startswith("."):
            if not _resolve_js_relative_import(generated_dir, config.repo_root, module_spec):
                return "unknown_import"
            continue
        if module_spec.startswith("node:"):
            continue
        package_name = _js_package_name(module_spec)
        if package_name not in known_packages:
            return "unknown_import"
    return None


@lru_cache(maxsize=8)
def _known_js_packages(repo_root_str: str) -> frozenset[str]:
    repo_root = Path(repo_root_str)
    known_packages = set(COMMON_JS_TEST_MODULES)
    package_json = repo_root / "package.json"
    if not package_json.exists():
        return frozenset(known_packages)
    try:
        body = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return frozenset(known_packages)
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        known_packages.update((body.get(section) or {}).keys())
    return frozenset(known_packages)


def _js_package_name(module_spec: str) -> str:
    if module_spec.startswith("@"):
        parts = module_spec.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else module_spec
    return module_spec.split("/")[0]


def _resolve_js_relative_import(base_dir: Path, repo_root: Path, module_spec: str) -> bool:
    base_path = (base_dir / PurePosixPath(module_spec)).resolve()
    candidates = [
        base_path,
        base_path.with_suffix(".js"),
        base_path.with_suffix(".jsx"),
        base_path.with_suffix(".ts"),
        base_path.with_suffix(".tsx"),
        base_path / "index.js",
        base_path / "index.ts",
        base_path / "index.tsx",
    ]
    for candidate in candidates:
        if candidate.exists() and repo_root in candidate.parents:
            return True
    return False


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

from __future__ import annotations

import ast
import re
from pathlib import Path

import esprima
from tree_sitter_languages import get_parser

from generation.test_models import GeneratedTest
from utils.logger import get_logger
from validation.test_naming import extract_test_names, has_duplicate_test_names


LOGGER = get_logger(__name__)


BANNED_OUTPUT_PATTERNS = (
    re.compile(r"(?<![\w*])\*{3,}(?![\w*])"),
    re.compile(r"(?<![\w?])\?{3,}(?![\w?])"),
    re.compile(r"(?<!\.)\.\.\.(?!\.)"),
    re.compile(r"\b(?:todo|tbd)\b", re.IGNORECASE),
    re.compile(r"\byour_module\b"),
    re.compile(r"\bcreate_engine\s*\("),
    re.compile(r"\b(?:requests|httpx)\.(?:get|post|put|delete|patch)\s*\("),
    re.compile(r"\b(?:sqlite3|psycopg|psycopg2)\.connect\s*\("),
    re.compile(r"\bopen\s*\([^\n,]+,\s*['\"](?:w|a|x)"),
    re.compile(r"\b(?:Path|pathlib\.Path)\([^\n]*\)\.(?:write_text|write_bytes|open)\s*\("),
)


def validate_content(language: str, content: str, source_file: str) -> tuple[bool, str]:
    """
    Master validation entry point called by test_generator.
    Returns (True, '') on fully valid content, (False, reason) on first failure.
    """
    if not content.strip():
        return False, "empty"

    # Check 1: completeness / truncation
    ok, reason = _check_completeness(content, language)
    if not ok:
        LOGGER.warning(f"[validate_content] completeness fail: {reason}")
        return False, reason

    # Check 2: imports
    ok, reason = _check_imports(content, source_file, language)
    if not ok:
        LOGGER.warning(f"[validate_content] import fail: {reason}")
        return False, reason

    # Check 3: placeholder patterns
    for pattern in BANNED_OUTPUT_PATTERNS:
        if pattern.search(content):
            LOGGER.warning(f"[validate_content] banned pattern matched")
            return False, "banned_pattern"

    # Check 4: universally banned JS/TS patterns
    _UNIVERSAL_JS_BANS = [
        # Deprecated Jest API — removed in v30
        re.compile(r"\.toThrowError\s*\("),
        # Getter spy on non-getter — always wrong for plain consts
        re.compile(
            r"jest\.spyOn\s*\([^)]+,\s*['\"][^'\"]+['\"],\s*['\"](?:get|set)['\"]"
        ),
    ]

    if language in ("javascript", "typescript"):
        for pattern in _UNIVERSAL_JS_BANS:
            if pattern.search(content):
                return False, "banned_js_pattern"

    # Check 5: syntax
    if not is_syntax_valid(language, content, source_file):
        return False, "syntax_error"

    # Check 6: at least one test function exists
    names = extract_test_names(language, content)
    if not names:
        return False, "no_tests"

    return True, ""


def validate_generated_tests(generated_tests: list[GeneratedTest]) -> tuple[list[GeneratedTest], list[GeneratedTest]]:
    valid: list[GeneratedTest] = []
    invalid: list[GeneratedTest] = []
    for generated_test in generated_tests:
        if _is_valid_generated_test(generated_test):
            valid.append(generated_test)
        else:
            invalid.append(generated_test)
    return valid, invalid


def _is_valid_generated_test(generated_test: GeneratedTest) -> bool:
    ok, reason = validate_content(
        generated_test.language,
        generated_test.content,
        generated_test.source_file,
    )
    if not ok:
        LOGGER.warning(f"[INVALID][{generated_test.test_id}] {reason}")
        return False

    if not generated_test.content.strip():
        return False
    if _contains_banned_output(generated_test.content):
        return False
    if has_duplicate_test_names(generated_test.language, generated_test.content):
        return False

    test_names = extract_test_names(generated_test.language, generated_test.content)
    if generated_test.generation_mode == "repair":
        if generated_test.repair_test_names:
            if set(test_names) != set(generated_test.repair_test_names):
                return False
        elif not 1 <= len(test_names) <= 20:
            return False
    elif generated_test.language in ("javascript", "typescript"):
        # Jest uses describe/it blocks — lower minimum, allow describe groups
        if not 1 <= len(test_names) <= 20:
            return False
    elif not 3 <= len(test_names) <= 8:
        return False

    if generated_test.language == "python":
        if any(
            not name.startswith(f"test_{generated_test.test_id}_")
            for name in test_names
        ):
            return False
    # JS/TS uses describe/it style — no prefix requirement
    return is_syntax_valid(generated_test.language, generated_test.content, generated_test.source_file)


def _contains_banned_output(content: str) -> bool:
    return any(pattern.search(content) for pattern in BANNED_OUTPUT_PATTERNS)


def _is_truncated(content: str, language: str) -> bool:
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return True
    last = lines[-1].rstrip()

    # Fix 2: Detect genuine mid-word cuts and single-letter orphan lines
    COMPLETE_ENDINGS = re.compile(
        r"""(\)|]|}|'|"|None|True|False|pass|continue|break|\d+)\s*$"""
    )
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        # Single orphan letter = truncation artifact
        if re.match(r"^\s*[a-zA-Z]\s*$", stripped):
            return True
        # Line ends with bare word chars AND is not a recognized complete ending
        if re.match(r".*[a-zA-Z_]\w*$", stripped):
            if not COMPLETE_ENDINGS.search(stripped):
                # Skip decorator lines — @decorator ends with identifier
                # but is never a truncation indicator
                if stripped.lstrip().startswith("@"):
                    continue
                # Make sure it's not just a keyword/builtin on its own line
                bare_keyword = re.match(
                    r"^\s*(pass|return|continue|break|raise|import\s+[\w,\s]+|from\s+[\w.]+\s+import\s+[\w,\s*]+)\s*$",
                    stripped,
                )
                if not bare_keyword:
                    return True

    # Fix 3: Trailing operator check — avoid false positive on "is not None"
    if re.search(r"(==|!=|<=|>=|(?<!\w)=(?!\w)|,|\()\s*$", last):
        return True
    if re.search(r"\b(and|or)\s*$", last):
        return True
    # "not" only counts if it is the very last word with nothing after
    if re.match(r".*\bnot\s*$", last):
        return True

    if language == "python":
        if last.endswith(":"):
            return True
        if re.match(r"^\s*(def|class|async\s+def|async\s+for|async\s+with)\s*$", last):
            return True

    if language in ("javascript", "typescript"):
        if re.match(r"^\s*(function|=>|async\s+function)\s*$", last):
            return True

    # Fix 1: Only flag severe bracket imbalance (> 3) to avoid false positives
    # from unmatched brackets inside string literals
    try:
        opens = content.count("(") + content.count("[") + content.count("{")
        closes = content.count(")") + content.count("]") + content.count("}")
        if opens - closes > 3:
            return True
    except Exception:
        pass

    return False


def _check_imports(content: str, source_file: str, language: str) -> tuple[bool, str]:
    """Returns (True, '') if imports look valid, (False, reason) if not."""
    if "your_module" in content:
        return False, "fake_import_your_module"

    suspicious_count = 0
    source_parts = set(Path(source_file).parts)

    for line in content.splitlines():
        stripped = line.strip()

        # Python imports
        if language == "python" and (stripped.startswith("import ") or stripped.startswith("from ")):
            # Reject obvious placeholder module names
            if re.search(r"\b(your_|example_|placeholder_|fake_|dummy_module)\w*", stripped):
                return False, "placeholder_import"

            # Extract module name
            match = re.match(r"(?:from|import)\s+([\w.]+)", stripped)
            if match:
                module = match.group(1)
                parts = module.split(".")
                # If module has 2+ segments and shares NO segment with source_file path, it's suspicious
                if len(parts) >= 2:
                    if not any(p in source_parts for p in parts):
                        suspicious_count += 1

        # JS/TS imports
        if language in ("javascript", "typescript"):
            match = re.search(r"""(?:from|require\()\s*['"]([^'"]+)['"]""", stripped)
            if match:
                module_path = match.group(1)
                if re.search(r"(your_module|example_module|placeholder)", module_path):
                    return False, "fake_import_js"

    if suspicious_count > 2:
        return False, f"too_many_suspicious_imports({suspicious_count})"

    return True, ""


def _check_completeness(content: str, language: str) -> tuple[bool, str]:
    """Returns (True, '') if content looks complete, (False, reason) if not."""
    if _is_truncated(content, language):
        return False, "truncated"

    # Stub body detection
    stub_patterns = [
        r"pass\s*#\s*TODO",
        r"raise\s+NotImplementedError",
        r"#\s*implement",
        r"#\s*fill\s+in",
    ]
    for pattern in stub_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, "stub_body"

    return True, ""


def is_syntax_valid(language: str, content: str, source_file: str = "generated") -> bool:
    try:
        if language == "python":
            ast.parse(content)
            return True
        if language == "javascript":
            esprima.parseModule(content, {"jsx": source_file.endswith(".jsx")})
            return True
        parser = get_parser("tsx" if source_file.endswith(".tsx") else "typescript")
        tree = parser.parse(content.encode("utf-8"))
        return not _contains_error(tree.root_node)
    except Exception:
        return False


def _contains_error(node) -> bool:
    if node.type == "ERROR" or node.has_error:
        return True
    return any(_contains_error(child) for child in node.children)

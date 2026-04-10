"""
Benchmark runner for TestGenEval.
Generates tests for a source file without needing a GitHub PR or git diff.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure agent modules are importable
_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

from diff.diff_analyzer import extract_functions_from_source
from generation.test_generator import generate_tests
from validation.syntax_validator import validate_generated_tests
from context.repo_context import GenerationTarget, FunctionChange
from context.project_context import parse_project_context


def _make_config(llm_api_url: str, llm_api_key: str, llm_model: str):
    """Build a minimal AgentConfig for benchmark use."""
    from agent.config import AgentConfig
    import tempfile
    return AgentConfig(
        repo_root=Path(tempfile.mkdtemp()),
        generated_tests_dir=Path(tempfile.mkdtemp()),
        llm_api_url=llm_api_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        project_context_raw="",
        llm_timeout_seconds=150,
        comment_on_pr=False,
        fail_on_generation_failure=False,
    )

def generate_tests_for_file(
    source_code: str,
    source_file_path: str,
    language: str,
    llm_api_url: str,
    llm_api_key: str,
    llm_model: str,
) -> str:
    """
    Main entry point for benchmark.
    Takes source file content, returns generated test file as string.
    """
    # Step 1: Extract all meaningful functions
    functions = extract_functions_from_source(
        source_code=source_code,
        file_path=source_file_path,
        language=language,
    )

    if not functions:
        print(f"  No functions extracted from {source_file_path}")
        return ""

    print(f"  Extracted {len(functions)} functions: {[f.function_name for f in functions]}")

    # Step 2: Build config
    config = _make_config(llm_api_url, llm_api_key, llm_model)

    # Step 3: Build correct module path from file path for import hints
    # django/db/migrations/serializer.py -> django.db.migrations.serializer
    if language == "python":
        module_path = (
            source_file_path
            .replace("\\", "/")
            .removesuffix(".py")
            .replace("/", ".")
        )
    else:
        module_path = source_file_path

    # Step 4: Build GenerationTargets
    targets = []
    for func in functions:
        # Build correct import hint so LLM never guesses the module path
        if language == "python":
            class_name = getattr(func, "enclosing_class_name", None) or ""
            if class_name:
                import_hint = f"from {module_path} import {class_name}"
            else:
                fn_name = func.function_name.split("_")[-1] if "_" in func.function_name else func.function_name
                import_hint = f"from {module_path} import {fn_name}"
        else:
            path_no_ext = (
                source_file_path
                .replace("\\", "/")
                .removesuffix(".tsx")
                .removesuffix(".ts")
                .removesuffix(".jsx")
                .removesuffix(".js")
            )
            symbol = getattr(func, "enclosing_class_name", None) or func.function_name
            import_hint = f"import {{ {symbol} }} from '{path_no_ext}'"

        target = GenerationTarget(
            test_id=func.function_name,
            source_file=source_file_path,
            language=language,
            function_change=func,
            dependencies=[],
            project_rules=[
                f"MANDATORY IMPORT: from {module_path} import {class_name or func.function_name}",
                "NEVER invent module paths. NEVER use task IDs as module names.",
                "Use ONLY unittest.TestCase — never import pytest.",
                "Use self.assertRaises() not pytest.raises().",
                "All test classes inherit from unittest.TestCase.",
                "Only assert behavior you can see explicitly in the source code.",
                "Never assert exceptions unless you see an explicit raise in the source.",
            ],
            pr_rules=[],
            existing_tests_text=f"# CORRECT IMPORT: from {module_path} import {class_name or func.function_name}",
            existing_test_names=[],
            generation_mode="replace",
            repair_test_names=[],
            repair_notes=[],
        )
        targets.append(target)

    # Step 5: Generate
    result = generate_tests(targets, config)

    # Step 6: Validate
    valid_tests, invalid = validate_generated_tests(result.generated_tests)

    if invalid:
        for inv, reason in invalid:
            print(f"  INVALID: {inv.test_id} reason={reason}")

    if not valid_tests:
        print(f"  All tests invalid for {source_file_path}")
        return ""

    print(f"  Valid tests: {len(valid_tests)}")

    # Step 7: Merge into one file
    return _merge_test_files(valid_tests, language, source_code=source_code)

'''
def _merge_test_files(valid_tests, language: str) -> str:
    """Merge multiple generated test objects into one test file."""
    imports = set()
    bodies = []

    for test in valid_tests:
        content = test.content.strip()
        if not content:
            continue

        lines = content.splitlines()
        body_lines = []
        in_header = True
        in_multiline_import = False

        for line in lines:
            stripped = line.strip()

            # Track multi-line imports (lines with unclosed parenthesis)
            if in_multiline_import:
                imports.add(line)  # add continuation lines as-is
                if ")" in line:
                    in_multiline_import = False
                continue

            if in_header and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped == ""
            ):
                if stripped.startswith("import ") or stripped.startswith("from "):
                    # Check if this is a multi-line import
                    if stripped.endswith("(") and ")" not in stripped:
                        in_multiline_import = True
                        # Skip incomplete multi-line imports entirely
                        # They will appear as complete single-line imports elsewhere
                        continue
                    imports.add(stripped)
            else:
                in_header = False
                body_lines.append(line)

        if body_lines:
            bodies.append("\n".join(body_lines).strip())

    # Deduplicate and sort imports, skip any remaining incomplete ones
    clean_imports = sorted(
        imp for imp in imports
        if imp.startswith(("import ", "from ")) and not imp.rstrip().endswith("(")
    )

    if not clean_imports and not bodies:
        return ""

    result = "\n".join(clean_imports)
    result += "\n\n"
    result += "\n\n\n".join(bodies)
    return result.strip() + "\n"
'''

'''
def _merge_test_files(valid_tests, language: str, source_code: str = "") -> str:
    """Merge multiple generated test objects into one test file."""
    import ast as _ast
    
    # Get all names defined in source code
    defined_names = set()
    if source_code and language == "python":
        try:
            tree = _ast.parse(source_code)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                    defined_names.add(node.name)
                elif isinstance(node, _ast.Assign):
                    for target in node.targets:
                        if isinstance(target, _ast.Name):
                            defined_names.add(target.id)
        except SyntaxError:
            pass
    
    imports = set()
    bodies = []

    for test in valid_tests:
        content = test.content.strip()
        if not content:
            continue

        lines = content.splitlines()
        body_lines = []
        in_header = True
        skip_multiline = False

        for line in lines:
            stripped = line.strip()

            if skip_multiline:
                if ")" in line:
                    skip_multiline = False
                continue

            if in_header and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped == ""
            ):
                if stripped.startswith("from ") or stripped.startswith("import "):
                    # Skip multi-line imports entirely
                    if stripped.endswith("(") and ")" not in stripped:
                        skip_multiline = True
                        continue
                    
                    # Validate from X import Y — check Y exists in source
                    if stripped.startswith("from ") and " import " in stripped and defined_names:
                        import_part = stripped.split(" import ", 1)[1]
                        symbols = [s.strip().split(" as ")[0].strip() for s in import_part.split(",")]
                        valid_symbols = []
                        for sym in symbols:
                            # Keep if: in defined_names, or is a known safe symbol
                            if (sym in defined_names 
                                or sym in ("MagicMock", "patch", "AsyncMock", "Mock",
                                          "pytest", "ANY", "call", "sentinel")):
                                valid_symbols.append(sym)
                        
                        if valid_symbols:
                            module = stripped.split(" import ")[0].replace("from ", "")
                            imports.add(f"from {module} import {', '.join(valid_symbols)}")
                    else:
                        imports.add(stripped)
            else:
                in_header = False
                body_lines.append(line)

        if body_lines:
            bodies.append("\n".join(body_lines).strip())

    clean_imports = sorted(
        imp for imp in imports
        if not imp.rstrip().endswith("(")
    )

    if not clean_imports and not bodies:
        return ""

    result = "\n".join(clean_imports)
    result += "\n\n"
    result += "\n\n\n".join(bodies)
    return result.strip() + "\n"
'''

def _merge_test_files(valid_tests, language: str, source_code: str = "") -> str:
    """Merge multiple generated test objects into one test file."""
    import ast as _ast

    # Get ONLY top-level importable names from source
    # (classes and module-level functions — NOT methods inside classes)
    top_level_names = set()
    if source_code and language == "python":
        try:
            tree = _ast.parse(source_code)
            for node in tree.body:  # tree.body = top level only
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                    top_level_names.add(node.name)
                elif isinstance(node, _ast.Assign):
                    for target in node.targets:
                        if isinstance(target, _ast.Name):
                            top_level_names.add(target.id)
        except SyntaxError:
            pass

    imports = set()
    bodies = []

    for test in valid_tests:
        content = test.content.strip()
        if not content:
            continue

        lines = content.splitlines()
        body_lines = []
        in_header = True
        skip_multiline = False

        for line in lines:
            stripped = line.strip()

            if skip_multiline:
                if ")" in line:
                    skip_multiline = False
                continue

            if in_header and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped == ""
            ):
                if stripped.startswith("from ") or stripped.startswith("import "):
                    # Skip multi-line imports
                    if stripped.endswith("(") and ")" not in stripped:
                        skip_multiline = True
                        continue

                    # Validate from X import Y,Z against top-level names
                    if (stripped.startswith("from ")
                            and " import " in stripped
                            and top_level_names):
                        module = stripped.split(" import ")[0].replace("from ", "").strip()
                        import_part = stripped.split(" import ", 1)[1].strip()
                        symbols = [
                            s.strip().split(" as ")[0].strip()
                            for s in import_part.split(",")
                        ]
                        # Keep symbol if it's a top-level name OR
                        # it's from a non-source module (stdlib, pytest, etc.)
                        from_source_module = any(
                            part in module
                            for part in ["django", "flask", "fastapi", "src", "app", "lib"]
                        )
                        if from_source_module:
                            valid_syms = [
                                s for s in symbols
                                if s in top_level_names
                                or s.startswith("_") and s in top_level_names
                            ]
                            if valid_syms:
                                imports.add(f"from {module} import {', '.join(valid_syms)}")
                        else:
                            imports.add(stripped)
                    else:
                        imports.add(stripped)
            else:
                in_header = False
                body_lines.append(line)

        if body_lines:
            bodies.append("\n".join(body_lines).strip())

    clean_imports = sorted(
    imp for imp in imports
    if not imp.rstrip().endswith("(")
    and imp.strip() != "import pytest"
    and not imp.strip().startswith("import pytest")
    )
    # Add unittest import at top
    clean_imports = ["import unittest"] + [i for i in clean_imports if i != "import unittest"]

    # Remove pytest imports — Docker containers may not have pytest
    clean_imports = [
        imp for imp in clean_imports
        if imp.strip() not in ("import pytest", "import pytest as pt")
        and not imp.strip().startswith("from pytest")
    ]

    # Ensure unittest is imported
    if any("unittest" not in imp for imp in clean_imports) or not any("unittest" in imp for imp in clean_imports):
        clean_imports = ["import unittest"] + clean_imports


    if not clean_imports and not bodies:
        return ""

    result = "\n".join(clean_imports)
    result += "\n\n"
    result += "\n\n\n".join(bodies)
    return result.strip() + "\n"
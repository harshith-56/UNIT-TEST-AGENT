from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from agent.config import test_framework_for_language
from context.dependency_resolver import DependencyContext
from context.repo_context import GenerationTarget
from context.token_budget import MAX_CONTEXT_TOKENS, MAX_INPUT_TOKENS, estimate_tokens, trim_rules_to_budget


class SkipGeneration(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMInput:
    function_name: str
    language: str
    test_framework: str
    primary_code_block: str
    dependencies: list[str]
    project_rules: list[str]
    pr_rules: list[str]
    existing_tests: str
    import_hints: list[str]


def build_llm_input(target: GenerationTarget, generated_tests_dir: Path | str = Path("tests/ai_generated")) -> LLMInput:
    dependency_entries = [_dependency_entry(dependency) for dependency in target.dependencies]
    project_rules, pr_rules = _fit_context_rules(target.project_rules, target.pr_rules)
    existing_tests = _fit_existing_tests(_build_existing_tests_reference(target))

    llm_input = LLMInput(
        function_name=target.test_id,
        language=target.language,
        test_framework=test_framework_for_language(target.language),
        primary_code_block=target.function_change.source_code.strip(),
        dependencies=[entry["active"] for entry in dependency_entries],
        project_rules=project_rules,
        pr_rules=pr_rules,
        existing_tests=existing_tests,
        import_hints=_build_import_hints(target, Path(generated_tests_dir)),
    )
    return _fit_llm_input_to_budget(llm_input, dependency_entries)


def _detect_external_calls(source_code: str) -> list[str]:
    hints = []
    seen = set()

    local_names = set(re.findall(
        r"^\s{4,}(\w+)\s*=",
        source_code,
        re.MULTILINE,
    ))

    for obj, method in re.findall(r"\b(\w+)\.(\w+)\s*\(", source_code):
        key = f"{obj}.{method}"
        if key in seen:
            continue
        if obj in local_names:
            continue
        if obj in {
            "self", "cls", "str", "int", "float", "bool",
            "list", "dict", "set", "tuple", "type",
            "re", "os", "sys", "math", "json", "time",
            "datetime", "timedelta", "pathlib", "Path",
            "console", "Math", "JSON", "Object", "Array",
            "Promise", "Error", "String", "Number",
        }:
            continue
        seen.add(key)
        hints.append(
            f"calls {obj}.{method}() — '{obj}' is external to this "
            f"function and must be mocked in tests, not called for real"
        )

    for name in re.findall(r"\b(\w+)\s*\[", source_code):
        key = f"subscript:{name}"
        if key in seen:
            continue
        if name in local_names:
            continue
        if name in {"self", "cls"}:
            continue
        seen.add(key)
        hints.append(
            f"accesses '{name}[...]' — '{name}' is external state "
            f"(module-level or closure). Must be patched in tests, "
            f"not assigned directly as a local variable"
        )

    first_line = (
        source_code.strip().splitlines()[0]
        if source_code.strip() else ""
    )
    for param, default in re.findall(
        r"(\w+)\s*(?::\s*[\w\[\]| ,]+)?\s*=\s*([A-Z]\w*)\s*\(",
        first_line,
    ):
        hints.append(
            f"parameter '{param}' has default value {default}(...) — "
            f"this default only runs inside a framework request context. "
            f"In tests you must create a mock and pass it directly: "
            f"{param} = MagicMock()"
        )

    return hints


def _detect_ui_component(source_code: str, language: str) -> str:
    if language not in ("javascript", "typescript"):
        return ""

    jsx_patterns = [
        r"return\s*\(",
        r"return\s*<",
        r"=>\s*<",
        r"=>\s*\(",
        r"<[A-Z][a-zA-Z]+",
        r"<[a-z]+\s+[a-z]+=",
        r"</[a-zA-Z]",
    ]

    is_component = any(
        re.search(p, source_code)
        for p in jsx_patterns
    )

    if not is_component:
        return ""

    return (
        "====================\n"
        "UI COMPONENT DETECTED\n"
        "====================\n"
        "This function returns UI markup (JSX/TSX). It is a UI component.\n"
        "UI components CANNOT be tested by calling them directly.\n\n"
        "CORRECT approach — use the framework's render utility:\n\n"
        "  import { render, screen, fireEvent, waitFor } "
        "from '@testing-library/react'\n\n"
        "  it('renders correctly', () => {\n"
        "    render(<ComponentName />)\n"
        "    expect(screen.getByText('expected text')).toBeInTheDocument()\n"
        "  })\n\n"
        "  it('handles user interaction', async () => {\n"
        "    render(<ComponentName />)\n"
        "    fireEvent.change(screen.getByPlaceholderText('Username'), {\n"
        "      target: { value: 'testuser' },\n"
        "    })\n"
        "    fireEvent.click(screen.getByRole('button'))\n"
        "    await waitFor(() =>\n"
        "      expect(screen.getByText('success')).toBeInTheDocument()\n"
        "    )\n"
        "  })\n\n"
        "WRONG patterns — never do these:\n"
        "  const result = ComponentName({ props })           "
        "← calling component as plain function\n"
        "  const result = ComponentName({ props }, event)    "
        "← same mistake\n"
        "  result.handleSubmit(event)                        "
        "← component returns JSX, not methods\n\n"
        "For mocking API calls inside the component:\n"
        "  jest.mock('./path/to/api')\n"
        "  import { myApiFunction } from './path/to/api'\n"
        "  jest.mocked(myApiFunction).mockResolvedValue({ success: true })\n\n"
    )


def _build_mock_hints_section(source_code: str) -> str:
    hints = _detect_external_calls(source_code)
    if not hints:
        return ""
    return (
        "====================\n"
        "WHAT NEEDS MOCKING IN THIS FUNCTION\n"
        "====================\n"
        "The following external dependencies were detected.\n"
        "You MUST mock ALL of them in every test.\n"
        "Do not call the real versions.\n"
        + "".join(f"- {h}\n" for h in hints)
        + "\n"
    )


def build_prompt(llm_input: LLMInput) -> str:
    dependencies = "\n\n".join(llm_input.dependencies) or "- None"
    import_hints = "\n".join(f"- {hint}" for hint in llm_input.import_hints) or "- None"
    existing_tests = llm_input.existing_tests.strip() or "None"

    project_context = "\n".join(f"- {r}" for r in llm_input.project_rules) or "- None"
    pr_context = "\n".join(f"- {r}" for r in llm_input.pr_rules) or "- None"

    is_repair = "Repair only these failing tests:" in existing_tests

    if not is_repair:
        instructions = f"Generate 3 to 9 production-grade unit tests for {llm_input.function_name}."
    else:
        instructions = "Fix ONLY failing tests. Keep EXACT same names."

    return (
        "You are an EXPERT production-grade UNIT TEST generator.\n"
        "You write STRICT, HIGH-COVERAGE, FAILURE-DRIVEN tests.\n"
        "You do NOT guess. You do NOT improvise. You follow code strictly.\n\n"

        f"LANGUAGE: {llm_input.language}\n"
        f"TEST FRAMEWORK: {llm_input.test_framework}\n\n"

        "====================\n"
        "ABSOLUTE RULES (NON-NEGOTIABLE)\n"
        "====================\n"
        "- Output ONLY executable test code\n"
        "- NO markdown\n"
        "- NO explanations\n"
        "- NO comments\n"
        "- NO placeholders (***, ..., ???, TODO, TBD)\n"
        "- NEVER use 'your_module'\n"
        "- ALWAYS provide COMPLETE arguments\n"
        "- NEVER invent fields or parameters\n"
        "- ONLY use real imports from code, dependencies, or hints\n"
        "- NEVER cut off output mid-function — finish every function you start\n"
        "- NEVER generate more tests than you can complete — fewer complete tests beats more broken ones\n\n"

        "====================\n"
        "TEST TYPE (STRICT)\n"
        "====================\n"
        "- Generate ONLY UNIT tests\n"
        "- DO NOT use TestClient\n"
        "- DO NOT make HTTP calls\n"
        "- DO NOT create real DB engines\n\n"

        "====================\n"
        "PROJECT CONTEXT\n"
        "====================\n"
        f"{project_context}\n\n"

        "====================\n"
        "PR CONTEXT\n"
        "====================\n"
        f"{pr_context}\n\n"

        "====================\n"
        "FUNCTION UNDER TEST\n"
        "====================\n"
        f"{llm_input.primary_code_block}\n\n"
        + _detect_ui_component(
            llm_input.primary_code_block, llm_input.language
        )
        + _build_mock_hints_section(llm_input.primary_code_block)
        +
        "====================\n"
        "IMPORT HINTS (MANDATORY — USE ONLY THESE)\n"
        "====================\n"
        f"{import_hints}\n\n"
        "IMPORT RULES:\n"
        "- You MUST import ONLY from the module paths listed above\n"
        "- NEVER invent module paths not listed above\n"
        "- NEVER use the repository or project name as a package prefix\n"
        "- NEVER import from modules not explicitly listed in import hints above\n"
        "- If a symbol is not importable from the listed paths, do NOT import it\n"
        "- For Python: use exactly the module path shown (e.g. 'backend.main')\n"
        "- For JS/TS: use exactly the relative path shown (e.g. './Signup')\n"
        "- If a type is shown in DEPENDENCIES with a source file path, "
        "import it from that exact module path\n"
        "- Example: if DEPENDENCIES shows 'SignupRequest (backend/schemas.py)' "
        "then import it as: from backend.schemas import SignupRequest\n\n"

        "====================\n"
        "DEPENDENCIES\n"
        "====================\n"
        f"{dependencies}\n\n"

        "====================\n"
        "EXISTING TESTS (DO NOT DUPLICATE)\n"
        "====================\n"
        f"{existing_tests}\n\n"

        "====================\n"
        "TEST GENERATION RULES\n"
        "====================\n"
        f"{instructions}\n"
        "- Each test must cover a UNIQUE scenario\n"
        "- Cover ALL branches and return paths\n"
        "- Cover valid, edge, boundary, invalid, and error cases\n\n"

        "====================\n"
        "ASSERTION STRENGTH (CRITICAL)\n"
        "====================\n"
        "- DO NOT write weak assertions\n"
        "- DO NOT use 'assert result is not None'\n"
        "- Assertions must validate REAL behavior\n\n"

        "====================\n"
        "INPUT RULES\n"
        "====================\n"
        "- Use ONLY visible schema/classes\n"
        "- NEVER leave arguments incomplete\n"
        "- If unsure, use VALID dummy values\n"
        "- Example:\n"
        "  SignupRequest(username='user', email='a@b.com', password='validpass123')\n\n"

        "====================\n"
        "BEHAVIOR RULES\n"
        "====================\n"
        "- Infer behavior ONLY from given code\n"
        "- DO NOT assume hidden logic\n"
        "- DO NOT invent outputs\n\n"

        "====================\n"
        "ADVERSARIAL TESTING (MANDATORY)\n"
        "====================\n"
        "- Try to BREAK the function\n"
        "- Include invalid inputs\n"
        "- Include wrong types\n"
        "- Include edge cases\n"
        "- Include boundary values\n\n"

        "====================\n"
        "MOCKING RULES (READ CAREFULLY)\n"
        "====================\n"
        "Identify which category the function under test falls into:\n\n"
        "CATEGORY A — Pure function (no I/O, no state, no calls to external systems)\n"
        "  Example (Python): def add(a, b): return a + b\n"
        "  Example (TS/JS): function add(a, b) { return a + b }\n"
        "  Rule: NO mocking. Call directly and assert the return value.\n\n"

        "CATEGORY B — Function with INJECTED dependencies (db, client, service passed as args)\n"
        "  Example (Python): def create_user(payload, db: Session)\n"
        "  Example (TS/JS): function createUser(payload, db) { return db.insert(payload) }\n"
        "  Rule: Pass a mock directly. DO NOT rely on framework injection mechanisms.\n"
        "  Correct (Python):\n"
        "    from unittest.mock import MagicMock\n"
        "    db = MagicMock()\n"
        "    db.query.return_value.filter.return_value.first.return_value = None\n"
        "    result = create_user(payload, db)\n"
        "  Correct (TS/JS):\n"
        "    const db = { insert: jest.fn().mockReturnValue({ id: 1 }) }\n"
        "    const result = createUser(payload, db)\n"
        "  WRONG (Python): def test_x(db = Depends(get_db)) ← Depends() does nothing in pytest\n"
        "  WRONG (TS/JS): jest.mock('db') without passing it explicitly ← hides real dependency flow\n\n"

        "CATEGORY C — Function that reads/writes MODULE-LEVEL state\n"
        "  Example (Python): _sessions[token] = {...} where _sessions is a module-level dict\n"
        "  Example (TS/JS): export const sessions = {}; sessions[token] = {...}\n"
        "  Rule: Use patch()/mocking tools to replace module-level state.\n"
        "  Correct (Python):\n"
        "    from unittest.mock import patch\n"
        "    with patch('mymodule._sessions', {}) as mock_sessions:\n"
        "        result = my_function('token')\n"
        "  Correct (TS/JS):\n"
        "    jest.mock('./sessionStore', () => ({ sessions: {} }))\n"
        "    import { sessions, getSession } from './sessionStore'\n"
        "    sessions['abc'] = { user: 1 }\n"
        "    const result = getSession('abc')\n"
        "  WRONG (Python): _sessions = {} ← local variable, not patched\n"
        "  WRONG (TS/JS): sessions = {} ← does not override imported module binding\n\n"

        "CATEGORY D — Function that CALLS other functions internally\n"
        "  Example (Python): def signup(payload, db): error = validate_signup(payload)\n"
        "  Example (TS/JS): function signup(payload, db) { validateSignup(payload); return db.insert(payload) }\n"
        "  Rule: Mock the called function ONLY if it has side effects or is not under test.\n"
        "  Correct (Python):\n"
        "    with patch('mymodule.validate_signup', return_value=None):\n"
        "        result = signup(payload, db)\n"
        "  Correct (TS/JS):\n"
        "    jest.mock('./module', () => ({ ...jest.requireActual('./module'), validateSignup: jest.fn() }))\n"
        "    const result = signup(payload, db)\n"
        "  WRONG (Python): Not mocking when validate_signup has side effects\n"
        "  WRONG (TS/JS): Importing before mocking ← mock will not apply\n\n"

        "CATEGORY E — Async / side-effect functions\n"
        "  Example (Python): async def fetch_user(client): return await client.get('/user')\n"
        "  Example (TS/JS): async function fetchUser(api) { const res = await api.get('/user'); return res.data }\n"
        "  Rule: Always handle async correctly and mock async dependencies.\n"
        "  Correct (Python):\n"
        "    client = AsyncMock()\n"
        "    client.get.return_value = {'id': 1}\n"
        "    result = await fetch_user(client)\n"
        "  Correct (TS/JS):\n"
        "    const api = { get: jest.fn().mockResolvedValue({ data: { id: 1 } }) }\n"
        "    const result = await fetchUser(api)\n"
        "  WRONG (Python): Missing await\n"
        "  WRONG (TS/JS): Not using await ← test may pass incorrectly\n\n"

        "CATEGORY F — Frontend UI Component (React / TSX / JSX)\n"
        "  Example (TSX): function Button({ label }) { return <button>{label}</button> }\n"
        "  Example (TSX with state): function Counter() { const [count, setCount] = useState(0); return <button onClick={() => setCount(count+1)}>{count}</button> }\n"
        "  Rule: Test rendered output and behavior, NOT internal implementation.\n"
        "  Use a rendering library (e.g., React Testing Library).\n"
        "  Correct (TS/JS):\n"
        "    import { render, screen, fireEvent } from '@testing-library/react'\n"
        "    render(<Button label='Click' />)\n"
        "    expect(screen.getByText('Click')).toBeInTheDocument()\n"
        "  Interaction example:\n"
        "    render(<Counter />)\n"
        "    fireEvent.click(screen.getByRole('button'))\n"
        "    expect(screen.getByText('1')).toBeInTheDocument()\n"
        "  Mock external hooks/APIs if needed (e.g., fetch, context, services)\n"
        "  WRONG:\n"
        "    Testing internal state directly\n"
        "    Calling component as a normal function instead of rendering\n\n"

        "====================\n"
        "GENERATOR FUNCTIONS\n"
        "====================\n"
        "- If the function under test contains 'yield', it is a generator\n"
        "- Generators MUST be tested with next() or list():\n"
        "  CORRECT:   db = next(get_db())\n"
        "  CORRECT:   results = list(get_items())\n"
        "  WRONG:     db = get_db(); db.query(...)  ← this will crash\n"
        "  WRONG:     result = get_db(); assert result.execute()\n"
        "- NEVER call methods directly on the return value of a generator function\n\n"

        "====================\n"
        "COMPLETENESS (NON-NEGOTIABLE)\n"
        "====================\n"
        "- Your output must be 100% complete with no cut-off lines\n"
        "- The last line must be a complete statement — closing paren, bracket, or a pass\n"
        "- Never end output with: =, ==, ,, (, and, or, not\n"
        "- If running out of space: stop BEFORE starting a new test, not in the middle of one\n\n"

        "====================\n"
        "FINAL OUTPUT\n"
        "====================\n"
        "- ONLY executable test code\n"
    )


def build_retry_prompt(llm_input: LLMInput, failure_reason: str, attempt_number: int) -> str:
    base = build_prompt(llm_input)

    correction = (
        "\n\n====================\n"
        "PREVIOUS OUTPUT REJECTED\n"
        "====================\n"
        f"SPECIFIC FAILURE REASON: {failure_reason}\n\n"
        "You MUST fix exactly this issue. Do not change anything else.\n\n"
        "General rules that must also hold:\n"
        "- Remove ALL placeholders (***, ..., ???)\n"
        "- Provide COMPLETE arguments\n"
        "- DO NOT use your_module\n"
        "- Use valid imports from hints only\n"
        "- Generate ONLY UNIT tests\n"
        "- DO NOT use DB, TestClient, or create_engine\n"
        "- Ensure every function is complete before starting the next\n"
        "- If the function has injected dependencies (db, client, "
        "session as args): pass MagicMock() directly, never Depends()\n"
        "- If the function uses module-level state (_var[x] = y): "
        "use patch('module.path._varname', value) not direct assignment\n"
        "- All external calls (db, network, file) MUST be mocked\n"
    )

    if failure_reason == "truncated":
        correction += (
            "\nYOUR OUTPUT WAS CUT OFF. Rules:\n"
            "- Generate the minimum required tests (3-6) but keep each one short\n"
            "- Make absolutely sure the last test is fully closed\n"
            "- The final line must be a complete statement\n"
        )

    if failure_reason in ("fake_import_your_module", "placeholder_import", "fake_import_js"):
        correction += (
            "\nYOUR IMPORTS WERE INVALID. Rules:\n"
            "- Only import from paths listed in IMPORT HINTS above\n"
            "- Never use your_module, example_module, or any placeholder path\n"
        )

    if "generator" in failure_reason or failure_reason == "generator_not_unwrapped":
        correction += (
            "\nGENERATOR USAGE WAS WRONG. Rules:\n"
            "- Use next(function()) to get a single value\n"
            "- Use list(function()) to get all values\n"
            "- Never call methods on the raw generator object\n"
        )

    if failure_reason == "banned_pattern":
        correction += (
            "\nYOUR OUTPUT CONTAINED A BANNED PATTERN. Rules:\n"
            "- No create_engine, real DB connections, or HTTP calls\n"
            "- No file writes (open with 'w', Path.write_text, etc)\n"
            "- No placeholders (***, ???, TODO, TBD)\n"
            "- No 'your_module' anywhere\n"
            "- Replace any banned pattern with a proper mock\n"
        )

    if failure_reason == "stub_body":
        correction += (
            "\nYOUR OUTPUT CONTAINED STUB FUNCTIONS. Rules:\n"
            "- Every test must have real assertions\n"
            "- No 'pass', no 'raise NotImplementedError'\n"
            "- No '# implement' or '# fill in' comments\n"
            "- Write the complete test body for every test function\n"
        )

    if "suspicious_imports" in failure_reason:
        correction += (
            "\nYOUR IMPORTS WERE SUSPICIOUS. Rules:\n"
            "- ONLY import from the paths listed in IMPORT HINTS\n"
            "- Do not import from invented module paths\n"
            "- Do not use the repository name as a package prefix\n"
        )

    if failure_reason == "missing_screen_import":
        correction += (
            "\nYOU USED screen BUT DID NOT IMPORT IT. Fix:\n"
            "- Change your import to include screen:\n"
            "  import { render, screen, fireEvent, waitFor } "
            "from '@testing-library/react'\n"
        )

    if failure_reason == "no_tests":
        correction += (
            "\nNO TEST FUNCTIONS WERE DETECTED. Rules:\n"
            "- For Python: every test must start with def test_\n"
            "- For JS/TS: every test must use it('...') or test('...')\n"
            "- Do not put tests inside classes for Python\n"
            "- Generate at least 3 test functions\n"
        )

    if attempt_number >= 5 and _detect_ui_component(
        llm_input.primary_code_block, llm_input.language
    ):
        correction += (
            "\nUI COMPONENT RETRY — simplify your approach:\n"
            "- Use render(<ComponentName />) — nothing else\n"
            "- Use screen.getByText(), screen.getByPlaceholderText(), "
            "screen.getByRole() to find elements\n"
            "- Use fireEvent.change() and fireEvent.click() for interactions\n"
            "- Use await waitFor(() => ...) for async assertions\n"
            "- Do NOT call the component as a function\n"
            "- Do NOT try to access handleSubmit or any method from the result\n"
            "- Import screen: { render, screen, fireEvent, waitFor }\n"
        )

    if attempt_number >= 7:
        correction += (
            f"\n\nCRITICAL — attempt {attempt_number} of 12:\n"
            "- Generate the MINIMUM number of tests required — "
            "3-6 tests for JS/TS, 3-6 tests for Python\n"
            "- The last line of your output must close all open blocks\n"
        )

    return base + correction


def _build_existing_tests_reference(target: GenerationTarget) -> str:
    if target.generation_mode != "repair":
        return target.existing_tests_text

    lines: list[str] = []
    if target.repair_test_names:
        lines.append("Repair only these failing tests:")
        lines.extend(f"- {name}" for name in target.repair_test_names)
    if target.repair_notes:
        lines.append("Failure notes:")
        lines.extend(f"- {note}" for note in target.repair_notes)
    if target.existing_tests_text.strip():
        lines.append("Existing tests:")
        lines.append(target.existing_tests_text.strip())
    return "\n".join(lines).strip()


def _build_import_hints(target: GenerationTarget, generated_tests_dir: Path) -> list[str]:
    if target.language == "python":
        module_path = _python_module_path(target.source_file)
        func_name = target.function_change.function_name.split(".")[-1]
        class_name = target.function_change.enclosing_class_name
        hints = [f"Source file: {target.source_file}"]
        if module_path:
            hints.append(f"Module path: {module_path}")
            if class_name:
                hints.append(f"Import the class: from {module_path} import {class_name}")
            else:
                hints.append(f"Import the function: from {module_path} import {func_name}")
            hints.append(f"Only import from '{module_path}' — do NOT invent other module paths")
            # Hint that sibling modules may contain dependency types
            package = ".".join(module_path.split(".")[:-1])
            if package:
                hints.append(
                    f"If you need request/response types or models, "
                    f"check sibling modules in the '{package}' package. "
                    f"Look at the DEPENDENCIES section for available types "
                    f"and their source files — import from those exact paths."
                )
        # Add import hints for dependency types
        # These are sibling modules whose types the LLM will need
        for dep in target.dependencies:
            if not dep.source_file:
                continue
            dep_module_path = _python_module_path(dep.source_file)
            if not dep_module_path or dep_module_path == module_path:
                continue
            # Extract the type names the LLM is likely to need
            # by scanning the dependency content for class definitions
            import re as _re
            class_names = _re.findall(
                r"^class\s+(\w+)", dep.content, _re.MULTILINE
            )
            if class_names:
                names_str = ", ".join(class_names[:5])
                hints.append(
                    f"For types from {dep.source_file}: "
                    f"from {dep_module_path} import {names_str}"
                )
        return hints

    source_file_str = target.source_file.replace("\\", "/")
    source_file_str = re.sub(r"^[A-Za-z]:[/\\]", "", source_file_str)
    source_file_str = source_file_str.lstrip("/")
    source_no_suffix = PurePosixPath(source_file_str).with_suffix("")
    generated_dir = PurePosixPath(generated_tests_dir.as_posix())
    relative_import = posixpath.relpath(source_no_suffix.as_posix(), generated_dir.as_posix())
    if not relative_import.startswith("."):
        relative_import = f"./{relative_import}"
    symbol_name = target.function_change.enclosing_class_name or target.function_change.function_name.split(".")[0]
    return [
        f"Source file: {target.source_file}",
        f"Import path (relative from test file): '{relative_import}'",
        f"Example: import {symbol_name} from '{relative_import}'",
        f"ONLY use this exact path — never use absolute or repo-name paths",
    ]


def _python_module_path(source_file: str) -> str:
    path = PurePosixPath(source_file)
    if path.suffix != ".py":
        return ""
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _dependency_entry(dependency: DependencyContext) -> dict[str, str]:
    location = f" ({dependency.source_file})" if dependency.source_file else ""
    full = f"- {dependency.name}{location}\n{dependency.content.strip()}"
    summary = f"- {dependency.name}{location}\n{dependency.summary.strip()}"
    return {"active": full, "summary": summary}


def _fit_context_rules(project_rules: list[str], pr_rules: list[str]) -> tuple[list[str], list[str]]:
    if not project_rules and not pr_rules:
        return [], []

    project_budget = MAX_CONTEXT_TOKENS // 2
    pr_budget = MAX_CONTEXT_TOKENS - project_budget

    trimmed_project = trim_rules_to_budget(project_rules, project_budget)
    trimmed_pr = trim_rules_to_budget(pr_rules, pr_budget)

    return trimmed_project, trimmed_pr


def _fit_existing_tests(existing_tests: str) -> str:
    if not existing_tests.strip():
        return ""

    kept_lines: list[str] = []
    used_tokens = 0
    for line in existing_tests.splitlines():
        line_tokens = estimate_tokens(line)
        if kept_lines and used_tokens + line_tokens > 250:
            break
        kept_lines.append(line)
        used_tokens += line_tokens
    return "\n".join(kept_lines).strip()


def _fit_llm_input_to_budget(llm_input: LLMInput, dependency_entries: list[dict[str, str]]) -> LLMInput:
    fitted_input = llm_input

    while estimate_tokens(build_prompt(fitted_input)) > MAX_INPUT_TOKENS:
        trimmed_input = _trim_context(fitted_input)
        if trimmed_input != fitted_input:
            fitted_input = trimmed_input
            continue

        summarized_input = _summarize_dependencies(fitted_input, dependency_entries)
        if summarized_input != fitted_input:
            fitted_input = summarized_input
            continue

        reduced_existing_tests = _trim_existing_tests(fitted_input)
        if reduced_existing_tests != fitted_input:
            fitted_input = reduced_existing_tests
            continue

        break

    if estimate_tokens(build_prompt(fitted_input)) > MAX_INPUT_TOKENS:
        raise SkipGeneration(f"Prompt exceeds MAX_INPUT_TOKENS for {llm_input.function_name}")

    return fitted_input


def _trim_context(llm_input: LLMInput) -> LLMInput:
    if llm_input.pr_rules:
        return replace(llm_input, pr_rules=llm_input.pr_rules[:-1])
    if llm_input.project_rules:
        return replace(llm_input, project_rules=llm_input.project_rules[:-1])
    return llm_input


def _summarize_dependencies(llm_input: LLMInput, dependency_entries: list[dict[str, str]]) -> LLMInput:
    active_dependencies = list(llm_input.dependencies)
    for index, current_dependency in enumerate(active_dependencies):
        summary_dependency = dependency_entries[index]["summary"]
        if current_dependency != summary_dependency:
            active_dependencies[index] = summary_dependency
            return replace(llm_input, dependencies=active_dependencies)
    return llm_input


def _trim_existing_tests(llm_input: LLMInput) -> LLMInput:
    if not llm_input.existing_tests.strip():
        return llm_input

    lines = llm_input.existing_tests.splitlines()
    if len(lines) <= 1:
        return replace(llm_input, existing_tests="")
    return replace(llm_input, existing_tests="\n".join(lines[:-1]).strip())
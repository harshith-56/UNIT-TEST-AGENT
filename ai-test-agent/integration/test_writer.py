from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.config import AgentConfig
from context.repo_context import MaintenanceAction
from generation.test_models import GeneratedTest
from integration.test_mapping import load_test_mapping, mapping_key, save_test_mapping
from utils.file_utils import ensure_directory


@dataclass(frozen=True)
class WriteResult:
    written_paths: list[Path]
    test_mapping: dict[str, dict]
    maintenance_changes: int = 0


def get_test_file_path(
    generated_tests_dir: Path,
    source_file: str,
    test_id: str,
    language: str,
) -> Path:
    """
    Compute the path for a single-function test file.

    source_file: "inventory_system/services/order_service.py"
    test_id:     "OrderService_create_order" or "create_order"

    Returns:
        <generated_tests_dir>/inventory_system/services/order_service/
            test_OrderService_create_order.py
    """
    sanitized_source = _sanitize_source_path(source_file)
    source_no_ext = Path(sanitized_source).with_suffix("")

    if language == "typescript":
        ext = ".test.ts"
    elif language == "javascript":
        ext = ".test.js"
    else:
        ext = ".py"

    test_filename = f"test_{test_id}{ext}"
    test_dir = generated_tests_dir / source_no_ext
    return test_dir / test_filename


def write_generated_tests(
    repo_root: Path,
    generated_tests: list[GeneratedTest],
    maintenance_actions: list[MaintenanceAction],
    config: AgentConfig,
) -> WriteResult:
    ensure_directory(config.generated_tests_dir)
    mapping = load_test_mapping(config.generated_tests_dir)
    written_paths: list[Path] = []
    maintenance_changes = 0

    for action in maintenance_actions:
        changed = _apply_maintenance_action(config.generated_tests_dir, action, mapping)
        if changed:
            maintenance_changes += 1

    for generated_test in generated_tests:
        destination = get_test_file_path(
            config.generated_tests_dir,
            generated_test.source_file,
            generated_test.test_id,
            generated_test.language,
        )
        _write_generated_test(destination, config.generated_tests_dir, generated_test, mapping)
        written_paths.append(destination.relative_to(repo_root))

    save_test_mapping(config.generated_tests_dir, mapping)
    unique_paths = sorted(dict.fromkeys(written_paths))
    return WriteResult(
        written_paths=unique_paths,
        test_mapping=mapping,
        maintenance_changes=maintenance_changes,
    )


def _apply_maintenance_action(
    generated_tests_dir: Path,
    action: MaintenanceAction,
    mapping: dict[str, dict],
) -> bool:
    if action.action_type == "deletion":
        return _delete_function_test_file(
            generated_tests_dir, action.source_file, action.function_name, mapping
        )
    elif action.action_type == "rename":
        # Delete the old test file; the renamed function gets new tests via
        # the normal generation flow (repo_context.py falls through to target creation).
        old_name = action.previous_name or action.function_name
        return _delete_function_test_file(
            generated_tests_dir, action.source_file, old_name, mapping
        )
    return False


def _delete_function_test_file(
    generated_tests_dir: Path,
    source_file: str,
    function_name: str,
    mapping: dict[str, dict],
) -> bool:
    key = mapping_key(source_file, function_name)
    entry = mapping.get(key)
    changed = False

    if entry and entry.get("target_file"):
        test_file = generated_tests_dir / entry["target_file"]
        if test_file.exists():
            test_file.unlink()
            changed = True
            _remove_empty_dirs(test_file.parent, generated_tests_dir)

    mapping.pop(key, None)
    return changed


def _remove_empty_dirs(directory: Path, stop_at: Path) -> None:
    """Remove directory and its ancestors if empty, stopping at stop_at."""
    current = directory
    while current != stop_at and current != current.parent:
        if not current.is_dir():
            break
        try:
            current.rmdir()  # raises OSError if not empty
            current = current.parent
        except OSError:
            break


def _write_generated_test(
    destination: Path,
    generated_tests_dir: Path,
    generated_test: GeneratedTest,
    mapping: dict[str, dict],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if generated_test.language == "python":
        _ensure_init_files_for_path(destination.parent, generated_tests_dir)

    # Each file contains exactly one function's tests — always overwrite
    destination.write_text(generated_test.content.strip() + "\n", encoding="utf-8")

    target_file = destination.relative_to(generated_tests_dir).as_posix()
    entry = {
        "source_file": generated_test.source_file,
        "language": generated_test.language,
        "function_name": generated_test.function_name,
        "test_id": generated_test.test_id,
        "target_file": target_file,
        "test_names": sorted(dict.fromkeys(generated_test.test_names)),
    }
    mapping[mapping_key(generated_test.source_file, generated_test.function_name)] = entry


def _ensure_init_files_for_path(test_dir: Path, generated_tests_dir: Path) -> None:
    """
    Create __init__.py in every directory from generated_tests_dir down to
    test_dir (inclusive), so pytest can discover nested test files.
    """
    dirs: list[Path] = []
    current = test_dir
    while True:
        dirs.append(current)
        if current == generated_tests_dir:
            break
        parent = current.parent
        if parent == current:  # reached filesystem root
            break
        current = parent

    for d in dirs:
        if d.is_dir():
            init_file = d / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")


def _sanitize_source_path(source_file: str) -> str:
    """
    Strip top-level ALL_CAPS repo folder from source path for directory naming.
    e.g. ECOMMERCE_UNIT_TEST_AGENT_TESTING/backend/main.py -> backend/main.py
    e.g. backend/main.py -> backend/main.py (unchanged)
    """
    parts = Path(source_file.replace("\\", "/")).parts
    if not parts:
        return source_file
    first = parts[0]
    if (
        first == first.upper()
        and "_" in first
        and len(first) >= 8
        and first.replace("_", "").isalpha()
    ):
        remaining = "/".join(parts[1:])
        return remaining if remaining else source_file
    return source_file

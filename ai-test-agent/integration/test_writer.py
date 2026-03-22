from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from agent.config import AgentConfig
from context.repo_context import MaintenanceAction
from generation.test_models import GeneratedTest
from integration.test_mapping import load_test_mapping, mapping_key, save_test_mapping
from utils.file_utils import ensure_directory, sanitize_module_name
from validation.test_naming import extract_test_names


@dataclass(frozen=True)
class WriteResult:
    written_paths: list[Path]
    test_mapping: dict[str, dict]
    maintenance_changes: int = 0


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
        destination = _destination_for_action(config.generated_tests_dir, action)
        changed = _apply_maintenance_action(destination, action, mapping)
        if changed:
            maintenance_changes += 1
            written_paths.append(destination.relative_to(repo_root))

    for generated_test in generated_tests:
        destination = config.generated_tests_dir / _build_file_name(generated_test)
        _write_generated_test(destination, generated_test, mapping)
        written_paths.append(destination.relative_to(repo_root))

    save_test_mapping(config.generated_tests_dir, mapping)
    unique_paths = sorted(dict.fromkeys(written_paths))
    return WriteResult(written_paths=unique_paths, test_mapping=mapping, maintenance_changes=maintenance_changes)


def _destination_for_action(generated_tests_dir: Path, action: MaintenanceAction) -> Path:
    if action.language == "python":
        return generated_tests_dir / f"test_ai_generated_{sanitize_module_name(action.source_file)}.py"
    if action.language == "javascript":
        return generated_tests_dir / f"ai_generated_{sanitize_module_name(action.source_file)}.test.js"
    return generated_tests_dir / f"ai_generated_{sanitize_module_name(action.source_file)}.test.ts"


def _apply_maintenance_action(destination: Path, action: MaintenanceAction, mapping: dict[str, dict]) -> bool:
    if not destination.exists():
        _delete_mapping_entries(mapping, action)
        return False

    content = destination.read_text(encoding="utf-8")
    original = content
    if action.action_type == "deletion":
        content = _remove_function_tests(
            content,
            action.language,
            action.previous_name or action.function_name,
            action.previous_test_id or action.test_id,
            action.existing_test_names,
        )
        _delete_mapping_entries(mapping, action)
    elif action.action_type == "rename":
        old_name = action.previous_name or action.function_name
        old_test_id = action.previous_test_id or action.test_id
        content = _rename_function_tests(content, action.language, old_name, action.function_name, old_test_id, action.test_id)
        _rename_mapping_entry(mapping, action)
    else:
        return False

    if content == original:
        return False
    if extract_test_names(action.language, content):
        destination.write_text(content.strip() + "\n", encoding="utf-8")
    elif destination.exists():
        destination.unlink()
    return True


def _write_generated_test(destination: Path, generated_test: GeneratedTest, mapping: dict[str, dict]) -> None:
    existing_content = destination.read_text(encoding="utf-8") if destination.exists() else ""
    existing_imports, body = _extract_imports_and_body(existing_content)
    new_imports, new_body = _extract_imports_and_body(generated_test.content)
    merged_imports = _merge_imports(existing_imports, new_imports)

    if generated_test.generation_mode == "append":
        existing_block_body = _extract_function_block(body, generated_test.language, generated_test.function_name, generated_test.test_id)
        combined_body = existing_block_body.rstrip()
        if combined_body:
            combined_body += "\n\n" + new_body.strip()
        else:
            combined_body = new_body.strip()
        body = _replace_function_block(body, generated_test.language, generated_test.function_name, generated_test.test_id, combined_body)
        updated_test_names = sorted(dict.fromkeys(_mapped_test_names(mapping, generated_test) + generated_test.test_names))
    elif generated_test.generation_mode == "repair":
        if generated_test.repair_test_names:
            repaired_body = _remove_named_tests(body, generated_test.language, generated_test.repair_test_names)
            remaining = [name for name in _mapped_test_names(mapping, generated_test) if name not in generated_test.repair_test_names]
        else:
            repaired_body = _remove_function_tests(body, generated_test.language, generated_test.function_name, generated_test.test_id, [])
            remaining = []
        existing_block_body = _extract_function_block(repaired_body, generated_test.language, generated_test.function_name, generated_test.test_id)
        combined_body = existing_block_body.rstrip()
        if combined_body:
            combined_body += "\n\n" + new_body.strip()
        else:
            combined_body = new_body.strip()
        body = _replace_function_block(repaired_body, generated_test.language, generated_test.function_name, generated_test.test_id, combined_body)
        updated_test_names = sorted(dict.fromkeys(remaining + generated_test.test_names))
    else:
        body = _replace_function_block(body, generated_test.language, generated_test.function_name, generated_test.test_id, new_body.strip())
        updated_test_names = sorted(dict.fromkeys(generated_test.test_names))

    final_content = _compose_file_content(merged_imports, body)
    ensure_directory(destination.parent)
    destination.write_text(final_content, encoding="utf-8")

    entry = {
        "source_file": generated_test.source_file,
        "language": generated_test.language,
        "function_name": generated_test.function_name,
        "test_id": generated_test.test_id,
        "target_file": destination.name,
        "test_names": updated_test_names,
    }
    mapping[mapping_key(generated_test.source_file, generated_test.function_name)] = entry


def _mapped_test_names(mapping: dict[str, dict], generated_test: GeneratedTest) -> list[str]:
    entry = mapping.get(mapping_key(generated_test.source_file, generated_test.function_name), {})
    return list(entry.get("test_names") or [])


def _delete_mapping_entries(mapping: dict[str, dict], action: MaintenanceAction) -> None:
    for key in {
        mapping_key(action.source_file, action.function_name),
        mapping_key(action.source_file, action.previous_name or action.function_name),
    }:
        mapping.pop(key, None)


def _rename_mapping_entry(mapping: dict[str, dict], action: MaintenanceAction) -> None:
    old_key = mapping_key(action.source_file, action.previous_name or action.function_name)
    entry = mapping.pop(old_key, None)
    if entry is None:
        entry = {
            "source_file": action.source_file,
            "language": action.language,
            "target_file": _destination_for_action(Path("."), action).name,
            "test_names": action.existing_test_names,
        }
    renamed_tests = [name.replace(f"test_{action.previous_test_id}_", f"test_{action.test_id}_") for name in entry.get("test_names") or []]
    entry.update(
        {
            "function_name": action.function_name,
            "test_id": action.test_id,
            "test_names": sorted(dict.fromkeys(renamed_tests)),
        }
    )
    mapping[mapping_key(action.source_file, action.function_name)] = entry


def _replace_function_block(body: str, language: str, function_name: str, test_id: str, block_body: str) -> str:
    block = _wrap_block(function_name, language, block_body)
    marker_pattern = _marker_pattern(function_name, language)
    if marker_pattern.search(body):
        updated = marker_pattern.sub(block, body)
    else:
        stripped = _remove_function_tests(body, language, function_name, test_id, [])
        updated = (stripped.rstrip() + "\n\n" + block).strip() if stripped.strip() else block
    return updated.strip() + "\n"


def _extract_function_block(body: str, language: str, function_name: str, test_id: str) -> str:
    marker_match = _marker_pattern(function_name, language).search(body)
    if marker_match:
        return marker_match.group("body").strip()
    return _extract_tests_by_prefix(body, language, test_id).strip()


def _remove_function_tests(content: str, language: str, function_name: str, test_id: str, existing_test_names: list[str]) -> str:
    updated = _marker_pattern(function_name, language).sub("", content)
    names_to_remove = existing_test_names or extract_test_names(language, _extract_tests_by_prefix(updated, language, test_id))
    if names_to_remove:
        updated = _remove_named_tests(updated, language, names_to_remove)
    else:
        updated = _remove_tests_by_prefix(updated, language, test_id)
    return _clean_spacing(updated)


def _rename_function_tests(content: str, language: str, old_name: str, new_name: str, old_test_id: str, new_test_id: str) -> str:
    updated = content
    old_block = _marker_pattern(old_name, language)
    if old_block.search(updated):
        updated = old_block.sub(lambda match: _wrap_block(new_name, language, _rename_block_body(match.group("body"), old_name, new_name, old_test_id, new_test_id)), updated)
    else:
        extracted = _extract_tests_by_prefix(updated, language, old_test_id)
        if extracted:
            renamed_block = _wrap_block(new_name, language, _rename_block_body(extracted, old_name, new_name, old_test_id, new_test_id))
            updated = _remove_tests_by_prefix(updated, language, old_test_id).rstrip()
            updated = (updated + "\n\n" + renamed_block).strip()
    return _clean_spacing(updated)


def _rename_block_body(body: str, old_name: str, new_name: str, old_test_id: str, new_test_id: str) -> str:
    updated = body.replace(f"test_{old_test_id}_", f"test_{new_test_id}_")
    updated = updated.replace(old_name, new_name)
    old_bare = old_name.split(".")[-1]
    new_bare = new_name.split(".")[-1]
    if old_bare != new_bare:
        updated = re.sub(rf"\b{re.escape(old_bare)}\b", new_bare, updated)
    return updated


def _remove_named_tests(content: str, language: str, test_names: list[str]) -> str:
    updated = content
    for name in test_names:
        updated = re.sub(_test_pattern(language, re.escape(name)), "", updated, flags=re.MULTILINE | re.DOTALL)
    return _clean_spacing(updated)


def _remove_tests_by_prefix(content: str, language: str, test_id: str) -> str:
    prefix_pattern = rf"test_{re.escape(test_id)}_[A-Za-z0-9_]+"
    return re.sub(_test_pattern(language, prefix_pattern), "", content, flags=re.MULTILINE | re.DOTALL)


def _extract_tests_by_prefix(content: str, language: str, test_id: str) -> str:
    pattern = re.compile(_test_pattern(language, rf"test_{re.escape(test_id)}_[A-Za-z0-9_]+"), flags=re.MULTILINE | re.DOTALL)
    return "\n\n".join(match.group(0).strip() for match in pattern.finditer(content))


def _test_pattern(language: str, name_pattern: str) -> str:
    if language == "python":
        return rf"^def\s+{name_pattern}\(.*?(?=^def\s+test_|\Z)"
    return rf"^\s*(?:it|test)\(\s*['\"`]{name_pattern}['\"`].*?(?=^\s*(?:it|test)\(\s*['\"`]test_|\Z)"


def _marker_pattern(function_name: str, language: str) -> re.Pattern[str]:
    comment = _comment_prefix(language)
    start = re.escape(f"{comment} AI_TEST_AGENT_START function={function_name}")
    end = re.escape(f"{comment} AI_TEST_AGENT_END function={function_name}")
    return re.compile(rf"{start}\s*(?P<body>.*?){end}", flags=re.DOTALL)


def _wrap_block(function_name: str, language: str, block_body: str) -> str:
    comment = _comment_prefix(language)
    body = block_body.strip()
    return (
        f"{comment} AI_TEST_AGENT_START function={function_name}\n"
        f"{body}\n"
        f"{comment} AI_TEST_AGENT_END function={function_name}"
    )


def _comment_prefix(language: str) -> str:
    return "#" if language == "python" else "//"


def _extract_imports_and_body(content: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        else:
            body_lines.append(line)
    return imports, "\n".join(body_lines).strip()


def _merge_imports(existing: list[str], new: list[str]) -> list[str]:
    merged = list(dict.fromkeys(existing + new))
    merged.sort()
    return merged


def _compose_file_content(imports: list[str], body: str) -> str:
    if imports and body.strip():
        return "\n".join(imports) + "\n\n" + body.strip() + "\n"
    if imports:
        return "\n".join(imports) + "\n"
    return body.strip() + ("\n" if body.strip() else "")


def _clean_spacing(content: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", content)
    return cleaned.strip() + ("\n" if cleaned.strip() else "")


def _build_file_name(generated_test: GeneratedTest) -> str:
    module_name = sanitize_module_name(generated_test.source_file)
    if generated_test.language == "python":
        return f"test_ai_generated_{module_name}.py"
    if generated_test.language == "javascript":
        return f"ai_generated_{module_name}.test.js"
    return f"ai_generated_{module_name}.test.ts"

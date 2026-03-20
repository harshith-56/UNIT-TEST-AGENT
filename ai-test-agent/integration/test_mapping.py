from __future__ import annotations

import json
from pathlib import Path


MAPPING_FILE_NAME = ".ai_test_mapping.json"


def mapping_file_path(generated_tests_dir: Path) -> Path:
    return generated_tests_dir / MAPPING_FILE_NAME


def mapping_key(source_file: str, function_name: str) -> str:
    return f"{source_file}::{function_name}"


def load_test_mapping(generated_tests_dir: Path) -> dict[str, dict]:
    path = mapping_file_path(generated_tests_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_test_mapping(generated_tests_dir: Path, mapping: dict[str, dict]) -> None:
    path = mapping_file_path(generated_tests_dir)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")

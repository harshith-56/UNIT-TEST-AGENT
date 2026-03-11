from __future__ import annotations

from pathlib import Path


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, content: str, allowed_root: Path, overwrite: bool = False) -> None:
    resolved_root = allowed_root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ValueError(f"Refusing to write outside allowed root: {resolved_path}")
    if resolved_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved_path}")
    ensure_directory(resolved_path.parent)
    resolved_path.write_text(content, encoding="utf-8", newline="\n")


def sanitize_module_name(path: str) -> str:
    module_path = Path(path)
    relative_without_suffix = module_path.with_suffix("")
    sanitized = "_".join(part for part in relative_without_suffix.parts if part not in {".", ".."})
    return sanitized.replace("-", "_")

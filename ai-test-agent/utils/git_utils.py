from __future__ import annotations

import subprocess
from pathlib import Path


def run_git_command(args: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def get_diff(repo_root: Path, base_branch: str) -> str:
    result = run_git_command(["diff", "--unified=0", "--no-color", f"origin/{base_branch}...HEAD"], repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout


def get_changed_files(repo_root: Path, base_branch: str) -> list[str]:
    result = run_git_command(["diff", "--name-only", f"origin/{base_branch}...HEAD"], repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff --name-only failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

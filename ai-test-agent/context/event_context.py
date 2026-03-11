from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventContext:
    repository: str
    pull_request_number: int
    base_branch: str
    head_branch: str
    commit_sha: str


def load_event_context() -> EventContext:
    repository = _required_env("GITHUB_REPOSITORY")
    commit_sha = _required_env("GITHUB_SHA")
    event_payload = _load_event_payload()
    pull_request = event_payload.get("pull_request", {})

    pull_request_number = int(
        os.getenv("PR_NUMBER")
        or pull_request.get("number")
        or event_payload.get("number")
        or 0
    )
    if pull_request_number <= 0:
        raise RuntimeError("Pull request number is required")

    base_branch = (
        os.getenv("GITHUB_BASE_REF")
        or pull_request.get("base", {}).get("ref")
        or event_payload.get("base_ref")
    )
    head_branch = (
        os.getenv("GITHUB_HEAD_REF")
        or pull_request.get("head", {}).get("ref")
        or event_payload.get("head_ref")
    )
    if not base_branch or not head_branch:
        raise RuntimeError("Base branch and head branch are required")

    return EventContext(
        repository=repository,
        pull_request_number=pull_request_number,
        base_branch=base_branch,
        head_branch=head_branch,
        commit_sha=commit_sha,
    )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_event_payload() -> dict:
    event_path = os.getenv("GITHUB_EVENT_PATH", "").strip()
    if not event_path:
        return {}
    path = Path(event_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

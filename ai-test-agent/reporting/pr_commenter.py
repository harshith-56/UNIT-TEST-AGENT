from __future__ import annotations

import os

import requests

from context.event_context import EventContext
from execution.test_runner import TestRunResult


def post_pr_comment(
    event_context: EventContext,
    generated_count: int,
    written_count: int,
    test_results: list[TestRunResult],
) -> None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        return
    passed = sum(1 for result in test_results if result.returncode == 0)
    failed = sum(1 for result in test_results if result.returncode != 0)
    body = (
        "AI Test Agent summary\n\n"
        f"- Generated test files: {generated_count}\n"
        f"- Written test files: {written_count}\n"
        f"- Test commands passed: {passed}\n"
        f"- Test commands failed: {failed}\n"
    )
    url = f"https://api.github.com/repos/{event_context.repository}/issues/{event_context.pull_request_number}/comments"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        json={"body": body},
        timeout=30,
    )
    response.raise_for_status()

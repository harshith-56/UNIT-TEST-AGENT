from __future__ import annotations

import re
from pathlib import Path


TEST_NAME_PATTERN = re.compile(r"""(?:it|test)\(\s*['"`](.*?)['"`]""")


def extract_test_names(file_path: Path) -> set[str]:
    source = file_path.read_text(encoding="utf-8")
    return {match.group(1).strip() for match in TEST_NAME_PATTERN.finditer(source)}

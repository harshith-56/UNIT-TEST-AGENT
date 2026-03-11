from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("AI_TEST_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)

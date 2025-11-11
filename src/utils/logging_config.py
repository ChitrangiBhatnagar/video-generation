"""Configure consistent logging across scripts."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(log_level: str = "INFO", log_format: Optional[str] = None) -> None:
    """Configure root logging with a consistent, color-free formatter."""

    format_string = log_format or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


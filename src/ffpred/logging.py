"""Application logging configuration."""

from __future__ import annotations

import logging


def configure_logging(verbosity: int = 0, level_name: str = "INFO") -> None:
    """Configure application logs once at the CLI composition root."""
    level = logging.DEBUG if verbosity > 0 else getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

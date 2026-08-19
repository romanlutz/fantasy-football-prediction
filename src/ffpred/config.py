"""Typed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.errors import ConfigurationError


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Validated settings shared by CLI and library callers."""

    output_dir: Path = Path()
    history_start: int = 2009
    train_start: int = 2010
    test_year: int = 2014
    cache_dir: Path | None = None
    log_level: str = "INFO"
    scoring: ScoringConfig = field(default=DEFAULT_SCORING)

    def __post_init__(self) -> None:
        if not self.history_start < self.train_start <= self.test_year:
            raise ConfigurationError(
                "Expected history_start < train_start <= test_year"
            )

    @classmethod
    def from_env(cls) -> Settings:
        """Load process-level defaults, leaving CLI flags to override them."""
        cache_value = os.getenv("FFPRED_CACHE_DIR")
        return cls(
            output_dir=Path(os.getenv("FFPRED_OUTPUT_DIR", ".")),
            history_start=int(os.getenv("FFPRED_HISTORY_START", "2009")),
            train_start=int(os.getenv("FFPRED_TRAIN_START", "2010")),
            test_year=int(os.getenv("FFPRED_TEST_YEAR", "2014")),
            cache_dir=Path(cache_value) if cache_value else None,
            log_level=os.getenv("FFPRED_LOG_LEVEL", "INFO").upper(),
        )

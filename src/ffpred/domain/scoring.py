"""Configurable fantasy scoring rules."""

from __future__ import annotations

from dataclasses import dataclass

from ffpred.domain.models import QuarterbackGameStats


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringConfig:
    """Weights for standard quarterback fantasy scoring."""

    passing_yards_per_point: float = 25.0
    passing_touchdown: float = 4.0
    interception: float = -2.0
    rushing_yards_per_point: float = 10.0
    rushing_touchdown: float = 6.0
    fumble: float = -2.0
    two_point_conversion: float = 2.0


DEFAULT_SCORING = ScoringConfig()


def fantasy_score(
    stats: QuarterbackGameStats,
    config: ScoringConfig = DEFAULT_SCORING,
) -> float:
    """Calculate points from normalized game statistics."""
    return (
        stats.passing_yards / config.passing_yards_per_point
        + stats.passing_touchdowns * config.passing_touchdown
        + stats.passing_interceptions * config.interception
        + stats.rushing_yards / config.rushing_yards_per_point
        + stats.rushing_touchdowns * config.rushing_touchdown
        + stats.fumbles * config.fumble
        + (stats.passing_two_point_made + stats.rushing_two_point_made)
        * config.two_point_conversion
    )

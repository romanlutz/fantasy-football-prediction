"""Configurable fantasy scoring rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from ffpred.domain.models import DstGameStats, KickerGameStats, QuarterbackGameStats


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


#: Ascending (points_allowed <= threshold, points) tiers. The final tier's
#: threshold is a catch-all for any higher points-allowed total. This mirrors
#: common default team-defense scoring (e.g. 0 allowed -> 10, 35+ -> -4).
DEFAULT_POINTS_ALLOWED_TIERS: tuple[tuple[float, float], ...] = (
    (0.0, 10.0),
    (6.0, 7.0),
    (13.0, 4.0),
    (20.0, 1.0),
    (27.0, 0.0),
    (34.0, -1.0),
    (float("inf"), -4.0),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DstScoringConfig:
    """Weights for team defense/special-teams fantasy scoring."""

    sack: float = 1.0
    interception: float = 2.0
    fumble_recovery: float = 2.0
    touchdown: float = 6.0
    safety: float = 2.0
    blocked_kick: float = 2.0
    points_allowed_tiers: tuple[tuple[float, float], ...] = field(
        default=DEFAULT_POINTS_ALLOWED_TIERS
    )


DEFAULT_DST_SCORING = DstScoringConfig()


def points_allowed_score(
    points_allowed: float,
    tiers: tuple[tuple[float, float], ...] = DEFAULT_POINTS_ALLOWED_TIERS,
) -> float:
    """Look up the tiered points-allowed score for one points total."""
    for threshold, points in tiers:
        if points_allowed <= threshold:
            return points
    return tiers[-1][1]


def dst_fantasy_score(
    stats: DstGameStats,
    config: DstScoringConfig = DEFAULT_DST_SCORING,
) -> float:
    """Calculate team defense/special-teams points from game statistics."""
    return (
        stats.sacks * config.sack
        + stats.interceptions * config.interception
        + stats.fumble_recoveries * config.fumble_recovery
        + stats.touchdowns * config.touchdown
        + stats.safeties * config.safety
        + stats.blocked_kicks * config.blocked_kick
        + points_allowed_score(stats.points_allowed, config.points_allowed_tiers)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class KickerScoringConfig:
    """Weights for kicker fantasy scoring.

    Field goals are scored by distance band; the two common conventions (flat
    3-points-per-make, or distance-tiered) are both expressible by setting
    all three make weights equal or distinct, respectively.
    """

    field_goal_0_39: float = 3.0
    field_goal_40_49: float = 4.0
    field_goal_50_plus: float = 5.0
    field_goal_missed: float = 0.0
    extra_point_made: float = 1.0
    extra_point_missed: float = 0.0


DEFAULT_KICKER_SCORING = KickerScoringConfig()


def kicker_fantasy_score(
    stats: KickerGameStats,
    config: KickerScoringConfig = DEFAULT_KICKER_SCORING,
) -> float:
    """Calculate kicker points from game statistics."""
    return (
        stats.fg_made_0_39 * config.field_goal_0_39
        + stats.fg_made_40_49 * config.field_goal_40_49
        + stats.fg_made_50_plus * config.field_goal_50_plus
        + stats.fg_missed * config.field_goal_missed
        + stats.pat_made * config.extra_point_made
        + stats.pat_missed * config.extra_point_missed
    )

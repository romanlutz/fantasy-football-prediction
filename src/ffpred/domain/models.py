"""Immutable normalized domain records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ffpred.domain.identifiers import GameId, PlayerId, Season, TeamCode, Week


@dataclass(frozen=True, slots=True, order=True)
class GameKey:
    """Chronological key for a regular-season game."""

    season: Season
    week: Week


@dataclass(frozen=True, slots=True, kw_only=True)
class GameContext:
    """Provider-independent game metadata."""

    game_id: GameId
    game_date: date
    home_team: TeamCode
    away_team: TeamCode
    team: TeamCode
    opponent: TeamCode


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerProfile:
    """Stable player attributes needed for feature generation."""

    player_id: PlayerId
    name: str
    birth_date: date | None
    rookie_season: Season | None


@dataclass(frozen=True, slots=True, kw_only=True)
class QuarterbackGameStats:
    """Quarterback counting statistics for one game."""

    passing_attempts: float
    passing_yards: float
    passing_touchdowns: float
    passing_interceptions: float
    passing_two_point_attempts: float
    passing_two_point_made: float
    rushing_attempts: float
    rushing_yards: float
    rushing_touchdowns: float
    rushing_two_point_attempts: float
    rushing_two_point_made: float
    fumbles: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DefenseGameStats:
    """Statistics allowed by one defense in one game."""

    points_allowed: float
    passing_yards_allowed: float
    rushing_yards_allowed: float
    turnovers: float


@dataclass(frozen=True, slots=True, kw_only=True)
class QuarterbackGame:
    """A quarterback's normalized game record."""

    key: GameKey
    context: GameContext
    stats: QuarterbackGameStats


@dataclass(frozen=True, slots=True, kw_only=True)
class DefenseGame:
    """A defense's normalized game record."""

    key: GameKey
    context: GameContext
    stats: DefenseGameStats


@dataclass(slots=True, kw_only=True)
class QuarterbackHistory:
    """Player profile and games indexed by typed chronological keys."""

    profile: PlayerProfile
    games: dict[GameKey, QuarterbackGame] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class DefenseHistory:
    """Defense games indexed by typed chronological keys."""

    team: TeamCode
    games: dict[GameKey, DefenseGame] = field(default_factory=dict)

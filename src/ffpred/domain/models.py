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


@dataclass(frozen=True, slots=True, kw_only=True)
class DstGameStats:
    """A team's own defense/special-teams counting statistics for one game.

    Distinct from ``DefenseGameStats``, which records what an *opponent's*
    offense produced against a team's defense (used as opponent-context
    features elsewhere). These fields are the team's own defensive/special-
    teams production, which is what a team-defense (D/ST) fantasy roster
    slot is scored on.
    """

    points_allowed: float
    sacks: float
    interceptions: float
    fumble_recoveries: float
    touchdowns: float
    safeties: float
    blocked_kicks: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DstGame:
    """A team defense/special-teams normalized game record."""

    key: GameKey
    context: GameContext
    stats: DstGameStats


@dataclass(slots=True, kw_only=True)
class DstHistory:
    """Team defense/special-teams games indexed by typed chronological keys."""

    team: TeamCode
    games: dict[GameKey, DstGame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class KickerGameStats:
    """A kicker's counting statistics for one game.

    Field-goal makes are grouped into the distance bands most fantasy scoring
    formats use (0-39, 40-49, 50+) rather than nflverse's finer buckets, since
    scoring never distinguishes further than that.
    """

    fg_made_0_39: float
    fg_made_40_49: float
    fg_made_50_plus: float
    fg_missed: float
    pat_made: float
    pat_missed: float


@dataclass(frozen=True, slots=True, kw_only=True)
class KickerGame:
    """A kicker's normalized game record.

    Kicker scoring needs no opponent context, so this omits the full
    ``GameContext`` team-and-opponent scaffolding used by the QB/D/ST records
    and keeps only the identifiers a feature row needs.
    """

    key: GameKey
    game_id: GameId
    stats: KickerGameStats


@dataclass(slots=True, kw_only=True)
class KickerHistory:
    """A kicker's profile and games indexed by typed chronological keys."""

    player_id: PlayerId
    name: str
    games: dict[GameKey, KickerGame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceivingGameStats:
    """An RB/WR/TE's rushing and receiving counting statistics for one game.

    Two-point-conversion *attempts* are tracked for the QB record (which
    needs a play-by-play join to count them at all); receiving positions
    only need the *made* count for scoring, so this omits attempts and the
    play-by-play join it would require.
    """

    rushing_attempts: float
    rushing_yards: float
    rushing_touchdowns: float
    rushing_two_point_made: float
    receptions: float
    targets: float
    receiving_yards: float
    receiving_touchdowns: float
    receiving_two_point_made: float
    fumbles: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceivingGame:
    """An RB/WR/TE's normalized game record."""

    key: GameKey
    context: GameContext
    stats: ReceivingGameStats


@dataclass(slots=True, kw_only=True)
class ReceivingHistory:
    """An RB/WR/TE's profile and games indexed by typed chronological keys."""

    player_id: PlayerId
    name: str
    position: str
    games: dict[GameKey, ReceivingGame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class IdpGameStats:
    """An individual defensive player's (IDP) counting statistics for one game.

    Nflverse tackle attribution is known to be less consistently officiated
    than offensive box scores, and finer defensive charting (e.g. QB hits)
    is not populated in the earliest seasons; see acquire_idp_histories for
    the historical-coverage floor this project applies as a result.
    """

    solo_tackles: float
    assisted_tackles: float
    sacks: float
    interceptions: float
    passes_defended: float
    fumbles_forced: float
    touchdowns: float


@dataclass(frozen=True, slots=True, kw_only=True)
class IdpGame:
    """An IDP's normalized game record.

    Like the kicker record, this omits opponent context: no opponent-strength
    feature is computed for IDP in this first release.
    """

    key: GameKey
    game_id: GameId
    stats: IdpGameStats


@dataclass(slots=True, kw_only=True)
class IdpHistory:
    """An IDP's profile and games indexed by typed chronological keys."""

    player_id: PlayerId
    name: str
    position_group: str
    games: dict[GameKey, IdpGame] = field(default_factory=dict)

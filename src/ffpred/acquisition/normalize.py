"""Convert provider frames into normalized domain histories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from ffpred.acquisition.contracts import (
    DEFAULT_SEASONS,
    PBP_CONTRACT,
    PLAYER_STATS_CONTRACT,
    PLAYERS_CONTRACT,
    REGULAR_SEASON,
    SCHEDULES_CONTRACT,
    TEAM_STATS_CONTRACT,
)
from ffpred.acquisition.schema import validate_frame
from ffpred.domain.identifiers import GameId, PlayerId, Season, TeamCode, Week
from ffpred.domain.models import (
    DefenseGame,
    DefenseGameStats,
    DefenseHistory,
    GameContext,
    GameKey,
    PlayerProfile,
    QuarterbackGame,
    QuarterbackGameStats,
    QuarterbackHistory,
)
from ffpred.errors import DataAcquisitionError
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleRecord:
    """Normalized schedule row used while joining provider frames."""

    game_id: GameId
    game_date: date
    home_team: TeamCode
    away_team: TeamCode
    home_score: float
    away_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerRecord:
    """Normalized player metadata used while joining provider frames."""

    name: str
    birth_date: date | None
    rookie_season: Season | None


def _number(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    raise DataAcquisitionError(f"Expected a number, received {value!r}")


def _required_text(value: object, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise DataAcquisitionError(f"Expected non-empty text for {field}")


def _date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise DataAcquisitionError(
                f"Expected an ISO date for {field}, received {value!r}"
            ) from error
    raise DataAcquisitionError(f"Expected a date for {field}, received {value!r}")


def _optional_date(value: object, field: str) -> date | None:
    return None if value is None else _date(value, field)


def _optional_season(value: object) -> Season | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return Season(int(value))
    raise DataAcquisitionError(f"Expected a numeric rookie season, received {value!r}")


def _schedule_index(frame: pl.DataFrame) -> dict[GameId, ScheduleRecord]:
    schedules: dict[GameId, ScheduleRecord] = {}
    for row in validate_frame(frame, SCHEDULES_CONTRACT).iter_rows(named=True):
        game_id = GameId(_required_text(row["game_id"], "game_id"))
        schedules[game_id] = ScheduleRecord(
            game_id=game_id,
            game_date=_date(row["gameday"], "gameday"),
            home_team=TeamCode(_required_text(row["home_team"], "home_team")),
            away_team=TeamCode(_required_text(row["away_team"], "away_team")),
            home_score=_number(row["home_score"]),
            away_score=_number(row["away_score"]),
        )
    return schedules


def _player_index(frame: pl.DataFrame) -> dict[PlayerId, PlayerRecord]:
    players: dict[PlayerId, PlayerRecord] = {}
    for row in validate_frame(frame, PLAYERS_CONTRACT).iter_rows(named=True):
        raw_id = row["gsis_id"]
        if not raw_id:
            continue
        player_id = PlayerId(_required_text(raw_id, "gsis_id"))
        players[player_id] = PlayerRecord(
            name=_required_text(row["display_name"], "display_name"),
            birth_date=_optional_date(row["birth_date"], "birth_date"),
            rookie_season=_optional_season(row["rookie_season"]),
        )
    return players


def acquire_two_point_attempts(
    seasons: Iterable[int],
    provider: NflDataProvider,
) -> dict[tuple[GameId, PlayerId], tuple[int, int]]:
    """Count passing and rushing two-point attempts by game and player."""
    attempts: dict[tuple[GameId, PlayerId], list[int]] = {}
    columns = [
        "game_id",
        "season_type",
        "two_point_attempt",
        "passer_player_id",
        "rusher_player_id",
    ]
    for season in seasons:
        plays = validate_frame(provider.load_pbp(season), PBP_CONTRACT).select(columns)
        plays = plays.filter(
            (pl.col("season_type") == REGULAR_SEASON)
            & (pl.col("two_point_attempt") == 1)
        )
        for row in plays.iter_rows(named=True):
            raw_player_id = row["passer_player_id"] or row["rusher_player_id"]
            if not raw_player_id:
                continue
            key = (
                GameId(_required_text(row["game_id"], "game_id")),
                PlayerId(_required_text(raw_player_id, "player_id")),
            )
            counts = attempts.setdefault(key, [0, 0])
            counts[0 if row["passer_player_id"] else 1] += 1
    return {key: (counts[0], counts[1]) for key, counts in attempts.items()}


def _game_context(
    schedule: ScheduleRecord,
    *,
    team: TeamCode,
    opponent: TeamCode,
) -> GameContext:
    return GameContext(
        game_id=schedule.game_id,
        game_date=schedule.game_date,
        home_team=schedule.home_team,
        away_team=schedule.away_team,
        team=team,
        opponent=opponent,
    )


def acquire_quarterback_histories(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    *,
    provider: NflDataProvider | None = None,
    min_attempts: int = 5,
) -> dict[PlayerId, QuarterbackHistory]:
    """Acquire and normalize regular-season quarterback histories."""
    season_list = sorted(set(seasons))
    provider = provider or NflReadPyProvider()
    schedules = _schedule_index(provider.load_schedules(season_list))
    players = _player_index(provider.load_players())
    attempts = acquire_two_point_attempts(season_list, provider)
    frame = validate_frame(
        provider.load_player_stats(season_list), PLAYER_STATS_CONTRACT
    ).filter(
        (pl.col("season_type") == REGULAR_SEASON)
        & (pl.col("position") == "QB")
        & (pl.col("attempts") >= min_attempts)
    )

    histories: dict[PlayerId, QuarterbackHistory] = {}
    for row in frame.iter_rows(named=True):
        player_id = PlayerId(_required_text(row["player_id"], "player_id"))
        game_id = GameId(_required_text(row["game_id"], "game_id"))
        if game_id not in schedules:
            raise DataAcquisitionError(f"No schedule row found for game {game_id}")
        metadata = players.get(player_id)
        name = (
            metadata.name
            if metadata
            else _required_text(row["player_display_name"], "player_display_name")
        )
        history = histories.setdefault(
            player_id,
            QuarterbackHistory(
                profile=PlayerProfile(
                    player_id=player_id,
                    name=name,
                    birth_date=metadata.birth_date if metadata else None,
                    rookie_season=metadata.rookie_season if metadata else None,
                )
            ),
        )
        key = GameKey(Season(int(row["season"])), Week(int(row["week"])))
        passing_attempts, rushing_attempts = attempts.get((game_id, player_id), (0, 0))
        history.games[key] = QuarterbackGame(
            key=key,
            context=_game_context(
                schedules[game_id],
                team=TeamCode(_required_text(row["team"], "team")),
                opponent=TeamCode(
                    _required_text(row["opponent_team"], "opponent_team")
                ),
            ),
            stats=QuarterbackGameStats(
                passing_attempts=_number(row["attempts"]),
                passing_yards=_number(row["passing_yards"]),
                passing_touchdowns=_number(row["passing_tds"]),
                passing_interceptions=_number(row["passing_interceptions"]),
                passing_two_point_attempts=passing_attempts,
                passing_two_point_made=_number(row["passing_2pt_conversions"]),
                rushing_attempts=_number(row["carries"]),
                rushing_yards=_number(row["rushing_yards"]),
                rushing_touchdowns=_number(row["rushing_tds"]),
                rushing_two_point_attempts=rushing_attempts,
                rushing_two_point_made=_number(row["rushing_2pt_conversions"]),
                fumbles=_number(row["fumbles_total"]),
            ),
        )
    return histories


def acquire_defense_histories(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    *,
    provider: NflDataProvider | None = None,
) -> dict[TeamCode, DefenseHistory]:
    """Acquire defensive totals by attributing opponent offense to each defense."""
    season_list = sorted(set(seasons))
    provider = provider or NflReadPyProvider()
    schedules = _schedule_index(provider.load_schedules(season_list))
    frame = validate_frame(
        provider.load_team_stats(season_list), TEAM_STATS_CONTRACT
    ).filter(pl.col("season_type") == REGULAR_SEASON)

    histories: dict[TeamCode, DefenseHistory] = {}
    for row in frame.iter_rows(named=True):
        game_id = GameId(_required_text(row["game_id"], "game_id"))
        if game_id not in schedules:
            raise DataAcquisitionError(f"No schedule row found for game {game_id}")
        schedule = schedules[game_id]
        offense = TeamCode(_required_text(row["team"], "team"))
        defense = TeamCode(_required_text(row["opponent_team"], "opponent_team"))
        if offense == schedule.home_team:
            points_allowed = schedule.home_score
        elif offense == schedule.away_team:
            points_allowed = schedule.away_score
        else:
            raise DataAcquisitionError(
                f"Team {offense} is not listed in schedule for {game_id}"
            )
        key = GameKey(Season(int(row["season"])), Week(int(row["week"])))
        history = histories.setdefault(
            defense,
            DefenseHistory(team=defense),
        )
        history.games[key] = DefenseGame(
            key=key,
            context=_game_context(schedule, team=defense, opponent=offense),
            stats=DefenseGameStats(
                points_allowed=points_allowed,
                passing_yards_allowed=_number(row["passing_yards"]),
                rushing_yards_allowed=_number(row["rushing_yards"]),
                turnovers=_number(row["passing_interceptions"])
                + _number(row["fumbles_lost_total"]),
            ),
        )
    return histories

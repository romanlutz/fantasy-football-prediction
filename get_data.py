# Copyright (c) 2026 Roman Lutz
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import version
from typing import Any

import nflreadpy as nfl
import polars as pl

DEFAULT_SEASONS = tuple(range(2009, 2015))
REGULAR_SEASON = "REG"
NFLVERSE_DATA_URL = "https://github.com/nflverse/nflverse-data/releases"


def create_empty_entry(
    seasons: Iterable[int] = DEFAULT_SEASONS, max_week: int = 18
) -> dict[str, dict[str, dict[str, bool]]]:
    return {
        str(year): {
            str(week): {"played": False} for week in range(1, max_week + 1)
        }
        for year in seasons
    }


def _number(value: Any) -> int | float:
    return 0 if value is None else value


def _regular_season(frame: pl.DataFrame) -> pl.DataFrame:
    if "season_type" not in frame.columns:
        raise KeyError("Expected nflverse column 'season_type'")
    return frame.filter(pl.col("season_type") == REGULAR_SEASON)


def _schedule_index(schedules: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        row["game_id"]: row
        for row in schedules.iter_rows(named=True)
        if row.get("game_id")
    }


def _player_index(players: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        row["gsis_id"]: row
        for row in players.iter_rows(named=True)
        if row.get("gsis_id")
    }


def load_two_point_attempts(
    seasons: Iterable[int],
    load_pbp: Callable[[int], pl.DataFrame] = nfl.load_pbp,
) -> dict[tuple[str, str], dict[str, int]]:
    attempts: dict[tuple[str, str], dict[str, int]] = {}
    columns = [
        "game_id",
        "season_type",
        "two_point_attempt",
        "passer_player_id",
        "rusher_player_id",
    ]

    for season in seasons:
        plays = load_pbp(season).select(columns)
        plays = plays.filter(
            (pl.col("season_type") == REGULAR_SEASON)
            & (pl.col("two_point_attempt") == 1)
        )
        for play in plays.iter_rows(named=True):
            if play["passer_player_id"]:
                player_id = play["passer_player_id"]
                attempt_type = "passing"
            elif play["rusher_player_id"]:
                player_id = play["rusher_player_id"]
                attempt_type = "rushing"
            else:
                continue

            key = (play["game_id"], player_id)
            counts = attempts.setdefault(key, {"passing": 0, "rushing": 0})
            counts[attempt_type] += 1

    return attempts


def fetch_qb_stats(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    min_attempts: int = 5,
    *,
    player_stats: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
    two_point_attempts: Mapping[tuple[str, str], Mapping[str, int]] | None = None,
    load_pbp: Callable[[int], pl.DataFrame] = nfl.load_pbp,
) -> dict[str, dict[str, Any]]:
    season_list = sorted(set(seasons))
    player_stats = (
        nfl.load_player_stats(season_list) if player_stats is None else player_stats
    )
    players = nfl.load_players() if players is None else players
    schedules = nfl.load_schedules(season_list) if schedules is None else schedules
    if two_point_attempts is None:
        two_point_attempts = load_two_point_attempts(season_list, load_pbp)

    games = _schedule_index(schedules)
    player_details = _player_index(players)
    quarterbacks = _regular_season(player_stats).filter(
        (pl.col("position") == "QB") & (pl.col("attempts") >= min_attempts)
    )

    statistics: dict[str, dict[str, Any]] = {}
    for row in quarterbacks.iter_rows(named=True):
        player_id = row["player_id"]
        season = str(row["season"])
        week = str(row["week"])
        game = games.get(row["game_id"], {})
        details = player_details.get(player_id, {})
        attempts = two_point_attempts.get(
            (row["game_id"], player_id), {"passing": 0, "rushing": 0}
        )

        if player_id not in statistics:
            statistics[player_id] = create_empty_entry(season_list)
            statistics[player_id].update(
                {
                    "name": details.get("display_name")
                    or row.get("player_display_name"),
                    "birthdate": details.get("birth_date"),
                    "rookie_season": details.get("rookie_season"),
                }
            )

        statistics[player_id].setdefault(season, {})
        statistics[player_id][season].setdefault(week, {"played": False})
        statistics[player_id][season][week] = {
            "game_id": row["game_id"],
            "game_date": game.get("gameday"),
            "home": game.get("home_team"),
            "away": game.get("away_team"),
            "team": row.get("team"),
            "opponent": row.get("opponent_team"),
            "passing_attempts": _number(row.get("attempts")),
            "passing_yards": _number(row.get("passing_yards")),
            "passing_touchdowns": _number(row.get("passing_tds")),
            "passing_interceptions": _number(row.get("passing_interceptions")),
            "passing_two_point_attempts": attempts["passing"],
            "passing_two_point_made": _number(
                row.get("passing_2pt_conversions")
            ),
            "rushing_attempts": _number(row.get("carries")),
            "rushing_yards": _number(row.get("rushing_yards")),
            "rushing_touchdowns": _number(row.get("rushing_tds")),
            "rushing_two_point_attempts": attempts["rushing"],
            "rushing_two_point_made": _number(
                row.get("rushing_2pt_conversions")
            ),
            "fumbles": _number(row.get("fumbles_total")),
            "played": True,
        }

    return statistics


def fetch_defense_stats(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    *,
    team_stats: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    season_list = sorted(set(seasons))
    team_stats = nfl.load_team_stats(season_list) if team_stats is None else team_stats
    schedules = nfl.load_schedules(season_list) if schedules is None else schedules
    games = _schedule_index(schedules)
    regular_stats = _regular_season(team_stats)

    teams = {
        team
        for column in ("team", "opponent_team")
        for team in regular_stats[column].drop_nulls().to_list()
    }
    statistics: dict[str, dict[str, Any]] = {
        team: create_empty_entry(season_list) for team in teams
    }

    for row in regular_stats.iter_rows(named=True):
        offense = row["team"]
        defense = row["opponent_team"]
        if not offense or not defense:
            continue

        game = games.get(row["game_id"], {})
        if offense == game.get("home_team"):
            points_allowed = game.get("home_score")
        elif offense == game.get("away_team"):
            points_allowed = game.get("away_score")
        else:
            continue

        season = str(row["season"])
        week = str(row["week"])
        statistics[defense].setdefault(season, {})
        statistics[defense][season].setdefault(week, {"played": False})
        statistics[defense][season][week] = {
            "game_id": row["game_id"],
            "game_date": game.get("gameday"),
            "home": game.get("home_team"),
            "away": game.get("away_team"),
            "points_allowed": _number(points_allowed),
            "passing_yards_allowed": _number(row.get("passing_yards")),
            "rushing_yards_allowed": _number(row.get("rushing_yards")),
            "turnovers": _number(row.get("passing_interceptions"))
            + _number(row.get("fumbles_lost_total")),
            "played": True,
        }

    return statistics


def source_metadata() -> dict[str, str]:
    return {
        "client": "nflreadpy",
        "client_version": version("nflreadpy"),
        "data_source": NFLVERSE_DATA_URL,
    }


test_players = {
    "00-0029263": "Russell Wilson",
    "00-0023459": "Aaron Rodgers",
    "00-0026143": "Matt Ryan",
    "00-0020531": "Drew Brees",
    "00-0026158": "Joe Flacco",
    "00-0027973": "Andy Dalton",
    "00-0024226": "Jay Cutler",
    "00-0023436": "Alex Smith",
    "00-0029701": "Ryan Tannehill",
    "00-0019596": "Tom Brady",
    "00-0031280": "Derek Carr",
    "00-0022924": "Ben Roethlisberger",
    "00-0026625": "Brian Hoyer",
    "00-0021678": "Tony Romo",
    "00-0027974": "Colin Kaepernick",
    "00-0010346": "Peyton Manning",
    "00-0029668": "Andrew Luck",
    "00-0026498": "Matthew Stafford",
    "00-0022803": "Eli Manning",
    "00-0022942": "Philip Rivers",
    "00-0027939": "Cam Newton",
    "00-0031237": "Teddy Bridgewater",
    "00-0031407": "Blake Bortles",
    "00-0023541": "Kyle Orton",
}

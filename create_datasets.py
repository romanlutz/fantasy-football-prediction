# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from get_data import fetch_defense_stats, fetch_qb_stats, source_metadata

QB_FIELDS = (
    "passing_attempts",
    "passing_yards",
    "passing_touchdowns",
    "passing_interceptions",
    "passing_two_point_attempts",
    "passing_two_point_made",
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "rushing_two_point_attempts",
    "rushing_two_point_made",
    "fumbles",
)
DEFENSE_FIELDS = (
    "points_allowed",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "turnovers",
)


def _played_games_before(
    statistics: dict[str, dict[str, Any]], identifier: str, year: int, week: int
) -> list[tuple[int, int, dict[str, Any]]]:
    games: list[tuple[int, int, dict[str, Any]]] = []
    for season_key, season_data in statistics[identifier].items():
        if not str(season_key).isdigit() or not isinstance(season_data, dict):
            continue
        season = int(season_key)
        for week_key, game in season_data.items():
            if not str(week_key).isdigit() or not game.get("played"):
                continue
            game_week = int(week_key)
            if (season, game_week) < (year, week):
                games.append((season, game_week, game))
    return sorted(games, key=lambda item: (item[0], item[1]), reverse=True)


def last_game(
    statistics: dict[str, dict[str, Any]], identifier: str, year: int, week: int
) -> tuple[dict[str, Any] | None, int | None, int | None]:
    games = _played_games_before(statistics, identifier, year, week)
    if not games:
        return None, None, None
    game_year, game_week, game = games[0]
    return game, game_year, game_week


def last_k_games(
    k: int,
    statistics: dict[str, dict[str, Any]],
    identifier: str,
    year: int,
    week: int,
) -> list[dict[str, Any]]:
    return [
        game
        for _, _, game in _played_games_before(statistics, identifier, year, week)[:k]
    ]


def _average_stats(
    games: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, float] | None:
    if not games:
        return None
    return {
        field: float(sum(game[field] for game in games)) / len(games)
        for field in fields
    }


def average_defense_stats(
    games: list[dict[str, Any]],
) -> dict[str, float] | None:
    return _average_stats(games, DEFENSE_FIELDS)


def average_qb_stats(games: list[dict[str, Any]]) -> dict[str, float] | None:
    return _average_stats(games, QB_FIELDS)


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for date_format in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def calculate_age(
    birthdate: str | date | datetime, game_date: str | date | datetime
) -> float:
    born = _parse_date(birthdate)
    played = _parse_date(game_date)
    return (played - born).days / 365.2425


def fantasy_score(
    passing_yards: float,
    passing_touchdowns: float,
    interceptions: float,
    rushing_yards: float,
    rushing_touchdowns: float,
    fumbles: float,
    two_point: float,
) -> float:
    return (
        passing_yards / 25
        + passing_touchdowns * 4
        - interceptions * 2
        + rushing_yards / 10
        + rushing_touchdowns * 6
        - fumbles * 2
        + two_point * 2
    )


def create_row(
    qb_statistics: dict[str, dict[str, Any]],
    defense_statistics: dict[str, dict[str, Any]],
    rookie_statistics: dict[str, float],
    identifier: str,
    year: int,
    week: int,
) -> list[Any] | None:
    player = qb_statistics[identifier]
    current_game = player[str(year)][str(week)]
    if not player.get("birthdate") or not current_game.get("game_date"):
        return None

    last_game_qb = (
        average_qb_stats(last_k_games(1, qb_statistics, identifier, year, week))
        or rookie_statistics
    )
    last_10_qb = (
        average_qb_stats(last_k_games(10, qb_statistics, identifier, year, week))
        or rookie_statistics
    )

    opponent = current_game.get("opponent")
    if not opponent or opponent not in defense_statistics:
        return None
    last_game_defense = average_defense_stats(
        last_k_games(1, defense_statistics, opponent, year, week)
    )
    last_10_defense = average_defense_stats(
        last_k_games(10, defense_statistics, opponent, year, week)
    )
    if last_game_defense is None or last_10_defense is None:
        return None

    rookie_season = player.get("rookie_season")
    years_pro = max(0, year - int(rookie_season)) if rookie_season else 0
    score = fantasy_score(
        current_game["passing_yards"],
        current_game["passing_touchdowns"],
        current_game["passing_interceptions"],
        current_game["rushing_yards"],
        current_game["rushing_touchdowns"],
        current_game["fumbles"],
        current_game["rushing_two_point_made"] + current_game["passing_two_point_made"],
    )

    return [
        identifier,
        player["name"],
        calculate_age(player["birthdate"], current_game["game_date"]),
        years_pro,
        *(last_game_qb[field] for field in QB_FIELDS),
        *(last_10_qb[field] for field in QB_FIELDS),
        *(last_game_defense[field] for field in DEFENSE_FIELDS),
        *(last_10_defense[field] for field in DEFENSE_FIELDS),
        score,
    ]


def rookie_qb_average(
    qb_statistics: dict[str, dict[str, Any]],
    before: tuple[int, int] | None = None,
) -> dict[str, float] | None:
    games: list[dict[str, Any]] = []
    for player in qb_statistics.values():
        rookie_season = player.get("rookie_season")
        season_data = player.get(str(rookie_season), {}) if rookie_season else {}
        for week_key, game in season_data.items():
            if not str(week_key).isdigit() or not game.get("played"):
                continue
            if before is None or (int(rookie_season), int(week_key)) < before:
                games.append(game)
    return average_qb_stats(games)


def create_all_rows(
    qb_statistics: dict[str, dict[str, Any]],
    defense_statistics: dict[str, dict[str, Any]],
    start_year: int,
    end_year: int,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    rookie_stats_by_week: dict[tuple[int, int], dict[str, float] | None] = {}
    for year in range(start_year, end_year):
        for player_id, player in qb_statistics.items():
            for week_key, game in player.get(str(year), {}).items():
                if not str(week_key).isdigit() or not game.get("played"):
                    continue
                week = int(week_key)
                cutoff = (year, week)
                if cutoff not in rookie_stats_by_week:
                    rookie_stats_by_week[cutoff] = rookie_qb_average(
                        qb_statistics, before=cutoff
                    )
                rookie_stats = rookie_stats_by_week[cutoff]
                if rookie_stats is None:
                    continue
                row = create_row(
                    qb_statistics,
                    defense_statistics,
                    rookie_stats,
                    player_id,
                    year,
                    week,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _save_array(path: Path, rows: list[list[Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            f"No rows were generated for {path.stem}; check season availability "
            "and ensure the history range includes prior games"
        )
    array = np.asarray(rows, dtype=str)
    np.save(path, array, allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "rows": len(rows), "sha256": digest}


def generate_datasets(
    output_dir: Path,
    history_start: int = 2009,
    train_start: int = 2010,
    test_year: int = 2014,
) -> dict[str, Any]:
    seasons = range(history_start, test_year + 1)
    qb_statistics = fetch_qb_stats(seasons)
    defense_statistics = fetch_defense_stats(seasons)
    train_rows = create_all_rows(
        qb_statistics, defense_statistics, train_start, test_year
    )
    test_rows = create_all_rows(
        qb_statistics, defense_statistics, test_year, test_year + 1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **source_metadata(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_start": history_start,
        "train_start": train_start,
        "test_year": test_year,
        "outputs": {
            "train": _save_array(output_dir / "train.npy", train_rows),
            "test": _save_array(output_dir / "test.npy", test_rows),
        },
    }
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fantasy football datasets")
    parser.add_argument("--output-dir", type=Path, default=Path())
    parser.add_argument("--history-start", type=int, default=2009)
    parser.add_argument("--train-start", type=int, default=2010)
    parser.add_argument("--test-year", type=int, default=2014)
    args = parser.parse_args()

    manifest = generate_datasets(
        args.output_dir, args.history_start, args.train_start, args.test_year
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

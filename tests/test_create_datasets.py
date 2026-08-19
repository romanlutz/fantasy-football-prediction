from datetime import date

import numpy as np
import pytest

from create_datasets import (
    _save_array,
    average_qb_stats,
    calculate_age,
    fantasy_score,
    last_k_games,
    rookie_qb_average,
)
from metrics import mean_relative_error


def test_last_k_games_crosses_season_boundary() -> None:
    statistics = {
        "player": {
            "2009": {
                "17": {"played": True, "passing_yards": 200},
            },
            "2010": {
                "1": {"played": True, "passing_yards": 250},
                "2": {"played": False},
            },
        }
    }

    games = last_k_games(2, statistics, "player", 2010, 2)

    assert [game["passing_yards"] for game in games] == [250, 200]


def test_average_qb_stats_preserves_feature_contract() -> None:
    game = {
        "passing_attempts": 20,
        "passing_yards": 250,
        "passing_touchdowns": 2,
        "passing_interceptions": 1,
        "passing_two_point_attempts": 1,
        "passing_two_point_made": 1,
        "rushing_attempts": 4,
        "rushing_yards": 20,
        "rushing_touchdowns": 1,
        "rushing_two_point_attempts": 0,
        "rushing_two_point_made": 0,
        "fumbles": 1,
    }

    average = average_qb_stats([game, game])

    assert average == {key: float(value) for key, value in game.items()}


def test_age_uses_actual_game_date() -> None:
    assert calculate_age(date(1985, 1, 1), date(2015, 1, 1)) == pytest.approx(
        30.0, abs=0.01
    )


def test_standard_fantasy_score() -> None:
    assert fantasy_score(250, 2, 1, 20, 1, 1, 1) == 24


def test_mean_relative_error_uses_actual_values() -> None:
    result = mean_relative_error(np.array([10, 20]), np.array([12, 16]))
    assert result == pytest.approx(0.2)


def test_rookie_average_excludes_future_games() -> None:
    game = {
        "played": True,
        "passing_attempts": 20,
        "passing_yards": 100,
        "passing_touchdowns": 1,
        "passing_interceptions": 0,
        "passing_two_point_attempts": 0,
        "passing_two_point_made": 0,
        "rushing_attempts": 1,
        "rushing_yards": 2,
        "rushing_touchdowns": 0,
        "rushing_two_point_attempts": 0,
        "rushing_two_point_made": 0,
        "fumbles": 0,
    }
    future_game = {**game, "passing_yards": 999}
    statistics = {
        "past": {"rookie_season": 2009, "2009": {"1": game}},
        "future": {"rookie_season": 2014, "2014": {"1": future_game}},
    }

    average = rookie_qb_average(statistics, before=(2010, 1))

    assert average is not None
    assert average["passing_yards"] == 100


def test_empty_dataset_has_actionable_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="No rows were generated"):
        _save_array(tmp_path / "train.npy", [])

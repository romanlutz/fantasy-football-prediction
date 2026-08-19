import polars as pl

from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_quarterback_histories,
)
from ffpred.domain.identifiers import PlayerId, Season, TeamCode, Week
from ffpred.domain.models import GameKey
from ffpred.providers.fakes import FakeProvider


def _provider() -> FakeProvider:
    schedules = pl.DataFrame(
        [
            {
                "game_id": "2014_01_GB_SEA",
                "gameday": "2014-09-04",
                "home_team": "SEA",
                "away_team": "GB",
                "home_score": 36,
                "away_score": 16,
            }
        ]
    )
    return FakeProvider(
        player_stats=pl.DataFrame(
            [
                {
                    "player_id": "00-TEST",
                    "player_display_name": "Test Quarterback",
                    "position": "QB",
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "team": "GB",
                    "opponent_team": "SEA",
                    "attempts": 30,
                    "passing_yards": 250,
                    "passing_tds": 2,
                    "passing_interceptions": 1,
                    "passing_2pt_conversions": 1,
                    "carries": 3,
                    "rushing_yards": 20,
                    "rushing_tds": 0,
                    "rushing_2pt_conversions": 0,
                    "fumbles_total": 1,
                }
            ]
        ),
        players=pl.DataFrame(
            [
                {
                    "gsis_id": "00-TEST",
                    "display_name": "Test Quarterback",
                    "birth_date": "1985-01-02",
                    "rookie_season": 2008,
                }
            ]
        ),
        schedules=schedules,
        team_stats=pl.DataFrame(
            [
                {
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "team": "GB",
                    "opponent_team": "SEA",
                    "passing_yards": 189,
                    "rushing_yards": 80,
                    "passing_interceptions": 1,
                    "fumbles_lost_total": 1,
                }
            ]
        ),
        pbp_by_season={
            2014: pl.DataFrame(
                [
                    {
                        "game_id": "2014_01_GB_SEA",
                        "season_type": "REG",
                        "two_point_attempt": 1,
                        "passer_player_id": "00-TEST",
                        "rusher_player_id": None,
                    }
                ]
            )
        },
    )


def test_quarterback_acquisition_returns_typed_history() -> None:
    histories = acquire_quarterback_histories([2014], provider=_provider())

    history = histories[PlayerId("00-TEST")]
    game = history.games[GameKey(Season(2014), Week(1))]
    assert history.profile.name == "Test Quarterback"
    assert game.context.opponent == TeamCode("SEA")
    assert game.stats.passing_two_point_attempts == 1


def test_defense_acquisition_attributes_opponent_offense() -> None:
    histories = acquire_defense_histories([2014], provider=_provider())

    game = histories[TeamCode("SEA")].games[GameKey(Season(2014), Week(1))]
    assert game.stats.points_allowed == 16
    assert game.stats.passing_yards_allowed == 189
    assert game.stats.turnovers == 2

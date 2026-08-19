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


def test_defense_acquisition_reconciles_relocated_team_codes() -> None:
    """nflreadpy normalizes player_stats/team_stats team codes to each
    franchise's *current* abbreviation for every historical season (the 2010
    Chargers appear as "LAC"), while load_schedules reports the abbreviation
    contemporaneous to that season (the 2010 Chargers appear as "SD"). Without
    reconciling the two, acquire_defense_histories cannot match a relocated
    team's offense against the schedule's home/away codes for any season
    before its relocation.
    """
    provider = FakeProvider(
        schedules=pl.DataFrame(
            [
                {
                    "game_id": "2010_01_SD_KC",
                    "gameday": "2010-09-12",
                    "home_team": "KC",
                    "away_team": "SD",
                    "home_score": 16,
                    "away_score": 38,
                }
            ]
        ),
        team_stats=pl.DataFrame(
            [
                {
                    "season": 2010,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2010_01_SD_KC",
                    "team": "LAC",
                    "opponent_team": "KC",
                    "passing_yards": 300,
                    "rushing_yards": 120,
                    "passing_interceptions": 0,
                    "fumbles_lost_total": 0,
                },
                {
                    "season": 2010,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2010_01_SD_KC",
                    "team": "KC",
                    "opponent_team": "LAC",
                    "passing_yards": 150,
                    "rushing_yards": 70,
                    "passing_interceptions": 2,
                    "fumbles_lost_total": 1,
                },
            ]
        ),
    )

    histories = acquire_defense_histories([2010], provider=provider)

    key = GameKey(Season(2010), Week(1))
    # KC's defense allowed 38 points, scored by the Chargers' (LAC) offense.
    kc_defense = histories[TeamCode("KC")].games[key]
    assert kc_defense.stats.points_allowed == 38

    # The Chargers' (LAC) defense allowed 16 points, scored by KC's offense.
    lac_defense = histories[TeamCode("LAC")].games[key]
    assert lac_defense.stats.points_allowed == 16

import polars as pl

from ffpred.acquisition.nflverse import fetch_defense_stats, fetch_qb_stats


def test_fetch_qb_stats_maps_nflverse_fields() -> None:
    player_stats = pl.DataFrame(
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
    )
    players = pl.DataFrame(
        [
            {
                "gsis_id": "00-TEST",
                "display_name": "Test Quarterback",
                "birth_date": "1985-01-02",
                "rookie_season": 2008,
                "years_of_experience": 7,
            }
        ]
    )
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

    result = fetch_qb_stats(
        [2014],
        player_stats=player_stats,
        players=players,
        schedules=schedules,
        two_point_attempts={
            ("2014_01_GB_SEA", "00-TEST"): {"passing": 2, "rushing": 1}
        },
    )

    player = result["00-TEST"]
    game = player["2014"]["1"]
    assert player["rookie_season"] == 2008
    assert game["team"] == "GB"
    assert game["opponent"] == "SEA"
    assert game["passing_two_point_attempts"] == 2
    assert game["passing_two_point_made"] == 1
    assert game["rushing_two_point_attempts"] == 1


def test_fetch_defense_stats_uses_opponent_offense() -> None:
    team_stats = pl.DataFrame(
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
            },
            {
                "season": 2014,
                "week": 1,
                "season_type": "REG",
                "game_id": "2014_01_GB_SEA",
                "team": "SEA",
                "opponent_team": "GB",
                "passing_yards": 191,
                "rushing_yards": 207,
                "passing_interceptions": 0,
                "fumbles_lost_total": 0,
            },
        ]
    )
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

    result = fetch_defense_stats([2014], team_stats=team_stats, schedules=schedules)

    seattle = result["SEA"]["2014"]["1"]
    assert seattle["points_allowed"] == 16
    assert seattle["passing_yards_allowed"] == 189
    assert seattle["rushing_yards_allowed"] == 80
    assert seattle["turnovers"] == 2

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ffpred.providers.fakes import FakeProvider


def make_provider(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> FakeProvider:
    schedules: list[dict[str, object]] = []
    player_stats: list[dict[str, object]] = []
    team_stats: list[dict[str, object]] = []
    pbp_by_season: dict[int, pl.DataFrame] = {}
    for index, season in enumerate(seasons):
        game_id = f"{season}_01_GB_SEA"
        schedules.append(
            {
                "game_id": game_id,
                "gameday": (date(season, 9, 1) + timedelta(days=index)).isoformat(),
                "home_team": "SEA",
                "away_team": "GB",
                "home_score": 24 + index,
                "away_score": 20 + index,
            }
        )
        player_stats.append(
            {
                "player_id": "00-TEST",
                "player_display_name": "Test Quarterback",
                "position": "QB",
                "season": season,
                "week": 1,
                "season_type": "REG",
                "game_id": game_id,
                "team": "GB",
                "opponent_team": "SEA",
                "attempts": 30,
                "passing_yards": 200 + index * 10,
                "passing_tds": 2,
                "passing_interceptions": 1,
                "passing_2pt_conversions": 0,
                "carries": 3,
                "rushing_yards": 20,
                "rushing_tds": 0,
                "rushing_2pt_conversions": 0,
                "fumbles_total": 0,
            }
        )
        team_stats.append(
            {
                "season": season,
                "week": 1,
                "season_type": "REG",
                "game_id": game_id,
                "team": "GB",
                "opponent_team": "SEA",
                "passing_yards": 200 + index * 10,
                "rushing_yards": 100,
                "passing_interceptions": 1,
                "fumbles_lost_total": 0,
            }
        )
        pbp_by_season[season] = pl.DataFrame(
            {
                "game_id": [game_id],
                "season_type": ["REG"],
                "two_point_attempt": [0],
                "passer_player_id": [None],
                "rusher_player_id": [None],
            }
        )
    return FakeProvider(
        player_stats=pl.DataFrame(player_stats),
        team_stats=pl.DataFrame(team_stats),
        schedules=pl.DataFrame(schedules),
        players=pl.DataFrame(
            {
                "gsis_id": ["00-TEST"],
                "display_name": ["Test Quarterback"],
                "birth_date": ["1990-01-01"],
                "rookie_season": [2019],
            }
        ),
        pbp_by_season=pbp_by_season,
    )


def make_dst_provider(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> FakeProvider:
    """A provider whose team_stats carry the def_* columns acquire_dst_histories
    needs, across enough seasons/weeks for leakage-safe rolling features.
    """
    schedules: list[dict[str, object]] = []
    team_stats: list[dict[str, object]] = []
    for index, season in enumerate(seasons):
        for week in (1, 2):
            game_id = f"{season}_{week:02d}_GB_SEA"
            schedules.append(
                {
                    "game_id": game_id,
                    "gameday": (
                        date(season, 9, 1) + timedelta(days=index * 7 + week)
                    ).isoformat(),
                    "home_team": "SEA",
                    "away_team": "GB",
                    "home_score": 24 + index,
                    "away_score": 17 + index,
                }
            )
            team_stats.append(
                {
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "team": "SEA",
                    "opponent_team": "GB",
                    "def_sacks": 2 + index,
                    "def_interceptions": 1,
                    "def_tds": 0,
                    "def_safeties": 0,
                    "fumble_recovery_opp": 1,
                    "def_punt_blocks": 0,
                    "def_pat_blocks": 0,
                    "def_fg_blocks": 0,
                }
            )
            team_stats.append(
                {
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "team": "GB",
                    "opponent_team": "SEA",
                    "def_sacks": 1,
                    "def_interceptions": 0,
                    "def_tds": 0,
                    "def_safeties": 0,
                    "fumble_recovery_opp": 0,
                    "def_punt_blocks": 0,
                    "def_pat_blocks": 0,
                    "def_fg_blocks": 0,
                }
            )
    return FakeProvider(
        team_stats=pl.DataFrame(team_stats),
        schedules=pl.DataFrame(schedules),
    )


def make_kicker_provider(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> FakeProvider:
    """A provider whose player_stats carry the fg_*/pat_* columns
    acquire_kicker_histories needs, across enough seasons/weeks for
    leakage-safe rolling features.
    """
    player_stats: list[dict[str, object]] = []
    for index, season in enumerate(seasons):
        for week in (1, 2):
            game_id = f"{season}_{week:02d}_GB_SEA"
            player_stats.append(
                {
                    "player_id": "00-KICKER",
                    "player_display_name": "Test Kicker",
                    "position": "K",
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "fg_made_0_19": 0,
                    "fg_made_20_29": 1,
                    "fg_made_30_39": 1,
                    "fg_made_40_49": 1 + index % 2,
                    "fg_made_50_59": 0,
                    "fg_made_60_": 0,
                    "fg_missed": index % 2,
                    "pat_made": 2,
                    "pat_missed": 0,
                }
            )
    return FakeProvider(player_stats=pl.DataFrame(player_stats))


def make_receiving_provider(
    seasons: tuple[int, ...] = (2020, 2021, 2022),
) -> FakeProvider:
    """A provider whose player_stats/team_stats/schedules carry the columns
    acquire_receiving_histories and acquire_defense_histories need, across
    enough seasons/weeks for leakage-safe rolling features.
    """
    schedules: list[dict[str, object]] = []
    player_stats: list[dict[str, object]] = []
    team_stats: list[dict[str, object]] = []
    for index, season in enumerate(seasons):
        for week in (1, 2):
            game_id = f"{season}_{week:02d}_GB_SEA"
            schedules.append(
                {
                    "game_id": game_id,
                    "gameday": (
                        date(season, 9, 1) + timedelta(days=index * 7 + week)
                    ).isoformat(),
                    "home_team": "SEA",
                    "away_team": "GB",
                    "home_score": 24 + index,
                    "away_score": 17 + index,
                }
            )
            for player_id, name, position in (
                ("00-RB", "Test Running Back", "RB"),
                ("00-WR", "Test Wide Receiver", "WR"),
            ):
                player_stats.append(
                    {
                        "player_id": player_id,
                        "player_display_name": name,
                        "position": position,
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "game_id": game_id,
                        "team": "GB",
                        "opponent_team": "SEA",
                        "carries": 10 + index,
                        "rushing_yards": 40 + index * 5,
                        "rushing_tds": 1,
                        "rushing_2pt_conversions": 0,
                        "receptions": 3,
                        "targets": 5,
                        "receiving_yards": 25 + index,
                        "receiving_tds": 0,
                        "receiving_2pt_conversions": 0,
                        "fumbles_total": 0,
                    }
                )
            team_stats.append(
                {
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "team": "GB",
                    "opponent_team": "SEA",
                    "passing_yards": 200,
                    "rushing_yards": 90,
                    "passing_interceptions": 1,
                    "fumbles_lost_total": 0,
                }
            )
    return FakeProvider(
        player_stats=pl.DataFrame(player_stats),
        team_stats=pl.DataFrame(team_stats),
        schedules=pl.DataFrame(schedules),
    )

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


def make_injury_report_provider(season: int = 2023) -> FakeProvider:
    """A provider whose player_stats/schedules/pbp carry a quarterback who
    plays weeks 1, 2, and 4 (skipping week 3 entirely, as if inactive), plus
    an injuries frame reporting that quarterback as Out in week 3 and
    Questionable in week 4 -- enough to exercise both a missed-game event and
    a played-while-reported event with a real pace/delta comparison.
    """
    played_weeks = (1, 2, 4)
    schedules: list[dict[str, object]] = []
    player_stats: list[dict[str, object]] = []
    for week in played_weeks:
        game_id = f"{season}_{week:02d}_GB_SEA"
        schedules.append(
            {
                "game_id": game_id,
                "gameday": (date(season, 9, 1) + timedelta(days=week * 7)).isoformat(),
                "home_team": "SEA",
                "away_team": "GB",
                "home_score": 24,
                "away_score": 17,
            }
        )
        player_stats.append(
            {
                "player_id": "00-QB",
                "player_display_name": "Test Quarterback",
                "position": "QB",
                "season": season,
                "week": week,
                "season_type": "REG",
                "game_id": game_id,
                "team": "GB",
                "opponent_team": "SEA",
                "attempts": 30,
                # Week 4's much lower output than weeks 1-2 is what creates a
                # meaningful negative delta-vs-pace after the week 3 absence.
                "passing_yards": 300 if week != 4 else 50,
                "passing_tds": 3 if week != 4 else 0,
                "passing_interceptions": 0,
                "passing_2pt_conversions": 0,
                "carries": 0,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "rushing_2pt_conversions": 0,
                "fumbles_total": 0,
                # Present so RECEIVING_PLAYER_STATS_CONTRACT still validates
                # when --positions all is requested against a QB-only frame
                # (no RB/WR/TE rows survive the position filter either way).
                "receptions": 0,
                "targets": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "receiving_2pt_conversions": 0,
            }
        )
    injuries = pl.DataFrame(
        [
            {
                "season": season,
                "week": 3,
                "game_type": "REG",
                "team": "GB",
                "gsis_id": "00-QB",
                "full_name": "Test Quarterback",
                "report_status": "Out",
                "report_primary_injury": "Ankle",
            },
            {
                "season": season,
                "week": 4,
                "game_type": "REG",
                "team": "GB",
                "gsis_id": "00-QB",
                "full_name": "Test Quarterback",
                "report_status": "Questionable",
                "report_primary_injury": "Ankle",
            },
        ]
    )
    return FakeProvider(
        player_stats=pl.DataFrame(player_stats),
        schedules=pl.DataFrame(schedules),
        players=pl.DataFrame(
            {
                "gsis_id": ["00-QB"],
                "display_name": ["Test Quarterback"],
                "birth_date": ["1990-01-01"],
                "rookie_season": [2018],
            }
        ),
        pbp_by_season={
            season: pl.DataFrame(
                {
                    "game_id": [f"{season}_{week:02d}_GB_SEA" for week in played_weeks],
                    "season_type": ["REG"] * len(played_weeks),
                    "two_point_attempt": [0] * len(played_weeks),
                    "passer_player_id": [None] * len(played_weeks),
                    "rusher_player_id": [None] * len(played_weeks),
                }
            )
        },
        injuries=injuries,
    )


def make_idp_provider(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> FakeProvider:
    """A provider whose player_stats carry the def_* columns and
    position_group acquire_idp_histories needs, across enough seasons/weeks
    for leakage-safe rolling features.
    """
    player_stats: list[dict[str, object]] = []
    for index, season in enumerate(seasons):
        for week in (1, 2):
            game_id = f"{season}_{week:02d}_GB_SEA"
            player_stats.append(
                {
                    "player_id": "00-IDP",
                    "player_display_name": "Test Linebacker",
                    "position_group": "LB",
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "def_tackles_solo": 5 + index,
                    "def_tackles_with_assist": 1,
                    "def_sacks": index % 2,
                    "def_interceptions": 0,
                    "def_pass_defended": 1,
                    "def_fumbles_forced": 0,
                    "def_tds": 0,
                }
            )
    return FakeProvider(player_stats=pl.DataFrame(player_stats))

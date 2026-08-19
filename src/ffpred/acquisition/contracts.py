"""nflverse frame contracts used by the acquisition adapter."""

from __future__ import annotations

from ffpred.acquisition.schema import ColumnKind, FrameContract

DEFAULT_SEASONS = tuple(range(2009, 2015))
REGULAR_SEASON = "REG"

PLAYER_STATS_CONTRACT = FrameContract(
    name="player_stats",
    columns={
        "player_id": ColumnKind.TEXT,
        "player_display_name": ColumnKind.TEXT,
        "position": ColumnKind.TEXT,
        "season": ColumnKind.INTEGER,
        "week": ColumnKind.INTEGER,
        "season_type": ColumnKind.TEXT,
        "game_id": ColumnKind.TEXT,
        "team": ColumnKind.TEXT,
        "opponent_team": ColumnKind.TEXT,
        "attempts": ColumnKind.NUMBER,
        "passing_yards": ColumnKind.NUMBER,
        "passing_tds": ColumnKind.NUMBER,
        "passing_interceptions": ColumnKind.NUMBER,
        "passing_2pt_conversions": ColumnKind.NUMBER,
        "carries": ColumnKind.NUMBER,
        "rushing_yards": ColumnKind.NUMBER,
        "rushing_tds": ColumnKind.NUMBER,
        "rushing_2pt_conversions": ColumnKind.NUMBER,
        "fumbles_total": ColumnKind.NUMBER,
    },
    non_null=frozenset({"player_id", "season", "week", "game_id"}),
)
TEAM_STATS_CONTRACT = FrameContract(
    name="team_stats",
    columns={
        "season": ColumnKind.INTEGER,
        "week": ColumnKind.INTEGER,
        "season_type": ColumnKind.TEXT,
        "game_id": ColumnKind.TEXT,
        "team": ColumnKind.TEXT,
        "opponent_team": ColumnKind.TEXT,
        "passing_yards": ColumnKind.NUMBER,
        "rushing_yards": ColumnKind.NUMBER,
        "passing_interceptions": ColumnKind.NUMBER,
        "fumbles_lost_total": ColumnKind.NUMBER,
    },
    non_null=frozenset({"season", "week", "game_id"}),
)
SCHEDULES_CONTRACT = FrameContract(
    name="schedules",
    columns={
        "game_id": ColumnKind.TEXT,
        "gameday": ColumnKind.DATE,
        "home_team": ColumnKind.TEXT,
        "away_team": ColumnKind.TEXT,
        "home_score": ColumnKind.NUMBER,
        "away_score": ColumnKind.NUMBER,
    },
    non_null=frozenset({"game_id", "gameday", "home_team", "away_team"}),
)
PLAYERS_CONTRACT = FrameContract(
    name="players",
    columns={
        "gsis_id": ColumnKind.TEXT,
        "display_name": ColumnKind.TEXT,
        "birth_date": ColumnKind.DATE,
        "rookie_season": ColumnKind.NUMBER,
    },
)
PBP_CONTRACT = FrameContract(
    name="play_by_play",
    columns={
        "game_id": ColumnKind.TEXT,
        "season_type": ColumnKind.TEXT,
        "two_point_attempt": ColumnKind.NUMBER,
        "passer_player_id": ColumnKind.TEXT,
        "rusher_player_id": ColumnKind.TEXT,
    },
)

"""nflverse frame contracts used by the acquisition adapter."""

from __future__ import annotations

from ffpred.acquisition.schema import ColumnKind, FrameContract

DEFAULT_SEASONS = tuple(range(2009, 2015))
REGULAR_SEASON = "REG"

# nflreadpy normalizes `player_stats`/`team_stats` team columns to each
# franchise's *current* abbreviation for every historical season, but
# `load_schedules` reports the abbreviation that was actually in use during
# that season. Both refer to the same franchise, so joins that compare a
# schedule code against a stats code must reconcile them first. This is the
# complete set of nflverse team-code changes since 1999, verified against every
# completed season in nflreadpy 0.1.5: the St. Louis Rams (2016 relocation),
# San Diego Chargers (2017 relocation), and Oakland Raiders (2020 relocation).
RELOCATED_TEAM_CODES = {
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


def normalize_team_code(code: str) -> str:
    """Map a season-contemporaneous team code to its current-franchise code."""
    return RELOCATED_TEAM_CODES.get(code, code)


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
    # nflverse includes aggregate rows without a player_id; the acquisition
    # query filters to quarterback rows before requiring a concrete identifier.
    non_null=frozenset({"season", "week", "game_id"}),
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
DST_TEAM_STATS_CONTRACT = FrameContract(
    name="team_stats",
    columns={
        "season": ColumnKind.INTEGER,
        "week": ColumnKind.INTEGER,
        "season_type": ColumnKind.TEXT,
        "game_id": ColumnKind.TEXT,
        "team": ColumnKind.TEXT,
        "opponent_team": ColumnKind.TEXT,
        "def_sacks": ColumnKind.NUMBER,
        "def_interceptions": ColumnKind.NUMBER,
        "def_tds": ColumnKind.NUMBER,
        "def_safeties": ColumnKind.NUMBER,
        "fumble_recovery_opp": ColumnKind.NUMBER,
        "def_punt_blocks": ColumnKind.NUMBER,
        "def_pat_blocks": ColumnKind.NUMBER,
        "def_fg_blocks": ColumnKind.NUMBER,
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
KICKER_PLAYER_STATS_CONTRACT = FrameContract(
    name="player_stats",
    columns={
        "player_id": ColumnKind.TEXT,
        "player_display_name": ColumnKind.TEXT,
        "position": ColumnKind.TEXT,
        "season": ColumnKind.INTEGER,
        "week": ColumnKind.INTEGER,
        "season_type": ColumnKind.TEXT,
        "game_id": ColumnKind.TEXT,
        "fg_made_0_19": ColumnKind.NUMBER,
        "fg_made_20_29": ColumnKind.NUMBER,
        "fg_made_30_39": ColumnKind.NUMBER,
        "fg_made_40_49": ColumnKind.NUMBER,
        "fg_made_50_59": ColumnKind.NUMBER,
        "fg_made_60_": ColumnKind.NUMBER,
        "fg_missed": ColumnKind.NUMBER,
        "pat_made": ColumnKind.NUMBER,
        "pat_missed": ColumnKind.NUMBER,
    },
    non_null=frozenset({"season", "week", "game_id"}),
)
RECEIVING_PLAYER_STATS_CONTRACT = FrameContract(
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
        "carries": ColumnKind.NUMBER,
        "rushing_yards": ColumnKind.NUMBER,
        "rushing_tds": ColumnKind.NUMBER,
        "rushing_2pt_conversions": ColumnKind.NUMBER,
        "receptions": ColumnKind.NUMBER,
        "targets": ColumnKind.NUMBER,
        "receiving_yards": ColumnKind.NUMBER,
        "receiving_tds": ColumnKind.NUMBER,
        "receiving_2pt_conversions": ColumnKind.NUMBER,
        "fumbles_total": ColumnKind.NUMBER,
    },
    non_null=frozenset({"season", "week", "game_id"}),
)

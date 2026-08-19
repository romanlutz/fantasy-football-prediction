"""Named feature schema for the RB/WR/TE receiving-position target."""

from __future__ import annotations

import polars as pl

from ffpred.errors import SchemaValidationError
from ffpred.features.schema import DEFENSE_STAT_FIELDS

RECEIVING_STAT_FIELDS = (
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "rushing_two_point_made",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_touchdowns",
    "receiving_two_point_made",
    "fumbles",
)

RECEIVING_LAST_ONE_COLUMNS = tuple(
    f"receiving_last_1_{field}" for field in RECEIVING_STAT_FIELDS
)
RECEIVING_LAST_TEN_COLUMNS = tuple(
    f"receiving_last_10_{field}" for field in RECEIVING_STAT_FIELDS
)
DEFENSE_LAST_ONE_COLUMNS = tuple(
    f"defense_last_1_{field}" for field in DEFENSE_STAT_FIELDS
)
DEFENSE_LAST_TEN_COLUMNS = tuple(
    f"defense_last_10_{field}" for field in DEFENSE_STAT_FIELDS
)
MODEL_FEATURE_COLUMNS = (
    *RECEIVING_LAST_ONE_COLUMNS,
    *RECEIVING_LAST_TEN_COLUMNS,
    *DEFENSE_LAST_ONE_COLUMNS,
    *DEFENSE_LAST_TEN_COLUMNS,
)
IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "target_season",
    "target_week",
    "target_game_id",
)
LINEAGE_COLUMNS = (
    "receiving_history_through_season",
    "receiving_history_through_week",
    "defense_history_through_season",
    "defense_history_through_week",
)
TARGET_COLUMN = "fantasy_points"
FEATURE_COLUMNS = (
    *IDENTITY_COLUMNS,
    *LINEAGE_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

FEATURE_SCHEMA = {
    "player_id": pl.String,
    "player_name": pl.String,
    "position": pl.String,
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "receiving_history_through_season": pl.Int64,
    "receiving_history_through_week": pl.Int64,
    "defense_history_through_season": pl.Int64,
    "defense_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}


def validate_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted receiving-position training-data contract."""
    problems: list[str] = []
    if tuple(frame.columns) != FEATURE_COLUMNS:
        problems.append("column order does not match FEATURE_COLUMNS")
    for column, expected in FEATURE_SCHEMA.items():
        if column not in frame.schema:
            continue
        if frame.schema[column] != expected:
            problems.append(
                f"{column!r} is {frame.schema[column]}, expected {expected}"
            )
        if frame[column].null_count():
            problems.append(f"{column!r} contains null values")
    if problems:
        raise SchemaValidationError("receiving_feature_frame", problems)
    return frame

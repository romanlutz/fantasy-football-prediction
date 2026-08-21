"""Named feature schema for the kicker target."""

from __future__ import annotations

import polars as pl

from ffpred.errors import SchemaValidationError

KICKER_STAT_FIELDS = (
    "fg_made_0_39",
    "fg_made_40_49",
    "fg_made_50_plus",
    "fg_missed",
    "pat_made",
    "pat_missed",
)

KICKER_LAST_ONE_COLUMNS = tuple(
    f"kicker_last_1_{field}" for field in KICKER_STAT_FIELDS
)
KICKER_LAST_TEN_COLUMNS = tuple(
    f"kicker_last_10_{field}" for field in KICKER_STAT_FIELDS
)
MODEL_FEATURE_COLUMNS = (*KICKER_LAST_ONE_COLUMNS, *KICKER_LAST_TEN_COLUMNS)
IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "target_season",
    "target_week",
    "target_game_id",
)
LINEAGE_COLUMNS = (
    "kicker_history_through_season",
    "kicker_history_through_week",
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
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "kicker_history_through_season": pl.Int64,
    "kicker_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}


def validate_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted kicker training-data contract."""
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
        raise SchemaValidationError("kicker_feature_frame", problems)
    return frame

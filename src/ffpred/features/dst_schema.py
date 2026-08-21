"""Named feature schema for the team defense/special-teams (D/ST) target."""

from __future__ import annotations

import polars as pl

from ffpred.errors import SchemaValidationError

DST_STAT_FIELDS = (
    "points_allowed",
    "sacks",
    "interceptions",
    "fumble_recoveries",
    "touchdowns",
    "safeties",
    "blocked_kicks",
)

DST_LAST_ONE_COLUMNS = tuple(f"dst_last_1_{field}" for field in DST_STAT_FIELDS)
DST_LAST_TEN_COLUMNS = tuple(f"dst_last_10_{field}" for field in DST_STAT_FIELDS)
MODEL_FEATURE_COLUMNS = (*DST_LAST_ONE_COLUMNS, *DST_LAST_TEN_COLUMNS)
IDENTITY_COLUMNS = (
    "team",
    "target_season",
    "target_week",
    "target_game_id",
)
LINEAGE_COLUMNS = (
    "dst_history_through_season",
    "dst_history_through_week",
)
TARGET_COLUMN = "fantasy_points"
FEATURE_COLUMNS = (
    *IDENTITY_COLUMNS,
    *LINEAGE_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

FEATURE_SCHEMA = {
    "team": pl.String,
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "dst_history_through_season": pl.Int64,
    "dst_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}


def validate_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted D/ST training-data contract."""
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
        raise SchemaValidationError("dst_feature_frame", problems)
    return frame

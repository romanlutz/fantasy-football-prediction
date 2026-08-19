"""Named feature schema for the individual defensive player (IDP) target."""

from __future__ import annotations

import polars as pl

from ffpred.errors import SchemaValidationError

IDP_STAT_FIELDS = (
    "solo_tackles",
    "assisted_tackles",
    "sacks",
    "interceptions",
    "passes_defended",
    "fumbles_forced",
    "touchdowns",
)

IDP_LAST_ONE_COLUMNS = tuple(f"idp_last_1_{field}" for field in IDP_STAT_FIELDS)
IDP_LAST_TEN_COLUMNS = tuple(f"idp_last_10_{field}" for field in IDP_STAT_FIELDS)
MODEL_FEATURE_COLUMNS = (*IDP_LAST_ONE_COLUMNS, *IDP_LAST_TEN_COLUMNS)
IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "position_group",
    "target_season",
    "target_week",
    "target_game_id",
)
LINEAGE_COLUMNS = (
    "idp_history_through_season",
    "idp_history_through_week",
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
    "position_group": pl.String,
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "idp_history_through_season": pl.Int64,
    "idp_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}


def validate_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted IDP training-data contract."""
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
        raise SchemaValidationError("idp_feature_frame", problems)
    return frame

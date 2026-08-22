"""Named feature schema shared by persistence and training."""

from __future__ import annotations

from enum import StrEnum

import polars as pl

from ffpred.errors import SchemaValidationError

QB_STAT_FIELDS = (
    "passing_attempts",
    "passing_yards",
    "passing_touchdowns",
    "passing_interceptions",
    "passing_two_point_attempts",
    "passing_two_point_made",
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "rushing_two_point_attempts",
    "rushing_two_point_made",
    "fumbles",
)
DEFENSE_STAT_FIELDS = (
    "points_allowed",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "turnovers",
)


class FeatureField(StrEnum):
    """Stable names for model input columns."""

    AGE = "age"
    YEARS_PRO = "years_pro"


QB_LAST_ONE_COLUMNS = tuple(f"qb_last_1_{field}" for field in QB_STAT_FIELDS)
QB_LAST_TEN_COLUMNS = tuple(f"qb_last_10_{field}" for field in QB_STAT_FIELDS)
DEFENSE_LAST_ONE_COLUMNS = tuple(
    f"defense_last_1_{field}" for field in DEFENSE_STAT_FIELDS
)
DEFENSE_LAST_TEN_COLUMNS = tuple(
    f"defense_last_10_{field}" for field in DEFENSE_STAT_FIELDS
)
MODEL_FEATURE_COLUMNS = (
    FeatureField.AGE.value,
    FeatureField.YEARS_PRO.value,
    *QB_LAST_ONE_COLUMNS,
    *QB_LAST_TEN_COLUMNS,
    *DEFENSE_LAST_ONE_COLUMNS,
    *DEFENSE_LAST_TEN_COLUMNS,
)
IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "target_season",
    "target_week",
    "target_game_id",
)
LINEAGE_COLUMNS = (
    "qb_history_through_season",
    "qb_history_through_week",
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
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "qb_history_through_season": pl.Int64,
    "qb_history_through_week": pl.Int64,
    "defense_history_through_season": pl.Int64,
    "defense_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}

FORECAST_IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "target_season",
    "target_week",
    "target_game_id",
    "forecast_as_of",
    "history_through_season",
)
FORECAST_COLUMNS = (
    *FORECAST_IDENTITY_COLUMNS,
    *LINEAGE_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
FORECAST_SCHEMA = {
    "player_id": pl.String,
    "player_name": pl.String,
    "position": pl.String,
    "team": pl.String,
    "opponent": pl.String,
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "forecast_as_of": pl.String,
    "history_through_season": pl.Int64,
    "qb_history_through_season": pl.Int64,
    "qb_history_through_week": pl.Int64,
    "defense_history_through_season": pl.Int64,
    "defense_history_through_week": pl.Int64,
    **dict.fromkeys(MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}


def validate_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted training-data contract."""
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
        raise SchemaValidationError("feature_frame", problems)
    return frame


def validate_forecast_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce the persisted point-in-time forecast contract."""
    problems: list[str] = []
    if tuple(frame.columns) != FORECAST_COLUMNS:
        problems.append("column order does not match FORECAST_COLUMNS")
    for column, expected in FORECAST_SCHEMA.items():
        if column not in frame.schema:
            continue
        if frame.schema[column] != expected:
            problems.append(
                f"{column!r} is {frame.schema[column]}, expected {expected}"
            )
        if column != TARGET_COLUMN and frame[column].null_count():
            problems.append(f"{column!r} contains null values")
    if problems:
        raise SchemaValidationError("forecast_frame", problems)
    return frame

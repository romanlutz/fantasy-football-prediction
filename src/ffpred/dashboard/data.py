"""Data preparation for the prediction dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from pathlib import Path

import polars as pl

from ffpred.errors import FfpredError
from ffpred.features.all_positions import (
    INJURY_MISSED_COLUMN,
    INJURY_STATUS_COLUMN,
)

PREDICTION_COLUMN = "prediction"
ACTUAL_COLUMN = "fantasy_points"
CONSENSUS_MODEL = "Consensus"
REQUIRED_COLUMNS = frozenset(
    {
        "player_id",
        "player_name",
        "target_season",
        "target_week",
        PREDICTION_COLUMN,
    }
)
VIEW_COLUMNS = ("position", "team", "opponent")
SHORT_MODEL_NAME_LENGTH = 4
TIGHT_MODEL_SPREAD = 1.5
MIXED_MODEL_SPREAD = 3.0


class DashboardDataError(FfpredError):
    """Raised when a prediction artifact cannot power the dashboard."""


def model_name_from_path(path: Path) -> str:
    """Derive a concise display name from a prediction artifact path."""
    name = path.stem.removesuffix("-predictions").replace("-", " ").strip()
    return name.upper() if len(name) <= SHORT_MODEL_NAME_LENGTH else name.title()


def prepare_predictions(frame: pl.DataFrame, *, model_name: str) -> pl.DataFrame:
    """Validate and normalize one prediction artifact for dashboard use."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DashboardDataError(
            f"Prediction artifact is missing columns: {sorted(missing)}"
        )
    if frame.is_empty():
        raise DashboardDataError("Prediction artifact contains no rows")

    expressions: list[pl.Expr] = [
        pl.lit(model_name).alias("model"),
        pl.col("player_name").cast(pl.String),
        pl.col("target_season").cast(pl.Int64),
        pl.col("target_week").cast(pl.Int64),
        pl.col(PREDICTION_COLUMN).cast(pl.Float64),
    ]
    if ACTUAL_COLUMN in frame.columns:
        expressions.append(pl.col(ACTUAL_COLUMN).cast(pl.Float64))
    else:
        expressions.append(pl.lit(None, dtype=pl.Float64).alias(ACTUAL_COLUMN))
    if INJURY_MISSED_COLUMN in frame.columns:
        expressions.append(
            pl.col(INJURY_MISSED_COLUMN)
            .cast(pl.Boolean)
            .fill_null(False)
            .alias(INJURY_MISSED_COLUMN)
        )
    else:
        expressions.append(pl.lit(False).alias(INJURY_MISSED_COLUMN))
    if INJURY_STATUS_COLUMN in frame.columns:
        expressions.append(pl.col(INJURY_STATUS_COLUMN).cast(pl.String))
    else:
        expressions.append(pl.lit(None, dtype=pl.String).alias(INJURY_STATUS_COLUMN))

    defaults = {"position": "QB", "team": "N/A", "opponent": "N/A"}
    for column, default in defaults.items():
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String)
                .fill_null(default)
                .str.to_uppercase()
                .alias(column)
            )
        else:
            expressions.append(pl.lit(default).alias(column))

    return frame.with_columns(expressions).with_columns(
        (pl.col(PREDICTION_COLUMN) - pl.col(ACTUAL_COLUMN)).alias("error"),
        (pl.col(PREDICTION_COLUMN) - pl.col(ACTUAL_COLUMN))
        .abs()
        .alias("absolute_error"),
    )


def load_prediction_files(paths: Sequence[Path]) -> pl.DataFrame:
    """Load and combine prediction artifacts from disk."""
    if not paths:
        raise DashboardDataError("Choose at least one prediction artifact")
    frames = [
        prepare_predictions(
            pl.read_parquet(path),
            model_name=model_name_from_path(path),
        )
        for path in paths
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def model_choices(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return available model selections, including consensus when useful."""
    models = tuple(str(model) for model in sorted(frame["model"].unique().to_list()))
    return (CONSENSUS_MODEL, *models) if len(models) > 1 else models


def select_model(frame: pl.DataFrame, model: str) -> pl.DataFrame:
    """Select one model or average matching rows into a consensus."""
    if model != CONSENSUS_MODEL:
        return frame.filter(pl.col("model") == model).with_columns(
            pl.lit(0.0).alias("model_spread"),
            pl.lit(1).alias("model_count"),
        )

    identity = [
        column
        for column in (
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
            INJURY_STATUS_COLUMN,
            INJURY_MISSED_COLUMN,
        )
        if column in frame.columns
    ]
    result = (
        frame.group_by(identity)
        .agg(
            pl.col(PREDICTION_COLUMN).mean(),
            pl.col(PREDICTION_COLUMN).std().fill_null(0.0).alias("model_spread"),
            pl.col("model").n_unique().alias("model_count"),
            pl.col(ACTUAL_COLUMN).drop_nulls().first(),
        )
        .with_columns(
            pl.lit(CONSENSUS_MODEL).alias("model"),
            (pl.col(PREDICTION_COLUMN) - pl.col(ACTUAL_COLUMN)).alias("error"),
            (pl.col(PREDICTION_COLUMN) - pl.col(ACTUAL_COLUMN))
            .abs()
            .alias("absolute_error"),
        )
    )
    sort_columns = [
        column
        for column in ("player_name", "target_season", "target_week")
        if column in result.columns
    ]
    return result.sort(sort_columns)


def draft_board(
    frame: pl.DataFrame,
    *,
    season: int,
    positions: Sequence[str],
    minimum_games: int,
) -> pl.DataFrame:
    """Aggregate weekly predictions into a season-long draft board."""
    selected = frame.filter(
        (pl.col("target_season") == season) & pl.col("position").is_in(list(positions))
    )
    season_has_results = selected[ACTUAL_COLUMN].count() > 0
    board = (
        selected.group_by("player_id", "player_name", "position")
        .agg(
            pl.col(PREDICTION_COLUMN).sum().alias("projected_points"),
            pl.col(PREDICTION_COLUMN).mean().alias("points_per_game"),
            pl.col(PREDICTION_COLUMN).std().fill_null(0.0).alias("volatility"),
            pl.col("model_spread").mean().alias("model_spread"),
            pl.len().alias("projected_games"),
            pl.col(ACTUAL_COLUMN).count().alias("actual_games"),
            pl.col(ACTUAL_COLUMN).sum().alias("actual_points"),
            pl.col(INJURY_MISSED_COLUMN).sum().cast(pl.Int64).alias("injury_games"),
        )
        .filter(pl.col("projected_games") >= minimum_games)
        .with_columns(
            pl.when(
                pl.lit(season_has_results)
                & ((pl.col("actual_games") > 0) | (pl.col("injury_games") > 0))
            )
            .then(pl.col("actual_points").fill_null(0.0))
            .otherwise(None)
            .alias("actual_points"),
            pl.when(pl.lit(season_has_results))
            .then(pl.col("injury_games"))
            .otherwise(None)
            .alias("injury_games"),
            pl.when(pl.lit(season_has_results))
            .then(pl.col("actual_games"))
            .otherwise(None)
            .alias("actual_games"),
            pl.col("projected_points")
            .rank(method="ordinal", descending=True)
            .over("position")
            .cast(pl.Int64)
            .alias("position_rank"),
        )
        .with_columns(
            pl.when(pl.col("actual_games") > 0)
            .then(pl.col("actual_points") / pl.col("actual_games"))
            .otherwise(None)
            .alias("actual_points_per_game"),
            (pl.col("actual_points") - pl.col("projected_points")).alias(
                "projection_delta"
            ),
        )
        .with_columns(
            (pl.col("injury_games") * pl.col("points_per_game")).alias(
                "projected_pace_injury_points"
            ),
            (pl.col("injury_games") * pl.col("actual_points_per_game")).alias(
                "actual_pace_injury_points"
            ),
        )
        .with_columns(
            (pl.col("actual_points") + pl.col("projected_pace_injury_points")).alias(
                "projected_pace_adjusted_actual"
            ),
            (pl.col("actual_points") + pl.col("actual_pace_injury_points")).alias(
                "actual_pace_adjusted_actual"
            ),
        )
        .with_columns(
            (
                pl.col("projected_pace_adjusted_actual") - pl.col("projected_points")
            ).alias("projected_pace_adjusted_delta"),
            (
                (pl.col("projected_pace_adjusted_actual") - pl.col("projected_points"))
                / pl.col("projected_points")
                * 100.0
            ).alias("projected_pace_adjusted_delta_percent"),
            (pl.col("actual_pace_adjusted_actual") - pl.col("projected_points")).alias(
                "actual_pace_adjusted_delta"
            ),
            (
                (pl.col("actual_pace_adjusted_actual") - pl.col("projected_points"))
                / pl.col("projected_points")
                * 100.0
            ).alias("actual_pace_adjusted_delta_percent"),
        )
        .sort("projected_points", descending=True)
    )
    return board


def weekly_board(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
    positions: Sequence[str],
) -> pl.DataFrame:
    """Build a ranked board for one week."""
    return (
        frame.filter(
            (pl.col("target_season") == season)
            & (pl.col("target_week") == week)
            & pl.col("position").is_in(list(positions))
        )
        .with_columns(
            pl.col(PREDICTION_COLUMN)
            .rank(method="ordinal", descending=True)
            .over("position")
            .cast(pl.Int64)
            .alias("position_rank"),
            pl.when(pl.col("model_count") == 1)
            .then(pl.lit("Single model"))
            .when(pl.col("model_spread") <= TIGHT_MODEL_SPREAD)
            .then(pl.lit("Tight"))
            .when(pl.col("model_spread") <= MIXED_MODEL_SPREAD)
            .then(pl.lit("Mixed"))
            .otherwise(pl.lit("Wide"))
            .alias("model_agreement"),
        )
        .sort(PREDICTION_COLUMN, descending=True)
    )


def player_history(
    frame: pl.DataFrame,
    *,
    player_ids: Sequence[str],
    season: int,
) -> pl.DataFrame:
    """Return chronological weekly predictions for selected players."""
    return frame.filter(
        (pl.col("target_season") == season)
        & pl.col("player_id").is_in(list(player_ids))
    ).sort("target_week")


def model_scorecard(frame: pl.DataFrame) -> pl.DataFrame:
    """Calculate comparable error metrics for every model and consensus."""
    selections = list(frame["model"].unique().sort().to_list())
    if len(selections) > 1:
        selections.insert(0, CONSENSUS_MODEL)

    rows: list[dict[str, object]] = []
    for name in selections:
        selected = select_model(frame, name).drop_nulls(ACTUAL_COLUMN)
        if selected.is_empty():
            continue
        squared_error = selected["error"].pow(2).mean()
        assert isinstance(squared_error, float)
        rows.append(
            {
                "model": name,
                "samples": selected.height,
                "mae": selected["absolute_error"].mean(),
                "rmse": sqrt(squared_error),
                "bias": selected["error"].mean(),
            }
        )
    return pl.DataFrame(rows)

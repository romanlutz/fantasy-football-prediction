"""Vectorized, leakage-safe feature table construction for team D/ST."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import polars as pl

from ffpred.domain.identifiers import TeamCode
from ffpred.domain.models import DstHistory
from ffpred.domain.scoring import DEFAULT_DST_SCORING, DstScoringConfig
from ffpred.features.dst_schema import (
    DST_STAT_FIELDS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_frame,
)
from ffpred.features.rolling import rolling_expressions


def _dst_frame(histories: Mapping[TeamCode, DstHistory]) -> pl.DataFrame:
    rows = [
        {
            "team": history.team,
            "target_season": game.key.season,
            "target_week": game.key.week,
            "target_game_id": game.context.game_id,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "team", "target_season", "target_week"
    )


def _score_expression(config: DstScoringConfig) -> pl.Expr:
    tiers = config.points_allowed_tiers
    points_allowed_expr = pl.lit(tiers[-1][1])
    for threshold, points in reversed(tiers[:-1]):
        points_allowed_expr = (
            pl.when(pl.col("points_allowed") <= threshold)
            .then(pl.lit(points))
            .otherwise(points_allowed_expr)
        )
    return (
        pl.col("sacks") * config.sack
        + pl.col("interceptions") * config.interception
        + pl.col("fumble_recoveries") * config.fumble_recovery
        + pl.col("touchdowns") * config.touchdown
        + pl.col("safeties") * config.safety
        + pl.col("blocked_kicks") * config.blocked_kick
        + points_allowed_expr
    ).alias(TARGET_COLUMN)


def build_dst_feature_frame(
    histories: Mapping[TeamCode, DstHistory],
    *,
    scoring: DstScoringConfig = DEFAULT_DST_SCORING,
) -> pl.DataFrame:
    """Build a named D/ST feature table using only games before each target."""
    if not histories:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    frame = _dst_frame(histories).with_columns(
        rolling_expressions(DST_STAT_FIELDS, prefix="dst", group="team")
    )
    frame = (
        frame.filter(pl.col("dst_history_through_season").is_not_null())
        .with_columns(_score_expression(scoring))
        .select(FEATURE_COLUMNS)
        .with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
        .with_columns(pl.col(TARGET_COLUMN).cast(pl.Float64))
    )
    return validate_feature_frame(frame)

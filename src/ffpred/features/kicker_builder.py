"""Vectorized, leakage-safe feature table construction for kickers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import polars as pl

from ffpred.domain.identifiers import PlayerId
from ffpred.domain.models import KickerHistory
from ffpred.domain.scoring import DEFAULT_KICKER_SCORING, KickerScoringConfig
from ffpred.features.kicker_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    KICKER_STAT_FIELDS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_frame,
)
from ffpred.features.rolling import rolling_expressions


def _kicker_frame(histories: Mapping[PlayerId, KickerHistory]) -> pl.DataFrame:
    rows = [
        {
            "player_id": history.player_id,
            "player_name": history.name,
            "target_season": game.key.season,
            "target_week": game.key.week,
            "target_game_id": game.game_id,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "player_id", "target_season", "target_week"
    )


def _score_expression(config: KickerScoringConfig) -> pl.Expr:
    return (
        pl.col("fg_made_0_39") * config.field_goal_0_39
        + pl.col("fg_made_40_49") * config.field_goal_40_49
        + pl.col("fg_made_50_plus") * config.field_goal_50_plus
        + pl.col("fg_missed") * config.field_goal_missed
        + pl.col("pat_made") * config.extra_point_made
        + pl.col("pat_missed") * config.extra_point_missed
    ).alias(TARGET_COLUMN)


def build_kicker_feature_frame(
    histories: Mapping[PlayerId, KickerHistory],
    *,
    scoring: KickerScoringConfig = DEFAULT_KICKER_SCORING,
) -> pl.DataFrame:
    """Build a named kicker feature table using only games before each target.

    A kicker's own debut game has no prior game to roll features from and is
    dropped (unlike the QB pipeline's rookie-cohort fallback); this is a
    deliberate scope simplification for the first kicker release.
    """
    if not histories:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    frame = _kicker_frame(histories).with_columns(
        rolling_expressions(KICKER_STAT_FIELDS, prefix="kicker", group="player_id")
    )
    frame = (
        frame.filter(pl.col("kicker_history_through_season").is_not_null())
        .with_columns(_score_expression(scoring))
        .select(FEATURE_COLUMNS)
        .with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
        .with_columns(pl.col(TARGET_COLUMN).cast(pl.Float64))
    )
    return validate_feature_frame(frame)

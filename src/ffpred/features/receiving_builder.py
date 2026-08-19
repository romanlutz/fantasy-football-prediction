"""Vectorized, leakage-safe feature table construction for RB/WR/TE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import polars as pl

from ffpred.domain.identifiers import PlayerId, TeamCode
from ffpred.domain.models import DefenseHistory, ReceivingHistory
from ffpred.domain.scoring import DEFAULT_RECEIVING_SCORING, ReceivingScoringConfig
from ffpred.features.receiving_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    MODEL_FEATURE_COLUMNS,
    RECEIVING_STAT_FIELDS,
    TARGET_COLUMN,
    validate_feature_frame,
)
from ffpred.features.rolling import rolling_expressions
from ffpred.features.schema import DEFENSE_STAT_FIELDS


def _receiving_frame(histories: Mapping[PlayerId, ReceivingHistory]) -> pl.DataFrame:
    rows = [
        {
            "player_id": history.player_id,
            "player_name": history.name,
            "position": history.position,
            "target_season": game.key.season,
            "target_week": game.key.week,
            "target_game_id": game.context.game_id,
            "opponent": game.context.opponent,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "player_id", "target_season", "target_week"
    )


def _defense_frame(histories: Mapping[TeamCode, DefenseHistory]) -> pl.DataFrame:
    rows = [
        {
            "defense": history.team,
            "target_season": game.key.season,
            "target_week": game.key.week,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "defense", "target_season", "target_week"
    )


def _score_expression(config: ReceivingScoringConfig) -> pl.Expr:
    return (
        pl.col("rushing_yards") / config.rushing_yards_per_point
        + pl.col("rushing_touchdowns") * config.rushing_touchdown
        + pl.col("receiving_yards") / config.receiving_yards_per_point
        + pl.col("receiving_touchdowns") * config.receiving_touchdown
        + pl.col("receptions") * config.reception
        + (pl.col("rushing_two_point_made") + pl.col("receiving_two_point_made"))
        * config.two_point_conversion
        + pl.col("fumbles") * config.fumble
    ).alias(TARGET_COLUMN)


def build_receiving_feature_frame(
    receiving_histories: Mapping[PlayerId, ReceivingHistory],
    defense_histories: Mapping[TeamCode, DefenseHistory],
    *,
    scoring: ReceivingScoringConfig = DEFAULT_RECEIVING_SCORING,
) -> pl.DataFrame:
    """Build a named RB/WR/TE feature table using only games before each target.

    A player's own debut game has no prior game to roll from and is dropped
    (the same deliberate scope simplification the kicker pipeline uses,
    rather than reimplementing QB's rookie-cohort fallback for a third
    position).
    """
    if not receiving_histories:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    receiving = _receiving_frame(receiving_histories).with_columns(
        rolling_expressions(
            RECEIVING_STAT_FIELDS, prefix="receiving", group="player_id"
        )
    )
    defenses = (
        _defense_frame(defense_histories)
        .with_columns(
            rolling_expressions(DEFENSE_STAT_FIELDS, prefix="defense", group="defense")
        )
        .select(
            "defense",
            "target_season",
            "target_week",
            "defense_history_through_season",
            "defense_history_through_week",
            *(f"defense_last_1_{field}" for field in DEFENSE_STAT_FIELDS),
            *(f"defense_last_10_{field}" for field in DEFENSE_STAT_FIELDS),
        )
    )

    frame = (
        receiving.join(
            defenses,
            left_on=["opponent", "target_season", "target_week"],
            right_on=["defense", "target_season", "target_week"],
            how="left",
        )
        .filter(
            pl.col("receiving_history_through_season").is_not_null()
            & pl.col("defense_history_through_season").is_not_null()
        )
        .with_columns(_score_expression(scoring))
        .select(FEATURE_COLUMNS)
        .with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
        .with_columns(pl.col(TARGET_COLUMN).cast(pl.Float64))
    )
    return validate_feature_frame(frame)

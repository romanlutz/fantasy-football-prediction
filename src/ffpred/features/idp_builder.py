"""Vectorized, leakage-safe feature table construction for IDP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import polars as pl

from ffpred.domain.identifiers import PlayerId
from ffpred.domain.models import IdpHistory
from ffpred.domain.scoring import DEFAULT_IDP_SCORING, IdpScoringConfig
from ffpred.features.idp_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    IDP_STAT_FIELDS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_frame,
)
from ffpred.features.rolling import rolling_expressions


def _idp_frame(histories: Mapping[PlayerId, IdpHistory]) -> pl.DataFrame:
    rows = [
        {
            "player_id": history.player_id,
            "player_name": history.name,
            "position_group": history.position_group,
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


def _score_expression(config: IdpScoringConfig) -> pl.Expr:
    return (
        pl.col("solo_tackles") * config.solo_tackle
        + pl.col("assisted_tackles") * config.assisted_tackle
        + pl.col("sacks") * config.sack
        + pl.col("interceptions") * config.interception
        + pl.col("passes_defended") * config.pass_defended
        + pl.col("fumbles_forced") * config.fumble_forced
        + pl.col("touchdowns") * config.touchdown
    ).alias(TARGET_COLUMN)


def build_idp_feature_frame(
    histories: Mapping[PlayerId, IdpHistory],
    *,
    scoring: IdpScoringConfig = DEFAULT_IDP_SCORING,
) -> pl.DataFrame:
    """Build a named IDP feature table using only games before each target.

    A player's own debut game has no prior game to roll from and is dropped
    (the same deliberate scope simplification the kicker/receiving pipelines
    use). No opponent-context feature is computed in this first release.
    """
    if not histories:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    frame = _idp_frame(histories).with_columns(
        rolling_expressions(IDP_STAT_FIELDS, prefix="idp", group="player_id")
    )
    frame = (
        frame.filter(pl.col("idp_history_through_season").is_not_null())
        .with_columns(_score_expression(scoring))
        .select(FEATURE_COLUMNS)
        .with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
        .with_columns(pl.col(TARGET_COLUMN).cast(pl.Float64))
    )
    return validate_feature_frame(frame)

"""Chronological rolling-window helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

import polars as pl

from ffpred.domain.models import GameKey

GameRecord = TypeVar("GameRecord")


def previous_games(
    games: Mapping[GameKey, GameRecord],
    cutoff: GameKey,
    count: int,
) -> list[GameRecord]:
    """Return up to ``count`` games strictly before the cutoff, newest first."""
    if count < 1:
        raise ValueError("count must be positive")
    keys = sorted((key for key in games if key < cutoff), reverse=True)
    return [games[key] for key in keys[:count]]


def rolling_expressions(
    fields: Iterable[str],
    *,
    prefix: str,
    group: str,
) -> list[pl.Expr]:
    """Build leakage-safe last-game and 10-game rolling-mean expressions.

    Every expression is lagged with ``shift(1)`` before any rolling
    aggregation, so a target row's features only ever draw on games strictly
    before it. Two lineage columns record the most recent history period
    used, so callers can assert that invariant downstream.
    """
    expressions: list[pl.Expr] = [
        pl.col("target_season")
        .shift(1)
        .over(group)
        .alias(f"{prefix}_history_through_season"),
        pl.col("target_week")
        .shift(1)
        .over(group)
        .alias(f"{prefix}_history_through_week"),
    ]
    for field in fields:
        expressions.extend(
            (
                pl.col(field).shift(1).over(group).alias(f"{prefix}_last_1_{field}"),
                pl.col(field)
                .rolling_mean(window_size=10, min_samples=1)
                .shift(1)
                .over(group)
                .alias(f"{prefix}_last_10_{field}"),
            )
        )
    return expressions

"""Leakage-safe chronological validation splits."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.errors import ModelTrainingError


def chronological_folds(
    frame: pl.DataFrame,
    folds: int,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Split whole season/week periods so validation always follows training."""
    if folds < 2:
        raise ModelTrainingError("At least two folds are required")
    periods = sorted(set(frame.select("target_season", "target_week").iter_rows()))
    if len(periods) < folds + 1:
        raise ModelTrainingError(
            f"{folds} folds require at least {folds + 1} distinct periods"
        )
    chunks = np.array_split(np.asarray(periods[1:], dtype=np.int64), folds)
    season = frame["target_season"].to_numpy()
    week = frame["target_week"].to_numpy()
    for chunk in chunks:
        first_validation = tuple(chunk[0])
        validation_periods = {tuple(period) for period in chunk}
        train_mask = np.array(
            [
                (row_season, row_week) < first_validation
                for row_season, row_week in zip(season, week, strict=True)
            ]
        )
        validation_mask = np.array(
            [
                (row_season, row_week) in validation_periods
                for row_season, row_week in zip(season, week, strict=True)
            ]
        )
        yield (
            np.flatnonzero(train_mask).astype(np.int64),
            np.flatnonzero(validation_mask).astype(np.int64),
        )

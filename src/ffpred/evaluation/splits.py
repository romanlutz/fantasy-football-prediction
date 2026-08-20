"""Leakage-safe chronological validation splits."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.errors import ModelTrainingError

MINIMUM_FOLDS = 2
MINIMUM_CALIBRATION_PERIODS = 2


def chronological_folds(
    frame: pl.DataFrame,
    folds: int,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Split whole season/week periods so validation always follows training."""
    if folds < MINIMUM_FOLDS:
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


def chronological_calibration_split(
    frame: pl.DataFrame,
    calibration_fraction: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Reserve the latest whole periods for conformal calibration."""
    if not 0 < calibration_fraction < 1:
        raise ModelTrainingError("Calibration fraction must be between zero and one")
    periods = sorted(set(frame.select("target_season", "target_week").iter_rows()))
    if len(periods) < MINIMUM_CALIBRATION_PERIODS:
        raise ModelTrainingError(
            "Conformal calibration requires at least two distinct periods"
        )
    calibration_periods = min(
        len(periods) - 1,
        max(1, int(np.ceil(len(periods) * calibration_fraction))),
    )
    first_calibration = periods[-calibration_periods]
    season = frame["target_season"].to_numpy()
    week = frame["target_week"].to_numpy()
    training_mask = np.array(
        [
            (row_season, row_week) < first_calibration
            for row_season, row_week in zip(season, week, strict=True)
        ]
    )
    calibration_mask = ~training_mask
    return (
        np.flatnonzero(training_mask).astype(np.int64),
        np.flatnonzero(calibration_mask).astype(np.int64),
    )

"""Shared regression metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True, slots=True, kw_only=True)
class RegressionMetrics:
    """Regression metrics reported by every trainer."""

    rmse: float
    mae: float
    mre: float
    samples: int


def _paired_arrays(
    actual: ArrayLike,
    prediction: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual_array.shape != prediction_array.shape:
        raise ValueError("actual and prediction must have the same shape")
    if actual_array.size == 0:
        raise ValueError("metrics require at least one sample")
    return actual_array, prediction_array


def mean_relative_error(actual: ArrayLike, prediction: ArrayLike) -> float:
    """Return absolute relative error, excluding undefined zero targets."""
    actual_array, prediction_array = _paired_arrays(actual, prediction)
    nonzero = actual_array != 0
    if not np.any(nonzero):
        raise ValueError("Mean relative error is undefined when all targets are zero")
    relative = np.abs(prediction_array[nonzero] - actual_array[nonzero]) / np.abs(
        actual_array[nonzero]
    )
    return float(np.mean(relative))


def evaluate(actual: ArrayLike, prediction: ArrayLike) -> RegressionMetrics:
    """Calculate the canonical evaluation metric set."""
    actual_array, prediction_array = _paired_arrays(actual, prediction)
    return RegressionMetrics(
        rmse=float(mean_squared_error(actual_array, prediction_array) ** 0.5),
        mae=float(mean_absolute_error(actual_array, prediction_array)),
        mre=mean_relative_error(actual_array, prediction_array),
        samples=actual_array.size,
    )

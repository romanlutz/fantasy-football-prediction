"""Common training result model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ffpred.evaluation.metrics import RegressionMetrics
from ffpred.training.protocol import Regressor


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingResult:
    """Fitted estimator, predictions, and evaluation details."""

    estimator: Regressor
    predictions: NDArray[np.float64]
    metrics: RegressionMetrics
    feature_names: tuple[str, ...]

"""Model inference over target-free point-in-time forecast rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.datasets.io import read_forecast
from ffpred.features.schema import MODEL_FEATURE_COLUMNS
from ffpred.training.data import TrainingData
from ffpred.training.protocol import Regressor


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionData:
    """Forecast frame and typed model inputs."""

    frame: pl.DataFrame
    features: NDArray[np.float64]


def load_projection_data(path: Path) -> ProjectionData:
    """Load a forecast artifact for model inference."""
    frame = read_forecast(path)
    features = np.asarray(
        frame.select(MODEL_FEATURE_COLUMNS).to_numpy(),
        dtype=np.float64,
    )
    return ProjectionData(frame=frame, features=features)


def project(
    estimator: Regressor,
    train: TrainingData,
    forecast: ProjectionData,
) -> NDArray[np.float64]:
    """Fit on completed games and predict frozen forecast rows."""
    return np.asarray(
        estimator.fit(train.features, train.target).predict(forecast.features),
        dtype=np.float64,
    )

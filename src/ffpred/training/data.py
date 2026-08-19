"""Named dataset conversion at the scikit-learn boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.datasets.io import read_dataset
from ffpred.features.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingData:
    """Typed arrays plus their source frame and feature names."""

    frame: pl.DataFrame
    features: NDArray[np.float64]
    target: NDArray[np.float64]
    feature_names: tuple[str, ...]


def training_data_from_frame(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> TrainingData:
    """Convert named, schema-validated columns to typed NumPy arrays."""
    features = np.asarray(
        frame.select(feature_names).to_numpy(),
        dtype=np.float64,
    )
    target = np.asarray(frame[TARGET_COLUMN].to_numpy(), dtype=np.float64)
    return TrainingData(
        frame=frame,
        features=features,
        target=target,
        feature_names=feature_names,
    )


def load_training_data(
    path: Path,
    feature_names: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> TrainingData:
    """Load one Parquet split for model training or evaluation."""
    return training_data_from_frame(read_dataset(path), feature_names)

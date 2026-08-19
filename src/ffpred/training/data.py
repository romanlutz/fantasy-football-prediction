"""Named dataset conversion at the scikit-learn boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.datasets.io import Validator, read_dataset
from ffpred.features.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from ffpred.features.schema import validate_feature_frame as validate_qb_frame


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
    *,
    validator: Validator = validate_qb_frame,
) -> TrainingData:
    """Load one Parquet split for model training or evaluation."""
    frame = read_dataset(path, validator=validator)
    return training_data_from_frame(frame, feature_names)

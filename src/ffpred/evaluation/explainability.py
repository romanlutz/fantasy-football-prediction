"""Model-agnostic uncertainty and explainability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import shap
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error


class Predictor(Protocol):
    """Minimum prediction surface required by explanation diagnostics."""

    def predict(
        self,
        features: NDArray[np.float64],
        /,
    ) -> NDArray[np.float64]: ...


MINIMUM_BINS = 2
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class ConformalPredictionInterval:
    """Finite-sample split-conformal prediction interval."""

    coverage: float
    radius: float
    calibration_samples: int
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class AleCurve:
    """One-dimensional accumulated local effects curve."""

    feature: str
    bin_edges: tuple[float, ...]
    bin_centers: tuple[float, ...]
    effects: tuple[float, ...]
    samples: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PermutationImportance:
    """Held-out error increase after within-period feature permutation."""

    feature: str
    mean_mae_increase: float
    std_mae_increase: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapValues:
    """Portable model-agnostic SHAP values."""

    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    base_values: NDArray[np.float64]
    data: NDArray[np.float64]
    sample_indices: tuple[int, ...]


def conformal_prediction_interval(
    calibration_actual: ArrayLike,
    calibration_prediction: ArrayLike,
    prediction: ArrayLike,
    *,
    coverage: float = 0.9,
) -> ConformalPredictionInterval:
    """Calibrate a symmetric interval from held-out absolute residuals."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    actual = _one_dimensional(calibration_actual, "calibration_actual")
    calibrated = _one_dimensional(
        calibration_prediction,
        "calibration_prediction",
    )
    predicted = _one_dimensional(prediction, "prediction")
    if actual.shape != calibrated.shape:
        raise ValueError("calibration actual and prediction must have the same shape")
    if actual.size == 0:
        raise ValueError("conformal calibration requires at least one sample")

    residuals = np.abs(actual - calibrated)
    quantile = min(1.0, np.ceil((actual.size + 1) * coverage) / actual.size)
    radius = float(np.quantile(residuals, quantile, method="higher"))
    return ConformalPredictionInterval(
        coverage=coverage,
        radius=radius,
        calibration_samples=actual.size,
        lower=predicted - radius,
        upper=predicted + radius,
    )


def accumulated_local_effects(
    estimator: Predictor,
    features: NDArray[np.float64],
    feature_index: int,
    feature_name: str,
    *,
    bins: int = 10,
) -> AleCurve:
    """Estimate a centered first-order ALE curve for one numeric feature."""
    if bins < MINIMUM_BINS:
        raise ValueError("ALE requires at least two bins")
    if features.ndim != MATRIX_DIMENSIONS or not 0 <= feature_index < features.shape[1]:
        raise ValueError("feature_index must identify a feature column")
    values = features[:, feature_index]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if edges.size == 1:
        value = float(edges[0])
        return AleCurve(
            feature=feature_name,
            bin_edges=(value, value),
            bin_centers=(value,),
            effects=(0.0,),
            samples=(features.shape[0],),
        )

    assignments = np.searchsorted(edges[1:-1], values, side="right")
    differences = np.zeros(edges.size - 1, dtype=np.float64)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    for index in range(edges.size - 1):
        rows = np.flatnonzero(assignments == index)
        counts[index] = rows.size
        if rows.size == 0:
            continue
        lower = features[rows].copy()
        upper = features[rows].copy()
        lower[:, feature_index] = edges[index]
        upper[:, feature_index] = edges[index + 1]
        differences[index] = float(
            np.mean(estimator.predict(upper) - estimator.predict(lower))
        )

    effects = np.cumsum(differences)
    populated = counts > 0
    if np.any(populated):
        effects -= np.average(effects[populated], weights=counts[populated])
    centers = (edges[:-1] + edges[1:]) / 2
    return AleCurve(
        feature=feature_name,
        bin_edges=tuple(float(value) for value in edges),
        bin_centers=tuple(float(value) for value in centers),
        effects=tuple(float(value) for value in effects),
        samples=tuple(int(value) for value in counts),
    )


def temporal_permutation_importance(  # noqa: PLR0913
    estimator: Predictor,
    features: NDArray[np.float64],
    target: ArrayLike,
    periods: ArrayLike,
    feature_names: Sequence[str],
    *,
    repeats: int = 5,
    random_state: int = 42,
) -> tuple[PermutationImportance, ...]:
    """Permute within periods to preserve season-level feature distributions."""
    if repeats < 1:
        raise ValueError("permutation importance requires at least one repeat")
    actual = _one_dimensional(target, "target")
    period_values = np.asarray(periods).reshape(-1)
    if features.ndim != MATRIX_DIMENSIONS or features.shape[0] != actual.size:
        raise ValueError("features and target must have the same number of rows")
    if period_values.size != actual.size:
        raise ValueError("periods and target must have the same number of rows")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the feature columns")

    baseline = float(mean_absolute_error(actual, estimator.predict(features)))
    groups = tuple(
        np.flatnonzero(period_values == period) for period in np.unique(period_values)
    )
    random = np.random.default_rng(random_state)
    results: list[PermutationImportance] = []
    for feature_index, feature_name in enumerate(feature_names):
        increases: list[float] = []
        for _ in range(repeats):
            permuted = features.copy()
            for rows in groups:
                permuted[rows, feature_index] = random.permutation(
                    permuted[rows, feature_index]
                )
            error = float(mean_absolute_error(actual, estimator.predict(permuted)))
            increases.append(error - baseline)
        results.append(
            PermutationImportance(
                feature=feature_name,
                mean_mae_increase=float(np.mean(increases)),
                std_mae_increase=float(np.std(increases)),
            )
        )
    return tuple(
        sorted(results, key=lambda result: result.mean_mae_increase, reverse=True)
    )


def model_agnostic_shap_values(  # noqa: PLR0913
    estimator: Predictor,
    background: NDArray[np.float64],
    samples: NDArray[np.float64],
    feature_names: Sequence[str],
    *,
    max_background: int = 100,
    max_samples: int = 100,
    random_state: int = 42,
) -> ShapValues:
    """Calculate permutation SHAP values for any project regressor."""
    if max_background < 1 or max_samples < 1:
        raise ValueError("SHAP sample limits must be positive")
    if background.ndim != MATRIX_DIMENSIONS or samples.ndim != MATRIX_DIMENSIONS:
        raise ValueError("SHAP inputs must be two-dimensional")
    if background.shape[1] != samples.shape[1] or samples.shape[1] != len(
        feature_names
    ):
        raise ValueError("SHAP inputs and feature_names must have matching columns")

    random = np.random.default_rng(random_state)
    background_indices = _sample_row_indices(
        background.shape[0],
        max_background,
        random,
    )
    sample_indices = _sample_row_indices(samples.shape[0], max_samples, random)
    background_rows = background[background_indices]
    sample_rows = samples[sample_indices]
    explainer = shap.Explainer(
        estimator.predict,
        background_rows,
        algorithm="permutation",
        feature_names=list(feature_names),
        seed=random_state,
    )
    explanation = explainer(
        sample_rows,
        max_evals=2 * len(feature_names) + 1,
        silent=True,
    )
    if isinstance(explanation, list):
        raise TypeError("SHAP returned multiple outputs for a regression model")
    return ShapValues(
        feature_names=tuple(feature_names),
        values=np.asarray(explanation.values, dtype=np.float64),
        base_values=np.asarray(explanation.base_values, dtype=np.float64).reshape(-1),
        data=np.asarray(explanation.data, dtype=np.float64),
        sample_indices=tuple(int(index) for index in sample_indices),
    )


def _one_dimensional(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _sample_row_indices(
    rows: int,
    limit: int,
    random: np.random.Generator,
) -> NDArray[np.int64]:
    if rows <= limit:
        return np.arange(rows, dtype=np.int64)
    return np.sort(random.choice(rows, size=limit, replace=False)).astype(np.int64)

"""Support-vector regression training and model selection."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

from ffpred.evaluation.metrics import evaluate
from ffpred.evaluation.splits import chronological_folds
from ffpred.features.schema import MODEL_FEATURE_COLUMNS
from ffpred.training.data import TrainingData, training_data_from_frame
from ffpred.training.result import TrainingResult

Kernel = Literal["linear", "poly", "rbf", "sigmoid"]

MANUAL_FEATURE_COLUMNS = tuple(
    MODEL_FEATURE_COLUMNS[index]
    for index in (
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
        17,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SvrConfig:
    """Serializable SVR hyperparameters."""

    c: float = 0.25
    epsilon: float = 0.25
    kernel: Kernel = "linear"
    gamma: float | Literal["scale", "auto"] = "scale"
    degree: int = 3


DEFAULT_SVR_CONFIG = SvrConfig()


def create_estimator(config: SvrConfig) -> Pipeline:
    """Create a leakage-safe scaling and regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                SVR(
                    C=config.c,
                    epsilon=config.epsilon,
                    kernel=config.kernel,
                    gamma=config.gamma,
                    degree=config.degree,
                ),
            ),
        ]
    )


def create_scalable_estimator(config: SvrConfig = DEFAULT_SVR_CONFIG) -> Pipeline:
    """Create a linear SVR suitable for the larger all-position archive."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                LinearSVR(
                    C=config.c,
                    epsilon=config.epsilon,
                    dual="auto",
                    max_iter=20_000,
                    random_state=42,
                ),
            ),
        ]
    )


def candidate_configs() -> tuple[SvrConfig, ...]:
    """Return the historical search space as immutable configuration data."""
    configs: list[SvrConfig] = []
    for c_value, epsilon, kernel in itertools.product(
        (0.25, 0.5, 0.75, 1.0),
        (0.05, 0.1, 0.15, 0.2, 0.25),
        ("rbf", "linear", "sigmoid", "poly"),
    ):
        if kernel == "poly":
            configs.extend(
                SvrConfig(
                    c=c_value,
                    epsilon=epsilon,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                )
                for gamma in (0.05, 0.1, 0.15)
                for degree in (2, 3)
            )
        elif kernel in {"rbf", "sigmoid"}:
            configs.extend(
                SvrConfig(
                    c=c_value,
                    epsilon=epsilon,
                    kernel=kernel,
                    gamma=gamma,
                )
                for gamma in (0.05, 0.1, 0.15)
            )
        else:
            configs.append(SvrConfig(c=c_value, epsilon=epsilon, kernel=kernel))
    return tuple(configs)


def select_config(
    data: TrainingData,
    configs: Iterable[SvrConfig],
    *,
    folds: int = 5,
) -> SvrConfig:
    """Choose hyperparameters using strictly chronological validation."""
    candidates = tuple(configs)
    if not candidates:
        raise ValueError("At least one SVR configuration is required")
    scores: list[float] = []
    splits = tuple(chronological_folds(data.frame, folds))
    for config in candidates:
        errors: list[float] = []
        for train_indices, validation_indices in splits:
            estimator = create_estimator(config)
            prediction = estimator.fit(
                data.features[train_indices],
                data.target[train_indices],
            ).predict(data.features[validation_indices])
            errors.append(
                float(
                    mean_absolute_error(
                        data.target[validation_indices],
                        prediction,
                    )
                )
            )
        scores.append(float(np.mean(errors)))
    return candidates[int(np.argmin(scores))]


def train_svr(
    train: TrainingData,
    test: TrainingData,
    *,
    config: SvrConfig = DEFAULT_SVR_CONFIG,
) -> TrainingResult:
    """Fit and evaluate an SVR pipeline."""
    estimator = create_estimator(config)
    prediction = np.asarray(
        estimator.fit(train.features, train.target).predict(test.features),
        dtype=np.float64,
    )
    return TrainingResult(
        estimator=estimator,
        predictions=prediction,
        metrics=evaluate(test.target, prediction),
        feature_names=train.feature_names,
    )


def select_manual_features(data: TrainingData) -> TrainingData:
    """Select the historical hand-picked columns by stable names."""
    return training_data_from_frame(data.frame, MANUAL_FEATURE_COLUMNS)

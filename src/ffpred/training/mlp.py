"""Deterministic multilayer-perceptron regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ffpred.evaluation.metrics import evaluate
from ffpred.training.data import TrainingData
from ffpred.training.result import TrainingResult

Activation = Literal["identity", "logistic", "tanh", "relu"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MlpConfig:
    """Serializable MLP hyperparameters."""

    hidden_units: int = 50
    activation: Activation = "relu"
    max_iterations: int = 1000
    learning_rate: float = 0.001
    random_state: int = 42


DEFAULT_MLP_CONFIG = MlpConfig()


def create_estimator(config: MlpConfig) -> Pipeline:
    """Create a deterministic scaled MLP pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(config.hidden_units,),
                    activation=config.activation,
                    max_iter=config.max_iterations,
                    learning_rate_init=config.learning_rate,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def create_archive_estimator(config: MlpConfig = DEFAULT_MLP_CONFIG) -> Pipeline:
    """Create an early-stopping MLP for the larger all-position archive."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(config.hidden_units,),
                    activation=config.activation,
                    max_iter=min(config.max_iterations, 400),
                    learning_rate_init=config.learning_rate,
                    random_state=config.random_state,
                    early_stopping=True,
                    n_iter_no_change=15,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def train_mlp(
    train: TrainingData,
    test: TrainingData,
    *,
    config: MlpConfig = DEFAULT_MLP_CONFIG,
) -> TrainingResult:
    """Fit and evaluate a scaled deterministic MLP."""
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

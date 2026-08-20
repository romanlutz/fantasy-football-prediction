"""Explainable Boosting Machine training and explanation export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from ffpred.evaluation.explainability import (
    ConformalPredictionInterval,
    conformal_prediction_interval,
)
from ffpred.evaluation.metrics import evaluate
from ffpred.evaluation.splits import chronological_calibration_split
from ffpred.training.data import TrainingData
from ffpred.training.result import TrainingResult


@dataclass(frozen=True, slots=True, kw_only=True)
class EbmConfig:
    """Serializable EBM hyperparameters."""

    max_bins: int = 256
    interactions: int = 10
    max_rounds: int = 5_000
    learning_rate: float = 0.04
    min_samples_leaf: int = 4
    outer_bags: int = 8
    validation_size: float = 0.15
    calibration_fraction: float = 0.2
    interval_coverage: float = 0.9
    random_state: int = 42
    n_jobs: int = -2


@dataclass(frozen=True, slots=True, kw_only=True)
class GlobalTermExplanation:
    """Importance and learned response shape for one EBM term."""

    name: str
    importance: float
    shape: dict[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTermExplanation:
    """One term's additive contribution to a prediction."""

    name: str
    value: object
    contribution: float


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalExplanation:
    """Additive decomposition for one prediction row."""

    row: int
    actual: float
    prediction: float
    intercept: float
    terms: tuple[LocalTermExplanation, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EbmExplanations:
    """Structured EBM explanations suitable for JSON or later visualization."""

    feature_names: tuple[str, ...]
    global_terms: tuple[GlobalTermExplanation, ...]
    local: tuple[LocalExplanation, ...]

    def to_dict(
        self,
        identities: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Return a versioned, JSON-compatible explanation artifact."""
        if len(identities) != len(self.local):
            raise ValueError("Identity rows must match local explanation rows")
        return {
            "schema_version": 1,
            "model": "ExplainableBoostingRegressor",
            "feature_names": list(self.feature_names),
            "global": {
                "terms": [
                    {
                        "name": term.name,
                        "importance": term.importance,
                        "shape": term.shape,
                    }
                    for term in self.global_terms
                ]
            },
            "local": [
                {
                    "row": explanation.row,
                    "identity": _json_value(identities[explanation.row]),
                    "actual": explanation.actual,
                    "prediction": explanation.prediction,
                    "intercept": explanation.intercept,
                    "terms": [
                        {
                            "name": term.name,
                            "value": _json_value(term.value),
                            "contribution": term.contribution,
                        }
                        for term in explanation.terms
                    ],
                }
                for explanation in self.local
            ],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EbmTrainingResult(TrainingResult):
    """Training result augmented with native global and local explanations."""

    estimator: ExplainableBoostingRegressor
    explanations: EbmExplanations
    prediction_interval: ConformalPredictionInterval | None


DEFAULT_EBM_CONFIG = EbmConfig()


def create_estimator(
    config: EbmConfig,
    feature_names: tuple[str, ...] | None = None,
) -> ExplainableBoostingRegressor:
    """Create a deterministic Explainable Boosting Machine regressor."""
    return ExplainableBoostingRegressor(
        feature_names=list(feature_names) if feature_names is not None else None,
        max_bins=config.max_bins,
        interactions=config.interactions,
        max_rounds=config.max_rounds,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf,
        outer_bags=config.outer_bags,
        validation_size=config.validation_size,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_ebm(
    train: TrainingData,
    test: TrainingData,
    *,
    config: EbmConfig = DEFAULT_EBM_CONFIG,
) -> EbmTrainingResult:
    """Fit and evaluate an EBM with native global and local explanations."""
    estimator = create_estimator(config, train.feature_names)
    fit_features = train.features
    fit_target = train.target
    calibration_indices: np.ndarray | None = None
    if config.calibration_fraction:
        fit_indices, calibration_indices = chronological_calibration_split(
            train.frame,
            config.calibration_fraction,
        )
        fit_features = train.features[fit_indices]
        fit_target = train.target[fit_indices]
    estimator.fit(fit_features, fit_target)
    predictions = np.asarray(
        estimator.predict(test.features),
        dtype=np.float64,
    )
    interval = None
    if calibration_indices is not None:
        calibration_prediction = estimator.predict(train.features[calibration_indices])
        interval = conformal_prediction_interval(
            train.target[calibration_indices],
            calibration_prediction,
            predictions,
            coverage=config.interval_coverage,
        )
    explanations = _build_explanations(estimator, test, predictions)
    return EbmTrainingResult(
        estimator=estimator,
        predictions=predictions,
        metrics=evaluate(test.target, predictions),
        feature_names=train.feature_names,
        explanations=explanations,
        prediction_interval=interval,
    )


def write_ebm_explanations(
    path: Path,
    explanations: EbmExplanations,
    *,
    identities: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, object] | None = None,
) -> None:
    """Atomically write EBM explanations as portable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    artifact = explanations.to_dict(identities)
    if diagnostics is not None:
        artifact["diagnostics"] = _json_value(diagnostics)
    temporary.write_text(
        json.dumps(
            artifact,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_explanations(
    estimator: ExplainableBoostingRegressor,
    test: TrainingData,
    predictions: np.ndarray,
) -> EbmExplanations:
    global_explanation = estimator.explain_global()
    importances = np.asarray(estimator.term_importances(), dtype=np.float64)
    global_terms = tuple(
        GlobalTermExplanation(
            name=name,
            importance=float(importance),
            shape=_json_mapping(global_explanation.data(index)),
        )
        for index, (name, importance) in enumerate(
            zip(estimator.term_names_, importances, strict=True)
        )
    )

    local_explanation = estimator.explain_local(test.features, test.target)
    local_rows: list[LocalExplanation] = []
    for index, (actual, prediction) in enumerate(
        zip(test.target, predictions, strict=True)
    ):
        data = _mapping(local_explanation.data(index))
        names = _sequence(data["names"])
        values = _sequence(data["values"])
        scores = _sequence(data["scores"])
        terms = tuple(
            LocalTermExplanation(
                name=str(name),
                value=_local_term_value(
                    str(name),
                    value,
                    test,
                    index,
                ),
                contribution=_float_value(score),
            )
            for name, value, score in zip(names, values, scores, strict=True)
        )
        local_rows.append(
            LocalExplanation(
                row=index,
                actual=float(actual),
                prediction=float(prediction),
                intercept=float(estimator.intercept_),
                terms=terms,
            )
        )
    return EbmExplanations(
        feature_names=test.feature_names,
        global_terms=global_terms,
        local=tuple(local_rows),
    )


def _local_term_value(
    name: str,
    value: object,
    test: TrainingData,
    row: int,
) -> object:
    feature_indices = {
        feature: index for index, feature in enumerate(test.feature_names)
    }
    interaction_features = name.split(" & ")
    if len(interaction_features) == 1:
        return _json_value(value)
    if not all(feature in feature_indices for feature in interaction_features):
        raise ValueError(f"Unknown EBM interaction term {name!r}")
    return {
        feature: float(test.features[row, feature_indices[feature]])
        for feature in interaction_features
    }


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("InterpretML explanation data must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("InterpretML explanation values must be a sequence")
    return value


def _json_mapping(value: object) -> dict[str, object]:
    converted = _json_value(_mapping(value))
    if not isinstance(converted, dict):
        raise TypeError("InterpretML explanation data must be a mapping")
    return {str(key): item for key, item in converted.items()}


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("InterpretML explanation scores must be numeric")
    return float(value)


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return value

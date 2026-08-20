"""Typed command option records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ffpred.training.mlp import Activation


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplainabilityOptions:
    """Shared model-diagnostic output controls."""

    path: Path | None
    ale_bins: int
    permutation_repeats: int
    shap_background: int
    shap_samples: int
    random_state: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildOptions:
    """Dataset build command options."""

    output_dir: Path
    history_start: int
    train_start: int
    test_year: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceivingBuildOptions:
    """RB/WR/TE dataset build command options."""

    output_dir: Path
    history_start: int
    train_start: int
    test_year: int
    positions: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SvrOptions:
    """SVR command options."""

    train_path: Path
    test_path: Path
    predictions_path: Path
    position: str = "qb"
    explainability: ExplainabilityOptions
    manual_features: bool
    select_hyperparameters: bool
    folds: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MlpOptions:
    """MLP command options."""

    train_path: Path
    test_path: Path
    predictions_path: Path
    position: str = "qb"
    explainability: ExplainabilityOptions
    hidden_units: int
    activation: Activation
    iterations: int
    learning_rate: float
    random_state: int


@dataclass(frozen=True, slots=True, kw_only=True)
class EbmOptions:
    """EBM command options."""

    train_path: Path
    test_path: Path
    predictions_path: Path
    position: str = "qb"
    explainability: ExplainabilityOptions
    max_bins: int
    interactions: int
    max_rounds: int
    learning_rate: float
    min_samples_leaf: int
    outer_bags: int
    validation_size: float
    calibration_fraction: float
    interval_coverage: float
    random_state: int
    n_jobs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluateOptions:
    """Prediction evaluation command options."""

    predictions_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class InjuryReportOptions:
    """Injury-impact report command options."""

    output_path: Path
    start_season: int
    end_season: int
    positions: tuple[str, ...]
    trailing_window: int

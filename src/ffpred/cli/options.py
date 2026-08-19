"""Typed command option records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ffpred.training.mlp import Activation


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildOptions:
    """Dataset build command options."""

    output_dir: Path
    history_start: int
    train_start: int
    test_year: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SvrOptions:
    """SVR command options."""

    train_path: Path
    test_path: Path
    predictions_path: Path
    manual_features: bool
    select_hyperparameters: bool
    folds: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MlpOptions:
    """MLP command options."""

    train_path: Path
    test_path: Path
    predictions_path: Path
    hidden_units: int
    activation: Activation
    iterations: int
    learning_rate: float
    random_state: int


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluateOptions:
    """Prediction evaluation command options."""

    predictions_path: Path

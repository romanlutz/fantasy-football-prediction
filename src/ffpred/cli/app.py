"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Never

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.cli.options import (
    BuildOptions,
    EbmOptions,
    EvaluateOptions,
    ExplainabilityOptions,
    MlpOptions,
    ReceivingBuildOptions,
    SvrOptions,
)
from ffpred.config import Settings
from ffpred.datasets.builder import (
    DatasetBuildConfig,
    DstDatasetBuildConfig,
    IdpDatasetBuildConfig,
    KickerDatasetBuildConfig,
    ReceivingDatasetBuildConfig,
    build_datasets,
    build_dst_datasets,
    build_idp_datasets,
    build_kicker_datasets,
    build_receiving_datasets,
)
from ffpred.errors import FfpredError
from ffpred.evaluation.cohorts import residual_cohorts
from ffpred.evaluation.explainability import (
    ConformalPredictionInterval,
    accumulated_local_effects,
    model_agnostic_shap_values,
    temporal_permutation_importance,
)
from ffpred.evaluation.metrics import evaluate
from ffpred.features import dst_schema, idp_schema, kicker_schema, receiving_schema
from ffpred.features import schema as qb_schema
from ffpred.features.schema import TARGET_COLUMN
from ffpred.logging import configure_logging
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.training.data import TrainingData, load_training_data
from ffpred.training.ebm import (
    EbmConfig,
    train_ebm,
    write_ebm_explanations,
)
from ffpred.training.mlp import MlpConfig, train_mlp
from ffpred.training.result import TrainingResult
from ffpred.training.svr import (
    candidate_configs,
    select_config,
    select_manual_features,
    train_svr,
)

LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMN = "prediction"
#: Positions supported by the generic training and evaluation commands.
#: Adding a position here only requires a feature-schema module exposing
#: MODEL_FEATURE_COLUMNS, IDENTITY_COLUMNS, and validate_feature_frame;
#: TARGET_COLUMN is shared. RB/WR/TE share one feature schema (they differ
#: only in which players' rows were acquired at build time), so all three
#: map to the same receiving_schema module.
POSITION_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "qb": qb_schema.MODEL_FEATURE_COLUMNS,
    "dst": dst_schema.MODEL_FEATURE_COLUMNS,
    "k": kicker_schema.MODEL_FEATURE_COLUMNS,
    "rb": receiving_schema.MODEL_FEATURE_COLUMNS,
    "wr": receiving_schema.MODEL_FEATURE_COLUMNS,
    "te": receiving_schema.MODEL_FEATURE_COLUMNS,
    "idp": idp_schema.MODEL_FEATURE_COLUMNS,
}
POSITION_IDENTITY_COLUMNS: dict[str, tuple[str, ...]] = {
    "qb": qb_schema.IDENTITY_COLUMNS,
    "dst": dst_schema.IDENTITY_COLUMNS,
    "k": kicker_schema.IDENTITY_COLUMNS,
    "rb": receiving_schema.IDENTITY_COLUMNS,
    "wr": receiving_schema.IDENTITY_COLUMNS,
    "te": receiving_schema.IDENTITY_COLUMNS,
    "idp": idp_schema.IDENTITY_COLUMNS,
}
POSITION_VALIDATORS = {
    "qb": qb_schema.validate_feature_frame,
    "dst": dst_schema.validate_feature_frame,
    "k": kicker_schema.validate_feature_frame,
    "rb": receiving_schema.validate_feature_frame,
    "wr": receiving_schema.validate_feature_frame,
    "te": receiving_schema.validate_feature_frame,
    "idp": idp_schema.validate_feature_frame,
}
#: Maps the --position value on build-receiving-dataset to the acquired
#: nflverse position codes.
RECEIVING_BUILD_POSITIONS: dict[str, tuple[str, ...]] = {
    "rb": ("RB",),
    "wr": ("WR",),
    "te": ("TE",),
    "all": ("RB", "WR", "TE"),
}


def _parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffpred",
        description="Build and evaluate fantasy-football prediction models",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="build QB train/test datasets")
    _add_build_arguments(build, settings)

    build_dst = subparsers.add_parser(
        "build-dst-dataset", help="build team D/ST train/test datasets"
    )
    _add_build_arguments(build_dst, settings)

    build_kicker = subparsers.add_parser(
        "build-kicker-dataset", help="build kicker train/test datasets"
    )
    _add_build_arguments(build_kicker, settings)

    build_receiving = subparsers.add_parser(
        "build-receiving-dataset", help="build RB/WR/TE train/test datasets"
    )
    _add_build_arguments(build_receiving, settings)
    build_receiving.add_argument(
        "--position", choices=tuple(RECEIVING_BUILD_POSITIONS), default="all"
    )

    build_idp = subparsers.add_parser(
        "build-idp-dataset", help="build IDP train/test datasets"
    )
    _add_build_arguments(
        build_idp, settings, history_start=2010, train_start=2011, test_year=2014
    )

    svr = subparsers.add_parser("train-svr", help="train an SVR model")
    _add_dataset_arguments(svr, "svr-predictions.parquet")
    svr.add_argument(
        "--position", choices=tuple(POSITION_FEATURE_COLUMNS), default="qb"
    )
    svr.add_argument("--manual-features", action="store_true")
    svr.add_argument("--select-hyperparameters", action="store_true")
    svr.add_argument("--folds", type=int, default=5)
    svr.add_argument("--random-state", type=int, default=42)
    _add_explainability_arguments(svr)

    mlp = subparsers.add_parser("train-mlp", help="train an MLP model")
    _add_dataset_arguments(mlp, "mlp-predictions.parquet")
    mlp.add_argument(
        "--position", choices=tuple(POSITION_FEATURE_COLUMNS), default="qb"
    )
    mlp.add_argument("--hidden-units", type=int, default=50)
    mlp.add_argument(
        "--activation",
        choices=("identity", "logistic", "tanh", "relu"),
        default="relu",
    )
    mlp.add_argument("--iterations", type=int, default=1000)
    mlp.add_argument("--learning-rate", type=float, default=0.001)
    mlp.add_argument("--random-state", type=int, default=42)
    _add_explainability_arguments(mlp)

    ebm = subparsers.add_parser(
        "train-ebm",
        help="train an Explainable Boosting Machine",
    )
    _add_dataset_arguments(ebm, "ebm-predictions.parquet")
    ebm.add_argument(
        "--position", choices=tuple(POSITION_FEATURE_COLUMNS), default="qb"
    )
    ebm.add_argument("--max-bins", type=int, default=256)
    ebm.add_argument("--interactions", type=int, default=10)
    ebm.add_argument("--max-rounds", type=int, default=5_000)
    ebm.add_argument("--learning-rate", type=float, default=0.04)
    ebm.add_argument("--min-samples-leaf", type=int, default=4)
    ebm.add_argument("--outer-bags", type=int, default=8)
    ebm.add_argument("--validation-size", type=float, default=0.15)
    ebm.add_argument("--calibration-fraction", type=float, default=0.2)
    ebm.add_argument("--interval-coverage", type=float, default=0.9)
    ebm.add_argument("--random-state", type=int, default=42)
    ebm.add_argument("--jobs", type=int, default=-2)
    _add_explainability_arguments(ebm, default=Path("ebm-explanations.json"))

    evaluation = subparsers.add_parser(
        "evaluate",
        help="evaluate a prediction Parquet artifact",
    )
    evaluation.add_argument("predictions", type=Path)
    return parser


def _add_build_arguments(
    parser: argparse.ArgumentParser,
    settings: Settings,
    *,
    history_start: int | None = None,
    train_start: int | None = None,
    test_year: int | None = None,
) -> None:
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument(
        "--history-start",
        type=int,
        default=settings.history_start if history_start is None else history_start,
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=settings.train_start if train_start is None else train_start,
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=settings.test_year if test_year is None else test_year,
    )


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    prediction_default: str,
) -> None:
    parser.add_argument("--train", type=Path, default=Path("train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("test.parquet"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(prediction_default),
    )


def _add_explainability_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: Path | None = None,
) -> None:
    parser.add_argument("--explanations", type=Path, default=default)
    parser.add_argument("--ale-bins", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--shap-background", type=int, default=100)
    parser.add_argument("--shap-samples", type=int, default=25)


def _provider(settings: Settings) -> NflDataProvider:
    return NflReadPyProvider(
        cache_mode=("filesystem" if settings.cache_mode == "filesystem" else "off"),
        cache_dir=settings.cache_dir,
    )


def _write_predictions(
    path: Path,
    test_frame: pl.DataFrame,
    predictions: NDArray[np.float64],
    *,
    identity_columns: tuple[str, ...],
    additional_columns: Mapping[str, NDArray[np.float64]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = test_frame.select(*identity_columns, TARGET_COLUMN).with_columns(
        pl.Series(PREDICTION_COLUMN, predictions, dtype=pl.Float64)
    )
    if additional_columns:
        frame = frame.with_columns(
            *(
                pl.Series(name, values, dtype=pl.Float64)
                for name, values in additional_columns.items()
            )
        )
    frame.write_parquet(path, compression="zstd", statistics=True)


def _build_options(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        train_start=args.train_start,
        test_year=args.test_year,
    )


def _receiving_build_options(args: argparse.Namespace) -> ReceivingBuildOptions:
    return ReceivingBuildOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        train_start=args.train_start,
        test_year=args.test_year,
        positions=RECEIVING_BUILD_POSITIONS[args.position],
    )


def _svr_options(args: argparse.Namespace) -> SvrOptions:
    return SvrOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
        position=args.position,
        explainability=_explainability_options(args),
        manual_features=args.manual_features,
        select_hyperparameters=args.select_hyperparameters,
        folds=args.folds,
    )


def _mlp_options(args: argparse.Namespace) -> MlpOptions:
    return MlpOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
        position=args.position,
        explainability=_explainability_options(args),
        hidden_units=args.hidden_units,
        activation=args.activation,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )


def _ebm_options(args: argparse.Namespace) -> EbmOptions:
    return EbmOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
        position=args.position,
        explainability=_explainability_options(args),
        max_bins=args.max_bins,
        interactions=args.interactions,
        max_rounds=args.max_rounds,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        outer_bags=args.outer_bags,
        validation_size=args.validation_size,
        calibration_fraction=args.calibration_fraction,
        interval_coverage=args.interval_coverage,
        random_state=args.random_state,
        n_jobs=args.jobs,
    )


def _explainability_options(args: argparse.Namespace) -> ExplainabilityOptions:
    return ExplainabilityOptions(
        path=args.explanations,
        ale_bins=args.ale_bins,
        permutation_repeats=args.permutation_repeats,
        shap_background=args.shap_background,
        shap_samples=args.shap_samples,
        random_state=args.random_state,
    )


def _run_build(
    options: BuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    manifest = build_datasets(
        DatasetBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            test_year=options.test_year,
        ),
        provider=provider,
    )
    return {
        "manifest": str(options.output_dir / "dataset-manifest.json"),
        "train_rows": manifest.outputs["train"].rows,
        "test_rows": manifest.outputs["test"].rows,
    }


def _run_build_dst(
    options: BuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    manifest = build_dst_datasets(
        DstDatasetBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            test_year=options.test_year,
        ),
        provider=provider,
    )
    return {
        "manifest": str(options.output_dir / "dataset-manifest.json"),
        "train_rows": manifest.outputs["train"].rows,
        "test_rows": manifest.outputs["test"].rows,
    }


def _run_build_kicker(
    options: BuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    manifest = build_kicker_datasets(
        KickerDatasetBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            test_year=options.test_year,
        ),
        provider=provider,
    )
    return {
        "manifest": str(options.output_dir / "dataset-manifest.json"),
        "train_rows": manifest.outputs["train"].rows,
        "test_rows": manifest.outputs["test"].rows,
    }


def _run_build_receiving(
    options: ReceivingBuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    manifest = build_receiving_datasets(
        ReceivingDatasetBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            test_year=options.test_year,
            positions=options.positions,
        ),
        provider=provider,
    )
    return {
        "manifest": str(options.output_dir / "dataset-manifest.json"),
        "train_rows": manifest.outputs["train"].rows,
        "test_rows": manifest.outputs["test"].rows,
    }


def _run_build_idp(
    options: BuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    manifest = build_idp_datasets(
        IdpDatasetBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            test_year=options.test_year,
        ),
        provider=provider,
    )
    return {
        "manifest": str(options.output_dir / "dataset-manifest.json"),
        "train_rows": manifest.outputs["train"].rows,
        "test_rows": manifest.outputs["test"].rows,
    }


def _run_svr(options: SvrOptions) -> dict[str, object]:
    if options.manual_features and options.position != "qb":
        raise FfpredError("--manual-features is only supported for --position qb")
    feature_names = POSITION_FEATURE_COLUMNS[options.position]
    identity_columns = POSITION_IDENTITY_COLUMNS[options.position]
    validator = POSITION_VALIDATORS[options.position]
    train = load_training_data(options.train_path, feature_names, validator=validator)
    test = load_training_data(options.test_path, feature_names, validator=validator)
    if options.manual_features:
        train = select_manual_features(train)
        test = select_manual_features(test)
    config = (
        select_config(
            train,
            candidate_configs(),
            folds=options.folds,
        )
        if options.select_hyperparameters
        else None
    )
    result = train_svr(train, test, **({"config": config} if config else {}))
    _write_predictions(
        options.predictions_path,
        test.frame,
        result.predictions,
        identity_columns=identity_columns,
    )
    explanations_path = options.explainability.path
    if explanations_path is not None:
        _write_diagnostics(
            explanations_path,
            "SVR",
            _model_diagnostics(
                result,
                train,
                test,
                identity_columns,
                options.explainability,
            ),
        )
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "explanations": str(explanations_path) if explanations_path else None,
        "config": asdict(config) if config else None,
    }


def _run_mlp(options: MlpOptions) -> dict[str, object]:
    feature_names = POSITION_FEATURE_COLUMNS[options.position]
    identity_columns = POSITION_IDENTITY_COLUMNS[options.position]
    validator = POSITION_VALIDATORS[options.position]
    train = load_training_data(options.train_path, feature_names, validator=validator)
    test = load_training_data(options.test_path, feature_names, validator=validator)
    config = MlpConfig(
        hidden_units=options.hidden_units,
        activation=options.activation,
        max_iterations=options.iterations,
        learning_rate=options.learning_rate,
        random_state=options.random_state,
    )
    result = train_mlp(train, test, config=config)
    _write_predictions(
        options.predictions_path,
        test.frame,
        result.predictions,
        identity_columns=identity_columns,
    )
    explanations_path = options.explainability.path
    if explanations_path is not None:
        _write_diagnostics(
            explanations_path,
            "MLPRegressor",
            _model_diagnostics(
                result,
                train,
                test,
                identity_columns,
                options.explainability,
            ),
        )
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "explanations": str(explanations_path) if explanations_path else None,
        "config": asdict(config),
    }


def _run_ebm(options: EbmOptions) -> dict[str, object]:
    feature_names = POSITION_FEATURE_COLUMNS[options.position]
    identity_columns = POSITION_IDENTITY_COLUMNS[options.position]
    validator = POSITION_VALIDATORS[options.position]
    train = load_training_data(options.train_path, feature_names, validator=validator)
    test = load_training_data(options.test_path, feature_names, validator=validator)
    config = EbmConfig(
        max_bins=options.max_bins,
        interactions=options.interactions,
        max_rounds=options.max_rounds,
        learning_rate=options.learning_rate,
        min_samples_leaf=options.min_samples_leaf,
        outer_bags=options.outer_bags,
        validation_size=options.validation_size,
        calibration_fraction=options.calibration_fraction,
        interval_coverage=options.interval_coverage,
        random_state=options.random_state,
        n_jobs=options.n_jobs,
    )
    result = train_ebm(train, test, config=config)
    interval_columns = (
        {
            "prediction_lower": result.prediction_interval.lower,
            "prediction_upper": result.prediction_interval.upper,
        }
        if result.prediction_interval is not None
        else None
    )
    _write_predictions(
        options.predictions_path,
        test.frame,
        result.predictions,
        identity_columns=identity_columns,
        additional_columns=interval_columns,
    )
    diagnostics = _model_diagnostics(
        result,
        train,
        test,
        identity_columns,
        options.explainability,
        prediction_interval=result.prediction_interval,
    )
    explanations_path = options.explainability.path
    if explanations_path is None:
        raise ValueError("EBM explanations require an output path")
    write_ebm_explanations(
        explanations_path,
        result.explanations,
        identities=test.frame.select(identity_columns).to_dicts(),
        diagnostics=diagnostics,
    )
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "explanations": str(explanations_path),
        "config": asdict(config),
    }


def _model_diagnostics(  # noqa: PLR0913
    result: TrainingResult,
    train: TrainingData,
    test: TrainingData,
    identity_columns: tuple[str, ...],
    options: ExplainabilityOptions,
    *,
    prediction_interval: ConformalPredictionInterval | None = None,
) -> dict[str, object]:
    ale = tuple(
        accumulated_local_effects(
            result.estimator,
            train.features,
            index,
            name,
            bins=options.ale_bins,
        )
        for index, name in enumerate(train.feature_names)
    )
    permutation = temporal_permutation_importance(
        result.estimator,
        test.features,
        test.target,
        test.frame["target_season"],
        test.feature_names,
        repeats=options.permutation_repeats,
        random_state=options.random_state,
    )
    shap_values = model_agnostic_shap_values(
        result.estimator,
        train.features,
        test.features,
        test.feature_names,
        max_background=options.shap_background,
        max_samples=options.shap_samples,
        random_state=options.random_state,
    )
    categorical_cohorts = tuple(
        column
        for column in ("position", "position_group", "target_week")
        if column in test.frame
    )
    opponent_strength = tuple(
        name
        for name in test.feature_names
        if name.endswith("defense_last_10_points_allowed")
    )
    numeric_cohorts = tuple(
        column for column in ("years_pro", *opponent_strength) if column in test.frame
    )
    cohorts = residual_cohorts(
        test.frame,
        test.target,
        result.predictions,
        categorical_columns=categorical_cohorts,
        quantile_columns=numeric_cohorts,
    )
    return {
        "conformal_interval": (
            {
                "coverage": prediction_interval.coverage,
                "radius": prediction_interval.radius,
                "calibration_samples": prediction_interval.calibration_samples,
            }
            if prediction_interval is not None
            else None
        ),
        "ale": [asdict(curve) for curve in ale],
        "temporal_permutation_importance": [
            asdict(importance) for importance in permutation
        ],
        "shap": {
            "feature_names": list(shap_values.feature_names),
            "values": shap_values.values.tolist(),
            "base_values": shap_values.base_values.tolist(),
            "data": shap_values.data.tolist(),
            "sample_indices": list(shap_values.sample_indices),
            "identities": test.frame[list(shap_values.sample_indices)]
            .select(identity_columns)
            .to_dicts(),
        },
        "residual_cohorts": [asdict(cohort) for cohort in cohorts],
    }


def _write_diagnostics(
    path: Path,
    model: str,
    diagnostics: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": model,
                "diagnostics": diagnostics,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_evaluate(options: EvaluateOptions) -> dict[str, object]:
    frame = pl.read_parquet(options.predictions_path)
    required = {TARGET_COLUMN, PREDICTION_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise FfpredError(
            f"{options.predictions_path} is missing columns: {sorted(missing)}"
        )
    return {"metrics": asdict(evaluate(frame[TARGET_COLUMN], frame[PREDICTION_COLUMN]))}


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: NflDataProvider | None = None,
) -> int:
    """Execute one command and return a process exit code."""
    settings = Settings.from_env()
    args = _parser(settings).parse_args(argv)
    configure_logging(args.verbose, settings.log_level)
    try:
        if args.command == "build-dataset":
            output = _run_build(
                _build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "build-dst-dataset":
            output = _run_build_dst(
                _build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "build-kicker-dataset":
            output = _run_build_kicker(
                _build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "build-receiving-dataset":
            output = _run_build_receiving(
                _receiving_build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "build-idp-dataset":
            output = _run_build_idp(
                _build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "train-svr":
            output = _run_svr(_svr_options(args))
        elif args.command == "train-mlp":
            output = _run_mlp(_mlp_options(args))
        elif args.command == "train-ebm":
            output = _run_ebm(_ebm_options(args))
        else:
            output = _run_evaluate(EvaluateOptions(predictions_path=args.predictions))
    except FfpredError as error:
        LOGGER.error("%s", error)  # noqa: TRY400
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run() -> Never:
    """Console-script adapter."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()

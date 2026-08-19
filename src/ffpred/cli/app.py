"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Never

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.cli.options import (
    BuildOptions,
    EvaluateOptions,
    MlpOptions,
    SvrOptions,
)
from ffpred.config import Settings
from ffpred.datasets.builder import (
    DatasetBuildConfig,
    DstDatasetBuildConfig,
    build_datasets,
    build_dst_datasets,
)
from ffpred.errors import FfpredError
from ffpred.evaluation.metrics import evaluate
from ffpred.features import dst_schema
from ffpred.features import schema as qb_schema
from ffpred.features.schema import TARGET_COLUMN
from ffpred.logging import configure_logging
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.training.data import load_training_data
from ffpred.training.mlp import MlpConfig, train_mlp
from ffpred.training.svr import (
    candidate_configs,
    select_config,
    select_manual_features,
    train_svr,
)

LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMN = "prediction"
#: Positions supported by the generic train-svr/train-mlp/evaluate commands.
#: Adding a position here only requires a feature-schema module exposing
#: MODEL_FEATURE_COLUMNS, IDENTITY_COLUMNS, and validate_feature_frame;
#: TARGET_COLUMN is shared.
POSITION_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "qb": qb_schema.MODEL_FEATURE_COLUMNS,
    "dst": dst_schema.MODEL_FEATURE_COLUMNS,
}
POSITION_IDENTITY_COLUMNS: dict[str, tuple[str, ...]] = {
    "qb": qb_schema.IDENTITY_COLUMNS,
    "dst": dst_schema.IDENTITY_COLUMNS,
}
POSITION_VALIDATORS = {
    "qb": qb_schema.validate_feature_frame,
    "dst": dst_schema.validate_feature_frame,
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

    svr = subparsers.add_parser("train-svr", help="train an SVR model")
    _add_dataset_arguments(svr, "svr-predictions.parquet")
    svr.add_argument(
        "--position", choices=tuple(POSITION_FEATURE_COLUMNS), default="qb"
    )
    svr.add_argument("--manual-features", action="store_true")
    svr.add_argument("--select-hyperparameters", action="store_true")
    svr.add_argument("--folds", type=int, default=5)

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

    evaluation = subparsers.add_parser(
        "evaluate",
        help="evaluate a prediction Parquet artifact",
    )
    evaluation.add_argument("predictions", type=Path)
    return parser


def _add_build_arguments(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument("--history-start", type=int, default=settings.history_start)
    parser.add_argument("--train-start", type=int, default=settings.train_start)
    parser.add_argument("--test-year", type=int, default=settings.test_year)


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = test_frame.select(*identity_columns, TARGET_COLUMN).with_columns(
        pl.Series(PREDICTION_COLUMN, predictions, dtype=pl.Float64)
    )
    frame.write_parquet(path, compression="zstd", statistics=True)


def _build_options(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        train_start=args.train_start,
        test_year=args.test_year,
    )


def _svr_options(args: argparse.Namespace) -> SvrOptions:
    return SvrOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
        position=args.position,
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
        hidden_units=args.hidden_units,
        activation=args.activation,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
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
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
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
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "config": asdict(config),
    }


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
        elif args.command == "train-svr":
            output = _run_svr(_svr_options(args))
        elif args.command == "train-mlp":
            output = _run_mlp(_mlp_options(args))
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

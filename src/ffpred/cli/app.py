"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Never

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.cli.options import (
    BuildOptions,
    EvaluateOptions,
    ForecastArchiveOptions,
    ForecastBuildOptions,
    MlpOptions,
    ProjectionOptions,
    SvrOptions,
)
from ffpred.config import Settings
from ffpred.datasets.archive import ForecastArchiveConfig, build_forecast_archive
from ffpred.datasets.builder import DatasetBuildConfig, build_datasets
from ffpred.datasets.forecast import ForecastBuildConfig, build_forecast_datasets
from ffpred.errors import FfpredError
from ffpred.evaluation.metrics import evaluate
from ffpred.features.all_positions import ALL_POSITION_MODEL_FEATURE_COLUMNS
from ffpred.features.schema import IDENTITY_COLUMNS, TARGET_COLUMN
from ffpred.logging import configure_logging
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.training.data import load_training_data
from ffpred.training.mlp import MlpConfig, create_archive_estimator, train_mlp
from ffpred.training.mlp import create_estimator as create_mlp
from ffpred.training.projection import load_projection_data, project
from ffpred.training.svr import (
    DEFAULT_SVR_CONFIG,
    candidate_configs,
    create_scalable_estimator,
    select_config,
    select_manual_features,
    train_svr,
)
from ffpred.training.svr import create_estimator as create_svr

LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMN = "prediction"


def _parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffpred",
        description="Build and evaluate fantasy-football prediction models",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="build train/test datasets")
    build.add_argument("--output-dir", type=Path, default=settings.output_dir)
    build.add_argument("--history-start", type=int, default=settings.history_start)
    build.add_argument("--train-start", type=int, default=settings.train_start)
    build.add_argument("--test-year", type=int, default=settings.test_year)

    forecast = subparsers.add_parser(
        "build-forecast",
        help="build frozen training and future schedule datasets",
    )
    forecast.add_argument("--output-dir", type=Path, required=True)
    forecast.add_argument(
        "--history-start",
        type=int,
        default=settings.history_start,
    )
    forecast.add_argument("--train-start", type=int, default=settings.train_start)
    forecast.add_argument(
        "--history-through",
        type=int,
        required=True,
        dest="history_through_season",
    )
    forecast.add_argument("--target-year", type=int, required=True)
    forecast.add_argument("--as-of", type=date.fromisoformat)
    forecast.add_argument("--include-actuals", action="store_true")

    archive = subparsers.add_parser(
        "build-forecast-archive",
        help="build frozen all-position forecasts for a range of seasons",
    )
    archive.add_argument("--output-dir", type=Path, default=settings.output_dir)
    archive.add_argument("--history-start", type=int, default=1999)
    archive.add_argument("--first-target-year", type=int, default=2010)
    archive.add_argument(
        "--last-target-year",
        type=int,
        default=date.today().year,
    )
    archive.add_argument("--as-of", type=date.fromisoformat)

    svr = subparsers.add_parser("train-svr", help="train an SVR model")
    _add_dataset_arguments(svr, "svr-predictions.parquet")
    svr.add_argument("--manual-features", action="store_true")
    svr.add_argument("--select-hyperparameters", action="store_true")
    svr.add_argument("--folds", type=int, default=5)

    mlp = subparsers.add_parser("train-mlp", help="train an MLP model")
    _add_dataset_arguments(mlp, "mlp-predictions.parquet")
    mlp.add_argument("--hidden-units", type=int, default=50)
    mlp.add_argument(
        "--activation",
        choices=("identity", "logistic", "tanh", "relu"),
        default="relu",
    )
    mlp.add_argument("--iterations", type=int, default=1000)
    mlp.add_argument("--learning-rate", type=float, default=0.001)
    mlp.add_argument("--random-state", type=int, default=42)

    project_svr = subparsers.add_parser(
        "project-svr",
        help="fit SVR and predict a frozen forecast dataset",
    )
    _add_projection_arguments(project_svr, "svr-predictions.parquet")

    project_mlp = subparsers.add_parser(
        "project-mlp",
        help="fit MLP and predict a frozen forecast dataset",
    )
    _add_projection_arguments(project_mlp, "mlp-predictions.parquet")

    evaluation = subparsers.add_parser(
        "evaluate",
        help="evaluate a prediction Parquet artifact",
    )
    evaluation.add_argument("predictions", type=Path)
    return parser


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


def _add_projection_arguments(
    parser: argparse.ArgumentParser,
    prediction_default: str,
) -> None:
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "position",
            "team",
            "opponent",
            "forecast_as_of",
            "history_through_season",
            TARGET_COLUMN,
        )
        if column in test_frame.columns
    ]
    frame = test_frame.select(identity).with_columns(
        pl.Series(PREDICTION_COLUMN, predictions, dtype=pl.Float64),
    )
    frame.write_parquet(path, compression="zstd", statistics=True)


def _build_options(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        train_start=args.train_start,
        test_year=args.test_year,
    )


def _forecast_build_options(args: argparse.Namespace) -> ForecastBuildOptions:
    return ForecastBuildOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        train_start=args.train_start,
        history_through_season=args.history_through_season,
        target_year=args.target_year,
        as_of=args.as_of,
        include_actuals=args.include_actuals,
    )


def _forecast_archive_options(args: argparse.Namespace) -> ForecastArchiveOptions:
    return ForecastArchiveOptions(
        output_dir=args.output_dir,
        history_start=args.history_start,
        first_target_year=args.first_target_year,
        last_target_year=args.last_target_year,
        as_of=args.as_of,
    )


def _projection_options(args: argparse.Namespace) -> ProjectionOptions:
    return ProjectionOptions(
        train_path=args.train,
        forecast_path=args.forecast,
        predictions_path=args.predictions,
    )


def _svr_options(args: argparse.Namespace) -> SvrOptions:
    return SvrOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
        manual_features=args.manual_features,
        select_hyperparameters=args.select_hyperparameters,
        folds=args.folds,
    )


def _mlp_options(args: argparse.Namespace) -> MlpOptions:
    return MlpOptions(
        train_path=args.train,
        test_path=args.test,
        predictions_path=args.predictions,
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


def _run_forecast_build(
    options: ForecastBuildOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    result = build_forecast_datasets(
        ForecastBuildConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            train_start=options.train_start,
            history_through_season=options.history_through_season,
            target_year=options.target_year,
            as_of=options.as_of,
            include_actuals=options.include_actuals,
        ),
        provider=provider,
    )
    return {
        "training": result.training.path,
        "training_rows": result.training.rows,
        "forecast": result.forecast.path,
        "forecast_rows": result.forecast.rows,
        "manifest": str(result.manifest_path),
    }


def _run_forecast_archive(
    options: ForecastArchiveOptions,
    provider: NflDataProvider,
) -> dict[str, object]:
    result = build_forecast_archive(
        ForecastArchiveConfig(
            output_dir=options.output_dir,
            history_start=options.history_start,
            first_target_year=options.first_target_year,
            last_target_year=options.last_target_year,
            as_of=options.as_of,
        ),
        provider=provider,
    )
    return {
        "seasons": [
            {
                "target_year": season.target_year,
                "training": season.training.path,
                "training_rows": season.training.rows,
                "forecast": season.forecast.path,
                "forecast_rows": season.forecast.rows,
                "manifest": str(season.manifest_path),
            }
            for season in result.seasons
        ]
    }


def _run_svr(options: SvrOptions) -> dict[str, object]:
    train = load_training_data(options.train_path)
    test = load_training_data(options.test_path)
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
    _write_predictions(options.predictions_path, test.frame, result.predictions)
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "config": asdict(config) if config else None,
    }


def _run_mlp(options: MlpOptions) -> dict[str, object]:
    train = load_training_data(options.train_path)
    test = load_training_data(options.test_path)
    config = MlpConfig(
        hidden_units=options.hidden_units,
        activation=options.activation,
        max_iterations=options.iterations,
        learning_rate=options.learning_rate,
        random_state=options.random_state,
    )
    result = train_mlp(train, test, config=config)
    _write_predictions(options.predictions_path, test.frame, result.predictions)
    return {
        "metrics": asdict(result.metrics),
        "features": list(result.feature_names),
        "predictions": str(options.predictions_path),
        "config": asdict(config),
    }


def _run_projection(
    options: ProjectionOptions,
    *,
    model: str,
) -> dict[str, object]:
    train = load_training_data(options.train_path)
    forecast = load_projection_data(options.forecast_path)
    is_archive = train.feature_names == ALL_POSITION_MODEL_FEATURE_COLUMNS
    if model == "svr":
        estimator = (
            create_scalable_estimator()
            if is_archive
            else create_svr(DEFAULT_SVR_CONFIG)
        )
    else:
        estimator = (
            create_archive_estimator() if is_archive else create_mlp(MlpConfig())
        )
    predictions = project(estimator, train, forecast)
    _write_predictions(options.predictions_path, forecast.frame, predictions)
    scored = forecast.frame.with_columns(
        pl.Series(PREDICTION_COLUMN, predictions, dtype=pl.Float64)
    ).drop_nulls(TARGET_COLUMN)
    metrics = (
        asdict(evaluate(scored[TARGET_COLUMN], scored[PREDICTION_COLUMN]))
        if not scored.is_empty()
        else None
    )
    return {
        "metrics": metrics,
        "features": list(train.feature_names),
        "predictions": str(options.predictions_path),
        "forecast_rows": forecast.frame.height,
        "history_through_season": forecast.frame["history_through_season"][0],
        "target_year": forecast.frame["target_season"][0],
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
        elif args.command == "build-forecast":
            output = _run_forecast_build(
                _forecast_build_options(args),
                provider or _provider(settings),
            )
        elif args.command == "build-forecast-archive":
            output = _run_forecast_archive(
                _forecast_archive_options(args),
                provider or _provider(settings),
            )
        elif args.command == "train-svr":
            output = _run_svr(_svr_options(args))
        elif args.command == "train-mlp":
            output = _run_mlp(_mlp_options(args))
        elif args.command == "project-svr":
            output = _run_projection(_projection_options(args), model="svr")
        elif args.command == "project-mlp":
            output = _run_projection(_projection_options(args), model="mlp")
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

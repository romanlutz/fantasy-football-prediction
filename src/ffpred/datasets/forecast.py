"""Reproducible point-in-time forecast dataset construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from ffpred import __version__
from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_quarterback_histories,
)
from ffpred.datasets.io import file_sha256, write_dataset
from ffpred.datasets.manifest import DatasetArtifact
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.errors import ConfigurationError, EmptyDatasetError
from ffpred.features.builder import build_feature_frame
from ffpred.features.forecast import (
    ForecastFrameConfig,
    ForecastSources,
    build_forecast_frame,
)
from ffpred.features.schema import validate_forecast_frame
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider

FORECAST_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastBuildConfig:
    """All values that materially determine a point-in-time forecast."""

    output_dir: Path
    history_start: int
    train_start: int
    history_through_season: int
    target_year: int
    as_of: date | None = None
    include_actuals: bool = False
    scoring: ScoringConfig = DEFAULT_SCORING

    def __post_init__(self) -> None:
        if not (
            self.history_start
            < self.train_start
            <= self.history_through_season
            < self.target_year
        ):
            raise ConfigurationError(
                "Expected history_start < train_start <= "
                "history_through_season < target_year"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastBuildResult:
    """Paths and row counts produced by a forecast build."""

    training: DatasetArtifact
    forecast: DatasetArtifact
    manifest_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastManifestInputs:
    """Values serialized into a forecast manifest."""

    config: ForecastBuildConfig
    provider: ProvenanceProvider
    training: DatasetArtifact
    forecast: DatasetArtifact
    forecast_as_of: str


def _write_forecast(path: Path, frame: pl.DataFrame) -> DatasetArtifact:
    validate_forecast_frame(frame)
    if frame.is_empty():
        raise EmptyDatasetError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)
    return DatasetArtifact(
        path=str(path),
        rows=frame.height,
        columns=frame.width,
        sha256=file_sha256(path),
    )


def _write_manifest(
    path: Path,
    inputs: ForecastManifestInputs,
) -> None:
    config = inputs.config
    manifest = {
        "schema_version": FORECAST_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "provider": dict(inputs.provider.metadata()),
        "parameters": {
            "history_start": config.history_start,
            "train_start": config.train_start,
            "history_through_season": config.history_through_season,
            "target_year": config.target_year,
            "requested_as_of": config.as_of.isoformat() if config.as_of else None,
            "forecast_as_of": inputs.forecast_as_of,
            "include_actuals": config.include_actuals,
            "scoring": asdict(config.scoring),
        },
        "sources": {
            name: asdict(artifact)
            for name, artifact in sorted(inputs.provider.artifacts.items())
        },
        "outputs": {
            "training": asdict(inputs.training),
            "forecast": asdict(inputs.forecast),
        },
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_forecast_datasets(
    config: ForecastBuildConfig,
    *,
    provider: NflDataProvider | None = None,
) -> ForecastBuildResult:
    """Build frozen training and target-season feature artifacts."""
    seasons = tuple(range(config.history_start, config.history_through_season + 1))
    recording_provider = ProvenanceProvider(provider or NflReadPyProvider())
    quarterback_histories = acquire_quarterback_histories(
        seasons,
        provider=recording_provider,
    )
    defense_histories = acquire_defense_histories(
        seasons,
        provider=recording_provider,
    )
    historical = build_feature_frame(
        quarterback_histories,
        defense_histories,
        scoring=config.scoring,
    )
    training_frame = historical.filter(
        pl.col("target_season").is_between(
            config.train_start,
            config.history_through_season,
        )
    )
    actual_stats = (
        recording_provider.load_player_stats((config.target_year,))
        if config.include_actuals
        else None
    )
    forecast_frame = build_forecast_frame(
        ForecastSources(
            quarterback_histories=quarterback_histories,
            defense_histories=defense_histories,
            schedules=recording_provider.load_schedules((config.target_year,)),
            depth_charts=recording_provider.load_depth_charts((config.target_year,)),
            players=recording_provider.load_players(),
            actual_player_stats=actual_stats,
        ),
        ForecastFrameConfig(
            history_through_season=config.history_through_season,
            target_year=config.target_year,
            as_of=config.as_of,
            scoring=config.scoring,
        ),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training = write_dataset(config.output_dir / "training.parquet", training_frame)
    forecast = _write_forecast(
        config.output_dir / "forecast.parquet",
        forecast_frame,
    )
    manifest_path = config.output_dir / "forecast-manifest.json"
    _write_manifest(
        manifest_path,
        ForecastManifestInputs(
            config=config,
            provider=recording_provider,
            training=training,
            forecast=forecast,
            forecast_as_of=str(forecast_frame["forecast_as_of"][0]),
        ),
    )
    return ForecastBuildResult(
        training=training,
        forecast=forecast,
        manifest_path=manifest_path,
    )

"""Build a point-in-time all-position forecast archive."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from ffpred import __version__
from ffpred.datasets.io import file_sha256
from ffpred.datasets.manifest import DatasetArtifact
from ffpred.errors import ConfigurationError, EmptyDatasetError
from ffpred.features.all_positions import (
    ALL_POSITION_MODEL_FEATURE_COLUMNS,
    FANTASY_POSITIONS,
    ForecastFrameConfig,
    build_actual_frame,
    build_all_position_forecast_frame,
    build_all_position_training_frame,
    validate_all_position_frame,
)
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider

ARCHIVE_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastArchiveConfig:
    """Configuration for an expanding-window season archive."""

    output_dir: Path
    history_start: int = 1999
    first_target_year: int = 2010
    last_target_year: int = field(default_factory=lambda: date.today().year)
    as_of: date | None = None

    def __post_init__(self) -> None:
        if self.history_start + 1 >= self.first_target_year:
            raise ConfigurationError(
                "history_start must leave at least one completed training target "
                "before first_target_year"
            )
        if self.first_target_year > self.last_target_year:
            raise ConfigurationError(
                "first_target_year must not exceed last_target_year"
            )
        if self.last_target_year > date.today().year:
            raise ConfigurationError("last_target_year cannot be in the future")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchiveSeasonResult:
    """Artifacts produced for one target season."""

    target_year: int
    training: DatasetArtifact
    forecast: DatasetArtifact
    manifest_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastArchiveResult:
    """Complete list of season artifacts in an archive build."""

    seasons: tuple[ArchiveSeasonResult, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchiveManifestInputs:
    """Inputs serialized into one archive season manifest."""

    config: ForecastArchiveConfig
    provider: ProvenanceProvider
    result: ArchiveSeasonResult
    forecast_frame: pl.DataFrame


def _write_frame(
    path: Path,
    frame: pl.DataFrame,
    *,
    target_required: bool,
) -> DatasetArtifact:
    validate_all_position_frame(frame, target_required=target_required)
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
    inputs: ArchiveManifestInputs,
) -> None:
    config = inputs.config
    provider = inputs.provider
    result = inputs.result
    forecast_frame = inputs.forecast_frame
    relevant_sources = {
        name: asdict(artifact)
        for name, artifact in sorted(provider.artifacts.items())
        if not name.startswith(("depth_charts:", "injuries:", "rosters_weekly:"))
        or name
        in {
            f"depth_charts:{result.target_year}",
            f"injuries:{result.target_year}",
            f"rosters_weekly:{result.target_year}",
        }
    }
    manifest = {
        "schema_version": ARCHIVE_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "provider": dict(provider.metadata()),
        "parameters": {
            "history_start": config.history_start,
            "first_target_year": config.first_target_year,
            "last_target_year": config.last_target_year,
            "target_year": result.target_year,
            "history_through_season": result.target_year - 1,
            "forecast_as_of": str(forecast_frame["forecast_as_of"][0]),
            "positions": list(FANTASY_POSITIONS),
            "scoring": "standard_non_ppr",
            "model_features": list(ALL_POSITION_MODEL_FEATURE_COLUMNS),
            "injury_absence_rule": (
                "scheduled game with no fantasy result and either an Out injury "
                "report or reserve-list weekly roster status"
            ),
        },
        "sources": relevant_sources,
        "outputs": {
            "training": asdict(result.training),
            "forecast": asdict(result.forecast),
        },
        "roster_coverage": {
            "teams": forecast_frame["team"].n_unique(),
            "team_position_pairs": forecast_frame.select(
                "team",
                "position",
            )
            .unique()
            .height,
            "expected_team_position_pairs": (
                forecast_frame["team"].n_unique() * len(FANTASY_POSITIONS)
            ),
            "players_by_position": {
                str(row["position"]): int(row["players"])
                for row in (
                    forecast_frame.group_by("position")
                    .agg(pl.col("player_id").n_unique().alias("players"))
                    .sort("position")
                    .iter_rows(named=True)
                )
            },
        },
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_forecast_archive(
    config: ForecastArchiveConfig,
    *,
    provider: NflDataProvider | None = None,
) -> ForecastArchiveResult:
    """Build frozen all-position datasets for every requested target season."""
    recording_provider = ProvenanceProvider(provider or NflReadPyProvider())
    completed_through = min(config.last_target_year, date.today().year - 1)
    stats_seasons = tuple(range(config.history_start, completed_through + 1))
    schedule_seasons = tuple(range(config.history_start, config.last_target_year + 1))
    schedules = recording_provider.load_schedules(schedule_seasons)
    actuals = build_actual_frame(
        recording_provider.load_player_stats(stats_seasons),
        recording_provider.load_team_stats(stats_seasons),
        schedules,
    )
    all_training = build_all_position_training_frame(
        actuals,
        schedules,
        target_years=range(config.history_start + 1, completed_through + 1),
    )

    results: list[ArchiveSeasonResult] = []
    for target_year in range(
        config.first_target_year,
        config.last_target_year + 1,
    ):
        season_dir = config.output_dir / str(target_year)
        training_frame = all_training.filter(pl.col("target_season") < target_year)
        if target_year <= completed_through:
            injuries = recording_provider.load_injuries((target_year,))
            rosters_weekly = recording_provider.load_rosters_weekly((target_year,))
        else:
            injuries = pl.DataFrame()
            rosters_weekly = pl.DataFrame()
        forecast_frame = build_all_position_forecast_frame(
            actuals,
            schedules,
            recording_provider.load_depth_charts((target_year,)),
            config=ForecastFrameConfig(
                target_year=target_year,
                as_of=config.as_of,
                injuries=injuries,
                rosters_weekly=rosters_weekly,
            ),
        )
        training = _write_frame(
            season_dir / "training.parquet",
            training_frame,
            target_required=True,
        )
        forecast = _write_frame(
            season_dir / "forecast.parquet",
            forecast_frame,
            target_required=False,
        )
        result = ArchiveSeasonResult(
            target_year=target_year,
            training=training,
            forecast=forecast,
            manifest_path=season_dir / "forecast-manifest.json",
        )
        _write_manifest(
            result.manifest_path,
            ArchiveManifestInputs(
                config=config,
                provider=recording_provider,
                result=result,
                forecast_frame=forecast_frame,
            ),
        )
        results.append(result)
    return ForecastArchiveResult(seasons=tuple(results))

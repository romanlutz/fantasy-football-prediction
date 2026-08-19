"""End-to-end reproducible dataset construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffpred import __version__
from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_dst_histories,
    acquire_quarterback_histories,
)
from ffpred.datasets.io import write_dataset
from ffpred.datasets.manifest import (
    BuildParameters,
    DatasetManifest,
)
from ffpred.domain.scoring import (
    DEFAULT_DST_SCORING,
    DEFAULT_SCORING,
    DstScoringConfig,
    ScoringConfig,
)
from ffpred.errors import ConfigurationError
from ffpred.features.builder import build_feature_frame
from ffpred.features.dst_builder import build_dst_feature_frame
from ffpred.features.dst_schema import FEATURE_SCHEMA as DST_FEATURE_SCHEMA
from ffpred.features.dst_schema import validate_feature_frame as validate_dst_frame
from ffpred.features.schema import FEATURE_SCHEMA
from ffpred.logging import configure_logging
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider

LOGGER = logging.getLogger(__name__)


def _validate_season_range(
    history_start: int, train_start: int, test_year: int
) -> None:
    if not history_start < train_start <= test_year:
        raise ConfigurationError("Expected history_start < train_start <= test_year")


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetBuildConfig:
    """All values that materially determine generated datasets."""

    output_dir: Path = Path()
    history_start: int = 2009
    train_start: int = 2010
    test_year: int = 2014
    scoring: ScoringConfig = DEFAULT_SCORING

    def __post_init__(self) -> None:
        _validate_season_range(self.history_start, self.train_start, self.test_year)


DEFAULT_BUILD_CONFIG = DatasetBuildConfig()


@dataclass(frozen=True, slots=True, kw_only=True)
class DstDatasetBuildConfig:
    """All values that materially determine generated D/ST datasets."""

    output_dir: Path = Path()
    history_start: int = 2009
    train_start: int = 2010
    test_year: int = 2014
    scoring: DstScoringConfig = DEFAULT_DST_SCORING

    def __post_init__(self) -> None:
        _validate_season_range(self.history_start, self.train_start, self.test_year)


DEFAULT_DST_BUILD_CONFIG = DstDatasetBuildConfig()


def _feature_schema_sha256(schema: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {column: str(dtype) for column, dtype in schema.items()},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_datasets(
    config: DatasetBuildConfig = DEFAULT_BUILD_CONFIG,
    *,
    provider: NflDataProvider | None = None,
) -> DatasetManifest:
    """Acquire, engineer, split, persist, and describe train/test datasets."""
    seasons = tuple(range(config.history_start, config.test_year + 1))
    recording_provider = ProvenanceProvider(provider or NflReadPyProvider())
    quarterback_histories = acquire_quarterback_histories(
        seasons,
        provider=recording_provider,
    )
    defense_histories = acquire_defense_histories(
        seasons,
        provider=recording_provider,
    )
    features = build_feature_frame(
        quarterback_histories,
        defense_histories,
        scoring=config.scoring,
    )
    train = features.filter(
        pl.col("target_season").is_between(
            config.train_start,
            config.test_year,
            closed="left",
        )
    )
    test = features.filter(pl.col("target_season") == config.test_year)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": write_dataset(config.output_dir / "train.parquet", train),
        "test": write_dataset(config.output_dir / "test.parquet", test),
    }
    manifest = DatasetManifest(
        generated_at=datetime.now(UTC).isoformat(),
        package_version=__version__,
        provider=dict(recording_provider.metadata()),
        parameters=BuildParameters(
            history_start=config.history_start,
            train_start=config.train_start,
            test_year=config.test_year,
            scoring=asdict(config.scoring),
        ),
        feature_schema_sha256=_feature_schema_sha256(FEATURE_SCHEMA),
        sources=dict(recording_provider.artifacts),
        outputs=outputs,
    )
    manifest.write(config.output_dir / "dataset-manifest.json")
    return manifest


def build_dst_datasets(
    config: DstDatasetBuildConfig = DEFAULT_DST_BUILD_CONFIG,
    *,
    provider: NflDataProvider | None = None,
) -> DatasetManifest:
    """Acquire, engineer, split, persist, and describe team D/ST datasets."""
    seasons = tuple(range(config.history_start, config.test_year + 1))
    recording_provider = ProvenanceProvider(provider or NflReadPyProvider())
    histories = acquire_dst_histories(seasons, provider=recording_provider)
    features = build_dst_feature_frame(histories, scoring=config.scoring)
    train = features.filter(
        pl.col("target_season").is_between(
            config.train_start,
            config.test_year,
            closed="left",
        )
    )
    test = features.filter(pl.col("target_season") == config.test_year)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": write_dataset(
            config.output_dir / "train.parquet", train, validator=validate_dst_frame
        ),
        "test": write_dataset(
            config.output_dir / "test.parquet", test, validator=validate_dst_frame
        ),
    }
    manifest = DatasetManifest(
        generated_at=datetime.now(UTC).isoformat(),
        package_version=__version__,
        provider=dict(recording_provider.metadata()),
        parameters=BuildParameters(
            history_start=config.history_start,
            train_start=config.train_start,
            test_year=config.test_year,
            scoring=asdict(config.scoring),
        ),
        feature_schema_sha256=_feature_schema_sha256(DST_FEATURE_SCHEMA),
        sources=dict(recording_provider.artifacts),
        outputs=outputs,
    )
    manifest.write(config.output_dir / "dataset-manifest.json")
    return manifest


def main() -> None:
    """Run the legacy dataset-building command."""
    parser = argparse.ArgumentParser(description="Build fantasy football datasets")
    parser.add_argument("--output-dir", type=Path, default=Path())
    parser.add_argument("--history-start", type=int, default=2009)
    parser.add_argument("--train-start", type=int, default=2010)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    configure_logging(args.verbose)

    manifest = build_datasets(
        DatasetBuildConfig(
            output_dir=args.output_dir,
            history_start=args.history_start,
            train_start=args.train_start,
            test_year=args.test_year,
        )
    )
    LOGGER.info(
        "wrote %d training rows and %d test rows to %s",
        manifest.outputs["train"].rows,
        manifest.outputs["test"].rows,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

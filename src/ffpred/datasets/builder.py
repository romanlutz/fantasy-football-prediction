"""End-to-end reproducible dataset construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffpred import __version__
from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_quarterback_histories,
)
from ffpred.datasets.io import write_dataset
from ffpred.datasets.manifest import (
    BuildParameters,
    DatasetManifest,
)
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.features.builder import build_feature_frame
from ffpred.features.schema import FEATURE_SCHEMA
from ffpred.logging import configure_logging
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider

LOGGER = logging.getLogger(__name__)


def _feature_schema_sha256() -> str:
    encoded = json.dumps(
        {column: str(dtype) for column, dtype in FEATURE_SCHEMA.items()},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_datasets(
    output_dir: Path,
    history_start: int = 2009,
    train_start: int = 2010,
    test_year: int = 2014,
    *,
    provider: NflDataProvider | None = None,
    scoring: ScoringConfig = DEFAULT_SCORING,
) -> DatasetManifest:
    """Acquire, engineer, split, persist, and describe train/test datasets."""
    seasons = tuple(range(history_start, test_year + 1))
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
        scoring=scoring,
    )
    train = features.filter(
        pl.col("target_season").is_between(
            train_start,
            test_year,
            closed="left",
        )
    )
    test = features.filter(pl.col("target_season") == test_year)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": write_dataset(output_dir / "train.parquet", train),
        "test": write_dataset(output_dir / "test.parquet", test),
    }
    manifest = DatasetManifest(
        generated_at=datetime.now(UTC).isoformat(),
        package_version=__version__,
        provider=dict(recording_provider.metadata()),
        parameters=BuildParameters(
            history_start=history_start,
            train_start=train_start,
            test_year=test_year,
            scoring=asdict(scoring),
        ),
        feature_schema_sha256=_feature_schema_sha256(),
        sources=dict(recording_provider.artifacts),
        outputs=outputs,
    )
    manifest.write(output_dir / "dataset-manifest.json")
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

    manifest = generate_datasets(
        args.output_dir,
        args.history_start,
        args.train_start,
        args.test_year,
    )
    LOGGER.info(
        "wrote %d training rows and %d test rows to %s",
        manifest.outputs["train"].rows,
        manifest.outputs["test"].rows,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

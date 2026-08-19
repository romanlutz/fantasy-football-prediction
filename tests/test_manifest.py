from ffpred.datasets.manifest import (
    BuildParameters,
    DatasetArtifact,
    DatasetManifest,
)
from ffpred.providers.provenance import SourceArtifact


def test_manifest_round_trip_is_lossless() -> None:
    manifest = DatasetManifest(
        generated_at="2026-01-01T00:00:00+00:00",
        package_version="1.0.0",
        provider={"client": "fake"},
        parameters=BuildParameters(
            history_start=2020,
            train_start=2021,
            test_year=2025,
            scoring={"passing_touchdown": 4.0},
        ),
        feature_schema_sha256="schema",
        sources={
            "players": SourceArtifact(
                name="players",
                rows=1,
                sha256="source",
                schema={"player_id": "String"},
            )
        },
        outputs={
            "train": DatasetArtifact(
                path="train.parquet",
                rows=1,
                columns=43,
                sha256="output",
            )
        },
    )

    assert DatasetManifest.from_json(manifest.to_json()) == manifest

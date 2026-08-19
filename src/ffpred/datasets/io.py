"""Typed Parquet dataset persistence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import polars as pl

from ffpred.datasets.manifest import DatasetArtifact
from ffpred.errors import DatasetIntegrityError, EmptyDatasetError
from ffpred.features.schema import validate_feature_frame

Validator = Callable[[pl.DataFrame], pl.DataFrame]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 identity of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    path: Path,
    frame: pl.DataFrame,
    *,
    validator: Validator = validate_feature_frame,
) -> DatasetArtifact:
    """Validate and atomically persist a feature table as Parquet."""
    validator(frame)
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


def read_dataset(
    path: Path,
    *,
    expected_sha256: str | None = None,
    validator: Validator = validate_feature_frame,
) -> pl.DataFrame:
    """Read and validate a persisted feature table."""
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise DatasetIntegrityError(
                f"Checksum mismatch for {path}: "
                f"expected {expected_sha256}, got {actual}"
            )
    return validator(pl.read_parquet(path))

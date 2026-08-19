from pathlib import Path

import polars as pl
import pytest

from ffpred.datasets.io import read_dataset, write_dataset
from ffpred.errors import DatasetIntegrityError, EmptyDatasetError
from ffpred.features.schema import FEATURE_SCHEMA


def _feature_frame() -> pl.DataFrame:
    values: dict[str, list[object]] = {}
    for column, dtype in FEATURE_SCHEMA.items():
        if dtype == pl.String:
            values[column] = ["value"]
        elif dtype == pl.Int64:
            values[column] = [1]
        else:
            values[column] = [1.0]
    return pl.DataFrame(values, schema=FEATURE_SCHEMA)


def test_parquet_round_trip_preserves_schema(tmp_path: Path) -> None:
    path = tmp_path / "dataset.parquet"

    artifact = write_dataset(path, _feature_frame())
    restored = read_dataset(path, expected_sha256=artifact.sha256)

    assert restored.schema == _feature_frame().schema
    assert restored.to_dicts() == _feature_frame().to_dicts()


def test_empty_dataset_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(EmptyDatasetError, match="No rows were generated"):
        write_dataset(
            tmp_path / "train.parquet",
            pl.DataFrame(schema=FEATURE_SCHEMA),
        )


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.parquet"
    write_dataset(path, _feature_frame())

    with pytest.raises(DatasetIntegrityError, match="Checksum mismatch"):
        read_dataset(path, expected_sha256="0" * 64)

from pathlib import Path

import polars as pl

from ffpred.datasets.builder import generate_datasets
from ffpred.datasets.manifest import DatasetManifest
from ffpred.features.schema import FEATURE_COLUMNS
from tests.factories import make_provider


def test_generate_datasets_builds_reproducible_artifacts(tmp_path: Path) -> None:
    manifest = generate_datasets(
        tmp_path,
        history_start=2020,
        train_start=2021,
        test_year=2022,
        provider=make_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    restored_manifest = DatasetManifest.read(tmp_path / "dataset-manifest.json")
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert train["target_season"].to_list() == [2021]
    assert test["target_season"].to_list() == [2022]
    assert restored_manifest == manifest
    assert set(manifest.outputs) == {"train", "test"}
    assert "play_by_play:2020" in manifest.sources
    assert manifest.sources["player_stats:2020-2021-2022"].rows == 3

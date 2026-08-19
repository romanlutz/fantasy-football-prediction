from pathlib import Path

import polars as pl

from ffpred.datasets.builder import generate_datasets
from ffpred.features.schema import MODEL_FEATURE_COLUMNS
from tests.factories import make_provider


def test_every_feature_history_precedes_its_target(tmp_path: Path) -> None:
    generate_datasets(
        tmp_path,
        history_start=2020,
        train_start=2021,
        test_year=2022,
        provider=make_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    for row in pl.concat([train, test]).iter_rows(named=True):
        target = (row["target_season"], row["target_week"])
        assert (
            row["qb_history_through_season"],
            row["qb_history_through_week"],
        ) < target
        assert (
            row["defense_history_through_season"],
            row["defense_history_through_week"],
        ) < target


def test_train_and_test_games_are_disjoint(tmp_path: Path) -> None:
    generate_datasets(
        tmp_path,
        history_start=2020,
        train_start=2021,
        test_year=2022,
        provider=make_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    assert set(train["target_game_id"]).isdisjoint(test["target_game_id"])
    assert train.select(MODEL_FEATURE_COLUMNS).width == len(MODEL_FEATURE_COLUMNS)

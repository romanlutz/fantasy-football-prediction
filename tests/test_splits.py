import polars as pl
import pytest

from ffpred.errors import ModelTrainingError
from ffpred.evaluation.splits import chronological_folds
from ffpred.training.data import training_data_from_frame
from ffpred.training.svr import (
    MANUAL_FEATURE_COLUMNS,
    SvrConfig,
    candidate_configs,
    select_config,
    select_manual_features,
)
from tests.test_training import _frame


def test_chronological_folds_never_split_or_reverse_periods() -> None:
    frame = pl.DataFrame(
        {
            "target_season": [2020, 2020, 2020, 2020, 2020, 2020],
            "target_week": [1, 1, 2, 2, 3, 3],
        }
    )

    for train, validation in chronological_folds(frame, 2):
        train_periods = {
            tuple(row)
            for row in frame[train].select("target_season", "target_week").iter_rows()
        }
        validation_periods = {
            tuple(row)
            for row in frame[validation]
            .select("target_season", "target_week")
            .iter_rows()
        }
        assert max(train_periods) < min(validation_periods)
        assert train_periods.isdisjoint(validation_periods)


def test_chronological_folds_reject_insufficient_periods() -> None:
    with pytest.raises(ModelTrainingError, match="distinct periods"):
        tuple(
            chronological_folds(
                pl.DataFrame({"target_season": [2025], "target_week": [1]}),
                2,
            )
        )


def test_svr_search_space_and_manual_features_are_stable() -> None:
    assert len(candidate_configs()) == 260
    data = training_data_from_frame(_frame(16, season=2020))

    selected = select_manual_features(data)

    assert selected.feature_names == MANUAL_FEATURE_COLUMNS
    assert selected.features.shape[1] == len(MANUAL_FEATURE_COLUMNS)


def test_select_config_uses_chronological_folds() -> None:
    data = training_data_from_frame(_frame(16, season=2020))
    candidates = (
        SvrConfig(c=0.25, kernel="linear"),
        SvrConfig(c=1.0, kernel="linear"),
    )

    assert select_config(data, candidates, folds=2) in candidates

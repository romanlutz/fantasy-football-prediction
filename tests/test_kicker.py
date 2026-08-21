"""Tests for the kicker prediction vertical."""

from pathlib import Path

import polars as pl
import pytest

from ffpred.acquisition.normalize import acquire_kicker_histories
from ffpred.datasets.builder import KickerDatasetBuildConfig, build_kicker_datasets
from ffpred.domain.identifiers import PlayerId, Season, Week
from ffpred.domain.models import GameKey, KickerGameStats
from ffpred.domain.scoring import (
    DEFAULT_KICKER_SCORING,
    KickerScoringConfig,
    kicker_fantasy_score,
)
from ffpred.features.kicker_builder import build_kicker_feature_frame
from ffpred.features.kicker_schema import FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from ffpred.providers.fakes import FakeProvider
from tests.factories import make_kicker_provider


def _provider() -> FakeProvider:
    return FakeProvider(
        player_stats=pl.DataFrame(
            [
                {
                    "player_id": "00-TEST",
                    "player_display_name": "Test Kicker",
                    "position": "K",
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "fg_made_0_19": 1,
                    "fg_made_20_29": 1,
                    "fg_made_30_39": 0,
                    "fg_made_40_49": 1,
                    "fg_made_50_59": 1,
                    "fg_made_60_": 0,
                    "fg_missed": 1,
                    "pat_made": 3,
                    "pat_missed": 1,
                }
            ]
        )
    )


def test_kicker_acquisition_groups_field_goals_by_distance_band() -> None:
    histories = acquire_kicker_histories([2014], provider=_provider())

    game = histories[PlayerId("00-TEST")].games[GameKey(Season(2014), Week(1))]
    assert game.stats.fg_made_0_39 == 2
    assert game.stats.fg_made_40_49 == 1
    assert game.stats.fg_made_50_plus == 1
    assert game.stats.fg_missed == 1
    assert game.stats.pat_made == 3
    assert game.stats.pat_missed == 1


def test_kicker_fantasy_score_default_config() -> None:
    stats = KickerGameStats(
        fg_made_0_39=2,
        fg_made_40_49=1,
        fg_made_50_plus=1,
        fg_missed=1,
        pat_made=3,
        pat_missed=1,
    )

    # 2*3 + 1*4 + 1*5 + 1*0 + 3*1 + 1*0 = 6 + 4 + 5 + 0 + 3 + 0 = 18
    assert kicker_fantasy_score(stats) == 18


def test_kicker_feature_frame_is_named_and_leakage_safe() -> None:
    histories = acquire_kicker_histories(
        (2020, 2021, 2022), provider=make_kicker_provider()
    )

    frame = build_kicker_feature_frame(histories)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.height > 0
    for row in frame.iter_rows(named=True):
        target = (row["target_season"], row["target_week"])
        assert (
            row["kicker_history_through_season"],
            row["kicker_history_through_week"],
        ) < target


def test_empty_kicker_histories_produce_empty_schema() -> None:
    frame = build_kicker_feature_frame({})

    assert frame.is_empty()
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_custom_kicker_scoring_changes_the_target() -> None:
    histories = acquire_kicker_histories(
        (2020, 2021, 2022), provider=make_kicker_provider()
    )
    flat_scoring = KickerScoringConfig(
        field_goal_0_39=3.0, field_goal_40_49=3.0, field_goal_50_plus=3.0
    )

    default_frame = build_kicker_feature_frame(histories)
    flat_frame = build_kicker_feature_frame(histories, scoring=flat_scoring)

    assert (
        default_frame["fantasy_points"].to_list()
        != flat_frame["fantasy_points"].to_list()
    )


def test_build_kicker_datasets_builds_reproducible_artifacts(tmp_path: Path) -> None:
    manifest = build_kicker_datasets(
        KickerDatasetBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            test_year=2022,
        ),
        provider=make_kicker_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert train.select(MODEL_FEATURE_COLUMNS).width == len(MODEL_FEATURE_COLUMNS)
    assert set(train["target_season"].to_list()) == {2021}
    assert set(test["target_season"].to_list()) == {2022}
    assert set(manifest.outputs) == {"train", "test"}


def test_default_kicker_scoring_matches_report_convention() -> None:
    assert DEFAULT_KICKER_SCORING.field_goal_0_39 == pytest.approx(3.0)
    assert DEFAULT_KICKER_SCORING.field_goal_50_plus > (
        DEFAULT_KICKER_SCORING.field_goal_0_39
    )

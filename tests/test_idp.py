"""Tests for the individual defensive player (IDP) prediction vertical."""

from pathlib import Path

import polars as pl
import pytest

from ffpred.acquisition.normalize import acquire_idp_histories
from ffpred.datasets.builder import IdpDatasetBuildConfig, build_idp_datasets
from ffpred.domain.identifiers import PlayerId, Season, Week
from ffpred.domain.models import GameKey, IdpGameStats
from ffpred.domain.scoring import (
    DEFAULT_IDP_SCORING,
    IdpScoringConfig,
    idp_fantasy_score,
)
from ffpred.features.idp_builder import build_idp_feature_frame
from ffpred.features.idp_schema import FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from ffpred.providers.fakes import FakeProvider
from tests.factories import make_idp_provider


def _provider() -> FakeProvider:
    return FakeProvider(
        player_stats=pl.DataFrame(
            [
                {
                    "player_id": "00-TEST",
                    "player_display_name": "Test Linebacker",
                    "position_group": "LB",
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "def_tackles_solo": 8,
                    "def_tackles_with_assist": 2,
                    "def_sacks": 1,
                    "def_interceptions": 1,
                    "def_pass_defended": 2,
                    "def_fumbles_forced": 1,
                    "def_tds": 1,
                },
                {
                    "player_id": "00-OFFENSE",
                    "player_display_name": "Test Wide Receiver",
                    "position_group": "WR",
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "def_tackles_solo": 0,
                    "def_tackles_with_assist": 0,
                    "def_sacks": 0,
                    "def_interceptions": 0,
                    "def_pass_defended": 0,
                    "def_fumbles_forced": 0,
                    "def_tds": 0,
                },
            ]
        )
    )


def test_idp_acquisition_maps_nflverse_fields() -> None:
    histories = acquire_idp_histories([2014], provider=_provider())

    history = histories[PlayerId("00-TEST")]
    game = history.games[GameKey(Season(2014), Week(1))]
    assert history.position_group == "LB"
    assert game.stats.solo_tackles == 8
    assert game.stats.assisted_tackles == 2
    assert game.stats.sacks == 1
    assert game.stats.interceptions == 1
    assert game.stats.passes_defended == 2
    assert game.stats.fumbles_forced == 1
    assert game.stats.touchdowns == 1


def test_idp_acquisition_excludes_non_defensive_position_groups() -> None:
    histories = acquire_idp_histories([2014], provider=_provider())

    assert PlayerId("00-OFFENSE") not in histories


def test_idp_fantasy_score_default_config() -> None:
    stats = IdpGameStats(
        solo_tackles=8,
        assisted_tackles=2,
        sacks=1,
        interceptions=1,
        passes_defended=2,
        fumbles_forced=1,
        touchdowns=1,
    )

    # 8*1 + 2*0.5 + 1*4 + 1*6 + 2*2 + 1*2 + 1*6 = 8+1+4+6+4+2+6 = 31
    assert idp_fantasy_score(stats, DEFAULT_IDP_SCORING) == pytest.approx(31.0)


def test_idp_feature_frame_is_named_and_leakage_safe() -> None:
    histories = acquire_idp_histories((2020, 2021, 2022), provider=make_idp_provider())

    frame = build_idp_feature_frame(histories)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.height > 0
    assert set(frame["position_group"].to_list()) == {"LB"}
    for row in frame.iter_rows(named=True):
        target = (row["target_season"], row["target_week"])
        assert (
            row["idp_history_through_season"],
            row["idp_history_through_week"],
        ) < target


def test_empty_idp_histories_produce_empty_schema() -> None:
    frame = build_idp_feature_frame({})

    assert frame.is_empty()
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_custom_idp_scoring_changes_the_target() -> None:
    histories = acquire_idp_histories((2020, 2021, 2022), provider=make_idp_provider())
    heavier_tackles = IdpScoringConfig(solo_tackle=2.0)

    default_frame = build_idp_feature_frame(histories)
    custom_frame = build_idp_feature_frame(histories, scoring=heavier_tackles)

    assert (
        default_frame["fantasy_points"].to_list()
        != custom_frame["fantasy_points"].to_list()
    )


def test_build_idp_datasets_builds_reproducible_artifacts(tmp_path: Path) -> None:
    manifest = build_idp_datasets(
        IdpDatasetBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            test_year=2022,
        ),
        provider=make_idp_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert train.select(MODEL_FEATURE_COLUMNS).width == len(MODEL_FEATURE_COLUMNS)
    assert set(train["target_season"].to_list()) == {2021}
    assert set(test["target_season"].to_list()) == {2022}
    assert set(manifest.outputs) == {"train", "test"}


def test_default_idp_build_config_uses_a_2010_history_floor() -> None:
    assert IdpDatasetBuildConfig().history_start == 2010

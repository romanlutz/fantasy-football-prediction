"""Tests for the RB/WR/TE receiving-position prediction vertical."""

from pathlib import Path

import polars as pl
import pytest

from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_receiving_histories,
)
from ffpred.datasets.builder import (
    ReceivingDatasetBuildConfig,
    build_receiving_datasets,
)
from ffpred.domain.identifiers import PlayerId, Season, TeamCode, Week
from ffpred.domain.models import GameKey, ReceivingGameStats
from ffpred.domain.scoring import (
    DEFAULT_RECEIVING_SCORING,
    FULL_PPR_RECEIVING_SCORING,
    HALF_PPR_RECEIVING_SCORING,
    ReceivingScoringConfig,
    receiving_fantasy_score,
)
from ffpred.features.receiving_builder import build_receiving_feature_frame
from ffpred.features.receiving_schema import FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from ffpred.providers.fakes import FakeProvider
from tests.factories import make_receiving_provider


def _provider() -> FakeProvider:
    return FakeProvider(
        player_stats=pl.DataFrame(
            [
                {
                    "player_id": "00-TEST",
                    "player_display_name": "Test Running Back",
                    "position": "RB",
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "team": "GB",
                    "opponent_team": "SEA",
                    "attempts": 0,
                    "carries": 15,
                    "rushing_yards": 80,
                    "rushing_tds": 1,
                    "rushing_2pt_conversions": 0,
                    "receptions": 4,
                    "targets": 5,
                    "receiving_yards": 30,
                    "receiving_tds": 1,
                    "receiving_2pt_conversions": 1,
                    "fumbles_total": 1,
                }
            ]
        ),
        schedules=pl.DataFrame(
            [
                {
                    "game_id": "2014_01_GB_SEA",
                    "gameday": "2014-09-04",
                    "home_team": "SEA",
                    "away_team": "GB",
                    "home_score": 36,
                    "away_score": 16,
                }
            ]
        ),
    )


def test_receiving_acquisition_maps_nflverse_fields() -> None:
    histories = acquire_receiving_histories([2014], provider=_provider())

    history = histories[PlayerId("00-TEST")]
    game = history.games[GameKey(Season(2014), Week(1))]
    assert history.position == "RB"
    assert game.context.opponent == TeamCode("SEA")
    assert game.stats.rushing_yards == 80
    assert game.stats.receptions == 4
    assert game.stats.receiving_two_point_made == 1
    assert game.stats.team_targets == 5
    assert game.stats.team_rushing_attempts == 15
    assert game.stats.team_offensive_plays == 15


def test_receiving_acquisition_filters_by_requested_positions() -> None:
    histories = acquire_receiving_histories([2014], ("WR",), provider=_provider())

    assert histories == {}


def test_receiving_fantasy_score_reception_weight_controls_ppr_format() -> None:
    stats = ReceivingGameStats(
        rushing_attempts=15,
        rushing_yards=80,
        rushing_touchdowns=1,
        rushing_two_point_made=0,
        receptions=4,
        targets=5,
        receiving_yards=30,
        receiving_touchdowns=0,
        receiving_two_point_made=0,
        fumbles=0,
    )

    standard = receiving_fantasy_score(stats, DEFAULT_RECEIVING_SCORING)
    half_ppr = receiving_fantasy_score(stats, HALF_PPR_RECEIVING_SCORING)
    full_ppr = receiving_fantasy_score(stats, FULL_PPR_RECEIVING_SCORING)

    assert half_ppr == pytest.approx(standard + 4 * 0.5)
    assert full_ppr == pytest.approx(standard + 4 * 1.0)


def test_receiving_feature_frame_is_named_and_leakage_safe() -> None:
    provider = make_receiving_provider()
    receiving_histories = acquire_receiving_histories(
        (2020, 2021, 2022), provider=provider
    )
    defense_histories = acquire_defense_histories((2020, 2021, 2022), provider=provider)

    frame = build_receiving_feature_frame(receiving_histories, defense_histories)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.height > 0
    assert set(frame["position"].to_list()) <= {"RB", "WR"}
    row = frame.filter(
        (pl.col("player_id") == "00-WR")
        & (pl.col("target_season") == 2020)
        & (pl.col("target_week") == 2)
    ).row(0, named=True)
    assert row["receiving_last_1_target_share"] == pytest.approx(0.5)
    assert row["receiving_last_1_carry_share"] == pytest.approx(10 / 23)
    assert row["receiving_last_1_team_pass_attempts"] == 30
    assert row["receiving_last_1_team_rushing_attempts"] == 23
    assert row["receiving_last_1_team_offensive_plays"] == 53
    assert row["receiving_last_1_team_pass_rate"] == pytest.approx(30 / 53)
    for row in frame.iter_rows(named=True):
        target = (row["target_season"], row["target_week"])
        assert (
            row["receiving_history_through_season"],
            row["receiving_history_through_week"],
        ) < target
        assert (
            row["defense_history_through_season"],
            row["defense_history_through_week"],
        ) < target


def test_empty_receiving_histories_produce_empty_schema() -> None:
    frame = build_receiving_feature_frame({}, {})

    assert frame.is_empty()
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_custom_receiving_scoring_changes_the_target() -> None:
    provider = make_receiving_provider()
    receiving_histories = acquire_receiving_histories(
        (2020, 2021, 2022), provider=provider
    )
    defense_histories = acquire_defense_histories((2020, 2021, 2022), provider=provider)
    ppr = ReceivingScoringConfig(reception=1.0)

    standard_frame = build_receiving_feature_frame(
        receiving_histories, defense_histories
    )
    ppr_frame = build_receiving_feature_frame(
        receiving_histories, defense_histories, scoring=ppr
    )

    assert (
        standard_frame["fantasy_points"].to_list()
        != ppr_frame["fantasy_points"].to_list()
    )


def test_build_receiving_datasets_can_filter_to_one_position(tmp_path: Path) -> None:
    manifest = build_receiving_datasets(
        ReceivingDatasetBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            test_year=2022,
            positions=("WR",),
        ),
        provider=make_receiving_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert train.select(MODEL_FEATURE_COLUMNS).width == len(MODEL_FEATURE_COLUMNS)
    assert set(train["position"].to_list()) == {"WR"}
    assert set(train["target_season"].to_list()) == {2021}
    assert set(test["target_season"].to_list()) == {2022}
    assert set(manifest.outputs) == {"train", "test"}

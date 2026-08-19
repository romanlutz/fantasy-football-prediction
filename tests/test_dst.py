"""Tests for the team defense/special-teams (D/ST) prediction vertical."""

from pathlib import Path

import polars as pl
import pytest

from ffpred.acquisition.normalize import acquire_dst_histories
from ffpred.datasets.builder import DstDatasetBuildConfig, build_dst_datasets
from ffpred.domain.identifiers import Season, TeamCode, Week
from ffpred.domain.models import GameKey
from ffpred.domain.scoring import (
    DEFAULT_DST_SCORING,
    DstScoringConfig,
    points_allowed_score,
)
from ffpred.features.dst_builder import build_dst_feature_frame
from ffpred.features.dst_schema import FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from ffpred.providers.fakes import FakeProvider
from tests.factories import make_dst_provider


def _provider() -> FakeProvider:
    return FakeProvider(
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
        team_stats=pl.DataFrame(
            [
                {
                    "season": 2014,
                    "week": 1,
                    "season_type": "REG",
                    "game_id": "2014_01_GB_SEA",
                    "team": "SEA",
                    "opponent_team": "GB",
                    "def_sacks": 4,
                    "def_interceptions": 2,
                    "def_tds": 1,
                    "def_safeties": 0,
                    "fumble_recovery_opp": 1,
                    "def_punt_blocks": 0,
                    "def_pat_blocks": 0,
                    "def_fg_blocks": 1,
                }
            ]
        ),
    )


def test_dst_acquisition_attributes_stats_to_their_own_team() -> None:
    histories = acquire_dst_histories([2014], provider=_provider())

    game = histories[TeamCode("SEA")].games[GameKey(Season(2014), Week(1))]
    assert game.stats.points_allowed == 16
    assert game.stats.sacks == 4
    assert game.stats.interceptions == 2
    assert game.stats.touchdowns == 1
    assert game.stats.fumble_recoveries == 1
    assert game.stats.blocked_kicks == 1


def test_points_allowed_score_uses_ascending_tiers() -> None:
    assert points_allowed_score(0) == 10
    assert points_allowed_score(6) == 7
    assert points_allowed_score(13) == 4
    assert points_allowed_score(34) == -1
    assert points_allowed_score(35) == -4
    assert points_allowed_score(100) == -4


def test_dst_feature_frame_matches_vectorized_and_scalar_scoring() -> None:
    histories = acquire_dst_histories((2020, 2021, 2022), provider=make_dst_provider())

    frame = build_dst_feature_frame(histories)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.height > 0
    for team in histories.values():
        for key, game in team.games.items():
            if key.season == 2020:
                continue  # first season has no prior history to roll from
            row = frame.filter(
                (pl.col("team") == team.team)
                & (pl.col("target_season") == key.season)
                & (pl.col("target_week") == key.week)
            )
            if row.is_empty():
                continue
            expected = (
                game.stats.sacks * DEFAULT_DST_SCORING.sack
                + game.stats.interceptions * DEFAULT_DST_SCORING.interception
                + game.stats.fumble_recoveries * DEFAULT_DST_SCORING.fumble_recovery
                + game.stats.touchdowns * DEFAULT_DST_SCORING.touchdown
                + game.stats.safeties * DEFAULT_DST_SCORING.safety
                + game.stats.blocked_kicks * DEFAULT_DST_SCORING.blocked_kick
                + points_allowed_score(game.stats.points_allowed)
            )
            assert row.row(0, named=True)["fantasy_points"] == pytest.approx(expected)


def test_dst_feature_frame_is_leakage_safe() -> None:
    histories = acquire_dst_histories((2020, 2021, 2022), provider=make_dst_provider())

    frame = build_dst_feature_frame(histories)

    for row in frame.iter_rows(named=True):
        target = (row["target_season"], row["target_week"])
        assert (
            row["dst_history_through_season"],
            row["dst_history_through_week"],
        ) < target


def test_empty_dst_histories_produce_empty_schema() -> None:
    frame = build_dst_feature_frame({})

    assert frame.is_empty()
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_custom_dst_scoring_changes_the_target() -> None:
    histories = acquire_dst_histories((2020, 2021, 2022), provider=make_dst_provider())
    lenient = DstScoringConfig(sack=5.0)

    default_frame = build_dst_feature_frame(histories)
    custom_frame = build_dst_feature_frame(histories, scoring=lenient)

    assert (
        default_frame["fantasy_points"].to_list()
        != custom_frame["fantasy_points"].to_list()
    )


def test_build_dst_datasets_builds_reproducible_artifacts(tmp_path: Path) -> None:
    manifest = build_dst_datasets(
        DstDatasetBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            test_year=2022,
        ),
        provider=make_dst_provider(),
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    test = pl.read_parquet(tmp_path / "test.parquet")
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert train.select(MODEL_FEATURE_COLUMNS).width == len(MODEL_FEATURE_COLUMNS)
    assert set(train["target_season"].to_list()) == {2021}
    assert set(test["target_season"].to_list()) == {2022}
    assert set(manifest.outputs) == {"train", "test"}

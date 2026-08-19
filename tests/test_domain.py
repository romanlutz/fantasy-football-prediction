from pathlib import Path

import pytest

from ffpred.config import Settings
from ffpred.domain.models import QuarterbackGameStats
from ffpred.domain.scoring import ScoringConfig, fantasy_score
from ffpred.errors import ConfigurationError


def test_default_scoring_reproduces_standard_score() -> None:
    stats = QuarterbackGameStats(
        passing_attempts=30,
        passing_yards=250,
        passing_touchdowns=2,
        passing_interceptions=1,
        passing_two_point_attempts=1,
        passing_two_point_made=1,
        rushing_attempts=3,
        rushing_yards=20,
        rushing_touchdowns=1,
        rushing_two_point_attempts=0,
        rushing_two_point_made=0,
        fumbles=1,
    )

    assert fantasy_score(stats) == 24


def test_scoring_configuration_is_extensible() -> None:
    stats = QuarterbackGameStats(
        passing_attempts=1,
        passing_yards=100,
        passing_touchdowns=0,
        passing_interceptions=0,
        passing_two_point_attempts=0,
        passing_two_point_made=0,
        rushing_attempts=0,
        rushing_yards=0,
        rushing_touchdowns=0,
        rushing_two_point_attempts=0,
        rushing_two_point_made=0,
        fumbles=0,
    )

    assert fantasy_score(stats, ScoringConfig(passing_yards_per_point=20)) == 5


def test_settings_reject_invalid_season_ranges() -> None:
    with pytest.raises(ConfigurationError):
        Settings(history_start=2014, train_start=2010, test_year=2014)


def test_settings_load_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFPRED_OUTPUT_DIR", "artifacts")
    monkeypatch.setenv("FFPRED_TEST_YEAR", "2025")

    settings = Settings.from_env()

    assert settings.output_dir == Path("artifacts")
    assert settings.test_year == 2025

import polars as pl
import pytest

from ffpred.acquisition.coaching import (
    acquire_season_head_coaches,
    head_coach_changed,
)
from ffpred.domain.identifiers import Season, TeamCode
from ffpred.errors import SchemaValidationError
from ffpred.providers.fakes import FakeProvider


def _provider(rows: list[dict[str, object]]) -> FakeProvider:
    return FakeProvider(schedules=pl.DataFrame(rows))


def test_season_head_coaches_uses_earliest_game_per_team_per_season() -> None:
    provider = _provider(
        [
            {
                "season": 2013,
                "week": 1,
                "home_team": "SEA",
                "away_team": "GB",
                "home_coach": "Pete Carroll",
                "away_coach": "Mike McCarthy",
            },
            {
                "season": 2014,
                "week": 1,
                "home_team": "SEA",
                "away_team": "GB",
                "home_coach": "Pete Carroll",
                "away_coach": "Mike McCarthy",
            },
            # A later week must not override the season's first-game coach.
            {
                "season": 2014,
                "week": 8,
                "home_team": "SEA",
                "away_team": "GB",
                "home_coach": "Pete Carroll",
                "away_coach": "Interim Coach",
            },
        ]
    )

    coaches = acquire_season_head_coaches([2013, 2014], provider=provider)

    assert coaches[(TeamCode("SEA"), Season(2014))] == "Pete Carroll"
    assert coaches[(TeamCode("GB"), Season(2014))] == "Mike McCarthy"


def test_head_coach_changed_compares_consecutive_seasons() -> None:
    provider = _provider(
        [
            {
                "season": 2013,
                "week": 1,
                "home_team": "SEA",
                "away_team": "GB",
                "home_coach": "Pete Carroll",
                "away_coach": "Old Coach",
            },
            {
                "season": 2014,
                "week": 1,
                "home_team": "SEA",
                "away_team": "GB",
                "home_coach": "Pete Carroll",
                "away_coach": "New Coach",
            },
        ]
    )

    coaches = acquire_season_head_coaches([2013, 2014], provider=provider)

    assert head_coach_changed(coaches, TeamCode("GB"), Season(2014)) is True
    assert head_coach_changed(coaches, TeamCode("SEA"), Season(2014)) is False
    # 2013 has no prior season in the requested range to compare against.
    assert head_coach_changed(coaches, TeamCode("SEA"), Season(2013)) is False
    # Unknown team/season combinations must not raise.
    assert head_coach_changed(coaches, TeamCode("NEW"), Season(2014)) is False


def test_relocated_team_codes_are_normalized_to_current_franchise() -> None:
    provider = _provider(
        [
            {
                "season": 2015,
                "week": 1,
                "home_team": "NE",
                "away_team": "STL",
                "home_coach": "Bill Belichick",
                "away_coach": "Jeff Fisher",
            },
            {
                "season": 2016,
                "week": 1,
                "home_team": "NE",
                "away_team": "LA",
                "home_coach": "Bill Belichick",
                "away_coach": "Jeff Fisher",
            },
        ]
    )

    coaches = acquire_season_head_coaches([2015, 2016], provider=provider)

    assert coaches[(TeamCode("LA"), Season(2015))] == "Jeff Fisher"
    assert head_coach_changed(coaches, TeamCode("LA"), Season(2016)) is False


def test_missing_coach_columns_raise_schema_validation_error() -> None:
    provider = _provider(
        [
            {
                "season": 2014,
                "week": 1,
                "home_team": "SEA",
                "away_team": "GB",
            }
        ]
    )

    with pytest.raises(SchemaValidationError):
        acquire_season_head_coaches([2014], provider=provider)

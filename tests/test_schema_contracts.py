import polars as pl
import pytest

from ffpred.acquisition.contracts import INJURY_REPORTS_CONTRACT, SCHEDULES_CONTRACT
from ffpred.acquisition.schema import validate_frame
from ffpred.errors import SchemaValidationError


def test_schedule_contract_accepts_provider_shape() -> None:
    frame = pl.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI"],
            "gameday": ["2025-09-04"],
            "home_team": ["PHI"],
            "away_team": ["DAL"],
            "home_score": [24],
            "away_score": [20],
        }
    )

    assert validate_frame(frame, SCHEDULES_CONTRACT) is frame


def test_schedule_contract_reports_all_schema_problems() -> None:
    frame = pl.DataFrame(
        {
            "game_id": [None],
            "gameday": [1],
            "home_team": ["PHI"],
            "away_team": ["DAL"],
            "home_score": [24],
        }
    )

    with pytest.raises(SchemaValidationError) as error:
        validate_frame(frame, SCHEDULES_CONTRACT)

    assert "missing column 'away_score'" in str(error.value)
    assert "'gameday' is Int64" in str(error.value)
    assert "'game_id' contains null values" in str(error.value)


def test_injury_reports_contract_accepts_provider_shape() -> None:
    frame = pl.DataFrame(
        {
            "season": [2023],
            "week": [5],
            "game_type": ["REG"],
            "team": ["KC"],
            "gsis_id": ["00-QB"],
            "full_name": ["Test Quarterback"],
            "report_status": ["Out"],
            "report_primary_injury": ["Ankle"],
        }
    )

    assert validate_frame(frame, INJURY_REPORTS_CONTRACT) is frame


def test_injury_reports_contract_allows_null_status_and_gsis_id() -> None:
    """Most injuries rows have a null gsis_id or report_status (e.g. a
    practice-only listing, or a row for a non-roster staff member); those are
    filtered out downstream in acquire_injury_reports, not rejected here.
    """
    frame = pl.DataFrame(
        {
            "season": [2023],
            "week": [5],
            "game_type": ["REG"],
            "team": ["KC"],
            "gsis_id": [None],
            "full_name": [None],
            "report_status": [None],
            "report_primary_injury": [None],
        }
    )

    assert validate_frame(frame, INJURY_REPORTS_CONTRACT) is frame

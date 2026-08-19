import polars as pl
import pytest

from ffpred.acquisition.contracts import SCHEDULES_CONTRACT
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

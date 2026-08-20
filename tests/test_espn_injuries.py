import polars as pl
import pytest

from ffpred.acquisition.schema import ColumnKind, FrameContract
from ffpred.domain.identifiers import PlayerId, TeamCode
from ffpred.domain.models import InjuryStatus
from ffpred.errors import SchemaValidationError
from ffpred.providers.espn import (
    build_espn_injury_snapshot,
    espn_injury_snapshot_frame,
    fetch_espn_injuries,
)


def _payload(**injury_overrides: object) -> dict[str, object]:
    injury = {
        "status": "Questionable",
        "shortComment": "Limited in practice.",
        "longComment": None,
        "date": "2025-09-10T12:00Z",
        "details": {"type": "Hamstring"},
        "athlete": {
            "displayName": "Test Player",
            "links": [
                {"href": "https://www.espn.com/nfl/player/_/id/999123/test-player"}
            ],
            "team": {"abbreviation": "KC"},
        },
    }
    injury.update(injury_overrides)
    return {
        "injuries": [
            {
                "id": "22",
                "displayName": "Test Team",
                "abbreviation": "KC",
                "injuries": [injury],
            }
        ]
    }


def test_fetch_espn_injuries_parses_a_well_formed_record() -> None:
    records = fetch_espn_injuries(_payload)

    assert len(records) == 1
    record = records[0]
    assert record.espn_athlete_id == "999123"
    assert record.player_name == "Test Player"
    assert record.team == TeamCode("KC")
    assert record.status_text == "Questionable"
    assert record.body_part == "Hamstring"
    assert record.comment == "Limited in practice."
    assert record.reported_at == "2025-09-10T12:00Z"


def test_fetch_espn_injuries_falls_back_to_team_abbreviation() -> None:
    """When an individual athlete's team block is missing, fall back to the
    enclosing team entry's own abbreviation rather than dropping the record.
    """
    payload = _payload(
        athlete={
            "displayName": "No Team Block",
            "links": [
                {"href": "https://www.espn.com/nfl/player/_/id/555/no-team-block"}
            ],
        }
    )

    records = fetch_espn_injuries(lambda: payload)

    assert len(records) == 1
    assert records[0].team == TeamCode("KC")


def test_fetch_espn_injuries_skips_records_missing_athlete_id_or_status() -> None:
    no_id_payload = _payload(
        athlete={"displayName": "No ID", "links": [], "team": {"abbreviation": "KC"}}
    )
    no_status_payload = _payload(status=None)

    assert fetch_espn_injuries(lambda: no_id_payload) == ()
    assert fetch_espn_injuries(lambda: no_status_payload) == ()


def test_fetch_espn_injuries_handles_empty_or_malformed_payload() -> None:
    assert fetch_espn_injuries(lambda: {}) == ()
    assert fetch_espn_injuries(lambda: {"injuries": []}) == ()


def _players_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["00-TEST"],
            "espn_id": ["999123"],
        }
    )


def test_build_espn_injury_snapshot_crosswalks_and_maps_status() -> None:
    records = fetch_espn_injuries(_payload)

    snapshots = build_espn_injury_snapshot(records, _players_frame())

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.player_id == PlayerId("00-TEST")
    assert snapshot.status is InjuryStatus.QUESTIONABLE
    assert snapshot.team == TeamCode("KC")
    assert snapshot.body_part == "Hamstring"


def test_build_espn_injury_snapshot_skips_unmapped_status() -> None:
    records = fetch_espn_injuries(lambda: _payload(status="Suspension"))

    assert build_espn_injury_snapshot(records, _players_frame()) == ()


def test_build_espn_injury_snapshot_skips_unmatched_athletes() -> None:
    records = fetch_espn_injuries(_payload)
    empty_players = pl.DataFrame({"gsis_id": [], "espn_id": []})

    assert build_espn_injury_snapshot(records, empty_players) == ()


def test_build_espn_injury_snapshot_requires_crosswalk_columns() -> None:
    records = fetch_espn_injuries(_payload)
    players_missing_espn_id = pl.DataFrame({"gsis_id": ["00-TEST"]})

    with pytest.raises(SchemaValidationError):
        build_espn_injury_snapshot(records, players_missing_espn_id)


def test_espn_injury_snapshot_frame_is_empty_with_named_schema_for_no_snapshots() -> (
    None
):
    frame = espn_injury_snapshot_frame(())

    assert frame.is_empty()
    assert "status" in frame.columns


def test_espn_injury_snapshot_frame_flattens_snapshots() -> None:
    records = fetch_espn_injuries(_payload)
    snapshots = build_espn_injury_snapshot(records, _players_frame())

    frame = espn_injury_snapshot_frame(snapshots)

    assert frame.height == 1
    assert frame["status"][0] == "Questionable"
    assert frame["player_id"][0] == "00-TEST"


def test_players_espn_crosswalk_contract_shape() -> None:
    """Documents the columns build_espn_injury_snapshot requires from the
    players frame, independent of the full PLAYERS_CONTRACT used elsewhere.
    """
    contract = FrameContract(
        name="players_espn_crosswalk",
        columns={"gsis_id": ColumnKind.TEXT, "espn_id": ColumnKind.TEXT},
    )
    assert contract.columns == {"gsis_id": ColumnKind.TEXT, "espn_id": ColumnKind.TEXT}

"""Live checks against ESPN's public (unofficial) injuries endpoint.

Kept separate from tests/live/test_nflreadpy_live.py: ESPN is an entirely
different, unofficial third-party service, not part of the nflverse
ecosystem those tests exercise.
"""

import pytest

from ffpred.providers.espn import build_espn_injury_snapshot, fetch_espn_injuries
from ffpred.providers.nflreadpy import NflReadPyProvider


@pytest.mark.live
def test_live_espn_injuries_endpoint_returns_parseable_records() -> None:
    """ESPN's endpoint is undocumented and may change or become unavailable
    without notice; this only asserts it still returns at least one
    record with a recognized designation, not any particular count or
    player, since injury reports vary week to week.
    """
    records = fetch_espn_injuries()

    assert records
    assert all(record.espn_athlete_id for record in records)
    assert all(record.status_text for record in records)


@pytest.mark.live
def test_live_espn_injury_snapshot_crosswalks_to_known_players() -> None:
    """At least some of today's ESPN-reported athletes should crosswalk to
    a GSIS-identified player via load_players()'s espn_id column, proving
    the join key is still valid against current nflverse player data.
    """
    provider = NflReadPyProvider(cache_mode="filesystem")
    records = fetch_espn_injuries()
    players = provider.load_players()

    snapshots = build_espn_injury_snapshot(records, players)

    assert snapshots
    assert all(snapshot.player_id for snapshot in snapshots)

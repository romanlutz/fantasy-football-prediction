import pytest

from ffpred.acquisition.contracts import (
    PBP_CONTRACT,
    PLAYER_STATS_CONTRACT,
    PLAYERS_CONTRACT,
    SCHEDULES_CONTRACT,
    TEAM_STATS_CONTRACT,
)
from ffpred.acquisition.schema import validate_frame
from ffpred.providers.nflreadpy import NflReadPyProvider

COMPLETED_SEASON = 2025


@pytest.mark.live
def test_live_nflreadpy_core_contracts() -> None:
    provider = NflReadPyProvider()

    schedules = provider.load_schedules((COMPLETED_SEASON,))
    players = provider.load_players()
    player_stats = provider.load_player_stats((COMPLETED_SEASON,))
    team_stats = provider.load_team_stats((COMPLETED_SEASON,))

    assert not validate_frame(schedules, SCHEDULES_CONTRACT).is_empty()
    assert not validate_frame(players, PLAYERS_CONTRACT).is_empty()
    assert not validate_frame(player_stats, PLAYER_STATS_CONTRACT).is_empty()
    assert not validate_frame(team_stats, TEAM_STATS_CONTRACT).is_empty()


@pytest.mark.live
@pytest.mark.live_slow
def test_live_nflreadpy_play_by_play_contract() -> None:
    frame = NflReadPyProvider().load_pbp(COMPLETED_SEASON)

    assert not validate_frame(frame, PBP_CONTRACT).is_empty()

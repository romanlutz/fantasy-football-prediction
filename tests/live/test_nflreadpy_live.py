import pytest

from ffpred.acquisition.contracts import (
    DST_TEAM_STATS_CONTRACT,
    IDP_PLAYER_STATS_CONTRACT,
    INJURY_REPORTS_CONTRACT,
    KICKER_PLAYER_STATS_CONTRACT,
    PBP_CONTRACT,
    PLAYER_STATS_CONTRACT,
    PLAYERS_CONTRACT,
    RECEIVING_PLAYER_STATS_CONTRACT,
    SCHEDULES_CONTRACT,
    TEAM_STATS_CONTRACT,
)
from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_dst_histories,
    acquire_idp_histories,
    acquire_injury_reports,
    acquire_kicker_histories,
    acquire_quarterback_histories,
    acquire_receiving_histories,
)
from ffpred.acquisition.schema import validate_frame
from ffpred.providers.nflreadpy import NflReadPyProvider

COMPLETED_SEASON = 2025
#: Latest season nflverse's injury-report source covers; it was retired
#: after the 2024 season with no replacement announced.
LAST_INJURY_REPORT_SEASON = 2024


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
def test_live_nflreadpy_position_contracts() -> None:
    """Validate the position-specific frame contracts added for D/ST, kicker,
    RB/WR/TE, and IDP against the same real player_stats/team_stats frames
    the core contract test already downloads.
    """
    provider = NflReadPyProvider(cache_mode="filesystem")

    player_stats = provider.load_player_stats((COMPLETED_SEASON,))
    team_stats = provider.load_team_stats((COMPLETED_SEASON,))

    assert not validate_frame(team_stats, DST_TEAM_STATS_CONTRACT).is_empty()
    assert not validate_frame(player_stats, KICKER_PLAYER_STATS_CONTRACT).is_empty()
    assert not validate_frame(player_stats, RECEIVING_PLAYER_STATS_CONTRACT).is_empty()
    assert not validate_frame(player_stats, IDP_PLAYER_STATS_CONTRACT).is_empty()


@pytest.mark.live
@pytest.mark.live_slow
def test_live_nflreadpy_play_by_play_contract() -> None:
    frame = NflReadPyProvider().load_pbp(COMPLETED_SEASON)

    assert not validate_frame(frame, PBP_CONTRACT).is_empty()


@pytest.mark.live
@pytest.mark.live_slow
def test_live_normalized_acquisition() -> None:
    provider = NflReadPyProvider(cache_mode="filesystem")

    quarterbacks = acquire_quarterback_histories(
        (COMPLETED_SEASON,),
        provider=provider,
    )
    defenses = acquire_defense_histories(
        (COMPLETED_SEASON,),
        provider=provider,
    )

    assert quarterbacks
    assert defenses
    assert all(
        game.key.season == COMPLETED_SEASON
        for history in quarterbacks.values()
        for game in history.games.values()
    )


@pytest.mark.live
def test_live_new_position_acquisition() -> None:
    """Acquisition smoke test for the four positions added after the
    original QB/D-ST pipeline: D/ST-as-target, kicker, RB/WR/TE, and IDP.
    None of these need a play-by-play join, so unlike QB acquisition this
    does not require the live_slow tier.
    """
    provider = NflReadPyProvider(cache_mode="filesystem")

    team_dst = acquire_dst_histories((COMPLETED_SEASON,), provider=provider)
    kickers = acquire_kicker_histories((COMPLETED_SEASON,), provider=provider)
    receivers = acquire_receiving_histories((COMPLETED_SEASON,), provider=provider)
    idp = acquire_idp_histories((COMPLETED_SEASON,), provider=provider)

    assert team_dst
    assert kickers
    assert receivers
    assert idp
    assert all(
        game.key.season == COMPLETED_SEASON
        for history in team_dst.values()
        for game in history.games.values()
    )
    assert all(
        game.key.season == COMPLETED_SEASON
        for history in kickers.values()
        for game in history.games.values()
    )
    assert all(
        game.key.season == COMPLETED_SEASON
        for history in receivers.values()
        for game in history.games.values()
    )
    assert all(
        game.key.season == COMPLETED_SEASON
        for history in idp.values()
        for game in history.games.values()
    )
    assert {history.position for history in receivers.values()} <= {"RB", "WR", "TE"}
    assert {history.position_group for history in idp.values()} <= {"DL", "LB", "DB"}


@pytest.mark.live
def test_live_injury_reports_contract_and_acquisition() -> None:
    """Validate the injury-report contract and acquisition against the last
    season nflverse's injury source covers. Unlike the other live acquisition
    tests, this deliberately does not use COMPLETED_SEASON: nflverse's injury
    source was retired after the 2024 season, so a 2025 request would return
    no data at all by design (see acquire_injury_reports).
    """
    provider = NflReadPyProvider(cache_mode="filesystem")

    injuries = provider.load_injuries((LAST_INJURY_REPORT_SEASON,))
    assert not validate_frame(injuries, INJURY_REPORTS_CONTRACT).is_empty()

    histories = acquire_injury_reports((LAST_INJURY_REPORT_SEASON,), provider=provider)
    assert histories
    assert all(
        report.key.season == LAST_INJURY_REPORT_SEASON
        for history in histories.values()
        for report in history.reports.values()
    )


@pytest.mark.live
def test_live_injury_reports_outside_coverage_window_return_no_data() -> None:
    """acquire_injury_reports should degrade gracefully -- not raise -- once
    seasons roll past nflverse's injury-source retirement after 2024.
    """
    histories = acquire_injury_reports(
        (COMPLETED_SEASON,), provider=NflReadPyProvider(cache_mode="filesystem")
    )

    assert histories == {}

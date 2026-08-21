from datetime import date

import pytest

from ffpred.domain.identifiers import GameId, PlayerId, Season, TeamCode, Week
from ffpred.domain.models import (
    GameContext,
    GameKey,
    InjuryHistory,
    InjuryReport,
    InjuryStatus,
    PlayerProfile,
    QuarterbackGame,
    QuarterbackGameStats,
    QuarterbackHistory,
    ReceivingGame,
    ReceivingGameStats,
    ReceivingHistory,
)
from ffpred.evaluation.injury_impact import (
    build_quarterback_injury_impact,
    build_receiving_injury_impact,
    injury_impact_frame,
)


def test_injury_status_orders_by_severity() -> None:
    assert InjuryStatus.QUESTIONABLE < InjuryStatus.DOUBTFUL < InjuryStatus.OUT
    assert InjuryStatus.NONE < InjuryStatus.QUESTIONABLE


def _qb_stats(**overrides: float) -> QuarterbackGameStats:
    base = {
        "passing_attempts": 0.0,
        "passing_yards": 0.0,
        "passing_touchdowns": 0.0,
        "passing_interceptions": 0.0,
        "passing_two_point_attempts": 0.0,
        "passing_two_point_made": 0.0,
        "rushing_attempts": 0.0,
        "rushing_yards": 0.0,
        "rushing_touchdowns": 0.0,
        "rushing_two_point_attempts": 0.0,
        "rushing_two_point_made": 0.0,
        "fumbles": 0.0,
    }
    base.update(overrides)
    return QuarterbackGameStats(**base)


def _context() -> GameContext:
    return GameContext(
        game_id=GameId("2023_01_KC_DEN"),
        game_date=date(2023, 9, 10),
        home_team=TeamCode("KC"),
        away_team=TeamCode("DEN"),
        team=TeamCode("KC"),
        opponent=TeamCode("DEN"),
    )


def _quarterback_history() -> QuarterbackHistory:
    """A quarterback who plays weeks 1 and 2, misses week 3 entirely, then
    returns for a much worse week 4 -- weeks 3 (Out) and 4 (Questionable)
    both carry an injury report.
    """
    profile = PlayerProfile(
        player_id=PlayerId("00-QB"),
        name="Test Quarterback",
        birth_date=date(1990, 1, 1),
        rookie_season=Season(2015),
    )
    history = QuarterbackHistory(profile=profile)
    for week, passing_yards, passing_touchdowns in (
        (1, 300, 3),
        (2, 300, 3),
        (4, 50, 0),
    ):
        key = GameKey(Season(2023), Week(week))
        history.games[key] = QuarterbackGame(
            key=key,
            context=_context(),
            stats=_qb_stats(
                passing_yards=passing_yards, passing_touchdowns=passing_touchdowns
            ),
        )
    return history


def _injury_history() -> InjuryHistory:
    injury = InjuryHistory(player_id=PlayerId("00-QB"), name="Test Quarterback")
    injury.reports[GameKey(Season(2023), Week(3))] = InjuryReport(
        key=GameKey(Season(2023), Week(3)),
        team=TeamCode("KC"),
        status=InjuryStatus.OUT,
        primary_injury="Ankle",
    )
    injury.reports[GameKey(Season(2023), Week(4))] = InjuryReport(
        key=GameKey(Season(2023), Week(4)),
        team=TeamCode("KC"),
        status=InjuryStatus.QUESTIONABLE,
        primary_injury="Ankle",
    )
    return injury


def test_quarterback_injury_impact_reports_missed_and_played_weeks() -> None:
    profile_id = PlayerId("00-QB")
    events = build_quarterback_injury_impact(
        {profile_id: _quarterback_history()},
        {profile_id: _injury_history()},
        trailing_window=4,
    )

    assert len(events) == 2
    missed, returned = events

    assert missed.key == GameKey(Season(2023), Week(3))
    assert missed.status is InjuryStatus.OUT
    assert missed.played is False
    assert missed.games_missed_since_last_played == 0
    assert missed.actual_fantasy_points is None
    # Standard scoring: 300/25 + 3*4 = 24 points/game for weeks 1-2.
    assert missed.pace_fantasy_points == pytest.approx(24.0)
    assert missed.delta_vs_pace is None

    assert returned.key == GameKey(Season(2023), Week(4))
    assert returned.status is InjuryStatus.QUESTIONABLE
    assert returned.played is True
    # Week 3 was the only week missed before this one.
    assert returned.games_missed_since_last_played == 1
    assert returned.pace_fantasy_points == pytest.approx(24.0)
    assert returned.actual_fantasy_points == pytest.approx(2.0)
    assert returned.delta_vs_pace == pytest.approx(2.0 - 24.0)


def test_quarterback_injury_impact_skips_players_with_no_reports_or_no_history() -> (
    None
):
    profile_id = PlayerId("00-QB")
    history = _quarterback_history()

    # No injury history at all for this player.
    assert build_quarterback_injury_impact({profile_id: history}, {}) == ()

    # An InjuryHistory entry exists for the target player but carries no
    # reports at all (defensive: acquire_injury_reports never actually
    # produces such an entry, since it only creates one alongside a report).
    empty_injury = InjuryHistory(player_id=profile_id, name="Test Quarterback")
    assert (
        build_quarterback_injury_impact(
            {profile_id: history}, {profile_id: empty_injury}
        )
        == ()
    )

    # An injury history exists for a player with no acquired game history.
    other_id = PlayerId("00-OTHER")
    other_injury = InjuryHistory(player_id=other_id, name="Ghost")
    other_injury.reports[GameKey(Season(2023), Week(1))] = InjuryReport(
        key=GameKey(Season(2023), Week(1)),
        team=TeamCode("KC"),
        status=InjuryStatus.OUT,
        primary_injury="Knee",
    )
    assert (
        build_quarterback_injury_impact({profile_id: history}, {other_id: other_injury})
        == ()
    )


def test_quarterback_injury_impact_has_no_pace_before_any_game_played() -> None:
    """A player reported injured before ever appearing in the acquired
    history (e.g. hurt before a debut) has no prior played game to average,
    so pace is None rather than raising.
    """
    player_id = PlayerId("00-ROOKIE")
    profile = PlayerProfile(
        player_id=player_id,
        name="Rookie QB",
        birth_date=date(2001, 1, 1),
        rookie_season=Season(2023),
    )
    history = QuarterbackHistory(profile=profile)
    injury = InjuryHistory(player_id=player_id, name="Rookie QB")
    injury.reports[GameKey(Season(2023), Week(1))] = InjuryReport(
        key=GameKey(Season(2023), Week(1)),
        team=TeamCode("KC"),
        status=InjuryStatus.OUT,
        primary_injury="Hamstring",
    )

    events = build_quarterback_injury_impact({player_id: history}, {player_id: injury})

    assert len(events) == 1
    assert events[0].pace_fantasy_points is None
    assert events[0].played is False


def _receiving_history() -> ReceivingHistory:
    history = ReceivingHistory(
        player_id=PlayerId("00-WR"), name="Test Receiver", position="WR"
    )
    for week, receiving_yards, receiving_touchdowns in ((1, 100, 1), (2, 100, 1)):
        key = GameKey(Season(2023), Week(week))
        history.games[key] = ReceivingGame(
            key=key,
            context=_context(),
            stats=ReceivingGameStats(
                rushing_attempts=0,
                rushing_yards=0,
                rushing_touchdowns=0,
                rushing_two_point_made=0,
                receptions=6,
                targets=8,
                receiving_yards=receiving_yards,
                receiving_touchdowns=receiving_touchdowns,
                receiving_two_point_made=0,
                fumbles=0,
            ),
        )
    return history


def test_receiving_injury_impact_uses_receiving_scoring() -> None:
    player_id = PlayerId("00-WR")
    injury = InjuryHistory(player_id=player_id, name="Test Receiver")
    injury.reports[GameKey(Season(2023), Week(3))] = InjuryReport(
        key=GameKey(Season(2023), Week(3)),
        team=TeamCode("KC"),
        status=InjuryStatus.DOUBTFUL,
        primary_injury="Hamstring",
    )

    events = build_receiving_injury_impact(
        {player_id: _receiving_history()}, {player_id: injury}
    )

    assert len(events) == 1
    event = events[0]
    assert event.position == "WR"
    assert event.status is InjuryStatus.DOUBTFUL
    assert event.played is False
    # Standard (non-PPR) scoring: 100/10 + 1*6 = 16 points/game for weeks 1-2.
    assert event.pace_fantasy_points == pytest.approx(16.0)


def test_receiving_injury_impact_skips_players_with_no_reports_or_no_history() -> None:
    player_id = PlayerId("00-WR")
    history = _receiving_history()

    assert build_receiving_injury_impact({player_id: history}, {}) == ()

    empty_injury = InjuryHistory(player_id=player_id, name="Test Receiver")
    assert (
        build_receiving_injury_impact({player_id: history}, {player_id: empty_injury})
        == ()
    )


def test_injury_impact_frame_is_empty_with_named_schema_for_no_events() -> None:
    frame = injury_impact_frame(())

    assert frame.is_empty()
    assert "delta_vs_pace" in frame.columns


def test_injury_impact_frame_flattens_events() -> None:
    profile_id = PlayerId("00-QB")
    events = build_quarterback_injury_impact(
        {profile_id: _quarterback_history()},
        {profile_id: _injury_history()},
    )

    frame = injury_impact_frame(events)

    assert frame.height == 2
    assert frame["report_status"].to_list() == ["Out", "Questionable"]
    assert frame["played"].to_list() == [False, True]

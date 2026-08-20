"""Injury-report impact analysis: was a player hurt, and how did it affect them.

For every week a player appears on the official injury report, this compares
what they were "on pace for" (their trailing average fantasy score over
recent games they actually played) against what actually happened that week:
whether they sat out entirely, and if they played, how their score compared
to their pace. This lets a prediction miss be checked against whether an
injury -- not a modeling failure -- explains it, and lets a player's eventual
return be compared against their pre-injury trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from ffpred.domain.identifiers import PlayerId, TeamCode, Week
from ffpred.domain.models import (
    GameKey,
    InjuryHistory,
    InjuryReport,
    InjuryStatus,
    QuarterbackHistory,
    ReceivingHistory,
)
from ffpred.domain.scoring import (
    DEFAULT_RECEIVING_SCORING,
    DEFAULT_SCORING,
    ReceivingScoringConfig,
    ScoringConfig,
    fantasy_score,
    receiving_fantasy_score,
)

#: Default number of a player's own prior played games averaged to compute
#: "pace" -- what they were on track for heading into a reported week.
DEFAULT_TRAILING_WINDOW = 4
#: Regular-season week range scanned for missed-game streaks. 18 covers every
#: modern-era regular season (17 games plus one bye) without over-scanning.
DEFAULT_MAX_WEEK = 18


@dataclass(frozen=True, slots=True, kw_only=True)
class InjuryImpactEvent:
    """One (player, reported week): pace, outcome, and games-missed streak."""

    player_id: PlayerId
    player_name: str
    position: str
    key: GameKey
    team: TeamCode
    status: InjuryStatus
    primary_injury: str | None
    played: bool
    games_missed_since_last_played: int
    pace_fantasy_points: float | None
    actual_fantasy_points: float | None
    delta_vs_pace: float | None


def _trailing_pace(
    sorted_keys: list[GameKey],
    scores: Mapping[GameKey, float],
    cutoff: GameKey,
    window: int,
) -> float | None:
    """Average score over up to ``window`` games strictly before ``cutoff``."""
    prior = [key for key in sorted_keys if key < cutoff][-window:]
    if not prior:
        return None
    return sum(scores[key] for key in prior) / len(prior)


def _build_events_for_player(  # noqa: PLR0913
    player_id: PlayerId,
    name: str,
    position: str,
    scores: Mapping[GameKey, float],
    reports: Mapping[GameKey, InjuryReport],
    *,
    trailing_window: int,
    max_week: int,
) -> list[InjuryImpactEvent]:
    sorted_keys = sorted(scores)
    seasons = sorted({key.season for key in {*scores, *reports}})

    events: list[InjuryImpactEvent] = []
    for season in seasons:
        missed_streak = 0
        for week in range(1, max_week + 1):
            key = GameKey(season, Week(week))
            played = key in scores
            report = reports.get(key)
            if report is not None:
                pace = _trailing_pace(sorted_keys, scores, key, trailing_window)
                actual = scores.get(key) if played else None
                events.append(
                    InjuryImpactEvent(
                        player_id=player_id,
                        player_name=name,
                        position=position,
                        key=key,
                        team=report.team,
                        status=report.status,
                        primary_injury=report.primary_injury,
                        played=played,
                        games_missed_since_last_played=missed_streak,
                        pace_fantasy_points=pace,
                        actual_fantasy_points=actual,
                        delta_vs_pace=(
                            actual - pace
                            if actual is not None and pace is not None
                            else None
                        ),
                    )
                )
            missed_streak = 0 if played else missed_streak + 1
    return events


def build_quarterback_injury_impact(
    quarterback_histories: Mapping[PlayerId, QuarterbackHistory],
    injury_histories: Mapping[PlayerId, InjuryHistory],
    *,
    scoring: ScoringConfig = DEFAULT_SCORING,
    trailing_window: int = DEFAULT_TRAILING_WINDOW,
    max_week: int = DEFAULT_MAX_WEEK,
) -> tuple[InjuryImpactEvent, ...]:
    """Build injury-impact events for every quarterback with a reported week."""
    events: list[InjuryImpactEvent] = []
    for player_id, injury_history in injury_histories.items():
        if not injury_history.reports:
            continue
        history = quarterback_histories.get(player_id)
        if history is None:
            continue
        scores = {
            key: fantasy_score(game.stats, scoring)
            for key, game in history.games.items()
        }
        events.extend(
            _build_events_for_player(
                player_id,
                history.profile.name,
                "QB",
                scores,
                injury_history.reports,
                trailing_window=trailing_window,
                max_week=max_week,
            )
        )
    return _sorted_events(events)


def build_receiving_injury_impact(
    receiving_histories: Mapping[PlayerId, ReceivingHistory],
    injury_histories: Mapping[PlayerId, InjuryHistory],
    *,
    scoring: ReceivingScoringConfig = DEFAULT_RECEIVING_SCORING,
    trailing_window: int = DEFAULT_TRAILING_WINDOW,
    max_week: int = DEFAULT_MAX_WEEK,
) -> tuple[InjuryImpactEvent, ...]:
    """Build injury-impact events for every RB/WR/TE with a reported week."""
    events: list[InjuryImpactEvent] = []
    for player_id, injury_history in injury_histories.items():
        if not injury_history.reports:
            continue
        history = receiving_histories.get(player_id)
        if history is None:
            continue
        scores = {
            key: receiving_fantasy_score(game.stats, scoring)
            for key, game in history.games.items()
        }
        events.extend(
            _build_events_for_player(
                player_id,
                history.name,
                history.position,
                scores,
                injury_history.reports,
                trailing_window=trailing_window,
                max_week=max_week,
            )
        )
    return _sorted_events(events)


def _sorted_events(
    events: list[InjuryImpactEvent],
) -> tuple[InjuryImpactEvent, ...]:
    return tuple(
        sorted(
            events,
            key=lambda event: (event.player_name, event.key.season, event.key.week),
        )
    )


def injury_impact_frame(events: tuple[InjuryImpactEvent, ...]) -> pl.DataFrame:
    """Flatten injury-impact events into a Polars frame for export/reporting."""
    rows = [
        {
            "player_id": event.player_id,
            "player_name": event.player_name,
            "position": event.position,
            "season": event.key.season,
            "week": event.key.week,
            "team": event.team,
            "report_status": event.status.name.title(),
            "primary_injury": event.primary_injury,
            "played": event.played,
            "games_missed_since_last_played": event.games_missed_since_last_played,
            "pace_fantasy_points": event.pace_fantasy_points,
            "actual_fantasy_points": event.actual_fantasy_points,
            "delta_vs_pace": event.delta_vs_pace,
        }
        for event in events
    ]
    schema = {
        "player_id": pl.String,
        "player_name": pl.String,
        "position": pl.String,
        "season": pl.Int64,
        "week": pl.Int64,
        "team": pl.String,
        "report_status": pl.String,
        "primary_injury": pl.String,
        "played": pl.Boolean,
        "games_missed_since_last_played": pl.Int64,
        "pace_fantasy_points": pl.Float64,
        "actual_fantasy_points": pl.Float64,
        "delta_vs_pace": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)

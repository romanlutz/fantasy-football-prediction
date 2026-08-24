"""Head-coach continuity signal derived from nflverse schedules.

Every other prior-season feature in this project (team pass rate, target
share, etc. -- see ``ffpred.features.all_positions``) is a numeric average
that cannot express a discontinuity like a new head coach installing a new
offensive scheme. nflverse schedules separately expose ``home_coach``/
``away_coach`` (head coach name, sourced from Pro-Football-Reference), which
is enough open data to at least flag *that* a team's coaching staff changed,
even without offensive-coordinator/play-caller detail (not available in
nflverse's open data).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ffpred.acquisition.contracts import DEFAULT_SEASONS, normalize_team_code
from ffpred.acquisition.schema import ColumnKind, FrameContract, validate_frame
from ffpred.domain.identifiers import Season, TeamCode
from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider

SCHEDULE_COACHES_CONTRACT = FrameContract(
    name="schedule_coaches",
    columns={
        "season": ColumnKind.INTEGER,
        "week": ColumnKind.INTEGER,
        "home_team": ColumnKind.TEXT,
        "away_team": ColumnKind.TEXT,
        "home_coach": ColumnKind.TEXT,
        "away_coach": ColumnKind.TEXT,
    },
    non_null=frozenset(
        {"season", "week", "home_team", "away_team", "home_coach", "away_coach"}
    ),
)

#: Each team-season mapped to its head coach of record.
SeasonHeadCoaches = Mapping[tuple[TeamCode, Season], str]


def acquire_season_head_coaches(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    *,
    provider: NflDataProvider | None = None,
) -> dict[tuple[TeamCode, Season], str]:
    """Map each team-season to the head coach of its earliest played game.

    The coach recorded in a team's earliest game of a season is treated as
    that season's coach of record, so an offseason hire is visible even if a
    team later fires someone mid-season. Pass the result to
    ``head_coach_changed`` to compare consecutive seasons.
    """
    season_list = sorted(set(seasons))
    provider = provider or NflReadPyProvider()
    frame = validate_frame(
        provider.load_schedules(season_list), SCHEDULE_COACHES_CONTRACT
    )

    earliest: dict[tuple[TeamCode, Season], tuple[int, str]] = {}
    for row in frame.iter_rows(named=True):
        season = Season(int(row["season"]))
        week = int(row["week"])
        for team_column, coach_column in (
            ("home_team", "home_coach"),
            ("away_team", "away_coach"),
        ):
            team = TeamCode(normalize_team_code(row[team_column]))
            coach = row[coach_column]
            key = (team, season)
            if key not in earliest or week < earliest[key][0]:
                earliest[key] = (week, coach)
    return {key: coach for key, (_, coach) in earliest.items()}


def head_coach_changed(
    season_coaches: SeasonHeadCoaches, team: TeamCode, season: Season
) -> bool:
    """Return whether ``team``'s head coach differs from the prior season's.

    Returns ``False`` when either season's coach of record is unknown (e.g.
    the earliest season in a requested range has no prior season to compare
    against), since the absence of a comparison is not evidence of a change.
    """
    current = season_coaches.get((team, season))
    previous = season_coaches.get((team, Season(int(season) - 1)))
    if current is None or previous is None:
        return False
    return current != previous

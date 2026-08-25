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

import polars as pl

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
    # home_coach/away_coach are deliberately excluded from non_null: nflverse
    # only populates them once a game has been played (they come from the
    # final box score via Pro-Football-Reference), so a schedules frame that
    # spans a not-yet-played season -- the normal case when forecasting the
    # current season -- has null coaches for its future weeks. Those rows
    # are skipped rather than rejected; see season_head_coaches_from_schedules.
    non_null=frozenset({"season", "week", "home_team", "away_team"}),
)

#: Each team-season mapped to its head coach of record.
SeasonHeadCoaches = Mapping[tuple[TeamCode, Season], str]

#: Optional schedule columns this module reads. A schedules frame lacking
#: either column carries no coaching-continuity signal at all (rather than
#: a validation failure), since most callers -- and every existing test
#: fixture in this project -- build minimal schedules without them.
COACH_COLUMNS = ("home_coach", "away_coach")


def season_head_coaches_from_schedules(
    schedules: pl.DataFrame,
) -> dict[tuple[TeamCode, Season], str]:
    """Map each team-season to the head coach of its earliest *known* game.

    The coach recorded in a team's earliest game with a known coach is
    treated as that season's coach of record, so an offseason hire is
    visible even if a team later fires someone mid-season. Rows with a null
    coach are skipped rather than recorded -- nflverse only populates
    ``home_coach``/``away_coach`` once a game has been played, so a
    schedules frame spanning a not-yet-played season (the normal case when
    forecasting the current season) has null coaches for its future weeks.
    A team-season with no played games yet simply has no entry, and
    ``head_coach_changed`` already treats an unknown season as "no change."
    Pass the result to ``head_coach_changed`` to compare consecutive
    seasons. Raises if ``schedules`` is missing the ``home_coach``/
    ``away_coach`` columns entirely; callers that may receive a schedules
    frame without those optional columns (e.g. minimal test fixtures)
    should check for them first, as ``ffpred.features.all_positions`` does.
    """
    frame = validate_frame(schedules, SCHEDULE_COACHES_CONTRACT)

    earliest: dict[tuple[TeamCode, Season], tuple[int, str]] = {}
    for row in frame.iter_rows(named=True):
        season = Season(int(row["season"]))
        week = int(row["week"])
        for team_column, coach_column in (
            ("home_team", "home_coach"),
            ("away_team", "away_coach"),
        ):
            coach = row[coach_column]
            if coach is None:
                continue
            team = TeamCode(normalize_team_code(row[team_column]))
            key = (team, season)
            if key not in earliest or week < earliest[key][0]:
                earliest[key] = (week, coach)
    return {key: coach for key, (_, coach) in earliest.items()}


def acquire_season_head_coaches(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    *,
    provider: NflDataProvider | None = None,
) -> dict[tuple[TeamCode, Season], str]:
    """Load schedules for ``seasons`` and map each team-season to its coach.

    A thin provider-fetching wrapper around
    ``season_head_coaches_from_schedules`` for callers that have not already
    loaded a schedules frame themselves.
    """
    season_list = sorted(set(seasons))
    provider = provider or NflReadPyProvider()
    return season_head_coaches_from_schedules(provider.load_schedules(season_list))


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


def team_head_coach_change_frame(schedules: pl.DataFrame) -> pl.DataFrame:
    """Build a tidy ``team``/``season``/``team_head_coach_changed`` frame.

    Vectorized-friendly counterpart to ``head_coach_changed`` for callers
    that join this signal into a feature table by team and season (see
    ``ffpred.features.all_positions``) rather than looking up one team at a
    time. Returns an empty frame with the right schema if ``schedules``
    lacks the optional coach columns, rather than raising, so callers can
    outer-join it and rely on ``fill_null(0.0)`` for "no known change."
    """
    schema = {
        "team": pl.String,
        "season": pl.Int64,
        "team_head_coach_changed": pl.Float64,
    }
    if not set(COACH_COLUMNS).issubset(schedules.columns):
        return pl.DataFrame(schema=schema)

    coaches = season_head_coaches_from_schedules(schedules)
    rows = [
        {
            "team": str(team),
            "season": int(season),
            "team_head_coach_changed": float(head_coach_changed(coaches, team, season)),
        }
        for (team, season) in coaches
    ]
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

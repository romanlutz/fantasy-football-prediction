"""ESPN's public injuries endpoint: an unofficial supplement to nflverse's
official injury-report source, which nflverse retired after the 2024 season
(see acquisition.contracts.INJURY_REPORTS_MAX_SEASON).

This module is deliberately kept separate from the acquisition/features/
datasets pipeline used elsewhere in this package. Unlike nflverse's injury
data, ESPN's endpoint only exposes a **current** snapshot -- there is no
reliable, documented way to request a specific past season/week from it (the
community-documented historical path returned 404 when verified against this
package's own tests) -- so it cannot backfill leakage-safe historical
training features the way ``acquire_injury_reports`` does. It is meant for
live/operational use: seeing today's report to sanity-check a prediction
made for an upcoming game, for seasons nflverse's own source no longer
covers.

The endpoint itself is undocumented and unofficial. It may change shape or
become unavailable without notice; every parsing step here fails soft
(skips a record) rather than raising, so a partially-changed response
degrades gracefully instead of crashing callers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl
import requests

from ffpred.acquisition.contracts import normalize_team_code
from ffpred.acquisition.schema import ColumnKind, FrameContract, validate_frame
from ffpred.domain.identifiers import PlayerId, TeamCode
from ffpred.domain.models import InjuryStatus

ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
)
#: Maps ESPN's free-text injury designations to this package's InjuryStatus.
#: ESPN's vocabulary is broader than nflverse's three official game
#: designations (Questionable/Doubtful/Out): statuses with no comparable
#: game-availability meaning (e.g. a roster/transaction state rather than an
#: injury designation) are intentionally left unmapped, and such records are
#: skipped rather than guessed at.
ESPN_STATUS_TO_INJURY_STATUS: Mapping[str, InjuryStatus] = {
    "Questionable": InjuryStatus.QUESTIONABLE,
    "Doubtful": InjuryStatus.DOUBTFUL,
    "Out": InjuryStatus.OUT,
    "Injured Reserve": InjuryStatus.OUT,
    "Out For Season": InjuryStatus.OUT,
    "Day-To-Day": InjuryStatus.QUESTIONABLE,
}
_ATHLETE_ID_PATTERN = re.compile(r"/id/(\d+)/")
#: Columns required to crosswalk ESPN's numeric athlete IDs to this
#: package's GSIS-based PlayerId, read from the same players frame
#: acquisition.normalize already validates via PLAYERS_CONTRACT.
PLAYERS_ESPN_CROSSWALK_CONTRACT = FrameContract(
    name="players_espn_crosswalk",
    columns={
        "gsis_id": ColumnKind.TEXT,
        "espn_id": ColumnKind.TEXT,
    },
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EspnInjuryRecord:
    """One player's current injury-report entry as reported by ESPN."""

    espn_athlete_id: str
    player_name: str
    team: TeamCode
    status_text: str
    body_part: str | None
    comment: str | None
    reported_at: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EspnInjurySnapshot:
    """One player's current ESPN injury status, crosswalked to a PlayerId."""

    player_id: PlayerId
    espn_athlete_id: str
    player_name: str
    team: TeamCode
    status: InjuryStatus
    body_part: str | None
    comment: str | None
    reported_at: str | None


def _default_fetch() -> dict[str, Any]:
    response = requests.get(ESPN_INJURIES_URL, timeout=15)
    response.raise_for_status()
    return response.json()


def _athlete_id(athlete: Mapping[str, Any]) -> str | None:
    """ESPN's athlete payload does not always populate a top-level ``id``
    field directly; the numeric ID is reliably present in every player-card
    link's URL instead (verified live), so that is used as the fallback.
    """
    raw_id = athlete.get("id")
    if raw_id:
        return str(raw_id)
    for link in athlete.get("links") or ():
        match = _ATHLETE_ID_PATTERN.search(link.get("href", ""))
        if match:
            return str(match.group(1))
    return None


def _parse_injury_entry(
    team_entry: Mapping[str, Any],
    injury_entry: Mapping[str, Any],
) -> EspnInjuryRecord | None:
    athlete = injury_entry.get("athlete") or {}
    athlete_id = _athlete_id(athlete)
    status_text = injury_entry.get("status")
    if not athlete_id or not status_text:
        return None

    team_info = athlete.get("team") or {}
    team_code = team_info.get("abbreviation") or team_entry.get("abbreviation")
    if not team_code:
        return None

    details = injury_entry.get("details") or {}
    return EspnInjuryRecord(
        espn_athlete_id=athlete_id,
        player_name=str(athlete.get("displayName") or ""),
        team=TeamCode(normalize_team_code(str(team_code))),
        status_text=str(status_text),
        body_part=details.get("type"),
        comment=injury_entry.get("shortComment") or injury_entry.get("longComment"),
        reported_at=injury_entry.get("date"),
    )


def fetch_espn_injuries(
    fetch: Callable[[], dict[str, Any]] = _default_fetch,
) -> tuple[EspnInjuryRecord, ...]:
    """Fetch and parse ESPN's current league-wide injury report.

    ``fetch`` defaults to a real HTTP GET against ``ESPN_INJURIES_URL`` but
    can be replaced (e.g. in tests) with any zero-argument callable
    returning the same JSON shape, without a network call.
    """
    payload = fetch()
    records: list[EspnInjuryRecord] = []
    for team_entry in payload.get("injuries") or ():
        for injury_entry in team_entry.get("injuries") or ():
            record = _parse_injury_entry(team_entry, injury_entry)
            if record is not None:
                records.append(record)
    return tuple(records)


def build_espn_injury_snapshot(
    records: tuple[EspnInjuryRecord, ...],
    players: pl.DataFrame,
) -> tuple[EspnInjurySnapshot, ...]:
    """Crosswalk ESPN injury records to this package's PlayerId via GSIS.

    Records for statuses outside ``ESPN_STATUS_TO_INJURY_STATUS`` (e.g. a
    roster/transaction state with no game-availability meaning) or for
    athletes with no matching ``espn_id`` in ``players`` are skipped rather
    than guessed at.
    """
    validate_frame(players, PLAYERS_ESPN_CROSSWALK_CONTRACT)
    espn_to_gsis = {
        row["espn_id"]: row["gsis_id"]
        for row in players.iter_rows(named=True)
        if row.get("espn_id") and row.get("gsis_id")
    }

    snapshots: list[EspnInjurySnapshot] = []
    for record in records:
        status = ESPN_STATUS_TO_INJURY_STATUS.get(record.status_text)
        gsis_id = espn_to_gsis.get(record.espn_athlete_id)
        if status is None or gsis_id is None:
            continue
        snapshots.append(
            EspnInjurySnapshot(
                player_id=PlayerId(gsis_id),
                espn_athlete_id=record.espn_athlete_id,
                player_name=record.player_name,
                team=record.team,
                status=status,
                body_part=record.body_part,
                comment=record.comment,
                reported_at=record.reported_at,
            )
        )
    return tuple(snapshots)


def espn_injury_snapshot_frame(
    snapshots: tuple[EspnInjurySnapshot, ...],
) -> pl.DataFrame:
    """Flatten ESPN injury snapshots into a Polars frame for export/reporting."""
    schema = {
        "player_id": pl.String,
        "espn_athlete_id": pl.String,
        "player_name": pl.String,
        "team": pl.String,
        "status": pl.String,
        "body_part": pl.String,
        "comment": pl.String,
        "reported_at": pl.String,
    }
    if not snapshots:
        return pl.DataFrame(schema=schema)
    rows = [
        {
            "player_id": snapshot.player_id,
            "espn_athlete_id": snapshot.espn_athlete_id,
            "player_name": snapshot.player_name,
            "team": snapshot.team,
            "status": snapshot.status.name.title(),
            "body_part": snapshot.body_part,
            "comment": snapshot.comment,
            "reported_at": snapshot.reported_at,
        }
        for snapshot in snapshots
    ]
    return pl.DataFrame(rows, schema=schema)

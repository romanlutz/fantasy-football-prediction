"""Re-extract only factual listings from the pinned Wikipedia revisions."""

from __future__ import annotations

import hashlib
import re

import polars as pl
import requests

from ffpred.errors import DataAcquisitionError
from ffpred.study.club_rosters import FACTS, REVISIONS

ACTIVE = {
    "quarterbacks": "QB",
    "running backs": "RB",
    "wide receivers": "WR",
    "tight ends": "TE",
}
INACTIVE = {"reserve lists": "reserve", "practice squad": "practice"}
POSITIONS = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}


def parse_roster(revision: int, text: str) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    clean = re.sub(r"<!--.*?-->|<noinclude>.*?</noinclude>", "", text, flags=re.S)
    clean = re.sub(r"\{\{abbrlink\|[^{}]*}}", "", clean, flags=re.I)
    section = ""
    seen = set()
    for line in clean.splitlines():
        heading = re.match(r"^\s*\|\s*([^=]+?)\s*=", line)
        if heading:
            section = heading[1].strip().lower()
            seen.add(section)
        if section not in ACTIVE and section not in INACTIVE:
            continue
        for fields in re.findall(r"\{\{\s*NFLplayer\s*\|([^{}]+)}}", line, re.I):
            args = [p.strip() for p in fields.split("|")]
            if section in ACTIVE:
                rows.append((revision, args[1], ACTIVE[section], "active"))
            else:
                pos = next((POSITIONS[p] for p in args[2:] if p in POSITIONS), None)
                if pos:
                    rows.append((revision, args[1], pos, INACTIVE[section]))
    if not set(ACTIVE) <= seen or {r[2] for r in rows if r[3] == "active"} != set(
        ACTIVE.values()
    ):
        raise DataAcquisitionError(f"Incomplete active offense in revision {revision}")
    return rows


def write_facts(rows: list[tuple[int, str, str, str]]) -> None:
    pl.DataFrame(
        rows, schema=["revision", "name", "position", "section"], orient="row"
    ).unique().sort("revision", "position", "name", "section").write_csv(FACTS)


def main() -> None:
    index = pl.read_csv(REVISIONS)
    rows = []
    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "ffpred historical roster study (https://github.com/romanlutz/fantasy-football-prediction)"
        )
        for chunk in index.iter_slices(25):
            want = {r["revision"]: r for r in chunk.iter_rows(named=True)}
            response = session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "prop": "revisions",
                    "rvslots": "main",
                    "rvprop": "ids|timestamp|content",
                    "revids": "|".join(str(i) for i in want),
                },
                timeout=60,
            )
            response.raise_for_status()
            for page in response.json()["query"]["pages"]:
                for rev in page["revisions"]:
                    source = want.pop(rev["revid"])
                    text = rev["slots"]["main"]["content"]
                    if (
                        hashlib.sha256(text.encode()).hexdigest() != source["sha256"]
                        or rev["timestamp"] != source["stamp"]
                    ):
                        raise DataAcquisitionError("Club revision fingerprint mismatch")
                    rows.extend(parse_roster(rev["revid"], text))
            if want:
                raise DataAcquisitionError("Missing pinned roster revisions")
    write_facts(rows)


if __name__ == "__main__":
    main()

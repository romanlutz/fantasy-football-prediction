"""Reconcile 2014 eligibility and scores against the original Game Center boxes."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile

import polars as pl
import requests

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import PLAYER_WEEK

PAPER_YEAR = 2014
PASSER_THRESHOLD = 5
ARCHIVE_URL = (
    "https://files.pythonhosted.org/packages/12/67/"
    "f83dbc4a7bcff8c72b5f6f2f4c3f84f9f1e9c5f5afe13eca152e64dd3449/nflgame-1.2.15.tar.gz"
)
ARCHIVE_HASH = "283dfed6b1e2f3c44fcce9a148bba355b00fa8103cf4cae92414b919186c46a0"


def historical_passers() -> pl.DataFrame:
    response = requests.get(ARCHIVE_URL, timeout=120)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != ARCHIVE_HASH:
        raise DataAcquisitionError("Historical nflgame archive checksum mismatch")
    return read_gamecenter(response.content)


def read_gamecenter(content: bytes) -> pl.DataFrame:
    rows = []
    common_teams: dict[str, set[str]] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(content))) as archive:

        def read(path: str) -> bytes:
            file = archive.extractfile("nflgame-1.2.15/nflgame/" + path)
            if file is None:
                raise DataAcquisitionError(f"Missing archived file {path}")
            return file.read()

        for eid, game in json.loads(read("schedule.json"))["games"]:
            if game["year"] != PAPER_YEAR or game["season_type"] != "REG":
                continue
            box = json.loads(gzip.decompress(read(f"gamecenter-json/{eid}.json.gz")))[
                eid
            ]
            teams = {box[side]["abbr"] for side in ("home", "away")}
            for side in ("home", "away"):
                stats = box[side]["stats"]
                for player, passing in stats["passing"].items():
                    if passing["att"] < PASSER_THRESHOLD:
                        continue
                    common_teams.setdefault(player, teams.copy()).intersection_update(
                        teams
                    )
                    rush = stats.get("rushing", {}).get(player, {})
                    fumbles = stats.get("fumbles", {}).get(player, {}).get("tot", 0)
                    points = (
                        passing["yds"] / 25
                        + passing["tds"] * 4
                        - passing["ints"] * 2
                        + rush.get("yds", 0) / 10
                        + rush.get("tds", 0) * 6
                        + 2 * (passing["twoptm"] + rush.get("twoptm", 0))
                        - 2 * fumbles
                    )
                    rows.append(
                        {
                            "season": PAPER_YEAR,
                            "week": game["week"],
                            "player_id": player,
                            "paper_points": round(points, 2),
                            "paper_name": passing["name"],
                        }
                    )
    eligible = [player for player, teams in common_teams.items() if teams]
    return pl.DataFrame(rows).filter(pl.col("player_id").is_in(eligible))


def reconciliation(
    panel: pl.DataFrame, boxes: pl.DataFrame, legacy: pl.DataFrame
) -> pl.DataFrame:
    c = pl.col
    paper_or_new = legacy.join(
        panel.filter(c("season") == PAPER_YEAR).select(
            *PLAYER_WEEK, "position", "team", "name", "roster_week"
        ),
        on=PLAYER_WEEK,
        how="full",
        coalesce=True,
        validate="1:1",
    )
    return (
        paper_or_new.join(
            boxes.select(
                *PLAYER_WEEK,
                "box_position",
                "player_display_name",
                "actual_team",
                "attempts",
                "points",
                "legacy_qb_rescore",
                "fumbles_total",
                "fumbles_lost_total",
                "receiving_yards",
                "receiving_tds",
            ),
            on=PLAYER_WEEK,
            how="left",
            validate="1:1",
        )
        .with_columns(
            c("points").fill_null(0),
            ((c("box_position") == "QB") & (c("attempts") >= PASSER_THRESHOLD))
            .fill_null(False)
            .alias("box_qb_eligible"),
            pl.coalesce("name", "player_display_name", "paper_name").alias("name"),
            pl.when(c("paper_points").is_null())
            .then(pl.lit("new_roster_candidate"))
            .when(c("position").is_not_null())
            .then(pl.lit("both"))
            .when(c("week") == 1)
            .then(pl.lit("no_prior_in_season_roster"))
            .otherwise(pl.lit("not_on_prior_eligible_roster"))
            .alias("membership"),
        )
        .with_columns(
            (c("legacy_qb_rescore") - c("paper_points"))
            .round(2)
            .alias("source_revision_delta"),
            (2 * (c("fumbles_total") - c("fumbles_lost_total"))).alias(
                "fumble_rule_delta"
            ),
        )
        .with_columns(
            (c("points") - c("legacy_qb_rescore") - c("fumble_rule_delta"))
            .round(2)
            .alias("other_scoring_delta")
        )
        .sort("week", "player_id")
    )

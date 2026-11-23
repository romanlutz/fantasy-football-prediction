"""Committed pre-opener affiliations, not retrospective season roster labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import requests

from ffpred.study.panel import POSITIONS, TEAM_CODES
from ffpred.study.role_evidence import ENTRY

SOURCES = Path(__file__).with_name("opening_sources.csv")
ALIASES = {
    **TEAM_CODES,
    "GBP": "GB",
    "KCC": "KC",
    "LVR": "LV",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TBB": "TB",
    "SDC": "LAC",
    "RAM": "LA",
}
c = pl.col


def opening_editions(end_year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames = []
    records = []
    for row in (
        pl.read_csv(SOURCES).filter(c("season") <= end_year).iter_rows(named=True)
    ):
        url = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{row['path']}"
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        if row["path"].endswith("json"):
            raw = pl.DataFrame(
                [
                    {
                        "player_id": k,
                        "name": v.get("full_name"),
                        "position": v.get("position"),
                        "team": v.get("team"),
                        "listed": v.get("status") == "ACT",
                        "rookie": False,
                    }
                    for k, v in json.loads(res.content).items()
                    if v.get("status") in {"ACT", "RES", "SUS", "PUP"}
                ]
            )
        else:
            csv = pl.read_csv(res.content, infer_schema=False, null_values=["NA", ""])
            raw = csv.select(
                c("gsis_id").alias("player_id"),
                "name",
                c("position" if "position" in csv.columns else "pos").alias("position"),
                "team",
                (~c("team").is_in(["FA", "FA*"])).alias("listed"),
                (c("draft_year") == str(row["season"]))
                .fill_null(False)
                .alias("rookie"),
            )
        clean = (
            raw.drop_nulls(["player_id", "position", "team"])
            .with_columns(
                c("team").replace(ALIASES),
                c("position").replace({"HB": "RB", "FB": "RB"}),
            )
            .filter(
                c("position").is_in(POSITIONS)
                & c("player_id").str.contains(r"^00-\d{7}$")
            )
            .sort("name")
            .unique(["player_id", "team", "listed", "position"], keep="first")
        )
        clean = clean.filter(pl.len().over("player_id") == 1)
        frames.append(
            clean.with_columns(
                pl.lit(row["season"]).alias("season"),
                pl.lit(row["stamp"])
                .str.to_datetime(time_zone="UTC")
                .alias("edition_at"),
                pl.lit(row["path"].endswith("json")).alias("has_active_status"),
            )
        )
        records.append(
            {
                **row,
                "url": url,
                "sha256": hashlib.sha256(res.content).hexdigest(),
                "unique_ids": clean.height,
            }
        )
    return pl.concat(frames).sort("season", "player_id"), pl.DataFrame(records)


def opening_evidence(
    clocks: pl.DataFrame,
    seeds: pl.DataFrame,
    boxes: pl.DataFrame,
    editions: pl.DataFrame,
) -> pl.DataFrame:
    cutoff = clocks.filter(c("opener")).group_by("season").agg(c("cutoff").min())
    allowed = editions.join(cutoff, on="season").filter(
        (c("edition_at") <= c("cutoff"))
        & (c("edition_at") >= c("cutoff") - pl.duration(days=30))
    )
    # A fantasy crosswalk can retain retired players. Only admit a league listing
    # with pre-existing candidacy, a recent box, a rookie flag or explicit ACT.
    known = (
        pl.concat(
            [
                seeds.join(
                    clocks.join(cutoff, on=["season", "cutoff"])
                    .filter(c("opener"))
                    .select("game_id", "team", "season"),
                    on=["game_id", "team"],
                ).select("season", "player_id"),
                *(
                    boxes.select((c("season") + n).alias("season"), "player_id")
                    for n in (1, 2)
                ),
            ],
            how="vertical_relaxed",
        )
        .unique()
        .with_columns(pl.lit(True).alias("known"))
    )
    return (
        allowed.join(known, on=["season", "player_id"], how="left")
        .filter(c("known").fill_null(False) | c("rookie") | c("has_active_status"))
        .drop("known", "cutoff")
    )


def apply_editions(
    pool: pl.DataFrame,
    evidence: pl.DataFrame,
    transfers: pl.DataFrame,
    reports: pl.DataFrame,
) -> pl.DataFrame:
    ev = evidence.select(
        "season", "player_id", c("team").alias("edition_team"), "edition_at", "listed"
    )
    moves = transfers.select("player_id", "move_at").sort("move_at")
    rows = (
        pool.join(ev, on=["season", "player_id"], how="left")
        .join(reports.select(*ENTRY, "report_at"), on=ENTRY, how="left")
        .sort("cutoff")
        .join_asof(
            moves,
            left_on="cutoff",
            right_on="move_at",
            by="player_id",
            check_sortedness=False,
        )
    )
    fresh = (
        c("opener")
        & c("edition_at").is_not_null()
        & (
            pl.max_horizontal(
                "move_at", "depth_at", "identity_at", "report_at"
            ).is_null()
            | (
                c("edition_at")
                >= pl.max_horizontal("move_at", "depth_at", "identity_at", "report_at")
            )
        )
    )
    sanctioned = (c("team") == c("edition_team")) & c("listed")
    # Affiliation is evidence of team, not a starter or activation announcement.
    return (
        rows.filter(~fresh | sanctioned)
        .with_columns(
            (c("opener") & sanctioned).fill_null(False).alias("edition_confirmed"),
            pl.when(c("opener") & sanctioned)
            .then(pl.max_horizontal("identity_at", "edition_at"))
            .otherwise(c("identity_at"))
            .alias("identity_at"),
        )
        .drop("move_at", "report_at")
    )

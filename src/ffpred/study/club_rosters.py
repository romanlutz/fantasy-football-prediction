"""Exact-name identity links for the revision-pinned team roster facts."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.role_evidence import position

REVISIONS = Path(__file__).with_name("club_revisions.csv")
FACTS = Path(__file__).with_name("club_roster_facts.csv")
c = pl.col


def name_key(name: str) -> str:
    letters = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub("[^a-z0-9]", "", letters.lower())


def club_editions(
    end_year: int, players: pl.DataFrame, rosters: pl.DataFrame, draft: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    meta = pl.read_csv(REVISIONS).filter(c("season") <= end_year)
    facts = pl.read_csv(FACTS).join(meta, on="revision", validate="m:1")
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in (
        players.select(
            "gsis_id",
            "display_name",
            "first_name",
            "football_name",
            "common_first_name",
            "last_name",
            "suffix",
        )
        .drop_nulls("gsis_id")
        .iter_rows(named=True)
    ):
        names = {row["display_name"]}
        for first in (
            row["first_name"],
            row["common_first_name"],
            row["football_name"],
        ):
            if first and row["last_name"]:
                full = f"{first} {row['last_name']}"
                names.add(full)
                if row["suffix"]:
                    names.add(f"{full} {row['suffix']}")
        for name in filter(None, names):
            aliases[name_key(name)].add(row["gsis_id"])
    # Distinguish namesakes only by earlier rosters or the already-held draft.
    # Never prefer the ID that later played for the new club.
    prior = (
        rosters.filter(c("game_type") == "REG")
        .select("season", "gsis_id", position("position"))
        .drop_nulls()
        .unique()
    )
    eligible = pl.concat(
        [prior.with_columns(c("season") + n) for n in (1, 2)]
        + [draft.select("season", "gsis_id", position("position")).drop_nulls()],
        how="vertical_relaxed",
    )
    known: dict[tuple[int, str], set[str]] = defaultdict(set)
    for year, pid, pos in eligible.iter_rows():
        known[year, pos].add(pid)
    links: list[str | None] = []
    for name, year, pos in facts.select("name", "season", "position").iter_rows():
        choices = aliases.get(name_key(name), set())
        if len(choices) > 1:
            choices = choices & known[year, pos]
        links.append(next(iter(choices)) if len(choices) == 1 else None)
    linked = facts.with_columns(pl.Series("player_id", links, dtype=pl.String))
    audit = linked.group_by("revision").agg(
        pl.len().alias("listings"), c("player_id").count().alias("linked")
    )
    evidence = linked.drop_nulls("player_id").select(
        "player_id",
        "name",
        "position",
        "team",
        "season",
        (c("section") == "active").alias("listed"),
        c("stamp").str.to_datetime(time_zone="UTC").alias("edition_at"),
        pl.lit(False).alias("rookie"),
        pl.lit(True).alias("has_active_status"),
        pl.lit(True).alias("club_roster"),
    )
    if (
        meta.select("season", "team").is_duplicated().any()
        or meta.join(audit, on="revision", how="anti").height
    ):
        raise DataAcquisitionError("Missing or duplicated club roster edition")
    return evidence, meta.join(audit, on="revision").sort("season", "team")

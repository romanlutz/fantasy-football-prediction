"""Timestamped affiliation changes, not announcements of playing time."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.matchup_pool import SEEDS
from ffpred.study.panel import POSITIONS
from ffpred.study.role_evidence import ENTRY, SIDE

LEDGER = Path(__file__).with_name("transaction_news.csv")
c = pl.col
ARRIVALS = ("claim", "signing", "trade", "agreement")
CLUB_PLAYER = ["season", "team", "player_id"]


def available_at(value: str) -> datetime:
    if "T" not in value:
        following = date.fromisoformat(value) + timedelta(days=1)
        return datetime.combine(following, time.min, ZoneInfo("America/New_York"))
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise DataAcquisitionError("Transaction time must have an explicit offset")
    return stamp


def read_transactions(path: Path = LEDGER) -> pl.DataFrame:
    data = pl.read_csv(path, schema_overrides={"games": pl.Int64})
    suspension = c("kind") == "opening_suspension"
    if (
        data.drop_nulls(
            [
                "season",
                "player_id",
                "name",
                "position",
                "old_team",
                "kind",
                "public_by",
                "source",
            ]
        ).height
        != data.height
        or data.filter(
            ~c("position").is_in(POSITIONS)
            | ~c("source").str.starts_with("https://")
            | ~c("kind").is_in([*ARRIVALS, "release", "practice", "opening_suspension"])
            | (c("kind").is_in([*ARRIVALS, "practice"]) != c("new_team").is_not_null())
            | (c("old_team") == c("new_team"))
            | (suspension != c("games").is_not_null())
            | (c("games") < 1)
        ).height
    ):
        raise DataAcquisitionError("Invalid dated affiliation record")
    rows = data.with_columns(
        pl.Series(
            "move_at", [available_at(s).astimezone(UTC) for s in data["public_by"]]
        )
    )
    if rows.select("player_id", "move_at").is_duplicated().any():
        raise DataAcquisitionError("Conflicting same-time affiliation records")
    return rows


def transaction_facts(events: pl.DataFrame, clocks: pl.DataFrame) -> pl.DataFrame:
    home = events.with_columns(
        c("old_team").alias("team"),
        pl.when(c("kind") == "opening_suspension")
        .then(pl.lit("suspended"))
        .otherwise(pl.lit("left"))
        .alias("state"),
    )
    away = events.drop_nulls("new_team").with_columns(
        c("new_team").alias("team"),
        pl.when(c("kind") == "practice")
        .then(pl.lit("practice"))
        .otherwise(pl.lit("joined"))
        .alias("state"),
    )
    return (
        clocks.sort("week")
        .with_columns(c("week").cum_count().over("season", "team").alias("team_game"))
        .join(pl.concat([home, away]), on=["season", "team"])
        .filter(
            (c("move_at") <= c("cutoff"))
            & ((c("state") != "suspended") | (c("team_game") <= c("games")))
        )
        .with_columns(
            (c("week") == c("week").min().over(*CLUB_PLAYER, "move_at")).alias("first")
        )
        .sort("move_at")
        .unique(ENTRY, keep="last")
    )


def transaction_seeds(seeds: pl.DataFrame, facts: pl.DataFrame) -> pl.DataFrame:
    arrivals = facts.filter((c("state") == "joined") & c("first")).with_columns(
        pl.lit("transaction").alias("roster_source")
    )
    return pl.concat([seeds, arrivals.select(SEEDS)]).unique(
        ENTRY, keep="first", maintain_order=True
    )


def apply_transactions(
    pool: pl.DataFrame, facts: pl.DataFrame, boxes: pl.DataFrame, clocks: pl.DataFrame
) -> pl.DataFrame:
    # A lagged roster can repeat an already released player. Allow a genuine
    # return: a subsequent club announcement, dated chart or earlier game box.
    past = (
        boxes.rename({"actual_team": "team"})
        .join(facts.select(CLUB_PLAYER).unique(), on=CLUB_PLAYER, how="semi")
        .join(clocks.select(*SIDE, c("cutoff").alias("box_lock")), on=SIDE)
        .select(*CLUB_PLAYER, "box_lock")
        .sort("box_lock")
    )
    checked = (
        pool.join(
            facts.select(*ENTRY, "state", "move_at", "first"), on=ENTRY, how="left"
        )
        .sort("cutoff")
        .join_asof(
            past,
            by=CLUB_PLAYER,
            left_on="cutoff",
            right_on="box_lock",
            allow_exact_matches=False,
            check_sortedness=False,
        )
    )
    returned = pl.max_horizontal("identity_at", "depth_at", "box_lock") > c("move_at")
    activated = c("kind").is_in(["activated", "starter"]) & (
        c("news_at") > c("move_at")
    )
    reject = ((c("state") == "left") & ~returned.fill_null(False)) | (
        ((c("state") == "suspended") | ((c("state") == "practice") & c("first")))
        & ~activated.fill_null(False)
    )
    return (
        checked.filter(~reject.fill_null(False))
        .with_columns(
            pl.when((c("state") == "joined") & c("first"))
            .then(pl.max_horizontal("identity_at", "move_at"))
            .otherwise(c("identity_at"))
            .alias("identity_at")
        )
        .select(pool.columns)
    )

"""Point-in-time sources for the weekly team-role experiment."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import POSITIONS, TEAM_CODES

TEAM_WEEK = ["season", "week", "team"]
SIDE = ["game_id", "team"]
ENTRY = [*SIDE, "player_id"]
STAMP = pl.Datetime("us", "UTC")
DEPTH_CHANGE = 2025
NEWS_FILE = Path(__file__).with_name("role-news.csv")


def position(column: str) -> pl.Expr:
    return pl.col(column).replace({"HB": "RB", "FB": "RB"}).alias("position")


def game_clocks(
    sides: pl.DataFrame, schedule: pl.DataFrame, *, include_openers: bool = False
) -> pl.DataFrame:
    c = pl.col
    kickoff = (
        pl.concat_str("gameday", "gametime", separator=" ")
        .str.to_datetime("%Y-%m-%d %H:%M")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )
    clocks = sides.select("season", "week", *SIDE, "opponent").join(
        schedule.select("game_id", (kickoff - pl.duration(hours=1)).alias("cutoff")),
        on="game_id",
        validate="m:1",
    )
    if clocks.height != sides.height or clocks["cutoff"].null_count():
        raise DataAcquisitionError("Missing scheduled kickoff for a completed game")
    clocks = clocks.sort("season", "team", "week").with_columns(
        c("week").shift().over("season", "team").alias("roster_week")
    )
    return clocks if include_openers else clocks.drop_nulls("roster_week")


def depth_evidence(
    clocks: pl.DataFrame, weekly: pl.DataFrame, daily: pl.DataFrame | None
) -> pl.DataFrame:
    c = pl.col
    old = weekly.filter(c("game_type") == "REG").select(
        "season",
        c("week").alias("roster_week"),
        c("club_code").replace(TEAM_CODES).alias("team"),
        c("gsis_id").alias("player_id"),
        c("full_name").alias("depth_name"),
        position("position"),
        c("depth_team").cast(pl.Int32, strict=False).alias("depth_rank"),
    )
    prior = clocks.join(old, on=["season", "roster_week", "team"]).with_columns(
        pl.lit(None, dtype=STAMP).alias("depth_at")
    )
    cols = [*ENTRY, "position", "depth_rank", "depth_at", "depth_name"]
    frames = [prior.select(cols)]
    if daily is not None:
        rows = daily.select(
            c("team").replace(TEAM_CODES),
            c("dt").str.to_datetime(time_zone="UTC").alias("depth_at"),
            c("gsis_id").alias("player_id"),
            c("player_name").alias("depth_name"),
            position("pos_abb"),
            c("pos_rank").alias("depth_rank"),
        )
        # Select one whole team snapshot: as-of per *player* retains departed players.
        stamps = rows.select("team", "depth_at").unique().sort("depth_at")
        current = (
            clocks.filter(c("season") == DEPTH_CHANGE)
            .sort("cutoff")
            .join_asof(
                stamps,
                left_on="cutoff",
                right_on="depth_at",
                by="team",
                strategy="backward",
                tolerance="7d",
                check_sortedness=False,
            )
            .join(rows, on=["team", "depth_at"])
        )
        frames.append(current.select(cols))
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(
            c("position").is_in(POSITIONS)
            & c("player_id").is_not_null()
            & (c("player_id") != "")
        )
        .sort("depth_rank", "position", nulls_last=True)
        .unique(ENTRY, keep="first")
    )


def injury_evidence(clocks: pl.DataFrame, reports: pl.DataFrame) -> pl.DataFrame:
    """A week label alone (notably in 2025) cannot place a report before kickoff."""
    c = pl.col
    rows = reports.filter(c("game_type") == "REG").select(
        c("season", "week").cast(pl.Int32),
        c("team").replace(TEAM_CODES),
        c("gsis_id").alias("player_id"),
        c("report_status").alias("designation"),
        c("full_name").alias("report_name"),
        position("position"),
        "practice_status",
        c("date_modified").cast(STAMP).alias("report_at"),
    )
    return (
        clocks.join(rows, on=TEAM_WEEK)
        .filter(c("report_at") <= c("cutoff"))
        .sort("report_at")
        .unique(ENTRY, keep="last")
        .select(
            *ENTRY,
            "report_at",
            "designation",
            "practice_status",
            "report_name",
            "position",
        )
    )


def read_news(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, schema_overrides={"max_opportunities": pl.Float64})


def official_news(clocks: pl.DataFrame, news: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    invalid = (
        ~c("kind").is_in(["starter", "activated", "released", "inactive", "limited"])
        | ~c("position").is_in(POSITIONS)
        | ~c("source").str.starts_with("https://")
        | ~c("published_at").str.contains(r"(?:Z|[+-]\d\d:\d\d)$")
        | ((c("kind") == "starter") & (c("position") != "QB"))
        | ((c("kind") == "limited") & ~(c("max_opportunities") >= 0).fill_null(False))
    )
    if (
        news.drop_nulls(
            [*ENTRY, "position", "name", "kind", "source", "published_at"]
        ).height
        != news.height
        or news.filter(invalid).height
    ):
        raise DataAcquisitionError("Invalid timestamped role news record")
    rows = (
        clocks.join(news, on=SIDE)
        .with_columns(
            c("published_at").str.to_datetime(time_zone="UTC").alias("news_at")
        )
        .filter(c("news_at") <= c("cutoff"))
        .sort("news_at")
    )
    if rows.select(*ENTRY, "news_at").is_duplicated().any():
        raise DataAcquisitionError("Ambiguous same-time player news")
    identity = rows.unique(ENTRY, keep="last").select(
        *ENTRY, "position", "name", c("news_at").alias("identity_at")
    )
    availability = (
        rows.filter(c("kind") != "limited")
        .unique(ENTRY, keep="last")
        .select(*ENTRY, "kind", "news_at", "source")
    )
    caps = (
        rows.filter(c("kind") == "limited")
        .unique(ENTRY, keep="last")
        .select(
            *ENTRY,
            "max_opportunities",
            c("source").alias("limit_source"),
            c("news_at").alias("limit_at"),
        )
    )
    return identity.join(availability, on=ENTRY, how="left").join(
        caps, on=ENTRY, how="left"
    )

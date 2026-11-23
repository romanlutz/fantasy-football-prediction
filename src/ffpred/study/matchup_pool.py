"""League-wide freeze, offseason seeds, unique affiliations and participation labels."""

from __future__ import annotations

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import PLAYER_WEEK, POSITIONS, TEAM_CODES
from ffpred.study.role_evidence import ENTRY, TEAM_WEEK, position

c = pl.col
WEEK = ["season", "week"]
SEEDS = [*ENTRY, "position", "name", "roster_source"]
SNAP_START = 2013


def id_map(players: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """Use only identifiers, never the player dictionary's current team or status."""
    primary, backup = (
        f.select("gsis_id", "pfr_id").drop_nulls().filter(c("pfr_id") != "").unique()
        for f in (players, rosters)
    )
    ids = pl.concat([primary, backup.join(primary, on="pfr_id", how="anti")])
    return ids.filter(pl.len().over("pfr_id") == 1).rename({"gsis_id": "player_id"})


def weekly_clocks(clocks: pl.DataFrame) -> pl.DataFrame:
    return clocks.sort("team", "season", "week").with_columns(
        c("cutoff").shift().over("team").alias("prior_at"),
        c("season").shift().over("team").alias("prior_season"),
        c("week").shift().over("team").alias("prior_week"),
        c("roster_week").is_null().alias("opener"),
        c("cutoff").min().over(WEEK).alias("cutoff"),
    )


def opener_seeds(
    clocks: pl.DataFrame,
    rosters: pl.DataFrame,
    draft: pl.DataFrame,
    boxes: pl.DataFrame,
) -> pl.DataFrame:
    games = clocks.filter(c("opener"))
    old = rosters.filter(
        (c("game_type") == "REG")
        & c("status_description_abbr").is_in(["A01", "I01", "I02", "R01"])
    ).select(
        c("season").alias("prior_season"),
        c("week").alias("prior_week"),
        c("team").replace(TEAM_CODES),
        c("gsis_id").alias("player_id"),
        position("position"),
        c("full_name").alias("name"),
    )
    rookies = draft.select(
        "season",
        c("team").replace(TEAM_CODES),
        c("gsis_id").alias("player_id"),
        position("position"),
        c("pfr_player_name").alias("name"),
    )
    keys = ["prior_season", "prior_week", "team", "player_id"]
    coded = rosters.filter(
        (c("game_type") == "REG") & c("status_description_abbr").is_not_null()
    ).select(
        c("season").alias("prior_season"),
        c("week").alias("prior_week"),
        c("team").replace(TEAM_CODES),
        c("gsis_id").alias("player_id"),
    )
    fallback = boxes.select(
        c("season").alias("prior_season"),
        c("week").alias("prior_week"),
        c("actual_team").alias("team"),
        "player_id",
        c("box_position").replace({"FB": "RB", "HB": "RB"}).alias("position"),
        c("player_display_name").alias("name"),
    ).join(coded, on=keys, how="anti")
    return (
        pl.concat(
            [
                games.join(frame, on=keys)
                .with_columns(pl.lit(label).alias("roster_source"))
                .select(SEEDS)
                for frame, keys, label in (
                    (rookies, ["season", "team"], "draft"),
                    (old, ["prior_season", "prior_week", "team"], "offseason_carry"),
                    (fallback, ["prior_season", "prior_week", "team"], "offseason_box"),
                )
            ]
        )
        .drop_nulls("player_id")
        .filter(c("position").is_in(POSITIONS) & (c("player_id") != ""))
        .unique(ENTRY, keep="first", maintain_order=True)
    )


def player_transfers(trades: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    # Pick trades can carry the ID of the player eventually drafted with that pick.
    moves = (
        trades.filter(c("pick_season").is_null() & c("pick_round").is_null())
        .join(ids, on="pfr_id")
        .select(
            "player_id",
            c("received").replace(TEAM_CODES).alias("team"),
            c("pfr_name").alias("name"),
            (c("trade_date") + pl.duration(days=1))
            .cast(pl.Datetime)
            .dt.replace_time_zone("America/New_York")
            .dt.convert_time_zone("UTC")
            .alias("move_at"),
        )
        .unique()
    )
    if moves.select("player_id", "move_at").is_duplicated().any():
        raise DataAcquisitionError("Ambiguous same-day player transfer")
    return moves.sort("move_at")


def trade_events(
    moves: pl.DataFrame, rosters: pl.DataFrame, clocks: pl.DataFrame
) -> pl.DataFrame:
    prior_pos = (
        rosters.filter(c("game_type") == "REG")
        .select(
            *WEEK,
            c("team").replace(TEAM_CODES),
            c("gsis_id").alias("player_id"),
            position("position"),
        )
        .join(clocks.select(*TEAM_WEEK, "prior_at", "cutoff"), on=TEAM_WEEK)
        .select(
            "player_id",
            "position",
            (c("season") * 100 + c("week")).alias("observed_period"),
        )
        .filter(c("position").is_in(POSITIONS))
        .unique()
        .sort("observed_period", "position")
    )
    dest = (
        clocks.join(moves, on="team")
        .filter((c("move_at") <= c("cutoff")) & (c("move_at") > c("prior_at")))
        .with_columns((c("season") * 100 + c("week")).alias("period"))
        .sort("period")
        .join_asof(
            prior_pos,
            left_on="period",
            right_on="observed_period",
            by="player_id",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .drop_nulls("position")
    )
    return (
        dest.select(*ENTRY, "name", "position", "move_at")
        .sort("move_at")
        .unique(ENTRY, keep="last")
    )


def resolve_teams(
    pool: pl.DataFrame, reports: pl.DataFrame, moves: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    moves = moves.select(
        "player_id", c("team").alias("destination"), c("move_at").alias("trade_at")
    )
    rows = (
        pool.join(reports.select(*ENTRY, "report_at"), on=ENTRY, how="left")
        .sort("cutoff")
        .join_asof(
            moves,
            left_on="cutoff",
            right_on="trade_at",
            by="player_id",
            check_sortedness=False,
        )
        .with_columns(
            pl.max_horizontal("identity_at", "depth_at", "report_at", "prior_at").alias(
                "evidence_at"
            ),
            pl.len().over(PLAYER_WEEK).alias("team_count"),
        )
    )
    stale = (
        (c("destination") != c("team"))
        & (c("trade_at") >= c("prior_at"))
        & (
            c("trade_at")
            >= pl.max_horizontal("prior_at", "identity_at", "depth_at", "report_at")
        )
    ).fill_null(False)
    rows = rows.with_columns(stale.alias("stale_trade")).sort(
        ["stale_trade", "evidence_at", "team"],
        descending=[False, True, False],
        nulls_last=True,
    )
    keep = (
        rows.filter(~c("stale_trade"))
        .unique(PLAYER_WEEK, keep="first", maintain_order=True)
        .select(ENTRY)
    )
    removed = rows.join(keep, on=ENTRY, how="anti")
    return pool.join(keep, on=ENTRY, how="semi"), removed.select(
        *PLAYER_WEEK,
        "team",
        "destination",
        "trade_at",
        "cutoff",
        "stale_trade",
        "team_count",
    )


def played_labels(
    pool: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame, boxes: pl.DataFrame
) -> pl.DataFrame:
    regular = snaps.filter(c("game_type") == "REG").with_columns(
        c("team").replace(TEAM_CODES)
    )
    covered = (
        regular.select(TEAM_WEEK).unique().with_columns(pl.lit(True).alias("covered"))
    )
    positive = (
        regular.filter(
            pl.sum_horizontal("offense_snaps", "defense_snaps", "st_snaps") > 0
        )
        .join(ids, left_on="pfr_player_id", right_on="pfr_id")
        .select(PLAYER_WEEK)
        .unique()
        .with_columns(pl.lit(True).alias("played_snap"))
    )
    # A statistical event also proves participation if the PFR crosswalk is missing.
    events = (
        boxes.filter(
            (c("points") != 0)
            | (
                pl.sum_horizontal(
                    "attempts", "carries", "targets", "receptions", "fumbles_total"
                )
                > 0
            )
        )
        .select(PLAYER_WEEK)
        .with_columns(pl.lit(True).alias("event"))
    )
    labels = (
        pool.select(*PLAYER_WEEK, "team")
        .join(covered, on=TEAM_WEEK, how="left")
        .join(
            ids.select("player_id").unique().with_columns(pl.lit(True).alias("mapped")),
            on="player_id",
            how="left",
        )
        .join(positive, on=PLAYER_WEEK, how="left")
        .join(events, on=PLAYER_WEEK, how="left")
    )
    if labels.filter((c("season") >= SNAP_START) & c("covered").is_null()).height:
        raise DataAcquisitionError("Missing team snap-count coverage")
    return labels.select(
        *PLAYER_WEEK,
        pl.when(c("played_snap").fill_null(False) | c("event").fill_null(False))
        .then(True)
        .when(c("covered") & c("mapped"))
        .then(False)
        .otherwise(None)
        .alias("played"),
    )

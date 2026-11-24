"""Strict outcome labels. None of the evidence here selects a current-game role."""

from __future__ import annotations

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.matchup_pool import SNAP_START
from ffpred.study.panel import (
    CHANNELS,
    HISTORY_FIELDS,
    OTHER,
    PLAYER_WEEK,
    SCORING,
    STATS,
    TEAM_CODES,
)
from ffpred.study.role_evidence import ENTRY, SIDE, TEAM_WEEK

c = pl.col


def superseded_at_game(
    pool: pl.DataFrame, clocks: pl.DataFrame, moves: pl.DataFrame, facts: pl.DataFrame
) -> pl.DataFrame:
    """Check departures *after* an issued forecast but before its own game."""
    rows = (
        pool.select(*ENTRY, "cutoff")
        .join(
            clocks.select(*SIDE, c("cutoff").alias("game_at")), on=SIDE, validate="m:1"
        )
        .sort("game_at")
        .join_asof(
            moves.select("player_id", c("team").alias("destination"), "move_at").sort(
                "move_at"
            ),
            left_on="game_at",
            right_on="move_at",
            by="player_id",
            check_sortedness=False,
        )
        .join(
            facts.select(*ENTRY, c("move_at").alias("fact_at"), "state"),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
    )
    last = pl.max_horizontal("fact_at", "move_at")
    left = (
        pl.when(c("fact_at") == last)
        .then(c("state") == "left")
        .otherwise(c("destination") != c("team"))
    )
    return rows.filter((last > c("cutoff")) & left).select(ENTRY)


def matchup_evidence(  # noqa: PLR0913 - independent source frames
    pool: pl.DataFrame,
    boxes: pl.DataFrame,
    snaps: pl.DataFrame,
    ids: pl.DataFrame,
    rosters: pl.DataFrame,
    *,
    sides: pl.DataFrame,
    game_news: pl.DataFrame,
    superseded: pl.DataFrame,
) -> pl.DataFrame:
    regular = snaps.filter(c("game_type") == "REG").with_columns(
        c("team").replace(TEAM_CODES)
    )
    coverage = regular.select(SIDE).unique().with_columns(pl.lit(True).alias("covered"))
    if (
        pool.filter(c("season") >= SNAP_START)
        .join(coverage, on=SIDE, how="anti")
        .height
    ):
        raise DataAcquisitionError("Missing matchup snap-count coverage")
    positive = (
        regular.filter(
            pl.sum_horizontal("offense_snaps", "defense_snaps", "st_snaps") > 0
        )
        .join(ids, left_on="pfr_player_id", right_on="pfr_id")
        .group_by(PLAYER_WEEK)
        .agg(
            c("team").first().alias("snap_team"),
            c("game_id").first().alias("snap_game"),
            pl.struct(SIDE).n_unique().alias("snap_sides"),
        )
    )
    weekly = (
        rosters.filter(c("game_type") == "REG")
        .select(
            "season",
            "week",
            c("team").replace(TEAM_CODES),
            c("gsis_id").alias("player_id"),
        )
        .drop_nulls()
        .join(
            sides.select(*TEAM_WEEK, "game_id"),
            on=TEAM_WEEK,
            how="left",
            validate="m:1",
        )
        .group_by(PLAYER_WEEK)
        .agg(
            c("team").first().alias("roster_team"),
            c("game_id").first().alias("roster_game"),
            c("team").n_unique().alias("roster_teams"),
        )
    )
    rows = (
        pool.select(*PLAYER_WEEK, *SIDE)
        .join(
            boxes.select(
                *PLAYER_WEEK, "actual_team", c("game_id").alias("actual_game_id")
            ),
            on=PLAYER_WEEK,
            how="left",
            validate="m:1",
        )
        .join(positive, on=PLAYER_WEEK, how="left", validate="m:1")
        .join(weekly, on=PLAYER_WEEK, how="left", validate="m:1")
        .join(
            superseded.with_columns(pl.lit(True).alias("departed")),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
        .join(
            game_news.select(*ENTRY, c("kind").alias("game_news")),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
        .join(coverage, on=SIDE, how="left", validate="m:1")
        .join(
            ids.select("player_id").unique().with_columns(pl.lit(True).alias("mapped")),
            on="player_id",
            how="left",
            validate="m:1",
        )
    )
    state = (
        pl.when(
            (c("snap_sides") > 1)
            | (c("snap_team") != c("actual_team"))
            | (c("snap_game") != c("actual_game_id"))
        )
        .then(pl.lit("conflicting_play"))
        .when((c("game_id") == c("actual_game_id")) & (c("team") == c("actual_team")))
        .then(pl.lit("same_box"))
        .when(c("actual_game_id").is_not_null())
        .then(pl.lit("other_box"))
        .when((c("game_id") == c("snap_game")) & (c("team") == c("snap_team")))
        .then(pl.lit("same_snap"))
        .when(c("snap_game").is_not_null())
        .then(pl.lit("other_snap"))
        .when(c("game_news") == "released")
        .then(pl.lit("superseded_release"))
        .when(c("game_news") == "inactive")
        .then(pl.lit("same_inactive"))
        .when(c("departed"))
        .then(pl.lit("superseded_move"))
        .when(c("roster_teams") > 1)
        .then(pl.lit("conflicting_roster"))
        .when((c("game_id") == c("roster_game")) & (c("team") == c("roster_team")))
        .then(pl.lit("same_roster"))
        .when(c("roster_team").is_not_null())
        .then(pl.lit("other_roster"))
        .otherwise(pl.lit("unverified"))
    )
    return rows.with_columns(state.alias("outcome_status"))


def strict_panel(
    pool: pl.DataFrame, evidence: pl.DataFrame, boxes: pl.DataFrame
) -> pl.DataFrame:
    rows = pool.join(evidence, on=[*PLAYER_WEEK, *SIDE], validate="1:1")
    scores = boxes.rename({"actual_team": "team"}).with_columns(
        (
            (c("special_teams_tds") + c("fumble_recovery_tds")) * SCORING.other_td
            + pl.sum_horizontal(c(f"{ch}_2pt_conversions") for ch in CHANNELS)
            * SCORING.conversion
            + c("receptions") * SCORING.reception
        )
        .round(2)
        .alias("other_points"),
        (
            (c("points") != 0)
            | (
                pl.sum_horizontal(
                    "attempts", "carries", "targets", "receptions", "fumbles_total"
                )
                > 0
            )
        ).alias("event"),
    )
    targets = list(dict.fromkeys([*HISTORY_FIELDS, *STATS, *OTHER]))
    matched = c("outcome_status").str.starts_with("same_")
    labeled = (
        rows.join(
            scores.select(*ENTRY, *targets, "event"),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            *(
                pl.when(matched).then(c(k).fill_null(0)).otherwise(None).alias(k)
                for k in targets
            ),
            pl.when(~matched)
            .then(None)
            .when(c("snap_game").is_not_null() | c("event").fill_null(False))
            .then(True)
            .when(c("outcome_status") == "same_inactive")
            .then(False)
            .when(c("covered") & c("mapped"))
            .then(False)
            .otherwise(None)
            .alias("played"),
            (c("outcome_status") == "same_box").alias("has_box"),
        )
        .drop("event", "covered", "mapped")
    )
    fields = [*HISTORY_FIELDS, *STATS[1::2]]
    observed = scores.select(*PLAYER_WEEK, *fields)
    # A player can be inactive for two clubs in the same week. Preserve both
    # game-specific labels but enter just one zero into his player-week history.
    zeros = (
        labeled.filter(matched & ~c("has_box"))
        .select(observed.columns)
        .unique(PLAYER_WEEK)
    )
    history = (
        pl.concat([observed, zeros], how="vertical_relaxed")
        .sort("player_id", "season", "week")
        .with_columns(
            (c("season") * 100 + c("week")).alias("lag_period"),
            c("season").alias("history_season"),
            c("week").alias("history_week"),
            c("week").cum_count().over("player_id").alias("prior_games"),
            c("points")
            .rolling_std(10, min_samples=2)
            .over("player_id")
            .alias("volatility"),
            *(
                c(f)
                .rolling_mean(n, min_samples=1)
                .over("player_id")
                .alias(f"last{n}_{f}")
                for n in (1, 10)
                for f in HISTORY_FIELDS
            ),
            *(
                c(f).rolling_mean(10, min_samples=1).over("player_id").alias("lag_" + f)
                for f in STATS[1::2]
            ),
        )
    )
    if history.select(PLAYER_WEEK).is_duplicated().any():
        raise DataAcquisitionError("Ambiguous verified matchup history")
    return (
        labeled.sort("period")
        .join_asof(
            history.drop(*fields, "season", "week").sort("lag_period"),
            by="player_id",
            left_on="period",
            right_on="lag_period",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(c("prior_games").fill_null(0))
        .drop("lag_period")
    )

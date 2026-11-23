"""RB opportunity estimates from earlier team-games, not the eventual week's usage."""

from __future__ import annotations

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import PLAYER_WEEK, TEAM_CODES
from ffpred.study.role_evidence import SIDE, TEAM_WEEK

c = pl.col
COUNTS = ("runs", "inside5_runs", "inside5_targets")
CARRY = (
    "team_runs",
    "share_runs",
    "expected_runs",
    "backfield_competition",
    "carry_history_known",
)
GOAL = (
    "team_inside5_runs",
    "share_inside5_runs",
    "expected_inside5_runs",
    "team_inside5_targets",
    "share_inside5_targets",
    "expected_inside5_targets",
)
GOAL_LINE = 5


def count_opportunities(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    plays = pbp.filter(
        (c("season_type") == "REG")
        & c("posteam").is_not_null()
        & (c("play_deleted").fill_null(0) == 0)
        & (c("two_point_attempt").fill_null(0) == 0)
    )
    close = c("yardline_100").is_between(1, GOAL_LINE).fill_null(False)
    run = (
        (c("play_type") == "run")
        & (c("rush_attempt") == 1)
        & (c("qb_kneel").fill_null(0) == 0)
    )
    target = (c("play_type") == "pass") & c("receiver_player_id").is_not_null()
    counts = plays.with_columns(
        c("posteam").replace(TEAM_CODES).alias("team"),
        run.alias("runs"),
        (run & close).alias("inside5_runs"),
        (target & close).alias("inside5_targets"),
    )
    keys = [*SIDE, "season", "week"]
    teams = counts.group_by(keys).agg(c(COUNTS).sum())
    carriers = counts.filter(run & c("rusher_player_id").is_not_null()).select(
        *keys,
        c("rusher_player_id").alias("player_id"),
        "runs",
        "inside5_runs",
        pl.lit(False).alias("inside5_targets"),
    )
    receivers = counts.filter(target).select(
        *keys,
        c("receiver_player_id").alias("player_id"),
        pl.lit(False).alias("runs"),
        pl.lit(False).alias("inside5_runs"),
        "inside5_targets",
    )
    players = (
        pl.concat([carriers, receivers])
        .group_by(*keys, "player_id")
        .agg(c(COUNTS).sum())
    )
    return teams, players


def include_snap_zeros(
    players: pl.DataFrame, teams: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame
) -> pl.DataFrame:
    played = (
        snaps.filter(
            (c("game_type") == "REG")
            & (pl.sum_horizontal("offense_snaps", "defense_snaps", "st_snaps") > 0)
        )
        .join(ids, left_on="pfr_player_id", right_on="pfr_id")
        .select(*PLAYER_WEEK, c("team").replace(TEAM_CODES))
        .unique()
        .join(teams.select(*TEAM_WEEK, "game_id"), on=TEAM_WEEK, validate="m:1")
    )
    # PBP identifies the offense directly. Do not assign a zero to either
    # team when a snap crosswalk maps one player-week to two franchises.
    zeros = (
        played.join(players, on=PLAYER_WEEK, how="anti")
        .filter(pl.len().over(PLAYER_WEEK) == 1)
        .with_columns(pl.lit(0, dtype=pl.UInt32).alias(k) for k in COUNTS)
    )
    result = pl.concat([players, zeros.select(players.columns)], how="vertical_relaxed")
    if result.select(PLAYER_WEEK).is_duplicated().any():
        raise DataAcquisitionError("Ambiguous team for observed weekly opportunities")
    return result


def rb_opportunities(
    board: pl.DataFrame, teams: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    rb = board.filter(c("position") == "RB").sort("season", "week", "team", "player_id")
    if rb.select(SIDE).unique().join(teams, on=SIDE, how="anti").height:
        raise DataAcquisitionError("Missing play-by-play for a forecast team-game")
    team_form = teams.sort("team", "season", "week").with_columns(
        c(k).shift().rolling_mean(8, min_samples=1).over("team").alias("team_" + k)
        for k in COUNTS
    )
    obs = players.select(*PLAYER_WEEK, "game_id", "team", *COUNTS)
    history = obs.join(
        teams.select(*SIDE, *(c(k).alias("budget_" + k) for k in COUNTS)),
        on=SIDE,
        validate="m:1",
    )
    # Ratios of sums shrink sparse goal-line shares toward the player's share
    # of all carries. A bye is not a zero; estimates can follow a traded player.
    history = (
        history.sort("player_id", "season", "week")
        .with_columns(
            (c("season") * 100 + c("week")).alias("source_period"),
            *(
                c(k)
                .rolling_sum(5, min_samples=1)
                .over("player_id")
                .alias("recent_" + k)
                for k in (*COUNTS, *("budget_" + v for v in COUNTS))
            ),
        )
        .with_columns(
            (c("recent_runs") / c("recent_budget_runs").clip(lower_bound=1)).alias(
                "share_runs"
            )
        )
    )
    history = history.with_columns(
        (
            (c("recent_" + k) + 3 * c("share_runs")) / (c("recent_budget_" + k) + 3)
        ).alias("share_" + k)
        for k in COUNTS[1:]
    )
    historic = history.select(
        "player_id", "source_period", *("share_" + k for k in COUNTS)
    ).sort("source_period")
    featured = (
        rb.join(
            team_form.select(*SIDE, *("team_" + k for k in COUNTS)),
            on=SIDE,
            validate="m:1",
        )
        .with_columns((c("season") * 100 + c("week")).alias("forecast_period"))
        .sort("forecast_period")
        .join_asof(
            historic,
            left_on="forecast_period",
            right_on="source_period",
            by="player_id",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            c("source_period").is_not_null().alias("carry_history_known"),
            (
                pl.when(~c("ruled_out"))
                .then(c("share_runs"))
                .otherwise(0)
                .sum()
                .over(SIDE)
                - pl.when(~c("ruled_out"))
                .then(c("share_runs").fill_null(0))
                .otherwise(0)
            ).alias("backfield_competition"),
            *(
                (c("share_" + k) * c("team_" + k)).alias("expected_" + k)
                for k in COUNTS
            ),
        )
    )
    return featured.sort("season", "week", "team", "player_id")

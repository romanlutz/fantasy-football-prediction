"""Freeze each team's role selections independently before its kickoff."""

from __future__ import annotations

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import PLAYER_WEEK, TEAM_CODES
from ffpred.study.role_evidence import ENTRY, SIDE, TEAM_WEEK

SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
LEAGUE = {"QB": 24, "RB": 36, "WR": 36, "TE": 24}
GROUP = [*SIDE, "position"]
VOLUME = ["attempts", "carries", "targets"]


def documented_absences(
    rosters: pl.DataFrame, injuries: pl.DataFrame, boxes: pl.DataFrame
) -> pl.DataFrame:
    """Outcome-side illness/injury markers; never used to select that same week."""
    c = pl.col
    rows = (
        pl.concat(
            [
                f.select(
                    c("season", "week").cast(pl.Int32),
                    c("team").replace(TEAM_CODES),
                    c("gsis_id").alias("player_id"),
                )
                for f in (
                    rosters.filter(
                        (c("game_type") == "REG")
                        & (c("status_description_abbr") == "R01")
                    ),
                    injuries.filter(
                        (c("game_type") == "REG")
                        & (c("report_status") == "Out")
                        & (
                            c("report_primary_injury").is_not_null()
                            | c("practice_primary_injury").is_not_null()
                        )
                    ),
                )
            ],
        )
        .drop_nulls()
        .unique()
    )
    return rows.join(boxes.select(PLAYER_WEEK), on=PLAYER_WEEK, how="anti")


def histories(
    base: pl.DataFrame, boxes: pl.DataFrame, absences: pl.DataFrame
) -> pl.DataFrame:
    """Workload after each observation, skipping confirmed missed games, not byes."""
    c = pl.col
    observed = boxes.select(*PLAYER_WEEK, *VOLUME)
    zeros = (
        base.select(PLAYER_WEEK)
        .unique()
        .join(observed, on=PLAYER_WEEK, how="anti")
        .join(absences, on=PLAYER_WEEK, how="anti")
        .with_columns(pl.lit(0).alias(v) for v in VOLUME)
    )
    return (
        pl.concat([observed, zeros.select(observed.columns)], how="vertical_relaxed")
        .sort("player_id", "season", "week")
        .with_columns(
            (c("season") * 100 + c("week")).alias("history_period"),
            *(
                c(v)
                .rolling_mean(n, min_samples=1)
                .over("player_id")
                .alias(f"prior{n}_{v}")
                for n in (3, 10)
                for v in VOLUME
            ),
        )
    )


def workload(attempts: pl.Expr, carries: pl.Expr, targets: pl.Expr) -> pl.Expr:
    return (
        pl.when(pl.col("position") == "QB")
        .then(attempts)
        .when(pl.col("position") == "RB")
        .then(carries + targets)
        .otherwise(targets)
    )


def role_pool(
    clocks: pl.DataFrame,
    base: pl.DataFrame,
    depth: pl.DataFrame,
    news: pl.DataFrame,
    reports: pl.DataFrame,
) -> pl.DataFrame:
    c = pl.col
    cols = [*ENTRY, "name", "position"]
    # Old charts rank existing candidates only. A recently dated chart or
    # official game-specific announcement can introduce a new candidate.
    practicing = reports.filter(
        c("position").is_in(list(SLOTS))
        & c("player_id").is_not_null()
        & (c("player_id") != "")
        & c("practice_status").is_in(
            ["Full Participation in Practice", "Limited Participation in Practice"]
        )
        & ~c("designation").fill_null("").is_in(["Out", "Doubtful"])
    ).rename({"report_name": "name"})
    identities = pl.concat(
        [
            news.select(cols),
            depth.filter(c("depth_at").is_not_null())
            .rename({"depth_name": "name"})
            .select(cols),
            base.select(cols),
            practicing.select(cols),
        ],
        how="vertical_relaxed",
    ).unique(ENTRY, keep="first", maintain_order=True)
    return (
        identities.join(clocks, on=SIDE, validate="m:1")
        .join(
            base.select(*ENTRY, "roster_source"), on=ENTRY, how="left", validate="1:1"
        )
        .join(
            depth.drop("position", "depth_name"), on=ENTRY, how="left", validate="1:1"
        )
        .join(news.drop("position", "name"), on=ENTRY, how="left", validate="1:1")
    )


def select_roles(
    pool: pl.DataFrame, reports: pl.DataFrame, history: pl.DataFrame
) -> pl.DataFrame:
    c = pl.col
    rows = (
        pool.join(
            reports.drop("report_name", "position"),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
        .with_columns((c("season") * 100 + c("week")).alias("period"))
        .sort("period")
        .join_asof(
            history.select(
                "player_id", "history_period", pl.selectors.starts_with("prior")
            ).sort("history_period"),
            left_on="period",
            right_on="history_period",
            by="player_id",
            allow_exact_matches=False,
            strategy="backward",
            check_sortedness=False,
        )
    )
    status = c("kind").is_in(["inactive", "released"]).fill_null(False) | (
        (c("designation") == "Out").fill_null(False)
        & (c("news_at").is_null() | (c("report_at") >= c("news_at")))
    )
    recent, longer = (workload(*(c(f"prior{n}_{v}") for v in VOLUME)) for n in (3, 10))
    usage = (recent + longer) / 2
    rows = rows.with_columns(
        status.alias("ruled_out"),
        pl.min_horizontal(usage.fill_null(0), c("max_opportunities")).alias(
            "expected_usage"
        ),
        (c("kind") == "starter").fill_null(False).alias("announced"),
        ((c("position") == "QB") & c("depth_at").is_not_null() & (c("depth_rank") == 1))
        .fill_null(False)
        .alias("current_qb1"),
        (c("designation") == "Doubtful").fill_null(False).alias("doubtful"),
    )
    if rows.filter(c("announced") & ~c("ruled_out")).select(SIDE).is_duplicated().any():
        raise DataAcquisitionError("Conflicting eligible announced starting QBs")
    ranked = (
        rows.filter(~c("ruled_out"))
        .sort(
            [
                "announced",
                "doubtful",
                "current_qb1",
                "expected_usage",
                "depth_rank",
                "player_id",
            ],
            descending=[True, False, True, True, False, False],
            nulls_last=True,
        )
        .with_columns(c("player_id").cum_count().over(GROUP).alias("team_rank"))
    )
    return rows.join(
        ranked.select(*ENTRY, "team_rank"), on=ENTRY, how="left"
    ).with_columns(
        (c("team_rank") <= c("position").replace_strict(SLOTS))
        .fill_null(False)
        .alias("team_role")
    )


def league_reference(base_panel: pl.DataFrame) -> pl.DataFrame:
    """Previous league-wide caps: no Sunday reports can change a Thursday pick."""
    c = pl.col
    return (
        base_panel.with_columns(
            workload(*(c(f"last10_{v}") for v in VOLUME)).alias("league_usage")
        )
        .drop_nulls("league_usage")
        .sort(["league_usage", "player_id"], descending=[True, False])
        .with_columns(
            c("player_id")
            .cum_count()
            .over("season", "week", "position")
            .alias("league_rank")
        )
        .filter(c("league_rank") <= c("position").replace_strict(LEAGUE))
        .select(*ENTRY, "league_rank")
    )


def role_outcomes(
    clocks: pl.DataFrame,
    schedule: pl.DataFrame,
    boxes: pl.DataFrame,
    rosters: pl.DataFrame,
    absences: pl.DataFrame,
) -> pl.DataFrame:
    """Outcome population independent of the selected names."""
    c = pl.col
    qbs = pl.concat(
        [
            schedule.select(
                "game_id",
                c(f"{s}_team").replace(TEAM_CODES).alias("team"),
                c(f"{s}_qb_id").alias("player_id"),
                pl.lit("QB").alias("position"),
            )
            for s in ("home", "away")
        ]
    ).drop_nulls()
    game_roles = (
        rosters.filter(c("game_type") == "REG")
        .select(
            "season",
            "week",
            c("team").replace(TEAM_CODES),
            c("gsis_id").alias("player_id"),
            c("position").replace({"FB": "RB", "HB": "RB"}).alias("game_position"),
        )
        .drop_nulls()
        .unique()
    )
    # If the game roster has conflicting position rows, use the box as fallback.
    game_roles = game_roles.filter(pl.len().over(*TEAM_WEEK, "player_id") == 1)
    skill = (
        boxes.rename({"actual_team": "team"})
        .join(game_roles, on=[*TEAM_WEEK, "player_id"], how="left")
        .with_columns(
            pl.coalesce(
                "game_position", c("box_position").replace({"FB": "RB", "HB": "RB"})
            ).alias("position")
        )
        .filter(c("position").is_in(["RB", "WR", "TE"]))
        .join(qbs.select(ENTRY), on=ENTRY, how="anti")
        .with_columns(workload(*(c(v) for v in VOLUME)).alias("usage"))
        .filter(c("usage") > 0)
        .filter(
            c("usage").rank("min", descending=True).over(GROUP)
            <= c("position").replace_strict(SLOTS)
        )
        .select(*ENTRY, "position")
    )
    return (
        pl.concat([qbs, skill])
        .join(clocks.select(*SIDE, "season", "week", "roster_week"), on=SIDE)
        .join(
            absences.rename({"week": "roster_week"}).with_columns(
                pl.lit(True).alias("injury_return")
            ),
            on=["season", "roster_week", "team", "player_id"],
            how="left",
        )
        .with_columns(c("injury_return").fill_null(False))
    )

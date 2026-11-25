"""Test which selected teammates receive targets after a pre-kickoff report."""

import polars as pl
from threadpoolctl import threadpool_limits

from ffpred.errors import DataAcquisitionError
from ffpred.study.methods import method_scores, paired_mae
from ffpred.study.panel import TEAM_CODES, WEIGHTS
from ffpred.study.pass_workload import active_shares
from ffpred.study.role_evidence import ENTRY, SIDE
from ffpred.study.run import PAPER_YEAR

c = pl.col
RECEIVERS = ("RB", "WR", "TE")
STATUSES = ("out", "limited")
WINDOW = 5
MIN_GAMES_TARGETED = 2
REGULAR_SHARE = 0.10
SIGNALS = tuple(f"{s}_{p}" for s in STATUSES for p in RECEIVERS)
FEATURES = (
    *SIGNALS,
    *(f"{s}_{modifier}" for s in SIGNALS for modifier in ("first", "recent")),
)


def regular_reports(
    board: pl.DataFrame, games: pl.DataFrame, boxes: pl.DataFrame
) -> pl.DataFrame:
    report = c("report_at") <= c("cutoff")
    news = c("news_at") <= c("cutoff")
    superseded = (
        news
        & c("kind").is_in(["activated", "starter"])
        & (c("news_at") > c("report_at"))
    ).fill_null(False)
    reported_out = (report & (c("designation") == "Out") & ~superseded).fill_null(False)
    inactive = (news & (c("kind") == "inactive")).fill_null(False)
    cap = (
        (c("limit_at") <= c("cutoff")) & c("max_opportunities").is_not_null()
    ).fill_null(False)
    practice = (
        report
        & (c("practice_status") == "Limited Participation in Practice")
        & ~superseded
    ).fill_null(False)
    flagged = (
        board.filter(c("position").is_in(RECEIVERS))
        .with_columns(
            pl.when(inactive)
            .then(pl.lit("inactive"))
            .when(reported_out)
            .then(pl.lit("out_report"))
            .when(cap)
            .then(pl.lit("announced_limit"))
            .when(practice)
            .then(pl.lit("limited_practice"))
            .alias("signal")
        )
        .drop_nulls("signal")
        .select(
            "season",
            "week",
            *ENTRY,
            "name",
            "position",
            "cutoff",
            "signal",
            pl.when(c("signal") == "inactive")
            .then(c("news_at"))
            .when(c("signal") == "announced_limit")
            .then(c("limit_at"))
            .otherwise(c("report_at"))
            .alias("signal_at"),
        )
    )
    # Look back by *this franchise's games*, not five calendar weeks or
    # appearances elsewhere. A long-absent player's targets are not newly vacant.
    previous = (
        games.sort("team", "season", "week")
        .select(
            *SIDE,
            *(
                c("game_id").shift(i).over("team").alias(str(i))
                for i in range(1, WINDOW + 1)
            ),
        )
        .unpivot(index=SIDE, value_name="prior_game")
        .drop_nulls("prior_game")
        .drop("variable")
    )
    player_counts = boxes.filter(
        (c("season_type") == "REG") & c("player_id").is_not_null()
    ).select(
        c("game_id").alias("prior_game"),
        c("team").replace(TEAM_CODES),
        "player_id",
        "targets",
    )
    mass = (
        flagged.select(ENTRY)
        .join(previous, on=SIDE)
        .join(
            games.select(
                c("game_id").alias("prior_game"),
                "team",
                c("targets").alias("team_targets"),
            ),
            on=["team", "prior_game"],
            validate="m:1",
        )
        .join(
            player_counts,
            on=["team", "prior_game", "player_id"],
            how="left",
            validate="m:1",
        )
        .group_by(ENTRY)
        .agg(
            pl.len().alias("history_games"),
            (c("targets") > 0).sum().alias("target_games"),
            (c("targets").sum() / c("team_targets").sum()).alias("recent_team_share"),
        )
    )
    return (
        flagged.join(mass, on=ENTRY, validate="1:1")
        .filter(
            (c("recent_team_share") >= REGULAR_SHARE)
            & (c("target_games") >= MIN_GAMES_TARGETED)
        )
        .with_columns(
            pl.when(c("signal").is_in(["inactive", "out_report"]))
            .then(pl.lit("out"))
            .otherwise(pl.lit("limited"))
            .alias("status")
        )
        .sort("season", "week", *ENTRY)
    )


def teammate_features(roles: pl.DataFrame, reports: pl.DataFrame) -> pl.DataFrame:
    def match(s: str, p: str) -> pl.Expr:
        return (c("status") == s) & (c("position") == p)

    totals = reports.group_by(SIDE).agg(
        c("recent_team_share").filter(match(s, p)).sum().alias(f"{s}_{p}")
        for s in STATUSES
        for p in RECEIVERS
    )
    return (
        roles.join(totals, on=SIDE, how="left", validate="m:1")
        .join(
            reports.select(*ENTRY, "status", "recent_team_share"),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                c(f"{s}_{p}").fill_null(0)
                - pl.when(match(s, p)).then(c("recent_team_share")).otherwise(0)
            )
            .clip(lower_bound=0)
            .alias(f"{s}_{p}")
            for s in STATUSES
            for p in RECEIVERS
        )
        .with_columns(
            *((c(k) * (c("team_rank") == 1)).alias(f"{k}_first") for k in SIGNALS),
            *(
                (c(k) * c("share_targets").fill_null(0)).alias(f"{k}_recent")
                for k in SIGNALS
            ),
            pl.when(pl.sum_horizontal(f"out_{p}" for p in RECEIVERS) > 0)
            .then(pl.lit("out"))
            .when(pl.sum_horizontal(f"limited_{p}" for p in RECEIVERS) > 0)
            .then(pl.lit("limited_only"))
            .otherwise(pl.lit("none"))
            .alias("peer_report"),
        )
    )


def inheritance_replay(
    control: pl.DataFrame, roles: pl.DataFrame, end_year: int
) -> pl.DataFrame:
    with threadpool_limits(limits=2):
        fitted = active_shares(
            roles,
            end_year,
            positions=RECEIVERS,
            units=("targets",),
            extra_features=FEATURES,
        )
    joined = control.join(
        roles.select(*ENTRY, "status", "peer_report", *SIGNALS),
        on=ENTRY,
        validate="1:1",
    ).join(
        fitted.rename({"active_share_targets": "refit_share"}),
        on=ENTRY,
        how="left",
        validate="1:1",
    )
    if (
        control.height != joined.height
        or joined.filter(
            c("position").is_in(RECEIVERS) & c("refit_share").is_null()
        ).height
    ):
        raise DataAcquisitionError(
            "Missing teammate target forecast for an issued role"
        )
    adjust = (c("peer_report") != "none") & c("position").is_in(RECEIVERS)
    old_share = c("allocated_share_targets")
    candidate = (
        c("refit_share")
        * c("play_probability")
        / c("uncapped_targets").sum().over(SIDE).clip(lower_bound=1)
    )
    # Leave QBs and players without a flagged *peer* alone, including the only
    # limited player himself. Never spend those players' share a second time.
    rows = joined.with_columns(
        pl.when(adjust).then(candidate).otherwise(0).alias("candidate")
    )
    available = (
        1 - pl.when(adjust).then(0).otherwise(old_share).sum().over(SIDE)
    ).clip(0, 1)
    wanted = c("candidate").sum().over(SIDE)
    scale = pl.when(wanted > available).then(available / wanted).otherwise(1)
    rows = rows.with_columns(
        pl.when(adjust)
        .then(c("candidate") * scale)
        .otherwise(old_share)
        .alias("inherit_share")
    ).with_columns(
        pl.when(adjust)
        .then(c("inherit_share") * c("budget_targets"))
        .otherwise(c("workload_targets"))
        .alias("inherit_targets")
    )
    delta = c("inherit_targets") - c("workload_targets")
    rows = rows.with_columns(
        (
            c(f"workload_{model}")
            + delta
            * (
                c(f"{model}_receiving_rate") * WEIGHTS["receiving_yards"]
                + c(f"{model}_receiving_td_rate") * WEIGHTS["receiving_tds"]
            )
        ).alias(f"inherit_{model}")
        for model in ("linear", "forest")
    ).with_columns(
        (
            c("workload_ensemble")
            + (
                (c("inherit_linear") - c("workload_linear"))
                + (c("inherit_forest") - c("workload_forest"))
            )
            / 4
        ).alias("inherit_ensemble")
    )
    return rows.drop("candidate").sort("season", "week", *ENTRY)


def inheritance_tables(
    pred: pl.DataFrame, reports: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    verified = pred.drop_nulls("points")
    scores, annual, paired = [], [], []
    for cohort in ("all", "out", "limited_only", "none"):
        rows = (
            verified if cohort == "all" else verified.filter(c("peer_report") == cohort)
        )
        if rows.is_empty():
            continue
        for metric, old, new in (
            ("points", "workload_ensemble", "inherit_ensemble"),
            ("targets", "workload_targets", "inherit_targets"),
        ):
            scored = rows.with_columns(c(metric).alias("points"))
            by_year, pooled = method_scores(scored, [old, new])
            tags = [pl.lit(cohort).alias("cohort"), pl.lit(metric).alias("metric")]
            scores.append(pooled.with_columns(tags))
            annual.append(by_year.with_columns(tags))
            paired.append(
                paired_mae(by_year, reference=old)
                .filter(c("model") == new)
                .with_columns(tags)
            )
    recipients = [
        verified.filter(c("position").is_in(RECEIVERS) & (c(f"{s}_{p}") > 0))
        .group_by("position", "team_rank")
        .agg(
            pl.lit(s).alias("donor_status"),
            pl.lit(p).alias("donor_position"),
            pl.len().alias("rows"),
            c("workload_targets", "inherit_targets", "targets").mean(),
            (c("targets") - c("workload_targets")).mean().alias("actual_minus_control"),
            (c("inherit_targets") - c("workload_targets"))
            .mean()
            .alias("predicted_change"),
        )
        for s in STATUSES
        for p in RECEIVERS
    ]
    sides = pred.select("season", *SIDE).unique()
    qualifying = reports.filter(c("season") >= PAPER_YEAR)
    per_side = qualifying.group_by(SIDE).agg(
        (c("status") == "out").sum().alias("out_regulars"),
        (c("status") == "limited").sum().alias("limited_regulars"),
    )
    coverage = (
        sides.join(per_side, on=SIDE, how="left")
        .with_columns(c("out_regulars", "limited_regulars").fill_null(0))
        .group_by("season")
        .agg(
            pl.len().alias("team_games"),
            c("out_regulars", "limited_regulars").sum(),
            (c("out_regulars") > 0).sum().alias("with_out"),
            ((c("out_regulars") == 0) & (c("limited_regulars") > 0))
            .sum()
            .alias("limited_only"),
        )
        .sort("season")
    )
    return {
        "scores.csv": pl.concat(scores).sort("cohort", "metric", "position", "model"),
        "annual.csv": pl.concat(annual).sort(
            "cohort", "metric", "season", "position", "model"
        ),
        "paired.csv": pl.concat(paired).sort("cohort", "metric", "position"),
        "recipients.csv": pl.concat(recipients).sort(
            "donor_status", "donor_position", "position", "team_rank"
        ),
        "coverage.csv": coverage,
        "target-leaders.csv": target_leaders(pred),
        "regular-reports.parquet": qualifying,
    }


def target_leaders(pred: pl.DataFrame) -> pl.DataFrame:
    eligible = pred.filter(
        c("position").is_in(RECEIVERS)
        & c("status").is_null()
        & (c("peer_report") != "none")
    )

    def pick(col: str, model: str) -> pl.Expr:
        return (
            c(col)
            .sort_by([f"{model}_targets", "player_id"], descending=[True, False])
            .first()
        )

    return (
        eligible.group_by(*SIDE, "peer_report")
        .agg(
            pl.len().alias("choices"),
            c("points").count().alias("verified"),
            c("targets").max().alias("best"),
            *(pick("player_id", m).alias(m) for m in ("workload", "inherit")),
            *(pick("targets", m).alias(f"{m}_actual") for m in ("workload", "inherit")),
        )
        .filter((c("choices") > 1) & (c("verified") == c("choices")) & (c("best") > 0))
        .group_by("peer_report")
        .agg(
            pl.len().alias("team_games"),
            (c("workload") != c("inherit")).sum().alias("changed_pick"),
            *(
                (c(f"{m}_actual") == c("best")).sum().alias(f"{m}_correct")
                for m in ("workload", "inherit")
            ),
            *(
                (c("best") - c(f"{m}_actual")).mean().alias(f"{m}_targets_missed")
                for m in ("workload", "inherit")
            ),
        )
        .sort("peer_report")
    )

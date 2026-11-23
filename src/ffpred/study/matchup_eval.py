"""Point, participation, component, distribution and league starter scorecards."""

from __future__ import annotations

import numpy as np
import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.matchup_models import NAMES
from ffpred.study.methods import method_scores
from ffpred.study.panel import CHANNELS, PLAYER_WEEK, WEIGHTS
from ffpred.study.run import PAPER_YEAR

c = pl.col
K = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
QUARTILE = 0.25
WEEK_POS = ["season", "week", "position"]


def starter_scores(
    pred: pl.DataFrame, universe: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Loss against a perfect top K, with best possible overlap across boundary ties."""
    selected = pred.select(*PLAYER_WEEK, "position", "points")
    all_rows = pl.concat(
        [
            selected,
            universe.join(selected, on=PLAYER_WEEK, how="anti").select(
                selected.columns
            ),
        ],
        how="vertical_relaxed",
    )
    boards = []
    for label, pool in (("league", all_rows), ("roles", selected)):
        oracle = pool.sort(
            ["points", "player_id"], descending=[True, False]
        ).with_columns(c("player_id").cum_count().over(WEEK_POS).alias("rank"))
        top = (
            oracle.filter(c("rank") <= c("position").replace_strict(K))
            .group_by(WEEK_POS)
            .agg(
                c("points").min().alias("boundary"),
                c("points").sum().alias("oracle_points"),
                (c("points") == c("points").min()).sum().alias("tie_slots"),
            )
        )
        for model in NAMES:
            picks = (
                pred.sort([model, "player_id"], descending=[True, False])
                .with_columns(c("player_id").cum_count().over(WEEK_POS).alias("rank"))
                .filter(c("rank") <= c("position").replace_strict(K))
                .join(top, on=WEEK_POS, validate="m:1")
            )
            boards.append(
                picks.group_by(WEEK_POS)
                .agg(
                    pl.lit(model).alias("model"),
                    pl.lit(label).alias("oracle"),
                    pl.len().alias("picked"),
                    (
                        (c("points") > c("boundary")).sum()
                        + pl.min_horizontal(
                            (c("points") == c("boundary")).sum(), c("tie_slots").first()
                        )
                    ).alias("hits"),
                    c("points").sum().alias("picked_points"),
                    c("oracle_points").first(),
                )
                .with_columns(
                    (c("oracle_points") - c("picked_points")).round(2).alias("regret")
                )
            )
    weekly = pl.concat(boards).sort(*WEEK_POS, "oracle", "model")
    if weekly.filter(
        (c("picked") != c("position").replace_strict(K)) | (c("regret") < 0)
    ).height:
        raise DataAcquisitionError(
            "Incomplete or inconsistent weekly starter comparison"
        )
    pooled = (
        weekly.group_by("position", "model", "oracle")
        .agg(
            pl.len().alias("weeks"),
            c("picked", "hits", "regret", "picked_points", "oracle_points").mean(),
            (c("hits") == c("picked")).sum().alias("perfect_weeks"),
        )
        .sort("position", "model", "oracle")
    )
    return weekly, pooled


def component_errors(pred: pl.DataFrame) -> pl.DataFrame:
    reports = []
    for name in ("linear", "forest"):
        total = (c(f"factor_{name}") - c("points")).sign()
        direct = (c("boost_role") - c("points")).sign()
        for stat in [*WEIGHTS, *CHANNELS.values()]:
            err = c(f"{name}_{stat}") - c(stat)
            weight = WEIGHTS.get(stat, 0.0)
            reports.append(
                pred.group_by("position").agg(
                    pl.lit(name).alias("technique"),
                    pl.lit(stat).alias("stat"),
                    pl.len().alias("rows"),
                    err.abs().mean().alias("stat_mae"),
                    err.pow(2).mean().sqrt().alias("stat_rmse"),
                    (err * weight * total).mean().alias("attributed_mae"),
                    (err * weight * direct).mean().alias("direct_bridge_mae"),
                )
            )
        delta = c("boost_role") - c(f"factor_{name}")
        reports.append(
            pred.group_by("position").agg(
                pl.lit(name).alias("technique"),
                pl.lit("direct_minus_components").alias("stat"),
                pl.len().alias("rows"),
                delta.abs().mean().alias("stat_mae"),
                delta.pow(2).mean().sqrt().alias("stat_rmse"),
                pl.lit(0.0).alias("attributed_mae"),
                (delta * direct).mean().alias("direct_bridge_mae"),
            )
        )
    return pl.concat(reports).sort("position", "technique", "stat")


def yardage_errors(pred: pl.DataFrame) -> pl.DataFrame:
    """Split a yardage miss into workload and remaining yield at forecast efficiency."""
    reports = []
    for name in ("linear", "forest"):
        for channel, unit in CHANNELS.items():
            rate = c(f"{name}_{channel}_rate")
            actual_yards = c(f"{channel}_yards")
            predicted_yards = c(f"{name}_{channel}_yards")
            volume_error = (c(f"{name}_{unit}") - c(unit)) * rate
            yard_error = predicted_yards - actual_yards
            sign = (c(f"factor_{name}") - c("points")).sign()
            w = WEIGHTS[f"{channel}_yards"]
            reports.append(
                pred.group_by("position").agg(
                    pl.lit(name).alias("technique"),
                    pl.lit(channel).alias("channel"),
                    ((rate - actual_yards / c(unit)).abs().filter(c(unit) > 0))
                    .mean()
                    .alias("rate_mae_when_used"),
                    (c(unit) > 0).sum().alias("games_used"),
                    (volume_error * w * sign).mean().alias("workload_mae_contribution"),
                    ((yard_error - volume_error) * w * sign)
                    .mean()
                    .alias("yield_mae_contribution"),
                )
            )
    return pl.concat(reports).sort("position", "technique", "channel")


def intervals(pred: pl.DataFrame) -> pl.DataFrame:
    reports = []
    for label, mask in (
        ("all", pl.lit(True)),
        ("opener", c("opener")),
        ("previous_absence", c("prior_absence")),
        ("narrowest_quarter", c("width_rank") <= QUARTILE),
        ("widest_quarter", c("width_rank") > 1 - QUARTILE),
    ):
        for level, lo, hi in ((0.5, "q25", "q75"), (0.8, "q10", "q90")):
            y = c("points")
            q = (1 - level) / 2
            reports.append(
                pred.with_columns((c(hi) - c(lo)).alias("width"))
                .with_columns(
                    (
                        c("width").rank("average").over("position")
                        / pl.len().over("position")
                    ).alias("width_rank")
                )
                .filter(mask)
                .group_by("position")
                .agg(
                    pl.lit(label).alias("cohort"),
                    pl.lit(level).alias("level"),
                    pl.len().alias("rows"),
                    y.is_between(c(lo), c(hi)).mean().alias("coverage"),
                    c("width").mean().alias("width"),
                    (c("ensemble") - y).abs().mean().alias("mean_forecast_mae"),
                    pl.max_horizontal(q * (y - c(lo)), (q - 1) * (y - c(lo)))
                    .mean()
                    .alias("lower_pinball"),
                    pl.max_horizontal((1 - q) * (y - c(hi)), -q * (y - c(hi)))
                    .mean()
                    .alias("upper_pinball"),
                )
            )
    return pl.concat(reports).sort("position", "cohort", "level")


def evaluate_matchups(
    pred: pl.DataFrame, data: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    if not np.isfinite(
        pred.select(*NAMES, "q10", "q25", "q75", "q90").to_numpy()
    ).all():
        raise DataAcquisitionError("Nonfinite held-out matchup prediction")
    annual, pooled = method_scores(pred, NAMES)
    weekly, starters = starter_scores(
        pred, data["universe"].filter(c("season") >= PAPER_YEAR)
    )
    slices = []
    for label, mask in (
        ("opener", c("opener")),
        ("in_season", ~c("opener")),
        ("previous_absence", c("prior_absence")),
    ):
        _, score = method_scores(pred.filter(mask), NAMES)
        slices.append(score.with_columns(pl.lit(label).alias("cohort")))
    # Fantasy managers get points wherever the player actually played; a stricter
    # matchup target gives no credit for points earned playing for another team.
    matchup_target = pred.with_columns(
        pl.when(c("actual_team").is_not_null() & (c("actual_team") != c("team")))
        .then(0.0)
        .otherwise(c("points"))
        .alias("points")
    )
    for label, mask in (
        ("matchup_target_opener", c("opener")),
        ("matchup_target_in_season", ~c("opener")),
    ):
        _, score = method_scores(matchup_target.filter(mask), NAMES)
        slices.append(score.with_columns(pl.lit(label).alias("cohort")))
    prob = pred.filter(c("played").is_not_null()).with_columns(
        (c("play_probability") * 10).floor().clip(0, 9).alias("decile")
    )

    def prob_stats(keys: list[str]) -> pl.DataFrame:
        return (
            prob.group_by(keys)
            .agg(
                pl.len().alias("rows"),
                c("play_probability").mean().alias("mean_probability"),
                c("played").mean().alias("played_rate"),
                (c("play_probability") - c("played").cast(pl.Float64))
                .pow(2)
                .mean()
                .alias("brier"),
            )
            .sort(keys)
        )

    return {
        "scores.csv": pooled,
        "annual.csv": annual,
        "cohorts.csv": pl.concat(slices),
        "intervals.csv": intervals(pred),
        "components.csv": component_errors(pred),
        "yardage.csv": yardage_errors(pred),
        "starters-weekly.csv": weekly,
        "starters.csv": starters,
        "participation.csv": prob_stats(["position"]),
        "participation-calibration.csv": prob_stats(["position", "decile"]),
        "example-week.csv": example_week(pred),
    }


def example_week(pred: pl.DataFrame) -> pl.DataFrame:
    """A readable player-level forecast board near midseason of the last holdout."""
    midseason = 10
    year = pred.filter(c("season") == c("season").max())
    week = min(year["week"].unique().to_list(), key=lambda w: (abs(w - midseason), w))
    return (
        year.filter(c("week") == week)
        .sort(["position", "ensemble", "player_id"], descending=[False, True, False])
        .with_columns(c("player_id").cum_count().over("position").alias("rank"))
        .select(
            *WEEK_POS,
            "rank",
            (c("rank") <= c("position").replace_strict(K)).alias("would_start"),
            "team",
            "player_id",
            "name",
            "cutoff",
            "play_probability",
            "ensemble",
            "q25",
            "q75",
            "q10",
            "q90",
            "points",
            *(
                f"forest_{v}"
                for v in (
                    "active_carries",
                    "carries",
                    "rushing_rate",
                    "rushing_yards",
                    "rushing_tds",
                    "targets",
                    "receiving_yards",
                    "receiving_tds",
                    "attempts",
                    "passing_yards",
                    "passing_tds",
                )
            ),
        )
    )

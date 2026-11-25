"""Forecast team pass volume and allocate it across the pregame selected roles."""

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from ffpred.errors import DataAcquisitionError
from ffpred.study.matchup_models import FEATURES, NAMES, Array, ridge_mean
from ffpred.study.matchup_pool import SNAP_START
from ffpred.study.methods import method_scores, paired_mae
from ffpred.study.opportunities import CARRY
from ffpred.study.panel import PLAYER_WEEK, POSITIONS, TEAM_CODES, WEIGHTS
from ffpred.study.role_evidence import ENTRY, SIDE
from ffpred.study.run import PAPER_YEAR

c = pl.col
UNITS = ("attempts", "targets")
TEAM_STATS = (*UNITS, "carries", "sacks_suffered")
SHARES = tuple(f"share_{v}" for v in UNITS)
SHARE_FEATURES = [
    *FEATURES,
    *SHARES,
    "prior_appearances",
    *(f"peers_{s}" for s in SHARES),
]
TEAM_FEATURES = [
    f"{side}_{stat}_{n}"
    for side in ("own", "allowed")
    for stat in TEAM_STATS
    for n in (3, 8)
]
PAIRS = {
    "workload_linear": "factor_linear",
    "workload_forest": "factor_forest",
    "workload_ensemble": "ensemble",
}


def _design(
    train: pl.DataFrame, test: pl.DataFrame, features: list[str]
) -> tuple[Array, Array]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(train.select(features)), imputer.transform(
        test.select(features)
    )


def team_games(stats: pl.DataFrame) -> pl.DataFrame:
    games = (
        stats.filter(c("season_type") == "REG")
        .select(
            "season",
            "week",
            "game_id",
            c("team").replace(TEAM_CODES),
            c("opponent_team").replace(TEAM_CODES).alias("opponent"),
            *TEAM_STATS,
        )
        .sort("season", "week", "team")
    )
    if (
        games.select("season", "week", "team").is_duplicated().any()
        or games.select("season", "week", "opponent").is_duplicated().any()
        or games.drop_nulls().height != games.height
        or games.filter((c("targets") <= 0) | (c("targets") > c("attempts"))).height
    ):
        raise DataAcquisitionError("Missing or invalid game-side passing workload")
    return games.with_columns(
        c(stat)
        .shift()
        .rolling_mean(n, min_samples=1)
        .over(key)
        .alias(f"{side}_{stat}_{n}")
        for side, key in (("own", "team"), ("allowed", "opponent"))
        for stat in TEAM_STATS
        for n in (3, 8)
    )


def team_forecasts(games: pl.DataFrame, end_year: int) -> pl.DataFrame:
    folds = []
    for year in range(PAPER_YEAR, end_year + 1):
        prior = games.filter(c("season") < year)
        test = games.filter(c("season") == year)
        x, xt = _design(prior, test, TEAM_FEATURES)
        budgets = np.maximum(0, ridge_mean(x, prior.select(UNITS).to_numpy(), xt))
        budgets[:, 1] = np.minimum(budgets[:, 1], budgets[:, 0])
        folds.append(
            test.select(
                "season", "week", *SIDE, *UNITS, *(f"own_{u}_8" for u in UNITS)
            ).with_columns(
                pl.Series(f"budget_{u}", budgets[:, i]) for i, u in enumerate(UNITS)
            )
        )
    return pl.concat(folds)


def share_board(
    board: pl.DataFrame,
    games: pl.DataFrame,
    stats: pl.DataFrame,
    appearances: pl.DataFrame,
) -> pl.DataFrame:
    boxes = stats.filter(
        (c("season_type") == "REG") & c("player_id").is_not_null()
    ).select(
        "season",
        "week",
        "game_id",
        c("team").replace(TEAM_CODES),
        "player_id",
        *UNITS,
        "carries",
    )
    # Mapped positive snaps, plus official touches outside the selected role pool.
    observed = pl.concat(
        [
            boxes.filter(pl.sum_horizontal(*UNITS, "carries") > 0).select(
                *PLAYER_WEEK, *SIDE
            ),
            appearances.select(*PLAYER_WEEK, *SIDE),
        ],
        how="vertical_relaxed",
    ).unique()
    if observed.select(PLAYER_WEEK).is_duplicated().any():
        raise DataAcquisitionError("Ambiguous player-week in passing share history")
    totals = games.select(*SIDE, *(c(u).alias(f"actual_team_{u}") for u in UNITS))
    obs = (
        observed.join(
            boxes.select(*ENTRY, *UNITS), on=ENTRY, how="left", validate="1:1"
        )
        .with_columns(c(UNITS).fill_null(0))
        .join(totals, on=SIDE, how="left", validate="m:1")
    )
    if obs.drop_nulls().height != obs.height:
        raise DataAcquisitionError(
            "Missing team denominator for observed passing workload"
        )
    hist = obs.sort("player_id", "season", "week")
    hist = hist.with_columns(
        (c("season") * 100 + c("week")).alias("source_period"),
        c("week").cum_count().over("player_id").alias("prior_appearances"),
        *(
            (
                c(u).rolling_sum(5, min_samples=1).over("player_id")
                / c(f"actual_team_{u}").rolling_sum(5, min_samples=1).over("player_id")
            ).alias(s)
            for u, s in zip(UNITS, SHARES, strict=True)
        ),
    )
    roles = (
        board.filter(c("team_role"))
        .join(totals, on=SIDE, how="left", validate="m:1")
        .sort("period")
        .join_asof(
            hist.select(
                "player_id", "source_period", "prior_appearances", *SHARES
            ).sort("source_period"),
            by="player_id",
            left_on="period",
            right_on="source_period",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            c("prior_appearances").fill_null(0),
            *(
                (c(s).sum().over(SIDE) - c(s).fill_null(0)).alias(f"peers_{s}")
                for s in SHARES
            ),
        )
    )
    if roles.drop_nulls([f"actual_team_{u}" for u in UNITS]).height != roles.height:
        raise DataAcquisitionError("Missing team denominator for selected role")
    return roles


def active_shares(roles: pl.DataFrame, end_year: int) -> pl.DataFrame:
    folds = []
    for year in range(PAPER_YEAR, end_year + 1):
        for pos in POSITIONS:
            group = roles.filter(c("position") == pos)
            prior = group.filter(
                c("season").is_between(SNAP_START, year - 1)
                & c("played")
                & c("points").is_not_null()
            )
            test = group.filter(c("season") == year)
            features = [*SHARE_FEATURES, *CARRY] if pos == "RB" else SHARE_FEATURES
            x, xt = _design(prior, test, features)
            estimates = []
            for unit in UNITS:
                team = prior[f"actual_team_{unit}"].to_numpy()
                share = prior[unit].to_numpy() / team
                rate = ridge_mean(x, share, xt, weights=team).clip(0, 1)
                estimates.append(pl.Series(f"active_share_{unit}", rate))
            folds.append(test.select(ENTRY).with_columns(estimates))
    return pl.concat(folds)


def distribute(pred: pl.DataFrame) -> pl.DataFrame:
    rows = (
        pred.with_columns(
            (c("play_probability") * c(f"active_share_{u}")).alias(f"uncapped_{u}")
            for u in UNITS
        )
        .with_columns(
            (
                c(f"uncapped_{u}")
                / c(f"uncapped_{u}").sum().over(SIDE).clip(lower_bound=1)
            ).alias(f"allocated_share_{u}")
            for u in UNITS
        )
        .with_columns(
            (c(f"allocated_share_{u}") * c(f"budget_{u}")).alias(f"workload_{u}")
            for u in UNITS
        )
    )
    for method in ("linear", "forest"):
        adjust = []
        for ch, unit in (("passing", "attempts"), ("receiving", "targets")):
            # Price only the change in unconditional volume, preserving the
            # control's participation-weighted zero-opportunity allowance.
            delta = c(f"workload_{unit}") - c(f"{method}_{unit}")
            for stat, rate in (("yards", "rate"), ("tds", "td_rate")):
                key = f"{ch}_{stat}"
                change = delta * c(f"{method}_{ch}_{rate}")
                rows = rows.with_columns(
                    (c(f"{method}_{key}") + change).alias(f"workload_{method}_{key}")
                )
                adjust.append(change * WEIGHTS[key])
        rows = rows.with_columns(
            (c(f"factor_{method}") + pl.sum_horizontal(adjust)).alias(
                f"workload_{method}"
            )
        )
    return rows.with_columns(
        pl.mean_horizontal(
            "ridge_role", "boost_role", "workload_linear", "workload_forest"
        ).alias("workload_ensemble")
    )


def workload_replay(
    pred: pl.DataFrame, roles: pl.DataFrame, games: pl.DataFrame, end_year: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    with threadpool_limits(limits=2):
        budgets = team_forecasts(games, end_year)
        shares = active_shares(roles, end_year)
    joined = pred.join(shares, on=ENTRY, validate="1:1").join(
        budgets.select(
            *SIDE,
            *(c(u).alias(f"actual_team_{u}") for u in UNITS),
            *(f"budget_{u}" for u in UNITS),
        ),
        on=SIDE,
        validate="m:1",
    )
    if joined.height != pred.height:
        raise DataAcquisitionError("Missing passing workload for an issued role")
    return distribute(joined).sort("season", "week", *ENTRY), budgets


def _volume_tables(
    verified: pl.DataFrame, teams: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    player_rows, team_rows, paired_rows, diagnostic = [], [], [], []
    for u in UNITS:
        years = []
        for method in ("linear", "forest", "workload"):
            pred = c(f"{method}_{u}")
            error = pred - c(u)
            share = (
                c(f"allocated_share_{u}")
                if method == "workload"
                else pred / pl.when(c(f"budget_{u}") > 0).then(c(f"budget_{u}"))
            )
            years.append(
                verified.group_by("season", "position").agg(
                    pl.lit(method).alias("model"),
                    pl.len().alias("player_games"),
                    error.abs().mean().alias("mae"),
                )
            )
            player_rows.append(
                verified.group_by("position").agg(
                    pl.lit(u).alias("stat"),
                    pl.lit(method).alias("model"),
                    pl.len().alias("games"),
                    error.abs().mean().alias("mae"),
                    error.pow(2).mean().sqrt().alias("rmse"),
                    error.mean().alias("bias"),
                    ((share - c(u) / c(f"actual_team_{u}")).abs().mean() * 100).alias(
                        "share_mae_pp"
                    ),
                )
            )
        annual = pl.concat(years)
        paired_rows.extend(
            paired_mae(
                annual.filter(c("model").is_in(["workload", base])), reference=base
            )
            .filter(c("model") == "workload")
            .with_columns(pl.lit(u).alias("stat"))
            for base in ("linear", "forest")
        )
        true_total, budget, frac = (
            c(f"actual_team_{u}"),
            c(f"budget_{u}"),
            c(f"allocated_share_{u}"),
        )
        diagnostic.append(
            verified.group_by("position").agg(
                pl.lit(u).alias("stat"),
                (budget * frac - c(u)).abs().mean().alias("forecast_mae"),
                (true_total * frac - c(u)).abs().mean().alias("known_team_total_mae"),
                ((budget - true_total) * c(u) / true_total)
                .abs()
                .mean()
                .alias("known_player_share_mae"),
            )
        )
        for method, col in (
            ("offense_last8", f"own_{u}_8"),
            ("team_ridge", f"budget_{u}"),
        ):
            error = c(col) - c(u)
            team_rows.append(
                teams.select(
                    pl.lit(u).alias("stat"),
                    pl.lit(method).alias("model"),
                    pl.len().alias("team_games"),
                    error.abs().mean().alias("mae"),
                    error.pow(2).mean().sqrt().alias("rmse"),
                    error.mean().alias("bias"),
                )
            )
    return {
        "player-volume.csv": pl.concat(player_rows).sort("position", "stat", "model"),
        "volume-paired.csv": pl.concat(paired_rows).sort(
            "position", "stat", "reference"
        ),
        "volume-diagnostics.csv": pl.concat(diagnostic).sort("position", "stat"),
        "team-volume.csv": pl.concat(team_rows).sort("stat", "model"),
        "teams.parquet": teams,
    }


def workload_tables(
    forecasts: pl.DataFrame, teams: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    verified = forecasts.drop_nulls("points")
    annual, scores = method_scores(verified, [*NAMES, *PAIRS])
    paired = pl.concat(
        paired_mae(annual.filter(c("model").is_in([new, old])), reference=old).filter(
            c("model") == new
        )
        for new, old in PAIRS.items()
    )
    return {
        "scores.csv": scores,
        "annual.csv": annual,
        "paired.csv": paired.sort("position", "model"),
        **_volume_tables(verified, teams),
    }

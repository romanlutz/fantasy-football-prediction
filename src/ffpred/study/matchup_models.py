"""Expanding-season expected points, opportunity/rate models and outcome intervals."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ffpred.study.benchmark import BLOCKS, histogram_booster
from ffpred.study.matchup_pool import SNAP_START
from ffpred.study.panel import CHANNELS, OTHER, POSITIONS, STATS, WEIGHTS
from ffpred.study.run import PAPER_YEAR, TRAIN_START

FEATURES = [
    *BLOCKS["paper"],
    "expected_usage",
    "team_rank",
    "depth_rank",
    "dated_depth",
    "opener",
    "prior_absence",
    "questionable",
    "limited",
    "doubtful",
    "current_qb1",
    "announced",
    "history_gap",
    "volatility",
    *(f"lag_{v}" for v in STATS[1::2]),
]
NAMES = [
    "trailing",
    "ridge_role",
    "boost_broad",
    "boost_role",
    "factor_linear",
    "factor_forest",
    "ensemble",
]
Q = (0.1, 0.25, 0.75, 0.9)
c = pl.col
Array = NDArray[np.float64]


def _ridge(x: Array, y: Array, xt: Array, weights: Array | None = None) -> Array:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    return (
        Ridge(alpha=20)
        .fit(scaler.transform(x), y, sample_weight=weights)
        .predict(scaler.transform(xt))
    )


def _mean_fit(
    x: Array, y: Array, xt: Array, *, forest: bool, weights: Array | None = None
) -> Array:
    if not forest:
        return _ridge(x, y, xt, weights)
    scale = np.maximum(np.std(y, axis=0), 0.01)
    model = ExtraTreesRegressor(
        n_estimators=24, max_depth=10, min_samples_leaf=20, n_jobs=1, random_state=42
    )
    return model.fit(x, y / scale, sample_weight=weights).predict(xt) * scale


def _factor(
    train: pl.DataFrame, x: Array, xt: Array, p: Array, *, forest: bool
) -> dict[str, Array]:
    volume = train.select(CHANNELS.values()).to_numpy()
    nhat = np.maximum(0, _mean_fit(x, volume, xt, forest=forest))
    pred = {}
    for i, (ch, unit) in enumerate(CHANNELS.items()):
        n = volume[:, i]
        y = train.select(f"{ch}_yards", f"{ch}_tds").to_numpy()
        use = n > 0
        # Estimate E[total|X,played] / E[opportunities|X,played], not an
        # unweighted average of per-game ratios; zero-volume lateral plays survive.
        rates = (
            _mean_fit(x[use], y[use] / n[use, None], xt, forest=forest, weights=n[use])
            if np.any(use)
            else np.zeros((len(xt), 2))
        )
        rates[:, 1] = np.clip(rates[:, 1], 0, 1)
        orphan = y[~use].sum(axis=0) / len(y)
        pred[f"active_{unit}"] = nhat[:, i]
        pred[unit] = p * nhat[:, i]
        pred[f"{ch}_rate"] = rates[:, 0]
        pred[f"{ch}_yards"] = p * (nhat[:, i] * rates[:, 0] + orphan[0])
        pred[f"{ch}_tds"] = p * (nhat[:, i] * rates[:, 1] + orphan[1])
    other = _mean_fit(x, train.select(OTHER).to_numpy(), xt, forest=forest)
    for i, stat in enumerate(OTHER):
        pred[stat] = p * np.maximum(0, other[:, i])
    pred["points"] = sum(
        (pred[k] * v for k, v in WEIGHTS.items()), start=np.zeros(len(p))
    )
    return pred


def _intervals(
    train: pl.DataFrame, cal: pl.DataFrame, test: pl.DataFrame, features: list[str]
) -> dict[str, Array]:
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    x = impute.fit_transform(train.select(features))
    xt = impute.transform(pl.concat([cal.select(features), test.select(features)]))
    fitted = np.column_stack(
        [
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=60,
                max_leaf_nodes=12,
                min_samples_leaf=30,
                learning_rate=0.07,
                early_stopping=False,
                random_state=42,
            )
            .fit(x, train["points"].to_numpy())
            .predict(xt)
            for q in Q
        ]
    )
    fitted.sort(axis=1)
    prior, current = fitted[: len(cal)], fitted[len(cal) :].copy()
    y = cal["points"].to_numpy()
    for left, right, level in [(0, 3, 0.8), (1, 2, 0.5)]:
        score = np.maximum(prior[:, left] - y, y - prior[:, right])
        rank = min(len(y), int(np.ceil((len(y) + 1) * level)))
        widen = max(0, np.partition(score, rank - 1)[rank - 1])
        current[:, left] -= widen
        current[:, right] += widen
    current[:, 0] = np.minimum(current[:, 0], current[:, 1])
    current[:, 3] = np.maximum(current[:, 3], current[:, 2])
    return {f"q{round(q * 100)}": current[:, i] for i, q in enumerate(Q)}


def _fold(group: pl.DataFrame, year: int, features: list[str]) -> pl.DataFrame:
    earlier = group.filter((c("season") < year) & c("points").is_not_null())
    broad = earlier.filter(c("season") >= SNAP_START)
    roles = broad.filter(c("team_role"))
    test = group.filter((c("season") == year) & c("team_role"))
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    x = impute.fit_transform(roles.select(features))
    xt = impute.transform(test.select(features))
    y = roles["points"].to_numpy()
    preds = {
        "ridge_role": _ridge(x, y, xt),
        "boost_role": histogram_booster().fit(x, y).predict(xt),
    }
    broad_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), histogram_booster()
    )
    preds["boost_broad"] = broad_model.fit(
        broad.select(features), broad["points"].to_numpy()
    ).predict(test.select(features))
    preds["trailing"] = test["last10_points"].fill_null(float(y.mean())).to_numpy()
    known = roles["played"].is_not_null().to_numpy()
    played = roles["played"].fill_null(False).to_numpy()
    classes = np.unique(played[known])
    if len(classes) == 1:
        prob = np.full(test.height, float(classes[0]))
    else:
        prob = (
            make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
            .fit(x[known], played[known])
            .predict_proba(xt)[:, 1]
        )
    preds["play_probability"] = prob
    for label, forest in (("linear", False), ("forest", True)):
        stats = _factor(roles.filter(c("played")), x[played], xt, prob, forest=forest)
        preds[f"factor_{label}"] = stats.pop("points")
        preds.update({f"{label}_{name}": v for name, v in stats.items()})
    preds["ensemble"] = np.mean(
        [
            preds[k]
            for k in ("ridge_role", "boost_role", "factor_linear", "factor_forest")
        ],
        axis=0,
    )
    preds.update(
        _intervals(
            earlier.filter(
                (c("season") < year - 1) & c("team_role") & (c("season") >= TRAIN_START)
            ),
            earlier.filter((c("season") == year - 1) & c("team_role")),
            test,
            features,
        )
    )
    return test.select(
        "season",
        "week",
        "game_id",
        "player_id",
        "name",
        "position",
        "team",
        "actual_team",
        *(
            ["actual_game_id", "outcome_status", "snap_team", "roster_team", "opponent"]
            if "outcome_status" in test.columns
            else []
        ),
        "opener",
        "team_rank",
        "prior_absence",
        "volatility",
        "cutoff",
        "points",
        "played",
        *WEIGHTS,
        *CHANNELS.values(),
    ).with_columns(pl.Series(k, v) for k, v in preds.items())


def project(
    board: pl.DataFrame,
    end_year: int,
    *,
    extra_features: tuple[str, ...] = (),
    positions: tuple[str, ...] = POSITIONS,
) -> pl.DataFrame:
    with threadpool_limits(limits=2):
        return pl.concat(
            [
                _fold(
                    board.filter(c("position") == pos),
                    year,
                    [*FEATURES, *extra_features],
                )
                for year in range(PAPER_YEAR, end_year + 1)
                for pos in positions
            ]
        )

"""Paired historical and modern regressors on the frozen team-role cohort."""

from __future__ import annotations

import numpy as np
import polars as pl
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from threadpoolctl import threadpool_limits

from ffpred.errors import DataAcquisitionError
from ffpred.study.benchmark import BLOCKS, histogram_booster
from ffpred.study.bridge import PAPER_YEAR

KEY = ["season", "week", "game_id", "team", "position", "player_id"]
REFERENCE = ["trailing10", "svr_paper", "boost_paper"]
CHALLENGERS = [
    "rbf_approx",
    "mlp_sigmoid",
    "mlp_relu",
    "ridge",
    "boost_absolute",
    "ebm",
]
FEATURES = BLOCKS["paper"]


def _models(
    n_train: int,
) -> dict[str, Pipeline | HistGradientBoostingRegressor | ExplainableBoostingRegressor]:
    return {
        "rbf_approx": make_pipeline(
            StandardScaler(),
            Nystroem(gamma=0.05, n_components=min(128, n_train), random_state=42),
            LinearSVR(C=0.25, epsilon=0.25, max_iter=20_000, random_state=42),
        ),
        "mlp_sigmoid": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(50,),
                activation="logistic",
                solver="sgd",
                learning_rate_init=0.01,
                momentum=0,
                alpha=0,
                max_iter=50,
                n_iter_no_change=51,
                random_state=42,
            ),
        ),
        "mlp_relu": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="adam",
                max_iter=50,
                n_iter_no_change=51,
                random_state=42,
            ),
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "boost_absolute": histogram_booster("absolute_error"),
        "ebm": ExplainableBoostingRegressor(
            interactions=0,
            max_bins=64,
            outer_bags=1,
            validation_size=0,
            max_rounds=200,
            smoothing_rounds=0,
            learning_rate=0.04,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=42,
        ),
    }


def compare_methods(
    train_panel: pl.DataFrame, forecast_panel: pl.DataFrame, reference: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    c = pl.col
    ref = reference.filter(c("team_role")).select(*KEY, "points", *REFERENCE)
    selected = forecast_panel.filter(c("team_role")).select(*KEY, "points", *FEATURES)
    data = selected.join(ref, on=KEY, validate="1:1")
    if (
        data.height != ref.height
        or data.height != selected.height
        or not data["points"].equals(data["points_right"])
    ):
        raise DataAcquisitionError(
            "Model comparison requires identical team-role rows and scores"
        )
    frames = []
    exact = []
    with threadpool_limits(limits=2):
        for (year, pos), test in (
            data.sort(KEY).partition_by(["season", "position"], as_dict=True).items()
        ):
            train = train_panel.filter(
                (c("season") < year) & (c("position") == pos)
            ).sort(KEY)
            x, xt = (f.select(FEATURES).to_numpy() for f in (train, test))
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x = imputer.fit_transform(x)
            xt = imputer.transform(xt)
            y = train["points"].to_numpy()
            results = test.select(*KEY, "points", *REFERENCE)
            for name, model in _models(train.height).items():
                results = results.with_columns(
                    pl.Series(name, model.fit(x, y).predict(xt))
                )
            frames.append(results)
            if year == PAPER_YEAR:
                model = make_pipeline(
                    StandardScaler(),
                    SVR(C=0.25, epsilon=0.25, kernel="rbf", gamma=0.05),
                )
                exact.append(
                    results.select(*KEY, "points", "rbf_approx").with_columns(
                        pl.Series("rbf_exact", model.fit(x, y).predict(xt))
                    )
                )
    predictions = pl.concat(frames)
    annual, pooled = method_scores(predictions, [*REFERENCE, *CHALLENGERS])
    _, kernel = method_scores(pl.concat(exact), ["rbf_approx", "rbf_exact"])
    return {
        "method-predictions.parquet": predictions,
        "methods-by-season.csv": annual,
        "methods-pooled.csv": pooled,
        "methods-paired-mae.csv": paired_mae(annual),
        "methods-exact-rbf-2014.csv": kernel,
    }


def method_scores(
    pred: pl.DataFrame, names: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    c = pl.col
    if not np.isfinite(pred.select("points", *names).to_numpy()).all():
        raise DataAcquisitionError("Nonfinite point forecast or outcome")
    rows = pred.unpivot(
        index=["season", "week", "position", "points"],
        on=names,
        variable_name="model",
        value_name="forecast",
    ).with_columns((c("forecast") - c("points")).alias("error"))
    group = ["position", "model"]
    weekly = rows.group_by("season", "week", *group).agg(
        pl.corr("forecast", "points", method="spearman").fill_nan(None).alias("rho")
    )

    def summary(keys: list[str]) -> pl.DataFrame:
        return (
            rows.group_by(keys)
            .agg(
                pl.len().alias("player_games"),
                c("error").abs().mean().alias("mae"),
                c("error").pow(2).mean().sqrt().alias("rmse"),
                c("error").mean().alias("bias"),
            )
            .join(
                weekly.group_by(keys).agg(
                    c("rho").mean().alias("weekly_spearman"),
                    c("rho").count().alias("ranked_weeks"),
                ),
                on=keys,
            )
            .sort(keys)
        )

    return summary(["season", *group]), summary(group)


def paired_mae(annual: pl.DataFrame) -> pl.DataFrame:
    """Resample paired seasons, preserving player-game weights within each draw."""
    scores = []
    for (position,), group in annual.partition_by("position", as_dict=True).items():
        base = group.filter(pl.col("model") == "svr_paper").sort("season")
        n = base["player_games"].to_numpy()
        weights = (
            np.random.default_rng(42).multinomial(
                len(n), np.ones(len(n)) / len(n), size=4000
            )
            * n
        )
        for (model,), model_years in group.partition_by("model", as_dict=True).items():
            frame = model_years.sort("season")
            if not frame.select("season", "player_games").equals(
                base.select("season", "player_games")
            ):
                raise DataAcquisitionError(
                    "Unpaired model comparison seasons or counts"
                )
            delta = (frame["mae"] - base["mae"]).to_numpy()
            low, high = np.quantile(
                weights @ delta / weights.sum(axis=1), [0.025, 0.975]
            )
            scores.append(
                {
                    "position": position,
                    "model": model,
                    "reference": "svr_paper",
                    "mae_delta": np.average(delta, weights=n),
                    "low95": low,
                    "high95": high,
                    "seasons_better": int((delta < 0).sum()),
                    "seasons": len(n),
                }
            )
    return pl.DataFrame(scores).sort("position", "model")


def role_error_slices(
    pred: pl.DataFrame,
    selections: pl.DataFrame,
    actual: pl.DataFrame,
    absences: pl.DataFrame,
) -> pl.DataFrame:
    """Score subsets after prediction, including unsuccessful returns."""
    c = pl.col
    rows = (
        pred.join(selections.select(*KEY, "roster_week"), on=KEY, validate="1:1")
        .join(actual.select(*KEY, "injury_return"), on=KEY, how="left", validate="1:1")
        .join(
            absences.rename({"week": "roster_week"}).with_columns(
                pl.lit(True).alias("missed_prior")
            ),
            on=["season", "roster_week", "team", "player_id"],
            how="left",
            validate="m:1",
        )
    )
    if rows.height != pred.height:
        raise DataAcquisitionError("Missing team-role rows in forecast diagnostics")
    masks = {
        "all": pl.lit(True),
        "no_trade_conflict": ~pl.struct("season", "week", "player_id").is_duplicated(),
        "missed_previous_game_injured": c("missed_prior"),
        "realized_return_role": c("injury_return"),
        "correct_role": c("injury_return").is_not_null(),
        "missed_role": c("injury_return").is_null(),
    }
    slices = []
    for label, mask in masks.items():
        part = rows.filter(mask).unpivot(
            index=["position", "points"],
            on=[*REFERENCE, *CHALLENGERS],
            variable_name="model",
            value_name="forecast",
        )
        err = c("forecast") - c("points")
        slices.append(
            part.group_by("position", "model").agg(
                pl.lit(label).alias("cohort"),
                pl.len().alias("slots"),
                (c("points") == 0).sum().alias("zero_scores"),
                (c("points") < 0).sum().alias("negative_scores"),
                err.abs().mean().alias("mae"),
                err.pow(2).mean().sqrt().alias("rmse"),
                err.mean().alias("bias"),
            )
        )
    return pl.concat(slices).sort("cohort", "position", "model")

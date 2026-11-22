"""Expanding-window, same-row benchmark for the weekly forecasting study."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from ffpred.study.defense import ADJUSTED, PAPER
from ffpred.study.panel import PLAYER_FEATURES, PLAYER_WEEK, POSITIONS
from ffpred.training.svr import (
    DEFAULT_SVR_CONFIG,
    create_estimator,
    create_scalable_estimator,
)

BLOCKS = {
    "none": PLAYER_FEATURES,
    "paper": (*PLAYER_FEATURES, *PAPER),
    "adjusted": (*PLAYER_FEATURES, *PAPER, *ADJUSTED),
}


def histogram_booster(
    loss: Literal["squared_error", "absolute_error"] = "squared_error",
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        learning_rate=0.07,
        early_stopping=False,
        random_state=42,
    )


def backtest(
    panel: pl.DataFrame,
    test_years: range,
    *,
    exact_svr: bool = False,
    forecast_panel: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fit on completed earlier seasons. All columns below predict the same rows."""
    frames = []
    with threadpool_limits(limits=2):
        for year in test_years:
            for position in POSITIONS:
                group = panel.filter(pl.col("position") == position)
                train = group.filter(pl.col("season") < year)
                test = (
                    group
                    if forecast_panel is None
                    else forecast_panel.filter(pl.col("position") == position)
                ).filter(pl.col("season") == year)
                identity = ["team"] if forecast_panel is not None else []
                target = train["points"].to_numpy()
                forecast = test.select(
                    *PLAYER_WEEK, "game_id", *identity, "position", "points"
                ).with_columns(
                    test["last10_points"]
                    .fill_null(float(np.mean(target)))
                    .alias("trailing10")
                )
                for block, features in BLOCKS.items():
                    x, xt = (f.select(features).to_numpy() for f in (train, test))
                    models = {
                        "svr": (
                            create_estimator(DEFAULT_SVR_CONFIG)
                            if exact_svr
                            else create_scalable_estimator()
                        ),
                        "boost": histogram_booster(),
                    }
                    for name, model in models.items():
                        estimator = make_pipeline(
                            SimpleImputer(strategy="median", keep_empty_features=True),
                            model,
                        )
                        pred = estimator.fit(x, target).predict(xt)
                        forecast = forecast.with_columns(
                            pl.Series(f"{name}_{block}", pred)
                        )
                frames.append(forecast)
    return pl.concat(frames)


def scorecard(predictions: pl.DataFrame, *, by_year: bool) -> pl.DataFrame:
    keys = ["position", "model"]
    if by_year:
        keys.insert(0, "season")
    long = predictions.unpivot(
        index=["season", "position", "points"],
        on=["trailing10", *(f"{m}_{b}" for b in BLOCKS for m in ("svr", "boost"))],
        variable_name="model",
        value_name="forecast",
    ).with_columns((pl.col("forecast") - pl.col("points")).alias("error"))
    return (
        long.group_by(keys)
        .agg(
            pl.len().alias("player_games"),
            pl.col("error").abs().mean().alias("mae"),
            pl.col("error").pow(2).mean().sqrt().alias("rmse"),
            pl.col("error").mean().alias("bias"),
        )
        .sort(keys)
    )

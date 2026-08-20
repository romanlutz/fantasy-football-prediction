import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from ffpred.evaluation.cohorts import residual_cohorts
from ffpred.evaluation.explainability import (
    accumulated_local_effects,
    conformal_prediction_interval,
    model_agnostic_shap_values,
    temporal_permutation_importance,
)


def test_conformal_interval_uses_finite_sample_corrected_residual_quantile() -> None:
    interval = conformal_prediction_interval(
        [10, 10, 10, 10],
        [9, 12, 9, 12],
        [5, 8],
        coverage=0.75,
    )

    assert interval.radius == 2
    np.testing.assert_array_equal(interval.lower, [3, 6])
    np.testing.assert_array_equal(interval.upper, [7, 10])


def test_ale_curve_recovers_increasing_linear_effect() -> None:
    feature = np.linspace(0, 10, 40)
    noise = np.tile([0.0, 1.0], 20)
    features = np.column_stack((feature, noise))
    estimator = LinearRegression().fit(features, 2 * feature)

    curve = accumulated_local_effects(
        estimator,
        features,
        0,
        "signal",
        bins=5,
    )

    assert curve.effects == tuple(sorted(curve.effects))
    assert sum(curve.samples) == features.shape[0]
    assert np.average(curve.effects, weights=curve.samples) == pytest.approx(0)


def test_temporal_permutation_importance_identifies_signal() -> None:
    signal = np.tile(np.arange(10, dtype=float), 2)
    noise = np.tile([0.0, 1.0], 10)
    features = np.column_stack((signal, noise))
    target = 3 * signal
    periods = np.repeat([2024, 2025], 10)
    estimator = LinearRegression().fit(features, target)

    importance = temporal_permutation_importance(
        estimator,
        features,
        target,
        periods,
        ("signal", "noise"),
        repeats=3,
        random_state=7,
    )

    assert importance[0].feature == "signal"
    assert importance[0].mean_mae_increase > 0
    assert importance[1].mean_mae_increase == pytest.approx(0, abs=1e-12)


def test_model_agnostic_shap_values_reconstruct_predictions() -> None:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    target = 2 * features[:, 0] - features[:, 1]
    estimator = LinearRegression().fit(features, target)

    explanation = model_agnostic_shap_values(
        estimator,
        features,
        features[:2],
        ("first", "second"),
        max_background=4,
        max_samples=2,
        random_state=7,
    )

    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, estimator.predict(explanation.data))
    assert explanation.sample_indices == (0, 1)


def test_residual_cohorts_report_position_week_and_numeric_bins() -> None:
    frame = pl.DataFrame(
        {
            "position": ["WR", "WR", "TE", "TE"],
            "target_week": [1, 2, 1, 2],
            "years_pro": [1.0, 2.0, 8.0, 9.0],
        }
    )

    cohorts = residual_cohorts(
        frame,
        [10, 10, 10, 10],
        [8, 12, 9, 9],
        categorical_columns=("position", "target_week"),
        quantile_columns=("years_pro",),
        quantiles=2,
    )

    wr = next(
        cohort
        for cohort in cohorts
        if cohort.column == "position" and cohort.value == "WR"
    )
    assert wr.samples == 2
    assert wr.bias == 0
    assert wr.mae == 2
    assert {cohort.value for cohort in cohorts if cohort.column == "years_pro"} == {
        "Q1",
        "Q2",
    }

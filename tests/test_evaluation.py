import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ffpred.evaluation.metrics import evaluate, mean_relative_error


def test_evaluate_reports_shared_metric_contract() -> None:
    result = evaluate([10, 20], [12, 16])

    assert result.rmse == pytest.approx((20 / 2) ** 0.5)
    assert result.mae == 3
    assert result.mre == pytest.approx(0.2)
    assert result.samples == 2


@given(
    values=st.lists(
        st.floats(
            min_value=0.1,
            max_value=1_000,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=20,
    ),
    scale=st.floats(
        min_value=0.1,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_mean_relative_error_is_scale_invariant(
    values: list[float],
    scale: float,
) -> None:
    actual = np.asarray(values)
    prediction = actual * 1.1

    assert mean_relative_error(actual, prediction) == pytest.approx(
        mean_relative_error(actual * scale, prediction * scale)
    )


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate([1, 2], [1])

"""Named evaluation cohorts retained from the original experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from ffpred.domain.identifiers import PlayerId

MINIMUM_QUANTILES = 2

LEGACY_2014_QUARTERBACKS = frozenset(
    PlayerId(player_id)
    for player_id in (
        "00-0029263",
        "00-0023459",
        "00-0026143",
        "00-0020531",
        "00-0026158",
        "00-0027973",
        "00-0024226",
        "00-0023436",
        "00-0029701",
        "00-0019596",
        "00-0031280",
        "00-0022924",
        "00-0026625",
        "00-0021678",
        "00-0027974",
        "00-0010346",
        "00-0029668",
        "00-0026498",
        "00-0022803",
        "00-0022942",
        "00-0027939",
        "00-0031237",
        "00-0031407",
        "00-0023541",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidualCohort:
    """Error metrics for one categorical or numeric cohort."""

    column: str
    value: str
    samples: int
    bias: float
    mae: float
    rmse: float
    lower_bound: float | None = None
    upper_bound: float | None = None


def residual_cohorts(  # noqa: PLR0913
    frame: pl.DataFrame,
    actual: ArrayLike,
    prediction: ArrayLike,
    *,
    categorical_columns: tuple[str, ...] = (),
    quantile_columns: tuple[str, ...] = (),
    quantiles: int = 3,
) -> tuple[ResidualCohort, ...]:
    """Summarize held-out residuals by named and quantile-binned cohorts."""
    if quantiles < MINIMUM_QUANTILES:
        raise ValueError("numeric cohorts require at least two quantiles")
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if (
        actual_values.shape != predicted_values.shape
        or frame.height != actual_values.size
    ):
        raise ValueError("frame, actual, and prediction must have the same rows")

    cohorts: list[ResidualCohort] = []
    for column in categorical_columns:
        _require_column(frame, column)
        values = frame[column].to_numpy()
        for value in sorted(set(values), key=str):
            rows = np.flatnonzero(values == value)
            cohorts.append(
                _residual_cohort(
                    column,
                    str(value),
                    rows,
                    actual_values,
                    predicted_values,
                )
            )

    for column in quantile_columns:
        _require_column(frame, column)
        values = np.asarray(frame[column].to_numpy(), dtype=np.float64)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))
        if edges.size == 1:
            cohorts.append(
                _residual_cohort(
                    column,
                    "Q1",
                    np.arange(frame.height),
                    actual_values,
                    predicted_values,
                    lower_bound=float(edges[0]),
                    upper_bound=float(edges[0]),
                )
            )
            continue
        assignments = np.searchsorted(edges[1:-1], values, side="right")
        for index in range(edges.size - 1):
            rows = np.flatnonzero(assignments == index)
            if rows.size == 0:
                continue
            cohorts.append(
                _residual_cohort(
                    column,
                    f"Q{index + 1}",
                    rows,
                    actual_values,
                    predicted_values,
                    lower_bound=float(edges[index]),
                    upper_bound=float(edges[index + 1]),
                )
            )
    return tuple(cohorts)


def _residual_cohort(  # noqa: PLR0913
    column: str,
    value: str,
    rows: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> ResidualCohort:
    residual = actual[rows] - prediction[rows]
    return ResidualCohort(
        column=column,
        value=value,
        samples=rows.size,
        bias=float(np.mean(residual)),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.mean(residual**2) ** 0.5),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _require_column(frame: pl.DataFrame, column: str) -> None:
    if column not in frame:
        raise ValueError(f"cohort column {column!r} is missing")

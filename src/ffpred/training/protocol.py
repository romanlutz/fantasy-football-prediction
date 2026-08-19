"""Structural model interfaces."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray


class Regressor(Protocol):
    """Minimum estimator surface used by the training layer."""

    def fit(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> Self: ...

    def predict(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

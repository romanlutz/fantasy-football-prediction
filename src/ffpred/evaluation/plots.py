# Copyright (c) 2026 Roman Lutz
# SPDX-License-Identifier: MIT

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import ArrayLike

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def histogram(
    y_vals: ArrayLike,
    prediction: ArrayLike,
    output: Path = Path("absolute_error_distribution.pdf"),
) -> None:
    errors = np.abs(
        np.asarray(y_vals, dtype=float) - np.asarray(prediction, dtype=float)
    )
    plt.hist(errors, bins=range(35), rwidth=1.0, histtype="bar")
    plt.xlabel("Absolute Error")
    plt.ylabel("Number of data cases")
    plt.title("Absolute Error Distribution")
    plt.savefig(output)
    plt.close("all")

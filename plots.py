# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def histogram(
    y_vals: np.ndarray,
    prediction: np.ndarray,
    output: Path = Path("absolute_error_distribution.pdf"),
) -> None:
    errors = np.abs(np.asarray(y_vals, dtype=float) - np.asarray(prediction, dtype=float))
    plt.hist(errors, bins=range(35), rwidth=1.0, histtype="bar")
    plt.xlabel("Absolute Error")
    plt.ylabel("Number of data cases")
    plt.title("Absolute Error Distribution")
    plt.savefig(output)
    plt.close("all")

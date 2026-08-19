# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

import numpy as np


def mean_relative_error(y: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    nonzero = actual != 0
    if not np.any(nonzero):
        raise ValueError("Mean relative error is undefined when all targets are zero")
    return float(
        np.mean(np.abs(predicted[nonzero] - actual[nonzero]) / np.abs(actual[nonzero]))
    )

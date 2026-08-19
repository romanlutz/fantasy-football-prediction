# Copyright (c) 2026 Roman Lutz
# SPDX-License-Identifier: MIT

import numpy as np


def mean_relative_error(y: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    nonzero = actual != 0
    if not np.any(nonzero):
        raise ValueError("Mean relative error is undefined when all targets are zero")
    return float(np.mean(np.abs(predicted[nonzero] - actual[nonzero]) / np.abs(actual[nonzero])))

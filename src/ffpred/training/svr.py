# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ffpred.acquisition.nflverse import test_players
from ffpred.evaluation.metrics import mean_relative_error
from ffpred.evaluation.plots import histogram

MANUAL_FEATURE_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    10,
    13,
    14,
    15,
    16,
    17,
    20,
    21,
    22,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
]


def load_data(
    train_path: Path, test_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(train_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    if train.ndim != 2 or train.shape[1] != 37:
        raise ValueError(f"{train_path} must contain rows with 37 columns")
    if test.ndim != 2 or test.shape[1] != 37:
        raise ValueError(f"{test_path} must contain rows with 37 columns")
    train_x = train[:, 2:36].astype(float)
    train_y = train[:, 36].astype(float)
    test_x = test[:, 2:36].astype(float)
    test_y = test[:, 36].astype(float)
    return train_x, train_y, test_x, test_y, test


def hyperparameter_selection(
    regressors: list[SVR], x: np.ndarray, y: np.ndarray, folds: int
) -> SVR:
    cross_validation = KFold(n_splits=folds, shuffle=True, random_state=42)
    average_errors: list[float] = []
    for regressor in regressors:
        errors = []
        for train, validation in cross_validation.split(x):
            scaler = StandardScaler()
            fold_train = scaler.fit_transform(x[train])
            fold_validation = scaler.transform(x[validation])
            prediction = (
                clone(regressor).fit(fold_train, y[train]).predict(fold_validation)
            )
            errors.append(mean_absolute_error(y[validation], prediction))
        average_errors.append(float(np.mean(errors)))
    return regressors[int(np.argmin(average_errors))]


def candidate_regressors() -> list[SVR]:
    regressors: list[SVR] = []
    for c_value, epsilon, kernel in itertools.product(
        (0.25, 0.5, 0.75, 1.0),
        (0.05, 0.1, 0.15, 0.2, 0.25),
        ("rbf", "linear", "sigmoid", "poly"),
    ):
        if kernel == "poly":
            regressors.extend(
                SVR(
                    C=c_value,
                    epsilon=epsilon,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                )
                for gamma in (0.05, 0.1, 0.15)
                for degree in (2, 3)
            )
        elif kernel in {"rbf", "sigmoid"}:
            regressors.extend(
                SVR(C=c_value, epsilon=epsilon, kernel=kernel, gamma=gamma)
                for gamma in (0.05, 0.1, 0.15)
            )
        else:
            regressors.append(SVR(C=c_value, epsilon=epsilon, kernel=kernel))
    return regressors


def evaluate(actual: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_squared_error(actual, prediction) ** 0.5),
        float(mean_absolute_error(actual, prediction)),
        mean_relative_error(actual, prediction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an SVR model")
    parser.add_argument("--train", type=Path, default=Path("train.npy"))
    parser.add_argument("--test", type=Path, default=Path("test.npy"))
    parser.add_argument(
        "--prediction-output", type=Path, default=Path("prediction.npy")
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        default=Path("absolute_error_distribution.pdf"),
    )
    parser.add_argument(
        "--feature-selection",
        choices=("none", "manual", "rfecv"),
        default="none",
    )
    parser.add_argument("--select-hyperparameters", action="store_true")
    parser.add_argument("--no-histogram", action="store_true")
    parser.add_argument("--show-predictions", action="store_true")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y, test = load_data(args.train, args.test)
    if args.feature_selection == "rfecv":
        selector = RFECV(
            estimator=SVR(kernel="linear"),
            step=3,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        train_x = selector.fit_transform(train_x, train_y)
        test_x = selector.transform(test_x)
        print("Feature rankings:", selector.ranking_)
    elif args.feature_selection == "manual":
        train_x = train_x[:, MANUAL_FEATURE_INDICES]
        test_x = test_x[:, MANUAL_FEATURE_INDICES]

    if args.select_hyperparameters:
        regressor = hyperparameter_selection(
            candidate_regressors(), train_x, train_y, 5
        )
    else:
        regressor = SVR(C=0.25, epsilon=0.25, kernel="linear")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    prediction = regressor.fit(train_x, train_y).predict(test_x)
    np.save(args.prediction_output, prediction, allow_pickle=False)

    print("RMSE, MAE, MRE (all):", *evaluate(test_y, prediction))
    selected = np.array(
        [player_id in test_players for player_id in test[:, 0]], dtype=bool
    )
    if np.any(selected):
        print(
            "RMSE, MAE, MRE (selected players):",
            *evaluate(test_y[selected], prediction[selected]),
        )
        if args.show_predictions:
            print(list(zip(test_y[selected], prediction[selected], strict=True)))

    if not args.no_histogram:
        histogram(test_y, prediction, args.histogram_output)


if __name__ == "__main__":
    main()

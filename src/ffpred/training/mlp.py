# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ffpred.acquisition.nflverse import test_players
from ffpred.evaluation.metrics import mean_relative_error
from ffpred.training.svr import load_data


def evaluate(actual: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_squared_error(actual, prediction) ** 0.5),
        float(mean_absolute_error(actual, prediction)),
        mean_relative_error(actual, prediction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate neural-network regressors"
    )
    parser.add_argument("--train", type=Path, default=Path("train.npy"))
    parser.add_argument("--test", type=Path, default=Path("test.npy"))
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument(
        "--hidden-units", type=int, nargs="+", default=[10, 25, 50, 100]
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=("logistic", "tanh"),
        default=["logistic", "tanh"],
    )
    parser.add_argument("--output", type=Path, default=Path("neural_net_output.txt"))
    args = parser.parse_args()

    train_x, train_y, test_x, test_y, test = load_data(args.train, args.test)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    selected = np.array(
        [player_id in test_players for player_id in test[:, 0]], dtype=bool
    )

    with args.output.open("w", encoding="utf-8") as output:
        output.write(
            "epochs hidden_units activation "
            "RMSE_all MAE_all MRE_all RMSE_selected MAE_selected MRE_selected\n"
        )
        for epochs in args.epochs:
            for hidden_units in args.hidden_units:
                for activation in args.activations:
                    model = MLPRegressor(
                        hidden_layer_sizes=(hidden_units,),
                        activation=activation,
                        learning_rate_init=0.01,
                        max_iter=epochs,
                        random_state=42,
                    )
                    prediction = model.fit(train_x, train_y).predict(test_x)
                    all_metrics = evaluate(test_y, prediction)
                    selected_metrics = (
                        evaluate(test_y[selected], prediction[selected])
                        if np.any(selected)
                        else (float("nan"),) * 3
                    )
                    values = (*all_metrics, *selected_metrics)
                    output.write(
                        f"{epochs} {hidden_units} {activation} "
                        + " ".join(f"{value:.6f}" for value in values)
                        + "\n"
                    )


if __name__ == "__main__":
    main()

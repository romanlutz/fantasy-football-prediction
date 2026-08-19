import numpy as np
import polars as pl

from ffpred.features.schema import FEATURE_SCHEMA, TARGET_COLUMN
from ffpred.training.data import training_data_from_frame
from ffpred.training.mlp import MlpConfig, train_mlp
from ffpred.training.svr import SvrConfig, train_svr


def _frame(rows: int, *, season: int) -> pl.DataFrame:
    values: dict[str, list[object]] = {}
    for column, dtype in FEATURE_SCHEMA.items():
        if dtype == pl.String:
            values[column] = [f"{column}-{index}" for index in range(rows)]
        elif dtype == pl.Int64:
            values[column] = [season + index // 4 for index in range(rows)]
        else:
            values[column] = [float(index + 1) for index in range(rows)]
    values["target_week"] = [index % 4 + 1 for index in range(rows)]
    values[TARGET_COLUMN] = [float(index * 2 + 1) for index in range(rows)]
    return pl.DataFrame(values, schema=FEATURE_SCHEMA)


def test_svr_training_returns_typed_deterministic_result() -> None:
    train = training_data_from_frame(_frame(16, season=2020))
    test = training_data_from_frame(_frame(4, season=2025))

    first = train_svr(train, test, config=SvrConfig(kernel="linear"))
    second = train_svr(train, test, config=SvrConfig(kernel="linear"))

    np.testing.assert_allclose(first.predictions, second.predictions)
    assert first.metrics.samples == 4
    assert first.feature_names == train.feature_names


def test_mlp_training_is_reproducible_with_fixed_seed() -> None:
    train = training_data_from_frame(_frame(16, season=2020))
    test = training_data_from_frame(_frame(4, season=2025))
    config = MlpConfig(
        hidden_units=4,
        max_iterations=2_000,
        learning_rate=0.01,
        random_state=7,
    )

    first = train_mlp(train, test, config=config)
    second = train_mlp(train, test, config=config)

    np.testing.assert_allclose(first.predictions, second.predictions)

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ffpred.features.schema import FEATURE_SCHEMA, TARGET_COLUMN
from ffpred.training.data import training_data_from_frame
from ffpred.training.ebm import EbmConfig, train_ebm, write_ebm_explanations
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


def test_ebm_exports_global_shapes_and_additive_local_explanations(
    tmp_path: Path,
) -> None:
    train = training_data_from_frame(_frame(16, season=2020))
    test = training_data_from_frame(_frame(4, season=2025))
    config = EbmConfig(
        max_bins=16,
        interactions=0,
        max_rounds=20,
        learning_rate=0.05,
        min_samples_leaf=2,
        outer_bags=1,
        validation_size=0,
        random_state=7,
        n_jobs=1,
    )

    result = train_ebm(train, test, config=config)
    explanations_path = tmp_path / "explanations.json"
    identities = [{"player_id": f"player-{index}"} for index in range(4)]
    write_ebm_explanations(
        explanations_path,
        result.explanations,
        identities=identities,
    )
    artifact = json.loads(explanations_path.read_text(encoding="utf-8"))

    assert result.metrics.samples == 4
    assert result.prediction_interval is not None
    assert result.prediction_interval.calibration_samples > 0
    assert np.all(result.prediction_interval.lower <= result.prediction_interval.upper)
    assert len(artifact["global"]["terms"]) == len(train.feature_names)
    assert artifact["local"][0]["identity"] == {"player_id": "player-0"}
    assert "scores" in artifact["global"]["terms"][0]["shape"]
    for local in artifact["local"]:
        contribution_sum = local["intercept"] + sum(
            term["contribution"] for term in local["terms"]
        )
        assert contribution_sum == pytest.approx(local["prediction"])


def test_ebm_local_interactions_retain_both_feature_values() -> None:
    train = training_data_from_frame(_frame(24, season=2020))
    test = training_data_from_frame(_frame(4, season=2025))
    result = train_ebm(
        train,
        test,
        config=EbmConfig(
            max_bins=16,
            interactions=2,
            max_rounds=20,
            min_samples_leaf=2,
            outer_bags=1,
            validation_size=0,
            calibration_fraction=0,
            n_jobs=1,
        ),
    )

    interaction = next(
        term for term in result.explanations.local[0].terms if " & " in term.name
    )

    assert isinstance(interaction.value, dict)
    assert set(interaction.value) == set(interaction.name.split(" & "))

import json
from pathlib import Path

import polars as pl
import pytest

from ffpred.cli.app import main
from tests.factories import (
    make_dst_provider,
    make_kicker_provider,
    make_provider,
    make_receiving_provider,
)


def test_build_dataset_command_uses_injected_provider(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "build-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_provider(),
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["train_rows"] == 1
    assert output["test_rows"] == 1


def test_train_and_evaluate_commands_write_parquet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "build-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_provider(),
    )
    capsys.readouterr()
    predictions = tmp_path / "predictions.parquet"

    train_result = main(
        [
            "train-svr",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
        ]
    )
    train_output = json.loads(capsys.readouterr().out)
    evaluation_result = main(["evaluate", str(predictions)])
    evaluation_output = json.loads(capsys.readouterr().out)

    assert train_result == 0
    assert evaluation_result == 0
    assert predictions.exists()
    assert train_output["metrics"]["samples"] == 1
    assert evaluation_output["metrics"] == train_output["metrics"]
    assert "prediction" in pl.read_parquet(predictions).columns


def test_dst_build_train_and_evaluate_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_result = main(
        [
            "build-dst-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_dst_provider(),
    )
    build_output = json.loads(capsys.readouterr().out)
    predictions = tmp_path / "dst-predictions.parquet"

    train_result = main(
        [
            "train-svr",
            "--position",
            "dst",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
        ]
    )
    train_output = json.loads(capsys.readouterr().out)
    evaluation_result = main(["evaluate", str(predictions)])
    evaluation_output = json.loads(capsys.readouterr().out)

    assert build_result == 0
    assert train_result == 0
    assert evaluation_result == 0
    assert build_output["train_rows"] > 0
    assert build_output["test_rows"] > 0
    assert predictions.exists()
    assert evaluation_output["metrics"] == train_output["metrics"]
    prediction_frame = pl.read_parquet(predictions)
    assert "prediction" in prediction_frame.columns
    assert "team" in prediction_frame.columns


def test_manual_features_rejected_for_non_qb_position(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "build-dst-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_dst_provider(),
    )
    capsys.readouterr()

    result = main(
        [
            "train-svr",
            "--position",
            "dst",
            "--manual-features",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
        ]
    )

    assert result == 2


def test_kicker_build_train_and_evaluate_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_result = main(
        [
            "build-kicker-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_kicker_provider(),
    )
    build_output = json.loads(capsys.readouterr().out)
    predictions = tmp_path / "kicker-predictions.parquet"

    train_result = main(
        [
            "train-mlp",
            "--position",
            "k",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
        ]
    )
    train_output = json.loads(capsys.readouterr().out)
    evaluation_result = main(["evaluate", str(predictions)])
    evaluation_output = json.loads(capsys.readouterr().out)

    assert build_result == 0
    assert train_result == 0
    assert evaluation_result == 0
    assert build_output["train_rows"] > 0
    assert build_output["test_rows"] > 0
    assert predictions.exists()
    assert evaluation_output["metrics"] == train_output["metrics"]
    prediction_frame = pl.read_parquet(predictions)
    assert "prediction" in prediction_frame.columns
    assert "player_id" in prediction_frame.columns


def test_receiving_build_train_and_evaluate_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_result = main(
        [
            "build-receiving-dataset",
            "--position",
            "wr",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_receiving_provider(),
    )
    build_output = json.loads(capsys.readouterr().out)
    predictions = tmp_path / "wr-predictions.parquet"

    train_result = main(
        [
            "train-svr",
            "--position",
            "wr",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
        ]
    )
    train_output = json.loads(capsys.readouterr().out)
    evaluation_result = main(["evaluate", str(predictions)])
    evaluation_output = json.loads(capsys.readouterr().out)

    assert build_result == 0
    assert train_result == 0
    assert evaluation_result == 0
    assert build_output["train_rows"] > 0
    assert build_output["test_rows"] > 0
    assert predictions.exists()
    assert evaluation_output["metrics"] == train_output["metrics"]
    prediction_frame = pl.read_parquet(predictions)
    assert "prediction" in prediction_frame.columns
    assert set(pl.read_parquet(tmp_path / "train.parquet")["position"]) == {"WR"}

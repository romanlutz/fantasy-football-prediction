import json
from pathlib import Path

import polars as pl
import pytest

from ffpred.cli.app import main
from tests.factories import make_provider


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

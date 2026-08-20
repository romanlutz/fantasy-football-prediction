import json
from pathlib import Path

import polars as pl
import pytest

from ffpred.cli.app import main
from tests.factories import (
    make_dst_provider,
    make_idp_provider,
    make_injury_report_provider,
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
    explanations = tmp_path / "svr-explanations.json"

    train_result = main(
        [
            "train-svr",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
            "--explanations",
            str(explanations),
            "--shap-samples",
            "1",
            "--shap-background",
            "1",
            "--permutation-repeats",
            "1",
        ]
    )
    train_output = json.loads(capsys.readouterr().out)
    evaluation_result = main(["evaluate", str(predictions)])
    evaluation_output = json.loads(capsys.readouterr().out)

    assert train_result == 0
    assert evaluation_result == 0
    assert predictions.exists()
    assert explanations.exists()
    explanation_data = json.loads(explanations.read_text(encoding="utf-8"))
    assert train_output["metrics"]["samples"] == 1
    assert train_output["explanations"] == str(explanations)
    assert explanation_data["model"] == "SVR"
    assert explanation_data["diagnostics"]["shap"]["identities"][0]["player_id"]
    assert evaluation_output["metrics"] == train_output["metrics"]
    assert "prediction" in pl.read_parquet(predictions).columns


def test_train_ebm_writes_predictions_and_explanations(
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
    predictions = tmp_path / "ebm-predictions.parquet"
    explanations = tmp_path / "ebm-explanations.json"

    result = main(
        [
            "train-ebm",
            "--train",
            str(tmp_path / "train.parquet"),
            "--test",
            str(tmp_path / "test.parquet"),
            "--predictions",
            str(predictions),
            "--explanations",
            str(explanations),
            "--interactions",
            "0",
            "--max-rounds",
            "5",
            "--min-samples-leaf",
            "1",
            "--outer-bags",
            "1",
            "--validation-size",
            "0",
            "--calibration-fraction",
            "0",
            "--jobs",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    explanation_data = json.loads(explanations.read_text(encoding="utf-8"))

    assert result == 0
    assert predictions.exists()
    assert output["explanations"] == str(explanations)
    assert explanation_data["schema_version"] == 1
    assert set(explanation_data["diagnostics"]) == {
        "ale",
        "conformal_interval",
        "residual_cohorts",
        "shap",
        "temporal_permutation_importance",
    }
    assert (
        explanation_data["diagnostics"]["shap"]["identities"][0]["player_id"]
        == explanation_data["local"][0]["identity"]["player_id"]
    )
    prediction_frame = pl.read_parquet(predictions)
    assert (
        explanation_data["local"][0]["identity"]["player_id"]
        == prediction_frame["player_id"][0]
    )


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


def test_idp_build_train_and_evaluate_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_result = main(
        [
            "build-idp-dataset",
            "--output-dir",
            str(tmp_path),
            "--history-start",
            "2020",
            "--train-start",
            "2021",
            "--test-year",
            "2022",
        ],
        provider=make_idp_provider(),
    )
    build_output = json.loads(capsys.readouterr().out)
    predictions = tmp_path / "idp-predictions.parquet"

    train_result = main(
        [
            "train-svr",
            "--position",
            "idp",
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
    assert "position_group" in prediction_frame.columns


def test_injury_report_command_writes_pace_comparison_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "injuries.csv"

    result = main(
        [
            "injury-report",
            "--output",
            str(output_path),
            "--start-season",
            "2023",
            "--end-season",
            "2023",
            "--positions",
            "qb",
        ],
        provider=make_injury_report_provider(season=2023),
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["events"] == 2
    assert output["missed_games"] == 1
    assert output["played_while_reported"] == 1
    assert output_path.exists()

    frame = pl.read_csv(output_path)
    assert frame.height == 2
    assert set(frame["report_status"]) == {"Out", "Questionable"}
    # The player's return week should show a real, negative delta versus
    # their pre-injury pace (see make_injury_report_provider).
    returned = frame.filter(pl.col("played"))
    assert returned["delta_vs_pace"][0] < 0


def test_injury_report_command_defaults_to_all_positions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "injuries.csv"

    result = main(
        [
            "injury-report",
            "--output",
            str(output_path),
            "--start-season",
            "2023",
            "--end-season",
            "2023",
        ],
        provider=make_injury_report_provider(season=2023),
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    # No RB/WR/TE data exists in this provider, so only the QB's two
    # reported weeks should appear even though every position was requested.
    assert output["events"] == 2

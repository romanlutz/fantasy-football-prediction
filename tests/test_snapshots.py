import pytest
from syrupy.assertion import SnapshotAssertion

from ffpred.cli.app import main
from ffpred.features.schema import FEATURE_SCHEMA


def test_feature_schema_snapshot(snapshot: SnapshotAssertion) -> None:
    assert {column: str(dtype) for column, dtype in FEATURE_SCHEMA.items()} == snapshot


def test_cli_help_snapshot(
    snapshot: SnapshotAssertion,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == snapshot

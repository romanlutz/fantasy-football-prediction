from pathlib import Path

from ffpred.evaluation.plots import histogram


def test_histogram_writes_headless_pdf(tmp_path: Path) -> None:
    output = tmp_path / "errors.pdf"

    histogram([1, 2, 3], [1, 3, 5], output)

    assert output.read_bytes().startswith(b"%PDF")

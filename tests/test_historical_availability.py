from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ffpred.errors import DataAcquisitionError
from ffpred.study.historical_availability import (
    AVAILABILITY,
    apply_historical_availability,
    read_availability,
)
from ffpred.study.role_evidence import ENTRY


def test_review_covers_the_frozen_no_record_games() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/weekly-study/opportunity/transactions/unresolved.csv"
    )
    c = pl.col
    cohort = pl.read_csv(path).filter(
        (c("board") == "transactions")
        & ~c("opener")
        & (c("season") <= 2016)
        & (c("evidence") == "no_record")
    )
    facts = read_availability()
    cols = ["season", "week", *ENTRY, "name"]
    assert cohort.height == 111
    assert_frame_equal(cohort.select(cols).sort(ENTRY), facts.select(cols).sort(ENTRY))


def test_cutoff_exact_matchup_and_newer_activation() -> None:
    r = (
        read_availability()
        .filter(pl.col("name") == "Marcedes Lewis")
        .row(0, named=True)
    )
    stamp, team, game = r["available_at"], r["team"], r["game_id"]
    after, before = stamp + timedelta(hours=1), stamp - timedelta(hours=1)
    pool = (
        pl.DataFrame(
            [
                (game, team, before, None, None),
                (game, team, stamp, None, None),
                (game, "BUF", after, None, None),
                ("return_game", team, after, None, None),
                (game, team, after, "activated", before),
                (game, team, after, "activated", after),
                (game, team, stamp, "activated", after),
            ],
            schema=["game_id", "team", "cutoff", "kind", "news_at"],
            orient="row",
        )
        .with_columns(pl.lit(r["player_id"]).alias("player_id"))
        .with_row_index("row")
    )
    fixed = apply_historical_availability(pool)
    assert_frame_equal(fixed.sort("row"), pool[[0, 2, 3, 5]])


def test_rejects_duplicate_matchup_evidence(tmp_path: Path) -> None:
    rows = pl.read_csv(AVAILABILITY).head(1)
    path = tmp_path / "invalid.csv"
    pl.concat([rows, rows]).write_csv(path)
    with pytest.raises(DataAcquisitionError, match="availability"):
        read_availability(path)

"""Game-specific, dated eligibility facts for the 2014-16 no-record audit."""

from pathlib import Path

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.role_evidence import ENTRY

AVAILABILITY = Path(__file__).with_name("historical_availability.csv")
REASONS = {
    "injured_reserve",
    "nonfootball_reserve",
    "exempt",
    "released",
    "traded",
    "suspended",
    "practice_squad",
    "retired",
    "did_not_report",
}


def read_availability(path: Path = AVAILABILITY) -> pl.DataFrame:
    data = pl.read_csv(path)
    keys = [*ENTRY, "reason", "available_at", "source"]
    if (
        data.drop_nulls(keys).height != data.height
        or data.select(ENTRY).is_duplicated().any()
        or not set(data["reason"]) <= REASONS
        or not data["source"].str.starts_with("https://").all()
        or not data["available_at"].str.ends_with("Z").all()
    ):
        raise DataAcquisitionError("Invalid game-specific availability evidence")
    return data.with_columns(pl.col("available_at").str.to_datetime(time_zone="UTC"))


def apply_historical_availability(pool: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    known = read_availability().select(*ENTRY, "available_at")
    rows = pool.join(known, on=ENTRY, how="left", validate="m:1")
    activation = (
        c("kind").is_in(["activated", "starter"])
        & (c("news_at") > c("available_at"))
        & (c("news_at") <= c("cutoff"))
    )
    barred = (c("available_at") <= c("cutoff")) & ~activation.fill_null(False)
    return rows.filter(~barred.fill_null(False)).select(pool.columns)

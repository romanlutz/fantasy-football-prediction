"""Team defense snapshots fixed before the first game of each target week."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ffpred.errors import DataAcquisitionError
from ffpred.study.panel import TEAM_CODES

RAW = ("points_allowed", "passing_yards", "rushing_yards", "turnovers")
PAPER = tuple(f"opp_last{n}_{f}" for n in (1, 10) for f in RAW)
RATES = (
    "rush_ypc",
    "rush_epa",
    "rush_explosive",
    "pass_ypd",
    "pass_epa",
    "pass_explosive",
)
ADJUSTED = tuple(f"opp_adjusted_{rate}" for rate in RATES)
HALF_LIFE = 4
PRIOR_PLAYS = 50
OFFSEASON_RETENTION = 0.25


class Totals(NamedTuple):
    season: int
    numerator: NDArray[np.float64]
    exposure: NDArray[np.float64]


def _decayed(
    state: Totals | None, season: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if state is None:
        return np.zeros(len(RATES)), np.zeros(len(RATES))
    keep = OFFSEASON_RETENTION ** (season - state.season)
    return state.numerator * keep, state.exposure * keep


def _update(state: Totals | None, observation: Totals) -> Totals:
    total, plays = _decayed(state, observation.season)
    decay = 0.5 ** (1 / HALF_LIFE)
    return Totals(
        observation.season,
        total * decay + observation.numerator,
        plays * decay + observation.exposure,
    )


def defense_snapshots(sides: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    offense = stats.filter(c("season_type") == "REG").with_columns(
        c("team").replace(TEAM_CODES),
        (c("passing_interceptions") + c("fumbles_lost_total")).alias("turnovers"),
        (c("attempts") + c("sacks_suffered")).alias("dropbacks"),
    )
    games = sides.join(
        offense, on=["game_id", "team"], how="left", validate="1:1"
    ).rename({"offense_score": "points_allowed"})
    fields = [
        *RAW,
        "carries",
        "dropbacks",
        "rushing_epa",
        "passing_epa",
        "rushing_10",
        "passing_20",
    ]
    if games.select(fields).null_count().row(0) != (0,) * len(fields):
        raise DataAcquisitionError(
            "Incomplete team box for a completed regular-season game"
        )
    games = games.sort("season", "week", "team").with_columns(
        c(f)
        .shift()
        .rolling_mean(n, min_samples=1)
        .over("opponent")
        .alias(f"opp_last{n}_{f}")
        for n in (1, 10)
        for f in RAW
    )
    attack: dict[str, Totals] = {}
    defense: dict[str, Totals] = {}
    league_sum, league_plays = np.zeros(len(RATES)), np.zeros(len(RATES))
    snapshots = []
    for week in games.partition_by(["season", "week"], maintain_order=True):
        mean = league_sum / np.maximum(1, league_plays)
        updates = []
        for row in week.iter_rows(named=True):
            year, team, opponent = row["season"], row["team"], row["opponent"]
            observed = np.array(
                [
                    row[f]
                    for f in (
                        "rushing_yards",
                        "rushing_epa",
                        "rushing_10",
                        "passing_yards",
                        "passing_epa",
                        "passing_20",
                    )
                ]
            )
            plays = np.repeat([row["carries"], row["dropbacks"]], 3)
            a, ap = _decayed(attack.get(team), year)
            expected = (a + PRIOR_PLAYS * mean) / (ap + PRIOR_PLAYS)
            d, dp = _decayed(defense.get(opponent), year)
            snapshots.append(
                dict(zip(ADJUSTED, mean + d / (dp + PRIOR_PLAYS), strict=True))
            )
            updates.append(
                (
                    team,
                    opponent,
                    Totals(year, observed, plays),
                    Totals(year, observed - expected * plays, plays),
                )
            )
        for team, opponent, actual, residual in updates:
            attack[team] = _update(attack.get(team), actual)
            defense[opponent] = _update(defense.get(opponent), residual)
            league_sum += actual.numerator
            league_plays += actual.exposure
    return games.select("game_id", "opponent", *PAPER).hstack(pl.DataFrame(snapshots))

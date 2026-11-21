"""Pregame candidates, a position-independent scoring ledger and player histories."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffpred.acquisition.contracts import RELOCATED_TEAM_CODES
from ffpred.errors import DataAcquisitionError

POSITIONS = ("QB", "RB", "WR", "TE")
PLAYER_WEEK = ["season", "week", "player_id"]
TEAM_CODES = {
    **RELOCATED_TEAM_CODES,
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
    "LAR": "LA",
}


@dataclass(frozen=True)
class WeeklyScoring:
    """Standard non-PPR individual offense and return scoring, version 1."""

    pass_yard: float = 0.04
    scrimmage_yard: float = 0.1
    pass_td: float = 4.0
    other_td: float = 6.0
    interception: float = -2.0
    lost_fumble: float = -2.0
    conversion: float = 2.0
    reception: float = 0.0

    def points(self) -> pl.Expr:
        c = pl.col
        return (
            c("passing_yards") * self.pass_yard
            + (c("rushing_yards") + c("receiving_yards")) * self.scrimmage_yard
            + c("passing_tds") * self.pass_td
            + (
                c("rushing_tds")
                + c("receiving_tds")
                + c("special_teams_tds")
                + c("fumble_recovery_tds")
            )
            * self.other_td
            + c("passing_interceptions") * self.interception
            + c("fumbles_lost_total") * self.lost_fumble
            + (
                c("passing_2pt_conversions")
                + c("rushing_2pt_conversions")
                + c("receiving_2pt_conversions")
            )
            * self.conversion
            + c("receptions") * self.reception
        ).round(2)


SCORING = WeeklyScoring()
USAGE = ("attempts", "carries", "targets", "receptions")
HISTORY_FIELDS = ("points", *USAGE, "passing_yards", "rushing_yards", "receiving_yards")
PLAYER_FEATURES = (
    "prior_games",
    *(f"last{n}_{f}" for n in (1, 10) for f in HISTORY_FIELDS),
)


def box_scores(stats: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    numeric = (*SCORING.points().meta.root_names(), *USAGE, "fumbles_total")
    regular = stats.filter(
        (c("season_type") == "REG")
        & c("player_id").is_not_null()
        & (c("player_id") != "")
    ).select(
        *PLAYER_WEEK,
        "game_id",
        "player_display_name",
        c("position").alias("box_position"),
        c("team").replace(TEAM_CODES).alias("actual_team"),
        c(list(dict.fromkeys(numeric))).fill_null(0),
    )
    legacy = (
        c("passing_yards") / 25
        + 4 * c("passing_tds")
        - 2 * c("passing_interceptions")
        + c("rushing_yards") / 10
        + 6 * c("rushing_tds")
        - 2 * c("fumbles_total")
        + 2 * (c("passing_2pt_conversions") + c("rushing_2pt_conversions"))
    )
    if regular.select(PLAYER_WEEK).is_duplicated().any():
        raise DataAcquisitionError("Non-unique regular-season player-week in stats")
    return regular.with_columns(
        SCORING.points().alias("points"), legacy.round(2).alias("legacy_qb_rescore")
    )


def schedule_sides(schedules: pl.DataFrame) -> pl.DataFrame:
    regular = schedules.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    sides = [
        regular.select(
            "season",
            "week",
            "game_id",
            pl.col(f"{side}_team").replace(TEAM_CODES).alias("team"),
            pl.col(f"{other}_team").replace(TEAM_CODES).alias("opponent"),
            pl.col(f"{side}_score").alias("offense_score"),
        )
        for side, other in (("home", "away"), ("away", "home"))
    ]
    return pl.concat(sides).sort("season", "week", "team")


def candidates(
    sides: pl.DataFrame, rosters: pl.DataFrame, boxes: pl.DataFrame
) -> pl.DataFrame:
    """Use the most recent completed team game's roster, never a target-week row."""
    c = pl.col
    games = sides.with_columns(
        c("week").shift().over("season", "team").alias("roster_week")
    )
    roster = rosters.filter(
        (c("game_type") == "REG") & (c("gsis_id") != "") & c("gsis_id").is_not_null()
    ).select(
        "season",
        c("week").alias("roster_week"),
        c("team").replace(TEAM_CODES),
        c("gsis_id").alias("player_id"),
        c("full_name").alias("name"),
        c("position").replace({"FB": "RB", "HB": "RB"}),
        "status_description_abbr",
    )
    join_keys = ["season", "team", "roster_week"]
    if games.drop_nulls("roster_week").join(roster, on=join_keys, how="anti").height:
        raise DataAcquisitionError("Missing prior team roster for a scheduled matchup")
    # Historical `status` can say RES even in weeks a player started. Use game
    # status codes, not that field. Some 2016 team-games are only partly coded.
    player_keys = [*join_keys, "player_id"]
    coded_players = (
        roster.filter(c("status_description_abbr").is_not_null())
        .select(player_keys)
        .unique()
    )
    roster_fields = [*join_keys, "player_id", "name", "position"]
    weekly = roster.filter(c("status_description_abbr").is_in(["A01", "I01", "I02"]))
    prior_boxes = boxes.select(
        "season",
        c("week").alias("roster_week"),
        c("actual_team").alias("team"),
        "player_id",
        c("player_display_name").alias("box_name"),
        c("box_position").replace({"FB": "RB", "HB": "RB"}),
    ).join(coded_players, on=player_keys, how="anti")
    fallback = prior_boxes.join(
        roster.select(roster_fields).unique(), on=player_keys, how="left"
    ).with_columns(
        pl.coalesce("position", "box_position"), pl.coalesce("name", "box_name")
    )
    evidence = pl.concat(
        [
            source.select(*roster_fields, pl.lit(label).alias("roster_source"))
            for source, label in ((weekly, "weekly_roster"), (fallback, "prior_box"))
        ]
    )
    pool = (
        games.join(evidence, on=join_keys)
        .filter(c("position").is_in(POSITIONS))
        .unique()
        .filter(c("roster_week") == c("roster_week").max().over(PLAYER_WEEK))
    )
    if pool.select(PLAYER_WEEK).is_duplicated().any():
        raise DataAcquisitionError(
            "Conflicting candidates with equally recent roster evidence"
        )
    return pool.drop("offense_score")


def player_panel(pool: pl.DataFrame, boxes: pl.DataFrame) -> pl.DataFrame:
    """Score the player-week even when a roster or predicted matchup is stale."""
    observed = boxes.select(*PLAYER_WEEK, *HISTORY_FIELDS)
    zeros = pool.select(PLAYER_WEEK).join(observed, on=PLAYER_WEEK, how="anti")
    history = pl.concat(
        [
            observed,
            zeros.with_columns(
                pl.lit(0).cast(observed.schema[f]).alias(f) for f in HISTORY_FIELDS
            ).select(observed.columns),
        ]
    ).sort("player_id", "season", "week")
    lagged = history.with_columns(
        (pl.col("week").cum_count().over("player_id") - 1).alias("prior_games"),
        pl.col("season").shift().over("player_id").alias("history_season"),
        pl.col("week").shift().over("player_id").alias("history_week"),
        *(
            pl.col(f)
            .shift()
            .rolling_mean(n, min_samples=1)
            .over("player_id")
            .alias(f"last{n}_{f}")
            for n in (1, 10)
            for f in HISTORY_FIELDS
        ),
    )
    return (
        pool.join(
            lagged.select(
                *PLAYER_WEEK,
                "points",
                "history_season",
                "history_week",
                *PLAYER_FEATURES,
            ),
            on=PLAYER_WEEK,
            validate="1:1",
        )
        .join(
            boxes.select(
                *PLAYER_WEEK, "actual_team", pl.col("game_id").alias("actual_game_id")
            ),
            on=PLAYER_WEEK,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("actual_game_id").is_not_null().alias("has_box"))
        .sort(*PLAYER_WEEK)
    )

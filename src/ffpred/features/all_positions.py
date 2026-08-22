"""Leakage-safe preseason features for standard fantasy positions."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from ffpred.acquisition.contracts import REGULAR_SEASON
from ffpred.errors import DataAcquisitionError, SchemaValidationError
from ffpred.features.schema import TARGET_COLUMN

LOGGER = logging.getLogger(__name__)
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
PLAYER_POSITIONS = FANTASY_POSITIONS[:-1]
POSITION_DEPTH_LIMITS: Mapping[str, int] = {
    "QB": 1,
    "RB": 3,
    "WR": 3,
    "TE": 2,
    "K": 1,
}
DST_POINTS_ALLOWED_LIMITS = (0, 6, 13, 20, 27, 34)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastFrameConfig:
    """Target and outcome metadata used to assemble one forecast frame."""

    target_year: int
    as_of: date | None = None
    injuries: pl.DataFrame | None = None
    rosters_weekly: pl.DataFrame | None = None


ALL_POSITION_MODEL_FEATURE_COLUMNS = (
    "player_last_1_points",
    "player_last_5_points",
    "player_last_10_points",
    "player_previous_season_ppg",
    "player_previous_season_games",
    "player_career_ppg",
    "player_career_games",
    "team_position_ppg",
    "opponent_position_points_allowed",
    "target_week_normalized",
    "is_home",
    *(f"position_{position.lower()}" for position in FANTASY_POSITIONS),
)
ALL_POSITION_IDENTITY_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "target_season",
    "target_week",
    "target_game_id",
    "forecast_as_of",
    "history_through_season",
)
INJURY_STATUS_COLUMN = "injury_status"
INJURY_MISSED_COLUMN = "injury_missed_game"
ALL_POSITION_LINEAGE_COLUMNS = (
    "player_history_through_season",
    "opponent_history_through_season",
)
ALL_POSITION_COLUMNS = (
    *ALL_POSITION_IDENTITY_COLUMNS,
    INJURY_STATUS_COLUMN,
    INJURY_MISSED_COLUMN,
    *ALL_POSITION_LINEAGE_COLUMNS,
    *ALL_POSITION_MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
ALL_POSITION_SCHEMA = {
    "player_id": pl.String,
    "player_name": pl.String,
    "position": pl.String,
    "team": pl.String,
    "opponent": pl.String,
    "target_season": pl.Int64,
    "target_week": pl.Int64,
    "target_game_id": pl.String,
    "forecast_as_of": pl.String,
    "history_through_season": pl.Int64,
    INJURY_STATUS_COLUMN: pl.String,
    INJURY_MISSED_COLUMN: pl.Boolean,
    "player_history_through_season": pl.Int64,
    "opponent_history_through_season": pl.Int64,
    **dict.fromkeys(ALL_POSITION_MODEL_FEATURE_COLUMNS, pl.Float64),
    TARGET_COLUMN: pl.Float64,
}

_ACTUAL_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "season",
    "week",
    "game_id",
    "is_home",
    TARGET_COLUMN,
)


def _require_columns(
    frame: pl.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataAcquisitionError(f"{source} is missing columns: {sorted(missing)}")


def _normalized_team(column: str) -> pl.Expr:
    return pl.col(column).replace_strict(
        {"STL": "LA", "SD": "LAC", "OAK": "LV"},
        default=pl.col(column),
        return_dtype=pl.String,
    )


def _points_allowed_score() -> pl.Expr:
    points = pl.col("_points_allowed")
    shutout, low, moderate, average, high, very_high = DST_POINTS_ALLOWED_LIMITS
    return (
        pl.when(points == shutout)
        .then(10.0)
        .when(points <= low)
        .then(7.0)
        .when(points <= moderate)
        .then(4.0)
        .when(points <= average)
        .then(1.0)
        .when(points <= high)
        .then(0.0)
        .when(points <= very_high)
        .then(-1.0)
        .otherwise(-4.0)
    )


def _schedule_context(schedules: pl.DataFrame) -> pl.DataFrame:
    required = (
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    _require_columns(schedules, required, source="schedules")
    return schedules.select(required).with_columns(
        _normalized_team("home_team").alias("home_team"),
        _normalized_team("away_team").alias("away_team"),
    )


def _player_actuals(
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    required = (
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "opponent_team",
        "fantasy_points",
        "attempts",
        "carries",
        "targets",
        "receptions",
        "fg_att",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "pat_made",
    )
    _require_columns(player_stats, required, source="player_stats")
    schedule = _schedule_context(schedules).select(
        "game_id",
        "home_team",
        "away_team",
    )
    kicker_points = (
        (
            pl.col("fg_made_0_19").fill_null(0)
            + pl.col("fg_made_20_29").fill_null(0)
            + pl.col("fg_made_30_39").fill_null(0)
        )
        * 3.0
        + pl.col("fg_made_40_49").fill_null(0) * 4.0
        + (pl.col("fg_made_50_59").fill_null(0) + pl.col("fg_made_60_").fill_null(0))
        * 5.0
        + pl.col("pat_made").fill_null(0)
    )
    participated = (pl.col("fantasy_points").fill_null(0).abs() > 0) | (
        pl.sum_horizontal(
            pl.col("attempts").fill_null(0),
            pl.col("carries").fill_null(0),
            pl.col("targets").fill_null(0),
            pl.col("receptions").fill_null(0),
            pl.col("fg_att").fill_null(0),
        )
        > 0
    )
    return (
        player_stats.filter(
            (pl.col("season_type") == REGULAR_SEASON)
            & pl.col("position").is_in(PLAYER_POSITIONS)
            & pl.col("player_id").is_not_null()
            & participated
        )
        .join(schedule, on="game_id", how="inner")
        .with_columns(
            _normalized_team("team").alias("team"),
            _normalized_team("opponent_team").alias("opponent"),
        )
        .with_columns(
            (pl.col("team") == pl.col("home_team")).cast(pl.Float64).alias("is_home"),
            pl.when(pl.col("position") == "K")
            .then(kicker_points)
            .otherwise(pl.col("fantasy_points").fill_null(0))
            .cast(pl.Float64)
            .alias(TARGET_COLUMN),
        )
        .select(
            pl.col("player_id").cast(pl.String),
            pl.col("player_display_name").cast(pl.String).alias("player_name"),
            pl.col("position").cast(pl.String),
            pl.col("team").cast(pl.String),
            pl.col("opponent").cast(pl.String),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("game_id").cast(pl.String),
            pl.col("is_home").cast(pl.Float64),
            pl.col(TARGET_COLUMN).cast(pl.Float64),
        )
    )


def _dst_actuals(
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    required = (
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "opponent_team",
        "def_sacks",
        "def_interceptions",
        "fumble_recovery_opp",
        "def_tds",
        "special_teams_tds",
        "def_safeties",
        "def_punt_blocks",
        "def_pat_blocks",
        "def_fg_blocks",
    )
    _require_columns(team_stats, required, source="team_stats")
    schedule = _schedule_context(schedules).select(
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    frame = (
        team_stats.filter(pl.col("season_type") == REGULAR_SEASON)
        .join(schedule, on="game_id", how="inner")
        .with_columns(
            _normalized_team("team").alias("team"),
            _normalized_team("opponent_team").alias("opponent"),
        )
        .with_columns(
            pl.when(pl.col("team") == pl.col("home_team"))
            .then(pl.col("away_score"))
            .otherwise(pl.col("home_score"))
            .fill_null(0)
            .alias("_points_allowed"),
            (pl.col("team") == pl.col("home_team")).cast(pl.Float64).alias("is_home"),
        )
        .with_columns(
            (
                pl.col("def_sacks").fill_null(0)
                + pl.col("def_interceptions").fill_null(0) * 2.0
                + pl.col("fumble_recovery_opp").fill_null(0) * 2.0
                + (
                    pl.col("def_tds").fill_null(0)
                    + pl.col("special_teams_tds").fill_null(0)
                )
                * 6.0
                + pl.col("def_safeties").fill_null(0) * 2.0
                + (
                    pl.col("def_punt_blocks").fill_null(0)
                    + pl.col("def_pat_blocks").fill_null(0)
                    + pl.col("def_fg_blocks").fill_null(0)
                )
                * 2.0
                + _points_allowed_score()
            )
            .cast(pl.Float64)
            .alias(TARGET_COLUMN)
        )
    )
    return frame.select(
        (pl.lit("DST-") + pl.col("team")).alias("player_id"),
        (pl.col("team") + pl.lit(" D/ST")).alias("player_name"),
        pl.lit("DST").alias("position"),
        pl.col("team").cast(pl.String),
        pl.col("opponent").cast(pl.String),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("game_id").cast(pl.String),
        pl.col("is_home").cast(pl.Float64),
        pl.col(TARGET_COLUMN).cast(pl.Float64),
    )


def build_actual_frame(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """Create standard-scoring game results for every supported position."""
    return (
        pl.concat(
            [
                _player_actuals(player_stats, schedules),
                _dst_actuals(team_stats, schedules),
            ],
            how="vertical",
        )
        .select(_ACTUAL_COLUMNS)
        .sort("season", "week", "game_id", "position", "player_id")
    )


def build_injury_absence_frame(
    injuries: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Identify source-backed weekly absences due to injury."""
    schema = {
        "player_id": pl.String,
        "season": pl.Int64,
        "week": pl.Int64,
        INJURY_STATUS_COLUMN: pl.String,
    }
    sources: list[pl.DataFrame] = []
    if not injuries.is_empty():
        _require_columns(
            injuries,
            ("season", "game_type", "week", "gsis_id", "report_status"),
            source="injuries",
        )
        sources.append(
            injuries.filter(
                (pl.col("game_type") == REGULAR_SEASON)
                & (pl.col("report_status").str.to_lowercase() == "out")
                & pl.col("gsis_id").is_not_null()
                & (pl.col("gsis_id") != "")
            ).select(
                pl.col("gsis_id").cast(pl.String).alias("player_id"),
                pl.col("season").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.lit("Out").alias(INJURY_STATUS_COLUMN),
            )
        )
    if not rosters_weekly.is_empty():
        _require_columns(
            rosters_weekly,
            ("season", "game_type", "week", "gsis_id", "status"),
            source="rosters_weekly",
        )
        sources.append(
            rosters_weekly.filter(
                (pl.col("game_type") == REGULAR_SEASON)
                & (pl.col("status") == "RES")
                & pl.col("gsis_id").is_not_null()
                & (pl.col("gsis_id") != "")
            ).select(
                pl.col("gsis_id").cast(pl.String).alias("player_id"),
                pl.col("season").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.lit("Reserve list").alias(INJURY_STATUS_COLUMN),
            )
        )
    if not sources:
        return pl.DataFrame(schema=schema)
    return (
        pl.concat(sources, how="vertical")
        .unique()
        .group_by("player_id", "season", "week")
        .agg(
            pl.col(INJURY_STATUS_COLUMN)
            .unique()
            .sort()
            .str.join(" / ")
            .alias(INJURY_STATUS_COLUMN)
        )
        .sort("season", "week", "player_id")
    )


def _date_value(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as error:
            raise DataAcquisitionError(
                f"Expected an ISO date for {field}, received {value!r}"
            ) from error
    raise DataAcquisitionError(f"Expected a date for {field}, received {value!r}")


def _forecast_date(
    schedules: pl.DataFrame,
    target_year: int,
    requested: date | None,
) -> date:
    target = schedules.filter(
        (pl.col("season") == target_year) & (pl.col("game_type") == REGULAR_SEASON)
    )
    if target.is_empty():
        raise DataAcquisitionError(
            f"No regular-season schedule exists for {target_year}"
        )
    first_game = min(_date_value(value, "gameday") for value in target["gameday"])
    return min(requested or date.today(), first_game - timedelta(days=1))


def _historical_depth_candidates(
    depth_charts: pl.DataFrame,
    target_year: int,
) -> pl.DataFrame:
    required = (
        "season",
        "club_code",
        "week",
        "game_type",
        "depth_team",
        "formation",
        "gsis_id",
        "position",
        "full_name",
    )
    _require_columns(depth_charts, required, source="historical depth_charts")
    return (
        depth_charts.filter(
            (pl.col("season") == target_year)
            & (pl.col("week") == 1)
            & (pl.col("game_type") == REGULAR_SEASON)
            & pl.col("position").is_in(PLAYER_POSITIONS)
            & pl.col("gsis_id").is_not_null()
            & (
                ((pl.col("position") == "K") & (pl.col("formation") == "Special Teams"))
                | ((pl.col("position") != "K") & (pl.col("formation") == "Offense"))
            )
        )
        .select(
            _normalized_team("club_code").cast(pl.String).alias("team"),
            pl.col("gsis_id").cast(pl.String).alias("player_id"),
            pl.col("full_name").cast(pl.String).alias("player_name"),
            pl.col("position").cast(pl.String),
            pl.col("depth_team").cast(pl.Int64).alias("_rank"),
        )
        .drop_nulls("player_name")
        .unique(("team", "player_id", "position"))
    )


def _daily_depth_candidates(
    depth_charts: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    required = ("dt", "team", "player_name", "gsis_id", "pos_abb", "pos_rank")
    _require_columns(depth_charts, required, source="daily depth_charts")
    return (
        depth_charts.with_columns(
            pl.col("dt").cast(pl.String).str.slice(0, 10).str.to_date().alias("_date"),
            pl.when(pl.col("pos_abb") == "PK")
            .then(pl.lit("K"))
            .otherwise(pl.col("pos_abb"))
            .alias("_position"),
        )
        .filter(
            (pl.col("_date") <= as_of)
            & pl.col("_position").is_in(PLAYER_POSITIONS)
            & pl.col("gsis_id").is_not_null()
            & pl.col("player_name").is_not_null()
        )
        .with_columns(pl.col("_date").max().over("team").alias("_team_date"))
        .filter(pl.col("_date") == pl.col("_team_date"))
        .select(
            _normalized_team("team").cast(pl.String).alias("team"),
            pl.col("gsis_id").cast(pl.String).alias("player_id"),
            pl.col("player_name").cast(pl.String),
            pl.col("_position").cast(pl.String).alias("position"),
            pl.col("pos_rank").cast(pl.Int64).alias("_rank"),
        )
        .unique(("team", "player_id", "position"))
    )


def _depth_candidates(
    depth_charts: pl.DataFrame,
    *,
    target_year: int,
    as_of: date,
) -> pl.DataFrame:
    candidates = (
        _daily_depth_candidates(depth_charts, as_of)
        if "dt" in depth_charts.columns
        else _historical_depth_candidates(depth_charts, target_year)
    )
    limit = pl.col("position").replace_strict(
        POSITION_DEPTH_LIMITS,
        return_dtype=pl.Int64,
    )
    return (
        candidates.with_columns(
            pl.col("_rank").min().over("team", "position").alias("_minimum_rank")
        )
        .filter(
            pl.when(pl.col("position").is_in(("QB", "K")))
            .then(pl.col("_rank") == pl.col("_minimum_rank"))
            .otherwise(pl.col("_rank") <= limit)
        )
        .drop("_rank", "_minimum_rank")
        .sort("team", "position", "player_name")
    )


def _complete_roster(
    candidates: pl.DataFrame,
    actuals: pl.DataFrame,
    matchups: pl.DataFrame,
    *,
    target_year: int,
) -> pl.DataFrame:
    teams = matchups.select("team").unique()
    expected = teams.join(
        pl.DataFrame({"position": list(PLAYER_POSITIONS)}),
        how="cross",
    )
    available = candidates.select("team", "position").unique()
    missing = expected.join(available, on=["team", "position"], how="anti")
    if missing.is_empty():
        return candidates

    limit = pl.col("position").replace_strict(
        POSITION_DEPTH_LIMITS,
        return_dtype=pl.Int64,
    )
    fallback = (
        actuals.filter(
            (pl.col("season") == target_year - 1)
            & pl.col("position").is_in(PLAYER_POSITIONS)
        )
        .group_by("team", "position", "player_id", "player_name")
        .agg(pl.col(TARGET_COLUMN).sum().alias("_previous_points"))
        .with_columns(
            pl.col("_previous_points")
            .rank(method="ordinal", descending=True)
            .over("team", "position")
            .alias("_rank")
        )
        .filter(pl.col("_rank") <= limit)
        .join(missing, on=["team", "position"], how="inner")
        .select("team", "player_id", "player_name", "position")
    )
    completed = pl.concat([candidates, fallback], how="vertical").unique(
        ("team", "player_id", "position")
    )
    unresolved = expected.join(
        completed.select("team", "position").unique(),
        on=["team", "position"],
        how="anti",
    )
    if not unresolved.is_empty():
        gaps = ", ".join(
            f"{row['team']} {row['position']}"
            for row in unresolved.sort("team", "position").iter_rows(named=True)
        )
        raise DataAcquisitionError(
            f"No current or prior-season roster candidate exists for {gaps}"
        )
    LOGGER.warning(
        "Filled %d missing depth-chart team-position assignments for %d "
        "from prior-season production",
        missing.height,
        target_year,
    )
    return completed.sort("team", "position", "player_name")


def _position_baselines(history: pl.DataFrame, target_year: int) -> pl.DataFrame:
    recent = history.filter(pl.col("season") >= max(0, target_year - 3))
    return recent.group_by("position").agg(
        pl.col(TARGET_COLUMN).mean().alias("_baseline_ppg"),
        pl.col("season").max().cast(pl.Int64).alias("_baseline_lineage"),
    )


def _player_profiles(history: pl.DataFrame, target_year: int) -> pl.DataFrame:
    keys = ["player_id", "position"]
    ordered = history.sort("player_id", "position", "season", "week")
    career = ordered.group_by(keys).agg(
        pl.col(TARGET_COLUMN).mean().alias("player_career_ppg"),
        pl.len().cast(pl.Float64).alias("player_career_games"),
        pl.col("season").max().cast(pl.Int64).alias("_player_lineage"),
        pl.col(TARGET_COLUMN).last().alias("player_last_1_points"),
    )
    last_five = (
        ordered.group_by(keys, maintain_order=True)
        .tail(5)
        .group_by(keys)
        .agg(pl.col(TARGET_COLUMN).mean().alias("player_last_5_points"))
    )
    last_ten = (
        ordered.group_by(keys, maintain_order=True)
        .tail(10)
        .group_by(keys)
        .agg(pl.col(TARGET_COLUMN).mean().alias("player_last_10_points"))
    )
    previous = (
        ordered.filter(pl.col("season") == target_year - 1)
        .group_by(keys)
        .agg(
            pl.col(TARGET_COLUMN).mean().alias("player_previous_season_ppg"),
            pl.len().cast(pl.Float64).alias("player_previous_season_games"),
        )
    )
    return (
        career.join(last_five, on=keys)
        .join(last_ten, on=keys)
        .join(
            previous,
            on=keys,
            how="left",
        )
    )


def _context_profiles(
    history: pl.DataFrame, target_year: int
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:
    previous = history.filter(pl.col("season") == target_year - 1)
    team = previous.group_by("team", "position").agg(
        pl.col(TARGET_COLUMN).mean().alias("team_position_ppg")
    )
    opponent = previous.group_by("opponent", "position").agg(
        pl.col(TARGET_COLUMN).mean().alias("opponent_position_points_allowed")
    )
    return team, opponent


def _feature_rows(
    base: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    target_year: int,
    forecast_as_of: date,
) -> pl.DataFrame:
    if INJURY_STATUS_COLUMN not in base.columns:
        base = base.with_columns(
            pl.lit(None, dtype=pl.String).alias(INJURY_STATUS_COLUMN)
        )
    if INJURY_MISSED_COLUMN not in base.columns:
        base = base.with_columns(pl.lit(False).alias(INJURY_MISSED_COLUMN))
    history = actuals.filter(pl.col("season") < target_year)
    if history.is_empty():
        raise DataAcquisitionError(
            f"No historical fantasy results exist before {target_year}"
        )
    profiles = _player_profiles(history, target_year)
    baselines = _position_baselines(history, target_year)
    team_context, opponent_context = _context_profiles(history, target_year)
    frame = (
        base.join(profiles, on=["player_id", "position"], how="left")
        .join(baselines, on="position", how="left")
        .join(team_context, on=["team", "position"], how="left")
        .join(opponent_context, on=["opponent", "position"], how="left")
        .with_columns(
            pl.col("player_last_1_points")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("player_last_1_points"),
            pl.col("player_last_5_points")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("player_last_5_points"),
            pl.col("player_last_10_points")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("player_last_10_points"),
            pl.col("player_previous_season_ppg")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("player_previous_season_ppg"),
            pl.col("player_previous_season_games")
            .fill_null(0.0)
            .alias("player_previous_season_games"),
            pl.col("player_career_ppg")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("player_career_ppg"),
            pl.col("player_career_games").fill_null(0.0).alias("player_career_games"),
            pl.col("team_position_ppg")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("team_position_ppg"),
            pl.col("opponent_position_points_allowed")
            .fill_null(pl.col("_baseline_ppg"))
            .alias("opponent_position_points_allowed"),
            (pl.col("week") / 18.0).cast(pl.Float64).alias("target_week_normalized"),
            pl.col("is_home").cast(pl.Float64),
            pl.coalesce("_player_lineage", "_baseline_lineage")
            .cast(pl.Int64)
            .alias("player_history_through_season"),
            pl.lit(target_year - 1)
            .cast(pl.Int64)
            .alias("opponent_history_through_season"),
            *(
                (pl.col("position") == position)
                .cast(pl.Float64)
                .alias(f"position_{position.lower()}")
                for position in FANTASY_POSITIONS
            ),
            pl.lit(forecast_as_of.isoformat()).alias("forecast_as_of"),
            pl.lit(target_year - 1).cast(pl.Int64).alias("history_through_season"),
        )
        .select(
            pl.col("player_id").cast(pl.String),
            pl.col("player_name").cast(pl.String),
            pl.col("position").cast(pl.String),
            pl.col("team").cast(pl.String),
            pl.col("opponent").cast(pl.String),
            pl.col("season").cast(pl.Int64).alias("target_season"),
            pl.col("week").cast(pl.Int64).alias("target_week"),
            pl.col("game_id").cast(pl.String).alias("target_game_id"),
            "forecast_as_of",
            "history_through_season",
            pl.col(INJURY_STATUS_COLUMN).cast(pl.String),
            pl.col(INJURY_MISSED_COLUMN).cast(pl.Boolean),
            "player_history_through_season",
            "opponent_history_through_season",
            *ALL_POSITION_MODEL_FEATURE_COLUMNS,
            pl.col(TARGET_COLUMN).cast(pl.Float64),
        )
    )
    return validate_all_position_frame(frame, target_required=False)


def build_all_position_training_frame(
    actuals: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    target_years: Iterable[int],
) -> pl.DataFrame:
    """Build expanding-window preseason training rows for completed seasons."""
    schedule = _schedule_context(schedules)
    frames: list[pl.DataFrame] = []
    for target_year in sorted(set(target_years)):
        target = actuals.filter(pl.col("season") == target_year)
        if target.is_empty():
            continue
        frames.append(
            _feature_rows(
                target,
                actuals,
                target_year=target_year,
                forecast_as_of=_forecast_date(schedule, target_year, None),
            )
        )
    if not frames:
        raise DataAcquisitionError("No completed target seasons produced training rows")
    frame = pl.concat(frames, how="vertical")
    return validate_all_position_frame(frame, target_required=True)


def _matchups(schedules: pl.DataFrame, target_year: int) -> pl.DataFrame:
    schedule = _schedule_context(schedules).filter(
        (pl.col("season") == target_year) & (pl.col("game_type") == REGULAR_SEASON)
    )
    away = schedule.select(
        _normalized_team("away_team").alias("team"),
        _normalized_team("home_team").alias("opponent"),
        "season",
        "week",
        "game_id",
        pl.lit(0.0).alias("is_home"),
    )
    home = schedule.select(
        _normalized_team("home_team").alias("team"),
        _normalized_team("away_team").alias("opponent"),
        "season",
        "week",
        "game_id",
        pl.lit(1.0).alias("is_home"),
    )
    return pl.concat([away, home], how="vertical").sort("week", "game_id", "team")


def build_all_position_forecast_frame(
    actuals: pl.DataFrame,
    schedules: pl.DataFrame,
    depth_charts: pl.DataFrame,
    *,
    config: ForecastFrameConfig,
) -> pl.DataFrame:
    """Build one frozen all-position forecast frame for a target season."""
    target_year = config.target_year
    forecast_as_of = _forecast_date(schedules, target_year, config.as_of)
    candidates = _depth_candidates(
        depth_charts,
        target_year=target_year,
        as_of=forecast_as_of,
    )
    matchups = _matchups(schedules, target_year)
    candidates = _complete_roster(
        candidates,
        actuals,
        matchups,
        target_year=target_year,
    )
    dst = (
        matchups.select("team")
        .unique()
        .with_columns(
            (pl.lit("DST-") + pl.col("team")).alias("player_id"),
            (pl.col("team") + pl.lit(" D/ST")).alias("player_name"),
            pl.lit("DST").alias("position"),
        )
    )
    roster = pl.concat([candidates, dst], how="vertical")
    injury_absences = build_injury_absence_frame(
        config.injuries if config.injuries is not None else pl.DataFrame(),
        (
            config.rosters_weekly
            if config.rosters_weekly is not None
            else pl.DataFrame()
        ),
    )
    base = (
        matchups.join(roster, on="team", how="inner")
        # Historical outcomes are attached only after every model feature has
        # been derived from seasons before target_year.
        .join(
            actuals.select(
                pl.col("player_id"),
                pl.col("game_id"),
                pl.col(TARGET_COLUMN),
            ),
            on=["player_id", "game_id"],
            how="left",
        )
        .join(
            injury_absences,
            on=["player_id", "season", "week"],
            how="left",
        )
        .with_columns(
            (
                pl.col(TARGET_COLUMN).is_null()
                & pl.col(INJURY_STATUS_COLUMN).is_not_null()
            ).alias(INJURY_MISSED_COLUMN)
        )
        .select(
            "player_id",
            "player_name",
            "position",
            "team",
            "opponent",
            "season",
            "week",
            "game_id",
            "is_home",
            INJURY_STATUS_COLUMN,
            INJURY_MISSED_COLUMN,
            TARGET_COLUMN,
        )
    )
    return _feature_rows(
        base,
        actuals,
        target_year=target_year,
        forecast_as_of=forecast_as_of,
    )


def validate_all_position_frame(
    frame: pl.DataFrame,
    *,
    target_required: bool,
) -> pl.DataFrame:
    """Enforce the persisted all-position feature contract."""
    problems: list[str] = []
    if tuple(frame.columns) != ALL_POSITION_COLUMNS:
        problems.append("column order does not match ALL_POSITION_COLUMNS")
    for column, expected in ALL_POSITION_SCHEMA.items():
        if column not in frame.schema:
            continue
        if frame.schema[column] != expected:
            problems.append(
                f"{column!r} is {frame.schema[column]}, expected {expected}"
            )
        null_allowed = column == INJURY_STATUS_COLUMN or (
            column == TARGET_COLUMN and not target_required
        )
        if not null_allowed and frame[column].null_count():
            problems.append(f"{column!r} contains null values")
    unsupported = (
        set(frame["position"].unique().to_list()) - set(FANTASY_POSITIONS)
        if "position" in frame.columns
        else set()
    )
    if unsupported:
        problems.append(f"unsupported positions: {sorted(unsupported)}")
    if problems:
        raise SchemaValidationError("all_position_frame", problems)
    return frame

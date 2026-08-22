"""Point-in-time feature construction for future schedule rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from ffpred.acquisition.contracts import (
    DEPTH_CHARTS_CONTRACT,
    FORECAST_SCHEDULES_CONTRACT,
    PLAYERS_CONTRACT,
    REGULAR_SEASON,
    normalize_team_code,
)
from ffpred.acquisition.schema import validate_frame
from ffpred.domain.identifiers import PlayerId, TeamCode
from ffpred.domain.models import (
    DefenseGame,
    DefenseHistory,
    GameKey,
    QuarterbackGame,
    QuarterbackHistory,
)
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.errors import DataAcquisitionError
from ffpred.features.schema import (
    DEFENSE_STAT_FIELDS,
    FORECAST_COLUMNS,
    FORECAST_SCHEMA,
    MODEL_FEATURE_COLUMNS,
    QB_STAT_FIELDS,
    TARGET_COLUMN,
    validate_forecast_frame,
)

GameRecord = QuarterbackGame | DefenseGame


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastSources:
    """Historical and current provider data needed for a forecast."""

    quarterback_histories: Mapping[PlayerId, QuarterbackHistory]
    defense_histories: Mapping[TeamCode, DefenseHistory]
    schedules: pl.DataFrame
    depth_charts: pl.DataFrame
    players: pl.DataFrame
    actual_player_stats: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastFrameConfig:
    """Cutoff and target values for frozen feature construction."""

    history_through_season: int
    target_year: int
    as_of: date | None = None
    scoring: ScoringConfig = DEFAULT_SCORING


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


def _text(value: object, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise DataAcquisitionError(f"Expected non-empty text for {field}")


def _integer(value: object, field: str) -> int:
    if isinstance(value, int):
        return value
    raise DataAcquisitionError(f"Expected an integer for {field}")


def _latest_games(
    games: Iterable[GameRecord],
    count: int,
) -> list[GameRecord]:
    return sorted(games, key=lambda game: game.key, reverse=True)[:count]


def _stat_features(
    games: Iterable[GameRecord],
    fields: tuple[str, ...],
    *,
    prefix: str,
) -> tuple[dict[str, float], GameKey]:
    latest = _latest_games(games, 10)
    if not latest:
        raise DataAcquisitionError(f"No {prefix} history is available")
    features: dict[str, float] = {}
    for field in fields:
        features[f"{prefix}_last_1_{field}"] = float(getattr(latest[0].stats, field))
        features[f"{prefix}_last_10_{field}"] = sum(
            float(getattr(game.stats, field)) for game in latest
        ) / len(latest)
    return features, latest[0].key


def _rookie_fallback(
    histories: Mapping[PlayerId, QuarterbackHistory],
) -> tuple[dict[str, float], GameKey]:
    games = [
        game
        for history in histories.values()
        if history.profile.rookie_season is not None
        for game in history.games.values()
        if game.key.season == history.profile.rookie_season
    ]
    features, latest = _stat_features(games, QB_STAT_FIELDS, prefix="qb")
    for field in QB_STAT_FIELDS:
        average = sum(float(getattr(game.stats, field)) for game in games) / len(games)
        features[f"qb_last_1_{field}"] = average
        features[f"qb_last_10_{field}"] = average
    return features, latest


def _player_metadata(
    players: pl.DataFrame,
) -> dict[str, tuple[date | None, int | None]]:
    records: dict[str, tuple[date | None, int | None]] = {}
    for row in validate_frame(players, PLAYERS_CONTRACT).iter_rows(named=True):
        raw_id = row["gsis_id"]
        if not raw_id:
            continue
        birth_date = (
            None
            if row["birth_date"] is None
            else _date_value(row["birth_date"], "birth_date")
        )
        rookie_season = (
            None if row["rookie_season"] is None else int(row["rookie_season"])
        )
        records[str(raw_id)] = (birth_date, rookie_season)
    return records


def _schedule_rows(
    schedules: pl.DataFrame,
    target_year: int,
) -> list[dict[str, object]]:
    return list(
        validate_frame(schedules, FORECAST_SCHEDULES_CONTRACT)
        .filter(
            (pl.col("season") == target_year) & (pl.col("game_type") == REGULAR_SEASON)
        )
        .sort("week", "game_id")
        .iter_rows(named=True)
    )


def _forecast_as_of(
    schedule_rows: list[dict[str, object]],
    requested: date | None,
) -> date:
    first_game = min(_date_value(row["gameday"], "gameday") for row in schedule_rows)
    latest_preseason = first_game - timedelta(days=1)
    return min(requested or date.today(), latest_preseason)


def _starters(
    depth_charts: pl.DataFrame,
    *,
    as_of: date,
) -> dict[str, tuple[str, str]]:
    rows = validate_frame(depth_charts, DEPTH_CHARTS_CONTRACT)
    dated = rows.with_columns(
        pl.col("dt").cast(pl.String).str.slice(0, 10).str.to_date().alias("_date")
    ).filter(
        (pl.col("_date") <= as_of)
        & (pl.col("pos_abb") == "QB")
        & (pl.col("pos_rank") == 1)
    )
    if dated.is_empty():
        raise DataAcquisitionError(
            f"No QB1 depth-chart snapshot is available on or before {as_of}"
        )
    latest = dated["_date"].max()
    assert isinstance(latest, date)
    current = dated.filter(pl.col("_date") == latest)
    starters = {
        normalize_team_code(_text(row["team"], "team")): (
            _text(row["gsis_id"], "gsis_id"),
            _text(row["player_name"], "player_name"),
        )
        for row in current.iter_rows(named=True)
    }
    return starters


def _actual_points(
    player_stats: pl.DataFrame | None,
    scoring: ScoringConfig,
) -> dict[tuple[str, str], float]:
    if player_stats is None:
        return {}
    required = {
        "player_id",
        "game_id",
        "position",
        "season_type",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_2pt_conversions",
        "rushing_yards",
        "rushing_tds",
        "rushing_2pt_conversions",
        "fumbles_total",
    }
    missing = required - set(player_stats.columns)
    if missing:
        raise DataAcquisitionError(
            f"Actual player stats are missing columns: {sorted(missing)}"
        )
    frame = player_stats.filter(
        (pl.col("position") == "QB")
        & (pl.col("season_type") == REGULAR_SEASON)
        & pl.col("player_id").is_not_null()
    ).with_columns(
        (
            pl.col("passing_yards").fill_null(0) / scoring.passing_yards_per_point
            + pl.col("passing_tds").fill_null(0) * scoring.passing_touchdown
            + pl.col("passing_interceptions").fill_null(0) * scoring.interception
            + pl.col("rushing_yards").fill_null(0) / scoring.rushing_yards_per_point
            + pl.col("rushing_tds").fill_null(0) * scoring.rushing_touchdown
            + pl.col("fumbles_total").fill_null(0) * scoring.fumble
            + (
                pl.col("passing_2pt_conversions").fill_null(0)
                + pl.col("rushing_2pt_conversions").fill_null(0)
            )
            * scoring.two_point_conversion
        ).alias(TARGET_COLUMN)
    )
    return {
        (str(row["player_id"]), str(row["game_id"])): float(row[TARGET_COLUMN])
        for row in frame.select("player_id", "game_id", TARGET_COLUMN).iter_rows(
            named=True
        )
    }


def build_forecast_frame(
    sources: ForecastSources,
    config: ForecastFrameConfig,
) -> pl.DataFrame:
    """Build frozen matchup features using no stats after the history cutoff."""
    if config.target_year <= config.history_through_season:
        raise ValueError("target_year must be after history_through_season")
    schedule_rows = _schedule_rows(sources.schedules, config.target_year)
    if not schedule_rows:
        raise DataAcquisitionError(
            f"No regular-season schedule exists for {config.target_year}"
        )
    forecast_date = _forecast_as_of(schedule_rows, config.as_of)
    starters = _starters(sources.depth_charts, as_of=forecast_date)
    metadata = _player_metadata(sources.players)
    rookie_features, rookie_lineage = _rookie_fallback(sources.quarterback_histories)
    actuals = _actual_points(sources.actual_player_stats, config.scoring)

    rows: list[dict[str, object]] = []
    for game in schedule_rows:
        game_date = _date_value(game["gameday"], "gameday")
        home_team = normalize_team_code(_text(game["home_team"], "home_team"))
        away_team = normalize_team_code(_text(game["away_team"], "away_team"))
        for team, opponent in ((away_team, home_team), (home_team, away_team)):
            if team not in starters:
                raise DataAcquisitionError(f"No QB1 assignment is available for {team}")
            player_id, player_name = starters[team]
            history = sources.quarterback_histories.get(PlayerId(player_id))
            if history and history.games:
                qb_features, qb_lineage = _stat_features(
                    history.games.values(),
                    QB_STAT_FIELDS,
                    prefix="qb",
                )
                birth_date = history.profile.birth_date
                rookie_season = history.profile.rookie_season
            else:
                qb_features = rookie_features
                qb_lineage = rookie_lineage
                birth_date, rookie_season = metadata.get(player_id, (None, None))
            if birth_date is None:
                raise DataAcquisitionError(
                    f"No birth date is available for QB1 {player_name} ({player_id})"
                )
            defense = sources.defense_histories.get(TeamCode(opponent))
            if defense is None:
                raise DataAcquisitionError(
                    f"No defense history is available for {opponent}"
                )
            defense_features, defense_lineage = _stat_features(
                defense.games.values(),
                DEFENSE_STAT_FIELDS,
                prefix="defense",
            )
            game_id = _text(game["game_id"], "game_id")
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": "QB",
                    "team": team,
                    "opponent": opponent,
                    "target_season": config.target_year,
                    "target_week": _integer(game["week"], "week"),
                    "target_game_id": game_id,
                    "forecast_as_of": forecast_date.isoformat(),
                    "history_through_season": config.history_through_season,
                    "qb_history_through_season": int(qb_lineage.season),
                    "qb_history_through_week": int(qb_lineage.week),
                    "defense_history_through_season": int(defense_lineage.season),
                    "defense_history_through_week": int(defense_lineage.week),
                    "age": (game_date - birth_date).days / 365.2425,
                    "years_pro": float(
                        max(
                            0,
                            config.target_year - (rookie_season or config.target_year),
                        )
                    ),
                    **qb_features,
                    **defense_features,
                    TARGET_COLUMN: actuals.get((player_id, game_id)),
                }
            )
    frame = pl.DataFrame(rows, schema=FORECAST_SCHEMA).select(FORECAST_COLUMNS)
    return validate_forecast_frame(
        frame.with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
    )

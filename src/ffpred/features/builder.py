"""Vectorized, leakage-safe feature table construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict

import polars as pl

from ffpred.domain.identifiers import PlayerId, TeamCode
from ffpred.domain.models import (
    DefenseHistory,
    GameKey,
    QuarterbackHistory,
)
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.features.schema import (
    DEFENSE_STAT_FIELDS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    MODEL_FEATURE_COLUMNS,
    QB_STAT_FIELDS,
    TARGET_COLUMN,
    validate_feature_frame,
)


def _quarterback_frame(
    histories: Mapping[PlayerId, QuarterbackHistory],
) -> pl.DataFrame:
    rows = [
        {
            "player_id": history.profile.player_id,
            "player_name": history.profile.name,
            "birth_date": history.profile.birth_date,
            "rookie_season": history.profile.rookie_season,
            "target_season": game.key.season,
            "target_week": game.key.week,
            "target_game_id": game.context.game_id,
            "game_date": game.context.game_date,
            "opponent": game.context.opponent,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "player_id", "target_season", "target_week"
    )


def _defense_frame(histories: Mapping[TeamCode, DefenseHistory]) -> pl.DataFrame:
    rows = [
        {
            "defense": history.team,
            "target_season": game.key.season,
            "target_week": game.key.week,
            **asdict(game.stats),
        }
        for history in histories.values()
        for game in history.games.values()
    ]
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "defense", "target_season", "target_week"
    )


def _rolling_expressions(
    fields: Iterable[str],
    *,
    prefix: str,
    group: str,
) -> list[pl.Expr]:
    expressions: list[pl.Expr] = [
        pl.col("target_season")
        .shift(1)
        .over(group)
        .alias(f"{prefix}_history_through_season"),
        pl.col("target_week")
        .shift(1)
        .over(group)
        .alias(f"{prefix}_history_through_week"),
    ]
    for field in fields:
        expressions.extend(
            (
                pl.col(field).shift(1).over(group).alias(f"{prefix}_last_1_{field}"),
                pl.col(field)
                .rolling_mean(window_size=10, min_samples=1)
                .shift(1)
                .over(group)
                .alias(f"{prefix}_last_10_{field}"),
            )
        )
    return expressions


def _rookie_average_rows(
    histories: Mapping[PlayerId, QuarterbackHistory],
    cutoffs: Iterable[GameKey],
) -> list[dict[str, float | int]]:
    rookie_games = [
        game
        for history in histories.values()
        if history.profile.rookie_season is not None
        for game in history.games.values()
        if game.key.season == history.profile.rookie_season
    ]
    rows: list[dict[str, float | int]] = []
    for cutoff in sorted(set(cutoffs)):
        prior = [game.stats for game in rookie_games if game.key < cutoff]
        if not prior:
            continue
        row: dict[str, float | int] = {
            "target_season": cutoff.season,
            "target_week": cutoff.week,
        }
        for field in QB_STAT_FIELDS:
            row[f"rookie_{field}"] = sum(
                getattr(stats, field) for stats in prior
            ) / len(prior)
        rows.append(row)
    return rows


def _score_expression(config: ScoringConfig) -> pl.Expr:
    return (
        pl.col("passing_yards") / config.passing_yards_per_point
        + pl.col("passing_touchdowns") * config.passing_touchdown
        + pl.col("passing_interceptions") * config.interception
        + pl.col("rushing_yards") / config.rushing_yards_per_point
        + pl.col("rushing_touchdowns") * config.rushing_touchdown
        + pl.col("fumbles") * config.fumble
        + (pl.col("passing_two_point_made") + pl.col("rushing_two_point_made"))
        * config.two_point_conversion
    ).alias(TARGET_COLUMN)


def build_feature_frame(
    quarterback_histories: Mapping[PlayerId, QuarterbackHistory],
    defense_histories: Mapping[TeamCode, DefenseHistory],
    *,
    scoring: ScoringConfig = DEFAULT_SCORING,
) -> pl.DataFrame:
    """Build a named feature table using only games before each target."""
    if not quarterback_histories:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    quarterbacks = _quarterback_frame(quarterback_histories)
    defenses = _defense_frame(defense_histories)

    quarterbacks = quarterbacks.with_columns(
        _rolling_expressions(QB_STAT_FIELDS, prefix="qb", group="player_id")
    )
    defenses = defenses.with_columns(
        _rolling_expressions(DEFENSE_STAT_FIELDS, prefix="defense", group="defense")
    ).select(
        "defense",
        "target_season",
        "target_week",
        "defense_history_through_season",
        "defense_history_through_week",
        *(f"defense_last_1_{field}" for field in DEFENSE_STAT_FIELDS),
        *(f"defense_last_10_{field}" for field in DEFENSE_STAT_FIELDS),
    )

    cutoffs = (
        GameKey(row["target_season"], row["target_week"])
        for row in quarterbacks.select("target_season", "target_week").iter_rows(
            named=True
        )
    )
    rookie_rows = _rookie_average_rows(quarterback_histories, cutoffs)
    if rookie_rows:
        quarterbacks = quarterbacks.join(
            pl.DataFrame(rookie_rows),
            on=["target_season", "target_week"],
            how="left",
        )
        quarterbacks = quarterbacks.with_columns(
            *(
                pl.coalesce(
                    pl.col(f"qb_last_1_{field}"),
                    pl.col(f"rookie_{field}"),
                ).alias(f"qb_last_1_{field}")
                for field in QB_STAT_FIELDS
            ),
            *(
                pl.coalesce(
                    pl.col(f"qb_last_10_{field}"),
                    pl.col(f"rookie_{field}"),
                ).alias(f"qb_last_10_{field}")
                for field in QB_STAT_FIELDS
            ),
        )

    frame = (
        quarterbacks.join(
            defenses,
            left_on=["opponent", "target_season", "target_week"],
            right_on=["defense", "target_season", "target_week"],
            how="left",
        )
        .filter(
            pl.col("birth_date").is_not_null()
            & pl.col("defense_history_through_season").is_not_null()
            & pl.col("qb_last_1_passing_attempts").is_not_null()
        )
        .with_columns(
            ((pl.col("game_date") - pl.col("birth_date")).dt.total_days() / 365.2425)
            .cast(pl.Float64)
            .alias("age"),
            (
                pl.col("target_season")
                - pl.col("rookie_season").fill_null(pl.col("target_season"))
            )
            .clip(lower_bound=0)
            .cast(pl.Float64)
            .alias("years_pro"),
            _score_expression(scoring),
        )
        .select(FEATURE_COLUMNS)
        .with_columns(pl.col(MODEL_FEATURE_COLUMNS).cast(pl.Float64))
        .with_columns(pl.col(TARGET_COLUMN).cast(pl.Float64))
    )
    return validate_feature_frame(frame)

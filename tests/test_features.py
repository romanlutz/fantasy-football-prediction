from datetime import date

from ffpred.domain.identifiers import GameId, PlayerId, Season, TeamCode, Week
from ffpred.domain.models import (
    DefenseGame,
    DefenseGameStats,
    DefenseHistory,
    GameContext,
    GameKey,
    PlayerProfile,
    QuarterbackGame,
    QuarterbackGameStats,
    QuarterbackHistory,
)
from ffpred.features.builder import build_feature_frame
from ffpred.features.schema import FEATURE_COLUMNS


def _context(game_id: str, team: str, opponent: str, day: int) -> GameContext:
    return GameContext(
        game_id=GameId(game_id),
        game_date=date(2014, 9, day),
        home_team=TeamCode(team),
        away_team=TeamCode(opponent),
        team=TeamCode(team),
        opponent=TeamCode(opponent),
    )


def _qb_stats(yards: float) -> QuarterbackGameStats:
    return QuarterbackGameStats(
        passing_attempts=20,
        passing_yards=yards,
        passing_touchdowns=2,
        passing_interceptions=1,
        passing_two_point_attempts=0,
        passing_two_point_made=0,
        rushing_attempts=2,
        rushing_yards=10,
        rushing_touchdowns=0,
        rushing_two_point_attempts=0,
        rushing_two_point_made=0,
        fumbles=0,
    )


def _defense_stats(yards: float) -> DefenseGameStats:
    return DefenseGameStats(
        points_allowed=20,
        passing_yards_allowed=yards,
        rushing_yards_allowed=100,
        turnovers=1,
    )


def test_feature_frame_is_named_typed_and_leakage_safe() -> None:
    week_one = GameKey(Season(2014), Week(1))
    week_two = GameKey(Season(2014), Week(2))
    quarterbacks = {
        PlayerId("qb"): QuarterbackHistory(
            profile=PlayerProfile(
                player_id=PlayerId("qb"),
                name="Quarterback",
                birth_date=date(1990, 1, 1),
                rookie_season=Season(2013),
            ),
            games={
                week_one: QuarterbackGame(
                    key=week_one,
                    context=_context("g1", "GB", "SEA", 1),
                    stats=_qb_stats(100),
                ),
                week_two: QuarterbackGame(
                    key=week_two,
                    context=_context("g2", "GB", "SEA", 8),
                    stats=_qb_stats(300),
                ),
            },
        )
    }
    defenses = {
        TeamCode("SEA"): DefenseHistory(
            team=TeamCode("SEA"),
            games={
                week_one: DefenseGame(
                    key=week_one,
                    context=_context("d1", "SEA", "GB", 1),
                    stats=_defense_stats(150),
                ),
                week_two: DefenseGame(
                    key=week_two,
                    context=_context("d2", "SEA", "GB", 8),
                    stats=_defense_stats(999),
                ),
            },
        )
    }

    frame = build_feature_frame(quarterbacks, defenses)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["target_week"] == 2
    assert row["qb_last_1_passing_yards"] == 100
    assert row["defense_last_1_passing_yards_allowed"] == 150
    assert (
        row["qb_history_through_season"],
        row["qb_history_through_week"],
    ) < (row["target_season"], row["target_week"])
    assert (
        row["defense_history_through_season"],
        row["defense_history_through_week"],
    ) < (row["target_season"], row["target_week"])


def test_empty_histories_produce_empty_schema() -> None:
    frame = build_feature_frame({}, {})

    assert frame.is_empty()
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_rookie_fallback_records_the_cohort_history_period_used() -> None:
    """A player's own debut game has no prior game of their own, so
    qb_last_1/10 fall back to the rookie cohort's average. The lineage
    columns must then record that cohort history's most recent period,
    never a null the leakage-safety invariant cannot check.
    """
    week_one = GameKey(Season(2014), Week(1))
    week_two = GameKey(Season(2014), Week(2))
    quarterbacks = {
        PlayerId("earlier-rookie"): QuarterbackHistory(
            profile=PlayerProfile(
                player_id=PlayerId("earlier-rookie"),
                name="Earlier Rookie",
                birth_date=date(1990, 1, 1),
                rookie_season=Season(2014),
            ),
            games={
                week_one: QuarterbackGame(
                    key=week_one,
                    context=_context("g1", "GB", "SEA", 1),
                    stats=_qb_stats(100),
                ),
            },
        ),
        PlayerId("week-two-debut"): QuarterbackHistory(
            profile=PlayerProfile(
                player_id=PlayerId("week-two-debut"),
                name="Week Two Debut",
                birth_date=date(1991, 1, 1),
                rookie_season=Season(2014),
            ),
            games={
                # This player's only recorded game is their own debut: there
                # is no earlier game of their own for qb_last_1/10 to shift
                # from, so the row must rely entirely on the rookie fallback.
                week_two: QuarterbackGame(
                    key=week_two,
                    context=_context("g3", "NYG", "SEA", 8),
                    stats=_qb_stats(150),
                ),
            },
        ),
    }
    defenses = {
        TeamCode("SEA"): DefenseHistory(
            team=TeamCode("SEA"),
            games={
                week_one: DefenseGame(
                    key=week_one,
                    context=_context("d1", "SEA", "GB", 1),
                    stats=_defense_stats(150),
                ),
                week_two: DefenseGame(
                    key=week_two,
                    context=_context("d2", "SEA", "NYG", 8),
                    stats=_defense_stats(999),
                ),
            },
        )
    }

    frame = build_feature_frame(quarterbacks, defenses)

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["player_id"] == "week-two-debut"
    assert row["qb_last_1_passing_yards"] == 100
    assert row["qb_history_through_season"] == 2014
    assert row["qb_history_through_week"] == 1
    assert (
        row["qb_history_through_season"],
        row["qb_history_through_week"],
    ) < (row["target_season"], row["target_week"])

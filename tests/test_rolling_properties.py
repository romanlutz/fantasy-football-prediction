import pytest
from hypothesis import given
from hypothesis import strategies as st

from ffpred.domain.identifiers import Season, Week
from ffpred.domain.models import GameKey
from ffpred.features.rolling import previous_games


@given(
    weeks=st.sets(st.integers(min_value=1, max_value=18), max_size=18),
    cutoff_week=st.integers(min_value=1, max_value=19),
    count=st.integers(min_value=1, max_value=20),
)
def test_previous_games_are_bounded_ordered_and_before_cutoff(
    weeks: set[int],
    cutoff_week: int,
    count: int,
) -> None:
    games = {GameKey(Season(2025), Week(week)): week for week in weeks}
    cutoff = GameKey(Season(2025), Week(cutoff_week))

    result = previous_games(games, cutoff, count)

    assert len(result) <= count
    assert result == sorted(result, reverse=True)
    assert all(week < cutoff_week for week in result)


def test_previous_games_requires_positive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        previous_games({}, GameKey(Season(2025), Week(1)), 0)

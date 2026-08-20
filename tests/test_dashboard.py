from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from ffpred.dashboard.data import (
    CONSENSUS_MODEL,
    DashboardDataError,
    draft_board,
    model_choices,
    model_name_from_path,
    model_scorecard,
    prepare_predictions,
    select_model,
    weekly_board,
)


def _predictions(offset: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["a", "a", "b", "b"],
            "player_name": ["Alpha QB", "Alpha QB", "Bravo QB", "Bravo QB"],
            "target_season": [2025, 2025, 2025, 2025],
            "target_week": [1, 2, 1, 2],
            "target_game_id": ["g1", "g2", "g3", "g4"],
            "fantasy_points": [20.0, 24.0, 18.0, 16.0],
            "prediction": [
                19.0 + offset,
                21.0 + offset,
                17.0 + offset,
                15.0 + offset,
            ],
        }
    )


def test_prepare_predictions_adds_dashboard_defaults() -> None:
    prepared = prepare_predictions(_predictions(), model_name="SVR")

    assert prepared["model"].unique().to_list() == ["SVR"]
    assert prepared["position"].unique().to_list() == ["QB"]
    assert prepared["team"].unique().to_list() == ["N/A"]
    assert prepared["absolute_error"].to_list() == [1.0, 3.0, 1.0, 1.0]


def test_prepare_predictions_rejects_incomplete_artifact() -> None:
    with pytest.raises(DashboardDataError, match="player_name"):
        prepare_predictions(
            _predictions().drop("player_name"),
            model_name="SVR",
        )


def test_consensus_averages_models_and_preserves_actuals() -> None:
    frame = pl.concat(
        [
            prepare_predictions(_predictions(), model_name="SVR"),
            prepare_predictions(_predictions(2.0), model_name="MLP"),
        ]
    )

    consensus = select_model(frame, CONSENSUS_MODEL)

    assert model_choices(frame) == ("Consensus", "MLP", "SVR")
    assert consensus["prediction"].to_list() == [20.0, 22.0, 18.0, 16.0]
    assert consensus["model_count"].unique().to_list() == [2]
    assert consensus["model_spread"].min() == pytest.approx(2**0.5)


def test_consensus_preserves_forecast_provenance() -> None:
    frame = pl.concat(
        [
            prepare_predictions(_predictions(), model_name="SVR"),
            prepare_predictions(_predictions(2.0), model_name="MLP"),
        ]
    ).with_columns(
        pl.lit("2025-09-03").alias("forecast_as_of"),
        pl.lit(2024).alias("history_through_season"),
    )

    consensus = select_model(frame, CONSENSUS_MODEL)

    assert consensus["forecast_as_of"].unique().to_list() == ["2025-09-03"]
    assert consensus["history_through_season"].unique().to_list() == [2024]


def test_draft_and_weekly_boards_rank_the_selected_horizon() -> None:
    prepared = select_model(
        prepare_predictions(_predictions(), model_name="SVR"),
        "SVR",
    )

    draft = draft_board(
        prepared,
        season=2025,
        positions=["QB"],
        minimum_games=2,
    )
    weekly = weekly_board(
        prepared,
        season=2025,
        week=1,
        positions=["QB"],
    )

    assert draft["player_name"].to_list() == ["Alpha QB", "Bravo QB"]
    assert draft["projected_points"].to_list() == [40.0, 32.0]
    assert draft["position_rank"].to_list() == [1, 2]
    assert weekly["player_name"].to_list() == ["Alpha QB", "Bravo QB"]
    assert weekly["model_agreement"].unique().to_list() == ["Single model"]


def test_draft_board_adds_injury_adjusted_actual() -> None:
    prepared = prepare_predictions(
        _predictions().with_columns(
            pl.Series("fantasy_points", [20.0, None, 18.0, 16.0]),
            pl.Series("injury_missed_game", [False, True, False, False]),
            pl.Series("injury_status", [None, "Out", None, None]),
        ),
        model_name="SVR",
    ).with_columns(
        pl.lit(0.0).alias("model_spread"),
        pl.lit(1).alias("model_count"),
    )

    player = draft_board(
        prepared,
        season=2025,
        positions=["QB"],
        minimum_games=2,
    ).filter(pl.col("player_id") == "a")

    assert player["projected_points"][0] == 40.0
    assert player["actual_points"][0] == 20.0
    assert player["injury_games"][0] == 1
    assert player["injury_estimated_points"][0] == 20.0
    assert player["availability_adjusted_actual"][0] == 40.0
    assert player["adjusted_delta_percent"][0] == 0.0


def test_model_scorecard_includes_consensus() -> None:
    frame = pl.concat(
        [
            prepare_predictions(_predictions(), model_name="SVR"),
            prepare_predictions(_predictions(2.0), model_name="MLP"),
        ]
    )

    scorecard = model_scorecard(frame)

    assert set(scorecard["model"]) == {"Consensus", "MLP", "SVR"}
    consensus = scorecard.filter(pl.col("model") == CONSENSUS_MODEL).row(
        0,
        named=True,
    )
    assert consensus["mae"] == pytest.approx(0.5)
    assert consensus["bias"] == pytest.approx(-0.5)


def test_model_name_from_path_is_concise() -> None:
    assert model_name_from_path(Path("svr-predictions.parquet")) == "SVR"
    assert model_name_from_path(Path("experimental-model.parquet")) == (
        "Experimental Model"
    )


def test_dashboard_app_renders_all_workspaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _predictions().write_parquet(tmp_path / "svr-predictions.parquet")
    app_path = Path(__file__).parents[1] / "src" / "ffpred" / "dashboard" / "app.py"
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.radio[0].options == [
        "Draft Board",
        "Weekly Decisions",
        "Model Room",
    ]
    headings = {
        "Draft Board": "Draft board",
        "Weekly Decisions": "Weekly decisions",
        "Model Room": "Model room",
    }
    for workspace in app.radio[0].options:
        app.radio[0].set_value(workspace).run()
        assert not app.exception
        assert headings[workspace] in [header.value for header in app.header]


def test_dashboard_exposes_all_artifact_seasons_and_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    pl.DataFrame(
        {
            "player_id": [f"player-{index}" for index in range(len(positions))] * 2,
            "player_name": [f"Player {position}" for position in positions] * 2,
            "position": positions * 2,
            "target_season": [2025] * len(positions) + [2026] * len(positions),
            "target_week": [1] * (len(positions) * 2),
            "target_game_id": [
                f"{season}-{position}"
                for season in (2025, 2026)
                for position in positions
            ],
            "fantasy_points": [10.0] * len(positions) + [None] * len(positions),
            "prediction": [9.0] * (len(positions) * 2),
        }
    ).write_parquet(tmp_path / "svr-predictions.parquet")
    app_path = Path(__file__).parents[1] / "src" / "ffpred" / "dashboard" / "app.py"
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(app_path, default_timeout=30).run()
    season = next(widget for widget in app.selectbox if widget.label == "Season")
    position = next(widget for widget in app.multiselect if widget.label == "Position")

    assert not app.exception
    assert season.options == ["2026", "2025"]
    assert position.options == ["DST", "K", "QB", "RB", "TE", "WR"]

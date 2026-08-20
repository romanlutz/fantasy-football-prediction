from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ffpred.cli.app import main
from ffpred.datasets.archive import ForecastArchiveConfig, build_forecast_archive
from ffpred.errors import ConfigurationError
from ffpred.features.all_positions import (
    ALL_POSITION_MODEL_FEATURE_COLUMNS,
    FANTASY_POSITIONS,
    build_actual_frame,
)
from ffpred.providers.fakes import FakeProvider
from ffpred.training.data import load_training_data
from ffpred.training.projection import load_projection_data


def _schedule(season: int) -> dict[str, object]:
    return {
        "game_id": f"{season}_01_GB_SEA",
        "season": season,
        "game_type": "REG",
        "week": 1,
        "gameday": f"{season}-09-10",
        "home_team": "SEA",
        "away_team": "GB",
        "home_score": 20,
        "away_score": 24,
    }


def _player_row(
    season: int,
    *,
    position: str,
    team: str,
    opponent: str,
) -> dict[str, object]:
    player_id = f"{team}-{position}"
    return {
        "player_id": player_id,
        "player_display_name": f"{team} {position}",
        "position": position,
        "season": season,
        "week": 1,
        "season_type": "REG",
        "game_id": f"{season}_01_GB_SEA",
        "team": team,
        "opponent_team": opponent,
        "fantasy_points": 12.0,
        "attempts": 25 if position == "QB" else 0,
        "carries": 12 if position == "RB" else 0,
        "targets": 8 if position in {"WR", "TE"} else 0,
        "receptions": 5 if position in {"WR", "TE"} else 0,
        "fg_att": 3 if position == "K" else 0,
        "fg_made_0_19": 1 if position == "K" else 0,
        "fg_made_20_29": 0,
        "fg_made_30_39": 0,
        "fg_made_40_49": 1 if position == "K" else 0,
        "fg_made_50_59": 1 if position == "K" else 0,
        "fg_made_60_": 0,
        "pat_made": 2 if position == "K" else 0,
    }


def _team_row(
    season: int,
    *,
    team: str,
    opponent: str,
) -> dict[str, object]:
    return {
        "season": season,
        "week": 1,
        "season_type": "REG",
        "game_id": f"{season}_01_GB_SEA",
        "team": team,
        "opponent_team": opponent,
        "def_sacks": 2,
        "def_interceptions": 1,
        "fumble_recovery_opp": 1,
        "def_tds": 1,
        "special_teams_tds": 0,
        "def_safeties": 1,
        "def_punt_blocks": 1,
        "def_pat_blocks": 0,
        "def_fg_blocks": 0,
    }


def _depth_charts(target_year: int) -> pl.DataFrame:
    rows = [
        {
            "season": target_year,
            "club_code": team,
            "week": 1,
            "game_type": "REG",
            "depth_team": "1",
            "formation": "Special Teams" if position == "K" else "Offense",
            "gsis_id": f"{team}-{position}",
            "position": position,
            "full_name": f"{team} {position}",
        }
        for team in ("GB", "SEA")
        for position in ("QB", "RB", "WR", "TE", "K")
    ]
    return pl.DataFrame(rows)


def _provider() -> FakeProvider:
    seasons = (2008, 2009, 2010)
    schedules = pl.DataFrame([_schedule(season) for season in seasons])
    player_stats = pl.DataFrame(
        [
            _player_row(
                season,
                position=position,
                team=team,
                opponent=("SEA" if team == "GB" else "GB"),
            )
            for season in seasons
            for team in ("GB", "SEA")
            for position in ("QB", "RB", "WR", "TE", "K")
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_row(
                season,
                team=team,
                opponent=("SEA" if team == "GB" else "GB"),
            )
            for season in seasons
            for team in ("GB", "SEA")
        ]
    )
    return FakeProvider(
        player_stats=player_stats,
        team_stats=team_stats,
        schedules=schedules,
        depth_charts=_depth_charts(2010),
    )


@pytest.mark.parametrize(
    ("history_start", "first_target", "last_target"),
    [
        (2010, 2010, 2011),
        (2009, 2010, 2011),
        (2009, 2011, 2010),
        (2009, 2010, date.today().year + 1),
    ],
)
def test_archive_config_rejects_invalid_year_ranges(
    tmp_path: Path,
    history_start: int,
    first_target: int,
    last_target: int,
) -> None:
    with pytest.raises(ConfigurationError):
        ForecastArchiveConfig(
            output_dir=tmp_path,
            history_start=history_start,
            first_target_year=first_target,
            last_target_year=last_target,
        )


def test_archive_default_target_year_tracks_current_year(tmp_path: Path) -> None:
    config = ForecastArchiveConfig(output_dir=tmp_path)

    assert config.last_target_year == date.today().year


def test_standard_scoring_includes_kickers_and_defenses() -> None:
    provider = _provider()
    actuals = build_actual_frame(
        provider.player_stats,
        provider.team_stats,
        provider.schedules,
    )

    kicker = actuals.filter(
        (pl.col("season") == 2010) & (pl.col("player_id") == "GB-K")
    )[0]
    defense = actuals.filter(
        (pl.col("season") == 2010) & (pl.col("player_id") == "DST-GB")
    )[0]

    assert kicker["fantasy_points"][0] == 14.0
    assert defense["fantasy_points"][0] == 17.0


def test_archive_fills_missing_depth_position_from_prior_season(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider()
    provider.depth_charts = provider.depth_charts.filter(
        ~((pl.col("club_code") == "GB") & (pl.col("position") == "K"))
    )

    result = build_forecast_archive(
        ForecastArchiveConfig(
            output_dir=tmp_path,
            history_start=2008,
            first_target_year=2010,
            last_target_year=2010,
        ),
        provider=provider,
    )
    forecast = pl.read_parquet(result.seasons[0].forecast.path)
    manifest = json.loads(result.seasons[0].manifest_path.read_text())

    assert "GB-K" in forecast["player_id"]
    assert manifest["roster_coverage"]["team_position_pairs"] == 12
    assert manifest["roster_coverage"]["expected_team_position_pairs"] == 12
    assert "prior-season production" in caplog.text


def test_archive_builds_every_position_with_frozen_lineage(
    tmp_path: Path,
) -> None:
    result = build_forecast_archive(
        ForecastArchiveConfig(
            output_dir=tmp_path,
            history_start=2008,
            first_target_year=2010,
            last_target_year=2010,
        ),
        provider=_provider(),
    )
    season = result.seasons[0]
    training = pl.read_parquet(season.training.path)
    forecast = pl.read_parquet(season.forecast.path)

    assert set(forecast["position"]) == set(FANTASY_POSITIONS)
    assert forecast["history_through_season"].unique().to_list() == [2009]
    assert forecast["player_history_through_season"].max() == 2009
    assert forecast["opponent_history_through_season"].max() == 2009
    null_counts = forecast.select(ALL_POSITION_MODEL_FEATURE_COLUMNS).null_count()
    assert sum(null_counts.row(0)) == 0
    assert training["target_season"].max() == 2009
    assert season.manifest_path.exists()


def test_projection_commands_detect_all_position_features(
    tmp_path: Path,
    capsys,
) -> None:
    season = build_forecast_archive(
        ForecastArchiveConfig(
            output_dir=tmp_path,
            history_start=2008,
            first_target_year=2010,
            last_target_year=2010,
        ),
        provider=_provider(),
    ).seasons[0]
    predictions = tmp_path / "svr-predictions.parquet"

    exit_code = main(
        [
            "project-svr",
            "--train",
            season.training.path,
            "--forecast",
            season.forecast.path,
            "--predictions",
            str(predictions),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["features"] == list(ALL_POSITION_MODEL_FEATURE_COLUMNS)
    assert set(pl.read_parquet(predictions)["position"]) == set(FANTASY_POSITIONS)
    assert load_training_data(Path(season.training.path)).features.shape[1] == len(
        ALL_POSITION_MODEL_FEATURE_COLUMNS
    )
    assert load_projection_data(Path(season.forecast.path)).features.shape[1] == len(
        ALL_POSITION_MODEL_FEATURE_COLUMNS
    )

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ffpred.cli.app import main
from ffpred.datasets.forecast import ForecastBuildConfig, build_forecast_datasets
from ffpred.errors import ConfigurationError
from tests.factories import make_provider


def _forecast_provider():
    provider = make_provider((2020, 2021, 2022))
    provider.players = provider.players.with_columns(
        pl.lit(2020).alias("rookie_season")
    )
    provider.schedules = provider.schedules.with_columns(
        pl.col("game_id").str.slice(0, 4).cast(pl.Int64).alias("season"),
        pl.lit("REG").alias("game_type"),
        pl.lit(1, dtype=pl.Int64).alias("week"),
    )
    provider.schedules = pl.concat(
        [
            provider.schedules,
            pl.DataFrame(
                {
                    "game_id": ["2023_01_GB_SEA"],
                    "gameday": ["2023-09-10"],
                    "home_team": ["SEA"],
                    "away_team": ["GB"],
                    "home_score": [0],
                    "away_score": [0],
                    "season": [2023],
                    "game_type": ["REG"],
                    "week": [1],
                }
            ),
        ],
        how="diagonal_relaxed",
    )
    reciprocal_stats = [
        {
            "season": season,
            "week": 1,
            "season_type": "REG",
            "game_id": f"{season}_01_GB_SEA",
            "team": "SEA",
            "opponent_team": "GB",
            "passing_yards": 180,
            "rushing_yards": 90,
            "passing_interceptions": 1,
            "fumbles_lost_total": 0,
        }
        for season in (2020, 2021, 2022)
    ]
    provider.team_stats = pl.concat(
        [provider.team_stats, pl.DataFrame(reciprocal_stats)],
        how="diagonal_relaxed",
    )
    provider.players = pl.concat(
        [
            provider.players,
            pl.DataFrame(
                {
                    "gsis_id": ["00-ROOKIE"],
                    "display_name": ["Rookie Quarterback"],
                    "birth_date": ["2000-01-01"],
                    "rookie_season": [2023],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    provider.depth_charts = pl.DataFrame(
        {
            "dt": ["2023-08-30T12:00:00Z", "2023-08-30T12:00:00Z"],
            "team": ["GB", "SEA"],
            "player_name": ["Test Quarterback", "Rookie Quarterback"],
            "gsis_id": ["00-TEST", "00-ROOKIE"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1, 1],
        }
    )
    return provider


def test_forecast_config_requires_strict_season_order(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        ForecastBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            history_through_season=2023,
            target_year=2023,
        )


def test_build_forecast_freezes_prior_season_features(tmp_path: Path) -> None:
    result = build_forecast_datasets(
        ForecastBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            history_through_season=2022,
            target_year=2023,
            as_of=date(2023, 8, 31),
        ),
        provider=_forecast_provider(),
    )

    forecast = pl.read_parquet(result.forecast.path)
    assert forecast.height == 2
    assert set(forecast["player_name"]) == {
        "Test Quarterback",
        "Rookie Quarterback",
    }
    assert forecast["forecast_as_of"].unique().to_list() == ["2023-08-31"]
    assert forecast["history_through_season"].unique().to_list() == [2022]
    assert forecast["qb_history_through_season"].max() == 2022
    assert forecast["defense_history_through_season"].max() == 2022
    assert forecast["fantasy_points"].null_count() == 2
    assert Path(result.training.path).exists()
    assert result.manifest_path.exists()


@pytest.mark.parametrize("model", ["svr", "mlp"])
def test_projection_commands_write_future_predictions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
) -> None:
    result = build_forecast_datasets(
        ForecastBuildConfig(
            output_dir=tmp_path,
            history_start=2020,
            train_start=2021,
            history_through_season=2022,
            target_year=2023,
            as_of=date(2023, 8, 31),
        ),
        provider=_forecast_provider(),
    )
    predictions = tmp_path / f"{model}-predictions.parquet"

    exit_code = main(
        [
            f"project-{model}",
            "--train",
            result.training.path,
            "--forecast",
            result.forecast.path,
            "--predictions",
            str(predictions),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    frame = pl.read_parquet(predictions)

    assert exit_code == 0
    assert output["metrics"] is None
    assert output["history_through_season"] == 2022
    assert output["target_year"] == 2023
    assert frame.height == 2
    assert frame["prediction"].null_count() == 0
    assert set(frame["team"]) == {"GB", "SEA"}

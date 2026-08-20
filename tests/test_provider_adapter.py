from pathlib import Path
from unittest.mock import Mock

import polars as pl

from ffpred.providers.nflreadpy import NFLVERSE_DATA_URL, NflReadPyProvider


def test_nflreadpy_adapter_delegates_all_operations(monkeypatch) -> None:
    frame = pl.DataFrame({"value": [1]})
    load_player_stats = Mock(return_value=frame)
    load_team_stats = Mock(return_value=frame)
    load_schedules = Mock(return_value=frame)
    load_depth_charts = Mock(return_value=frame)
    load_players = Mock(return_value=frame)
    load_pbp = Mock(return_value=frame)
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.nfl.load_player_stats",
        load_player_stats,
    )
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.nfl.load_team_stats",
        load_team_stats,
    )
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.nfl.load_schedules",
        load_schedules,
    )
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.nfl.load_depth_charts",
        load_depth_charts,
    )
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.nfl.load_players",
        load_players,
    )
    monkeypatch.setattr("ffpred.providers.nflreadpy.nfl.load_pbp", load_pbp)
    provider = NflReadPyProvider()

    assert provider.load_player_stats((2024, 2025)) is frame
    assert provider.load_team_stats((2025,)) is frame
    assert provider.load_schedules((2025,)) is frame
    assert provider.load_depth_charts((2025,)) is frame
    assert provider.load_players() is frame
    assert provider.load_pbp(2025) is frame
    load_player_stats.assert_called_once_with([2024, 2025])
    load_team_stats.assert_called_once_with([2025])
    load_schedules.assert_called_once_with([2025])
    load_depth_charts.assert_called_once_with([2025])
    load_players.assert_called_once_with()
    load_pbp.assert_called_once_with(2025)


def test_nflreadpy_adapter_reports_reproducibility_metadata() -> None:
    metadata = NflReadPyProvider().metadata()

    assert metadata["client"] == "nflreadpy"
    assert metadata["client_version"]
    assert metadata["data_source"] == NFLVERSE_DATA_URL


def test_nflreadpy_adapter_applies_explicit_cache_settings(monkeypatch) -> None:
    update_config = Mock()
    monkeypatch.setattr(
        "ffpred.providers.nflreadpy.update_config",
        update_config,
    )

    NflReadPyProvider(
        cache_mode="filesystem",
        cache_dir=Path("cache"),
    )

    update_config.assert_called_once_with(
        cache_mode="filesystem",
        cache_dir=Path("cache"),
    )

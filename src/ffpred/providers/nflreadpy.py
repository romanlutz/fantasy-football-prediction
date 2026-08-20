"""nflreadpy implementation of the provider protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

NFLVERSE_DATA_URL = "https://github.com/nflverse/nflverse-data/releases"


class NflReadPyProvider:
    """Load nflverse release data through nflreadpy."""

    def __init__(
        self,
        *,
        cache_mode: Literal["off", "memory", "filesystem"] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        options: dict[str, object] = {}
        if cache_mode is not None:
            options["cache_mode"] = cache_mode
        if cache_dir is not None:
            options["cache_dir"] = cache_dir
        if options:
            update_config(**options)

    def load_player_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_player_stats(list(seasons))

    def load_team_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_team_stats(list(seasons))

    def load_schedules(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_schedules(list(seasons))

    def load_depth_charts(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_depth_charts(list(seasons))

    def load_players(self) -> pl.DataFrame:
        return nfl.load_players()

    def load_pbp(self, season: int) -> pl.DataFrame:
        return nfl.load_pbp(season)

    def metadata(self) -> Mapping[str, str]:
        return {
            "client": "nflreadpy",
            "client_version": version("nflreadpy"),
            "data_source": NFLVERSE_DATA_URL,
        }

"""nflreadpy implementation of the provider protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import version

import nflreadpy as nfl
import polars as pl

NFLVERSE_DATA_URL = "https://github.com/nflverse/nflverse-data/releases"


class NflReadPyProvider:
    """Load nflverse release data through nflreadpy."""

    def load_player_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_player_stats(list(seasons))

    def load_team_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_team_stats(list(seasons))

    def load_schedules(self, seasons: Sequence[int]) -> pl.DataFrame:
        return nfl.load_schedules(list(seasons))

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

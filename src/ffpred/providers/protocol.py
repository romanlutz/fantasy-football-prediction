"""Structural interface for NFL data providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import polars as pl


class NflDataProvider(Protocol):
    """Provider operations required by the acquisition layer."""

    def load_player_stats(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_team_stats(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_schedules(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_depth_charts(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_injuries(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_rosters_weekly(self, seasons: Sequence[int]) -> pl.DataFrame: ...

    def load_players(self) -> pl.DataFrame: ...

    def load_pbp(self, season: int) -> pl.DataFrame: ...

    def metadata(self) -> Mapping[str, str]: ...

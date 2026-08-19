"""In-memory provider for deterministic tests and examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import polars as pl


@dataclass(slots=True, kw_only=True)
class FakeProvider:
    """Return caller-supplied frames without network access."""

    player_stats: pl.DataFrame = field(default_factory=pl.DataFrame)
    team_stats: pl.DataFrame = field(default_factory=pl.DataFrame)
    schedules: pl.DataFrame = field(default_factory=pl.DataFrame)
    players: pl.DataFrame = field(default_factory=pl.DataFrame)
    pbp_by_season: dict[int, pl.DataFrame] = field(default_factory=dict)
    source_metadata: dict[str, str] = field(
        default_factory=lambda: {
            "client": "fake",
            "client_version": "1",
            "data_source": "memory",
        }
    )

    def load_player_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        del seasons
        return self.player_stats

    def load_team_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        del seasons
        return self.team_stats

    def load_schedules(self, seasons: Sequence[int]) -> pl.DataFrame:
        del seasons
        return self.schedules

    def load_players(self) -> pl.DataFrame:
        return self.players

    def load_pbp(self, season: int) -> pl.DataFrame:
        return self.pbp_by_season[season]

    def metadata(self) -> Mapping[str, str]:
        return self.source_metadata

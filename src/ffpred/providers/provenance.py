"""Provider decorator that records source-frame provenance."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from ffpred.providers.protocol import NflDataProvider


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceArtifact:
    """Content identity and schema of one provider result."""

    name: str
    rows: int
    sha256: str
    schema: Mapping[str, str]


def fingerprint_frame(name: str, frame: pl.DataFrame) -> SourceArtifact:
    """Fingerprint exact frame content using uncompressed Arrow IPC."""
    digest = hashlib.sha256()
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as buffer:
        frame.write_ipc(buffer, compression="uncompressed")
        buffer.seek(0)
        for chunk in iter(lambda: buffer.read(1024 * 1024), b""):
            digest.update(chunk)
    return SourceArtifact(
        name=name,
        rows=frame.height,
        sha256=digest.hexdigest(),
        schema={column: str(dtype) for column, dtype in frame.schema.items()},
    )


class ProvenanceProvider:
    """Record hashes while transparently delegating provider calls."""

    def __init__(self, provider: NflDataProvider) -> None:
        self._provider = provider
        self.artifacts: dict[str, SourceArtifact] = {}

    def _record(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        self.artifacts[name] = fingerprint_frame(name, frame)
        return frame

    @staticmethod
    def _season_key(seasons: Sequence[int]) -> str:
        return "-".join(str(season) for season in seasons)

    def load_player_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"player_stats:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_player_stats(seasons))

    def load_team_stats(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"team_stats:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_team_stats(seasons))

    def load_schedules(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"schedules:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_schedules(seasons))

    def load_depth_charts(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"depth_charts:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_depth_charts(seasons))

    def load_injuries(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"injuries:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_injuries(seasons))

    def load_rosters_weekly(self, seasons: Sequence[int]) -> pl.DataFrame:
        name = f"rosters_weekly:{self._season_key(seasons)}"
        return self._record(name, self._provider.load_rosters_weekly(seasons))

    def load_players(self) -> pl.DataFrame:
        return self._record("players", self._provider.load_players())

    def load_pbp(self, season: int) -> pl.DataFrame:
        name = f"play_by_play:{season}"
        return self._record(name, self._provider.load_pbp(season))

    def metadata(self) -> Mapping[str, str]:
        return self._provider.metadata()

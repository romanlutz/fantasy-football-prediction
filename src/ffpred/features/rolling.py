"""Chronological rolling-window helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from ffpred.domain.models import GameKey

GameRecord = TypeVar("GameRecord")


def previous_games(
    games: Mapping[GameKey, GameRecord],
    cutoff: GameKey,
    count: int,
) -> list[GameRecord]:
    """Return up to ``count`` games strictly before the cutoff, newest first."""
    if count < 1:
        raise ValueError("count must be positive")
    keys = sorted((key for key in games if key < cutoff), reverse=True)
    return [games[key] for key in keys[:count]]

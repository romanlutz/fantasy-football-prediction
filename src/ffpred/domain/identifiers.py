"""Strong identifier types used across package boundaries."""

from typing import NewType

PlayerId = NewType("PlayerId", str)
TeamCode = NewType("TeamCode", str)
GameId = NewType("GameId", str)
Season = NewType("Season", int)
Week = NewType("Week", int)

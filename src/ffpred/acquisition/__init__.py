"""NFL data acquisition."""

from ffpred.acquisition.normalize import (
    acquire_defense_histories,
    acquire_quarterback_histories,
)

__all__ = ["acquire_defense_histories", "acquire_quarterback_histories"]

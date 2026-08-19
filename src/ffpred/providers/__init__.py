"""NFL data provider adapters."""

from ffpred.providers.nflreadpy import NflReadPyProvider
from ffpred.providers.protocol import NflDataProvider

__all__ = ["NflDataProvider", "NflReadPyProvider"]

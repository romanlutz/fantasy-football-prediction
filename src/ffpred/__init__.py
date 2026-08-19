"""Fantasy football prediction tools."""

from ffpred.config import Settings
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig

__all__ = ["DEFAULT_SCORING", "ScoringConfig", "Settings", "__version__"]

__version__ = "0.1.0"

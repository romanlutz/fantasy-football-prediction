"""Fantasy football prediction tools."""

__version__ = "0.1.0"

from ffpred.config import Settings
from ffpred.datasets.builder import DatasetBuildConfig, build_datasets
from ffpred.datasets.manifest import DatasetManifest
from ffpred.domain.scoring import DEFAULT_SCORING, ScoringConfig
from ffpred.evaluation.metrics import RegressionMetrics, evaluate
from ffpred.training.mlp import MlpConfig, train_mlp
from ffpred.training.svr import SvrConfig, train_svr

__all__ = [
    "DEFAULT_SCORING",
    "DatasetBuildConfig",
    "DatasetManifest",
    "MlpConfig",
    "RegressionMetrics",
    "ScoringConfig",
    "Settings",
    "SvrConfig",
    "__version__",
    "build_datasets",
    "evaluate",
    "train_mlp",
    "train_svr",
]

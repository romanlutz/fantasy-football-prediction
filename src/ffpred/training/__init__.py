"""Model training implementations."""

from ffpred.training.mlp import MlpConfig, train_mlp
from ffpred.training.svr import SvrConfig, train_svr

__all__ = ["MlpConfig", "SvrConfig", "train_mlp", "train_svr"]

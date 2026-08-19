"""Project-specific exception hierarchy."""

from __future__ import annotations

from pathlib import Path


class FfpredError(Exception):
    """Base class for errors callers may handle."""


class ConfigurationError(FfpredError):
    """Raised when configuration is internally inconsistent."""


class DataAcquisitionError(FfpredError):
    """Raised when source data cannot be acquired."""


class SchemaValidationError(DataAcquisitionError):
    """Raised when provider data does not satisfy its schema contract."""

    def __init__(self, dataset: str, problems: list[str]) -> None:
        self.dataset = dataset
        self.problems = tuple(problems)
        super().__init__(f"{dataset} schema is invalid: {'; '.join(problems)}")


class DatasetGenerationError(FfpredError):
    """Raised when a requested dataset cannot be generated."""


class EmptyDatasetError(DatasetGenerationError):
    """Raised when feature engineering produces no rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"No rows were generated for {path.stem}; check season availability "
            "and ensure the history range includes prior games"
        )


class ModelTrainingError(FfpredError):
    """Raised when model fitting cannot proceed."""

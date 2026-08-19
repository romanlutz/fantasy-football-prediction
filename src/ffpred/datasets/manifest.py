"""Versioned dataset manifest serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ffpred.providers.provenance import SourceArtifact

MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetArtifact:
    """Metadata for one persisted feature table."""

    path: str
    rows: int
    columns: int
    sha256: str
    format: str = "parquet"


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildParameters:
    """Parameters that materially determine dataset contents."""

    history_start: int
    train_start: int
    test_year: int
    scoring: dict[str, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetManifest:
    """Complete identity and provenance for generated datasets."""

    generated_at: str
    package_version: str
    provider: dict[str, str]
    parameters: BuildParameters
    feature_schema_sha256: str
    sources: dict[str, SourceArtifact]
    outputs: dict[str, DatasetArtifact]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_json(self) -> str:
        """Serialize deterministically for reviews and checksums."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, value: str) -> DatasetManifest:
        """Deserialize and validate the manifest schema version."""
        raw: dict[str, Any] = json.loads(value)
        version = raw.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema version: {version}")
        return cls(
            schema_version=version,
            generated_at=raw["generated_at"],
            package_version=raw["package_version"],
            provider=dict(raw["provider"]),
            parameters=BuildParameters(**raw["parameters"]),
            feature_schema_sha256=raw["feature_schema_sha256"],
            sources={
                name: SourceArtifact(**artifact)
                for name, artifact in raw["sources"].items()
            },
            outputs={
                name: DatasetArtifact(**artifact)
                for name, artifact in raw["outputs"].items()
            },
        )

    def write(self, path: Path) -> None:
        """Write the manifest atomically."""
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> DatasetManifest:
        """Read a UTF-8 manifest."""
        return cls.from_json(path.read_text(encoding="utf-8"))

"""Runtime schema contracts at the untyped DataFrame boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from ffpred.errors import SchemaValidationError


class ColumnKind(StrEnum):
    """Provider-neutral column type categories."""

    DATE = "date"
    INTEGER = "integer"
    NUMBER = "number"
    TEXT = "text"


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameContract:
    """Expected columns, type families, and non-null keys."""

    name: str
    columns: Mapping[str, ColumnKind]
    non_null: frozenset[str] = frozenset()


def _matches_kind(dtype: pl.DataType, kind: ColumnKind) -> bool:
    if kind is ColumnKind.TEXT:
        return dtype == pl.String
    if kind is ColumnKind.INTEGER:
        return dtype.is_integer()
    if kind is ColumnKind.NUMBER:
        return dtype.is_numeric()
    return dtype == pl.String or dtype.is_temporal()


def validate_frame(frame: pl.DataFrame, contract: FrameContract) -> pl.DataFrame:
    """Validate a provider frame before domain conversion."""
    problems: list[str] = []
    for column, kind in contract.columns.items():
        if column not in frame.schema:
            problems.append(f"missing column {column!r}")
            continue
        dtype = frame.schema[column]
        if dtype == pl.Null and column not in contract.non_null:
            continue
        if not _matches_kind(dtype, kind):
            problems.append(f"{column!r} is {dtype}, expected {kind}")
    problems.extend(
        f"{column!r} contains null values"
        for column in contract.non_null
        if column in frame.columns and frame[column].null_count()
    )
    if problems:
        raise SchemaValidationError(contract.name, problems)
    return frame

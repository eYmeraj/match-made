"""Column profiling: type inference and sample values."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel

from matchmade.cleaning import clean_boolean, clean_date, clean_number
from matchmade.schemas import BooleanCleaning, ColumnType, DateCleaning, NumberCleaning

PROFILE_SAMPLE_ROWS = 20_000


class ColumnProfile(BaseModel):
    name: str
    inferred_type: ColumnType
    null_fraction: float
    distinct: int
    samples: list[str]


def _parse_rate(values: pl.Series, expr: pl.Expr) -> float:
    if values.is_empty():
        return 0.0
    parsed = pl.DataFrame({"v": values}).select(expr.alias("p")).to_series()
    return 1 - parsed.null_count() / len(values)


def infer_type(values: pl.Series) -> ColumnType:
    """Infer a column type from its non-null raw string values."""
    v = pl.col("v")
    if values.is_empty():
        return ColumnType.TEXT
    distinct = values.n_unique()
    if distinct <= 2 and _parse_rate(values, clean_boolean(v, BooleanCleaning())) == 1:
        return ColumnType.BOOLEAN
    # Plain digit strings (ids, years, yyyymmdd) are numbers, not dates.
    looks_dated = values.str.contains(r"[-/.\s]|[A-Za-z]").sum() / len(values)
    if looks_dated > 0.5 and _parse_rate(values, clean_date(v, DateCleaning())) >= 0.9:
        return ColumnType.DATE
    if _parse_rate(values, clean_number(v, NumberCleaning())) >= 0.95 and not (
        values.str.contains(r"[A-Za-z]{2,}").any()
    ):
        return ColumnType.NUMBER
    if distinct <= 50 and distinct / len(values) < 0.05:
        return ColumnType.CATEGORICAL
    return ColumnType.TEXT


def profile_frame(frame: pl.DataFrame) -> list[ColumnProfile]:
    sample = frame.head(PROFILE_SAMPLE_ROWS)
    profiles = []
    for name in frame.columns:
        column = sample.get_column(name)
        values = column.drop_nulls()
        samples = values.unique(maintain_order=True).head(8).to_list()
        profiles.append(
            ColumnProfile(
                name=name,
                inferred_type=infer_type(values),
                null_fraction=round(column.null_count() / max(len(column), 1), 4),
                distinct=values.n_unique(),
                samples=samples,
            )
        )
    return profiles

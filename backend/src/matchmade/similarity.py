"""Type-aware similarity between aligned pairs of cleaned values.

Every function returns a Float32 series in [0, 1] aligned with its inputs, with null
where either side is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from matchmade.schemas import (
    Comparison,
    DateComparison,
    NumberComparison,
    TextComparison,
    TextMetric,
)

_SCORERS: dict[TextMetric, tuple[Callable[..., Any], float]] = {
    TextMetric.RATIO: (fuzz.ratio, 100.0),
    TextMetric.TOKEN_SORT: (fuzz.token_sort_ratio, 100.0),
    TextMetric.TOKEN_SET: (fuzz.token_set_ratio, 100.0),
    TextMetric.PARTIAL: (fuzz.partial_ratio, 100.0),
    TextMetric.WRATIO: (fuzz.WRatio, 100.0),
    TextMetric.JARO_WINKLER: (JaroWinkler.normalized_similarity, 1.0),
}


def text_similarity(a: pl.Series, b: pl.Series, metric: TextMetric) -> pl.Series:
    scorer, scale = _SCORERS[metric]
    missing = (a.is_null() | b.is_null()).to_numpy()
    if len(a) == 0:
        return pl.Series(values=[], dtype=pl.Float32)
    scores = cpdist(
        a.fill_null("").to_list(),
        b.fill_null("").to_list(),
        scorer=scorer,
        dtype=np.float32,
        workers=-1,
    )
    scores = np.asarray(scores, dtype=np.float32) / np.float32(scale)
    scores[missing] = np.nan
    return pl.Series(values=scores, dtype=pl.Float32).fill_nan(None)


def _decay(diff: pl.Expr, tolerance: float) -> pl.Expr:
    if tolerance == 0:
        return (diff == 0).cast(pl.Float32)
    return (1 - diff / tolerance).clip(0, 1).cast(pl.Float32)


def _pairwise(a: pl.Series, b: pl.Series, expr: pl.Expr) -> pl.Series:
    frame = pl.DataFrame({"a": a, "b": b})
    return frame.select(
        pl.when(pl.col("a").is_null() | pl.col("b").is_null())
        .then(None)
        .otherwise(expr)
        .cast(pl.Float32)
        .alias("s")
    ).to_series()


def number_similarity(a: pl.Series, b: pl.Series, comp: NumberComparison) -> pl.Series:
    diff = (pl.col("a") - pl.col("b")).abs()
    if comp.tolerance_mode == "relative":
        scale = pl.max_horizontal(pl.col("a").abs(), pl.col("b").abs())
        diff = pl.when(scale == 0).then(0.0).otherwise(diff / scale)
    return _pairwise(a, b, _decay(diff, comp.tolerance))


def date_similarity(a: pl.Series, b: pl.Series, comp: DateComparison) -> pl.Series:
    diff = (pl.col("a") - pl.col("b")).dt.total_days().abs()
    return _pairwise(a, b, _decay(diff, comp.tolerance_days))


def exact_similarity(a: pl.Series, b: pl.Series) -> pl.Series:
    return _pairwise(a, b, (pl.col("a") == pl.col("b")).cast(pl.Float32))


def similarity(comp: Comparison, a: pl.Series, b: pl.Series) -> pl.Series:
    if isinstance(comp, TextComparison):
        return text_similarity(a, b, comp.metric)
    if isinstance(comp, NumberComparison):
        return number_similarity(a, b, comp)
    if isinstance(comp, DateComparison):
        return date_similarity(a, b, comp)
    return exact_similarity(a, b)

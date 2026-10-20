"""Scoring candidate pairs and selecting matches."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import numpy as np
import polars as pl

from matchmade.blocking import ROW, value_column
from matchmade.cleaning import clean
from matchmade.schemas import Cardinality, Comparison, MissingPolicy
from matchmade.similarity import similarity

SCORE = "score"


def score_column(comparison_id: str) -> str:
    return f"s_{comparison_id}"


def cleaned_frame(
    raw: pl.DataFrame, comparisons: Sequence[Comparison], side: Literal["a", "b"]
) -> pl.DataFrame:
    """Clean the compared columns of one dataset; row ``i`` keeps ``__row == i``."""
    exprs = [
        clean(pl.col(c.column_for(side)), c.type, c.cleaning_for(side)).alias(value_column(c.id))
        for c in comparisons
    ]
    return raw.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias(ROW), *exprs)


def score_pairs(
    pairs: pl.DataFrame,
    a: pl.DataFrame,
    b: pl.DataFrame,
    comparisons: Sequence[Comparison],
    missing: MissingPolicy,
) -> pl.DataFrame:
    """Score aligned (a, b) row pairs; returns a, b, s_<id>... and score in [0, 100]."""
    a_idx, b_idx = pairs.get_column("a"), pairs.get_column("b")
    columns: dict[str, pl.Series] = {"a": a_idx, "b": b_idx}
    for comp in comparisons:
        col = value_column(comp.id)
        columns[score_column(comp.id)] = similarity(
            comp, a.get_column(col).gather(a_idx), b.get_column(col).gather(b_idx)
        )
    frame = pl.DataFrame(columns)

    weighted, weights = [], []
    for comp in comparisons:
        s = pl.col(score_column(comp.id))
        if missing == MissingPolicy.MISMATCH:
            weighted.append(s.fill_null(0) * comp.weight)
            weights.append(pl.lit(comp.weight))
        else:
            weighted.append(s * comp.weight)
            weights.append(pl.when(s.is_not_null()).then(comp.weight).otherwise(0.0))
    total_weight = pl.sum_horizontal(weights)
    score = pl.when(total_weight > 0).then(pl.sum_horizontal(weighted) / total_weight * 100)
    return frame.with_columns(score.cast(pl.Float32).alias(SCORE)).drop_nulls(SCORE)


def score_in_chunks(
    pairs: pl.DataFrame,
    a: pl.DataFrame,
    b: pl.DataFrame,
    comparisons: Sequence[Comparison],
    missing: MissingPolicy,
    keep_from: float,
    chunk_size: int,
    on_progress: Callable[[float], None] | None = None,
) -> Iterator[pl.DataFrame]:
    total = max(pairs.height, 1)
    for offset in range(0, pairs.height, chunk_size):
        chunk = pairs.slice(offset, chunk_size)
        scored = score_pairs(chunk, a, b, comparisons, missing)
        yield scored.filter(pl.col(SCORE) >= keep_from)
        if on_progress:
            on_progress(min(offset + chunk_size, total) / total)


# --- Assignment -----------------------------------------------------------------------


def _ranked(candidates: pl.DataFrame) -> pl.DataFrame:
    # A strict total order: higher score first, ties broken by row indices.
    return candidates.sort([SCORE, "a", "b"], descending=[True, False, False]).with_row_index(
        "rank"
    )


def _greedy(ranked: pl.DataFrame) -> pl.DataFrame:
    a = ranked.get_column("a").to_numpy()
    b = ranked.get_column("b").to_numpy()
    used_a: set[int] = set()
    used_b: set[int] = set()
    keep = np.zeros(len(a), dtype=bool)
    for i in range(len(a)):
        ai, bi = int(a[i]), int(b[i])
        if ai not in used_a and bi not in used_b:
            used_a.add(ai)
            used_b.add(bi)
            keep[i] = True
    return ranked.filter(pl.Series(keep))


def one_to_one(candidates: pl.DataFrame, max_rounds: int = 50) -> pl.DataFrame:
    """Greedy 1:1 matching: repeatedly take the best remaining pair.

    Implemented as rounds of "mutual best" selection: a pair that is the best for both
    its A row and its B row is exactly what the greedy algorithm would pick, and all
    such pairs can be taken at once. Long chains fall back to a sequential sweep.
    """
    remaining = _ranked(candidates)
    chosen: list[pl.DataFrame] = []
    for _ in range(max_rounds):
        if remaining.is_empty():
            break
        best_a = remaining.group_by("a").agg(pl.col("rank").min())
        best_b = remaining.group_by("b").agg(pl.col("rank").min())
        mutual = best_a.join(best_b, on="rank").select("rank")
        picked = remaining.join(mutual, on="rank", how="semi")
        chosen.append(picked)
        remaining = remaining.join(picked.select("a"), on="a", how="anti").join(
            picked.select("b"), on="b", how="anti"
        )
    if not remaining.is_empty():
        chosen.append(_greedy(remaining.sort("rank")))
    if not chosen:
        return candidates.clear()
    return pl.concat(chosen).sort("rank").drop("rank")


def best_per_a(candidates: pl.DataFrame) -> pl.DataFrame:
    return _ranked(candidates).unique("a", keep="first").sort("rank").drop("rank")


def assign(candidates: pl.DataFrame, threshold: float, cardinality: Cardinality) -> pl.DataFrame:
    above = candidates.filter(pl.col(SCORE) >= threshold)
    match cardinality:
        case Cardinality.ONE_TO_ONE:
            return one_to_one(above)
        case Cardinality.BEST_PER_A:
            return best_per_a(above)
        case Cardinality.ALL:
            return _ranked(above).drop("rank")

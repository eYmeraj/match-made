"""Candidate pair generation.

Comparing every A row with every B row is infeasible at a million rows each, so blocking
rules pick the pairs worth scoring. Rules are OR-ed together:

- key rules join A and B on a derived key (e.g. first token of the name AND birth year);
- TF-IDF rules take each A value's top-k nearest B values by character n-gram cosine.

Both sides arrive as "cleaned frames": a ``__row`` column holding the original row index
plus one ``c_<comparison id>`` column per comparison with the cleaned, typed values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jellyfish
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn

from matchmade.schemas import (
    Blocking,
    BlockingKeyPart,
    BlockingRule,
    ExactBlockingRule,
    KeyTransform,
    TfidfBlockingRule,
)

ROW = "__row"

# Called with the completed fraction (0-1) of the current step.
ProgressFn = Callable[[float], None]


def value_column(comparison_id: str) -> str:
    return f"c_{comparison_id}"


class BlockingError(ValueError):
    pass


@dataclass
class RuleStats:
    rule_index: int
    pairs: int
    skipped_blocks: int = 0
    skipped_pairs: int = 0


@dataclass
class BlockingResult:
    pairs: pl.DataFrame  # columns a, b (UInt32), unique
    rules: list[RuleStats] = field(default_factory=list)


# --- Key rules ----------------------------------------------------------------------


def _phonetic(series: pl.Series) -> pl.Series:
    uniques = series.drop_nulls().unique()
    codes = [jellyfish.metaphone(v) or None for v in uniques.to_list()]
    mapping = pl.DataFrame(
        {"v": uniques, "k": pl.Series(codes, dtype=pl.String)},
        schema={"v": series.dtype, "k": pl.String},
    )
    return (
        pl.DataFrame({"v": series})
        .join(mapping, on="v", how="left", maintain_order="left")
        .get_column("k")
    )


def derive_key(series: pl.Series, part: BlockingKeyPart) -> pl.Series:
    s = series
    match part.transform:
        case KeyTransform.EXACT:
            key = s.cast(pl.String)
        case KeyTransform.PREFIX:
            key = s.cast(pl.String).str.slice(0, int(part.param or 1))
        case KeyTransform.FIRST_TOKEN:
            key = s.cast(pl.String).str.split(" ").list.first()
        case KeyTransform.SORTED_TOKENS:
            key = s.cast(pl.String).str.split(" ").list.sort().list.join(" ")
        case KeyTransform.PHONETIC:
            key = _phonetic(s.cast(pl.String))
        case KeyTransform.YEAR:
            key = s.dt.year().cast(pl.String)
        case KeyTransform.YEAR_MONTH:
            key = s.dt.strftime("%Y-%m")
        case KeyTransform.ROUND:
            bucket = float(part.param or 1)
            key = (s / bucket).floor().cast(pl.Int64).cast(pl.String)
    return pl.select(pl.when(key.str.len_chars() > 0).then(key)).to_series()


def _keyed(frame: pl.DataFrame, rule: ExactBlockingRule) -> pl.DataFrame:
    keys = {
        f"k{i}": derive_key(frame.get_column(value_column(p.comparison_id)), p)
        for i, p in enumerate(rule.parts)
    }
    return pl.DataFrame({ROW: frame.get_column(ROW), **keys}).drop_nulls()


@dataclass
class _KeyBlocks:
    a: pl.DataFrame
    b: pl.DataFrame
    sizes: pl.DataFrame  # one row per key with n_a, n_b, pairs, oversized


def _key_blocks(
    a: pl.DataFrame, b: pl.DataFrame, rule: ExactBlockingRule, max_block_pairs: int
) -> _KeyBlocks:
    ka, kb = _keyed(a, rule), _keyed(b, rule)
    key_cols = [f"k{i}" for i in range(len(rule.parts))]
    sizes = (
        ka.group_by(key_cols)
        .len("n_a")
        .join(kb.group_by(key_cols).len("n_b"), on=key_cols)
        .with_columns(pairs=pl.col("n_a").cast(pl.Int64) * pl.col("n_b"))
        .with_columns(oversized=pl.col("pairs") > max_block_pairs)
    )
    return _KeyBlocks(ka, kb, sizes)


def _rule_stats_from_sizes(index: int, sizes: pl.DataFrame) -> RuleStats:
    kept = sizes.filter(~pl.col("oversized"))
    skipped = sizes.filter(pl.col("oversized"))
    return RuleStats(
        rule_index=index,
        pairs=int(kept.get_column("pairs").sum()),
        skipped_blocks=skipped.height,
        skipped_pairs=int(skipped.get_column("pairs").sum()),
    )


def _key_pairs(blocks: _KeyBlocks, n_parts: int) -> pl.DataFrame:
    key_cols = [f"k{i}" for i in range(n_parts)]
    allowed = blocks.sizes.filter(~pl.col("oversized")).select(key_cols)
    a = blocks.a.join(allowed, on=key_cols, how="semi")
    return a.join(blocks.b, on=key_cols, suffix="_b").select(
        pl.col(ROW).alias("a"), pl.col(f"{ROW}_b").alias("b")
    )


# --- TF-IDF rules -------------------------------------------------------------------

# Exact top-k cosine over all A x B value pairs costs sum_f df_a(f) * df_b(f), which reaches
# ~1e12 at a million rows because common n-grams are shared by tens of thousands of values.
# Instead, candidates are the B values sharing the most of each A value's rarest n-grams
# (each row keeps at most _RARE_FEATURES), then re-ranked by the full cosine. Oversampling
# before re-ranking recovers neighbours whose rare n-grams happen to differ.
_RARE_FEATURES = 12
_OVERSAMPLE = 4
# Chunking bounds memory and keeps each thread's accumulator over B values cache-sized.
_CHUNK_A = 50_000
_CHUNK_B = 100_000


def _n_rows(m: sp.csr_matrix) -> int:
    return len(m.indptr) - 1


def _rarest_features(m: sp.csr_matrix, idf: np.ndarray, k: int) -> sp.csr_matrix:
    """Binary matrix keeping each row's k highest-idf features."""
    rows = np.repeat(np.arange(_n_rows(m)), np.diff(m.indptr))
    order = np.lexsort((-idf[m.indices], rows))
    rank = np.arange(len(order)) - m.indptr[rows[order]]
    keep = np.sort(order[rank < k])
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows[keep], minlength=_n_rows(m)))])
    return sp.csr_matrix(
        (np.ones(len(keep), dtype=np.float32), m.indices[keep], indptr), shape=m.shape
    )


def _nearest_values(
    ma: sp.csr_matrix,
    mb: sp.csr_matrix,
    idf: np.ndarray,
    rule: TfidfBlockingRule,
    on_progress: ProgressFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into ma and mb of each A value's top-k neighbours by cosine."""
    rare_a = _rarest_features(ma, idf, _RARE_FEATURES)
    rare_b = _rarest_features(mb, idf, _RARE_FEATURES)
    n_a, n_b = _n_rows(ma), _n_rows(mb)
    b_chunks = [rare_b[s : s + _CHUNK_B].T.tocsr() for s in range(0, n_b, _CHUNK_B)]
    n_candidates = rule.top_k * _OVERSAMPLE
    found_a, found_b = [], []
    for start in range(0, n_a, _CHUNK_A):
        stop = min(start + _CHUNK_A, n_a)
        parts = [
            sp_matmul_topn(rare_a[start:stop], chunk, top_n=n_candidates, n_threads=-1)
            for chunk in b_chunks
        ]
        shared = (parts[0] if len(parts) == 1 else zip_sp_matmul_topn(n_candidates, parts)).tocoo()
        cosine = np.asarray(ma[start:stop][shared.row].multiply(mb[shared.col]).sum(axis=1))
        best = (
            pl.DataFrame({"a": shared.row + start, "b": shared.col, "s": cosine.ravel()})
            .filter(pl.col("s") >= rule.min_similarity)
            .sort("a", "s", descending=[False, True])
            .group_by("a", maintain_order=True)
            .head(rule.top_k)
        )
        found_a.append(best.get_column("a").to_numpy())
        found_b.append(best.get_column("b").to_numpy())
        on_progress(stop / n_a)
    return np.concatenate(found_a), np.concatenate(found_b)


def _value_rows(frame: pl.DataFrame, comparison_id: str) -> pl.DataFrame:
    col = value_column(comparison_id)
    return frame.select(ROW, pl.col(col).cast(pl.String).alias("v")).drop_nulls()


def _value_pair_sizes(
    value_pairs: pl.DataFrame, a_rows: pl.DataFrame, b_rows: pl.DataFrame, max_block_pairs: int
) -> pl.DataFrame:
    """Row pairs behind each (A value, B value) pair, flagged like oversized key blocks.

    A value repeated on both sides, such as a placeholder like "n a", would otherwise
    multiply into more row pairs than fit in memory.
    """
    return (
        value_pairs.join(a_rows.group_by("v").len("n_a").rename({"v": "va"}), on="va")
        .join(b_rows.group_by("v").len("n_b").rename({"v": "vb"}), on="vb")
        .with_columns(pairs=pl.col("n_a").cast(pl.Int64) * pl.col("n_b"))
        .with_columns(oversized=pl.col("pairs") > max_block_pairs)
    )


def _tfidf_pairs(
    a: pl.DataFrame,
    b: pl.DataFrame,
    rule: TfidfBlockingRule,
    max_block_pairs: int,
    on_progress: ProgressFn,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Candidate row pairs, and the value pair sizes they came from."""
    a_rows, b_rows = _value_rows(a, rule.comparison_id), _value_rows(b, rule.comparison_id)
    a_values = a_rows.get_column("v").unique(maintain_order=True)
    b_values = b_rows.get_column("v").unique(maintain_order=True)
    empty = (
        pl.DataFrame(schema={"a": pl.UInt32, "b": pl.UInt32}),
        pl.DataFrame(schema={"pairs": pl.Int64, "oversized": pl.Boolean}),
    )
    if a_values.is_empty() or b_values.is_empty():
        on_progress(1.0)
        return empty

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        dtype=np.float32,  # pyright: ignore[reportArgumentType]
        sublinear_tf=True,
    )
    vectorizer.fit(pl.concat([a_values, b_values]).unique().to_list())
    ma = vectorizer.transform(a_values.to_list()).tocsr()
    mb = vectorizer.transform(b_values.to_list()).tocsr()
    rows, cols = _nearest_values(ma, mb, vectorizer.idf_, rule, on_progress)
    if len(rows) == 0:
        return empty

    value_pairs = pl.DataFrame({"va": a_values.gather(rows), "vb": b_values.gather(cols)})
    sizes = _value_pair_sizes(value_pairs, a_rows, b_rows, max_block_pairs)
    pairs = (
        sizes.filter(~pl.col("oversized"))
        .select("va", "vb")
        .join(a_rows.rename({ROW: "a", "v": "va"}), on="va")
        .join(b_rows.rename({ROW: "b", "v": "vb"}), on="vb")
        .select("a", "b")
    )
    return pairs, sizes


def _tfidf_estimate(
    a: pl.DataFrame, b: pl.DataFrame, rule: TfidfBlockingRule, index: int, max_block_pairs: int
) -> RuleStats:
    """Rough pair count from average duplication; values equal on both sides are always
    each other's neighbours, so the ones that would be skipped are known exactly."""
    a_rows, b_rows = _value_rows(a, rule.comparison_id), _value_rows(b, rule.comparison_id)
    if b_rows.is_empty():
        return RuleStats(rule_index=index, pairs=0)
    rows_per_value = b_rows.height / max(b_rows.get_column("v").n_unique(), 1)
    pairs = int(a_rows.height * min(rule.top_k * rows_per_value, b_rows.height))
    shared = a_rows.select(pl.col("v").unique()).join(b_rows.select(pl.col("v").unique()), on="v")
    identical = shared.select(pl.col("v").alias("va"), pl.col("v").alias("vb"))
    skipped = _rule_stats_from_sizes(
        index, _value_pair_sizes(identical, a_rows, b_rows, max_block_pairs)
    )
    return RuleStats(
        rule_index=index,
        pairs=pairs,
        skipped_blocks=skipped.skipped_blocks,
        skipped_pairs=skipped.skipped_pairs,
    )


# --- Public API ---------------------------------------------------------------------


def estimate(a: pl.DataFrame, b: pl.DataFrame, blocking: Blocking) -> list[RuleStats]:
    """Cheap per-rule pair counts (exact for key rules, approximate for TF-IDF)."""
    stats = []
    for i, rule in enumerate(blocking.rules):
        if isinstance(rule, ExactBlockingRule):
            blocks = _key_blocks(a, b, rule, blocking.max_block_pairs)
            stats.append(_rule_stats_from_sizes(i, blocks.sizes))
        else:
            stats.append(_tfidf_estimate(a, b, rule, i, blocking.max_block_pairs))
    return stats


def _rule_pairs(
    a: pl.DataFrame,
    b: pl.DataFrame,
    rule: BlockingRule,
    index: int,
    max_block_pairs: int,
    on_progress: ProgressFn,
) -> tuple[pl.DataFrame, RuleStats]:
    if isinstance(rule, ExactBlockingRule):
        blocks = _key_blocks(a, b, rule, max_block_pairs)
        return _key_pairs(blocks, len(rule.parts)), _rule_stats_from_sizes(index, blocks.sizes)
    pairs, sizes = _tfidf_pairs(a, b, rule, max_block_pairs, on_progress)
    return pairs, _rule_stats_from_sizes(index, sizes)


def candidate_pairs(
    a: pl.DataFrame,
    b: pl.DataFrame,
    blocking: Blocking,
    max_candidates: int,
    on_progress: ProgressFn = lambda _: None,
) -> BlockingResult:
    if not blocking.rules:
        total = a.height * b.height
        if total > max_candidates:
            raise BlockingError(
                f"Comparing every row pair would create {total:,} pairs "
                f"(limit {max_candidates:,}). Add a blocking rule."
            )
        pairs = a.select(pl.col(ROW).alias("a")).join(b.select(pl.col(ROW).alias("b")), how="cross")
        return BlockingResult(pairs=pairs, rules=[])

    for stat in estimate(a, b, blocking):
        rule = blocking.rules[stat.rule_index]
        if isinstance(rule, ExactBlockingRule) and stat.pairs > max_candidates:
            raise BlockingError(
                f"Blocking rule {stat.rule_index + 1} would create {stat.pairs:,} pairs "
                f"(limit {max_candidates:,}). Make it more specific."
            )

    frames, stats = [], []
    n_rules = len(blocking.rules)
    for i, rule in enumerate(blocking.rules):

        def rule_progress(fraction: float, i: int = i) -> None:
            on_progress((i + fraction) / n_rules)

        pairs, stat = _rule_pairs(a, b, rule, i, blocking.max_block_pairs, rule_progress)
        frames.append(pairs.cast({"a": pl.UInt32, "b": pl.UInt32}))
        stats.append(stat)
        rule_progress(1.0)

    pairs = pl.concat(frames).unique()
    if pairs.height > max_candidates:
        raise BlockingError(
            f"Blocking rules together create {pairs.height:,} pairs "
            f"(limit {max_candidates:,}). Make them more specific."
        )
    return BlockingResult(pairs=pairs.sort("a", "b"), rules=stats)

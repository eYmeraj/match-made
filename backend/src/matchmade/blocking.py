"""Candidate pair generation.

Comparing every A row with every B row is infeasible at a million rows each, so blocking
rules pick the pairs worth scoring. Rules are OR-ed together:

- key rules join A and B on a derived key (e.g. first token of the name AND birth year);
- TF-IDF rules take each A value's top-k nearest B values by character n-gram cosine.

Both sides arrive as "cleaned frames": a ``__row`` column holding the original row index
plus one ``c_<comparison id>`` column per comparison with the cleaned, typed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jellyfish
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from matchmade.schemas import (
    Blocking,
    BlockingKeyPart,
    BlockingRule,
    ExactBlockingRule,
    KeyTransform,
    TfidfBlockingRule,
)

ROW = "__row"


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


def _tfidf_pairs(a: pl.DataFrame, b: pl.DataFrame, rule: TfidfBlockingRule) -> pl.DataFrame:
    col = value_column(rule.comparison_id)
    a_rows = a.select(ROW, pl.col(col).cast(pl.String).alias("v")).drop_nulls()
    b_rows = b.select(ROW, pl.col(col).cast(pl.String).alias("v")).drop_nulls()
    a_values = a_rows.get_column("v").unique(maintain_order=True)
    b_values = b_rows.get_column("v").unique(maintain_order=True)
    empty = pl.DataFrame(schema={"a": pl.UInt32, "b": pl.UInt32})
    if a_values.is_empty() or b_values.is_empty():
        return empty

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        dtype=np.float32,  # pyright: ignore[reportArgumentType]
        sublinear_tf=True,
    )
    vectorizer.fit(pl.concat([a_values, b_values]).unique().to_list())
    ma = vectorizer.transform(a_values.to_list())
    mb = vectorizer.transform(b_values.to_list())
    product = sp_matmul_topn(
        ma, mb.T, top_n=rule.top_k, threshold=rule.min_similarity, n_threads=-1
    ).tocoo()
    if product.nnz == 0:
        return empty

    value_pairs = pl.DataFrame(
        {
            "va": a_values.gather(product.row.astype(np.int64)),
            "vb": b_values.gather(product.col.astype(np.int64)),
        }
    )
    return (
        value_pairs.join(a_rows.rename({ROW: "a", "v": "va"}), on="va")
        .join(b_rows.rename({ROW: "b", "v": "vb"}), on="vb")
        .select("a", "b")
    )


def _tfidf_upper_bound(a: pl.DataFrame, b: pl.DataFrame, rule: TfidfBlockingRule) -> int:
    col = value_column(rule.comparison_id)
    b_vals = b.get_column(col).drop_nulls()
    if b_vals.is_empty():
        return 0
    rows_per_value = len(b_vals) / max(b_vals.n_unique(), 1)
    n_a = a.get_column(col).drop_nulls().len()
    return int(n_a * min(rule.top_k * rows_per_value, len(b_vals)))


# --- Public API ---------------------------------------------------------------------


def estimate(a: pl.DataFrame, b: pl.DataFrame, blocking: Blocking) -> list[RuleStats]:
    """Cheap per-rule pair counts (exact for key rules, an upper bound for TF-IDF)."""
    stats = []
    for i, rule in enumerate(blocking.rules):
        if isinstance(rule, ExactBlockingRule):
            blocks = _key_blocks(a, b, rule, blocking.max_block_pairs)
            stats.append(_rule_stats_from_sizes(i, blocks.sizes))
        else:
            stats.append(RuleStats(rule_index=i, pairs=_tfidf_upper_bound(a, b, rule)))
    return stats


def _rule_pairs(
    a: pl.DataFrame, b: pl.DataFrame, rule: BlockingRule, index: int, max_block_pairs: int
) -> tuple[pl.DataFrame, RuleStats]:
    if isinstance(rule, ExactBlockingRule):
        blocks = _key_blocks(a, b, rule, max_block_pairs)
        return _key_pairs(blocks, len(rule.parts)), _rule_stats_from_sizes(index, blocks.sizes)
    pairs = _tfidf_pairs(a, b, rule)
    return pairs, RuleStats(rule_index=index, pairs=pairs.height)


def candidate_pairs(
    a: pl.DataFrame, b: pl.DataFrame, blocking: Blocking, max_candidates: int
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
    for i, rule in enumerate(blocking.rules):
        pairs, stat = _rule_pairs(a, b, rule, i, blocking.max_block_pairs)
        frames.append(pairs.cast({"a": pl.UInt32, "b": pl.UInt32}))
        stats.append(stat)

    pairs = pl.concat(frames).unique()
    if pairs.height > max_candidates:
        raise BlockingError(
            f"Blocking rules together create {pairs.height:,} pairs "
            f"(limit {max_candidates:,}). Make them more specific."
        )
    return BlockingResult(pairs=pairs.sort("a", "b"), rules=stats)

"""Turning persisted candidates into matches, statistics and downloadable files."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Literal

import polars as pl
from pydantic import BaseModel

from matchmade.matching import SCORE, assign, score_column
from matchmade.schemas import Cardinality, MatchConfig
from matchmade.storage import JobSummary, NotFoundError, Storage

XLSX_MAX_ROWS = 1_048_575  # Excel's sheet limit minus the header row.
HISTOGRAM_BIN = 2


class ExportKind(StrEnum):
    MERGED = "merged"
    UNMATCHED_A = "unmatched_a"
    UNMATCHED_B = "unmatched_b"


class JoinMode(StrEnum):
    MATCHED = "matched"  # only matched pairs
    LEFT = "left"  # every A row, with its match if any
    FULL = "full"  # every A and B row


ExportFormat = Literal["csv", "xlsx"]


class ExportTooLargeError(ValueError):
    pass


class HistogramBin(BaseModel):
    start: float
    count: int


class ResultStats(BaseModel):
    threshold: float
    cardinality: Cardinality
    matches: int
    matched_a: int
    matched_b: int
    unmatched_a: int
    unmatched_b: int
    mean_score: float | None


class PairValue(BaseModel):
    comparison_id: str
    a: str | None
    b: str | None
    score: float | None


class MatchRow(BaseModel):
    row_a: int
    row_b: int
    score: float
    values: list[PairValue]


class MatchPage(BaseModel):
    total: int
    rows: list[MatchRow]


def load_summary(storage: Storage, job_id: str) -> JobSummary:
    path = storage.job_dir(job_id) / "summary.json"
    if not path.exists():
        raise NotFoundError(job_id)
    return JobSummary.model_validate_json(path.read_text())


def _candidates(storage: Storage, job_id: str) -> pl.LazyFrame:
    return pl.scan_parquet(storage.job_dir(job_id) / "candidates" / "*.parquet")


class _MatchCache:
    """Assignment on millions of candidates takes seconds; keep the latest few."""

    def __init__(self, size: int = 4) -> None:
        self._size = size
        self._items: OrderedDict[tuple[str, float, Cardinality], pl.DataFrame] = OrderedDict()
        self._lock = Lock()

    def get(self, storage: Storage, job_id: str, threshold: float, cardinality: Cardinality):
        key = (job_id, threshold, cardinality)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        candidates = _candidates(storage, job_id).filter(pl.col(SCORE) >= threshold).collect()
        matches = assign(candidates, threshold, cardinality)
        with self._lock:
            self._items[key] = matches
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return matches


_cache = _MatchCache()


def matches(
    storage: Storage, job_id: str, threshold: float, cardinality: Cardinality
) -> pl.DataFrame:
    summary = load_summary(storage, job_id)
    if threshold < summary.keep_from:
        raise ValueError(
            f"Threshold must be at least {summary.keep_from:g} for this run; "
            "lower-scoring pairs were not kept."
        )
    return _cache.get(storage, job_id, threshold, cardinality)


def histogram(storage: Storage, job_id: str) -> list[HistogramBin]:
    """Distribution of each A row's best candidate score."""
    best = (
        _candidates(storage, job_id)
        .group_by("a")
        .agg(pl.col(SCORE).max())
        .select(((pl.col(SCORE) // HISTOGRAM_BIN) * HISTOGRAM_BIN).clip(0, 100 - HISTOGRAM_BIN))
        .group_by(SCORE)
        .len()
        .sort(SCORE)
        .collect()
    )
    return [HistogramBin(start=float(s), count=int(n)) for s, n in best.iter_rows()]


def stats(storage: Storage, job_id: str, threshold: float, cardinality: Cardinality) -> ResultStats:
    summary = load_summary(storage, job_id)
    found = matches(storage, job_id, threshold, cardinality)
    matched_a = found.get_column("a").n_unique() if found.height else 0
    matched_b = found.get_column("b").n_unique() if found.height else 0
    mean = found.get_column(SCORE).mean() if found.height else None
    return ResultStats(
        threshold=threshold,
        cardinality=cardinality,
        matches=found.height,
        matched_a=matched_a,
        matched_b=matched_b,
        unmatched_a=summary.rows_a - matched_a,
        unmatched_b=summary.rows_b - matched_b,
        mean_score=round(float(mean), 2) if mean is not None else None,  # type: ignore[arg-type]
    )


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def match_page(
    storage: Storage,
    job_id: str,
    threshold: float,
    cardinality: Cardinality,
    order: Literal["asc", "desc"],
    offset: int,
    limit: int,
) -> MatchPage:
    config = storage.job_config(job_id)
    found = matches(storage, job_id, threshold, cardinality)
    page = found.sort([SCORE, "a", "b"], descending=[order == "desc", False, False]).slice(
        offset, limit
    )
    raw_a = storage.read_dataset(config.dataset_a, list({c.column_a for c in config.comparisons}))
    raw_b = storage.read_dataset(config.dataset_b, list({c.column_b for c in config.comparisons}))
    a_rows = raw_a[page.get_column("a").to_list()]
    b_rows = raw_b[page.get_column("b").to_list()]

    rows = []
    for i, pair in enumerate(page.iter_rows(named=True)):
        values = [
            PairValue(
                comparison_id=c.id,
                a=_as_str(a_rows[i, c.column_a]),
                b=_as_str(b_rows[i, c.column_b]),
                score=None
                if pair[score_column(c.id)] is None
                else round(pair[score_column(c.id)] * 100, 1),
            )
            for c in config.comparisons
        ]
        rows.append(
            MatchRow(row_a=pair["a"], row_b=pair["b"], score=round(pair[SCORE], 2), values=values)
        )
    return MatchPage(total=found.height, rows=rows)


# --- Exports --------------------------------------------------------------------------


def _output_names(a_cols: list[str], b_cols: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    shared = set(a_cols) & set(b_cols)
    a_map = {c: f"{c} (A)" if c in shared else c for c in a_cols}
    b_map = {c: f"{c} (B)" if c in shared else c for c in b_cols}
    return a_map, b_map


def _score_columns(config: MatchConfig, taken: set[str]) -> dict[str, str]:
    names = {}
    for c in config.comparisons:
        base = (
            f"score: {c.column_a}"
            if c.column_a == c.column_b
            else (f"score: {c.column_a} ~ {c.column_b}")
        )
        name, n = base, 1
        while name in taken:
            n += 1
            name = f"{base} ({n})"
        taken.add(name)
        names[score_column(c.id)] = name
    return names


def build_export(
    storage: Storage,
    job_id: str,
    kind: ExportKind,
    threshold: float,
    cardinality: Cardinality,
    join: JoinMode,
) -> pl.DataFrame:
    config = storage.job_config(job_id)
    found = matches(storage, job_id, threshold, cardinality)
    a = storage.read_dataset(config.dataset_a).with_row_index("__a", offset=0)
    b = storage.read_dataset(config.dataset_b).with_row_index("__b", offset=0)

    if kind == ExportKind.UNMATCHED_A:
        return a.join(found.select(pl.col("a").alias("__a")), on="__a", how="anti").drop("__a")
    if kind == ExportKind.UNMATCHED_B:
        return b.join(found.select(pl.col("b").alias("__b")), on="__b", how="anti").drop("__b")

    a_cols = [c for c in a.columns if c != "__a"]
    b_cols = [c for c in b.columns if c != "__b"]
    a_map, b_map = _output_names(a_cols, b_cols)
    scores = _score_columns(config, set(a_map.values()) | set(b_map.values()))
    pairs = found.select(
        pl.col("a").alias("__a"),
        pl.col("b").alias("__b"),
        pl.col(SCORE).round(2).alias("match_score"),
        *[(pl.col(src) * 100).round(1).alias(dst) for src, dst in scores.items()],
    )
    a_side = a.rename(a_map)
    b_side = b.rename(b_map)
    if join == JoinMode.MATCHED:
        merged = pairs.join(a_side, on="__a").join(b_side, on="__b")
    elif join == JoinMode.LEFT:
        merged = a_side.join(pairs, on="__a", how="left").join(b_side, on="__b", how="left")
    else:
        merged = a_side.join(pairs, on="__a", how="full", coalesce=True).join(
            b_side, on="__b", how="full", coalesce=True
        )
    merged = merged.sort(["__a", "__b"], nulls_last=True).with_columns(
        (pl.col("__a") + 1).alias("row (A)"), (pl.col("__b") + 1).alias("row (B)")
    )
    return merged.select(
        "match_score", *scores.values(), "row (A)", "row (B)", *a_map.values(), *b_map.values()
    )


def export_file(
    storage: Storage,
    job_id: str,
    kind: ExportKind,
    fmt: ExportFormat,
    threshold: float,
    cardinality: Cardinality,
    join: JoinMode,
) -> Path:
    key = f"{kind}-{threshold}-{cardinality}-{join}-{fmt}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    directory = storage.job_dir(job_id) / "exports"
    directory.mkdir(exist_ok=True)
    path = directory / f"{kind}-{digest}.{fmt}"
    if path.exists():
        return path

    frame = build_export(storage, job_id, kind, threshold, cardinality, join)
    tmp = path.with_suffix(f".tmp.{fmt}")
    if fmt == "xlsx":
        if frame.height > XLSX_MAX_ROWS:
            raise ExportTooLargeError(
                f"{frame.height:,} rows exceed Excel's limit of {XLSX_MAX_ROWS:,}. Use CSV."
            )
        frame.write_excel(tmp, worksheet=kind.value, autofit=frame.height <= 10_000)
    else:
        frame.write_csv(tmp)
    tmp.replace(path)
    return path

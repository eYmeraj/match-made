"""Auto-suggested comparisons and blocking from two profiled datasets."""

from __future__ import annotations

import re

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from matchmade.cleaning import DEFAULT_CLEANING, clean
from matchmade.profiling import ColumnProfile
from matchmade.schemas import (
    Blocking,
    BlockingKeyPart,
    BooleanComparison,
    CategoricalComparison,
    Cleaning,
    ColumnType,
    Comparison,
    DateCleaning,
    DateComparison,
    ExactBlockingRule,
    KeyTransform,
    NumberComparison,
    TextCleaning,
    TextComparison,
    TfidfBlockingRule,
)
from matchmade.storage import Storage

_SAMPLE_ROWS = 20_000
_MIN_PAIR_SCORE = 0.55
_MIN_DISTINCT_FOR_OVERLAP = 3
_FUZZY_PROBES = 100
_FUZZY_TARGETS = 2_000
_FUZZY_MATCH = 85
_ORGANISATION_CLEANING = TextCleaning(remove_legal_suffixes=True, expand_abbreviations=True)
# Below this many row pairs, comparing everything is cheaper than blocking.
_FULL_COMPARE_LIMIT = 1_000_000

_ORGANISATION_HINTS = re.compile(
    r"compan|org|firm|vendor|supplier|customer|client|employer|business|merchant|account|"
    r"entity|partner|brand"
)
_NAME_HINTS = re.compile(r"name|title|label")


def _normalise_name(name: str) -> str:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[^a-z0-9]+", " ", spaced.lower()).strip()


def _name_similarity(a: str, b: str) -> float:
    return fuzz.token_set_ratio(_normalise_name(a), _normalise_name(b)) / 100


def _compatible(a: ColumnType, b: ColumnType) -> ColumnType | None:
    if a == b:
        return a
    loose = {ColumnType.TEXT, ColumnType.CATEGORICAL}
    if {a, b} <= loose:
        return ColumnType.TEXT
    return None


def _cleaned_uniques(values: pl.Series, column_type: ColumnType) -> pl.Series:
    match column_type:
        case ColumnType.TEXT:
            cleaning: Cleaning = _ORGANISATION_CLEANING
        case ColumnType.DATE:
            cleaning = DateCleaning(day_first=_looks_day_first(values))
        case _:
            cleaning = DEFAULT_CLEANING[column_type]()
    frame = pl.DataFrame({"v": values}).select(clean(pl.col("v"), column_type, cleaning))
    return frame.to_series().drop_nulls().unique()


def _value_overlap(a: pl.Series, b: pl.Series, column_type: ColumnType) -> float:
    """Share of A's distinct values that also occur (fuzzily, for text) in B."""
    if column_type == ColumnType.BOOLEAN:
        return 0.0  # two possible values overlap trivially
    ua, ub = _cleaned_uniques(a, column_type), _cleaned_uniques(b, column_type)
    if min(len(ua), len(ub)) < _MIN_DISTINCT_FOR_OVERLAP:
        return 0.0
    if column_type == ColumnType.TEXT:
        probe = ua.sample(min(len(ua), _FUZZY_PROBES), seed=0).to_list()
        targets = ub.sample(min(len(ub), _FUZZY_TARGETS), seed=0).to_list()
        best = cdist(probe, targets, scorer=fuzz.token_sort_ratio, workers=-1).max(axis=1)
        return float((best >= _FUZZY_MATCH).mean())
    return len(set(ua.to_list()) & set(ub.to_list())) / min(len(ua), len(ub))


def _looks_day_first(values: pl.Series) -> bool:
    """True when some value can only be read as day/month (first part above 12)."""
    return bool(values.drop_nulls().str.contains(r"^\s*(1[3-9]|2\d|3[01])[/.-]\d{1,2}[/.-]").any())


def _comparison(
    cid: str,
    column_a: str,
    column_b: str,
    column_type: ColumnType,
    day_first: tuple[bool, bool] = (False, False),
) -> Comparison:
    names = f"{_normalise_name(column_a)} {_normalise_name(column_b)}"
    match column_type:
        case ColumnType.TEXT:
            organisation = bool(_ORGANISATION_HINTS.search(names))
            return TextComparison(
                id=cid,
                column_a=column_a,
                column_b=column_b,
                weight=2.0 if organisation or _NAME_HINTS.search(names) else 1.0,
                cleaning=TextCleaning(
                    remove_legal_suffixes=organisation, expand_abbreviations=organisation
                ),
            )
        case ColumnType.NUMBER:
            return NumberComparison(id=cid, column_a=column_a, column_b=column_b)
        case ColumnType.DATE:
            day_first_a, day_first_b = day_first
            return DateComparison(
                id=cid,
                column_a=column_a,
                column_b=column_b,
                cleaning=DateCleaning(day_first=day_first_a),
                cleaning_b=DateCleaning(day_first=day_first_b)
                if day_first_b != day_first_a
                else None,
            )
        case ColumnType.BOOLEAN:
            return BooleanComparison(id=cid, column_a=column_a, column_b=column_b)
        case ColumnType.CATEGORICAL:
            return CategoricalComparison(id=cid, column_a=column_a, column_b=column_b)


def suggest_comparisons(storage: Storage, dataset_a: str, dataset_b: str) -> list[Comparison]:
    meta_a, meta_b = storage.dataset_meta(dataset_a), storage.dataset_meta(dataset_b)
    sample_a = storage.read_dataset(dataset_a).head(_SAMPLE_ROWS)
    sample_b = storage.read_dataset(dataset_b).head(_SAMPLE_ROWS)

    scored: list[tuple[float, ColumnProfile, ColumnProfile, ColumnType]] = []
    for pa in meta_a.columns:
        for pb in meta_b.columns:
            column_type = _compatible(pa.inferred_type, pb.inferred_type)
            if column_type is None:
                continue
            name = _name_similarity(pa.name, pb.name)
            overlap = _value_overlap(sample_a[pa.name], sample_b[pb.name], column_type)
            score = max(name, 0.6 * name + 0.6 * overlap)
            if score >= _MIN_PAIR_SCORE:
                scored.append((score, pa, pb, column_type))

    used_a: set[str] = set()
    used_b: set[str] = set()
    chosen: list[tuple[ColumnProfile, ColumnProfile, ColumnType]] = []
    for _, pa, pb, column_type in sorted(scored, key=lambda s: -s[0]):
        if pa.name not in used_a and pb.name not in used_b:
            used_a.add(pa.name)
            used_b.add(pb.name)
            chosen.append((pa, pb, column_type))
    order = {p.name: i for i, p in enumerate(meta_a.columns)}
    chosen.sort(key=lambda c: order[c[0].name])
    return [
        _comparison(
            f"c{i + 1}",
            pa.name,
            pb.name,
            t,
            day_first=(_looks_day_first(sample_a[pa.name]), _looks_day_first(sample_b[pb.name])),
        )
        for i, (pa, pb, t) in enumerate(chosen)
    ]


def suggest_blocking(
    storage: Storage, dataset_a: str, dataset_b: str, comparisons: list[Comparison]
) -> Blocking:
    rows_a = storage.dataset_meta(dataset_a).rows
    rows_b = storage.dataset_meta(dataset_b).rows
    if rows_a * rows_b <= _FULL_COMPARE_LIMIT or not comparisons:
        return Blocking()

    meta_a = {c.name: c for c in storage.dataset_meta(dataset_a).columns}
    texts = [c for c in comparisons if c.type == ColumnType.TEXT]
    if texts:
        best = max(texts, key=lambda c: (c.weight, meta_a[c.column_a].distinct))
        return Blocking(rules=[TfidfBlockingRule(comparison_id=best.id)])
    best = max(comparisons, key=lambda c: meta_a[c.column_a].distinct)
    return Blocking(
        rules=[
            ExactBlockingRule(
                parts=[BlockingKeyPart(comparison_id=best.id, transform=KeyTransform.EXACT)]
            )
        ]
    )

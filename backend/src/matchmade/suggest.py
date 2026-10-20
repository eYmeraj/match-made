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

# Rows used to detect date formats; value overlap always uses whole columns, because in
# large files the first rows of A and B rarely describe the same entities.
_SAMPLE_ROWS = 20_000
_MIN_PAIR_SCORE = 0.35
_MIN_DISTINCT_FOR_OVERLAP = 3
# Columns of small integers (codes, counts) share values by chance.
_MIN_DISTINCT_NUMBERS_FOR_OVERLAP = 50
# Numbers or dates that mostly agree, but not exactly, get a tolerance.
_EXACT_AGREEMENT = 0.9
_NUMBER_TOLERANCE = 0.05
_DATE_TOLERANCE_DAYS = 31
_FUZZY_PROBES = 100
_FUZZY_MATCH = 85
_OVERLAP_TEXT_CLEANING = TextCleaning(remove_legal_suffixes=True)
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


class _Uniques:
    """Distinct cleaned values per whole column, computed once per suggestion."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame
        self._cache: dict[tuple[str, ColumnType], pl.Series] = {}

    def get(self, column: str, column_type: ColumnType) -> pl.Series:
        key = (column, column_type)
        if key not in self._cache:
            values = self._frame.get_column(column).drop_nulls().unique()
            self._cache[key] = _normalised(values, column_type).drop_nulls().unique()
        return self._cache[key]


def _normalised(values: pl.Series, column_type: ColumnType) -> pl.Series:
    v = pl.col("v")
    match column_type:
        case ColumnType.TEXT:
            # Abbreviation expansion is left out: it triples the cost on millions of values,
            # and the fuzzy probe tolerates what it would fix.
            expr = clean(v, column_type, _OVERLAP_TEXT_CLEANING)
        case ColumnType.DATE:
            expr = clean(v, column_type, DateCleaning(day_first=_looks_day_first(values)))
        case _:
            expr = clean(v, column_type, DEFAULT_CLEANING[column_type]())
    return pl.DataFrame({"v": values}).select(expr).to_series()


def _value_overlap(ua: pl.Series, ub: pl.Series, column_type: ColumnType) -> float:
    """Share of A's distinct values that also occur (fuzzily, for text) in B."""
    if column_type == ColumnType.BOOLEAN:
        return 0.0  # two possible values overlap trivially
    min_distinct = (
        _MIN_DISTINCT_NUMBERS_FOR_OVERLAP
        if column_type == ColumnType.NUMBER
        else _MIN_DISTINCT_FOR_OVERLAP
    )
    if min(len(ua), len(ub)) < min_distinct:
        return 0.0
    exact = int(ua.is_in(ub.implode()).sum()) / len(ua)
    if column_type != ColumnType.TEXT or exact >= _MIN_PAIR_SCORE:
        return exact
    # Typos and formatting hide exact matches; probe a sample of A against all of B.
    probe = ua.sample(min(len(ua), _FUZZY_PROBES), seed=0).to_list()
    best = cdist(
        probe, ub.to_list(), scorer=fuzz.token_sort_ratio, score_cutoff=_FUZZY_MATCH, workers=-1
    ).max(axis=1)
    return max(exact, float((best >= _FUZZY_MATCH).mean()))


def _looks_day_first(values: pl.Series) -> bool:
    """True when some value can only be read as day/month (first part above 12)."""
    return bool(values.drop_nulls().str.contains(r"^\s*(1[3-9]|2\d|3[01])[/.-]\d{1,2}[/.-]").any())


def _comparison(
    cid: str,
    column_a: str,
    column_b: str,
    column_type: ColumnType,
    overlap: float,
    day_first: tuple[bool, bool] = (False, False),
) -> Comparison:
    approximate = overlap < _EXACT_AGREEMENT
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
            return NumberComparison(
                id=cid,
                column_a=column_a,
                column_b=column_b,
                tolerance=_NUMBER_TOLERANCE if approximate else 0.0,
                tolerance_mode="relative" if approximate else "absolute",
            )
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
                tolerance_days=_DATE_TOLERANCE_DAYS if approximate else 0,
            )
        case ColumnType.BOOLEAN:
            return BooleanComparison(id=cid, column_a=column_a, column_b=column_b)
        case ColumnType.CATEGORICAL:
            return CategoricalComparison(id=cid, column_a=column_a, column_b=column_b)


def suggest_comparisons(storage: Storage, dataset_a: str, dataset_b: str) -> list[Comparison]:
    meta_a, meta_b = storage.dataset_meta(dataset_a), storage.dataset_meta(dataset_b)
    frame_a, frame_b = storage.read_dataset(dataset_a), storage.read_dataset(dataset_b)
    sample_a, sample_b = frame_a.head(_SAMPLE_ROWS), frame_b.head(_SAMPLE_ROWS)
    uniques_a, uniques_b = _Uniques(frame_a), _Uniques(frame_b)

    scored: list[tuple[float, float, ColumnProfile, ColumnProfile, ColumnType]] = []
    for pa in meta_a.columns:
        for pb in meta_b.columns:
            column_type = _compatible(pa.inferred_type, pb.inferred_type)
            if column_type is None:
                continue
            name = _name_similarity(pa.name, pb.name)
            overlap = _value_overlap(
                uniques_a.get(pa.name, column_type),
                uniques_b.get(pb.name, column_type),
                column_type,
            )
            # Either a similar name or shared values is evidence on its own.
            score = max(name, overlap)
            if score >= _MIN_PAIR_SCORE:
                scored.append((score, overlap, pa, pb, column_type))

    used_a: set[str] = set()
    used_b: set[str] = set()
    chosen: list[tuple[float, ColumnProfile, ColumnProfile, ColumnType]] = []
    for _, overlap, pa, pb, column_type in sorted(scored, key=lambda s: -s[0]):
        if pa.name not in used_a and pb.name not in used_b:
            used_a.add(pa.name)
            used_b.add(pb.name)
            chosen.append((overlap, pa, pb, column_type))
    order = {p.name: i for i, p in enumerate(meta_a.columns)}
    chosen.sort(key=lambda c: order[c[1].name])
    return [
        _comparison(
            f"c{i + 1}",
            pa.name,
            pb.name,
            t,
            overlap,
            day_first=(_looks_day_first(sample_a[pa.name]), _looks_day_first(sample_b[pb.name])),
        )
        for i, (overlap, pa, pb, t) in enumerate(chosen)
    ]


def _is_exact(comparison: Comparison) -> bool:
    if isinstance(comparison, NumberComparison):
        return comparison.tolerance == 0
    if isinstance(comparison, DateComparison):
        return comparison.tolerance_days == 0
    return True


def suggest_key_rule(
    comparisons: list[Comparison], distinct: dict[str, int]
) -> ExactBlockingRule | None:
    """Block on the most selective comparison an exact key cannot break."""
    exact = [c for c in comparisons if _is_exact(c) and c.type != ColumnType.BOOLEAN]
    if exact:
        best = max(exact, key=lambda c: distinct.get(c.column_a, 0))
        return ExactBlockingRule(
            parts=[BlockingKeyPart(comparison_id=best.id, transform=KeyTransform.EXACT)]
        )
    # Dates within a tolerance mostly still share their year.
    dates = [c for c in comparisons if c.type == ColumnType.DATE]
    if dates:
        best = max(dates, key=lambda c: distinct.get(c.column_a, 0))
        return ExactBlockingRule(
            parts=[BlockingKeyPart(comparison_id=best.id, transform=KeyTransform.YEAR)]
        )
    return None


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
    rule = suggest_key_rule(comparisons, {name: p.distinct for name, p in meta_a.items()})
    return Blocking(rules=[rule] if rule else [])

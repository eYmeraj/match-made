import datetime as dt

import polars as pl
import pytest

from matchmade.schemas import DateComparison, NumberComparison, TextMetric
from matchmade.similarity import (
    date_similarity,
    exact_similarity,
    number_similarity,
    text_similarity,
)


def approx(values):
    return [None if v is None else pytest.approx(v, abs=1e-3) for v in values]


def test_text_metrics_are_normalised_and_handle_missing():
    a = pl.Series(["acme widgets", "acme", None, "abc"])
    b = pl.Series(["widgets acme", "acme", "x", "xyz"])
    assert text_similarity(a, b, TextMetric.TOKEN_SORT).to_list() == approx([1, 1, None, 0])
    ratio = text_similarity(a, b, TextMetric.RATIO).to_list()
    assert 0 < ratio[0] < 1
    jw = text_similarity(pl.Series(["martha"]), pl.Series(["marhta"]), TextMetric.JARO_WINKLER)
    assert jw.to_list() == approx([0.9611])


def test_text_similarity_on_empty_input():
    assert text_similarity(
        pl.Series([], dtype=pl.String), pl.Series([], dtype=pl.String), TextMetric.RATIO
    ).is_empty()


def test_number_absolute_and_relative_tolerance():
    a = pl.Series([10.0, 10.0, 10.0, None, 0.0])
    b = pl.Series([10.0, 11.0, 30.0, 1.0, 0.0])
    exact = NumberComparison(id="n", column_a="x", column_b="x")
    assert number_similarity(a, b, exact).to_list() == approx([1, 0, 0, None, 1])
    absolute = NumberComparison(id="n", column_a="x", column_b="x", tolerance=4)
    assert number_similarity(a, b, absolute).to_list() == approx([1, 0.75, 0, None, 1])
    relative = NumberComparison(
        id="n", column_a="x", column_b="x", tolerance=0.5, tolerance_mode="relative"
    )
    # |10-11|/11 = 0.0909 -> 1 - 0.0909/0.5
    assert number_similarity(a, b, relative).to_list() == approx([1, 0.8182, 0, None, 1])


def test_date_tolerance_in_days():
    a = pl.Series([dt.date(2020, 1, 1)] * 3)
    b = pl.Series([dt.date(2020, 1, 1), dt.date(2020, 1, 6), dt.date(2019, 12, 1)])
    comp = DateComparison(id="d", column_a="x", column_b="x", tolerance_days=10)
    assert date_similarity(a, b, comp).to_list() == approx([1, 0.5, 0])


def test_exact_similarity():
    assert exact_similarity(
        pl.Series([True, True, None]), pl.Series([True, False, True])
    ).to_list() == approx([1, 0, None])

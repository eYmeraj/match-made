import random

import polars as pl
import pytest

from matchmade.blocking import ROW
from matchmade.matching import _greedy, _ranked, assign, best_per_a, one_to_one, score_pairs
from matchmade.schemas import (
    BooleanComparison,
    Cardinality,
    MissingPolicy,
    NumberComparison,
    TextComparison,
)


def cands(rows):
    return pl.DataFrame(
        rows, schema={"a": pl.UInt32, "b": pl.UInt32, "score": pl.Float32}, orient="row"
    )


def test_score_is_weighted_mean_with_missing_policy():
    comps = [
        TextComparison(id="name", column_a="n", column_b="n", weight=3),
        NumberComparison(id="age", column_a="x", column_b="x"),
    ]
    a = pl.DataFrame(
        {ROW: pl.Series([0, 1], dtype=pl.UInt32), "c_name": ["acme", "acme"], "c_age": [30.0, None]}
    )
    b = pl.DataFrame({ROW: pl.Series([0], dtype=pl.UInt32), "c_name": ["acme"], "c_age": [31.0]})
    pairs = pl.DataFrame({"a": [0, 1], "b": [0, 0]}, schema={"a": pl.UInt32, "b": pl.UInt32})

    ignore = score_pairs(pairs, a, b, comps, MissingPolicy.IGNORE)
    # Row 1 has no age: ignored, only the name counts; as a mismatch it costs 1/4.
    assert ignore.get_column("score").to_list() == pytest.approx([75.0, 100.0], abs=0.01)
    mismatch = score_pairs(pairs, a, b, comps, MissingPolicy.MISMATCH)
    assert mismatch.get_column("score").to_list() == pytest.approx([75.0, 75.0], abs=0.01)
    assert ignore.columns == ["a", "b", "s_name", "s_age", "score"]


def test_pairs_with_nothing_to_compare_are_dropped():
    comps = [BooleanComparison(id="f", column_a="f", column_b="f")]
    a = pl.DataFrame(
        {ROW: pl.Series([0], dtype=pl.UInt32), "c_f": pl.Series([None], dtype=pl.Boolean)}
    )
    b = pl.DataFrame({ROW: pl.Series([0], dtype=pl.UInt32), "c_f": [True]})
    pairs = pl.DataFrame({"a": [0], "b": [0]}, schema={"a": pl.UInt32, "b": pl.UInt32})
    assert score_pairs(pairs, a, b, comps, MissingPolicy.IGNORE).is_empty()
    assert score_pairs(pairs, a, b, comps, MissingPolicy.MISMATCH).get_column(
        "score"
    ).to_list() == [0.0]


def test_cardinalities():
    c = cands([(0, 0, 90), (0, 1, 95), (1, 1, 92), (2, 1, 85), (2, 2, 70)])
    assert sorted(one_to_one(c).select("a", "b").iter_rows()) == [(0, 1), (2, 2)]
    assert sorted(best_per_a(c).select("a", "b").iter_rows()) == [(0, 1), (1, 1), (2, 1)]
    assert assign(c, 80, Cardinality.ALL).height == 4


def test_one_to_one_respects_threshold_before_pairing():
    c = cands([(0, 1, 95), (2, 1, 85), (2, 2, 70)])
    # 2-2 falls below the threshold, so row 2 stays unmatched once B row 1 is taken.
    assert sorted(assign(c, 80, Cardinality.ONE_TO_ONE).select("a", "b").iter_rows()) == [(0, 1)]


@pytest.mark.parametrize("seed", range(20))
def test_mutual_best_rounds_equal_sequential_greedy(seed):
    rng = random.Random(seed)
    rows = {(rng.randrange(30), rng.randrange(30)) for _ in range(200)}
    c = cands([(a, b, rng.choice([50, 60, 70, 80, 90, 99.5])) for a, b in rows])
    expected = sorted(_greedy(_ranked(c)).select("a", "b").iter_rows())
    assert sorted(one_to_one(c).select("a", "b").iter_rows()) == expected
    assert sorted(one_to_one(c, max_rounds=1).select("a", "b").iter_rows()) == expected


def test_empty_candidates():
    empty = cands([])
    for cardinality in Cardinality:
        assert assign(empty, 50, cardinality).is_empty()

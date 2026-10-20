import datetime as dt

import polars as pl
import pytest

from matchmade.blocking import ROW, BlockingError, candidate_pairs, derive_key, estimate
from matchmade.schemas import (
    Blocking,
    BlockingKeyPart,
    ExactBlockingRule,
    KeyTransform,
    TfidfBlockingRule,
)


def frame(**columns):
    n = len(next(iter(columns.values())))
    return pl.DataFrame(
        {ROW: pl.Series(range(n), dtype=pl.UInt32), **{f"c_{k}": v for k, v in columns.items()}}
    )


def pairs_of(result):
    return sorted(result.pairs.iter_rows())


@pytest.mark.parametrize(
    ("values", "part", "expected"),
    [
        (["acme widgets", None], BlockingKeyPart(comparison_id="x"), ["acme widgets", None]),
        (
            ["acme widgets"],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.PREFIX, param=3),
            ["acm"],
        ),
        (
            ["acme widgets"],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.FIRST_TOKEN),
            ["acme"],
        ),
        (
            ["widgets acme"],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.SORTED_TOKENS),
            ["acme widgets"],
        ),
        (
            ["smith", "smyth"],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.PHONETIC),
            ["SM0", "SM0"],
        ),
        (
            [dt.date(2020, 5, 3)],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.YEAR),
            ["2020"],
        ),
        (
            [dt.date(2020, 5, 3)],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.YEAR_MONTH),
            ["2020-05"],
        ),
        (
            [123.0, 99.0, -1.0],
            BlockingKeyPart(comparison_id="x", transform=KeyTransform.ROUND, param=100),
            ["1", "0", "-1"],
        ),
    ],
)
def test_derive_key(values, part, expected):
    assert derive_key(pl.Series(values), part).to_list() == expected


def test_key_rules_are_and_within_rule_and_or_across_rules():
    a = frame(name=["acme", "beta", "gamma", None], year=[dt.date(2020, 1, 1)] * 4)
    b = frame(
        name=["acme", "beta", "delta", "gamma"],
        year=[dt.date(2020, 1, 1), dt.date(2021, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
    )
    both = ExactBlockingRule(
        parts=[
            BlockingKeyPart(comparison_id="name"),
            BlockingKeyPart(comparison_id="year", transform=KeyTransform.YEAR),
        ]
    )
    result = candidate_pairs(a, b, Blocking(rules=[both]), max_candidates=100)
    assert pairs_of(result) == [(0, 0), (2, 3)]

    by_prefix = ExactBlockingRule(
        parts=[BlockingKeyPart(comparison_id="name", transform=KeyTransform.PREFIX, param=1)]
    )
    result = candidate_pairs(a, b, Blocking(rules=[both, by_prefix]), max_candidates=100)
    assert pairs_of(result) == [(0, 0), (1, 1), (2, 3)]
    assert [s.pairs for s in result.rules] == [2, 3]


def test_oversized_blocks_are_skipped_and_reported():
    a = frame(k=["x"] * 3 + ["y"])
    b = frame(k=["x"] * 3 + ["y"])
    rule = ExactBlockingRule(parts=[BlockingKeyPart(comparison_id="k")])
    result = candidate_pairs(a, b, Blocking(rules=[rule], max_block_pairs=4), max_candidates=100)
    assert pairs_of(result) == [(3, 3)]
    assert result.rules[0].skipped_blocks == 1
    assert result.rules[0].skipped_pairs == 9
    assert estimate(a, b, Blocking(rules=[rule], max_block_pairs=4))[0].pairs == 1


def test_no_rules_compares_everything_within_the_limit():
    a, b = frame(k=["x", "y"]), frame(k=["z", "w", "v"])
    assert len(pairs_of(candidate_pairs(a, b, Blocking(), max_candidates=6))) == 6
    with pytest.raises(BlockingError, match="Add a blocking rule"):
        candidate_pairs(a, b, Blocking(), max_candidates=5)


def test_rule_over_the_candidate_limit_fails_early():
    a, b = frame(k=["x"] * 10), frame(k=["x"] * 10)
    rule = ExactBlockingRule(parts=[BlockingKeyPart(comparison_id="k")])
    with pytest.raises(BlockingError, match="Blocking rule 1 would create 100 pairs"):
        candidate_pairs(a, b, Blocking(rules=[rule]), max_candidates=50)


def test_tfidf_finds_fuzzy_neighbours_and_expands_duplicate_values():
    a = frame(name=["acme widgets", "globex corporation", "initech", None])
    b = frame(name=["acme widget", "globex corp", "initech", "initech", "umbrella"])
    rule = TfidfBlockingRule(comparison_id="name", top_k=1, min_similarity=0.3)
    result = candidate_pairs(a, b, Blocking(rules=[rule]), max_candidates=100)
    assert pairs_of(result) == [(0, 0), (1, 1), (2, 2), (2, 3)]


def test_tfidf_with_no_values():
    a = frame(name=pl.Series([None], dtype=pl.String))
    b = frame(name=["x"])
    rule = TfidfBlockingRule(comparison_id="name")
    assert candidate_pairs(a, b, Blocking(rules=[rule]), max_candidates=100).pairs.is_empty()

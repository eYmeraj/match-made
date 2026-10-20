import datetime as dt
import random

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from matchmade import blocking
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


def test_rarest_features_keeps_the_highest_idf_features_per_row_as_binary():
    m = sp.csr_matrix(np.array([[0.5, 0.2, 0.9, 0.1], [0.3, 0.0, 0.0, 0.4]], dtype=np.float32))
    idf = np.array([1.0, 4.0, 2.0, 3.0])
    rare = blocking._rarest_features(m, idf, 2).toarray()
    assert rare.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]


def _random_names(rng: random.Random, n: int) -> list[str]:
    syllables = ["ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa"]
    return ["".join(rng.choices(syllables, k=rng.randint(2, 4))) for _ in range(n)]


def _exact_top_k(a_values, b_values, top_k, min_similarity):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True)
    vectorizer.fit(sorted(set(a_values) | set(b_values)))
    ma = sp.csr_matrix(vectorizer.transform(a_values))
    mb = sp.csr_matrix(vectorizer.transform(b_values))
    cosine = (ma @ mb.T).toarray()
    expected = set()
    for i, row in enumerate(cosine):
        for j in np.argsort(-row, kind="stable")[:top_k]:
            if row[j] >= min_similarity:
                expected.add((i, int(j)))
    return expected


def test_tfidf_matches_exact_top_k_when_pruning_is_lossless():
    # Names this short have at most 10 trigrams, so no feature is pruned and the
    # candidate search must agree with the exact cosine top-k (up to ties).
    rng = random.Random(7)
    a_values = list(dict.fromkeys(_random_names(rng, 300)))
    b_values = list(dict.fromkeys(_random_names(rng, 300)))
    rule = TfidfBlockingRule(comparison_id="name", top_k=3, min_similarity=0.4)
    result = candidate_pairs(
        frame(name=a_values), frame(name=b_values), Blocking(rules=[rule]), max_candidates=10**6
    )
    found = set(result.pairs.iter_rows())
    expected = _exact_top_k(a_values, b_values, top_k=3, min_similarity=0.4)
    assert len(found ^ expected) <= 0.02 * len(expected)


def test_tfidf_results_do_not_depend_on_chunking(monkeypatch):
    rng = random.Random(11)
    a = frame(name=_random_names(rng, 400))
    b = frame(name=_random_names(rng, 500))
    blocks = Blocking(rules=[TfidfBlockingRule(comparison_id="name", top_k=5, min_similarity=0.3)])
    whole = candidate_pairs(a, b, blocks, max_candidates=10**6)
    monkeypatch.setattr(blocking, "_CHUNK_A", 37)
    monkeypatch.setattr(blocking, "_CHUNK_B", 53)
    progress: list[float] = []
    chunked = candidate_pairs(a, b, blocks, max_candidates=10**6, on_progress=progress.append)
    # Ties between equally rare candidates can resolve differently across chunks.
    assert (
        len(set(whole.pairs.iter_rows()) ^ set(chunked.pairs.iter_rows()))
        <= 0.02 * whole.pairs.height
    )
    assert progress == sorted(progress)
    assert len(progress) > 5
    assert progress[-1] == 1.0


def test_tfidf_skips_value_pairs_that_multiply_into_too_many_rows():
    # Placeholders like "N/A" repeat on both sides; 3,000 x 3,000 rows would be 9M pairs.
    names = [f"company {i}" for i in range(500)] + ["unknown"] * 3000
    a, b = frame(name=names), frame(name=names)
    blocks = Blocking(
        rules=[TfidfBlockingRule(comparison_id="name", top_k=1)], max_block_pairs=1_000_000
    )
    result = candidate_pairs(a, b, blocks, max_candidates=10**6)
    assert result.pairs.height == 500
    (stats,) = result.rules
    assert (stats.skipped_blocks, stats.skipped_pairs) == (1, 9_000_000)
    # The estimate warns about the same placeholder before anything runs.
    (estimated,) = estimate(a, b, blocks)
    assert (estimated.skipped_blocks, estimated.skipped_pairs) == (1, 9_000_000)

import polars as pl

from matchmade.schemas import ColumnType
from matchmade.suggest import _value_overlap


def test_text_overlap_finds_values_that_only_match_fuzzily():
    names = [f"company number {i} holdings" for i in range(200)]
    typos = [n.replace("company", "compnay") for n in names]
    unrelated = [f"{i} main street" for i in range(200)]
    a, b = pl.Series(names), pl.Series(typos)
    assert _value_overlap(a, b, ColumnType.TEXT) == 1.0
    assert _value_overlap(a, pl.Series(unrelated), ColumnType.TEXT) == 0.0

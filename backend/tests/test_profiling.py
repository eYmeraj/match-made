import polars as pl
import pytest

from matchmade.profiling import infer_type, profile_frame
from matchmade.schemas import ColumnType


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["yes", "no", "yes"], ColumnType.BOOLEAN),
        (["2020-01-01", "31/12/2020", "3 Feb 2021"], ColumnType.DATE),
        (["20200101", "20200102", "20200103"], ColumnType.NUMBER),
        (["$1,200", "3.5", "-2"], ColumnType.NUMBER),
        (["1.5e3", "2E-2", "7"], ColumnType.NUMBER),
        (["1,200 EUR", "USD 30", "15"], ColumnType.NUMBER),
        (["A0000001", "A0000002", "A0000003"], ColumnType.TEXT),
        (["12a", "13b", "14c"], ColumnType.TEXT),
        (["1", "2", "3", "4"], ColumnType.NUMBER),
        (["Acme Inc", "Globex", "Initech"], ColumnType.TEXT),
        (["12 Main St", "14 High Rd"], ColumnType.TEXT),
        (["red", "blue"] * 100, ColumnType.CATEGORICAL),
        ([], ColumnType.TEXT),
    ],
)
def test_infer_type(values, expected):
    assert infer_type(pl.Series(values, dtype=pl.String)) == expected


def test_profile_frame():
    frame = pl.DataFrame({"a": ["x", None, "y", "x"]})
    (profile,) = profile_frame(frame)
    assert profile.name == "a"
    assert profile.distinct == 2

import datetime as dt

import polars as pl
import pytest

from matchmade.cleaning import clean
from matchmade.schemas import (
    BooleanCleaning,
    CategoricalCleaning,
    ColumnType,
    DateCleaning,
    NumberCleaning,
    Replacement,
    TextCleaning,
)


def run(values, column_type, opts):
    return (
        pl.DataFrame({"v": values}, schema={"v": pl.String})
        .select(clean(pl.col("v"), column_type, opts))
        .to_series()
        .to_list()
    )


def test_text_defaults_normalise_case_accents_punctuation_and_whitespace():
    assert run(["  Café   Ünion, Inc. ", "", None, "!!!"], ColumnType.TEXT, TextCleaning()) == [
        "cafe union inc",
        None,
        None,
        None,
    ]


def test_text_company_preset_removes_suffixes_and_expands_abbreviations():
    opts = TextCleaning(remove_legal_suffixes=True, expand_abbreviations=True)
    assert run(
        ["ACME & Co.", "A.B.C. Int'l Ltd", "Smith Bros., L.L.C.", "Corp of Cos"],
        ColumnType.TEXT,
        opts,
    ) == ["acme and", "abc international", "smith brothers", "of cos"]


def test_text_suffix_removal_keeps_words_containing_suffixes():
    opts = TextCleaning(remove_legal_suffixes=True)
    assert run(["Incognito Coffee Company"], ColumnType.TEXT, opts) == ["incognito coffee"]


def test_text_stopwords_digits_and_replacements():
    opts = TextCleaning(
        remove_stopwords=True,
        remove_digits=True,
        replacements=[
            Replacement(find="saint", replace="st"),
            Replacement(find=r"\bx+\b", replace="", regex=True),
        ],
    )
    assert run(["The Bank of Saint Louis 2 xx"], ColumnType.TEXT, opts) == ["bank st louis"]


def test_text_can_keep_case_and_punctuation():
    opts = TextCleaning(lowercase=False, strip_accents=False, remove_punctuation=False)
    assert run(["Café-Bar"], ColumnType.TEXT, opts) == ["Café-Bar"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.50", 1234.5),
        ("(12)", -12.0),
        ("-3", -3.0),
        ("USD -7", -7.0),
        ("1e3", 1000.0),
        ("  42 kg ", 42.0),
        ("abc", None),
        (None, None),
        ("inf", None),
    ],
)
def test_number_parsing(raw, expected):
    assert run([raw], ColumnType.NUMBER, NumberCleaning()) == [expected]


def test_number_decimal_comma_and_percent():
    opts = NumberCleaning(decimal_separator=",", percent_as_fraction=True)
    assert run(["1.234,5", "1.234", "12,5%"], ColumnType.NUMBER, opts) == [1234.5, 1234.0, 0.125]


@pytest.mark.parametrize(
    ("raw", "day_first", "expected"),
    [
        ("2020-01-31", False, dt.date(2020, 1, 31)),
        ("2020-01-31T10:00:00", False, dt.date(2020, 1, 31)),
        ("2020-01-31 00:00:00", False, dt.date(2020, 1, 31)),
        ("01/02/2020", False, dt.date(2020, 1, 2)),
        ("01/02/2020", True, dt.date(2020, 2, 1)),
        ("13/02/2020", False, dt.date(2020, 2, 13)),
        ("3 Feb 2021", False, dt.date(2021, 2, 3)),
        ("feb 3, 2021", False, dt.date(2021, 2, 3)),
        ("31.12.1999", False, dt.date(1999, 12, 31)),
        ("20200105", False, dt.date(2020, 1, 5)),
        ("not a date", False, None),
    ],
)
def test_date_parsing(raw, day_first, expected):
    assert run([raw], ColumnType.DATE, DateCleaning(day_first=day_first)) == [expected]


def test_date_explicit_format():
    assert run(["2020|31|01", "2020-01-31"], ColumnType.DATE, DateCleaning(format="%Y|%d|%m")) == [
        dt.date(2020, 1, 31),
        None,
    ]


def test_boolean_values_are_configurable():
    assert run(["Yes", " n ", "TRUE", "maybe", None], ColumnType.BOOLEAN, BooleanCleaning()) == [
        True,
        False,
        True,
        None,
        None,
    ]
    opts = BooleanCleaning(true_values=["active"], false_values=["inactive"])
    assert run(["Active", "yes"], ColumnType.BOOLEAN, opts) == [True, None]


def test_categorical():
    assert run([" Île-de-France ", "  "], ColumnType.CATEGORICAL, CategoricalCleaning()) == [
        "ile-de-france",
        None,
    ]


def test_mismatched_options_are_rejected():
    with pytest.raises(ValueError, match="do not match"):
        clean(pl.col("v"), ColumnType.NUMBER, TextCleaning())

"""Type-aware cleaning, expressed as Polars expressions over raw string columns.

Every dataset column is stored as strings. Cleaning turns a raw column into a typed,
normalised column (text -> String, number -> Float64, date -> Date, boolean -> Boolean)
where values that are empty or cannot be parsed become null.
"""

from __future__ import annotations

import re

import polars as pl

from matchmade.schemas import (
    BooleanCleaning,
    CategoricalCleaning,
    Cleaning,
    ColumnType,
    DateCleaning,
    NumberCleaning,
    TextCleaning,
)

LEGAL_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "company", "co", "limited", "ltd",
    "llc", "llp", "lp", "plc", "gmbh", "mbh", "ag", "kg", "sa", "sas", "sarl", "srl",
    "spa", "bv", "nv", "oy", "oyj", "ab", "as", "asa", "aps", "kk", "pty", "pvt",
    "pte", "bhd", "sdn", "sl", "se", "lda", "ltda", "holdings", "group",
]  # fmt: skip

STOPWORDS = [
    "the", "of", "and", "a", "an", "for", "in", "on", "at", "to", "de", "la", "le",
    "el", "les", "der", "die", "das", "und", "y", "et",
]  # fmt: skip

ABBREVIATIONS = {
    "intl": "international",
    "int l": "international",
    "natl": "national",
    "assn": "association",
    "assoc": "associates",
    "bros": "brothers",
    "dept": "department",
    "univ": "university",
    "inst": "institute",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "mgmt": "management",
    "hldgs": "holdings",
    "grp": "group",
    "sys": "systems",
    "ctr": "center",
    "centre": "center",
    "hosp": "hospital",
    "ave": "avenue",
    "blvd": "boulevard",
    "rd": "road",
    "mt": "mount",
}

_DATE_FORMATS_ISO = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"]
_DATE_FORMATS_DAY_FIRST = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]
_DATE_FORMATS_MONTH_FIRST = ["%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y"]
_DATE_FORMATS_NAMED = [
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%b-%Y", "%d-%b-%y", "%d %b %y",
]  # fmt: skip


def _words_pattern(words: list[str]) -> str:
    # Longest first so "inc" does not shadow "incorporated" in the alternation.
    ordered = sorted(words, key=len, reverse=True)
    return r"(?i)\b(?:" + "|".join(re.escape(w) for w in ordered) + r")\b"


def _collapse_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()


def _empty_to_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.str.len_chars() > 0).then(expr).otherwise(None)


def _strip_accents(expr: pl.Expr) -> pl.Expr:
    return expr.str.normalize("NFKD").str.replace_all(r"\p{M}", "")


def clean_text(expr: pl.Expr, opts: TextCleaning) -> pl.Expr:
    e = expr.cast(pl.String)
    if opts.lowercase:
        e = e.str.to_lowercase()
    if opts.strip_accents:
        e = _strip_accents(e)
    if opts.remove_punctuation:
        # Collapse dotted initialisms ("L.L.C." -> "LLC") before dropping punctuation.
        e = e.str.replace_all(r"\b(\p{L})\.", "$1")
    if opts.expand_abbreviations:
        e = e.str.replace_all("&", " and ", literal=True)
    if opts.remove_punctuation:
        e = e.str.replace_all(r"[^\p{L}\p{N}\s]", " ")
    if opts.expand_abbreviations:
        e = _collapse_whitespace(e)
        for short, full in ABBREVIATIONS.items():
            e = e.str.replace_all(rf"(?i)\b{re.escape(short)}\b", full)
    if opts.remove_legal_suffixes:
        e = e.str.replace_all(_words_pattern(LEGAL_SUFFIXES), " ")
    if opts.remove_stopwords:
        e = e.str.replace_all(_words_pattern(STOPWORDS), " ")
    if opts.remove_digits:
        e = e.str.replace_all(r"\p{N}", " ")
    for r in opts.replacements:
        e = e.str.replace_all(r.find, r.replace, literal=not r.regex)
    return _empty_to_null(_collapse_whitespace(e))


def clean_categorical(expr: pl.Expr, opts: CategoricalCleaning) -> pl.Expr:
    e = expr.cast(pl.String)
    if opts.lowercase:
        e = e.str.to_lowercase()
    if opts.strip_accents:
        e = _strip_accents(e)
    return _empty_to_null(_collapse_whitespace(e))


def clean_number(expr: pl.Expr, opts: NumberCleaning) -> pl.Expr:
    raw = expr.cast(pl.String).str.strip_chars()

    negative = raw.str.contains(r"^\(.*\)$") | raw.str.contains(r"^-|-$|^\D+-")
    digits = raw.str.replace_all(r"[^0-9.,]", "")
    if opts.decimal_separator == ",":
        digits = digits.str.replace_all(".", "", literal=True).str.replace_all(
            ",", ".", literal=True
        )
    else:
        digits = digits.str.replace_all(",", "", literal=True)
    parsed = digits.cast(pl.Float64, strict=False)
    parsed = pl.when(negative).then(-parsed).otherwise(parsed)

    # A direct cast handles scientific notation, but would misread "1.234" as 1.234
    # when "." is the thousands separator.
    direct = raw.cast(pl.Float64, strict=False)
    value = pl.coalesce(direct, parsed) if opts.decimal_separator == "." else parsed
    if opts.percent_as_fraction:
        value = pl.when(raw.str.contains("%", literal=True)).then(value / 100).otherwise(value)
    return pl.when(value.is_finite()).then(value).otherwise(None)


def date_formats(opts: DateCleaning) -> list[str]:
    if opts.format:
        return [opts.format]
    ambiguous = (
        _DATE_FORMATS_DAY_FIRST + _DATE_FORMATS_MONTH_FIRST
        if opts.day_first
        else _DATE_FORMATS_MONTH_FIRST + _DATE_FORMATS_DAY_FIRST
    )
    return _DATE_FORMATS_ISO + ambiguous + _DATE_FORMATS_NAMED


def clean_date(expr: pl.Expr, opts: DateCleaning) -> pl.Expr:
    raw = _collapse_whitespace(expr.cast(pl.String))
    if opts.format:
        return raw.str.strptime(pl.Date, opts.format, strict=False)
    # Datetimes such as "2020-01-31T10:00:00" or Excel's "2020-01-31 00:00:00".
    iso_prefix = raw.str.extract(r"^(\d{4}-\d{2}-\d{2})(?:[T ].*)?$", 1)
    candidates = [iso_prefix.str.strptime(pl.Date, "%Y-%m-%d", strict=False)]
    candidates += [raw.str.strptime(pl.Date, f, strict=False) for f in date_formats(opts)]
    return pl.coalesce(candidates)


def clean_boolean(expr: pl.Expr, opts: BooleanCleaning) -> pl.Expr:
    e = expr.cast(pl.String).str.strip_chars().str.to_lowercase()
    true_values = [v.strip().lower() for v in opts.true_values]
    false_values = [v.strip().lower() for v in opts.false_values]
    return (
        pl.when(e.is_in(true_values))
        .then(pl.lit(True))
        .when(e.is_in(false_values))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )


def clean(expr: pl.Expr, column_type: ColumnType, opts: Cleaning) -> pl.Expr:
    match column_type, opts:
        case ColumnType.TEXT, TextCleaning():
            return clean_text(expr, opts)
        case ColumnType.CATEGORICAL, CategoricalCleaning():
            return clean_categorical(expr, opts)
        case ColumnType.NUMBER, NumberCleaning():
            return clean_number(expr, opts)
        case ColumnType.DATE, DateCleaning():
            return clean_date(expr, opts)
        case ColumnType.BOOLEAN, BooleanCleaning():
            return clean_boolean(expr, opts)
    raise ValueError(f"cleaning options do not match column type {column_type}")


DEFAULT_CLEANING: dict[ColumnType, type[Cleaning]] = {
    ColumnType.TEXT: TextCleaning,
    ColumnType.CATEGORICAL: CategoricalCleaning,
    ColumnType.NUMBER: NumberCleaning,
    ColumnType.DATE: DateCleaning,
    ColumnType.BOOLEAN: BooleanCleaning,
}

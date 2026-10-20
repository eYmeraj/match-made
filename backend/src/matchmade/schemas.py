"""Pydantic models shared by the API, the worker and the matching pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

import polars as pl
from pydantic import BaseModel, Field, model_validator


class ColumnType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class TextMetric(StrEnum):
    RATIO = "ratio"
    TOKEN_SORT = "token_sort"
    TOKEN_SET = "token_set"
    PARTIAL = "partial"
    WRATIO = "wratio"
    JARO_WINKLER = "jaro_winkler"


class Replacement(BaseModel):
    find: str = Field(min_length=1)
    replace: str = ""
    regex: bool = False

    @model_validator(mode="after")
    def _check_regex(self) -> Replacement:
        # Compile with Polars' own engine: it lacks some Python features (e.g. lookbehind).
        if self.regex:
            try:
                pl.select(pl.lit("").str.contains(self.find))
            except pl.exceptions.PolarsError as e:
                lines = str(e).splitlines()
                reason = next((x[7:] for x in lines if x.startswith("error: ")), "invalid syntax")
                raise ValueError(f"Invalid regular expression {self.find!r}: {reason}") from e
        return self


# --- Cleaning options, one model per column type -------------------------------------


class TextCleaning(BaseModel):
    lowercase: bool = True
    strip_accents: bool = True
    expand_abbreviations: bool = False
    remove_punctuation: bool = True
    remove_legal_suffixes: bool = False
    remove_stopwords: bool = False
    remove_digits: bool = False
    replacements: list[Replacement] = Field(default_factory=list)


class NumberCleaning(BaseModel):
    decimal_separator: Literal[".", ","] = "."
    percent_as_fraction: bool = False


class DateCleaning(BaseModel):
    # None means "try the common formats"; otherwise a strftime-style format string.
    format: str | None = None
    day_first: bool = False


class BooleanCleaning(BaseModel):
    true_values: list[str] = Field(default_factory=lambda: ["true", "t", "yes", "y", "1", "on"])
    false_values: list[str] = Field(default_factory=lambda: ["false", "f", "no", "n", "0", "off"])


class CategoricalCleaning(BaseModel):
    lowercase: bool = True
    strip_accents: bool = True


Cleaning = TextCleaning | NumberCleaning | DateCleaning | BooleanCleaning | CategoricalCleaning


# --- Comparisons: a column pair + type + cleaning + similarity ------------------------


class _ComparisonBase(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    column_a: str
    column_b: str
    weight: float = Field(default=1.0, gt=0, le=100)

    def cleaning_for(self, side: Literal["a", "b"]) -> Cleaning:
        cleaning = self.cleaning  # pyright: ignore[reportAttributeAccessIssue]
        if side == "b":
            return getattr(self, "cleaning_b", None) or cleaning
        return cleaning

    def column_for(self, side: Literal["a", "b"]) -> str:
        return self.column_a if side == "a" else self.column_b


class TextComparison(_ComparisonBase):
    type: Literal[ColumnType.TEXT] = ColumnType.TEXT
    cleaning: TextCleaning = Field(default_factory=TextCleaning)
    metric: TextMetric = TextMetric.TOKEN_SORT


class NumberComparison(_ComparisonBase):
    type: Literal[ColumnType.NUMBER] = ColumnType.NUMBER
    cleaning: NumberCleaning = Field(default_factory=NumberCleaning)
    # Files from different sources can format numbers differently; None = same as A.
    cleaning_b: NumberCleaning | None = None
    # Similarity decays linearly from 1 (equal) to 0 (difference >= tolerance).
    tolerance: float = Field(default=0.0, ge=0)
    tolerance_mode: Literal["absolute", "relative"] = "absolute"


class DateComparison(_ComparisonBase):
    type: Literal[ColumnType.DATE] = ColumnType.DATE
    cleaning: DateCleaning = Field(default_factory=DateCleaning)
    # Files from different sources can format dates differently; None = same as A.
    cleaning_b: DateCleaning | None = None
    tolerance_days: int = Field(default=0, ge=0)


class BooleanComparison(_ComparisonBase):
    type: Literal[ColumnType.BOOLEAN] = ColumnType.BOOLEAN
    cleaning: BooleanCleaning = Field(default_factory=BooleanCleaning)


class CategoricalComparison(_ComparisonBase):
    type: Literal[ColumnType.CATEGORICAL] = ColumnType.CATEGORICAL
    cleaning: CategoricalCleaning = Field(default_factory=CategoricalCleaning)


Comparison = Annotated[
    TextComparison | NumberComparison | DateComparison | BooleanComparison | CategoricalComparison,
    Field(discriminator="type"),
]

# --- Blocking -------------------------------------------------------------------------


class KeyTransform(StrEnum):
    EXACT = "exact"
    PREFIX = "prefix"
    FIRST_TOKEN = "first_token"
    SORTED_TOKENS = "sorted_tokens"
    PHONETIC = "phonetic"
    YEAR = "year"
    YEAR_MONTH = "year_month"
    ROUND = "round"


class BlockingKeyPart(BaseModel):
    comparison_id: str
    transform: KeyTransform = KeyTransform.EXACT
    # PREFIX: number of characters. ROUND: bucket size.
    param: float | None = Field(default=None, gt=0)


class ExactBlockingRule(BaseModel):
    """Candidate pairs share the same derived key; parts are AND-ed together."""

    kind: Literal["key"] = "key"
    parts: list[BlockingKeyPart] = Field(min_length=1)


class TfidfBlockingRule(BaseModel):
    """Candidate pairs are the top-k nearest neighbours by character n-gram TF-IDF."""

    kind: Literal["tfidf"] = "tfidf"
    comparison_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0, lt=1)


BlockingRule = Annotated[ExactBlockingRule | TfidfBlockingRule, Field(discriminator="kind")]


class Blocking(BaseModel):
    # Rules are OR-ed: a pair is a candidate if any rule produces it.
    # With no rules every A row is compared with every B row (small inputs only).
    rules: list[BlockingRule] = Field(default_factory=list)
    max_block_pairs: int = Field(default=1_000_000, ge=1)


# --- Run configuration ----------------------------------------------------------------


class Cardinality(StrEnum):
    ONE_TO_ONE = "one_to_one"
    BEST_PER_A = "best_per_a"
    ALL = "all"


class MissingPolicy(StrEnum):
    # Skip comparisons where either side is missing and re-normalise the weights.
    IGNORE = "ignore"
    # Treat a missing value as a complete mismatch.
    MISMATCH = "mismatch"


class MatchConfig(BaseModel):
    dataset_a: str
    dataset_b: str
    comparisons: list[Comparison] = Field(min_length=1)
    blocking: Blocking = Field(default_factory=Blocking)
    threshold: float = Field(default=80, ge=0, le=100)
    cardinality: Cardinality = Cardinality.ONE_TO_ONE
    missing: MissingPolicy = MissingPolicy.IGNORE

    @model_validator(mode="after")
    def _check_references(self) -> MatchConfig:
        ids = [c.id for c in self.comparisons]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison ids must be unique")
        by_id = {c.id: c for c in self.comparisons}
        for rule in self.blocking.rules:
            refs = (
                [p.comparison_id for p in rule.parts]
                if isinstance(rule, ExactBlockingRule)
                else [rule.comparison_id]
            )
            for ref in refs:
                if ref not in by_id:
                    raise ValueError(f"blocking rule references unknown comparison '{ref}'")
            if isinstance(rule, TfidfBlockingRule) and by_id[rule.comparison_id].type not in (
                ColumnType.TEXT,
                ColumnType.CATEGORICAL,
            ):
                raise ValueError("TF-IDF blocking needs a text or categorical comparison")
            if isinstance(rule, ExactBlockingRule):
                for part in rule.parts:
                    _check_transform(by_id[part.comparison_id].type, part)
        return self


_TRANSFORMS_BY_TYPE: dict[ColumnType, set[KeyTransform]] = {
    ColumnType.TEXT: {
        KeyTransform.EXACT,
        KeyTransform.PREFIX,
        KeyTransform.FIRST_TOKEN,
        KeyTransform.SORTED_TOKENS,
        KeyTransform.PHONETIC,
    },
    ColumnType.CATEGORICAL: {
        KeyTransform.EXACT,
        KeyTransform.PREFIX,
        KeyTransform.FIRST_TOKEN,
        KeyTransform.PHONETIC,
    },
    ColumnType.NUMBER: {KeyTransform.EXACT, KeyTransform.ROUND},
    ColumnType.DATE: {KeyTransform.EXACT, KeyTransform.YEAR, KeyTransform.YEAR_MONTH},
    ColumnType.BOOLEAN: {KeyTransform.EXACT},
}


def allowed_transforms(column_type: ColumnType) -> set[KeyTransform]:
    return _TRANSFORMS_BY_TYPE[column_type]


def _check_transform(column_type: ColumnType, part: BlockingKeyPart) -> None:
    if part.transform not in _TRANSFORMS_BY_TYPE[column_type]:
        raise ValueError(
            f"blocking transform '{part.transform}' is not valid for {column_type} columns"
        )
    if part.transform in (KeyTransform.PREFIX, KeyTransform.ROUND) and part.param is None:
        raise ValueError(f"blocking transform '{part.transform}' needs a param")

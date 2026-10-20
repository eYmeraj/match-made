"""Reading uploaded CSV/TSV/Excel files into all-string Polars frames."""

from __future__ import annotations

import codecs
import csv
import shutil
from pathlib import Path
from typing import Literal

import fastexcel
import polars as pl
from charset_normalizer import from_bytes
from pydantic import BaseModel

FileFormat = Literal["csv", "xlsx"]

_SNIFF_BYTES = 1 << 20
_DELIMITERS = ",;\t|"


class ParseOptions(BaseModel):
    sheet: str | None = None
    delimiter: str | None = None
    encoding: str | None = None


class ParseInfo(BaseModel):
    format: FileFormat
    sheet: str | None = None
    sheets: list[str] = []
    delimiter: str | None = None
    encoding: str | None = None


class ReadError(ValueError):
    pass


def detect_format(filename: str) -> FileFormat:
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return "xlsx"
    if suffix in (".csv", ".tsv", ".txt", ""):
        return "csv"
    raise ReadError(f"Unsupported file type '{suffix}'. Use CSV, TSV or XLSX.")


def detect_encoding(sample: bytes) -> str:
    try:
        # A multi-byte character may be cut at the end of the sample.
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        match = from_bytes(sample).best()
        return match.encoding if match else "latin-1"


def detect_delimiter(text_sample: str, filename: str) -> str:
    lines = text_sample.splitlines()[:50]
    try:
        return csv.Sniffer().sniff("\n".join(lines), delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return "\t" if filename.lower().endswith(".tsv") else ","


def _to_utf8(path: Path, encoding: str) -> Path:
    if codecs.lookup(encoding).name == "utf-8":
        return path
    target = path.with_suffix(path.suffix + ".utf8")
    with (
        open(path, encoding=encoding, errors="replace", newline="") as src,
        open(target, "w", encoding="utf-8", newline="") as dst,
    ):
        shutil.copyfileobj(src, dst, length=_SNIFF_BYTES)
    return target


def _dedupe_headers(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for i, raw in enumerate(names):
        name = raw.strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 0)
        result.append(name)
    return result


def read_csv(path: Path, filename: str, options: ParseOptions) -> tuple[pl.DataFrame, ParseInfo]:
    with open(path, "rb") as f:
        sample = f.read(_SNIFF_BYTES)
    encoding = options.encoding or detect_encoding(sample)
    try:
        text_sample = sample.decode(encoding, errors="ignore")
    except LookupError as e:
        raise ReadError(f"Unknown encoding '{encoding}'") from e
    text_sample = text_sample.removeprefix("\ufeff")
    delimiter = options.delimiter or detect_delimiter(text_sample, filename)
    if len(delimiter) != 1:
        raise ReadError("The delimiter must be a single character")

    source = _to_utf8(path, encoding)
    try:
        frame = pl.read_csv(
            source,
            separator=delimiter,
            # Headers are read as the first data row so that duplicate or blank names
            # get readable replacements instead of Polars' "_duplicated_0" suffixes.
            has_header=False,
            infer_schema=False,
            encoding="utf8-lossy",
            truncate_ragged_lines=True,
        )
    except pl.exceptions.PolarsError as e:
        raise ReadError(f"Could not parse the file as delimited text: {e}") from e
    finally:
        if source != path:
            source.unlink(missing_ok=True)

    if frame.height == 0:
        raise ReadError("The file is empty")
    header = [(v or "").lstrip("\ufeff") for v in frame.row(0)]
    frame = frame.slice(1)
    frame.columns = _dedupe_headers(header)
    return frame, ParseInfo(format="csv", delimiter=delimiter, encoding=encoding)


def read_xlsx(path: Path, options: ParseOptions) -> tuple[pl.DataFrame, ParseInfo]:
    try:
        sheets = fastexcel.read_excel(str(path)).sheet_names
    except Exception as e:  # fastexcel raises its own exception hierarchy
        raise ReadError(f"Could not open the workbook: {e}") from e
    if not sheets:
        raise ReadError("The workbook has no sheets")
    sheet = options.sheet or sheets[0]
    if sheet not in sheets:
        raise ReadError(f"Sheet '{sheet}' not found")
    try:
        frame = pl.read_excel(path, sheet_name=sheet, engine="calamine", infer_schema_length=0)
    except Exception as e:
        raise ReadError(f"Could not read sheet '{sheet}': {e}") from e

    frame.columns = _dedupe_headers(frame.columns)
    # Excel dates come through as "2020-01-31 00:00:00"; drop the empty time part.
    frame = frame.with_columns(pl.all().str.replace(r"^(\d{4}-\d{2}-\d{2}) 00:00:00$", "$1"))
    return frame, ParseInfo(format="xlsx", sheet=sheet, sheets=sheets)


def read_file(path: Path, filename: str, options: ParseOptions) -> tuple[pl.DataFrame, ParseInfo]:
    fmt = detect_format(filename)
    frame, info = read_xlsx(path, options) if fmt == "xlsx" else read_csv(path, filename, options)
    if frame.width == 0:
        raise ReadError("The file has no columns")
    # Blank cells become null so they are treated as missing everywhere downstream.
    frame = frame.with_columns(pl.when(pl.all().str.strip_chars() != "").then(pl.all()).name.keep())
    return frame, info

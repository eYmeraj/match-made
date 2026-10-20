import datetime as dt
from pathlib import Path

import pytest

from matchmade.readers import ParseOptions, ReadError, detect_format, read_file


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_csv_detects_delimiter_and_keeps_everything_as_strings(tmp_path):
    path = write(tmp_path, "a.csv", b"id;amount\n007;1,5\n008;\n")
    frame, info = read_file(path, "a.csv", ParseOptions())
    assert info.delimiter == ";"
    assert frame.to_dict(as_series=False) == {"id": ["007", "008"], "amount": ["1,5", None]}


def test_tsv_bom_blank_and_duplicate_headers(tmp_path):
    path = write(tmp_path, "a.tsv", "﻿name\tname\t\n x \t  \tz\n".encode())
    frame, info = read_file(path, "a.tsv", ParseOptions())
    assert info.delimiter == "\t"
    assert frame.columns == ["name", "name_1", "column_3"]
    assert frame.row(0) == (" x ", None, "z")


def test_non_utf8_is_transcoded(tmp_path):
    path = write(tmp_path, "a.csv", "name,city\nMüller,Zürich\n".encode("latin-1"))
    frame, info = read_file(path, "a.csv", ParseOptions())
    assert info.encoding != "utf-8"
    assert frame.row(0) == ("Müller", "Zürich")
    frame, _ = read_file(path, "a.csv", ParseOptions(encoding="latin-1"))
    assert frame.row(0) == ("Müller", "Zürich")


def test_explicit_options_and_errors(tmp_path):
    path = write(tmp_path, "a.csv", b"a|b\n1|2\n")
    frame, _ = read_file(path, "a.csv", ParseOptions(delimiter="|"))
    assert frame.columns == ["a", "b"]
    with pytest.raises(ReadError, match="single character"):
        read_file(path, "a.csv", ParseOptions(delimiter="||"))
    with pytest.raises(ReadError, match="Unknown encoding"):
        read_file(path, "a.csv", ParseOptions(encoding="klingon"))
    with pytest.raises(ReadError, match="Unsupported"):
        detect_format("a.pdf")


def test_xlsx_sheets_and_dates(tmp_path):
    path = tmp_path / "book.xlsx"
    import xlsxwriter

    workbook = xlsxwriter.Workbook(path)
    first = workbook.add_worksheet("First")
    first.write_row(0, 0, ["name", "when", "n"])
    first.write_row(1, 0, ["Acme", "", 2])
    first.write_datetime(
        1, 1, dt.datetime(2020, 1, 31), workbook.add_format({"num_format": "yyyy-mm-dd"})
    )
    second = workbook.add_worksheet("Second")
    second.write_row(0, 0, ["other"])
    second.write_row(1, 0, ["x"])
    workbook.close()

    frame, info = read_file(path, "book.xlsx", ParseOptions())
    assert info.sheets == ["First", "Second"]
    assert info.sheet == "First"
    assert frame.to_dict(as_series=False) == {"name": ["Acme"], "when": ["2020-01-31"], "n": ["2"]}
    frame, info = read_file(path, "book.xlsx", ParseOptions(sheet="Second"))
    assert frame.columns == ["other"]
    with pytest.raises(ReadError, match="not found"):
        read_file(path, "book.xlsx", ParseOptions(sheet="Nope"))


def test_corrupt_xlsx(tmp_path):
    path = write(tmp_path, "bad.xlsx", b"not a zip")
    with pytest.raises(ReadError, match="Could not open"):
        read_file(path, "bad.xlsx", ParseOptions())

import csv
import datetime as dt
import io
import random

import polars as pl
import pytest

A_CSV = """id,Company Name,Founded,Revenue,Public,Country
1,Acme Widgets Inc.,2001-03-15,1200000,true,US
2,Globex Corporation,1989-07-01,98000000,true,US
3,Initech LLC,1999-02-19,4500000,false,US
4,Umbrella Corp,1970-10-10,250000000,true,UK
5,Café Müller GmbH,2010-05-05,310000,false,DE
6,Stark Industries,1940-01-01,,true,US
7,Nobody Matches Me Ltd,2020-01-01,1,false,FR
"""

B_CSV = """vendor;incorporated;revenue_usd;listed;country
ACME WIDGETS;15/03/2001;"$1,200,000";Y;us
Globex Corp.;01/07/1989;"$98,000,000";Y;us
Initech;20/02/1999;"$4,400,000";N;us
Umbrella Corporation Ltd;10/10/1970;"$250,000,000";Y;uk
Cafe Muller;05/05/2010;"$310,000";N;de
Stark Industries Inc;01/01/1940;;Y;us
Wayne Enterprises;17/04/1939;"$5,000,000";Y;us
"""


def upload(client, name, content):
    response = client.post("/api/datasets", files={"file": (name, content.encode(), "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def make_config(a, b, **overrides):
    config = {
        "dataset_a": a["id"],
        "dataset_b": b["id"],
        "comparisons": [
            {
                "id": "name",
                "type": "text",
                "column_a": "Company Name",
                "column_b": "vendor",
                "weight": 3,
                "metric": "token_sort",
                "cleaning": {"remove_legal_suffixes": True, "expand_abbreviations": True},
            },
            {
                "id": "founded",
                "type": "date",
                "column_a": "Founded",
                "column_b": "incorporated",
                "cleaning": {"day_first": True},
                "tolerance_days": 30,
            },
            {
                "id": "revenue",
                "type": "number",
                "column_a": "Revenue",
                "column_b": "revenue_usd",
                "tolerance": 0.1,
                "tolerance_mode": "relative",
            },
        ],
        "blocking": {
            "rules": [{"kind": "tfidf", "comparison_id": "name", "top_k": 3, "min_similarity": 0.1}]
        },
        "threshold": 80,
        "cardinality": "one_to_one",
    }
    config.update(overrides)
    return config


@pytest.fixture
def datasets(client):
    return upload(client, "companies.csv", A_CSV), upload(client, "vendors.csv", B_CSV)


def test_upload_profiles_columns(datasets):
    a, b = datasets
    assert a["rows"] == 7
    assert {c["name"]: c["inferred_type"] for c in a["columns"]} == {
        "id": "number",
        "Company Name": "text",
        "Founded": "date",
        "Revenue": "number",
        "Public": "boolean",
        "Country": "text",
    }
    assert b["parse"]["delimiter"] == ";"
    assert {c["name"]: c["inferred_type"] for c in b["columns"]} == {
        "vendor": "text",
        "incorporated": "date",
        "revenue_usd": "number",
        "listed": "boolean",
        "country": "text",
    }


def test_rows_and_reparse(client, datasets):
    _, b = datasets
    rows = client.get(f"/api/datasets/{b['id']}/rows", params={"limit": 2}).json()
    assert rows["total"] == 7
    assert rows["columns"][0] == "vendor"
    assert rows["rows"] == [
        ["ACME WIDGETS", "15/03/2001", "$1,200,000", "Y", "us"],
        ["Globex Corp.", "01/07/1989", "$98,000,000", "Y", "us"],
    ]

    # A wrong delimiter fails cleanly and leaves the dataset as it was.
    wrong = client.post(f"/api/datasets/{b['id']}/parse", json={"delimiter": ","})
    assert wrong.status_code == 422
    assert client.get(f"/api/datasets/{b['id']}").json()["parse"]["delimiter"] == ";"

    tabbed = client.post(f"/api/datasets/{b['id']}/parse", json={"delimiter": "\t"}).json()
    assert [c["name"] for c in tabbed["columns"]] == [
        "vendor;incorporated;revenue_usd;listed;country"
    ]


def test_unsupported_and_unknown(client):
    response = client.post("/api/datasets", files={"file": ("x.pdf", b"%PDF", "application/pdf")})
    assert response.status_code == 422
    assert "Unsupported" in response.json()["detail"]
    assert client.get("/api/datasets/" + "0" * 32).status_code == 404
    assert client.get("/api/datasets/../../etc").status_code == 404
    assert client.get("/api/jobs/" + "f" * 32).status_code == 404


def test_cleaning_preview(client, datasets):
    _, b = datasets
    response = client.post(
        "/api/preview/cleaning",
        json={"dataset_id": b["id"], "column": "revenue_usd", "type": "number", "cleaning": {}},
    )
    body = response.json()
    assert body["parsed_fraction"] == 1
    assert body["samples"][0] == {"raw": "$1,200,000", "cleaned": "1200000.0"}

    response = client.post(
        "/api/preview/cleaning",
        json={
            "dataset_id": b["id"],
            "column": "incorporated",
            "type": "date",
            "cleaning": {"format": "%Y-%m-%d"},
        },
    )
    assert response.json()["parsed_fraction"] == 0
    assert "15/03/2001" in response.json()["failures"]


def test_suggest_pairs_columns_by_name_and_values(client, datasets):
    a, b = datasets
    suggestion = client.post(
        "/api/suggest", json={"dataset_a": a["id"], "dataset_b": b["id"]}
    ).json()
    comparisons = suggestion["comparisons"]
    assert [(c["id"], c["column_a"], c["column_b"], c["type"]) for c in comparisons] == [
        ("c1", "Company Name", "vendor", "text"),
        ("c2", "Founded", "incorporated", "date"),
        ("c3", "Revenue", "revenue_usd", "number"),
        ("c4", "Country", "country", "text"),
    ]
    company, founded = comparisons[0], comparisons[1]
    assert company["weight"] == 2
    assert company["cleaning"]["remove_legal_suffixes"] is True
    # A uses ISO dates, B uses day-first dates.
    assert founded["cleaning"]["day_first"] is False
    assert founded["cleaning_b"]["day_first"] is True
    assert suggestion["blocking"]["rules"] == []  # 7 x 7 rows: compare everything


def test_estimate(client, datasets):
    a, b = datasets
    body = client.post("/api/estimate", json=make_config(a, b)).json()
    assert body["all_pairs"] == 49
    assert body["rules"][0]["upper_bound"] is True


def test_invalid_config_is_rejected(client, datasets):
    a, b = datasets
    config = make_config(a, b)
    config["comparisons"][0]["column_a"] = "nope"
    response = client.post("/api/jobs", json=config)
    assert response.status_code == 422
    assert "nope" in response.json()["detail"]

    config = make_config(a, b)
    config["blocking"] = {
        "rules": [
            {"kind": "key", "parts": [{"comparison_id": "founded", "transform": "first_token"}]}
        ]
    }
    assert client.post("/api/jobs", json=config).status_code == 422


def run_job(client, run_worker, config):
    created = client.post("/api/jobs", json=config)
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "queued"
    run_worker()
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "succeeded", status
    return job_id, status


def test_full_match_flow(client, run_worker, datasets):
    a, b = datasets
    job_id, status = run_job(client, run_worker, make_config(a, b))
    assert status["summary"]["rows_a"] == 7
    assert status["summary"]["keep_from"] == 55

    params = {"threshold": 80, "cardinality": "one_to_one"}
    stats = client.get(f"/api/jobs/{job_id}/stats", params=params).json()
    assert stats["matches"] == 6
    assert stats["unmatched_a"] == 1
    assert stats["unmatched_b"] == 1

    page = client.get(
        f"/api/jobs/{job_id}/matches", params={**params, "order": "desc", "limit": 10}
    ).json()
    matched = {(r["row_a"], r["row_b"]) for r in page["rows"]}
    assert matched == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)}
    stark = next(r for r in page["rows"] if r["row_a"] == 5)
    revenue = next(v for v in stark["values"] if v["comparison_id"] == "revenue")
    assert revenue == {"comparison_id": "revenue", "a": None, "b": None, "score": None}

    histogram = client.get(f"/api/jobs/{job_id}/histogram").json()
    assert sum(b["count"] for b in histogram) >= 6

    # Raising the threshold changes the result without re-running the job:
    # Initech is a day and 2% of revenue off, so it drops out.
    strict = client.get(
        f"/api/jobs/{job_id}/stats", params={"threshold": 99, "cardinality": "one_to_one"}
    ).json()
    assert strict["matches"] == 5
    assert strict["unmatched_a"] == 2
    too_low = client.get(
        f"/api/jobs/{job_id}/stats", params={"threshold": 10, "cardinality": "one_to_one"}
    )
    assert too_low.status_code == 422

    merged = client.get(f"/api/jobs/{job_id}/export", params={**params, "kind": "merged"})
    assert merged.status_code == 200
    assert "merged.csv" in merged.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(merged.text)))
    assert len(rows) == 6
    assert list(rows[0])[:6] == [
        "match_score",
        "score: Company Name ~ vendor",
        "score: Founded ~ incorporated",
        "score: Revenue ~ revenue_usd",
        "row (A)",
        "row (B)",
    ]
    assert rows[0]["Company Name"] == "Acme Widgets Inc."
    assert rows[0]["vendor"] == "ACME WIDGETS"
    assert rows[0]["Country"] == "US"
    assert rows[0]["country"] == "us"

    left = client.get(
        f"/api/jobs/{job_id}/export", params={**params, "kind": "merged", "join": "left"}
    )
    assert len(list(csv.DictReader(io.StringIO(left.text)))) == 7
    full = client.get(
        f"/api/jobs/{job_id}/export", params={**params, "kind": "merged", "join": "full"}
    )
    assert len(list(csv.DictReader(io.StringIO(full.text)))) == 8

    unmatched_a = client.get(f"/api/jobs/{job_id}/export", params={**params, "kind": "unmatched_a"})
    assert [r["Company Name"] for r in csv.DictReader(io.StringIO(unmatched_a.text))] == [
        "Nobody Matches Me Ltd"
    ]
    unmatched_b = client.get(
        f"/api/jobs/{job_id}/export", params={**params, "kind": "unmatched_b", "format": "xlsx"}
    )
    assert unmatched_b.status_code == 200
    sheet = pl.read_excel(io.BytesIO(unmatched_b.content))
    assert sheet.get_column("vendor").to_list() == ["Wayne Enterprises"]


def test_exact_blocking_on_cleaned_values(client, run_worker, datasets):
    a, b = datasets
    rule = {"kind": "key", "parts": [{"comparison_id": "name", "transform": "exact"}]}
    job_id, status = run_job(client, run_worker, make_config(a, b, blocking={"rules": [rule]}))
    # Suffix removal makes the six real pairs share an exact key.
    assert status["summary"]["candidate_pairs"] == 6
    stats = client.get(f"/api/jobs/{job_id}/stats", params={"threshold": 80, "cardinality": "all"})
    assert stats.json()["matches"] == 6


def test_job_failure_is_reported(client, run_worker, datasets, monkeypatch):
    from matchmade.settings import get_settings

    monkeypatch.setenv("MATCHMADE_MAX_CANDIDATES", "10")
    get_settings.cache_clear()
    a, b = datasets
    job_id = client.post("/api/jobs", json=make_config(a, b, blocking={"rules": []})).json()["id"]
    run_worker()
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "failed"
    assert status["error"] == (
        "Comparing every row pair would create 49 pairs (limit 10). Add a blocking rule."
    )


def test_cancel_queued_job(client, datasets):
    a, b = datasets
    job_id = client.post("/api/jobs", json=make_config(a, b)).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["state"] == "cancelled"


def _noisy_pairs(n: int = 300, seed: int = 3):
    """A and B describe the same companies under unrelated column names, with noise."""
    rng = random.Random(seed)
    syllables = ["kor", "vex", "lum", "tra", "nio", "pel", "zar", "qui", "mon", "dra"]
    a_lines = ["company_id,Company Name,Founded,Revenue"]
    b_lines = ["vendor;incorporated;turnover"]
    for i in range(n):
        name = " ".join("".join(rng.choices(syllables, k=3)).title() for _ in range(2))
        day = dt.date(1950, 1, 1) + dt.timedelta(days=rng.randrange(25_000))
        revenue = rng.randrange(10_000, 50_000_000)
        a_lines.append(f"A{i:07d},{name} Ltd,{day.isoformat()},{revenue}")
        shifted = day + dt.timedelta(days=rng.choice([0, 0, 3, -20]))
        noisy_revenue = round(revenue * rng.uniform(0.98, 1.02))
        b_lines.append(f"{name.upper()};{shifted.strftime('%d/%m/%Y')};{noisy_revenue}")
    return "\n".join(a_lines) + "\n", "\n".join(b_lines) + "\n"


def test_suggest_pairs_columns_by_values_when_names_differ(client):
    a_csv, b_csv = _noisy_pairs()
    a, b = upload(client, "a.csv", a_csv), upload(client, "b.csv", b_csv)
    assert {c["name"]: c["inferred_type"] for c in a["columns"]}["company_id"] == "text"

    suggestion = client.post(
        "/api/suggest", json={"dataset_a": a["id"], "dataset_b": b["id"]}
    ).json()
    by_column = {c["column_a"]: c for c in suggestion["comparisons"]}
    assert {k: v["column_b"] for k, v in by_column.items()} == {
        "Company Name": "vendor",
        "Founded": "incorporated",
        "Revenue": "turnover",
    }
    # Values that only roughly agree get a tolerance instead of an exact comparison.
    assert by_column["Revenue"]["tolerance_mode"] == "relative"
    assert by_column["Revenue"]["tolerance"] == 0.05
    assert by_column["Founded"]["tolerance_days"] == 31

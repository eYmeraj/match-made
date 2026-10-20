"""Run the full pipeline on generate_synthetic.py output and report speed and accuracy.

uv run python scripts/benchmark.py ../.scratch/synthetic
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl

from matchmade import results
from matchmade.pipeline import run_match
from matchmade.readers import ParseOptions
from matchmade.schemas import (
    Blocking,
    BlockingKeyPart,
    Cardinality,
    DateCleaning,
    DateComparison,
    ExactBlockingRule,
    KeyTransform,
    MatchConfig,
    NumberComparison,
    TextCleaning,
    TextComparison,
    TfidfBlockingRule,
)
from matchmade.settings import Settings
from matchmade.storage import Storage


def load(storage: Storage, path: Path) -> str:
    dataset_id, directory = storage.new_dataset_dir()
    shutil.copy(path, directory / "source.csv")
    storage.parse_dataset(dataset_id, path.name, ParseOptions())
    return dataset_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--blocking", choices=["tfidf", "key", "both"], default="tfidf")
    parser.add_argument("--threshold", type=float, default=80)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="matchmade-bench-"))
    storage = Storage(root)
    settings = Settings(data_dir=root)
    t0 = time.perf_counter()
    a_id = load(storage, args.data / "a.csv")
    b_id = load(storage, args.data / "b.csv")
    print(f"upload+parse+profile: {time.perf_counter() - t0:.1f}s")

    rules = {
        "tfidf": [TfidfBlockingRule(comparison_id="name", top_k=5, min_similarity=0.5)],
        "key": [
            ExactBlockingRule(
                parts=[
                    BlockingKeyPart(comparison_id="name", transform=KeyTransform.FIRST_TOKEN),
                    BlockingKeyPart(comparison_id="founded", transform=KeyTransform.YEAR),
                ]
            )
        ],
    }
    chosen = rules["tfidf"] + rules["key"] if args.blocking == "both" else rules[args.blocking]
    config = MatchConfig(
        dataset_a=a_id,
        dataset_b=b_id,
        comparisons=[
            TextComparison(
                id="name",
                column_a="Company Name",
                column_b="vendor",
                weight=3,
                cleaning=TextCleaning(remove_legal_suffixes=True, expand_abbreviations=True),
            ),
            DateComparison(
                id="founded",
                column_a="Founded",
                column_b="incorporated",
                tolerance_days=31,
                cleaning_b=DateCleaning(day_first=True),
            ),
            NumberComparison(
                id="revenue",
                column_a="Revenue",
                column_b="annual_revenue",
                tolerance=0.05,
                tolerance_mode="relative",
            ),
        ],
        blocking=Blocking(rules=chosen),
        threshold=args.threshold,
        cardinality=Cardinality.ONE_TO_ONE,
    )
    job_id = storage.create_job(config)

    last = [0.0]

    def report(stage: str, progress: float) -> None:
        if progress - last[0] >= 0.1 or progress >= 1:
            last[0] = progress
            print(f"  {stage:<6} {progress:5.0%}  t={time.perf_counter() - t1:6.1f}s", flush=True)

    t1 = time.perf_counter()
    summary = run_match(storage, job_id, settings, report)
    print(f"pipeline: {time.perf_counter() - t1:.1f}s {summary.seconds}")
    print(f"candidate pairs: {summary.candidate_pairs:,}, kept: {summary.kept_pairs:,}")

    t2 = time.perf_counter()
    found = results.matches(storage, job_id, args.threshold, Cardinality.ONE_TO_ONE)
    print(f"assignment: {time.perf_counter() - t2:.1f}s, matches: {found.height:,}")

    truth = pl.read_csv(args.data / "truth.csv", schema={"row_a": pl.UInt32, "row_b": pl.UInt32})
    hits = found.join(truth, left_on=["a", "b"], right_on=["row_a", "row_b"], how="semi").height
    precision = hits / max(found.height, 1)
    recall = hits / truth.height
    print(f"precision: {precision:.4f}  recall: {recall:.4f}")

    t3 = time.perf_counter()
    path = results.export_file(
        storage, job_id, results.ExportKind.MERGED, "csv", args.threshold,
        Cardinality.ONE_TO_ONE, results.JoinMode.MATCHED,
    )  # fmt: skip
    print(
        f"export merged csv: {time.perf_counter() - t3:.1f}s ({path.stat().st_size / 1e6:.0f} MB)"
    )
    shutil.rmtree(root)


if __name__ == "__main__":
    main()

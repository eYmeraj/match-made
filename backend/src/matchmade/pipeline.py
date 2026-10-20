"""The matching job: clean -> block -> score -> persist candidates."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict

import polars as pl

from matchmade.blocking import candidate_pairs
from matchmade.matching import SCORE, cleaned_frame, score_in_chunks
from matchmade.schemas import MatchConfig
from matchmade.settings import Settings
from matchmade.storage import JobSummary, Storage, write_atomic

ProgressFn = Callable[[str, float], None]

# Overall progress share per stage; scoring dominates on large inputs.
_STAGES = {"load": (0.0, 0.05), "clean": (0.05, 0.15), "block": (0.15, 0.35), "score": (0.35, 1.0)}


def _reporter(report: ProgressFn) -> Callable[[str, float], None]:
    def stage(name: str, fraction: float) -> None:
        start, end = _STAGES[name]
        report(name, start + (end - start) * fraction)

    return stage


def keep_from(threshold: float, settings: Settings) -> float:
    return max(0.0, threshold - settings.keep_margin)


def run_match(storage: Storage, job_id: str, settings: Settings, report: ProgressFn) -> JobSummary:
    config: MatchConfig = storage.job_config(job_id)
    job_dir = storage.job_dir(job_id)
    stage = _reporter(report)
    seconds: dict[str, float] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        seconds[name] = round(now - t, 3)
        t = now

    stage("load", 0)
    raw_a = storage.read_dataset(config.dataset_a, list({c.column_a for c in config.comparisons}))
    raw_b = storage.read_dataset(config.dataset_b, list({c.column_b for c in config.comparisons}))
    lap("load")

    stage("clean", 0)
    a = cleaned_frame(raw_a, config.comparisons, "a")
    b = cleaned_frame(raw_b, config.comparisons, "b")
    del raw_a, raw_b
    lap("clean")

    stage("block", 0)
    blocked = candidate_pairs(
        a, b, config.blocking, settings.max_candidates, on_progress=lambda f: stage("block", f)
    )
    lap("block")

    stage("score", 0)
    parts_dir = job_dir / "candidates"
    parts_dir.mkdir(exist_ok=True)
    floor = keep_from(config.threshold, settings)
    kept = 0
    chunks = score_in_chunks(
        blocked.pairs,
        a,
        b,
        config.comparisons,
        config.missing,
        keep_from=floor,
        chunk_size=settings.score_chunk_size,
        on_progress=lambda f: stage("score", f),
    )
    for i, chunk in enumerate(chunks):
        kept += chunk.height
        chunk.write_parquet(parts_dir / f"part-{i:05d}.parquet")
    if kept == 0:
        _empty_candidates(config).write_parquet(parts_dir / "part-00000.parquet")
    lap("score")

    summary = JobSummary(
        rows_a=a.height,
        rows_b=b.height,
        candidate_pairs=blocked.pairs.height,
        kept_pairs=kept,
        keep_from=floor,
        rule_stats=[asdict(s) for s in blocked.rules],
        seconds=seconds,
    )
    write_atomic(job_dir / "summary.json", summary.model_dump_json())
    return summary


def _empty_candidates(config: MatchConfig) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {"a": pl.UInt32(), "b": pl.UInt32()}
    schema |= {f"s_{c.id}": pl.Float32() for c in config.comparisons}
    schema[SCORE] = pl.Float32()
    return pl.DataFrame(schema=schema)

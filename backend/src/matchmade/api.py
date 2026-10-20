"""HTTP API."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
from fastapi import APIRouter, Depends, FastAPI, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from redis import Redis
from starlette.background import BackgroundTask

from matchmade import blocking, jobs, results
from matchmade.cleaning import DEFAULT_CLEANING, clean
from matchmade.matching import cleaned_frame
from matchmade.readers import ParseOptions, ReadError, detect_format
from matchmade.schemas import (
    Blocking,
    Cardinality,
    Cleaning,
    ColumnType,
    Comparison,
    KeyTransform,
    MatchConfig,
    allowed_transforms,
)
from matchmade.settings import Settings, get_settings
from matchmade.storage import DatasetMeta, InUseError, NotFoundError, Storage
from matchmade.suggest import suggest_blocking, suggest_comparisons

_UPLOAD_CHUNK = 1 << 20
_PREVIEW_ROWS = 20_000


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


StorageDep = Annotated[Storage, Depends(get_storage)]
RedisDep = Annotated[Redis, Depends(get_redis)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

router = APIRouter(prefix="/api")


# --- meta -----------------------------------------------------------------------------


class Meta(BaseModel):
    transforms: dict[ColumnType, list[KeyTransform]]
    default_cleaning: dict[ColumnType, dict]


@router.get("/meta")
def meta() -> Meta:
    return Meta(
        transforms={t: sorted(allowed_transforms(t)) for t in ColumnType},
        default_cleaning={t: cls().model_dump() for t, cls in DEFAULT_CLEANING.items()},
    )


# --- datasets -------------------------------------------------------------------------


@router.post("/datasets")
def upload_dataset(file: UploadFile, storage: StorageDep) -> DatasetMeta:
    filename = Path(file.filename or "upload.csv").name
    fmt = detect_format(filename)
    dataset_id, directory = storage.new_dataset_dir()
    try:
        with open(directory / f"source.{fmt}", "wb") as out:
            shutil.copyfileobj(file.file, out, length=_UPLOAD_CHUNK)
        return storage.parse_dataset(dataset_id, filename, ParseOptions())
    except Exception:
        storage.delete_dataset(dataset_id)
        raise


@router.post("/datasets/{dataset_id}/parse")
def reparse_dataset(dataset_id: str, options: ParseOptions, storage: StorageDep) -> DatasetMeta:
    return storage.reparse_dataset(dataset_id, options)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, storage: StorageDep) -> DatasetMeta:
    return storage.dataset_meta(dataset_id)


class RowsPage(BaseModel):
    columns: list[str]
    rows: list[list[str | None]]
    total: int


@router.get("/datasets/{dataset_id}/rows")
def dataset_rows(
    dataset_id: str,
    storage: StorageDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> RowsPage:
    frame = storage.read_dataset(dataset_id)
    page = frame.slice(offset, limit)
    return RowsPage(
        columns=frame.columns, rows=[list(r) for r in page.iter_rows()], total=frame.height
    )


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, storage: StorageDep) -> None:
    storage.delete_dataset(dataset_id)


# --- cleaning preview -----------------------------------------------------------------


class CleaningPreviewRequest(BaseModel):
    dataset_id: str
    column: str
    type: ColumnType
    cleaning: dict = Field(default_factory=dict)


class CleanedSample(BaseModel):
    raw: str
    cleaned: str | None


class CleaningPreview(BaseModel):
    samples: list[CleanedSample]
    failures: list[str]
    parsed_fraction: float
    sampled_rows: int


@router.post("/preview/cleaning")
def preview_cleaning(body: CleaningPreviewRequest, storage: StorageDep) -> CleaningPreview:
    opts: Cleaning = DEFAULT_CLEANING[body.type].model_validate(body.cleaning)
    _require_columns(storage, body.dataset_id, [body.column])
    raw = storage.read_dataset(body.dataset_id, [body.column]).head(_PREVIEW_ROWS)
    frame = (
        raw.select(pl.col(body.column).alias("raw"))
        .drop_nulls()
        .with_columns(clean(pl.col("raw"), body.type, opts).cast(pl.String).alias("cleaned"))
    )
    distinct = frame.unique("raw", maintain_order=True)
    failures = distinct.filter(pl.col("cleaned").is_null()).head(8)
    return CleaningPreview(
        samples=[CleanedSample(**r) for r in distinct.head(12).iter_rows(named=True)],
        failures=failures.get_column("raw").to_list(),
        parsed_fraction=1 - frame.get_column("cleaned").null_count() / max(frame.height, 1),
        sampled_rows=frame.height,
    )


# --- configuration helpers ------------------------------------------------------------


class SuggestRequest(BaseModel):
    dataset_a: str
    dataset_b: str


class Suggestion(BaseModel):
    comparisons: list[Comparison]
    blocking: Blocking


@router.post("/suggest")
def suggest(body: SuggestRequest, storage: StorageDep) -> Suggestion:
    comparisons = suggest_comparisons(storage, body.dataset_a, body.dataset_b)
    return Suggestion(
        comparisons=comparisons,
        blocking=suggest_blocking(storage, body.dataset_a, body.dataset_b, comparisons),
    )


class RuleEstimate(BaseModel):
    rule_index: int
    pairs: int
    skipped_blocks: int
    skipped_pairs: int
    # TF-IDF counts are estimated from average duplication; key rule counts are exact.
    approximate: bool


class Estimate(BaseModel):
    rows_a: int
    rows_b: int
    all_pairs: int
    rules: list[RuleEstimate]
    max_candidates: int


def _require_columns(storage: Storage, dataset_id: str, columns: list[str]) -> None:
    available = {c.name for c in storage.dataset_meta(dataset_id).columns}
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")


def _validate_config(storage: Storage, config: MatchConfig) -> None:
    _require_columns(storage, config.dataset_a, [c.column_a for c in config.comparisons])
    _require_columns(storage, config.dataset_b, [c.column_b for c in config.comparisons])


@router.post("/estimate")
def estimate(config: MatchConfig, storage: StorageDep, settings: SettingsDep) -> Estimate:
    _validate_config(storage, config)
    raw_a = storage.read_dataset(config.dataset_a, list({c.column_a for c in config.comparisons}))
    raw_b = storage.read_dataset(config.dataset_b, list({c.column_b for c in config.comparisons}))
    a = cleaned_frame(raw_a, config.comparisons, "a")
    b = cleaned_frame(raw_b, config.comparisons, "b")
    stats = blocking.estimate(a, b, config.blocking)
    return Estimate(
        rows_a=a.height,
        rows_b=b.height,
        all_pairs=a.height * b.height,
        rules=[
            RuleEstimate(
                rule_index=s.rule_index,
                pairs=s.pairs,
                skipped_blocks=s.skipped_blocks,
                skipped_pairs=s.skipped_pairs,
                approximate=config.blocking.rules[s.rule_index].kind == "tfidf",
            )
            for s in stats
        ],
        max_candidates=settings.max_candidates,
    )


# --- jobs -----------------------------------------------------------------------------


@router.post("/jobs", status_code=201)
def create_job(
    config: MatchConfig, storage: StorageDep, connection: RedisDep, settings: SettingsDep
) -> jobs.JobStatus:
    _validate_config(storage, config)
    job_id = storage.create_job(config)
    jobs.enqueue(connection, job_id, settings)
    return jobs.JobStatus(id=job_id, state="queued")


@router.get("/jobs/{job_id}")
def job_status(job_id: str, storage: StorageDep, connection: RedisDep) -> jobs.JobStatus:
    return jobs.status(storage, connection, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, storage: StorageDep, connection: RedisDep) -> jobs.JobStatus:
    jobs.cancel(storage, connection, job_id)
    return jobs.status(storage, connection, job_id)


@router.get("/jobs/{job_id}/config")
def job_config(job_id: str, storage: StorageDep) -> MatchConfig:
    return storage.job_config(job_id)


@router.get("/jobs/{job_id}/histogram")
def job_histogram(job_id: str, storage: StorageDep) -> list[results.HistogramBin]:
    return results.histogram(storage, job_id)


Threshold = Annotated[float, Query(ge=0, le=100)]


@router.get("/jobs/{job_id}/stats")
def job_stats(
    job_id: str, threshold: Threshold, cardinality: Cardinality, storage: StorageDep
) -> results.ResultStats:
    return results.stats(storage, job_id, threshold, cardinality)


@router.get("/jobs/{job_id}/matches")
def job_matches(
    job_id: str,
    threshold: Threshold,
    cardinality: Cardinality,
    storage: StorageDep,
    order: Literal["asc", "desc"] = "asc",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> results.MatchPage:
    return results.match_page(storage, job_id, threshold, cardinality, order, offset, limit)


@router.get("/jobs/{job_id}/export")
def job_export(
    job_id: str,
    kind: results.ExportKind,
    threshold: Threshold,
    cardinality: Cardinality,
    storage: StorageDep,
    format: results.ExportFormat = "csv",
    join: results.JoinMode = results.JoinMode.MATCHED,
) -> FileResponse:
    path = results.export_file(storage, job_id, kind, format, threshold, cardinality, join)
    media = (
        "text/csv"
        if format == "csv"
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return FileResponse(
        path,
        media_type=media,
        filename=f"{kind.value}.{format}",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- app ------------------------------------------------------------------------------


def create_app(storage: Storage | None = None, connection: Redis | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Match Made")
    app.state.storage = storage or Storage(settings.data_dir)
    app.state.redis = connection or jobs.redis_connection(settings)
    app.include_router(router)

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    @app.exception_handler(InUseError)
    async def _in_use(_: Request, exc: InUseError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ReadError)
    @app.exception_handler(ValueError)
    async def _invalid(_: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, ValidationError):  # options validated inside an endpoint
            detail = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        else:
            detail = str(exc)
        return JSONResponse(status_code=422, content={"detail": detail})

    return app

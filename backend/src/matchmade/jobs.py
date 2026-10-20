"""Background execution of matching jobs on an RQ worker.

The job directory on disk is the source of truth for finished jobs (summary.json or
error.json); Redis only tracks queued and running jobs and their progress.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Literal

from pydantic import BaseModel
from redis import Redis
from rq import Queue, get_current_job
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job
from rq.job import JobStatus as RqStatus

from matchmade.blocking import BlockingError
from matchmade.pipeline import run_match
from matchmade.settings import Settings, get_settings
from matchmade.storage import JobSummary, Storage, write_atomic

log = logging.getLogger(__name__)

QUEUE = "matchmade"
_PROGRESS_INTERVAL = 0.5

State = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobStatus(BaseModel):
    id: str
    state: State
    stage: str | None = None
    progress: float = 0.0
    error: str | None = None
    summary: JobSummary | None = None


def redis_connection(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url)


def enqueue(connection: Redis, job_id: str, settings: Settings) -> None:
    Queue(QUEUE, connection=connection).enqueue(
        "matchmade.jobs.execute",
        job_id,
        job_id=job_id,
        job_timeout=settings.job_timeout_seconds,
        result_ttl=24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )


def _write_error(storage: Storage, job_id: str, message: str, state: State = "failed") -> None:
    write_atomic(
        storage.job_dir(job_id) / "error.json", json.dumps({"state": state, "error": message})
    )


def execute(job_id: str) -> None:
    """Entry point run by the RQ worker."""
    settings = get_settings()
    storage = Storage(settings.data_dir)
    rq_job = get_current_job()
    last = 0.0

    def report(stage: str, progress: float) -> None:
        nonlocal last
        now = time.monotonic()
        if rq_job is None or (now - last < _PROGRESS_INTERVAL and progress < 1):
            return
        last = now
        rq_job.meta.update(stage=stage, progress=round(progress, 4))
        rq_job.save_meta()

    try:
        run_match(storage, job_id, settings, report)
    except BlockingError as e:
        _write_error(storage, job_id, str(e))
    except Exception as e:
        log.exception("job %s failed", job_id)
        _write_error(storage, job_id, f"Unexpected error: {type(e).__name__}: {e}")
        raise


def status(storage: Storage, connection: Redis, job_id: str) -> JobStatus:
    job_dir = storage.job_dir(job_id)
    if (job_dir / "summary.json").exists():
        summary = JobSummary.model_validate_json((job_dir / "summary.json").read_text())
        return JobStatus(id=job_id, state="succeeded", progress=1.0, summary=summary)
    if (job_dir / "error.json").exists():
        data = json.loads((job_dir / "error.json").read_text())
        return JobStatus(id=job_id, state=data["state"], error=data["error"])

    try:
        rq_job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        return JobStatus(id=job_id, state="failed", error="The job was lost (worker restarted?)")
    rq_status = rq_job.get_status(refresh=True)
    meta = rq_job.get_meta(refresh=True)
    stage, progress = meta.get("stage"), float(meta.get("progress", 0.0))
    match rq_status:
        case RqStatus.QUEUED | RqStatus.DEFERRED | RqStatus.SCHEDULED:
            return JobStatus(id=job_id, state="queued")
        case RqStatus.STARTED:
            return JobStatus(id=job_id, state="running", stage=stage, progress=progress)
        case RqStatus.STOPPED | RqStatus.CANCELED:
            return JobStatus(id=job_id, state="cancelled", stage=stage, progress=progress)
        case RqStatus.FAILED:
            return JobStatus(id=job_id, state="failed", error="The job failed or timed out")
        case _:
            # FINISHED without summary.json means the summary write is still landing.
            return JobStatus(id=job_id, state="running", stage=stage, progress=progress)


def cancel(storage: Storage, connection: Redis, job_id: str) -> None:
    storage.job_dir(job_id)
    try:
        rq_job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        return
    if rq_job.get_status() == RqStatus.STARTED:
        with contextlib.suppress(InvalidJobOperation):
            send_stop_job_command(connection, job_id)
    else:
        rq_job.cancel()
    _write_error(storage, job_id, "Cancelled", state="cancelled")

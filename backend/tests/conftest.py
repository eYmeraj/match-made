from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker

from matchmade.api import create_app
from matchmade.jobs import QUEUE
from matchmade.settings import get_settings
from matchmade.storage import Storage


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("MATCHMADE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def storage(data_dir: Path) -> Storage:
    return Storage(data_dir)


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


@pytest.fixture
def client(storage: Storage, redis: fakeredis.FakeRedis) -> TestClient:
    return TestClient(create_app(storage=storage, connection=redis))


@pytest.fixture
def run_worker(redis: fakeredis.FakeRedis):
    """Drain the job queue in-process, like the real worker would."""

    def run() -> None:
        SimpleWorker([Queue(QUEUE, connection=redis)], connection=redis).work(burst=True)

    return run

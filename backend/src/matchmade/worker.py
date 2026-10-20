"""RQ worker entry point: ``matchmade-worker``."""

import logging

from rq import Worker

from matchmade.jobs import QUEUE, redis_connection
from matchmade.settings import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Worker([QUEUE], connection=redis_connection(settings)).work(with_scheduler=False)


if __name__ == "__main__":
    main()

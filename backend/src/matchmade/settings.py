from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MATCHMADE_")

    data_dir: Path = Path("data")
    redis_url: str = "redis://localhost:6379/0"
    # Hard cap on scored candidate pairs per run; bounds memory and runtime. 10M pairs peak
    # at ~2.3 GB in the worker and 26M did not fit Docker Desktop's default 3.8 GB VM.
    max_candidates: int = 15_000_000
    score_chunk_size: int = 2_000_000
    job_timeout_seconds: int = 6 * 60 * 60
    # Scores below `threshold - keep_margin` are discarded after scoring, so the
    # threshold can be lowered on the results page without re-running the job.
    keep_margin: float = 25.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

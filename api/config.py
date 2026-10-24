"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass


def _env_list(name, default=""):
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Origins allowed to call the API from another site. Empty = same-origin only,
    # which is all the bundled frontend needs.
    cors_origins: list[str]
    # Requests with a larger body are rejected with 413 before they are parsed.
    max_request_bytes: int
    # Threads PyTorch uses per worker; 0 keeps PyTorch's default (all cores).
    # With several workers, set this to roughly cores / workers.
    torch_threads: int
    # Serve the interactive API docs at /docs and /redoc.
    docs_enabled: bool
    # Only used by `python -m api.main`; in Docker, uvicorn reads UVICORN_HOST/UVICORN_PORT.
    host: str
    port: int


def load_settings():
    return Settings(
        cors_origins=_env_list("CORS_ORIGINS"),
        max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", 1_000_000)),
        torch_threads=int(os.getenv("TORCH_THREADS", 0)),
        docs_enabled=os.getenv("DOCS_ENABLED", "true").lower() in ("1", "true", "yes"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
    )

"""Shared HTTP and cache helpers for annotation data sources."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


USER_AGENT = "BiasTracker/0.1.0"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}


def get_session() -> requests.Session:
    """Create a requests session with BiasTracker defaults."""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def request_with_retries(
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    max_retries: int = 3,
    timeout: float = 60,
    backoff_factor: float = 1.5,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs: Any,
) -> requests.Response:
    """Perform an HTTP request with retries for transient failures."""
    http = session or get_session()
    attempts = max_retries + 1
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = http.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            _sleep_before_retry(attempt, backoff_factor)
            continue

        if response.status_code in retry_statuses:
            if attempt == attempts - 1:
                detail = _response_detail(response)
                raise RuntimeError(
                    f"{method.upper()} {url} failed after {attempts} attempts "
                    f"with HTTP {response.status_code}{detail}"
                )
            _sleep_before_retry(attempt, backoff_factor)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(
        f"{method.upper()} {url} failed after {attempts} attempts: {last_error}"
    ) from last_error


def ensure_cache_dir(cache_dir: str | Path) -> Path:
    """Create and return a cache directory."""
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_cache_key(value: str) -> str:
    """Return a short, stable filename-safe cache key."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def write_json_cache(path: str | Path, data: Any) -> None:
    """Write UTF-8 encoded JSON cache data."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True)


def read_json_cache(path: str | Path) -> Any | None:
    """Read JSON cache data, returning None when the file is absent."""
    cache_path = Path(path)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON cache file: {cache_path}") from exc


def _sleep_before_retry(attempt: int, backoff_factor: float) -> None:
    if backoff_factor <= 0:
        return
    time.sleep(backoff_factor * (2**attempt))


def _response_detail(response: requests.Response) -> str:
    text = response.text.strip()
    if not text:
        return ""
    return f": {text[:200]}"

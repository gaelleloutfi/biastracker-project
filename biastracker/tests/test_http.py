import json

import pytest
import requests

from biastracker.annotations._http import (
    ensure_cache_dir,
    get_session,
    read_json_cache,
    request_with_retries,
    safe_cache_key,
    write_json_cache,
)


class DummyResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_get_session_sets_biastracker_headers():
    session = get_session()

    assert session.headers["User-Agent"] == "BiasTracker/0.1.0"
    assert session.headers["Accept"] == "application/json"


def test_request_with_retries_retries_500_then_success(monkeypatch):
    monkeypatch.setattr("biastracker.annotations._http.time.sleep", lambda seconds: None)
    session = DummySession([DummyResponse(500, "try later"), DummyResponse(200, "ok")])

    response = request_with_retries("GET", "https://example.test/data", session=session)

    assert response.status_code == 200
    assert len(session.calls) == 2
    assert session.calls[0][2]["timeout"] == 60


def test_request_with_retries_retries_429_then_success(monkeypatch):
    monkeypatch.setattr("biastracker.annotations._http.time.sleep", lambda seconds: None)
    session = DummySession([DummyResponse(429, "rate limited"), DummyResponse(200, "ok")])

    response = request_with_retries("GET", "https://example.test/data", session=session)

    assert response.status_code == 200
    assert len(session.calls) == 2


def test_request_with_retries_does_not_retry_400(monkeypatch):
    monkeypatch.setattr("biastracker.annotations._http.time.sleep", lambda seconds: None)
    session = DummySession([DummyResponse(400, "bad request")])

    with pytest.raises(requests.HTTPError):
        request_with_retries("GET", "https://example.test/data", session=session)

    assert len(session.calls) == 1


def test_request_with_retries_raises_clear_runtime_error_after_retry_statuses(
    monkeypatch,
):
    monkeypatch.setattr("biastracker.annotations._http.time.sleep", lambda seconds: None)
    session = DummySession([DummyResponse(503, "busy"), DummyResponse(503, "still busy")])

    with pytest.raises(RuntimeError, match="failed after 2 attempts.*HTTP 503"):
        request_with_retries(
            "GET",
            "https://example.test/data",
            session=session,
            max_retries=1,
        )


def test_cache_read_write_and_missing(tmp_path):
    cache_dir = ensure_cache_dir(tmp_path / "cache")
    path = cache_dir / "data.json"
    data = {"ids": ["P12345", "Q67890"], "count": 2}

    assert cache_dir.exists()
    assert read_json_cache(path) is None

    write_json_cache(path, data)

    assert read_json_cache(path) == data


def test_read_json_cache_raises_clear_error_for_corrupt_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted JSON cache file"):
        read_json_cache(path)


def test_safe_cache_key_is_stable_and_short():
    value = json.dumps({"ids": ["P1", "P2"]}, sort_keys=True)

    assert safe_cache_key(value) == safe_cache_key(value)
    assert len(safe_cache_key(value)) == 20
    assert safe_cache_key(value).isalnum()

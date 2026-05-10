import pytest
import requests

from biastracker.annotations.uniprot_mapping import (
    extract_uniprot_mapping,
    fetch_id_mapping_results,
    map_ids_to_uniprot,
    poll_id_mapping_job,
    submit_id_mapping_job,
)


class DummyResponse:
    def __init__(self, payload, status_code=200, headers=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_submit_id_mapping_job_posts_unique_nonempty_ids():
    session = DummySession([DummyResponse({"jobId": "job-123"})])

    job_id = submit_id_mapping_job(
        ["ENSG1", "", "ENSG1", " ENSG2 "],
        from_ns="Ensembl",
        to_ns="UniProtKB",
        session=session,
    )

    assert job_id == "job-123"
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url.endswith("/idmapping/run")
    assert kwargs["data"] == {
        "from": "Ensembl",
        "to": "UniProtKB",
        "ids": "ENSG1,ENSG2",
    }


def test_submit_id_mapping_job_requires_ids():
    with pytest.raises(ValueError, match="No IDs provided"):
        submit_id_mapping_job(["", "  "], "Ensembl", "UniProtKB")


def test_submit_id_mapping_job_requires_job_id():
    session = DummySession([DummyResponse({})])

    with pytest.raises(RuntimeError, match="did not include a jobId"):
        submit_id_mapping_job(["ENSG1"], "Ensembl", "UniProtKB", session=session)


def test_poll_id_mapping_job_running_then_complete(monkeypatch):
    monkeypatch.setattr(
        "biastracker.annotations.uniprot_mapping.time.sleep",
        lambda seconds: None,
    )
    session = DummySession(
        [
            DummyResponse({"jobStatus": "RUNNING"}),
            DummyResponse({"results": [{"from": "ENSG1", "to": "P12345"}]}),
        ]
    )

    poll_id_mapping_job("job-123", session=session, poll_seconds=0)

    assert len(session.calls) == 2
    assert session.calls[0][1].endswith("/idmapping/status/job-123")


def test_poll_id_mapping_job_failure_raises():
    session = DummySession([DummyResponse({"jobStatus": "FAILED"})])

    with pytest.raises(RuntimeError, match="failed with status"):
        poll_id_mapping_job("job-123", session=session)


def test_fetch_id_mapping_results_one_page():
    session = DummySession(
        [DummyResponse({"results": [{"from": "ENSG1", "to": "P12345"}]})]
    )

    results = fetch_id_mapping_results("job-123", session=session)

    assert results == [{"from": "ENSG1", "to": "P12345"}]
    assert len(session.calls) == 1
    assert session.calls[0][1].endswith("/idmapping/results/job-123")


def test_fetch_id_mapping_results_follows_link_pagination():
    session = DummySession(
        [
            DummyResponse(
                {"results": [{"from": "ENSG1", "to": "P12345"}]},
                headers={"Link": '<https://rest.uniprot.org/next-page>; rel="next"'},
            ),
            DummyResponse({"results": [{"from": "ENSG2", "to": "Q67890"}]}),
        ]
    )

    results = fetch_id_mapping_results("job-123", session=session)

    assert results == [
        {"from": "ENSG1", "to": "P12345"},
        {"from": "ENSG2", "to": "Q67890"},
    ]
    assert session.calls[1][1] == "https://rest.uniprot.org/next-page"


def test_extract_uniprot_mapping_is_one_to_many_and_deduplicated():
    results = [
        {"from": "ENSG1", "to": "P12345"},
        {"from": "ENSG1", "to": {"primaryAccession": "Q67890"}},
        {"from": "ENSG1", "to": {"primaryAccession": "P12345"}},
        {"from": "ENSG2", "to": {"uniProtkbId": "GENE_HUMAN"}},
        {"from": "ENSG3", "to": {"id": "A0A000"}},
    ]

    mapping = extract_uniprot_mapping(results)

    assert mapping == {
        "ENSG1": ["P12345", "Q67890"],
        "ENSG2": ["GENE_HUMAN"],
        "ENSG3": ["A0A000"],
    }


def test_map_ids_to_uniprot_warns_for_unmapped_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "biastracker.annotations.uniprot_mapping.time.sleep",
        lambda seconds: None,
    )
    session = DummySession(
        [
            DummyResponse({"jobId": "job-123"}),
            DummyResponse({"results": []}),
            DummyResponse({"results": [{"from": "ENSG1", "to": "P12345"}]}),
        ]
    )

    with pytest.warns(RuntimeWarning, match="1 IDs were not mapped"):
        mapping = map_ids_to_uniprot(
            ["ENSG1", "ENSG2"],
            batch_size=500,
            cache_dir=tmp_path,
            session=session,
        )

    assert mapping == {"ENSG1": ["P12345"]}


def test_map_ids_to_uniprot_uses_cache(tmp_path):
    session = DummySession([])

    first = map_ids_to_uniprot(
        ["ENSG1"],
        cache_dir=tmp_path,
        session=DummySession(
            [
                DummyResponse({"jobId": "job-123"}),
                DummyResponse({"results": []}),
                DummyResponse({"results": [{"from": "ENSG1", "to": "P12345"}]}),
            ]
        ),
    )
    second = map_ids_to_uniprot(["ENSG1"], cache_dir=tmp_path, session=session)

    assert first == {"ENSG1": ["P12345"]}
    assert second == {"ENSG1": ["P12345"]}
    assert session.calls == []

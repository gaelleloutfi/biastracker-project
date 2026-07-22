"""Tests for biastracker.annotations.uniprot_go (offline)."""
from __future__ import annotations

import pytest

from biastracker.annotations.uniprot_go import (
    fetch_uniprot_go_tsv,
    load_uniprot_go_annotations,
    parse_uniprot_go_tsv,
)


class DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")


class DummySession:
    """Returns queued responses; records requests."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


_HEADER = (
    "Entry\tGene Ontology (biological process)\t"
    "Gene Ontology (cellular component)\tGene Ontology (molecular function)"
)
_TSV = (
    _HEADER + "\n"
    "P12345\tapoptotic process [GO:0006915]; cell cycle [GO:0007049]\t"
    "nucleus [GO:0005634]\tprotein binding [GO:0005515]\n"
    "Q99999\t\t\tRNA binding [GO:0003723]\n"
)


# ---------------------------------------------------------------------------
# parse_uniprot_go_tsv
# ---------------------------------------------------------------------------

def test_parse_extracts_terms_and_aspects():
    df = parse_uniprot_go_tsv(_TSV)
    p = df[df["primary_id"] == "P12345"]
    assert set(p["term_name"]) == {"apoptotic process", "cell cycle", "nucleus", "protein binding"}
    # term_id parsed from the [GO:...] suffix
    assert df.loc[df["term_name"] == "apoptotic process", "term_id"].iloc[0] == "GO:0006915"
    # aspect categories
    cats = dict(zip(df["term_name"], df["category"]))
    assert cats["apoptotic process"] == "biological_process"
    assert cats["nucleus"] == "cellular_component"
    assert cats["protein binding"] == "molecular_function"
    assert (df["source"] == "UniProt").all()


def test_parse_handles_empty_cells():
    df = parse_uniprot_go_tsv(_TSV)
    q = df[df["primary_id"] == "Q99999"]
    # Q99999 only has a molecular-function term; empty BP/CC cells are skipped.
    assert list(q["term_name"]) == ["RNA binding"]


def test_parse_empty_input_returns_typed_empty():
    df = parse_uniprot_go_tsv("")
    assert df.empty
    assert list(df.columns) == ["primary_id", "term_id", "term_name", "source", "category"]


def test_parse_missing_entry_column_raises():
    with pytest.raises(ValueError, match="Entry"):
        parse_uniprot_go_tsv("Gene Ontology (biological process)\nfoo [GO:1]\n")


def test_parse_deduplicates():
    tsv = _HEADER + "\nP1\tcell cycle [GO:0007049]; cell cycle [GO:0007049]\t\t\n"
    df = parse_uniprot_go_tsv(tsv)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# fetch + load (via DummySession)
# ---------------------------------------------------------------------------

def test_fetch_builds_accessions_request():
    sess = DummySession([DummyResponse(_TSV)])
    fetch_uniprot_go_tsv(["P12345", "Q99999"], aspects=("P", "F"), session=sess)
    method, url, kwargs = sess.calls[0]
    assert method == "GET"
    assert url.endswith("/uniprotkb/accessions")
    assert kwargs["params"]["accessions"] == "P12345,Q99999"
    assert kwargs["params"]["fields"] == "accession,go_p,go_f"
    assert kwargs["params"]["format"] == "tsv"


def test_fetch_rejects_empty_ids():
    with pytest.raises(ValueError, match="No accessions"):
        fetch_uniprot_go_tsv([], session=DummySession([]))


def test_unknown_aspect_raises():
    with pytest.raises(ValueError, match="Unknown GO aspect"):
        fetch_uniprot_go_tsv(["P1"], aspects=("Z",), session=DummySession([]))


def test_load_returns_annotationset(tmp_path):
    sess = DummySession([DummyResponse(_TSV)])
    ann = load_uniprot_go_annotations(
        ["P12345", "Q99999", "P12345"],  # duplicate collapsed
        session=sess, cache_dir=tmp_path,
    )
    assert ann.source == "UniProt"
    assert ann.metadata["n_input_ids"] == 2
    assert "GO:0005634" in set(ann.table["term_id"])
    assert "P12345" in ann.ids_for_term("nucleus")


def test_load_uses_cache_on_second_call(tmp_path):
    sess = DummySession([DummyResponse(_TSV)])  # only ONE response queued
    load_uniprot_go_annotations(["P12345", "Q99999"], session=sess, cache_dir=tmp_path)
    # Second call must hit the on-disk cache (no second HTTP call to pop).
    ann2 = load_uniprot_go_annotations(["P12345", "Q99999"], session=sess, cache_dir=tmp_path)
    assert len(sess.calls) == 1
    assert not ann2.table.empty


def test_refresh_bypasses_cache(tmp_path):
    # max_age_days=0 forces a re-fetch even though a fresh cache file exists.
    sess = DummySession([DummyResponse(_TSV), DummyResponse(_TSV)])
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path)
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path, max_age_days=0)
    assert len(sess.calls) == 2          # both calls hit the network


def test_stale_cache_is_refetched(tmp_path):
    import os
    sess = DummySession([DummyResponse(_TSV), DummyResponse(_TSV)])
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path)
    # Backdate the cache file to 40 days ago; default TTL is 30 days.
    for f in tmp_path.glob("go_*.csv"):
        old = os.stat(f).st_mtime - 40 * 86400
        os.utime(f, (old, old))
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path)
    assert len(sess.calls) == 2          # stale entry re-fetched


def test_ttl_none_never_expires(tmp_path):
    import os
    sess = DummySession([DummyResponse(_TSV)])   # only one response
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path)
    for f in tmp_path.glob("go_*.csv"):
        old = os.stat(f).st_mtime - 999 * 86400
        os.utime(f, (old, old))
    # max_age_days=None disables expiry -> still a cache hit, no 2nd call.
    load_uniprot_go_annotations(["P12345"], session=sess, cache_dir=tmp_path, max_age_days=None)
    assert len(sess.calls) == 1

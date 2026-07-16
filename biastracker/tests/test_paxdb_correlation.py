"""Tests for biastracker.analysis.paxdb."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biastracker.analysis.paxdb import (
    compute_spearman_correlation,
    load_paxdb_ppm,
    paxdb_abundance_agreement,
    prepare_paxdb_correlation_data,
)


def _ds(ids, expr) -> pd.DataFrame:
    return pd.DataFrame({"primary_id": ids, "mean_lfq": expr})


# ---------------------------------------------------------------------------
# load_paxdb_ppm
# ---------------------------------------------------------------------------

def test_load_paxdb_ppm(tmp_path):
    p = tmp_path / "paxdb.csv"
    pd.DataFrame({"primary_id": ["P1", "P2", "P1"], "paxdb_ppm": [10.0, 20.0, 30.0]}).to_csv(p, index=False)
    s = load_paxdb_ppm(p)
    assert s.loc["P2"] == 20.0
    assert s.loc["P1"] == 20.0          # duplicate accession -> mean(10, 30)


def test_load_paxdb_ppm_missing_file(tmp_path):
    s = load_paxdb_ppm(tmp_path / "nope.csv")
    assert s.empty


# ---------------------------------------------------------------------------
# prepare_paxdb_correlation_data
# ---------------------------------------------------------------------------

def test_prepare_matches_on_accession_partial_coverage():
    df = _ds(["P1", "P2", "P3"], [100.0, 1000.0, 10.0])
    paxdb = pd.Series({"P1": 5.0, "P2": 50.0}, name="paxdb_ppm")  # P3 not covered
    matched, meta = prepare_paxdb_correlation_data(df, "primary_id", paxdb)
    assert meta["n_input"] == 3
    assert meta["n_matched"] == 2
    assert set(matched["primary_id"]) == {"P1", "P2"}
    # log10 applied
    assert np.isclose(matched.loc[matched.primary_id == "P1", "paxdb_abundance"].iloc[0], np.log10(5.0))


def test_prepare_excludes_nonpositive_after_log():
    df = _ds(["P1", "P2", "P3"], [100.0, 0.0, -5.0])   # P2/P3 non-positive
    paxdb = pd.Series({"P1": 5.0, "P2": 50.0, "P3": 500.0}, name="paxdb_ppm")
    matched, meta = prepare_paxdb_correlation_data(df, "primary_id", paxdb, log_transform=True)
    assert meta["n_matched"] == 3
    assert meta["n_used"] == 1          # only P1 has a positive dataset abundance
    assert meta["n_excluded"] == 2
    assert list(matched["primary_id"]) == ["P1"]


def test_prepare_duplicate_dataset_accessions_averaged():
    df = _ds(["P1", "P1", "P2"], [100.0, 300.0, 10.0])
    paxdb = pd.Series({"P1": 5.0, "P2": 50.0}, name="paxdb_ppm")
    matched, meta = prepare_paxdb_correlation_data(df, "primary_id", paxdb, log_transform=False)
    assert meta["n_input"] == 2
    p1 = matched.loc[matched.primary_id == "P1", "dataset_abundance"].iloc[0]
    assert p1 == 200.0                   # mean(100, 300)


# ---------------------------------------------------------------------------
# compute_spearman_correlation
# ---------------------------------------------------------------------------

def test_spearman_known_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])   # perfectly monotonic
    res = compute_spearman_correlation(x, y)
    assert np.isclose(res["rho"], 1.0)
    assert res["n"] == 5
    assert res["message"] is None


def test_spearman_drops_invalid_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = compute_spearman_correlation(x, y)
    assert res["n"] == 3
    assert res["n_excluded"] == 2


def test_spearman_too_few_pairs():
    res = compute_spearman_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(res["rho"])
    assert res["n"] == 2
    assert "Too few" in res["message"]


def test_spearman_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        compute_spearman_correlation(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# paxdb_abundance_agreement (end-to-end, counts reported)
# ---------------------------------------------------------------------------

def test_agreement_reports_counts_and_correlation():
    df = _ds(
        ["P1", "P2", "P3", "P4", "P5"],
        [10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    paxdb = pd.Series(
        {"P1": 1.0, "P2": 10.0, "P3": 100.0, "P4": 1000.0, "P5": 10000.0},
        name="paxdb_ppm",
    )
    res = paxdb_abundance_agreement(df, "primary_id", paxdb)
    assert res.n_input == 5
    assert res.n_matched == 5
    assert res.n_used == 5
    assert np.isclose(res.rho, 1.0)
    assert res.matched_fraction == 1.0


def test_agreement_too_few_matches_message():
    df = _ds(["P1", "P2", "P3"], [10.0, 100.0, 1000.0])
    paxdb = pd.Series({"P1": 1.0, "Z9": 10.0}, name="paxdb_ppm")  # only P1 matches
    res = paxdb_abundance_agreement(df, "primary_id", paxdb)
    assert res.n_matched == 1
    assert np.isnan(res.rho)
    assert res.message is not None

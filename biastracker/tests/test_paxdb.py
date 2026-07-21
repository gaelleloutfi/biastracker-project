"""Tests for biastracker.analysis.paxdb (PaxDb abundance as a feature)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biastracker.dataset import BiasDataset
from biastracker.analysis.paxdb import (
    PAXDB_FEATURE,
    add_paxdb_abundance,
    load_paxdb_ppm,
)


def _ds(ids, ppm_col=None, contaminant=None) -> BiasDataset:
    table = pd.DataFrame({
        "primary_id": ids,
        "level": ["protein"] * len(ids),
        "sequence": ["SEQ"] * len(ids),
    })
    if contaminant is not None:
        table["is_contaminant"] = contaminant
    return BiasDataset(name="d", table=table, level="protein")


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
    assert load_paxdb_ppm(tmp_path / "nope.csv").empty


def test_load_paxdb_ppm_bad_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"acc": ["P1"], "ppm": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="primary_id"):
        load_paxdb_ppm(p)


# ---------------------------------------------------------------------------
# add_paxdb_abundance
# ---------------------------------------------------------------------------

def test_add_paxdb_abundance_log10_and_matching():
    ds = _ds(["P1", "P2", "P3"])
    paxdb = pd.Series({"P1": 100.0, "P2": 1000.0}, name="paxdb_ppm")  # P3 unmatched
    out = add_paxdb_abundance(ds, paxdb)
    col = out.table[PAXDB_FEATURE]
    assert np.isclose(col.iloc[0], 2.0)      # log10(100)
    assert np.isclose(col.iloc[1], 3.0)      # log10(1000)
    assert np.isnan(col.iloc[2])             # unmatched -> NaN
    assert out.metadata["paxdb_added"] is True
    assert out.metadata["paxdb_n_matched"] == 2


def test_add_paxdb_abundance_raw_when_no_log():
    ds = _ds(["P1"])
    out = add_paxdb_abundance(ds, pd.Series({"P1": 42.0}), log10=False)
    assert out.table[PAXDB_FEATURE].iloc[0] == 42.0


def test_add_paxdb_abundance_nonpositive_becomes_nan():
    ds = _ds(["P1", "P2"])
    out = add_paxdb_abundance(ds, pd.Series({"P1": 0.0, "P2": -5.0}))
    assert out.table[PAXDB_FEATURE].isna().all()


def test_add_paxdb_abundance_excludes_contaminants_by_default():
    ds = _ds(["P1", "P2"], contaminant=[False, True])
    paxdb = pd.Series({"P1": 100.0, "P2": 1000.0})
    out = add_paxdb_abundance(ds, paxdb)
    col = out.table[PAXDB_FEATURE]
    assert np.isclose(col.iloc[0], 2.0)
    assert np.isnan(col.iloc[1])             # contaminant excluded
    assert out.metadata["paxdb_n_matched"] == 1


def test_add_paxdb_abundance_can_keep_contaminants():
    ds = _ds(["P1", "P2"], contaminant=[False, True])
    paxdb = pd.Series({"P1": 100.0, "P2": 1000.0})
    out = add_paxdb_abundance(ds, paxdb, exclude_contaminants=False)
    assert np.isclose(out.table[PAXDB_FEATURE].iloc[1], 3.0)


def test_add_paxdb_abundance_no_matches_still_adds_column():
    ds = _ds(["P1", "P2"])
    out = add_paxdb_abundance(ds, pd.Series({"Z9": 100.0}))
    assert PAXDB_FEATURE in out.table.columns
    assert out.table[PAXDB_FEATURE].isna().all()
    assert out.metadata["paxdb_n_matched"] == 0


def test_add_paxdb_abundance_preserves_dataset_fields():
    ds = _ds(["P1"])
    out = add_paxdb_abundance(ds, pd.Series({"P1": 100.0}))
    assert out.level == "protein"
    assert out.id_col == "primary_id"
    assert "sequence" in out.table.columns   # BiasDataset invariants intact

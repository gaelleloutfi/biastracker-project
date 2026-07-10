import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from protperties.io_maxquant_pg import from_maxquant_proteingroups

def _make_tiny_proteingroups(tmp_path: Path) -> Path:
    data = {
        "Protein IDs": ["sp|P12345|NAME;tr|Q9XXX1|NAME", "sp|P67890|NAME", "sp|Q11111|NAME", "sp|P44444|NAME", "sp|P55555|NAME"],
        "LFQ intensity 1": [0.0, 1000.0, 500.0, 2000.0, 1000.0],
        "LFQ intensity 2": [2000.0, 1000.0, 0.0, 2000.0, 1000.0],
        "Reverse": ["", "", "+", "", ""],
        "Potential contaminant": ["", "", "", "+", ""],
        "Only identified by site": ["", "", "", "", " + "]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "proteinGroups.txt"
    df.to_csv(p, sep="\t", index=False)
    return p

def test_from_maxquant_proteingroups_mean_split(tmp_path):
    p = _make_tiny_proteingroups(tmp_path)
    # Testing split_ids=True (contaminants kept as ID-only by default)
    df = from_maxquant_proteingroups(p, drop_zeros=True, statistic="mean", split_ids=True, fetch_sequences=False)

    # Row 0 splits into 2 (P12345, Q9XXX1). Row 1 is P67890. Row 3 (P44444) is a
    # contaminant, now kept as ID-only. Reverse (Q11111) and OIBS (S33333) filtered.
    assert len(df) == 4
    assert "primary_id" in df.columns
    assert "mean_lfq" in df.columns
    assert "expression" in df.columns
    assert "n_samples_used" in df.columns
    assert "lfq_statistic" in df.columns
    assert "is_contaminant" in df.columns
    assert "Protein IDs" in df.columns

    assert df.loc[0, "primary_id"] == "P12345" # First part of split
    assert df.loc[0, "mean_lfq"] == 2000.0 # (NaN, 2000) -> 2000, drop_zeros=True
    assert df.loc[0, "n_samples_used"] == 1
    assert df.loc[0, "lfq_statistic"] == "mean"

    primary_ids = df["primary_id"].tolist()
    assert "P12345" in primary_ids
    assert "Q9XXX1" in primary_ids
    assert "P67890" in primary_ids
    # Contaminant retained, flagged, and never marked reverse/OIBS
    assert "P44444" in primary_ids
    contam = df[df["primary_id"] == "P44444"]
    assert bool(contam["is_contaminant"].iloc[0]) is True
    # Non-contaminants are flagged False
    assert bool(df.loc[df["primary_id"] == "P12345", "is_contaminant"].iloc[0]) is False


def test_from_maxquant_proteingroups_median_no_split(tmp_path):
    p = _make_tiny_proteingroups(tmp_path)
    # split_ids=False (default), drop_zeros=False
    df = from_maxquant_proteingroups(p, drop_zeros=False, statistic="median", split_ids=False, fetch_sequences=False)

    # First-ID per group: P12345, P67890, and contaminant P44444 (ID-only).
    assert len(df) == 3

    # P12345: median(0.0, 2000.0) = 1000.0
    # First id extracted
    assert df.iloc[0]["primary_id"] == "P12345"
    assert df.iloc[0]["median_lfq"] == 1000.0
    assert df.iloc[0]["n_samples_used"] == 2
    assert df.iloc[0]["lfq_statistic"] == "median"

    # Contaminant kept as ID-only with no sequence.
    contam = df[df["primary_id"] == "P44444"]
    assert len(contam) == 1
    assert bool(contam["is_contaminant"].iloc[0]) is True
    assert not contam["has_sequence"].iloc[0]


def test_from_maxquant_proteingroups_drop_contaminants(tmp_path):
    p = _make_tiny_proteingroups(tmp_path)
    df = from_maxquant_proteingroups(
        p, statistic="mean", split_ids=False, keep_contaminants=False, fetch_sequences=False
    )
    # P44444 (contaminant) dropped; Reverse/OIBS also gone -> P12345, P67890.
    assert "P44444" not in df["primary_id"].tolist()
    assert set(df["primary_id"]) == {"P12345", "P67890"}
    assert not df["is_contaminant"].any()


def test_from_maxquant_prefers_majority_protein_ids(tmp_path):
    # When both columns exist, the loader uses the first ID of 'Majority protein IDs'.
    data = {
        "Protein IDs": ["sp|P12345|N;sp|Q9XXX1|N", "sp|P67890|N;sp|O11111|N"],
        "Majority protein IDs": ["sp|Q9XXX1|N", "sp|O11111|N"],
        "LFQ intensity 1": [1000.0, 2000.0],
        "LFQ intensity 2": [1000.0, 2000.0],
    }
    p = tmp_path / "proteinGroups.txt"
    pd.DataFrame(data).to_csv(p, sep="\t", index=False)

    df = from_maxquant_proteingroups(p, fetch_sequences=False)
    assert len(df) == 2  # one row per group, no explosion
    assert df.iloc[0]["primary_id"] == "Q9XXX1"
    assert df.iloc[1]["primary_id"] == "O11111"

def test_from_maxquant_proteingroups_missing_columns(tmp_path):
    data = {
        "Protein IDs": ["A"],
        "Intensity 1": [100.0]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "bad.txt"
    df.to_csv(p, sep="\t", index=False)
    
    with pytest.raises(ValueError, match="No columns found with prefix 'LFQ intensity '"):
        from_maxquant_proteingroups(p)

def test_from_maxquant_proteingroups_on_test_data():
    path = Path("test_data/proteinGroups.txt")
    if not path.exists():
        pytest.skip(f"{path} not found")
        
    raw_df = pd.read_csv(path, sep="\t", low_memory=False)
    out = from_maxquant_proteingroups(path, drop_zeros=True, statistic="median", fetch_sequences=False)
    
    assert len(out) > 0, "Loader returned empty dataframe"
    assert "primary_id" in out.columns
    assert "expression" in out.columns
    assert "n_samples_used" in out.columns
    assert "lfq_statistic" in out.columns
    
    if "Reverse" in out.columns:
        assert not (out["Reverse"] == "+").any()
    if "Only identified by site" in out.columns:
        assert not (out["Only identified by site"].astype(str).str.strip() == "+").any()
    # Contaminants are now retained (ID-only) and flagged rather than dropped.
    if "Potential contaminant" in out.columns:
        contam_rows = out["Potential contaminant"].fillna("").astype(str).str.strip() == "+"
        assert (out.loc[contam_rows, "is_contaminant"]).all()
        assert not out.loc[contam_rows, "has_sequence"].any()
    
    lfq_cols = [c for c in raw_df.columns if c.startswith("LFQ intensity ")]
    assert len(lfq_cols) > 0
    
    assert (out["n_samples_used"] >= 0).all()
    assert (out["n_samples_used"] <= len(lfq_cols)).all()
    
    if "has_sequence" in out.columns:
        assert not out["has_sequence"].any()
    
    if hasattr(out, "attrs") and "level" in out.attrs:
        assert out.attrs["level"] == "protein"

import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from protperties.io_maxquant_pg import from_maxquant_proteingroups

def _make_tiny_proteingroups(tmp_path: Path) -> Path:
    data = {
        "Protein IDs": ["sp|P12345|NAME;tr|Q9XXX1|NAME", "sp|P67890|NAME", "sp|Q11111|NAME", "sp|R22222|NAME", "sp|S33333|NAME"],
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
    # Testing split_ids=True
    df = from_maxquant_proteingroups(p, drop_zeros=True, statistic="mean", split_ids=True, fetch_sequences=False)
    
    assert len(df) == 3  # Row 1 splits into 2 (P12345, Q9XXX1). Row 2 is P67890. Others filtered.
    assert "primary_id" in df.columns
    assert "mean_lfq" in df.columns
    assert "expression" in df.columns
    assert "n_samples_used" in df.columns
    assert "lfq_statistic" in df.columns
    assert "Protein IDs" in df.columns
    
    assert df.loc[0, "primary_id"] == "P12345" # First part of split
    assert df.loc[0, "mean_lfq"] == 2000.0 # (NaN, 2000) -> 2000, drop_zeros=True
    assert df.loc[0, "n_samples_used"] == 1
    assert df.loc[0, "lfq_statistic"] == "mean"
    
    # Q9XXX1 is on index 0 too because pandas explode keeps the original index, so let's reset or just check values
    # Actually explode keeps the original index, so we should check values
    primary_ids = df["primary_id"].tolist()
    assert "P12345" in primary_ids
    assert "Q9XXX1" in primary_ids
    assert "P67890" in primary_ids

def test_from_maxquant_proteingroups_median_no_split(tmp_path):
    p = _make_tiny_proteingroups(tmp_path)
    # split_ids=False, drop_zeros=False
    df = from_maxquant_proteingroups(p, drop_zeros=False, statistic="median", split_ids=False, fetch_sequences=False)
    
    assert len(df) == 2
    
    # P12345: median(0.0, 2000.0) = 1000.0
    # First id extracted
    assert df.iloc[0]["primary_id"] == "P12345"
    assert df.iloc[0]["median_lfq"] == 1000.0
    assert df.iloc[0]["n_samples_used"] == 2
    assert df.iloc[0]["lfq_statistic"] == "median"

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
        
    out = from_maxquant_proteingroups(path, drop_zeros=True, statistic="median", fetch_sequences=False)
    
    assert len(out) >= 0
    assert "primary_id" in out.columns
    assert "expression" in out.columns
    assert "n_samples_used" in out.columns
    assert "lfq_statistic" in out.columns
    
    if "has_sequence" in out.columns:
        assert not out["has_sequence"].any()
    
    if hasattr(out, "attrs") and "level" in out.attrs:
        assert out.attrs["level"] == "protein"

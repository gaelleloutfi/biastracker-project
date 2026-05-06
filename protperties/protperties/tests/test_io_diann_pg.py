import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from protperties.io_diann_pg import from_diann_pg_matrix

def _make_tiny_diann_pg(tmp_path: Path) -> Path:
    data = {
        "Protein.Group": ["P12345", "Q9XXX1", "O11111"],
        "Protein.Ids": ["P12345;P12346", "Q9XXX1", "O11111"],
        "Protein.Names": ["Name1", "Name2", "Name3"],
        "Genes": ["Gene1", "Gene2", "Gene3"],
        "First.Protein.Description": ["Desc1", "Desc2", "Desc3"],
        "C:\\sample1.raw": [0.0, 1000.0, 500.0],
        "C:\\sample2.raw": [2000.0, 1000.0, 0.0]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "report.pg_matrix.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p

def test_from_diann_pg_matrix_mean(tmp_path):
    p = _make_tiny_diann_pg(tmp_path)
    df = from_diann_pg_matrix(p, drop_zeros=True, statistic="mean", fetch_sequences=False)
    
    assert len(df) == 3
    assert "primary_id" in df.columns
    assert "mean_lfq" in df.columns
    assert "expression" in df.columns
    assert "n_samples_used" in df.columns
    assert "lfq_statistic" in df.columns
    assert "Protein.Ids" in df.columns  # Original column kept
    
    assert df.loc[0, "primary_id"] == "P12345"
    assert df.loc[1, "primary_id"] == "Q9XXX1"
    
    # 0 -> NaN for drop_zeros=True
    # P12345: [NaN, 2000.0] -> mean = 2000.0, n=1
    assert df.loc[0, "mean_lfq"] == 2000.0
    assert df.loc[0, "expression"] == 2000.0
    assert df.loc[0, "n_samples_used"] == 1
    assert df.loc[0, "lfq_statistic"] == "mean"
    
    # Q9XXX1: [1000.0, 1000.0] -> mean = 1000.0, n=2
    assert df.loc[1, "mean_lfq"] == 1000.0
    assert df.loc[1, "n_samples_used"] == 2

def test_from_diann_pg_matrix_median(tmp_path):
    p = _make_tiny_diann_pg(tmp_path)
    # drop_zeros = False
    df = from_diann_pg_matrix(p, drop_zeros=False, statistic="median", fetch_sequences=False)
    
    # P12345: [0.0, 2000.0] -> median = 1000.0, n=2
    assert df.loc[0, "median_lfq"] == 1000.0
    assert df.loc[0, "expression"] == 1000.0
    assert df.loc[0, "n_samples_used"] == 2
    assert df.loc[0, "lfq_statistic"] == "median"

def test_from_diann_pg_matrix_explicit_samples(tmp_path):
    p = _make_tiny_diann_pg(tmp_path)
    # Use only one sample column
    df = from_diann_pg_matrix(p, sample_cols=["C:\\sample1.raw"], drop_zeros=True, statistic="mean")
    
    # P12345: [NaN] -> mean = NaN, n=0
    assert pd.isna(df.loc[0, "mean_lfq"])
    assert df.loc[0, "n_samples_used"] == 0
    
    # Q9XXX1: [1000.0] -> mean = 1000.0, n=1
    assert df.loc[1, "mean_lfq"] == 1000.0
    assert df.loc[1, "n_samples_used"] == 1

def test_from_diann_pg_matrix_missing_columns(tmp_path):
    data = {
        "SomeColumn": ["A", "B", "C"],
        "Intensity": [100, 200, 300]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "bad.tsv"
    df.to_csv(p, sep="\t", index=False)
    
    with pytest.raises(ValueError, match="None of the candidate protein ID columns found"):
        from_diann_pg_matrix(p)

def test_from_diann_pg_matrix_strict_desc_rule(tmp_path):
    data = {
        "Protein.Group": ["P12345", "Q9XXX1"],
        "Protein.Ids": ["P12345", "Q9XXX1"],
        "First.Protein.Description": ["Desc1", "Desc2"],
        "SomeStringCol": ["A", "B"], # should be forced to NaN as it's after Description
        "C:\\sample1.raw": [100.0, 200.0]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "strict.tsv"
    df.to_csv(p, sep="\t", index=False)
    
    res = from_diann_pg_matrix(p, drop_zeros=True, statistic="mean", fetch_sequences=False)
    
    assert len(res) == 2
    # Because SomeStringCol is after First.Protein.Description, it's treated as a sample and coerced to NaN
    # C:\sample1.raw has [100.0, 200.0]. 
    # n_samples_used should be 1 for each row (since SomeStringCol is NaN)
    assert res.loc[0, "n_samples_used"] == 1
    assert res.loc[0, "mean_lfq"] == 100.0
    assert res.loc[1, "mean_lfq"] == 200.0

def test_from_diann_pg_matrix_fallback(tmp_path):
    # No First.Protein.Description
    data = {
        "Protein.Group": ["P12345", "Q9XXX1"],
        "Protein.Ids": ["P12345", "Q9XXX1"],
        "SomeMetadata": ["A", "B"], # string, should be skipped by the 80% rule
        "C:\\sample1.raw": [100.0, 200.0]
    }
    df = pd.DataFrame(data)
    p = tmp_path / "fallback.tsv"
    df.to_csv(p, sep="\t", index=False)
    
    res = from_diann_pg_matrix(p, drop_zeros=True, statistic="mean", fetch_sequences=False)
    
    assert len(res) == 2
    # SomeMetadata is skipped, C:\sample1.raw is kept.
    assert res.loc[0, "n_samples_used"] == 1
    assert res.loc[0, "mean_lfq"] == 100.0

def test_from_diann_pg_matrix_on_test_data():
    import pytest
    path = Path("test_data/report.pg_matrix.tsv")
    if not path.exists():
        pytest.skip(f"{path} not found")
        
    out = from_diann_pg_matrix(path, fetch_sequences=False)
    
    assert len(out) > 0
    assert "primary_id" in out.columns
    assert "expression" in out.columns
    assert "n_samples_used" in out.columns
    
    if "has_sequence" in out.columns:
        assert not out["has_sequence"].any()
        
    if hasattr(out, "attrs") and "level" in out.attrs:
        assert out.attrs["level"] == "protein"

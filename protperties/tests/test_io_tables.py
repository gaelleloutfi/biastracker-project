
import pandas as pd #to build tiny fake DIA-NN tables
import numpy as np
from pathlib import Path #to write a temporary parquet file

#now we import  the functions/classes we want to test from io_tables.py
from protperties.io_tables import (
    from_diann_parquet,
    summarize_miscleavage,
    FilterConfig,
    DedupConfig,
)

def _make_tiny_parquet(tmp_path: Path) -> Path: #for each testm pytest will give us a fresh temporary directory
    # we create very small Minimal DIA-NN-like dataframe with 3 rows: 
    # - 2 peptides "PEPTIDEK" and "ABCDEFGHIK"
    # - row 2 has Q.value = 0.02 (>0.01) on purpose to test the filtering
    # the same precursor "PEPTIDEK/2" appears in both R1 and R2 to test grouping/dedup
    df = pd.DataFrame({
        "Run": ["R1","R1","R2"],
        "Precursor.Id": ["PEPTIDEK/2", "ACDEFGHIK/3", "PEPTIDEK/2"],
        "Stripped.Sequence": ["PEPTIDEK", "ACDEFGHIK", "PEPTIDEK"],
        "Precursor.Charge": [2, 3, 2],
        "PEP": [0.005, 0.02, 0.003],          # one above 0.01 to test filter
        #"Global.Q.Value": [0.005, 0.005, 0.005],
        #"Lib.Q.Value": [0.005, 0.005, 0.005],
        "Decoy": [0, 0, 0],
        "Precursor.Normalised": [1e6, 5e5, 1.2e6],
        "Precursor.Quantity": [9e5, 4e5, 1.0e6],
    })
    p = tmp_path / "tiny.parquet" #this tiny file is saved in the temporary folder as tiny.parquet
    df.to_parquet(p, index=False)
    return p

def test_from_diann_parquet_filters_and_props(tmp_path):
    # checks that filtering works, properties are computed, lengths are sane,  missed cleavages for PEPTIDEK
    path = _make_tiny_parquet(tmp_path)
    # Use default filters: Q.Value <= 0.01 should remove the middle row
    out = from_diann_parquet(path)
    # Expect 2 rows after filtering
    assert len(out) == 2
    assert {"length","mw","pi","gravy","instability","aliphatic_index","missed_cleavages", "charge_at_pH"} <= set(out.columns)
    # Check that sequences computed give non-zero length
    assert (out["length"] > 0).all()
    # Missed cleavages for "PEPTIDEK" should be 0 (no internal K/R)
    assert (out.loc[out["Stripped.Sequence"]=="PEPTIDEK","missed_cleavages"] == 0).all()
    # The q values and quantity columns should have been dropped from the output
    for col in ("Q.Value","Global.Q.Value","Lib.Q.Value","Precursor.Quantity"):
        assert col not in out.columns
        
def test_dedup_max_on_normalised(tmp_path):
    # check for healthy deduplication
    path = _make_tiny_parquet(tmp_path)
    # Two rows share Run=R1 and Run=R2 for the same Precursor.Id "PEPTIDEK/2" across runs.
    # Here we test dedup by (Run, Precursor.Id) and pick max Precursor.Normalised.
    out = from_diann_parquet(
        path,
        dedup=DedupConfig(keys=("Run","Precursor.Id"), pick="max", on_col="Precursor.Normalised")
    )
    # After filtering, there are two rows for PEPTIDEK/2: R1 and R2 -> dedup keeps both because keys differ by Run
    assert len(out) == 2
    assert set(out["Run"]) == {"R1","R2"}

def test_summarize_miscleavage(tmp_path):
    #checks the healthy computation of digestion-quality metrics per run
    path = _make_tiny_parquet(tmp_path)
    df = from_diann_parquet(path)
    summary = summarize_miscleavage(df, groupby=("Run",))
    # Should have two groups (R1 and R2) after filtering
    assert set(summary["Run"]) == {"R1","R2"}
    assert {"n","percent_miscleavage","mean_missed_cleavages"} <= set(summary.columns)

def test_protein_id_normalization(tmp_path):
    """Test that protein IDs are normalized to new columns WITHOUT overwriting originals."""
    df = pd.DataFrame({
        "Run": ["R1", "R2", "R3"],
        "Precursor.Id": ["PEP1/2", "PEP2/2", "PEP3/2"],
        "Stripped.Sequence": ["PEPTIDEK", "ABCDEFK", "TESTPEP"],
        "Precursor.Charge": [2, 2, 2],
        "Protein.Ids": ["sp|P12345|PROT", "tr|Q9XXX1|TREMBL", "O43663-2"],
        "Protein.Group": ["sp|P12345|PROT", "Q9XXX1", "O43663"],
        "PEP": [0.005, 0.003, 0.001],
        "Decoy": [0, 0, 0],
        "Precursor.Normalised": [1e6, 2e6, 3e6],
    })
    path = tmp_path / "normalized.parquet"
    df.to_parquet(path, index=False)
    
    out = from_diann_parquet(path)
    
    # Check that all protein IDs are normalized in NEW columns
    assert len(out) == 3
    
    # CRITICAL: Original columns must remain unchanged
    assert out.iloc[0]["Protein.Ids"] == "sp|P12345|PROT"
    assert out.iloc[1]["Protein.Ids"] == "tr|Q9XXX1|TREMBL"
    assert out.iloc[2]["Protein.Ids"] == "O43663-2"
    
    assert out.iloc[0]["Protein.Group"] == "sp|P12345|PROT"
    assert out.iloc[1]["Protein.Group"] == "Q9XXX1"
    assert out.iloc[2]["Protein.Group"] == "O43663"
    
    # Check that NEW normalized columns are created
    assert "protein_primary_id" in out.columns
    assert "protein_group_primary_id" in out.columns
    
    assert out.iloc[0]["protein_primary_id"] == "P12345"  # normalized from sp|P12345|PROT
    assert out.iloc[1]["protein_primary_id"] == "Q9XXX1"  # normalized from tr|Q9XXX1|TREMBL
    assert out.iloc[2]["protein_primary_id"] == "O43663-2"  # isoform preserved
    
    assert out.iloc[0]["protein_group_primary_id"] == "P12345"  # normalized from sp|P12345|PROT
    assert out.iloc[1]["protein_group_primary_id"] == "Q9XXX1"  # already normalized
    assert out.iloc[2]["protein_group_primary_id"] == "O43663"  # already normalized

def test_semicolon_separated_ids_preserved(tmp_path):
    """Test that semicolon-separated protein IDs remain unchanged in original column."""
    df = pd.DataFrame({
        "Run": ["R1", "R2"],
        "Precursor.Id": ["PEP1/2", "PEP2/2"],
        "Stripped.Sequence": ["PEPTIDEK", "ABCDEFK"],
        "Precursor.Charge": [2, 2],
        # Semicolon-separated IDs (common in DIA-NN)
        "Protein.Ids": ["P12345;Q9XXX1;O43663", "sp|P00001|PROT1;tr|P00002|PROT2"],
        "Protein.Group": ["P12345;Q9XXX1", "P00001"],
        "PEP": [0.005, 0.003],
        "Decoy": [0, 0],
        "Precursor.Normalised": [1e6, 2e6],
    })
    path = tmp_path / "semicolon.parquet"
    df.to_parquet(path, index=False)
    
    out = from_diann_parquet(path)
    
    # CRITICAL: Original semicolon-separated IDs must be preserved
    assert out.iloc[0]["Protein.Ids"] == "P12345;Q9XXX1;O43663"
    assert out.iloc[1]["Protein.Ids"] == "sp|P00001|PROT1;tr|P00002|PROT2"
    
    assert out.iloc[0]["Protein.Group"] == "P12345;Q9XXX1"
    assert out.iloc[1]["Protein.Group"] == "P00001"
    
    # Check normalized columns extract the first ID
    assert "protein_primary_id" in out.columns
    assert out.iloc[0]["protein_primary_id"] == "P12345"  # First ID from semicolon list
    assert out.iloc[1]["protein_primary_id"] == "P00001"  # Normalized from sp|P00001|PROT1

def test_from_diann_parquet_on_test_data():
    import pytest
    path = Path("test_data/report.parquet")
    if not path.exists():
        pytest.skip(f"{path} not found")
        
    out = from_diann_parquet(path)
    
    assert len(out) > 0
    assert "Stripped.Sequence" in out.columns
    assert "Precursor.Charge" in out.columns
    assert "Protein.Ids" in out.columns
    
    assert "length" in out.columns
    assert "missed_cleavages" in out.columns
    
    if hasattr(out, "attrs") and "level" in out.attrs:
        assert out.attrs["level"] == "peptide"

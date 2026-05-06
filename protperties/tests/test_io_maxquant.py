import pandas as pd
from pathlib import Path
from protperties.io_maxquant import (
    from_maxquant_evidence,
    MaxQuantFilterConfig,
)

def _make_tiny_evidence(tmp_path: Path) -> Path:
    """ 
    Create a minimal synthetic maxquant evidence.txt for testing.
    columns included are: 
    Raw file, Sequence, Modified sequence, Charge, Leading razor protein,
    Protein names, Gene names, Intensity, PEP, Reverse.
    """
    df = pd.DataFrame({
        "Raw file": ["run1","run1","run2","run2","run2","run3"],
        "Sequence": ["PEPTIDEK", "PEPTIDEK", "AAAAAADLANR", "AAAAAADLANR", "AAAAAADLANR", "CONTAMPEPT"],
        "Modified sequence": ["PEPTIDEK", "PEPTIDEK", "AAAAAADLANR", "AAAAAADLANR", "AAAAAADLANR", "CONTAMPEPT"],
        "Charge": [2, 2, 2, 2, 2, 2],
        "Leading proteins": ["P12345", "P12345", "Q9JHS4", "REV_Q9JHS4", "P12345", "CON__P123"],
        "Leading razor protein": ["P12345", "P12345", "Q9JHS4", "Q9JHS4", "REV_P12345", "CON__P123"],
        "Protein names": ["TEST_PROT", "TEST_PROT", "CLPX_MOUSE", "CLPX_MOUSE", "TEST_PROT", "CONT"],
        "Gene names":["Test1", "Test1", "Clpx", "Clpx", "Test1", "CONT"],
        "Intensity": [1000.0,500.0,2000.0, 1000.0, 1000.0, 1000.0],
        "PEP": [0.005,0.02,0.003, 0.005, 0.005, 0.005],
        "Reverse": ["","", "1", "", "", ""],
        "Potential contaminant": ["", "", "", "", "", "+"],
    })
    path = tmp_path / "evidence.txt"
    df.to_csv(path, sep="\t", index=False)
    return path

def test_from_maxquant_basic_filters_and_mapping(tmp_path): 
    """
    Test that : 
    - PEP filtering works (PEP<=0.01)
    - reverse entries are removed
    - column mapping works
    - a precursor id is created from modified sequence + charge
    - peptide properties are computed
    """

    path = _make_tiny_evidence(tmp_path)
    out = from_maxquant_evidence(path)

    # pep filter removes row 2
    # reverse removes row 3
    # leading proteins rev removes row 4
    # leading razor protein rev removes row 5
    # row 6 (contaminant) is NOT removed
    # so row 1 and row 6 remain
    assert len(out) == 2
    row = out.iloc[0]
    #mapping checks

    assert row["Run"] == "run1"
    assert row["Stripped.Sequence"] == "PEPTIDEK"
    assert row["Precursor.Charge"] == 2
    assert row["Protein.Ids"] == "P12345"
    assert row["Protein.Names"] == "TEST_PROT"
    assert row["Genes"] == "Test1"
    assert row["Precursor.Normalised"] == 1000.0
    assert row["PEP"] == 0.005
    assert "Q.Value" not in out.columns

    #the precursor id must be constructed as sequence + charge
    assert row["Precursor.Id"] == "PEPTIDEK2"
    # basic properties must exist and be valid
    expected_props = {
        "length", "mw", "pi", "gravy", "instability",
        "aromaticity", "aliphatic_index", 
        "ext_reduced", "ext_cystine", "missed_cleavages",
        "charge_at_pH",
    }
    assert expected_props <= set(out.columns)
    assert row["length"] > 0
    #charge at ph must be a float in a reasonable range
    assert isinstance(row["charge_at_pH"], float)
    assert -10 < row["charge_at_pH"] < 10

def test_disable_pep_filter(tmp_path):
    """
    Test that disabling PEP filtering keeps rows with high PEP.
    """
    path = _make_tiny_evidence(tmp_path)
    cfg = MaxQuantFilterConfig(pep_max = None)
    out = from_maxquant_evidence(path, filters=cfg)

    # reverse filtering still removes the 3rd, 4th, and 5th rows
    # but since PEP filtering is off, row 2 remains alongside row 1 and row 6
    assert len(out) == 3
    assert set(out["Run"]) == {"run1", "run3"}

def test_protein_id_normalization(tmp_path):
    """
    Test that protein IDs are normalized to new column WITHOUT overwriting original.
    """
    df = pd.DataFrame({
        "Raw file": ["run1", "run2", "run3"],
        "Sequence": ["PEPTIDEK", "AAAAAAK", "TESTPEP"],
        "Modified sequence": ["PEPTIDEK", "AAAAAAK", "TESTPEP"],
        "Charge": [2, 2, 2],
        "Leading razor protein": ["sp|P12345|PROTEIN_NAME", "tr|Q9XXX1|TREMBL", "O43663-2"],
        "Protein names": ["PROT1", "PROT2", "PROT3"],
        "Gene names": ["Gene1", "Gene2", "Gene3"],
        "Intensity": [1000.0, 2000.0, 3000.0],
        "PEP": [0.005, 0.003, 0.001],
        "Reverse": ["", "", ""],
    })
    path = tmp_path / "evidence_normalized.txt"
    df.to_csv(path, sep="\t", index=False)
    
    out = from_maxquant_evidence(path)
    
    # Check that all protein IDs are normalized in NEW column
    assert len(out) == 3
    
    # CRITICAL: Original column must remain unchanged
    assert out.iloc[0]["Protein.Ids"] == "sp|P12345|PROTEIN_NAME"
    assert out.iloc[1]["Protein.Ids"] == "tr|Q9XXX1|TREMBL"
    assert out.iloc[2]["Protein.Ids"] == "O43663-2"
    
    # Check that NEW normalized column is created
    assert "protein_primary_id" in out.columns
    assert out.iloc[0]["protein_primary_id"] == "P12345"  # normalized from sp|P12345|PROTEIN_NAME
    assert out.iloc[1]["protein_primary_id"] == "Q9XXX1"  # normalized from tr|Q9XXX1|TREMBL
    assert out.iloc[2]["protein_primary_id"] == "O43663-2"  # isoform preserved

def test_semicolon_separated_ids_preserved(tmp_path):
    """Test that semicolon-separated protein IDs remain unchanged in original column."""
    df = pd.DataFrame({
        "Raw file": ["run1", "run2"],
        "Sequence": ["PEPTIDEK", "AAAAAAK"],
        "Modified sequence": ["PEPTIDEK", "AAAAAAK"],
        "Charge": [2, 2],
        # Semicolon-separated IDs
        "Leading razor protein": ["P12345;Q9XXX1;O43663", "sp|P00001|PROT1;tr|P00002|PROT2"],
        "Protein names": ["PROT1", "PROT2"],
        "Gene names": ["Gene1", "Gene2"],
        "Intensity": [1000.0, 2000.0],
        "PEP": [0.005, 0.003],
        "Reverse": ["", ""],
    })
    path = tmp_path / "evidence_semicolon.txt"
    df.to_csv(path, sep="\t", index=False)
    
    out = from_maxquant_evidence(path)
    
    # CRITICAL: Original semicolon-separated IDs must be preserved
    assert out.iloc[0]["Protein.Ids"] == "P12345;Q9XXX1;O43663"
    assert out.iloc[1]["Protein.Ids"] == "sp|P00001|PROT1;tr|P00002|PROT2"
    
    # Check normalized column extracts the first ID
    assert "protein_primary_id" in out.columns
    assert out.iloc[0]["protein_primary_id"] == "P12345"  # First ID from semicolon list
    assert out.iloc[1]["protein_primary_id"] == "P00001"  # Normalized from sp|P00001|PROT1

def test_from_maxquant_evidence_on_test_data():
    import pytest
    path = Path("test_data/evidence.txt")
    if not path.exists():
        pytest.skip(f"{path} not found")
        
    out = from_maxquant_evidence(path)
    
    assert len(out) > 0
    assert "Stripped.Sequence" in out.columns
    assert "Precursor.Charge" in out.columns
    assert "Protein.Ids" in out.columns
    assert "Precursor.Normalised" in out.columns
    
    assert "length" in out.columns
    assert "mw" in out.columns
    assert "missed_cleavages" in out.columns
    
    if hasattr(out, "attrs") and "level" in out.attrs:
        assert out.attrs["level"] == "peptide"

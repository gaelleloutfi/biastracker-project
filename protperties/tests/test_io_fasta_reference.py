import pytest
from pathlib import Path
from protperties.io_fasta_reference import from_fasta_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path: Path, filename: str, content: str) -> Path:
    """Write FASTA content to a file."""
    p = tmp_path / filename
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Basic FASTA parsing and peptide generation
# ---------------------------------------------------------------------------

def test_simple_fasta(tmp_path):
    """Test basic FASTA parsing with one protein."""
    fasta_content = """>test_protein_1
PEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, peptides_df = from_fasta_reference(path, mc=0, min_len=1, max_len=100)
    
    # Check proteins dataframe
    assert len(proteins_df) == 1
    assert proteins_df["primary_id"].iloc[0] == "test_protein_1"
    assert proteins_df["sequence"].iloc[0] == "PEPTIDEKR"
    assert proteins_df["level"].iloc[0] == "protein"
    assert proteins_df["length"].iloc[0] == 9
    
    # Check peptides dataframe
    # PEPTIDEKR with mc=0 should give: PEPTIDEK | R
    assert len(peptides_df) >= 1  # At least one peptide
    assert (peptides_df["level"] == "peptide").all()
    assert (peptides_df["primary_id"] == "test_protein_1").all()
    
    # Check that sequence is the canonical column (not peptide_sequence)
    assert "sequence" in peptides_df.columns
    assert "missed_cleavages" in peptides_df.columns
    # peptide_length should not exist (replaced by length from basic_props)
    assert "length" in peptides_df.columns


def test_multiple_proteins(tmp_path):
    """Test FASTA with multiple proteins."""
    fasta_content = """>protein_A
AKRPQR
>protein_B
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, peptides_df = from_fasta_reference(path, mc=0, min_len=1, max_len=100)
    
    # Check proteins
    assert len(proteins_df) == 2
    assert set(proteins_df["primary_id"]) == {"protein_A", "protein_B"}
    
    # Check that peptides are generated for both proteins
    assert len(peptides_df) > 0
    peptide_sources = set(peptides_df["primary_id"].unique())
    assert peptide_sources.issubset({"protein_A", "protein_B"})


def test_missed_cleavages(tmp_path):
    """Test that missed cleavages parameter works."""
    fasta_content = """>test_protein
AKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    # With mc=0, should get: AK | R
    proteins_df_0, peptides_df_0 = from_fasta_reference(path, mc=0, min_len=1, max_len=100)
    
    # With mc=1, should get: AK | R | AKR
    proteins_df_1, peptides_df_1 = from_fasta_reference(path, mc=1, min_len=1, max_len=100)
    
    # mc=1 should produce more or equal peptides than mc=0
    assert len(peptides_df_1) >= len(peptides_df_0)
    
    # Check that AKR (full sequence) is present when mc=1
    assert "AKR" in peptides_df_1["sequence"].values


def test_peptide_length_filtering(tmp_path):
    """Test that min_len and max_len parameters filter peptides correctly."""
    fasta_content = """>test_protein
AKRPEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    # Filter to only keep peptides of length 2-5
    proteins_df, peptides_df = from_fasta_reference(path, mc=0, min_len=2, max_len=5)
    
    # All peptides should be within the length range
    assert all(peptides_df["length"] >= 2)
    assert all(peptides_df["length"] <= 5)


# ---------------------------------------------------------------------------
# Physicochemical properties
# ---------------------------------------------------------------------------

def test_protein_properties(tmp_path):
    """Test that protein properties are computed correctly."""
    fasta_content = """>test_protein
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path)
    
    # Check that basic_props properties are present
    required_cols = ["length", "mw", "pi", "gravy", "instability",
                     "aromaticity", "aliphatic_index", "ext_reduced",
                     "ext_cystine", "charge_at_pH"]
    for col in required_cols:
        assert col in proteins_df.columns, f"Missing column: {col}"
    
    # Check that values are reasonable
    row = proteins_df.iloc[0]
    assert row["length"] == 8
    assert row["mw"] > 0
    assert isinstance(row["pi"], float)


def test_peptide_properties(tmp_path):
    """Test that peptide properties are computed correctly."""
    fasta_content = """>test_protein
PEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    _, peptides_df = from_fasta_reference(path, mc=0, min_len=1, max_len=100)
    
    # Check that basic_props properties are present
    required_cols = ["length", "mw", "pi", "gravy", "instability",
                     "aromaticity", "aliphatic_index", "ext_reduced",
                     "ext_cystine", "charge_at_pH"]
    for col in required_cols:
        assert col in peptides_df.columns, f"Missing column: {col}"
    
    # Check that missed_cleavages is computed
    assert "missed_cleavages" in peptides_df.columns
    assert all(peptides_df["missed_cleavages"] >= 0)


def test_missed_cleavages_computation(tmp_path):
    """Test that missed cleavages are computed correctly for peptides."""
    fasta_content = """>test_protein
AKRPQKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    # Allow mc=2 to get peptides with internal K/R
    _, peptides_df = from_fasta_reference(path, mc=2, min_len=3, max_len=100)
    
    # Find a peptide with missed cleavages
    # For example, if we have "AKR", it should have 1 missed cleavage (internal K)
    if "AKR" in peptides_df["sequence"].values:
        mc_value = peptides_df[peptides_df["sequence"] == "AKR"]["missed_cleavages"].iloc[0]
        assert mc_value == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_fasta(tmp_path):
    """Test with an empty FASTA file."""
    path = _write_fasta(tmp_path, "empty.fasta", "")
    
    proteins_df, peptides_df = from_fasta_reference(path)
    
    assert len(proteins_df) == 0
    assert len(peptides_df) == 0


def test_protein_with_no_valid_peptides(tmp_path):
    """Test protein that produces no peptides within length constraints."""
    fasta_content = """>test_protein
KR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    # KR with mc=0 gives: K | R, both length 1
    # If min_len=6, no peptides should be kept
    proteins_df, peptides_df = from_fasta_reference(path, mc=0, min_len=6, max_len=65)
    
    assert len(proteins_df) == 1  # Protein should still be in the output
    # All peptides from this protein should be filtered out due to length constraints
    assert len(peptides_df) == 0


def test_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        from_fasta_reference("/nonexistent/path/file.fasta")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_proteins_schema(tmp_path):
    """Test that proteins DataFrame has the expected schema."""
    fasta_content = """>test_protein
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path)
    
    # Check required columns from ensure_sequence_table
    assert "level" in proteins_df.columns
    assert "sequence" in proteins_df.columns
    assert "primary_id" in proteins_df.columns
    
    # Check that level is set correctly
    assert (proteins_df["level"] == "protein").all()


def test_peptides_schema(tmp_path):
    """Test that peptides DataFrame has the expected schema."""
    fasta_content = """>test_protein
PEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    _, peptides_df = from_fasta_reference(path, mc=0, min_len=1, max_len=100)
    
    # Check required columns from ensure_sequence_table
    assert "level" in peptides_df.columns
    assert "sequence" in peptides_df.columns
    assert "primary_id" in peptides_df.columns
    
    # Check peptide-specific columns
    assert "missed_cleavages" in peptides_df.columns
    assert "length" in peptides_df.columns  # From basic_props, not peptide_length
    
    # Check that level is set correctly
    assert (peptides_df["level"] == "peptide").all()


# ---------------------------------------------------------------------------
# UniProt ID normalization
# ---------------------------------------------------------------------------

def test_normalize_ids_true(tmp_path):
    """Test that UniProt IDs are normalized when normalize_ids=True."""
    fasta_content = """>sp|P12345|PROT_NAME Some protein description
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, peptides_df = from_fasta_reference(path, normalize_ids=True)
    
    # Primary ID should be normalized to just P12345
    assert proteins_df["primary_id"].iloc[0] == "P12345"
    
    # Header column should contain the full description
    assert "header" in proteins_df.columns
    assert "sp|P12345|PROT_NAME" in proteins_df["header"].iloc[0]
    
    # Peptides should also have the normalized protein ID
    assert (peptides_df["primary_id"] == "P12345").all()


def test_normalize_ids_false(tmp_path):
    """Test that raw IDs are used when normalize_ids=False."""
    fasta_content = """>sp|P12345|PROT_NAME Some protein description
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, peptides_df = from_fasta_reference(path, normalize_ids=False)
    
    # Primary ID should be the raw record.id (sp|P12345|PROT_NAME)
    assert proteins_df["primary_id"].iloc[0] == "sp|P12345|PROT_NAME"
    
    # Header column should still be present
    assert "header" in proteins_df.columns
    
    # Peptides should have the same raw protein ID
    assert (peptides_df["primary_id"] == "sp|P12345|PROT_NAME").all()


def test_normalize_ids_fallback(tmp_path):
    """Test that normalization falls back to raw ID if normalization fails."""
    fasta_content = """>my_custom_protein_123
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path, normalize_ids=True)
    
    # Since this is not a valid UniProt ID, it should fall back to raw ID
    assert proteins_df["primary_id"].iloc[0] == "my_custom_protein_123"


def test_header_column(tmp_path):
    """Test that header column is always present."""
    fasta_content = """>test_protein Extra description here
PEPTIDEK
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path)
    
    assert "header" in proteins_df.columns
    assert proteins_df["header"].iloc[0] == "test_protein Extra description here"


def test_protein_trypsin_sites_and_aa_composition(tmp_path):
    """Test that protein DataFrame includes trypsin_sites and amino acid composition columns."""
    fasta_content = """>test_protein
AKRP
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path)
    
    # Check trypsin_sites column exists and is correct
    assert "trypsin_sites" in proteins_df.columns
    # AKRP has 1 trypsin site (after K; R is followed by P so no cut there)
    assert proteins_df["trypsin_sites"].iloc[0] == 1
    
    # Check that all 20 amino acid composition columns exist
    aa_cols = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
    for col in aa_cols:
        assert col in proteins_df.columns, f"Missing column: {col}"
    
    # Verify amino acid composition sums to approximately 1.0
    row = proteins_df.iloc[0]
    aa_values = [row[col] for col in aa_cols]
    assert abs(sum(aa_values) - 1.0) < 0.001
    
    # Verify specific amino acids are present/absent
    # AKRP: A=0.25, K=0.25, R=0.25, P=0.25, all others=0
    assert row["aa_A"] == 0.25
    assert row["aa_K"] == 0.25
    assert row["aa_R"] == 0.25
    assert row["aa_P"] == 0.25
    assert row["aa_C"] == 0.0  # No cysteine
    assert row["aa_W"] == 0.0  # No tryptophan


def test_protein_structure_matches_build_protein_table(tmp_path):
    """Test that protein-level output structure matches build_protein_table."""
    from unittest.mock import patch
    from protperties.protein_table import build_protein_table
    import pandas as pd
    
    # Get columns from from_fasta_reference
    fasta_content = """>test_protein
PEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    fasta_proteins_df, _ = from_fasta_reference(path)
    fasta_cols = set(fasta_proteins_df.columns)
    
    # Get columns from build_protein_table (using mock to avoid network calls)
    with patch("protperties.protein_table.fetch_uniprot_sequences") as mock_fetch:
        mock_fetch.return_value = {"P12345": "PEPTIDEKR"}
        
        ids_df = pd.DataFrame({"accession": ["P12345"]})
        build_result = build_protein_table(ids_df, accession_col="accession")
        build_cols = set(build_result.columns)
    
    # Remove expected differences to compare core column sets:
    # - 'accession' is specific to build_protein_table (original accession column)
    # - 'header' is specific to from_fasta_reference (full FASTA description)
    # Note: 'expression' is now present in both outputs (from_fasta_reference now includes it),
    # so we don't need to exclude it from either side.
    build_core = build_cols - {'accession'}
    fasta_core = fasta_cols - {'header'}
    
    # The core columns should be identical
    assert build_core == fasta_core, f"Mismatch: {build_core ^ fasta_core}"
    
    # Verify critical columns are present in both
    critical_cols = {
        "primary_id", "sequence", "level", "length", "mw", "pi", 
        "gravy", "instability", "aromaticity", "aliphatic_index",
        "trypsin_sites", "expression"
    }
    assert critical_cols.issubset(fasta_core)
    assert critical_cols.issubset(build_core)
    
    # Verify all 20 aa columns are present in both
    aa_cols = {f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"}
    assert aa_cols.issubset(fasta_core)
    assert aa_cols.issubset(build_core)


def test_expression_column(tmp_path):
    """Test that expression column is present with default value."""
    fasta_content = """>test_protein
PEPTIDEKR
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    # Test with default expression value (0.0)
    proteins_df, _ = from_fasta_reference(path)
    assert "expression" in proteins_df.columns
    assert proteins_df["expression"].iloc[0] == 0.0
    
    # Test with custom expression value
    proteins_df, _ = from_fasta_reference(path, default_expression=100.0)
    assert proteins_df["expression"].iloc[0] == 100.0


def test_consistent_sequence_handling_with_non_canonical_aa(tmp_path):
    """Test that non-canonical amino acids are cleaned consistently for all property calculations."""
    # Create a FASTA with non-canonical amino acids (X, *, etc.)
    fasta_content = """>test_protein
PEPTIDE*XKRB
"""
    path = _write_fasta(tmp_path, "test.fasta", fasta_content)
    
    proteins_df, _ = from_fasta_reference(path)
    
    # The sequence should be cleaned (non-canonical removed)
    # PEPTIDE*XKRB -> PEPTIDEKR (removes *, X, B)
    cleaned_seq = proteins_df["sequence"].iloc[0]
    assert cleaned_seq == "PEPTIDEKR"
    
    # Length should match the cleaned sequence
    assert proteins_df["length"].iloc[0] == len(cleaned_seq)
    assert proteins_df["length"].iloc[0] == 9
    
    # Trypsin sites should be counted on the cleaned sequence
    # PEPTIDEKR has 2 trypsin sites (after K and after R)
    assert proteins_df["trypsin_sites"].iloc[0] == 2
    
    # Amino acid composition should sum to 1.0 (based on cleaned sequence)
    aa_cols = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
    aa_sum = sum(proteins_df[col].iloc[0] for col in aa_cols)
    assert abs(aa_sum - 1.0) < 0.001
    
    # Verify no non-canonical amino acids in composition
    # Columns for non-canonical amino acids like aa_X or aa_B should not exist
    assert "aa_X" not in proteins_df.columns
    assert "aa_B" not in proteins_df.columns


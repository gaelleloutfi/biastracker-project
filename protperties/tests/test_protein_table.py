"""
Tests for protein_table module (protein-level table builder).
"""
from unittest.mock import Mock, patch
import pandas as pd
import pytest

from protperties.protein_table import build_protein_table


class TestBuildProteinTable:
    """Tests for build_protein_table function."""
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_with_expression_col(self, mock_fetch):
        """Test build_protein_table with expression column provided."""
        # Mock UniProt fetch to return sequences
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
            "Q9XXX1": "ARNDCEQGHILKMFPSTWYV",
        }
        
        # Input DataFrame with expression data
        ids_df = pd.DataFrame({
            "accession": ["P12345", "Q9XXX1"],
            "intensity": [1000.0, 2000.0],
        })
        
        # Build protein table
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            expression_col="intensity",
        )
        
        # Verify basic structure
        assert "primary_id" in result.columns
        assert "sequence" in result.columns
        assert "expression" in result.columns
        assert "level" in result.columns
        assert "trypsin_sites" in result.columns
        
        # Verify level is set to protein
        assert (result["level"] == "protein").all()
        
        # Verify primary_id matches normalized accessions
        assert "P12345" in result["primary_id"].values
        assert "Q9XXX1" in result["primary_id"].values
        
        # Verify expression values are preserved
        p12345_row = result[result["primary_id"] == "P12345"]
        assert p12345_row["expression"].iloc[0] == 1000.0
        
        q9xxx1_row = result[result["primary_id"] == "Q9XXX1"]
        assert q9xxx1_row["expression"].iloc[0] == 2000.0
        
        # Verify physicochemical properties are present
        assert "length" in result.columns
        assert "mw" in result.columns
        assert "pi" in result.columns
        assert "gravy" in result.columns
        assert "instability" in result.columns
        assert "aromaticity" in result.columns
        assert "aliphatic_index" in result.columns
        assert "charge_at_pH" in result.columns
        
        # Verify amino acid composition columns are present
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert f"aa_{aa}" in result.columns
        
        # Verify fetch was called with correct accessions
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        accessions_arg = call_args[0][0]
        assert set(accessions_arg) == {"P12345", "Q9XXX1"}
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_without_expression_col(self, mock_fetch):
        """Test build_protein_table without expression column (uses default)."""
        # Mock UniProt fetch
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
        }
        
        # Input DataFrame without expression data
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        # Build protein table with default expression
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            expression_col=None,
            default_expression=5.0,
        )
        
        # Verify expression is set to default
        assert (result["expression"] == 5.0).all()
        
        # Verify other required columns exist
        assert "primary_id" in result.columns
        assert "sequence" in result.columns
        assert "level" in result.columns
        assert "trypsin_sites" in result.columns
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_expression_col_with_missing_values(self, mock_fetch):
        """Test that missing expression values are filled with default."""
        # Mock UniProt fetch
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
            "Q9XXX1": "ARNDCEQGHILKMFPSTWYV",
        }
        
        # Input DataFrame with missing expression values
        ids_df = pd.DataFrame({
            "accession": ["P12345", "Q9XXX1"],
            "intensity": [1000.0, None],
        })
        
        # Build protein table
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            expression_col="intensity",
            default_expression=10.0,
        )
        
        # Verify missing value is filled with default
        q9xxx1_row = result[result["primary_id"] == "Q9XXX1"]
        assert q9xxx1_row["expression"].iloc[0] == 10.0
        
        # Verify non-missing value is preserved
        p12345_row = result[result["primary_id"] == "P12345"]
        assert p12345_row["expression"].iloc[0] == 1000.0
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_invalid_accession_col(self, mock_fetch):
        """Test that ValueError is raised when accession_col doesn't exist."""
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        with pytest.raises(ValueError, match="accession_col 'wrong_col' not found"):
            build_protein_table(ids_df, accession_col="wrong_col")
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_fasta_header_normalization(self, mock_fetch):
        """Test that FASTA headers are normalized correctly."""
        # Mock UniProt fetch
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
        }
        
        # Input with FASTA-style accession
        ids_df = pd.DataFrame({
            "accession": ["sp|P12345|PROTEIN_NAME"],
        })
        
        # Build protein table
        result = build_protein_table(
            ids_df,
            accession_col="accession",
        )
        
        # Verify accession is normalized
        assert result["primary_id"].iloc[0] == "P12345"
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_trypsin_sites_computed(self, mock_fetch):
        """Test that trypsin_sites are computed correctly."""
        # Mock sequence with known trypsin sites
        # "AKRP" has 1 trypsin site (after K, R is followed by P so no cut)
        mock_fetch.return_value = {
            "P12345": "AKRP",
        }
        
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        result = build_protein_table(ids_df, accession_col="accession")
        
        # Verify trypsin_sites is computed
        assert result["trypsin_sites"].iloc[0] == 1
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_amino_acid_composition_columns(self, mock_fetch):
        """Test that all 20 amino acid composition columns are present."""
        mock_fetch.return_value = {
            "P12345": "ACDEFGHIKLMNPQRSTVWY",
        }
        
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        result = build_protein_table(ids_df, accession_col="accession")
        
        # Verify all 20 aa_X columns exist
        aa_cols = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
        for col in aa_cols:
            assert col in result.columns
        
        # Verify composition sums to approximately 1.0
        aa_values = [result[col].iloc[0] for col in aa_cols]
        assert abs(sum(aa_values) - 1.0) < 0.001
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_custom_ph_parameter(self, mock_fetch):
        """Test that custom pH is passed to basic_props."""
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
        }
        
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        # Build table with custom pH
        result = build_protein_table(ids_df, accession_col="accession", ph=7.0)
        
        # Verify charge_at_pH is computed (value will differ from default pH)
        assert "charge_at_pH" in result.columns
        assert result["charge_at_pH"].notna().all()
    
    def test_fetch_missing_sequences_false(self):
        """Test that fetch_missing_sequences=False uses existing sequences."""
        # Input DataFrame with sequences already present
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
            "sequence": ["MTEYKLVVVGAGGVGKSALTIQ"],
        })
        
        # Build table without fetching
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            fetch_missing_sequences=False,
        )
        
        # Verify sequence is preserved
        assert result["sequence"].iloc[0] == "MTEYKLVVVGAGGVGKSALTIQ"
        assert result["primary_id"].iloc[0] == "P12345"
        
        # Verify physicochemical properties are computed from existing sequence
        assert result["length"].iloc[0] == 22  # Length of "MTEYKLVVVGAGGVGKSALTIQ"
        assert result["mw"].iloc[0] > 0  # Molecular weight computed
        assert result["trypsin_sites"].iloc[0] > 0  # Has K residues
        
        # Verify amino acid composition computed (sequence starts with M)
        assert "aa_M" in result.columns
        assert result["aa_M"].iloc[0] > 0  # M is present in sequence
    
    def test_fetch_missing_sequences_false_no_sequence_col(self):
        """Test that ValueError is raised when fetch_missing_sequences=False and no sequence column."""
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        with pytest.raises(ValueError, match="no 'sequence' column found"):
            build_protein_table(
                ids_df,
                accession_col="accession",
                fetch_missing_sequences=False,
            )
    
    def test_fetch_missing_sequences_false_with_missing_sequence_values(self):
        """Test that sequences with missing/None values are dropped when fetch_missing_sequences=False."""
        # Input DataFrame with some missing sequences
        ids_df = pd.DataFrame({
            "accession": ["P12345", "Q9XXX1", "O43663"],
            "sequence": ["MTEYKLVVVGAGGVGKSALTIQ", None, ""],  # One valid, one None, one empty
        })
        
        # Build table without fetching - rows with None/empty sequences are dropped
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            fetch_missing_sequences=False,
        )
        
        # Verify only the row with valid sequence remains
        assert len(result) == 1
        assert "P12345" in result["primary_id"].values
        assert "Q9XXX1" not in result["primary_id"].values
        assert "O43663" not in result["primary_id"].values
        
        # Verify the valid sequence has proper properties
        assert result["length"].iloc[0] == 22  # Length of "MTEYKLVVVGAGGVGKSALTIQ"
    
    def test_fetch_missing_sequences_false_all_sequences_empty(self):
        """Test that ValueError is raised when all sequences are empty/None."""
        # Input DataFrame with all empty/None sequences
        ids_df = pd.DataFrame({
            "accession": ["P12345", "Q9XXX1"],
            "sequence": [None, ""],  # All invalid
        })
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="All sequences were NA or empty"):
            build_protein_table(
                ids_df,
                accession_col="accession",
                fetch_missing_sequences=False,
            )
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_multiple_proteins(self, mock_fetch):
        """Test with multiple proteins to ensure scalability."""
        # Mock multiple sequences
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQGHI",
            "O43663": "MKWVTFISL",
        }
        
        ids_df = pd.DataFrame({
            "accession": ["P12345", "Q9XXX1", "O43663"],
            "intensity": [100.0, 200.0, 300.0],
        })
        
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            expression_col="intensity",
        )
        
        # Verify all proteins are present
        assert len(result) == 3
        assert set(result["primary_id"]) == {"P12345", "Q9XXX1", "O43663"}
        
        # Verify each has properties computed
        for _, row in result.iterrows():
            assert row["length"] > 0
            assert row["mw"] > 0
            assert "trypsin_sites" in row
    
    @patch("protperties.protein_table.fetch_uniprot_sequences")
    def test_uniprot_fetch_kwargs_passed(self, mock_fetch):
        """Test that additional kwargs are passed to fetch_uniprot_sequences."""
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVGAGGVGKSALTIQ",
        }
        
        ids_df = pd.DataFrame({
            "accession": ["P12345"],
        })
        
        # Build table with custom fetch kwargs
        result = build_protein_table(
            ids_df,
            accession_col="accession",
            cache_dir="/tmp/custom_cache",
            batch_size=100,
        )
        
        # Verify kwargs were passed
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs.get("cache_dir") == "/tmp/custom_cache"
        assert call_kwargs.get("batch_size") == 100

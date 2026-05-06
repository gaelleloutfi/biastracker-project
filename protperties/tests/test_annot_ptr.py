"""
Tests for annot_ptr module (PTR annotation).
"""
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from protperties.annot_ptr import add_ptr_annotation


class TestAddPtrAnnotation:
    """Tests for add_ptr_annotation function."""
    
    @pytest.fixture
    def sample_ptr_table(self, tmp_path):
        """Create a sample PTR table Excel file."""
        ptr_data = pd.DataFrame({
            "Entry": ["P12345", "Q9XXX1", "O43663", "A0A024R1R8", "P99999"],
            "Entry Name": ["PROT1_HUMAN", "PROT2_HUMAN", "PROT3_HUMAN", "PROT4_HUMAN", "PROT5_HUMAN"],
            "PTR AML": [1557.36, 0.0, "N/A", 2360.90, 486.542],
        })
        
        ptr_path = tmp_path / "test_ptr.xlsx"
        ptr_data.to_excel(ptr_path, index=False)
        return ptr_path
    
    @pytest.fixture
    def sample_proteins_df(self):
        """Create a sample proteins DataFrame."""
        return pd.DataFrame({
            "primary_id": ["P12345", "Q9XXX1", "O43663", "A0A024R1R8", "P12345-2", "P99999", "P00000"],
            "protein_name": ["Prot1", "Prot2", "Prot3", "Prot4 long", "Prot1 isoform", "Prot5", "Unknown"],
        })
    
    def test_human_species_homo_sapiens(self, sample_proteins_df, sample_ptr_table):
        """Test annotation with species='Homo sapiens'."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="Homo sapiens"
        )
        
        # Check that PTR_AML column was added
        assert "PTR_AML" in result.columns
        
        # Check specific values
        assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
        assert pd.isna(result.loc[result["primary_id"] == "Q9XXX1", "PTR_AML"].iloc[0])  # 0 -> NaN
        assert pd.isna(result.loc[result["primary_id"] == "O43663", "PTR_AML"].iloc[0])  # N/A -> NaN
        assert result.loc[result["primary_id"] == "A0A024R1R8", "PTR_AML"].iloc[0] == 2360.90  # Long accession
        assert result.loc[result["primary_id"] == "P99999", "PTR_AML"].iloc[0] == 486.542
        assert pd.isna(result.loc[result["primary_id"] == "P00000", "PTR_AML"].iloc[0])  # Not in PTR table
    
    def test_human_species_lowercase(self, sample_proteins_df, sample_ptr_table):
        """Test annotation with species='human' (lowercase)."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # Should work the same as "Homo sapiens"
        assert "PTR_AML" in result.columns
        assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
    
    def test_human_species_taxid(self, sample_proteins_df, sample_ptr_table):
        """Test annotation with species='9606' (taxonomy ID)."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="9606"
        )
        
        # Should work the same as "human"
        assert "PTR_AML" in result.columns
        assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
    
    def test_human_species_case_insensitive(self, sample_proteins_df, sample_ptr_table):
        """Test that species matching is case-insensitive."""
        for species in ["HUMAN", "Human", "HuMaN", "HOMO SAPIENS", "Homo Sapiens"]:
            result = add_ptr_annotation(
                sample_proteins_df,
                sample_ptr_table,
                species=species
            )
            assert "PTR_AML" in result.columns
            assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
    
    def test_non_human_species_returns_unchanged(self, sample_proteins_df, sample_ptr_table):
        """Test that non-human species returns df unchanged with strict_human=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_ptr_annotation(
                sample_proteins_df,
                sample_ptr_table,
                species="Mus musculus",
                strict_human=True
            )
            
            # Should have warning
            assert len(w) == 1
            assert "only available for HUMAN" in str(w[0].message)
        
        # Should not have PTR_AML column at all when non-human
        assert "PTR_AML" not in result.columns
        
        # DataFrame should be unchanged (same columns as original)
        assert len(result) == len(sample_proteins_df)
        assert list(result.columns) == list(sample_proteins_df.columns)
    
    def test_species_none_returns_unchanged(self, sample_proteins_df, sample_ptr_table):
        """Test that species=None returns df unchanged with strict_human=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_ptr_annotation(
                sample_proteins_df,
                sample_ptr_table,
                species=None,
                strict_human=True
            )
            
            # Should have warning
            assert len(w) == 1
            assert "requires explicit species" in str(w[0].message)
        
        # Should not have PTR_AML column at all when species=None
        assert "PTR_AML" not in result.columns
        
        # DataFrame should be unchanged (same columns as original)
        assert len(result) == len(sample_proteins_df)
        assert list(result.columns) == list(sample_proteins_df.columns)
    
    def test_strict_human_false_allows_annotation(self, sample_proteins_df, sample_ptr_table):
        """Test that strict_human=False allows annotation regardless of species."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="mouse",  # Non-human
            strict_human=False
        )
        
        # Should annotate anyway
        assert "PTR_AML" in result.columns
        assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
    
    def test_isoform_matching(self, sample_ptr_table):
        """Test that isoform accessions match canonical accessions in PTR table."""
        # PTR table has P12345, proteins_df has P12345-2
        proteins_df = pd.DataFrame({
            "primary_id": ["P12345-2", "P12345-3", "Q9XXX1-1"],
        })
        
        result = add_ptr_annotation(
            proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # All isoforms of P12345 should match P12345 in PTR table
        assert result.loc[result["primary_id"] == "P12345-2", "PTR_AML"].iloc[0] == 1557.36
        assert result.loc[result["primary_id"] == "P12345-3", "PTR_AML"].iloc[0] == 1557.36
        # Q9XXX1 has PTR=0 which becomes NaN
        assert pd.isna(result.loc[result["primary_id"] == "Q9XXX1-1", "PTR_AML"].iloc[0])
    
    def test_zero_converted_to_nan(self, sample_proteins_df, sample_ptr_table):
        """Test that PTR values of 0 are converted to NaN."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # Q9XXX1 has PTR=0 in the table
        assert pd.isna(result.loc[result["primary_id"] == "Q9XXX1", "PTR_AML"].iloc[0])
    
    def test_na_string_converted_to_nan(self, sample_proteins_df, sample_ptr_table):
        """Test that N/A string values are converted to NaN."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # O43663 has PTR="N/A" in the table
        assert pd.isna(result.loc[result["primary_id"] == "O43663", "PTR_AML"].iloc[0])
    
    def test_valid_ptr_preserved(self, sample_proteins_df, sample_ptr_table):
        """Test that valid numeric PTR values are preserved."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # P12345 has valid PTR value
        assert result.loc[result["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
        # P99999 has valid PTR value
        assert result.loc[result["primary_id"] == "P99999", "PTR_AML"].iloc[0] == 486.542
    
    def test_custom_accession_column(self, sample_ptr_table):
        """Test using a custom accession column name."""
        proteins_df = pd.DataFrame({
            "uniprot_id": ["P12345", "Q9XXX1"],
            "name": ["Prot1", "Prot2"],
        })
        
        result = add_ptr_annotation(
            proteins_df,
            sample_ptr_table,
            accession_col="uniprot_id",
            species="human"
        )
        
        assert "PTR_AML" in result.columns
        assert result.loc[result["uniprot_id"] == "P12345", "PTR_AML"].iloc[0] == 1557.36
    
    def test_custom_output_column(self, sample_proteins_df, sample_ptr_table):
        """Test using a custom output column name."""
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human",
            ptr_out_col="PTR_custom"
        )
        
        assert "PTR_custom" in result.columns
        assert result.loc[result["primary_id"] == "P12345", "PTR_custom"].iloc[0] == 1557.36
    
    def test_missing_ptr_table_file(self, sample_proteins_df):
        """Test error when PTR table file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            add_ptr_annotation(
                sample_proteins_df,
                "/nonexistent/path/to/ptr.xlsx",
                species="human"
            )
    
    def test_missing_accession_column(self, sample_ptr_table):
        """Test error when accession column doesn't exist in proteins_df."""
        proteins_df = pd.DataFrame({
            "wrong_column": ["P12345"],
        })
        
        with pytest.raises(ValueError, match="not found in proteins_df"):
            add_ptr_annotation(
                proteins_df,
                sample_ptr_table,
                accession_col="primary_id",
                species="human"
            )
    
    def test_invalid_ptr_table_columns(self, tmp_path, sample_proteins_df):
        """Test error when PTR table has wrong columns."""
        invalid_ptr = pd.DataFrame({
            "WrongColumn1": ["P12345"],
            "WrongColumn2": [100],
        })
        
        ptr_path = tmp_path / "invalid_ptr.xlsx"
        invalid_ptr.to_excel(ptr_path, index=False)
        
        with pytest.raises(ValueError, match="missing expected columns"):
            add_ptr_annotation(
                sample_proteins_df,
                ptr_path,
                species="human"
            )
    
    def test_no_matching_accessions_warning(self, sample_ptr_table):
        """Test warning when no proteins match PTR table."""
        proteins_df = pd.DataFrame({
            "primary_id": ["XXXXXX", "YYYYYY"],  # Non-existent accessions
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_ptr_annotation(
                proteins_df,
                sample_ptr_table,
                species="human"
            )
            
            # Should have warning about no matches
            assert len(w) == 1
            assert "No PTR values found" in str(w[0].message)
        
        # All PTR values should be NaN
        assert result["PTR_AML"].isna().all()
    
    def test_original_dataframe_unchanged(self, sample_proteins_df, sample_ptr_table):
        """Test that the original DataFrame is not modified."""
        original_columns = sample_proteins_df.columns.tolist()
        original_len = len(sample_proteins_df)
        
        result = add_ptr_annotation(
            sample_proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # Original should be unchanged
        assert sample_proteins_df.columns.tolist() == original_columns
        assert len(sample_proteins_df) == original_len
        assert "PTR_AML" not in sample_proteins_df.columns
        
        # Result should have PTR_AML
        assert "PTR_AML" in result.columns
    
    def test_fasta_header_format(self, sample_ptr_table):
        """Test handling of FASTA header format accessions."""
        proteins_df = pd.DataFrame({
            "primary_id": ["sp|P12345|PROT1_HUMAN", "tr|Q9XXX1|PROT2_HUMAN"],
        })
        
        result = add_ptr_annotation(
            proteins_df,
            sample_ptr_table,
            species="human"
        )
        
        # Should extract accessions and match
        assert result.loc[result["primary_id"] == "sp|P12345|PROT1_HUMAN", "PTR_AML"].iloc[0] == 1557.36
        assert pd.isna(result.loc[result["primary_id"] == "tr|Q9XXX1|PROT2_HUMAN", "PTR_AML"].iloc[0])

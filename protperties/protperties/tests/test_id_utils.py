"""
Tests for id_utils module (UniProt accession normalization).
"""
import pytest
from protperties.id_utils import (
    normalize_uniprot_accession,
    extract_uniprot_accessions,
    canonicalize_isoform,
)


class TestNormalizeUniprotAccession:
    """Tests for normalize_uniprot_accession function."""
    
    def test_fasta_header_sp(self):
        """Test extraction from Swiss-Prot FASTA header."""
        assert normalize_uniprot_accession("sp|P12345|PROTEIN_NAME") == "P12345"
        assert normalize_uniprot_accession("sp|Q9XXX1|SOME_PROTEIN") == "Q9XXX1"
    
    def test_fasta_header_tr(self):
        """Test extraction from TrEMBL FASTA header."""
        assert normalize_uniprot_accession("tr|Q9XXX1|TR_PROTEIN") == "Q9XXX1"
        assert normalize_uniprot_accession("tr|A0A024R1R8|PROTEIN") == "A0A024R1R8"
    
    def test_plain_accession(self):
        """Test plain UniProt accession."""
        assert normalize_uniprot_accession("P12345") == "P12345"
        assert normalize_uniprot_accession("Q9XXX1") == "Q9XXX1"
        assert normalize_uniprot_accession("O43663") == "O43663"
    
    def test_isoform_accession(self):
        """Test accession with isoform suffix."""
        assert normalize_uniprot_accession("P12345-2") == "P12345-2"
        assert normalize_uniprot_accession("Q9XXX1-3") == "Q9XXX1-3"
        assert normalize_uniprot_accession("O43663-10") == "O43663-10"
    
    def test_semicolon_separated(self):
        """Test semicolon-separated list (returns first valid)."""
        assert normalize_uniprot_accession("P12345;Q9XXX1") == "P12345"
        assert normalize_uniprot_accession("P12345;Q9XXX1;O43663") == "P12345"
    
    def test_with_whitespace(self):
        """Test accessions with surrounding whitespace."""
        assert normalize_uniprot_accession("  P12345  ") == "P12345"
        assert normalize_uniprot_accession("  sp|P12345|NAME  ") == "P12345"
    
    def test_invalid_inputs(self):
        """Test invalid or malformed inputs."""
        assert normalize_uniprot_accession("invalid") is None
        assert normalize_uniprot_accession("12345") is None
        assert normalize_uniprot_accession("") is None
        assert normalize_uniprot_accession("   ") is None
        assert normalize_uniprot_accession("ABCDEF") is None
    
    def test_none_input(self):
        """Test None input."""
        assert normalize_uniprot_accession(None) is None
    
    def test_long_accessions(self):
        """Test 10-character accessions (6+4 format)."""
        assert normalize_uniprot_accession("A0A024R1R8") == "A0A024R1R8"
        assert normalize_uniprot_accession("A0A024R1R8-2") == "A0A024R1R8-2"
        assert normalize_uniprot_accession("sp|A0A024R1R8|NAME") == "A0A024R1R8"


class TestExtractUniprotAccessions:
    """Tests for extract_uniprot_accessions function."""
    
    def test_single_accession(self):
        """Test single accession extraction."""
        assert extract_uniprot_accessions("P12345") == ["P12345"]
    
    def test_semicolon_separated_list(self):
        """Test semicolon-separated list extraction."""
        result = extract_uniprot_accessions("P12345;Q9XXX1;O43663")
        assert result == ["P12345", "Q9XXX1", "O43663"]
    
    def test_mixed_formats(self):
        """Test list with mixed formats (FASTA headers and plain)."""
        result = extract_uniprot_accessions("sp|P12345|NAME;Q9XXX1;tr|O43663|PROT")
        assert result == ["P12345", "Q9XXX1", "O43663"]
    
    def test_with_isoforms(self):
        """Test extraction preserves isoforms."""
        result = extract_uniprot_accessions("P12345-2;Q9XXX1-3")
        assert result == ["P12345-2", "Q9XXX1-3"]
    
    def test_with_invalid_entries(self):
        """Test list with some invalid entries."""
        result = extract_uniprot_accessions("P12345;invalid;Q9XXX1;garbage")
        assert result == ["P12345", "Q9XXX1"]
    
    def test_empty_string(self):
        """Test empty string."""
        assert extract_uniprot_accessions("") == []
    
    def test_all_invalid(self):
        """Test string with no valid accessions."""
        assert extract_uniprot_accessions("invalid;garbage;nonsense") == []
    
    def test_none_input(self):
        """Test None input."""
        assert extract_uniprot_accessions(None) == []
    
    def test_with_spaces(self):
        """Test accessions with spaces."""
        result = extract_uniprot_accessions("P12345 ; Q9XXX1 ; O43663")
        assert result == ["P12345", "Q9XXX1", "O43663"]


class TestCanonicalizeIsoform:
    """Tests for canonicalize_isoform function."""
    
    def test_remove_isoform(self):
        """Test removal of isoform suffix."""
        assert canonicalize_isoform("P12345-2") == "P12345"
        assert canonicalize_isoform("Q9XXX1-3") == "Q9XXX1"
        assert canonicalize_isoform("O43663-10") == "O43663"
    
    def test_no_isoform(self):
        """Test accessions without isoform suffix."""
        assert canonicalize_isoform("P12345") == "P12345"
        assert canonicalize_isoform("Q9XXX1") == "Q9XXX1"
    
    def test_long_accession_with_isoform(self):
        """Test 10-character accession with isoform."""
        assert canonicalize_isoform("A0A024R1R8-2") == "A0A024R1R8"
    
    def test_fasta_header_unchanged(self):
        """Test that FASTA headers are not modified."""
        # canonicalize_isoform only works on plain accessions
        assert canonicalize_isoform("sp|P12345-2|NAME") == "sp|P12345-2|NAME"
    
    def test_invalid_input(self):
        """Test invalid inputs are returned unchanged."""
        assert canonicalize_isoform("invalid") == "invalid"
        assert canonicalize_isoform("") == ""
        assert canonicalize_isoform(None) is None
    
    def test_multiple_hyphens(self):
        """Test that only trailing -digit is considered an isoform."""
        # This should not match as an isoform pattern
        assert canonicalize_isoform("P12345-ABC") == "P12345-ABC"


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""
    
    def test_fasta_header_to_canonical(self):
        """Test extracting and canonicalizing from FASTA header."""
        acc = normalize_uniprot_accession("sp|P12345-2|PROTEIN_NAME")
        assert acc == "P12345-2"
        canonical = canonicalize_isoform(acc)
        assert canonical == "P12345"
    
    def test_extract_and_canonicalize_list(self):
        """Test extracting multiple accessions and canonicalizing them."""
        accessions = extract_uniprot_accessions("P12345-2;Q9XXX1-3;O43663")
        canonical = [canonicalize_isoform(acc) for acc in accessions]
        assert canonical == ["P12345", "Q9XXX1", "O43663"]
    
    def test_protein_groups_format(self):
        """Test common protein groups format from search engines."""
        # MaxQuant style
        accessions = extract_uniprot_accessions("sp|P12345|PROT1;sp|Q9XXX1|PROT2")
        assert accessions == ["P12345", "Q9XXX1"]
        
        # Plain semicolon-separated
        accessions = extract_uniprot_accessions("P12345;Q9XXX1;O43663")
        assert accessions == ["P12345", "Q9XXX1", "O43663"]

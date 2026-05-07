"""
Tests for reference_proteome module (UniProt reference proteome fetching).
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from protperties.reference_proteome import (
    build_reference_proteome,
    _find_reference_proteome,
    _download_proteome_fasta,
)


# ---------------------------------------------------------------------------
# Tests for _find_reference_proteome
# ---------------------------------------------------------------------------


class TestFindReferenceProteome:
    """Tests for finding reference proteome IDs."""
    
    @patch("protperties.reference_proteome.requests.get")
    def test_successful_single_proteome(self, mock_get):
        """Test successful query returning a single proteome."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "UP000005640",
                    "taxonomy": {"scientificName": "Homo sapiens"},
                    "proteomeType": "Reference proteome",
                }
            ]
        }
        mock_get.return_value = mock_response
        
        proteome_id = _find_reference_proteome("homo sapiens")
        
        assert proteome_id == "UP000005640"
        
        # Verify API call
        call_args = mock_get.call_args
        assert "https://rest.uniprot.org/proteomes/search" in call_args[0]
        params = call_args[1]["params"]
        assert "homo sapiens" in params["query"].lower()
        assert "Reference proteome" in params["query"]
    
    @patch("protperties.reference_proteome.requests.get")
    def test_multiple_proteomes_warning(self, mock_get):
        """Test that a warning is issued when multiple proteomes are found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "UP000005640",
                    "taxonomy": {"scientificName": "Homo sapiens"},
                },
                {
                    "id": "UP000002494",
                    "taxonomy": {"scientificName": "Homo sapiens neanderthalensis"},
                },
            ]
        }
        mock_get.return_value = mock_response
        
        with pytest.warns(UserWarning, match="Multiple reference proteomes found"):
            proteome_id = _find_reference_proteome("homo sapiens")
        
        # Should return the first one
        assert proteome_id == "UP000005640"
    
    @patch("protperties.reference_proteome.requests.get")
    def test_no_proteome_found(self, mock_get):
        """Test error when no proteome is found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="No reference proteome found"):
            _find_reference_proteome("nonexistent species")
    
    @patch("protperties.reference_proteome.requests.get")
    def test_network_error(self, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(RuntimeError, match="Failed to query UniProt proteomes API"):
            _find_reference_proteome("homo sapiens")
    
    @patch("protperties.reference_proteome.requests.get")
    def test_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Failed to query UniProt proteomes API"):
            _find_reference_proteome("homo sapiens")
    
    @patch("protperties.reference_proteome.requests.get")
    def test_missing_proteome_id(self, mock_get):
        """Test error when proteome result is missing ID field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "taxonomy": {"scientificName": "Test species"},
                    # Missing 'id' field
                }
            ]
        }
        mock_get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="missing ID"):
            _find_reference_proteome("test species")


# ---------------------------------------------------------------------------
# Tests for _download_proteome_fasta
# ---------------------------------------------------------------------------


class TestDownloadProteomeFasta:
    """Tests for downloading proteome FASTA files."""
    
    @patch("protperties.reference_proteome.requests.get")
    def test_successful_download_reviewed(self, mock_get, tmp_path):
        """Test successful FASTA download with reviewed filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b">sp|P12345|PROT1\n", b"MTEYKLVVVG\n"])
        mock_get.return_value = mock_response
        
        output_path = tmp_path / "test.fasta"
        _download_proteome_fasta("UP000005640", reviewed=True, output_path=output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify API call
        call_args = mock_get.call_args
        assert "https://rest.uniprot.org/uniprotkb/stream" in call_args[0]
        params = call_args[1]["params"]
        assert "UP000005640" in params["query"]
        assert "reviewed:true" in params["query"]
        assert params["format"] == "fasta"
    
    @patch("protperties.reference_proteome.requests.get")
    def test_successful_download_all(self, mock_get, tmp_path):
        """Test successful FASTA download without reviewed filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b">tr|Q9XXX1|PROT2\n", b"ARNDCEQ\n"])
        mock_get.return_value = mock_response
        
        output_path = tmp_path / "test.fasta"
        _download_proteome_fasta("UP000005640", reviewed=False, output_path=output_path)
        
        assert output_path.exists()
        
        # Verify API call doesn't include reviewed filter
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert "UP000005640" in params["query"]
        assert "reviewed" not in params["query"]
    
    @patch("protperties.reference_proteome.requests.get")
    def test_empty_response(self, mock_get, tmp_path):
        """Test error when downloaded FASTA is empty."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[])
        mock_get.return_value = mock_response
        
        output_path = tmp_path / "test.fasta"
        
        with pytest.raises(RuntimeError, match="empty"):
            _download_proteome_fasta("UP000005640", reviewed=True, output_path=output_path)
        
        # Verify cleanup: file should not exist after error
        assert not output_path.exists()
    
    @patch("protperties.reference_proteome.requests.get")
    def test_streaming_error_cleanup(self, mock_get, tmp_path):
        """Test that partial downloads are cleaned up on streaming error."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Simulate error during streaming
        def error_iter():
            yield b">sp|P12345|PROT1\n"
            raise Exception("Network interrupted")
        mock_response.iter_content = error_iter
        mock_get.return_value = mock_response
        
        output_path = tmp_path / "test.fasta"
        
        with pytest.raises(RuntimeError, match="Failed to download"):
            _download_proteome_fasta("UP000005640", reviewed=True, output_path=output_path)
        
        # Verify cleanup: partial file should be removed
        assert not output_path.exists()
    
    @patch("protperties.reference_proteome.requests.get")
    def test_http_error(self, mock_get, tmp_path):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response
        
        output_path = tmp_path / "test.fasta"
        
        with pytest.raises(RuntimeError, match="Failed to download"):
            _download_proteome_fasta("UP000005640", reviewed=True, output_path=output_path)


# ---------------------------------------------------------------------------
# Tests for build_reference_proteome (integration)
# ---------------------------------------------------------------------------


class TestBuildReferenceProteome:
    """Tests for the main build_reference_proteome function."""
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_basic_workflow_no_cache(self, mock_find, mock_download, tmp_path):
        """Test basic workflow when FASTA is not cached."""
        # Mock finding proteome
        mock_find.return_value = "UP000005640"
        
        # Create a temporary FASTA file for the download mock to "create"
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|TEST_HUMAN Test protein
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        # Call the function
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            reviewed=True,
            mc=0,
            min_len=1,
            max_len=100,
            cache_dir=cache_dir,
        )
        
        # Verify results
        assert len(proteins_df) > 0
        assert len(peptides_df) > 0
        assert "primary_id" in proteins_df.columns
        assert "sequence" in proteins_df.columns
        assert "level" in proteins_df.columns
        assert proteins_df["level"].iloc[0] == "protein"
        
        # Verify function calls
        mock_find.assert_called_once_with("homo sapiens")
        mock_download.assert_called_once()
        
        # Verify cache file was created
        assert (cache_dir / "UP000005640_reviewed.fasta").exists()
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_cache_hit_no_download(self, mock_find, mock_download, tmp_path):
        """Test that cached FASTA is reused without downloading."""
        # Mock finding proteome
        mock_find.return_value = "UP000005640"
        
        # Pre-create cached FASTA
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cached_fasta = cache_dir / "UP000005640_reviewed.fasta"
        cached_fasta.write_text(""">sp|P12345|TEST_HUMAN Test protein
PEPTIDEKR
""")
        
        # Call the function
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            reviewed=True,
            mc=0,
            min_len=1,
            max_len=100,
            cache_dir=cache_dir,
        )
        
        # Verify results
        assert len(proteins_df) > 0
        assert len(peptides_df) > 0
        
        # Verify download was NOT called (cached file was used)
        mock_download.assert_not_called()
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_reviewed_vs_all_different_cache(self, mock_find, mock_download, tmp_path):
        """Test that reviewed and all entries use different cache files."""
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|TEST
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        # Call with reviewed=True
        build_reference_proteome("homo sapiens", reviewed=True, cache_dir=cache_dir, min_len=1)
        
        # Call with reviewed=False
        mock_download.reset_mock()
        build_reference_proteome("homo sapiens", reviewed=False, cache_dir=cache_dir, min_len=1)
        
        # Both should have been downloaded (different cache files)
        assert (cache_dir / "UP000005640_reviewed.fasta").exists()
        assert (cache_dir / "UP000005640_all.fasta").exists()
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_default_cache_directory(self, mock_find, mock_download, tmp_path, monkeypatch):
        """Test that default cache directory is created."""
        # Change to tmp_path to avoid polluting the real filesystem
        monkeypatch.chdir(tmp_path)
        
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|TEST
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        # Call without specifying cache_dir (uses default)
        build_reference_proteome("homo sapiens", reviewed=True, min_len=1)
        
        # Check default cache directory was created
        default_cache = tmp_path / ".cache" / "protperties" / "uniprot_proteomes"
        assert default_cache.exists()
        assert (default_cache / "UP000005640_reviewed.fasta").exists()
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_peptide_parameters_passed_through(self, mock_find, mock_download, tmp_path):
        """Test that mc, min_len, max_len parameters are passed to from_fasta_reference."""
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            # Create a FASTA with peptides that will be within 3-8 length range
            # AKPEPTRKGHR cleaves after R (pos 6) and K (pos 7), but not after K at pos 1 (before P)
            # Generates: AKPEPTR (7) | K (1) | GHR (3) with mc=0
            # K (1) is filtered out, leaving AKPEPTR (7) and GHR (3)
            output_path.write_text(""">sp|P12345|TEST
AKPEPTRKGHR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        # Call with specific length constraints
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            reviewed=True,
            mc=0,
            min_len=3,
            max_len=8,
            cache_dir=cache_dir,
        )
        
        # Verify peptides are filtered by length
        # Should have AKPEPTR (7) and GHR (3)
        assert len(peptides_df) == 2
        assert all(peptides_df["length"] >= 3)
        assert all(peptides_df["length"] <= 8)
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_id_normalization(self, mock_find, mock_download, tmp_path):
        """Test that protein IDs are normalized."""
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|PROT_HUMAN Some protein
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            cache_dir=cache_dir,
            min_len=1,
        )
        
        # ID should be normalized to just the accession
        assert proteins_df["primary_id"].iloc[0] == "P12345"
        # Peptides should also have the normalized ID
        assert (peptides_df["primary_id"] == "P12345").all()
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_species_not_found(self, mock_find, mock_download, tmp_path):
        """Test error handling when species is not found."""
        mock_find.side_effect = RuntimeError("No reference proteome found")
        
        with pytest.raises(RuntimeError, match="No reference proteome found"):
            build_reference_proteome("nonexistent species", cache_dir=tmp_path)
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_download_failure(self, mock_find, mock_download, tmp_path):
        """Test error handling when download fails."""
        mock_find.return_value = "UP000005640"
        mock_download.side_effect = RuntimeError("Failed to download")
        
        with pytest.raises(RuntimeError, match="Failed to download"):
            build_reference_proteome("homo sapiens", cache_dir=tmp_path)
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_multiple_proteins_and_peptides(self, mock_find, mock_download, tmp_path):
        """Test with FASTA containing multiple proteins."""
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|PROT1_HUMAN Protein 1
AKRPEPTIDEKR
>sp|P67890|PROT2_HUMAN Protein 2
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            cache_dir=cache_dir,
            mc=0,
            min_len=1,
            max_len=100,
        )
        
        # Should have 2 proteins
        assert len(proteins_df) == 2
        assert set(proteins_df["primary_id"]) == {"P12345", "P67890"}
        
        # Should have peptides from both proteins
        assert len(peptides_df) > 0
        peptide_sources = set(peptides_df["primary_id"].unique())
        assert peptide_sources.issubset({"P12345", "P67890"})
    
    @patch("protperties.reference_proteome._download_proteome_fasta")
    @patch("protperties.reference_proteome._find_reference_proteome")
    def test_output_structure_matches_from_fasta_reference(self, mock_find, mock_download, tmp_path):
        """Test that output structure matches from_fasta_reference."""
        mock_find.return_value = "UP000005640"
        
        def create_fasta(proteome_id, reviewed, output_path):
            output_path.write_text(""">sp|P12345|TEST
PEPTIDEKR
""")
        mock_download.side_effect = create_fasta
        
        cache_dir = tmp_path / "cache"
        
        proteins_df, peptides_df = build_reference_proteome(
            "homo sapiens",
            cache_dir=cache_dir,
            min_len=1,
        )
        
        # Check protein DataFrame structure
        required_protein_cols = [
            "primary_id", "sequence", "level", "length", "mw", "pi",
            "gravy", "instability", "aromaticity", "aliphatic_index",
            "trypsin_sites", "expression", "header"
        ]
        for col in required_protein_cols:
            assert col in proteins_df.columns, f"Missing protein column: {col}"
        
        # Check peptide DataFrame structure
        required_peptide_cols = [
            "primary_id", "sequence", "level", "length", "missed_cleavages"
        ]
        for col in required_peptide_cols:
            assert col in peptides_df.columns, f"Missing peptide column: {col}"
        
        # Check level values
        assert (proteins_df["level"] == "protein").all()
        assert (peptides_df["level"] == "peptide").all()

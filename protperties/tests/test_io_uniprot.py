"""
Tests for io_uniprot module (UniProt sequence fetching with caching).
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from protperties.io_uniprot import (
    fetch_uniprot_sequences,
    _fetch_batch_from_uniprot,
)

# Default timeout value for testing (matches function signature default)
DEFAULT_TIMEOUT = 30




class TestFetchBatchFromUniprot:
    """Tests for batch fetching from UniProt (mocked)."""
    
    @patch("protperties.io_uniprot.requests.get")
    def test_successful_fetch_with_json_format(self, mock_get):
        """Test successful batch fetch uses accessions endpoint with json format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"primaryAccession": "P12345", "sequence": {"value": "MTEYKLVVVG"}},
                {"primaryAccession": "Q9XXX1", "sequence": {"value": "ARNDCEQ"}}
            ]
        }
        mock_get.return_value = mock_response
        
        result = _fetch_batch_from_uniprot(["P12345", "Q9XXX1"], timeout_s=DEFAULT_TIMEOUT)
        
        assert result == {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQ",
        }
        mock_get.assert_called_once()
        
        # Verify the URL is the accessions endpoint
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://rest.uniprot.org/uniprotkb/accessions"
        
        # Verify params
        params = call_args[1]["params"]
        assert params["accessions"] == "P12345,Q9XXX1"
        assert params["format"] == "json"
        assert call_args[1]["timeout"] == DEFAULT_TIMEOUT
    
    @patch("protperties.io_uniprot.requests.get")
    def test_empty_accessions(self, mock_get):
        """Test batch fetch with empty accessions list."""
        result = _fetch_batch_from_uniprot([], timeout_s=DEFAULT_TIMEOUT)
        assert result == {}
        mock_get.assert_not_called()
    
    @patch("protperties.io_uniprot.requests.get")
    def test_network_error_uses_warning(self, mock_get):
        """Test handling of network errors uses warnings.warn."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.warns(UserWarning, match="Failed to fetch batch from UniProt"):
            result = _fetch_batch_from_uniprot(["P12345"], timeout_s=DEFAULT_TIMEOUT)
        
        assert result == {}
    
    @patch("protperties.io_uniprot.requests.get")
    def test_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_get.return_value = mock_response
        
        with pytest.warns(UserWarning, match="Failed to fetch batch from UniProt"):
            result = _fetch_batch_from_uniprot(["P12345"], timeout_s=DEFAULT_TIMEOUT)
        
        assert result == {}
    
    @patch("protperties.io_uniprot.requests.get")
    def test_partial_results(self, mock_get):
        """Test when some accessions are not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"primaryAccession": "P12345", "sequence": {"value": "MTEYKLVVVG"}}
            ]
        }
        mock_get.return_value = mock_response
        
        result = _fetch_batch_from_uniprot(["P12345", "INVALID"], timeout_s=DEFAULT_TIMEOUT)
        
        # Only the valid one should be returned
        assert result == {"P12345": "MTEYKLVVVG"}


class TestFetchUniprotSequences:
    """Tests for main fetch_uniprot_sequences function."""
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_batch_query_construction(self, mock_fetch, tmp_path):
        """
        REAL TEST: Verify that fetch_uniprot_sequences with multiple accessions
        results in a single batch call with proper query construction.
        """
        # Mock the batch fetch to capture what query would be used
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQ",
        }
        
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences(
            ["P12345", "Q9XXX1"],
            cache_dir=cache_dir,
            batch_size=200,
            timeout_s=30,
        )
        
        # Verify results are correct
        assert result == {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQ",
        }
        
        # CRITICAL TEST: Verify batch fetch was called exactly ONCE
        assert mock_fetch.call_count == 1
        
        # Verify the call was made with both accessions in a single batch
        call_args = mock_fetch.call_args
        batch_accessions = call_args[0][0]
        assert set(batch_accessions) == {"P12345", "Q9XXX1"}
    
    @patch("protperties.io_uniprot.requests.get")
    def test_query_contains_comma_separated_accessions(self, mock_get, tmp_path):
        """
        REAL TEST: Verify the actual URL/params passed to requests.get contains
        the correct accessions formatting.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"primaryAccession": "P12345", "sequence": {"value": "MTEYKLVVVG"}},
                {"primaryAccession": "Q9XXX1", "sequence": {"value": "ARNDCEQ"}}
            ]
        }
        mock_get.return_value = mock_response
        
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences(
            ["P12345", "Q9XXX1"],
            cache_dir=cache_dir,
        )
        
        # Verify results
        assert len(result) == 2
        assert "P12345" in result
        assert "Q9XXX1" in result
        
        # CRITICAL TEST: Verify requests.get was called exactly ONCE
        assert mock_get.call_count == 1
        
        # Verify the query parameter structure
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        
        # CRITICAL ASSERTIONS
        assert params["accessions"] == "P12345,Q9XXX1"
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_cache_hit_no_network_call(self, mock_fetch, tmp_path):
        """
        REAL TEST: Verify that when all sequences are cached, 
        requests.get is NOT called at all.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        
        # Pre-populate cache with both sequences
        with open(cache_dir / "P12345.json", "w") as f:
            json.dump({"accession": "P12345", "sequence": "CACHED_SEQ1"}, f)
        with open(cache_dir / "Q9XXX1.json", "w") as f:
            json.dump({"accession": "Q9XXX1", "sequence": "CACHED_SEQ2"}, f)
        
        result = fetch_uniprot_sequences(
            ["P12345", "Q9XXX1"],
            cache_dir=cache_dir,
        )
        
        # Verify cached sequences are returned
        assert result == {
            "P12345": "CACHED_SEQ1",
            "Q9XXX1": "CACHED_SEQ2",
        }
        
        # CRITICAL TEST: No network call should have been made
        mock_fetch.assert_not_called()
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_missing_accession_single_warning(self, mock_fetch, tmp_path):
        """
        REAL TEST: Verify that when multiple accessions are missing,
        a SINGLE warning is issued with the total count (not per-ID spam).
        """
        # Mock returns only one of three requested accessions
        mock_fetch.return_value = {"P12345": "SEQUENCE"}
        
        cache_dir = tmp_path / "cache"
        
        # Request 3 accessions, but only 1 will be returned
        with pytest.warns(UserWarning, match=r"2 accession\(s\) could not be retrieved from UniProt"):
            result = fetch_uniprot_sequences(
                ["P12345", "INVALID1", "INVALID2"],
                cache_dir=cache_dir,
            )
        
        # Verify only the valid one is in the result
        assert result == {"P12345": "SEQUENCE"}
        assert len(result) == 1
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_basic_fetch_no_cache(self, mock_fetch, tmp_path):
        """Test basic fetching without cache hits."""
        mock_fetch.return_value = {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQ",
        }
        
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences(
            ["P12345", "Q9XXX1"],
            cache_dir=cache_dir,
            batch_size=200,
            timeout_s=30,
        )
        
        assert result == {
            "P12345": "MTEYKLVVVG",
            "Q9XXX1": "ARNDCEQ",
        }
        
        # Check that sequences were cached
        assert (cache_dir / "P12345.json").exists()
        assert (cache_dir / "Q9XXX1.json").exists()
        
        with open(cache_dir / "P12345.json") as f:
            cached = json.load(f)
            assert cached["accession"] == "P12345"
            assert cached["sequence"] == "MTEYKLVVVG"
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_partial_cache_hit(self, mock_fetch, tmp_path):
        """Test mixed cache hits and network fetches."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        
        # Cache one sequence
        with open(cache_dir / "P12345.json", "w") as f:
            json.dump({"accession": "P12345", "sequence": "CACHED"}, f)
        
        # Mock network fetch for the other
        mock_fetch.return_value = {"Q9XXX1": "FETCHED"}
        
        result = fetch_uniprot_sequences(
            ["P12345", "Q9XXX1"],
            cache_dir=cache_dir,
        )
        
        assert result == {
            "P12345": "CACHED",
            "Q9XXX1": "FETCHED",
        }
        
        # Should only fetch the uncached one
        mock_fetch.assert_called_once_with(["Q9XXX1"], DEFAULT_TIMEOUT)
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_batching(self, mock_fetch, tmp_path):
        """Test that large lists are batched."""
        # Create a list of 250 accessions
        accessions = [f"P{i:05d}" for i in range(250)]
        
        # Mock to return a sequence for each
        def mock_batch_fetch(batch, timeout):
            return {acc: f"SEQ_{acc}" for acc in batch}
        
        mock_fetch.side_effect = mock_batch_fetch
        
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences(
            accessions,
            cache_dir=cache_dir,
            batch_size=100,  # Use smaller batch for testing
        )
        
        # Should have made 3 calls (100 + 100 + 50)
        assert mock_fetch.call_count == 3
        assert len(result) == 250
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_empty_accessions_list(self, mock_fetch, tmp_path):
        """Test with empty accessions list."""
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences([], cache_dir=cache_dir)
        
        assert result == {}
        mock_fetch.assert_not_called()
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_corrupted_cache_file(self, mock_fetch, tmp_path):
        """Test handling of corrupted cache files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        
        # Create a corrupted cache file
        with open(cache_dir / "P12345.json", "w") as f:
            f.write("not valid json{{{")
        
        # Mock network fetch
        mock_fetch.return_value = {"P12345": "FETCHED"}
        
        result = fetch_uniprot_sequences(
            ["P12345"],
            cache_dir=cache_dir,
        )
        
        # Should fetch from network and overwrite corrupted cache
        assert result == {"P12345": "FETCHED"}
        mock_fetch.assert_called_once()
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_isoform_fallback(self, mock_fetch, tmp_path):
        """Test fallback to canonical accession when isoform is missing."""
        # First call gets nothing, second call (fallback) gets canonical
        def mock_fetch_side_effect(batch, timeout):
            if "A0AV96-2" in batch:
                return {} # Isoform missing
            elif "A0AV96" in batch:
                return {"A0AV96": "CANONICAL_SEQUENCE"}
            return {}
            
        mock_fetch.side_effect = mock_fetch_side_effect
        
        cache_dir = tmp_path / "cache"
        result = fetch_uniprot_sequences(
            ["A0AV96-2"],
            cache_dir=cache_dir,
        )
        
        # The result for the isoform should be the canonical sequence
        assert result == {
            "A0AV96-2": "CANONICAL_SEQUENCE"
        }
        
        # Check that isoform accessions are cached properly
        assert (cache_dir / "A0AV96-2.json").exists()
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_default_cache_dir(self, mock_fetch, tmp_path, monkeypatch):
        """Test that default cache directory is created."""
        # Change to tmp_path to avoid polluting the real filesystem
        monkeypatch.chdir(tmp_path)
        
        mock_fetch.return_value = {"P12345": "SEQUENCE"}
        
        result = fetch_uniprot_sequences(["P12345"])
        
        assert result == {"P12345": "SEQUENCE"}
        # Check default cache directory was created
        default_cache = tmp_path / ".cache" / "protperties" / "uniprot"
        assert default_cache.exists()
        assert (default_cache / "P12345.json").exists()
    
    @patch("protperties.io_uniprot._fetch_batch_from_uniprot")
    def test_cache_dir_as_string(self, mock_fetch, tmp_path):
        """Test that cache_dir can be passed as string."""
        mock_fetch.return_value = {"P12345": "SEQUENCE"}
        
        cache_dir = str(tmp_path / "cache")
        result = fetch_uniprot_sequences(
            ["P12345"],
            cache_dir=cache_dir,
        )
        
        assert result == {"P12345": "SEQUENCE"}
        assert Path(cache_dir).exists()
    
    def test_invalid_batch_size_zero(self, tmp_path):
        """Test that batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            fetch_uniprot_sequences(
                ["P12345"],
                cache_dir=tmp_path / "cache",
                batch_size=0,
            )
    
    def test_invalid_batch_size_negative(self, tmp_path):
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            fetch_uniprot_sequences(
                ["P12345"],
                cache_dir=tmp_path / "cache",
                batch_size=-10,
            )

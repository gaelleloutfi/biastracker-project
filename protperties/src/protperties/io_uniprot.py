"""
Module for fetching protein sequences from UniProt with disk caching.

This module provides functionality to retrieve protein sequences from UniProt's
REST API with support for batching requests and caching results to disk for
offline access and faster repeated queries.
"""

import json
import time
import warnings
from pathlib import Path
from urllib.parse import urlencode

import requests


def fetch_uniprot_sequences(
    accessions: list[str],
    cache_dir: str | Path = ".cache/protperties/uniprot",
    batch_size: int = 200,
    timeout_s: int = 30,
) -> dict[str, str]:
    """
    Fetch protein sequences from UniProt for a list of accessions.
    
    This function retrieves FASTA sequences from UniProt's REST API with support
    for disk caching and batch processing. Cached sequences are stored individually
    to support incremental updates and offline access.
    
    Args:
        accessions: List of UniProt accession IDs (e.g., ["P12345", "Q9XXX1"]).
        cache_dir: Directory path for caching sequences. Defaults to
            ".cache/protperties/uniprot". The directory will be created if it
            doesn't exist.
        batch_size: Maximum number of accessions to fetch in a single request.
            UniProt recommends keeping this under 200-300. Default is 200.
            Must be greater than 0.
        timeout_s: HTTP request timeout in seconds. Default is 30.
    
    Returns:
        Dictionary mapping accession IDs to their amino acid sequences (string).
        Sequences contain only amino acid letters (no FASTA headers or newlines).
        Missing or invalid accessions are not included in the result.
    
    Raises:
        ValueError: If batch_size is less than or equal to 0.
    
    Example:
        >>> sequences = fetch_uniprot_sequences(["P12345", "Q9XXX1"])
        >>> print(sequences["P12345"])
        'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM'
        
    Note:
        - Accessions with isoforms (e.g., "P12345-2") are supported.
        - The function is resilient to network errors and missing accessions.
        - Missing accessions are reported using warnings.warn().
        - Cached sequences are stored as JSON files in cache_dir.
        - Rate limiting: A 0.5 second delay is added between batches to be
          respectful to UniProt servers.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    result = {}
    to_fetch = []
    
    # Check cache first
    for acc in accessions:
        cached_file = cache_path / f"{acc}.json"
        if cached_file.exists():
            try:
                with open(cached_file, "r") as f:
                    data = json.load(f)
                    if "sequence" in data:
                        result[acc] = data["sequence"]
                        continue
            except (json.JSONDecodeError, IOError):
                # Cache file corrupted, will re-fetch
                pass
        to_fetch.append(acc)
    
    if not to_fetch:
        return result
    
    # Fetch in batches
    missing_count = 0
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        fetched = _fetch_batch_from_uniprot(batch, timeout_s)
        
        # Cache and collect results
        for acc in batch:
            if acc in fetched:
                sequence = fetched[acc]
                result[acc] = sequence
                
                # Write to cache
                cached_file = cache_path / f"{acc}.json"
                try:
                    with open(cached_file, "w") as f:
                        json.dump({"accession": acc, "sequence": sequence}, f)
                except IOError:
                    # Cache write failed, but we still have the result
                    pass
            else:
                missing_count += 1
        
        # Be polite to UniProt servers
        if i + batch_size < len(to_fetch):
            time.sleep(0.5)
    
    if missing_count > 0:
        warnings.warn(
            f"{missing_count} accession(s) could not be retrieved from UniProt",
            UserWarning,
            stacklevel=2
        )
    
    return result


def _fetch_batch_from_uniprot(
    accessions: list[str],
    timeout_s: int,
) -> dict[str, str]:
    """
    Fetch a batch of sequences from UniProt REST API.
    
    Uses UniProt's search/stream API endpoint with query syntax to retrieve 
    sequences in FASTA format for multiple accessions in a single request.
    Query format: (accession:P12345 OR accession:Q9XXX1 OR ...)
    
    Args:
        accessions: List of UniProt accession IDs to fetch.
        timeout_s: HTTP request timeout in seconds.
    
    Returns:
        Dictionary mapping accession IDs to amino acid sequences.
        Only successfully retrieved sequences are included.
    """
    if not accessions:
        return {}
    
    # Build query with OR logic: (accession:P12345 OR accession:Q9XXX1 OR ...)
    query_parts = [f"accession:{acc}" for acc in accessions]
    query = "(" + " OR ".join(query_parts) + ")"
    
    # UniProt search/stream endpoint supports query syntax
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "fasta",
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout_s)
        response.raise_for_status()
        fasta_text = response.text
    except Exception as e:
        warnings.warn(
            f"Failed to fetch batch from UniProt: {e}",
            UserWarning,
            stacklevel=2
        )
        return {}
    
    # Parse FASTA response
    return _parse_fasta(fasta_text)


def _parse_fasta(fasta_text: str) -> dict[str, str]:
    """
    Parse FASTA format text into a dictionary of sequences.
    
    Extracts accession IDs from FASTA headers and associates them with their
    sequences. Headers are expected to be in UniProt format:
    >sp|P12345|PROTEIN_NAME or >tr|Q9XXX1|PROTEIN_NAME
    
    Args:
        fasta_text: FASTA formatted text containing one or more sequences.
    
    Returns:
        Dictionary mapping accession IDs to amino acid sequences (no headers,
        no newlines, uppercase letters only).
    """
    sequences = {}
    current_acc = None
    current_seq_parts = []
    
    for line in fasta_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(">"):
            # Save previous sequence
            if current_acc is not None and current_seq_parts:
                sequences[current_acc] = "".join(current_seq_parts)
            
            # Parse new header
            # Format: >sp|P12345|PROTEIN_NAME or >tr|Q9XXX1|PROTEIN_NAME or >P12345
            header = line[1:]  # Remove '>'
            
            # Extract accession from different formats
            if "|" in header:
                # Format: sp|P12345|NAME or tr|P12345|NAME
                parts = header.split("|")
                if len(parts) >= 2:
                    current_acc = parts[1]
                else:
                    current_acc = parts[0]
            else:
                # Format: P12345 PROTEIN_NAME (take first word)
                current_acc = header.split()[0] if header else None
            
            current_seq_parts = []
        else:
            # Sequence line
            current_seq_parts.append(line)
    
    # Don't forget the last sequence
    if current_acc is not None and current_seq_parts:
        sequences[current_acc] = "".join(current_seq_parts)
    
    return sequences

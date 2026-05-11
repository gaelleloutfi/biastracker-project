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
    
    missing_count = 0
    missing_isoforms = {}
    
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
                if "-" in acc:
                    base_acc = acc.split("-")[0]
                    missing_isoforms[acc] = base_acc
                else:
                    missing_count += 1
        
        # Be polite to UniProt servers
        if i + batch_size < len(to_fetch):
            time.sleep(0.5)
            
    # Handle missing isoforms by falling back to their canonical forms
    if missing_isoforms:
        bases_to_fetch = list(set(missing_isoforms.values()))
        # Remove those we might have already fetched in this run
        bases_to_fetch = [b for b in bases_to_fetch if b not in result]
        
        # We can also check if the base is already in cache
        bases_to_actually_fetch = []
        resolved_bases = {}
        for b in bases_to_fetch:
            cached_file = cache_path / f"{b}.json"
            if cached_file.exists():
                try:
                    with open(cached_file, "r") as f:
                        data = json.load(f)
                        if "sequence" in data:
                            resolved_bases[b] = data["sequence"]
                            continue
                except (json.JSONDecodeError, IOError):
                    pass
            bases_to_actually_fetch.append(b)
            
        for i in range(0, len(bases_to_actually_fetch), batch_size):
            batch = bases_to_actually_fetch[i:i + batch_size]
            fetched = _fetch_batch_from_uniprot(batch, timeout_s)
            
            for base_acc in batch:
                if base_acc in fetched:
                    sequence = fetched[base_acc]
                    resolved_bases[base_acc] = sequence
                    
                    cached_file = cache_path / f"{base_acc}.json"
                    try:
                        with open(cached_file, "w") as f:
                            json.dump({"accession": base_acc, "sequence": sequence}, f)
                    except IOError:
                        pass
            
            if i + batch_size < len(bases_to_actually_fetch):
                time.sleep(0.5)
                
        # Now map the resolved bases back to the requested isoforms
        for iso_acc, base_acc in missing_isoforms.items():
            if base_acc in resolved_bases:
                sequence = resolved_bases[base_acc]
                result[iso_acc] = sequence
                
                # Cache the isoform as well so we don't have to resolve it again
                cached_file = cache_path / f"{iso_acc}.json"
                try:
                    with open(cached_file, "w") as f:
                        json.dump({"accession": iso_acc, "sequence": sequence}, f)
                except IOError:
                    pass
            else:
                missing_count += 1
    
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
    
    Uses UniProt's accessions endpoint to retrieve sequences in JSON format
    for multiple accessions in a single request.
    
    Args:
        accessions: List of UniProt accession IDs to fetch.
        timeout_s: HTTP request timeout in seconds.
    
    Returns:
        Dictionary mapping accession IDs (both primary and secondary) to amino
        acid sequences. Only successfully retrieved sequences are included.
    """
    if not accessions:
        return {}
    
    base_url = "https://rest.uniprot.org/uniprotkb/accessions"
    params = {
        "accessions": ",".join(accessions),
        "format": "json",
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        warnings.warn(
            f"Failed to fetch batch from UniProt: {e}",
            UserWarning,
            stacklevel=2
        )
        return {}
    
    fetched = {}
    if "results" in data:
        for entry in data["results"]:
            seq = entry.get("sequence", {}).get("value")
            if not seq:
                continue
                
            prim_acc = entry.get("primaryAccession")
            sec_accs = entry.get("secondaryAccessions", [])
            
            if prim_acc:
                fetched[prim_acc] = seq
            for sec_acc in sec_accs:
                fetched[sec_acc] = seq
                
    return fetched


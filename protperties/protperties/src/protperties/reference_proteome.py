"""
Module for fetching reference proteomes from UniProt and generating theoretical peptidomes.

This module provides functionality to query UniProt for species-specific reference
proteomes, download and cache FASTA files, and generate theoretical tryptic peptidomes
by reusing the from_fasta_reference pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from .io_fasta_reference import from_fasta_reference


def build_reference_proteome(
    species: str,
    reviewed: bool = True,
    mc: int = 0,
    min_len: int = 6,
    max_len: int = 65,
    cache_dir: str | Path = ".cache/protperties/uniprot_proteomes",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a reference proteome for a species from UniProt.
    
    This function queries UniProt for the reference proteome of a given species,
    downloads and caches the FASTA file locally, and generates a theoretical
    tryptic peptidome using the from_fasta_reference pipeline.
    
    Parameters
    ----------
    species : str
        Species name (e.g., "homo sapiens", "escherichia coli").
        Can be scientific name or common name.
    reviewed : bool, default=True
        If True, prefer reviewed (Swiss-Prot) entries over unreviewed (TrEMBL).
        When querying the proteome, this adds a filter for reviewed sequences.
    mc : int, default=0
        Maximum number of missed cleavages allowed when generating peptides.
    min_len : int, default=6
        Minimum peptide length to keep.
    max_len : int, default=65
        Maximum peptide length to keep.
    cache_dir : str or pathlib.Path, default=".cache/protperties/uniprot_proteomes"
        Directory path for caching downloaded FASTA files. The directory will
        be created if it doesn't exist. FASTA files are named by proteome ID
        and reviewed status.
    
    Returns
    -------
    proteins_df : pd.DataFrame
        DataFrame with one row per protein, containing:
        - primary_id: protein identifier
        - header: FASTA header description
        - sequence: amino acid sequence
        - expression: expression value (0.0)
        - length: sequence length
        - ... additional physicochemical properties
        - trypsin_sites: number of trypsin cleavage sites
        - aa_A, aa_C, ..., aa_Y: amino acid composition
        - level: "protein"
    peptides_df : pd.DataFrame
        DataFrame with one row per peptide, containing:
        - primary_id: source protein identifier
        - sequence: peptide amino acid sequence
        - length: peptide sequence length
        - missed_cleavages: number of missed cleavages
        - ... additional physicochemical properties
        - level: "peptide"
    
    Raises
    ------
    RuntimeError
        If no reference proteome is found for the species.
    RuntimeError
        If the FASTA download fails.
    
    Example
    -------
    >>> proteins_df, peptides_df = build_reference_proteome("homo sapiens")
    >>> print(f"Found {len(proteins_df)} proteins")
    >>> print(f"Generated {len(peptides_df)} peptides")
    
    Notes
    -----
    - The function first queries UniProt's proteomes API to find the reference
      proteome ID for the species.
    - If a cached FASTA file exists for that proteome ID, it is reused.
    - Otherwise, the FASTA file is downloaded from UniProt and cached.
    - The cached FASTA is then processed using from_fasta_reference() to generate
      the theoretical peptidome with computed physicochemical properties.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find the reference proteome ID for the species
    proteome_id = _find_reference_proteome(species)
    
    # Step 2: Generate cache filename based on proteome ID and reviewed status
    reviewed_suffix = "_reviewed" if reviewed else "_all"
    cache_filename = f"{proteome_id}{reviewed_suffix}.fasta"
    fasta_path = cache_path / cache_filename
    
    # Step 3: Download FASTA if not cached
    if not fasta_path.exists():
        _download_proteome_fasta(proteome_id, reviewed, fasta_path)
    
    # Step 4: Use from_fasta_reference to generate the peptidome
    proteins_df, peptides_df = from_fasta_reference(
        path=fasta_path,
        mc=mc,
        min_len=min_len,
        max_len=max_len,
        normalize_ids=True,
        default_expression=0.0,
    )
    
    return proteins_df, peptides_df


def _find_reference_proteome(species: str) -> str:
    """
    Find the reference proteome ID for a species.
    
    Queries UniProt's proteomes API to find the reference proteome for the
    given species. Prefers exact matches to the scientific name.
    
    Parameters
    ----------
    species : str
        Species name (scientific or common name).
    
    Returns
    -------
    str
        UniProt proteome ID (e.g., "UP000005640" for human).
    
    Raises
    ------
    RuntimeError
        If no reference proteome is found for the species.
    """
    # Build query for UniProt proteomes search
    # Format: (organism_name:species) AND (proteome_type:"Reference proteome")
    query = f'(organism_name:{species}) AND (proteome_type:"Reference proteome")'
    
    url = "https://rest.uniprot.org/proteomes/search"
    params = {
        "query": query,
        "format": "json",
        "size": 10,  # Get top 10 results
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise RuntimeError(
            f"Failed to query UniProt proteomes API for species '{species}': {e}"
        ) from e
    
    results = data.get("results", [])
    
    if not results:
        raise RuntimeError(
            f"No reference proteome found for species '{species}'. "
            "Please check the species name and try again."
        )
    
    # Take the first result (usually the most relevant)
    proteome = results[0]
    proteome_id = proteome.get("id")
    
    if not proteome_id:
        raise RuntimeError(
            f"Reference proteome found but missing ID for species '{species}'"
        )
    
    # Warn if multiple reference proteomes found
    if len(results) > 1:
        warnings.warn(
            f"Multiple reference proteomes found for '{species}'. "
            f"Using {proteome_id} ({proteome.get('taxonomy', {}).get('scientificName', 'unknown')})",
            UserWarning,
            stacklevel=3
        )
    
    return proteome_id


def _download_proteome_fasta(
    proteome_id: str,
    reviewed: bool,
    output_path: Path,
) -> None:
    """
    Download proteome FASTA file from UniProt.
    
    Downloads the complete FASTA file for a proteome from UniProt's
    uniprotkb/stream endpoint and saves it to disk.
    
    Parameters
    ----------
    proteome_id : str
        UniProt proteome ID (e.g., "UP000005640").
    reviewed : bool
        If True, only download reviewed (Swiss-Prot) entries.
        If False, download all entries (reviewed + unreviewed).
    output_path : Path
        Path where the FASTA file should be saved.
    
    Raises
    ------
    RuntimeError
        If the download fails or returns no data.
    """
    # Build query: proteome ID with optional reviewed filter
    query_parts = [f"(proteome:{proteome_id})"]
    if reviewed:
        query_parts.append("(reviewed:true)")
    query = " AND ".join(query_parts)
    
    # Use UniProt's stream endpoint for FASTA download
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "fasta",
    }
    
    try:
        response = requests.get(url, params=params, timeout=300, stream=True)
        response.raise_for_status()
        
        # Write to file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file is not empty
        if output_path.stat().st_size == 0:
            output_path.unlink()  # Clean up empty file
            raise RuntimeError(
                f"Downloaded FASTA file for proteome {proteome_id} is empty"
            )
            
    except Exception as e:
        # Clean up partial download on error
        if output_path.exists():
            output_path.unlink()
        
        raise RuntimeError(
            f"Failed to download FASTA for proteome {proteome_id}: {e}"
        ) from e

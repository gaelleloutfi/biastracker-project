"""
Utilities for normalizing UniProt accession IDs.

This module provides functions to extract and normalize UniProt accessions from
various formats commonly found in proteomics data, including:
- FASTA headers with database prefixes (sp|P12345|NAME, tr|Q9XXX1|...)
- Plain accessions (P12345, P12345-2)
- Semicolon-separated lists of accessions

Public API:
    normalize_uniprot_accession(raw: str) -> str | None
    extract_uniprot_accessions(raw: str) -> list[str]
    canonicalize_isoform(acc: str) -> str
"""
from __future__ import annotations

import re


def normalize_uniprot_accession(raw: str) -> str | None:
    """
    Extract and normalize a single UniProt accession from various formats.
    
    Accepts strings like:
    - "sp|P12345|NAME" → "P12345"
    - "tr|Q9XXX1|..." → "Q9XXX1"
    - "P12345" → "P12345"
    - "P12345-2" → "P12345-2" (isoform preserved)
    - "P12345;Q9XXX1" → "P12345" (takes first from semicolon-separated list)
    
    Parameters
    ----------
    raw : str
        Raw protein identifier string.
    
    Returns
    -------
    str or None
        Normalized UniProt accession, or None if no valid accession is found.
    
    Examples
    --------
    >>> normalize_uniprot_accession("sp|P12345|PROTEIN_NAME")
    'P12345'
    >>> normalize_uniprot_accession("P12345-2")
    'P12345-2'
    >>> normalize_uniprot_accession("P12345;Q9XXX1")
    'P12345'
    >>> normalize_uniprot_accession("invalid")
    None
    """
    if not raw or not isinstance(raw, str):
        return None
    
    raw = raw.strip()
    if not raw:
        return None
    
    # Pattern for UniProt accession: 
    # [OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
    # Optionally followed by -N for isoforms
    uniprot_pattern = r'([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?'
    
    # Try to extract from FASTA header format: sp|ACC|NAME or tr|ACC|NAME
    fasta_match = re.search(r'(?:sp|tr)\|([^|]+)\|', raw, flags = re.IGNORECASE)
    if fasta_match:
        acc_part = fasta_match.group(1)
        # Verify it matches UniProt pattern
        match = re.match(f'^{uniprot_pattern}$', acc_part)
        if match:
            return match.group(0)
    
    # Handle semicolon-separated lists - take first valid one
    if ';' in raw:
        parts = [p.strip() for p in raw.split(';')]
        for part in parts:
            result = normalize_uniprot_accession(part)
            if result:
                return result
        return None
    
    # Try to match plain accession (with or without isoform)
    match = re.match(f'^{uniprot_pattern}$', raw)
    if match:
        return match.group(0)
    
    return None


def extract_uniprot_accessions(raw: str) -> list[str]:
    """
    Extract all UniProt accessions from a string.
    
    Handles semicolon-separated lists and various formats.
    
    Parameters
    ----------
    raw : str
        Raw protein identifier string, possibly containing multiple accessions.
    
    Returns
    -------
    list[str]
        List of normalized UniProt accessions. Empty list if none found.
    
    Examples
    --------
    >>> extract_uniprot_accessions("P12345;Q9XXX1;O43663")
    ['P12345', 'Q9XXX1', 'O43663']
    >>> extract_uniprot_accessions("sp|P12345|NAME;tr|Q9XXX1|NAME2")
    ['P12345', 'Q9XXX1']
    >>> extract_uniprot_accessions("invalid")
    []
    """
    if not raw or not isinstance(raw, str):
        return []
    
    raw = raw.strip()
    if not raw:
        return []
    
    # Split by semicolon and normalize each part
    parts = [p.strip() for p in raw.split(';')]
    accessions = []
    
    for part in parts:
        if not part:
            continue
        acc = normalize_uniprot_accession(part)
        if acc:
            accessions.append(acc)
    
    return accessions


def canonicalize_isoform(acc: str) -> str:
    """
    Remove isoform suffix from a UniProt accession.
    
    Converts isoform-specific accessions to their canonical form.
    
    **Note**: This function only works on plain accessions. If your input contains
    FASTA headers (e.g., "sp|P12345-2|NAME"), call `normalize_uniprot_accession`
    first to extract the accession before canonicalizing.
    
    Parameters
    ----------
    acc : str
        UniProt accession, possibly with isoform suffix (e.g., "P12345-2").
    
    Returns
    -------
    str
        Canonical accession without isoform suffix (e.g., "P12345").
        If input doesn't look like a UniProt accession, returns it unchanged.
    
    Examples
    --------
    >>> canonicalize_isoform("P12345-2")
    'P12345'
    >>> canonicalize_isoform("P12345")
    'P12345'
    >>> # For FASTA headers, extract accession first:
    >>> acc = normalize_uniprot_accession("sp|P12345-2|NAME")
    >>> canonicalize_isoform(acc)
    'P12345'
    """
    if not acc or not isinstance(acc, str):
        return acc
    
    # Only strip isoform from plain accessions
    # Pattern: accession followed by -digits
    match = re.match(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})-\d+$', acc)
    if match:
        return match.group(1)
    
    return acc

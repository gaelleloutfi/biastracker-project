"""
PTR (Protein Turnover Rate) annotation for HUMAN proteins.

This module provides functionality to annotate protein DataFrames with PTR values
from the Human Proteome PTR reference table.

Public API:
    add_ptr_annotation(proteins_df, ptr_table_path, ...) -> pd.DataFrame
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .id_utils import normalize_uniprot_accession, canonicalize_isoform

if TYPE_CHECKING:
    from typing import Union


def add_ptr_annotation(
    proteins_df: pd.DataFrame,
    ptr_table_path: Union[str, Path],
    accession_col: str = "primary_id",
    species: Union[str, None] = None,
    strict_human: bool = True,
    ptr_out_col: str = "PTR_AML",
) -> pd.DataFrame:
    """
    Annotate proteins with PTR (Protein Turnover Rate) values for HUMAN proteins.
    
    This function adds PTR annotation from the Human Proteome PTR reference table.
    By default, it only annotates HUMAN proteins (strict_human=True) and requires
    explicit species information to prevent accidental mis-annotation.
    
    Parameters
    ----------
    proteins_df : pd.DataFrame
        DataFrame containing protein information with UniProt accessions.
    ptr_table_path : str or Path
        Path to the PTR reference Excel file (Human_Proteome_PTR.xlsx).
    accession_col : str, optional
        Column name in proteins_df containing UniProt accessions (default: "primary_id").
    species : str or None, optional
        Species identifier. Accepts "Homo sapiens", "human", or "9606" (case-insensitive).
        Required when strict_human=True for annotation to proceed.
    strict_human : bool, optional
        If True (default), only annotate when species is explicitly human.
        If False, proceed with annotation regardless of species.
    ptr_out_col : str, optional
        Output column name for PTR values (default: "PTR_AML").
    
    Returns
    -------
    pd.DataFrame
        Copy of proteins_df with PTR annotation added in ptr_out_col column.
        Returns unchanged copy if human-only conditions are not met.
    
    Notes
    -----
    - PTR values of 0 are treated as missing (converted to NaN).
    - Non-numeric PTR values are converted to NaN.
    - UniProt accessions are normalized to handle isoforms (e.g., P12345-2 matches P12345).
    - When strict_human=True and species is None or non-human, returns df unchanged with a warning.
    
    Examples
    --------
    >>> # Annotate human proteins
    >>> df = pd.DataFrame({'primary_id': ['P12345', 'Q9XXX1-2']})
    >>> result = add_ptr_annotation(df, 'data/Human_Proteome_PTR.xlsx', species='human')
    
    >>> # Non-human species returns unchanged
    >>> mouse_df = pd.DataFrame({'primary_id': ['P12345']})
    >>> result = add_ptr_annotation(mouse_df, 'data/Human_Proteome_PTR.xlsx', species='mouse')
    >>> # result equals mouse_df (no annotation added)
    """
    # A) HUMAN-ONLY rule
    if strict_human:
        if species is None:
            warnings.warn(
                "PTR annotation requires explicit species specification. "
                "No annotation applied (species=None).",
                stacklevel=2,
            )
            return proteins_df.copy()
        
        # Check if species is human (case-insensitive)
        species_lower = str(species).lower().strip()
        human_identifiers = {"homo sapiens", "human", "9606"}
        
        if species_lower not in human_identifiers:
            warnings.warn(
                f"PTR annotation is only available for HUMAN proteins. "
                f"Species '{species}' is not recognized as human. No annotation applied.",
                stacklevel=2,
            )
            return proteins_df.copy()
    
    # B) Load PTR table
    ptr_table_path = Path(ptr_table_path)
    if not ptr_table_path.exists():
        raise FileNotFoundError(f"PTR table not found: {ptr_table_path}")
    
    # Read Excel file
    ptr_df = pd.read_excel(ptr_table_path)
    
    # Validate expected columns
    expected_cols = {"Entry", "Entry Name", "PTR AML"}
    if not expected_cols.issubset(ptr_df.columns):
        missing = expected_cols - set(ptr_df.columns)
        raise ValueError(
            f"PTR table missing expected columns: {missing}. "
            f"Found columns: {list(ptr_df.columns)}"
        )
    
    # Helper function to normalize and canonicalize accessions
    def normalize_and_canonicalize(x):
        if pd.notna(x):
            acc = normalize_uniprot_accession(str(x))
            if acc:
                return canonicalize_isoform(acc)
        return None
    
    # Normalize UniProt accessions from Entry column
    ptr_df["accession_norm"] = ptr_df["Entry"].apply(normalize_and_canonicalize)
    
    # Drop rows with missing normalized accessions
    ptr_df = ptr_df[ptr_df["accession_norm"].notna()].copy()
    
    # C) Clean PTR values
    # Convert PTR AML to numeric (coerce errors to NaN)
    ptr_df["PTR_cleaned"] = pd.to_numeric(ptr_df["PTR AML"], errors="coerce")
    
    # Replace 0 with NaN (treat as missing)
    ptr_df["PTR_cleaned"] = ptr_df["PTR_cleaned"].replace(0, pd.NA)
    
    # Create mapping from normalized accession to PTR value
    ptr_mapping = ptr_df.set_index("accession_norm")["PTR_cleaned"].to_dict()
    
    # D) Join
    # Make a copy to avoid modifying original
    result_df = proteins_df.copy()
    
    # Check if accession column exists
    if accession_col not in result_df.columns:
        raise ValueError(
            f"Accession column '{accession_col}' not found in proteins_df. "
            f"Available columns: {list(result_df.columns)}"
        )
    
    # Normalize accessions in proteins_df (canonicalize isoforms)
    result_df["_accession_norm"] = result_df[accession_col].apply(normalize_and_canonicalize)
    
    # Map PTR values using normalized accessions
    result_df[ptr_out_col] = result_df["_accession_norm"].map(ptr_mapping)
    
    # Drop temporary normalization column
    result_df = result_df.drop(columns=["_accession_norm"])
    
    # E) Minimal logging
    # Count how many proteins have PTR values
    n_annotated = result_df[ptr_out_col].notna().sum()
    n_total = len(result_df)
    
    if n_annotated == 0 and n_total > 0:
        warnings.warn(
            f"No PTR values found for any of the {n_total} proteins. "
            "Check that accessions match the PTR reference table.",
            stacklevel=2,
        )
    
    return result_df

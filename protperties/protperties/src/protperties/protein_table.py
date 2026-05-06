"""
Protein-level table builder that joins UniProt sequences,
optional expression values, and computed physicochemical properties.

This module provides a function to build a canonical protein-level table
compatible with ensure_sequence_table(), without depending on any LFQ loader.
Expression data is optional.
"""
from __future__ import annotations

import pandas as pd

from .id_utils import normalize_uniprot_accession
from .io_uniprot import fetch_uniprot_sequences
from .features_basic import basic_props, amino_acid_composition
from .features_digest import trypsin_sites
from .schema import ensure_sequence_table


def build_protein_table(
    ids_df: pd.DataFrame,
    accession_col: str = "primary_id",
    expression_col: str | None = None,
    default_expression: float = 0.0,
    ph: float = 8.5,
    fetch_missing_sequences: bool = True,
    **uniprot_fetch_kwargs,
) -> pd.DataFrame:
    """
    Build a protein-level table with sequences, expression, and physicochemical properties.
    
    This function takes a DataFrame with UniProt accessions, optionally joins expression data,
    fetches sequences from UniProt, computes physicochemical properties, and returns a
    standardized protein-level table.
    
    Parameters
    ----------
    ids_df : pd.DataFrame
        Input DataFrame containing at least a column with UniProt accessions.
    accession_col : str, default="primary_id"
        Name of the column in ids_df containing UniProt accessions.
    expression_col : str or None, default=None
        Optional name of a column in ids_df containing expression values.
        If None, all proteins get default_expression.
    default_expression : float, default=0.0
        Default expression value to use when expression_col is None or has missing values.
    ph : float, default=8.5
        pH value for computing charge_at_pH in basic_props().
    fetch_missing_sequences : bool, default=True
        If True, fetches sequences from UniProt. If False, assumes sequences already
        exist in ids_df['sequence'].
    **uniprot_fetch_kwargs
        Additional keyword arguments passed to fetch_uniprot_sequences()
        (e.g., cache_dir, batch_size, timeout_s).
    
    Returns
    -------
    pd.DataFrame
        Standardized protein-level table containing:
        - primary_id: Normalized UniProt accession
        - sequence: Amino acid sequence
        - expression: Expression value (numeric)
        - level: "protein" (added by ensure_sequence_table)
        - trypsin_sites: Number of trypsin cleavage sites
        - aa_A, aa_C, ..., aa_Y: Amino acid composition (20 columns)
        - All properties from basic_props() (length, mw, pi, gravy, etc.)
    
    Raises
    ------
    ValueError
        If accession_col does not exist in ids_df.
        If no sequence column is found when fetch_missing_sequences=False.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from protperties.protein_table import build_protein_table
    >>> 
    >>> # With expression data
    >>> df = pd.DataFrame({
    ...     "accession": ["P12345", "Q9XXX1"],
    ...     "intensity": [1000.0, 2000.0]
    ... })
    >>> result = build_protein_table(df, accession_col="accession", expression_col="intensity")
    >>> 
    >>> # Without expression data
    >>> df = pd.DataFrame({"accession": ["P12345", "Q9XXX1"]})
    >>> result = build_protein_table(df, accession_col="accession")
    """
    # Step 1: Validate that accession_col exists in ids_df
    if accession_col not in ids_df.columns:
        raise ValueError(
            f"accession_col '{accession_col}' not found in ids_df. "
            f"Available columns: {list(ids_df.columns)}"
        )
    
    # Create a working copy to avoid modifying input
    df = ids_df.copy()
    
    # Step 2: Normalize accessions using id_utils.normalize_uniprot_accession
    df["primary_id"] = df[accession_col].apply(normalize_uniprot_accession)
    
    # Remove rows where normalization failed (None values)
    df = df.dropna(subset=["primary_id"])
    
    if len(df) == 0:
        original_values = list(ids_df[accession_col].unique())[:5]
        raise ValueError(
            f"No valid UniProt accessions found after normalization. "
            f"Original values (first 5): {original_values}"
        )
    
    # Step 3: Attach expression
    if expression_col is not None and expression_col in df.columns:
        df["expression"] = pd.to_numeric(df[expression_col], errors="coerce")
        df["expression"] = df["expression"].fillna(default_expression)
    else:
        df["expression"] = default_expression
    
    # Step 4: Fetch UniProt sequences
    if fetch_missing_sequences:
        accessions_list = df["primary_id"].unique().tolist()
        sequences_dict = fetch_uniprot_sequences(accessions_list, **uniprot_fetch_kwargs)
        
        # Map sequences back to DataFrame
        df["sequence"] = df["primary_id"].map(sequences_dict)
        
        # Remove rows where sequence fetch failed
        df = df.dropna(subset=["sequence"])
        
        if len(df) == 0:
            raise ValueError(
                f"No sequences could be fetched from UniProt for accessions: {accessions_list}"
            )
    else:
        # Require sequence column to exist
        if "sequence" not in df.columns:
            raise ValueError(
                "fetch_missing_sequences=False but no 'sequence' column found in ids_df"
            )
        
        # Drop rows where sequence is NA or empty string (combined operation)
        df = df[df["sequence"].notna() & (df["sequence"].astype(str).str.strip() != "")]
        
        # Reset index after dropping
        df = df.reset_index(drop=True)
        
        # Raise error if no rows remain
        if len(df) == 0:
            raise ValueError(
                "fetch_missing_sequences=False: All sequences were NA or empty. "
                "No valid protein data remains after filtering."
            )
    
    # Step 5: Compute physicochemical properties
    # At this point, all sequences should be valid (non-empty strings)
    props_list = []
    for seq in df["sequence"]:
        props = basic_props(seq, ph=ph)
        props_list.append(props)
    
    # Convert list of dicts to DataFrame and concatenate
    props_df = pd.DataFrame(props_list)
    
    # Reset indices to ensure proper concatenation
    df = df.reset_index(drop=True)
    props_df = props_df.reset_index(drop=True)
    
    df = pd.concat([df, props_df], axis=1)
    
    # Step 6: Compute digestion-related properties
    df["trypsin_sites"] = df["sequence"].apply(trypsin_sites)
    
    # Step 7: Compute amino acid composition
    aa_comp_list = []
    for seq in df["sequence"]:
        aa_comp = amino_acid_composition(seq)
        aa_comp_list.append(aa_comp)
    
    aa_comp_df = pd.DataFrame(aa_comp_list)
    
    # Prefix amino acid columns with "aa_"
    aa_comp_df = aa_comp_df.rename(columns={aa: f"aa_{aa}" for aa in aa_comp_df.columns})
    
    # Reset indices and concatenate
    aa_comp_df = aa_comp_df.reset_index(drop=True)
    df = pd.concat([df, aa_comp_df], axis=1)
    
    # Step 8: Standardize final table
    df = ensure_sequence_table(df, level="protein", id_col="primary_id")
    
    # Step 9: Return final DataFrame
    return df

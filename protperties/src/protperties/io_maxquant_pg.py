"""
IO utilities for MaxQuant proteinGroups.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .id_utils import extract_uniprot_accessions
from .schema import ensure_sequence_table
from .protein_table import build_protein_table


def from_maxquant_proteingroups(
    path: str | Path,
    protein_id_col: str = "Protein IDs",
    lfq_prefix: str = "LFQ intensity ",
    statistic: Literal["mean", "median"] = "mean",
    drop_zeros: bool = True,
    split_ids: bool = True,
    fetch_sequences: bool = False,
    **build_kwargs,
) -> pd.DataFrame:
    """
    Load a MaxQuant proteinGroups.txt file, compute LFQ expression, and return
    a standardized protein-level DataFrame.
    
    Note: Protein-level expression tables may not contain sequences. The resulting
    DataFrame will have a boolean `has_sequence` column. Physicochemical properties 
    are only computed when sequences are available or successfully fetched.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 2) Read tab-separated file
    df = pd.read_csv(path, sep="\t", low_memory=False)

    if protein_id_col not in df.columns:
        raise ValueError(f"Missing '{protein_id_col}' column.")

    # 3) Filter out Reverse and Contaminant
    for col in ["Reverse", "Potential contaminant", "Only identified by site"]:
        if col in df.columns:
            if col == "Only identified by site":
                is_plus = df[col].fillna("").astype(str).str.strip() == "+"
                df = df[~is_plus]
            else:
                df = df[df[col] != "+"]

    # 4) Detect LFQ columns
    lfq_cols = [c for c in df.columns if c.startswith(lfq_prefix)]
    if not lfq_cols:
        raise ValueError(f"No columns found with prefix '{lfq_prefix}'")

    # 5) Replace zeros
    expr_df = df[lfq_cols].copy()
    if drop_zeros:
        expr_df = expr_df.replace(0.0, np.nan)
    
    # Compute n_samples_used
    df["n_samples_used"] = expr_df.notna().sum(axis=1)

    # 6) Compute statistic
    if statistic == "mean":
        df["mean_lfq"] = expr_df.mean(axis=1)
        df["expression"] = df["mean_lfq"]
    elif statistic == "median":
        df["median_lfq"] = expr_df.median(axis=1)
        df["expression"] = df["median_lfq"]
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    
    df["lfq_statistic"] = statistic

    # 7 & 8) ID processing (split_ids)
    if split_ids:
        # Extract all accessions as a list
        df["primary_id"] = df[protein_id_col].apply(
            lambda x: extract_uniprot_accessions(str(x)) if pd.notna(x) else []
        )
        # Explode into multiple rows
        df = df.explode("primary_id")
        # Ensure exploded None values are dropped
        df = df.dropna(subset=["primary_id"]).reset_index(drop=True)
    else:
        # Take the first UniProt ID for each group
        def _extract_primary(x):
            if pd.isna(x): return None
            accs = extract_uniprot_accessions(str(x))
            return accs[0] if accs else None

        df["primary_id"] = df[protein_id_col].apply(_extract_primary)
        df = df.dropna(subset=["primary_id"])
    
    if df.empty:
        df["sequence"] = pd.Series(dtype=str)
        df["has_sequence"] = False
        df["lfq_statistic"] = statistic
        return ensure_sequence_table(df, level="protein", id_col="primary_id")

    # 10 & 11) Fetch sequences and properties if requested
    if fetch_sequences:
        res = build_protein_table(
            df,
            accession_col="primary_id",
            expression_col="expression",
            fetch_missing_sequences=True,
            **build_kwargs
        )
        res["has_sequence"] = True
        return res
    else:
        if "Sequence" in df.columns:
            df = df.rename(columns={"Sequence": "sequence"})
            df["has_sequence"] = True
        elif "sequence" not in df.columns:
            df["sequence"] = pd.NA
            df["has_sequence"] = False
        else:
            df["has_sequence"] = df["sequence"].notna()
            
        return ensure_sequence_table(df, level="protein", id_col="primary_id")

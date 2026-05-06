"""
IO utilities for DIA-NN report.pg_matrix.tsv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .id_utils import extract_uniprot_accessions
from .schema import ensure_sequence_table
from .protein_table import build_protein_table


def from_diann_pg_matrix(
    path: str | Path,
    protein_id_col_candidates: tuple[str, ...] = ("Protein.Ids", "Protein.Group", "Protein"),
    sample_cols: list[str] | None = None,
    statistic: Literal["mean", "median"] = "mean",
    drop_zeros: bool = True,
    fetch_sequences: bool = False,
    **build_kwargs,
) -> pd.DataFrame:
    """
    Load a DIA-NN report.pg_matrix.tsv file, compute expression across samples,
    and return a standardized protein-level DataFrame.
    
    Note: Protein-level expression tables may not contain sequences. The resulting
    DataFrame will have a boolean `has_sequence` column. Physicochemical properties 
    are only computed when sequences are available or successfully fetched.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read tab-separated file
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # 2) Identify the protein ID column
    id_col = next((c for c in protein_id_col_candidates if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"None of the candidate protein ID columns found: {protein_id_col_candidates}")

    # 3 & 4) Sample columns logic
    annotation_cols = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"}
    
    if sample_cols is not None:
        missing = [c for c in sample_cols if c not in df.columns]
        if missing:
            raise ValueError(f"The following explicit sample columns are missing: {missing}")
        final_sample_cols = sample_cols
    else:
        if "First.Protein.Description" in df.columns:
            desc_idx = df.columns.get_loc("First.Protein.Description")
            final_sample_cols = df.columns[desc_idx + 1:].tolist()
            for c in final_sample_cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            potential_sample_cols = [c for c in df.columns if c not in annotation_cols and c != id_col]
            final_sample_cols = []
            for c in potential_sample_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    final_sample_cols.append(c)
                else:
                    converted = pd.to_numeric(df[c], errors="coerce")
                    if len(df) > 0 and converted.notna().sum() / len(df) >= 0.8:
                        df[c] = converted
                        final_sample_cols.append(c)

    if not final_sample_cols:
        raise ValueError("No numeric sample columns found in the matrix.")

    # 5) Compute expression
    expr_df = df[final_sample_cols].copy()
    if drop_zeros:
        expr_df = expr_df.replace(0.0, np.nan)
    
    df["n_samples_used"] = expr_df.notna().sum(axis=1)

    # 6 & 7) Compute statistic and expression
    if statistic == "mean":
        df["mean_lfq"] = expr_df.mean(axis=1)
        df["expression"] = df["mean_lfq"]
    elif statistic == "median":
        df["median_lfq"] = expr_df.median(axis=1)
        df["expression"] = df["median_lfq"]
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    
    df["lfq_statistic"] = statistic

    # 8) Extract UniProt ID
    def _extract_primary(x):
        if pd.isna(x): return None
        accs = extract_uniprot_accessions(str(x))
        return accs[0] if accs else None

    df["primary_id"] = df[id_col].apply(_extract_primary)
    df = df.dropna(subset=["primary_id"])
    
    if df.empty:
        df["sequence"] = pd.Series(dtype=str)
        df["has_sequence"] = False
        df["lfq_statistic"] = statistic
        return ensure_sequence_table(df, level="protein", id_col="primary_id")

    # 11) Fetch sequences and properties if requested
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

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
    protein_id_col: str = "Majority protein IDs",
    lfq_prefix: str = "LFQ intensity ",
    statistic: Literal["mean", "median"] = "mean",
    drop_zeros: bool = True,
    split_ids: bool = False,
    keep_contaminants: bool = True,
    fetch_sequences: bool = False,
    **build_kwargs,
) -> pd.DataFrame:
    """
    Load a MaxQuant proteinGroups.txt file, compute LFQ expression, and return
    a standardized protein-level DataFrame.

    One row is emitted per protein group. By default the representative accession
    is the *first* entry of the ``Majority protein IDs`` column (the leading
    proteins that carry the majority of the group's peptides); this avoids the
    row-count inflation that ``split_ids=True`` produces by exploding every group
    into one row per accession. If ``protein_id_col`` is absent, the loader falls
    back to ``Protein IDs``.

    Reverse hits and "Only identified by site" (OIBS) entries are always removed.
    Potential contaminants are, by default (``keep_contaminants=True``), retained
    as **ID-only** rows: they are flagged with ``is_contaminant=True`` and no
    sequence is fetched for them, so they remain available for contaminant ORA
    without biasing the physicochemical statistics (their property columns stay
    NaN and are dropped by downstream plots/stats).

    Note: Protein-level expression tables may not contain sequences. The resulting
    DataFrame will have a boolean `has_sequence` column. Physicochemical properties
    are only computed when sequences are available or successfully fetched.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 2) Read tab-separated file
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Resolve the protein ID column, falling back to 'Protein IDs' when the
    # preferred 'Majority protein IDs' column is not present.
    if protein_id_col not in df.columns:
        fallback = "Protein IDs" if protein_id_col != "Protein IDs" else "Majority protein IDs"
        if fallback in df.columns:
            protein_id_col = fallback
        else:
            raise ValueError(
                f"Missing protein ID column: neither '{protein_id_col}' nor '{fallback}' present."
            )

    # 3) Always remove Reverse and 'Only identified by site' (OIBS) hits.
    for col in ["Reverse", "Only identified by site"]:
        if col in df.columns:
            is_plus = df[col].fillna("").astype(str).str.strip() == "+"
            df = df[~is_plus]

    # 3b) Contaminants: flag them (keep_contaminants) or drop them.
    if "Potential contaminant" in df.columns:
        is_contam = df["Potential contaminant"].fillna("").astype(str).str.strip() == "+"
    else:
        is_contam = pd.Series(False, index=df.index)
    if keep_contaminants:
        df = df.copy()
        df["is_contaminant"] = is_contam.to_numpy()
    else:
        df = df[~is_contam].copy()
        df["is_contaminant"] = False
    df = df.reset_index(drop=True)

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
        df = df.dropna(subset=["primary_id"]).reset_index(drop=True)

    if "is_contaminant" not in df.columns:
        df["is_contaminant"] = False

    if df.empty:
        df["sequence"] = pd.Series(dtype=str)
        df["has_sequence"] = False
        df["lfq_statistic"] = statistic
        return ensure_sequence_table(df, level="protein", id_col="primary_id")

    # 10 & 11) Fetch sequences and properties if requested
    if fetch_sequences:
        # Contaminants are kept as ID-only rows: no sequence is fetched for them,
        # so they stay out of the physicochemical statistics while remaining
        # available (by primary_id) for contaminant ORA.
        contam = df[df["is_contaminant"]].copy()
        clean = df[~df["is_contaminant"]].copy()

        if not clean.empty:
            res = build_protein_table(
                clean,
                accession_col="primary_id",
                expression_col="expression",
                fetch_missing_sequences=True,
                **build_kwargs
            )
            res["has_sequence"] = True
        else:
            res = pd.DataFrame()

        if not contam.empty:
            contam["sequence"] = pd.NA
            contam["has_sequence"] = False
            combined = pd.concat([res, contam], ignore_index=True) if not res.empty else contam
        else:
            combined = res

        return ensure_sequence_table(combined, level="protein", id_col="primary_id")
    else:
        if "Sequence" in df.columns:
            df = df.rename(columns={"Sequence": "sequence"})
            df["has_sequence"] = True
        elif "sequence" not in df.columns:
            df["sequence"] = pd.NA
            df["has_sequence"] = False
        else:
            df["has_sequence"] = df["sequence"].notna()

        # Contaminants are ID-only: drop any sequence so they never enter the
        # physicochemical statistics.
        if df["is_contaminant"].any():
            df.loc[df["is_contaminant"], "sequence"] = pd.NA
            df.loc[df["is_contaminant"], "has_sequence"] = False

        return ensure_sequence_table(df, level="protein", id_col="primary_id")

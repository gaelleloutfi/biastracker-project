"""
IO utilities for tabular inputs (DIA-NN .parquet).

This module: 
- loads DIA-NN parquet,
- applies recommended filters (q-values, decoys), see GitHub,
- computes peptide-level properties from Stripped.Sequence,
- optionally deduplicates (e.g., one row per Run x Precursor.Id), 
- provides summaries like % missed cleavages per group. 

List of peptide-level properties that are being computed here: 
length, molecular weight, isoelectric point (pI), gravy, instability,
aromaticity, aliphatic_index, ext_reduced, ext_cystine, missed_cleavages
Public API:
    from_diann_parquet(...)
    summarize_miscleavage(...)
"""

# Basically for each peptide sequence in our DiaNN output (Stripped.Sequence),
# we calculate properties that describe that specific peptide molecule, 
# independent of which protein or experimental condition it came from

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd

from .features_basic import basic_props
from .features_digest import missed_cleavages_in_peptide as mc_pep
from .schema import ensure_sequence_table
from .id_utils import normalize_uniprot_accession
from ._compute_utils import compute_props_on_series

# --------------------------------------------------------------
# Column presets (robust to missing columns) 


CORE_COLS = [
    "Run",
    "Precursor.Id",
    "Stripped.Sequence",
    "Precursor.Charge",
    "Protein.Group",
    "Protein.Ids",
    "Protein.Names",
    "Genes",
    "PEP",
    "Q.Value",
    "Global.Q.Value",
    "Lib.Q.Value",
    "Decoy",
    "Precursor.Normalised",
    "Precursor.Quantity",    
]

@dataclass 
class FilterConfig: 
    """ Recommended DIA-NN main-report filters.
    
    These filters resitrict which rows are retained from the DIA-NN .parquet table
    before computing peptide properties. Each filter is only applied if the 
    corresponding column exists in the file. 

    Attributes:
    -----------
    pep_max : float, default = 0.01
        Upper bound for the posterior error probability (PEP).
        This is the main filter: keeps only precursors identified at or 
        below a given error probability threshold (1% by default).

    q_value_max : float, default = 0.01
        Upper bound for the run-specific **precursor q-value** (`Q.Value`).
        This is the most common filter: keeps only precursors identified at or
        below a give false-discovery rate (FDR) threshold (1% by default).

    lib_q_value_max : float | None, default = 0.01
        Upper bound for the **library entry q-value** (`Lib.Q.Value`).
        Applies when using a spectral library or Match-Between-Runs (MBR).
        Set to None to skip this filter. 
    
    global_q_value_max : float | None, default = 0.01
        Upper bound for the **global precursor q-value** (`Global.Q.Value`),
        which reflects FDR across the entire dataset rather than within a single run. 
        Usually combined with `lib_q_value_max` when using a shared library. 

    pg_q_value_max : float | None, default = None
        Upper bound for the **protein-group q-value** (`PG.Q.Value`).
        Used when u also want to discard low-confidence protein groups.
        Typica thresholds range from 0.01 to 0.05. Set to None to disable. 

    channel_q_value_max : float | None, default = None
        Maximum allowed **channel q-value** (`Channel.Q.Value`).
        rlevant for multiplexed (plexDIA) experiments.
        Values around 0.01-0.5 are common depending on data quality.

    exclude_decoys : bool, default = True
        Whether to remove **decoy entries** (`Decoy ==1`).
        Decoys are synthetic sequences used to estimate FDR, and should
        normally be excluded from quantitative analyses. 
    
    """
    pep_max: float = 0.01
    q_value_max: Optional[float] = 0.01
    lib_q_value_max: Optional[float] = 0.01
    global_q_value_max: Optional[float] = 0.01
    pg_q_value_max: Optional[float] = None
    channel_q_value_max: Optional[float] = None
    exclude_decoys: bool = True
    
@dataclass
class DedupConfig:
    """
    Deduplication strategy after computing properties.
    - keys: group keys to deduplicate on (default one row per Run x Precursor.Id)
    - pick: "max" or "min" on `on_col` (falls back to first non-null if column missing)
    - on_col: column to choose representative row (e.g., Precursor.Normalised, Precursor.Quantity)    
    """
    keys: tuple = ("Run","Precursor.Id")
    pick: str = "max"
    on_col: str ="Precursor.Normalised"
    # we chose to do the deduplication after computing the properties 
    # because even if for now the deduplication picks based on `Precursor.Normalised`,
    # in the future we might decide to pick based on something like:
    # - pepetides with the lowest instability (from computed properties), or
    # - the most stable average MW per protein group. 

def _existing_columns(df: pd.DataFrame, wanted: Iterable[str]) -> List[str]:
    """
    Return only the columns from `wanted` that are actually present in the DataFrame.

    This helper prevents KeyErrors when selecting columns from DIA-NN outputs,
    since not all expected fields (e.g. "Lib.Q.Value", "Channel.Q.Value") are guaranteed
    to exist in every export.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (typically loaded from DIA-NN .parquet).
    wanted : Iterable[str]
        List or iterable of desired column names.

    Returns
    -------
    list of str
        Subset of `wanted` that are found in `df.columns`.
    """
    cols = [c for c in wanted if c in df.columns]
    return cols

def _apply_filters(df: pd.DataFrame, cfg: FilterConfig):
    """
    Apply DIA-NN confidence filters to a DataFrame, skipping those whose columns
    are absent.

    This function uses the thresholds defined in a `FilterConfig` object to
    remove low-confidence or decoy identifications before computing properties.

    Applied in this order:
        1. Exclude decoys (`Decoy == 1`) if enabled.
        2. Filter rows with `PEP` > cfg.pep_max.
        3. Optionally filter on Q.value, Lib.Q.Value, Global.Q.Value,
           PG.Q.Value, and Channel.Q.Value if thresholds are defined.

    Parameters
    ----------
    df : pandas.DataFrame
        DIA-NN output table (main .parquet report).
    cfg : FilterConfig
        Configuration with maximum pep, q-values and flags.

    Returns
    -------
    pandas.DataFrame
        Filtered table, re-indexed to consecutive rows.
    """
    mask = pd.Series(True, index=df.index)
    if cfg.exclude_decoys and "Decoy" in df.columns:
        mask &= (df["Decoy"]==0) | (df["Decoy"].isna())

    if "PEP" in df.columns and cfg.pep_max is not None:
        mask &= df["PEP"] <= cfg.pep_max

    if "Q.Value" in df.columns and cfg.q_value_max is not None:
        mask &= df["Q.Value"] <= cfg.q_value_max

    if cfg.lib_q_value_max is not None and "Lib.Q.Value" in df.columns:
        mask &= df["Lib.Q.Value"] <= cfg.lib_q_value_max

    if cfg.global_q_value_max is not None and "Global.Q.Value" in df.columns:
        mask &= df["Global.Q.Value"] <= cfg.global_q_value_max

    if cfg.pg_q_value_max is not None and "PG.Q.Value" in df.columns:
        mask &= df["PG.Q.Value"] <= cfg.pg_q_value_max

    if cfg.channel_q_value_max is not None and "Channel.Q.Value" in df.columns:
        mask &= df["Channel.Q.Value"] <= cfg.channel_q_value_max

    return df.loc[mask].reset_index(drop=True)

def _compute_props_on_series(seq: pd.Series, ph: float = 8.5) -> pd.DataFrame:
    """
    Compute physicochemical and digestion properties for each peptide sequence in a Series.
    
    This is a thin wrapper around compute_props_on_series() from _compute_utils.
    Kept for backward compatibility within this module.
    
    Parameters
    ----------
    seq : pandas.Series
        Series of peptide sequences (e.g., DIA-NN `Stripped.Sequence`).

    Returns
    -------
    pandas.DataFrame
        One row per peptide sequence with computed properties.
    """
    return compute_props_on_series(seq, include_missed_cleavages=True, ph=ph)


def _deduplicate(df: pd.DataFrame, cfg: DedupConfig) -> pd.DataFrame:
    """
    Collapse duplicate entries (e.g. same peptide precursor across channels)
    into a single representative row.

    Deduplication is typically done *after* computing properties, so that
    all rows have identical derived columns and we can select the representative
    row based on quantitative criteria (e.g. highest intensity).

    Logic:
      • If key columns (`cfg.keys`) are missing → return df unchanged.
      • If `cfg.on_col` (column to choose representative) is missing →
        keep the first non-null row per group.
      • Otherwise, for each group defined by `cfg.keys`, keep the row
        with min/max value of `cfg.on_col` depending on `cfg.pick`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing computed peptide properties.
    cfg : DedupConfig
        Configuration specifying which columns define duplicates and how
        to choose a representative row.

    Returns
    -------
    pandas.DataFrame
        Deduplicated DataFrame, re-indexed.
    """
    # If key columns missing, skip dedup.
    if not set(cfg.keys).issubset(df.columns):
        return df

    # If 'on_col' missing, choose first non-null per group.
    if cfg.on_col not in df.columns:
        return (
            df.sort_values(list(cfg.keys))
              .groupby(list(cfg.keys), dropna=False, as_index=False)
              .nth(0)
              .reset_index(drop=True)
        )

    # Choose row by min/max value in on_col (ties: keep first).
    agg_func = "idxmax" if cfg.pick == "max" else "idxmin"
    idx = (
        df.groupby(list(cfg.keys), dropna=False)[cfg.on_col]
          .agg(agg_func)  # returns index positions
    )
    return df.loc[idx.values].reset_index(drop=True)


def from_diann_parquet(
    path: str | Path,
    filters: Optional[FilterConfig] = None,
    dedup: Optional[DedupConfig] = None,
    keep_cols: Optional[Iterable[str]] = None,
    seq_col: str = "Stripped.Sequence",
    ph: float = 8.5,
) -> pd.DataFrame:
    """
    Load a DIA-NN main report (.parquet), filter low-confidence precursors,
    compute peptide properties from the sequence column, and optionally deduplicate.

    Typical workflow:
      1. Load the main report using pandas.
      2. Apply filters (PEP, decoys, optional q_values) from FilterConfig.
      3. Compute physicochemical properties from the peptide sequences.
      4. Deduplicate repeated peptide IDs (optional).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the DIA-NN `.parquet` report.
    filters : FilterConfig, optional
        PEP, Q-value and decoy filtering thresholds. If None, defaults are used.
    dedup : DedupConfig, optional
        Deduplication strategy to reduce multiple entries per Run × Precursor.Id.
        Set to None to keep all rows.
    keep_cols : Iterable[str], optional
        Extra columns to keep in addition to the default CORE_COLS.
    seq_col : str, default = 'Stripped.Sequence'
        Column containing amino-acid sequences to compute properties on.

    Returns
    -------
    pandas.DataFrame
        Filtered, annotated table containing:
        - selected DIA-NN metadata columns
        - computed properties (length, MW, pI, GRAVY, etc.)
        - missed cleavage counts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)

    # Column subset to keep (existing only)
    wanted = CORE_COLS.copy()
    if keep_cols:
        wanted.extend(list(keep_cols))

    # Ensure we don't duplicate columns like 'Stripped.Sequence'
    # dict.fromkeys preserves order and removes duplicates
    wanted_unique = list(dict.fromkeys(wanted + [seq_col]))
    cols = _existing_columns(df, wanted_unique)

    df = df.loc[:, cols].copy()



    # Filters
    filters = filters or FilterConfig()
    df = _apply_filters(df, filters)

    # Sanity: need sequence column
    if seq_col not in df.columns:
        raise KeyError(
            f"Required sequence column '{seq_col}' not found. "
            "For DIA-NN, use 'Stripped.Sequence'."
        )

    # Add normalized protein ID columns WITHOUT overwriting originals
    if "Protein.Ids" in df.columns:
        df["protein_primary_id"] = df["Protein.Ids"].apply(normalize_uniprot_accession)
    if "Protein.Group" in df.columns:
        df["protein_group_primary_id"] = df["Protein.Group"].apply(normalize_uniprot_accession)
    
    # Compute properties on peptide sequences
    props_df = _compute_props_on_series(df[seq_col], ph=ph)
    out = pd.concat(
        [df.reset_index(drop=True), props_df.reset_index(drop=True)],
        axis=1,
    )
    # Optional deduplication (e.g., one row per Run x Precursor.Id)
    if dedup is not None:
        out = _deduplicate(out, dedup)

    # Drop internal filtering columns (note: column names are case-sensitive)
    cols_to_drop = ["Q.Value", "Global.Q.Value", "Lib.Q.Value",
                    "Precursor.Quantity"] 
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])
    out = ensure_sequence_table(out, level="peptide", id_col="Precursor.Id")
    return out


def summarize_miscleavage(
    df: pd.DataFrame,
    groupby: Iterable[str] = ("Run",),
    mc_column: str = "missed_cleavages",
) -> pd.DataFrame:
    """
    Summarize missed-cleavage statistics by groups (e.g., per Run or Protein.Group).

    Calculates:
        • n = number of peptides in each group
        • percent_miscleavage = 100 × proportion of peptides with ≥1 missed cleavage
        • mean_missed_cleavages = mean number of missed cleavages per peptide

    Parameters
    ----------
    df : pandas.DataFrame
        Output from `from_diann_parquet`, containing the `missed_cleavages` column.
    groupby : Iterable[str], default = ("Run",)
        Columns to group by (e.g., ["Run"], ["Protein.Group"], ["Run","Protein.Group"]).
    mc_column : str, default = "missed_cleavages"
        Name of the column containing missed-cleavage counts.

    Returns
    -------
    pandas.DataFrame
        One row per group with summary metrics:
        ['n', 'percent_miscleavage', 'mean_missed_cleavages'].
    """

    if mc_column not in df.columns:
        raise KeyError(f"Column '{mc_column}' not found in input DataFrame.")
    g = df.groupby(list(groupby), dropna=False)
    out = g[mc_column].agg(
        n="size",
        percent_miscleavage=lambda s: 100.0 * (s.ge(1).mean() if len(s) else 0.0),
        mean_missed_cleavages="mean",
    ).reset_index()
    return out


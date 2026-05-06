"""
IO utilities for MaxQuant tabular outputs (evidence.txt)

This module provides a small adapter to turn MaxQuant evidence.txt files
into a DIA-NN-like format table and compute the same set of peptide
properties as for DIA-NN (length, MW, isoelectric point, GRAVY, instability,
aromaticity, aliphatic index, extinction coefficient, missed cleavages).

The main entry point is:
    from_maxquant_evidence(...)

which:
    - loads a tab-separated MaxQuant evidence.txt,
    - applies simple quality filters (PEP, reverse hits), 
    - maps MaxQuant columns to DIA-NN-equivalent names.,
    - computes peptide properties using the same functions as for DIA-NN.
    - returns a tidy pandas DataFrame.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

from .features_basic import basic_props
from .features_digest import missed_cleavages_in_peptide as mc_pep
from .schema import ensure_sequence_table
from .id_utils import normalize_uniprot_accession
from ._compute_utils import compute_props_on_series

@dataclass 
class MaxQuantFilterConfig: 
    """
    Filtering Configuration for MaxQuant evidence.txt files.

    Parameters
    ----------
    pep_col : str
        Name of the posterior error probability column. (default: PEP)
    pep_max: float or None
        Maximum allowed PEP. Rows with PEP>pep_max will be filtered out.
        If none, no filtering on PEP is done.
    intensity_col : str
        Name of the intensity column to use as precursor quantity.
        By default, this uses the non-LFQ raw intensity column.
    reverse_col : str
        Column marking reverse hits (e.g., 'Reverse').
    contaminant_col : str
        Column marking potential contaminants (e.g., 'Potential contaminant').
        Note: Contaminants are currently NOT filtered out.
    exclude_reverse : bool
        If true, optionally drop rows where reverse_col is marked as reverse.
    exclude_contaminants : bool
        Currently not used. Contaminants are NOT filtered out.
    """
    pep_col: str = "PEP"
    pep_max: Optional[float] = 0.01
    intensity_col: str = "Intensity"
    reverse_col: str = "Reverse"
    contaminant_col: str = "Potential contaminant"
    exclude_reverse:  bool = True
    exclude_contaminants: bool = False
                  
def _load_evidence(path: str | Path) -> pd.DataFrame:
    """
    Load MaxQuant evidence.txt file into a pandas DataFrame.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to the MaxQuant evidence.txt file.

    Returns
    -------
    pandas.DataFrame
        Raw table as loaded from the disk.
    """
    df = pd.read_csv(path, sep="\t")
    return df

def _apply_mq_filters(df: pd.DataFrame, cfg: MaxQuantFilterConfig) -> pd.DataFrame:
    """
    Apply simple quality filters to MaxQuant evidence table.
    
    This typically includes: 
        - PEP threshold,
        - optional removal of reverse hits.
        
    Note: Contaminants are currently NOT filtered out.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw MaxQuant evidence table.
    cfg : MaxQuantFilterConfig
        Filtering configuration.

    Returns
    -------
    pandas.DataFrame
        Filtered table.
    """
    if df.empty:
        return df
    
    mask = pd.Series(True, index=df.index)

    # Filter on PEP if the column is present and a threshold is set. 
    if cfg.pep_col in df.columns and cfg.pep_max is not None:
        mask &= df[cfg.pep_col] <= cfg.pep_max

    # Exclude reverse hits
    if cfg.exclude_reverse:
        if cfg.reverse_col in df.columns:
            mask &= ~df[cfg.reverse_col].isin(["+", 1, "1", True])
        
        for col in ["Leading proteins", "Leading razor protein"]:
            if col in df.columns:
                mask &= ~df[col].fillna("").astype(str).str.strip().str.startswith("REV_")

    # Contaminants are currently not filtered out (kept in data)
    # If we later need to filter them, uncomment the following 2 lines.
    #if cfg.exclude_contaminants and cfg.contaminant_col in df.columns:
        #mask &= ~df[cfg.contaminant_col].isin(["+", 1, "1", True])

    #nb: in the example file we have it is 1

    return df.loc[mask].reset_index(drop=True)

def _compute_props_on_series(seq: pd.Series) -> pd.DataFrame:
    """
    Compute physicochemical properties and digestion properties for each peptide sequence in a Series.
    
    This is a thin wrapper around compute_props_on_series() from _compute_utils.
    Kept for backward compatibility within this module.
    
    Parameters
    -----------
    seq: pandas.Series
        Series of peptide sequences.
    
    Returns
    -------
    pandas.DataFrame
        one row per peptide sequence with computed properties.
    """
    return compute_props_on_series(seq, include_missed_cleavages=True)

def from_maxquant_evidence(
        path: str | Path, 
        filters: Optional[MaxQuantFilterConfig] = None,
        keep_cols: Optional[Iterable[str]] = None,
        )-> pd.DataFrame:
    """
    Load a MaxQuant evidence.txt, filter low confidence entries, map to DIANN-like
    .parquet columns, compute peptide properties from the sequence, and return
    a table. 
    
    Note: Reverse hits are optionally filtered based on the configuration, but
    contaminants are currently NOT filtered out.

    The goal here is to produce a dataframe that is as close as possible to the 
    report.parquet file structure used by `from_diann_parquet` so that the
    downstream code can treat both maxquant and diann data in the same way. 

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the MaxQuant evidence.txt file.
    filters : MaxQuantFilterConfig (optional)
        Filtering configuration. If None, a default config with PEP<=0.01
        and exclusion of reverse is used.
    keep_cols : Iterable[str] (optional)
        Optional list of columns to keep from the original MaxQuant table
        before adding properties. If none, all columns are kept.

    Returns
    -------
    pandas.DataFrame
        Processed table with renamed DIA-NN-like columns and computed
        peptide properties.
    """
    path= Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MaxQuant evidence file not found: {path}")
    
    filters = filters or MaxQuantFilterConfig()
    df = _load_evidence(path)
    # we apply the basic filters
    df= _apply_mq_filters(df, filters)    
    # rename columns with this mapping
    rename_map = {
        "Raw file": "Run",
        "Sequence": "Stripped.Sequence",
        "Modified sequence": "Modified.Sequence",
        "Charge": "Precursor.Charge",
        "Leading razor protein": "Protein.Ids",
        "Protein names": "Protein.Names",
        "Gene names": "Genes",
        filters.intensity_col: "Precursor.Normalised",
        filters.pep_col: "PEP",
    }
    #but only rename columns that truly exist in the input
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    # we just need to make sure there is a sequence column usable for computing properties.
    if "Stripped.Sequence" not in df.columns:
        raise ValueError(
            "Could not find a peptide sequence column. Expected 'Sequence' in the "
            "MaxQuant evidence table so it can be renamed to 'Stripped.Sequence'."
        )
    # reconstruct a Precursor.Id (modified sequence + charge)
    if "Precursor.Id" not in df.columns and {
        "Modified.Sequence",
        "Precursor.Charge",
    } <= set(df.columns):
        df["Precursor.Id"] = (
            df["Modified.Sequence"].astype(str)
            + df["Precursor.Charge"].astype(str)
        )
    #if modified sequence isnt available, we fall back on stripped.sequence
    elif "Precursor.Id" not in df.columns and { 
        "Stripped.Sequence", 
        "Precursor.Charge", 
    } <= set(df.columns):
        df["Precursor.Id"] = (
            df["Stripped.Sequence"].astype(str)
            + df["Precursor.Charge"].astype(str)
        )

    # Optionally restrict to a subset of original columns before adding properties.
    if keep_cols is not None:
        # Only keep those that exist in df.
        cols_to_keep = [c for c in keep_cols if c in df.columns]
        # Also keep the columns we know we need for downstream code.
        mandatory = [
            "Run",
            "Stripped.Sequence",
            "Precursor.Charge",
            "Precursor.Id",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "PEP",
            "Precursor.Normalised",
        ]
        for c in mandatory:
            if c in df.columns and c not in cols_to_keep:
                cols_to_keep.append(c)
        df = df[cols_to_keep]

    # Add normalized protein ID column WITHOUT overwriting original
    if "Protein.Ids" in df.columns:
        df["protein_primary_id"] = df["Protein.Ids"].apply(normalize_uniprot_accession)
    
    # Compute physicochemical properties on the peptide sequences.
    props_df = _compute_props_on_series(df["Stripped.Sequence"])
    out = pd.concat(
        [df.reset_index(drop=True), props_df.reset_index(drop=True)],
        axis=1,
    )

    out = ensure_sequence_table(out, level="peptide", id_col="Precursor.Id")
    return out

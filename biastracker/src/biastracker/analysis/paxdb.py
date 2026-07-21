"""PaxDb reference-abundance annotation.

PaxDb provides a reference proteome abundance (parts-per-million) per UniProt
accession. BiasTracker treats this abundance as **just another per-protein
metric**: :func:`add_paxdb_abundance` attaches a ``paxdb_log10_ppm`` column to a
dataset so it flows through the normal Distributions / Compare / descriptive-stat
machinery, exactly like the physicochemical features.

Conventions:

* Matching is on the dataset's accession column (``primary_id`` by default).
* Abundance is log10-transformed by default (PaxDb spans several orders of
  magnitude); log10 is only defined for positive ppm, so non-positive / missing
  values become ``NaN`` (and are naturally dropped by the plots and stats).
* Contaminants (rows flagged ``is_contaminant``) are excluded by default — they
  are kept ID-only in the dataset for the contaminant ORA but should not enter
  the abundance metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biastracker.dataset import BiasDataset

# Default location of the bundled PaxDb human abundance table
# (biastracker/data/raw/paxdb/human_abundance_uniprot.csv). paxdb.py lives at
# biastracker/src/biastracker/analysis/, so the data dir is parents[3].
DEFAULT_PAXDB_CSV = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "paxdb" / "human_abundance_uniprot.csv"
)

#: Feature column added by :func:`add_paxdb_abundance`.
PAXDB_FEATURE = "paxdb_log10_ppm"


def load_paxdb_ppm(path: str | Path | None = None) -> pd.Series:
    """Load the PaxDb abundance table as an accession → ppm Series.

    Returns an empty Series if the file is absent. Duplicate accessions are
    collapsed to their mean ppm.
    """
    path = Path(path) if path is not None else DEFAULT_PAXDB_CSV
    if not path.exists():
        return pd.Series(dtype=float, name="paxdb_ppm")
    df = pd.read_csv(path)
    if "primary_id" not in df.columns or "paxdb_ppm" not in df.columns:
        raise ValueError(
            "PaxDb file must have 'primary_id' and 'paxdb_ppm' columns; "
            f"found {list(df.columns)}"
        )
    ppm = pd.Series(
        pd.to_numeric(df["paxdb_ppm"], errors="coerce").to_numpy(dtype=float),
        index=df["primary_id"].astype(str),
        name="paxdb_ppm",
    )
    if ppm.index.has_duplicates:
        ppm = ppm.groupby(level=0).mean()
    return ppm


def add_paxdb_abundance(
    dataset: BiasDataset,
    paxdb_ppm: pd.Series,
    id_col: str | None = None,
    out_col: str = PAXDB_FEATURE,
    log10: bool = True,
    exclude_contaminants: bool = True,
) -> BiasDataset:
    """Return a copy of *dataset* with a PaxDb reference-abundance feature column.

    The new column (``out_col``, default ``paxdb_log10_ppm``) holds the PaxDb
    abundance for each protein, matched by accession. Non-positive / unmatched
    abundances are ``NaN``; contaminants are ``NaN`` when *exclude_contaminants*
    is ``True``. When PaxDb has no matches at all the column is still added (all
    ``NaN``) so downstream code can rely on its presence.
    """
    id_col = id_col or dataset.id_col
    table = dataset.table.copy()
    if id_col not in table.columns:
        raise ValueError(f"id_col '{id_col}' not found in dataset table.")

    ppm = pd.to_numeric(table[id_col].astype(str).map(paxdb_ppm), errors="coerce").to_numpy(dtype=float)
    if log10:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(ppm > 0, np.log10(ppm), np.nan)
    else:
        values = ppm

    series = pd.Series(values, index=table.index)
    if exclude_contaminants and "is_contaminant" in table.columns:
        series = series.mask(table["is_contaminant"].fillna(False).astype(bool))
    table[out_col] = series

    n_matched = int(np.isfinite(series.to_numpy(dtype=float)).sum())
    return BiasDataset(
        name=dataset.name,
        table=table,
        level=dataset.level,
        source_type=dataset.source_type,
        id_col=dataset.id_col,
        group_col=dataset.group_col,
        metadata={**dataset.metadata, "paxdb_added": True, "paxdb_n_matched": n_matched},
    )

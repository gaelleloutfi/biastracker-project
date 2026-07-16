"""PaxDb abundance-agreement analysis.

PaxDb provides a reference proteome abundance (parts-per-million) per UniProt
accession. Rather than using PaxDb only as an alternative fGSEA ranking, this
module compares the **dataset's own abundance** (mean LFQ/expression) against the
**PaxDb reference** and quantifies their agreement with a Spearman rank
correlation. Spearman is used because it assesses *rank* agreement — whether the
proteins that are abundant in the dataset are also generally abundant in the
reference — rather than absolute equality of values.

Handling conventions:

* Matching is on the accession column already used elsewhere (``primary_id`` by
  default).
* Duplicate accessions in the dataset are collapsed to their **mean** abundance
  before matching (PaxDb accessions are likewise de-duplicated by mean on load).
* Abundances are log10-transformed for an interpretable scatter. log10 is only
  defined for positive values, so zero/negative abundances are dropped; the
  count of excluded pairs is reported. Because Spearman is invariant to monotone
  transforms, the correlation over the positive matched pairs is unchanged by the
  log — the transform is for visualisation and axis labelling only.
* Missing and infinite values are dropped **after** alignment, and the excluded
  count is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from biastracker.analysis.ranking import compute_mean_expression

# Default location of the bundled PaxDb human abundance table
# (biastracker/data/raw/paxdb/human_abundance_uniprot.csv). paxdb.py lives at
# biastracker/src/biastracker/analysis/, so the data dir is parents[3].
DEFAULT_PAXDB_CSV = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "paxdb" / "human_abundance_uniprot.csv"
)

# Minimum matched pairs required for a meaningful Spearman correlation.
_MIN_PAIRS = 3


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


@dataclass
class PaxDbCorrelationResult:
    """Result of a PaxDb abundance-agreement analysis."""

    matched: pd.DataFrame          # columns: primary_id, dataset_abundance, paxdb_abundance
    rho: float
    p_value: float
    n_used: int                    # pairs used in the correlation
    n_matched: int                 # accessions matched to PaxDb (before validity filter)
    n_input: int                   # unique dataset accessions with a finite abundance
    n_excluded: int                # matched pairs dropped as non-positive/NaN/inf
    x_label: str
    y_label: str
    message: str | None = None     # set when the correlation could not be computed

    @property
    def matched_fraction(self) -> float:
        return self.n_matched / self.n_input if self.n_input else 0.0


def compute_spearman_correlation(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> dict:
    """Spearman rank correlation between *x* and *y*.

    Pairs containing NaN/inf are dropped before the correlation. Returns a dict
    with ``rho``, ``p_value``, ``n`` (pairs used), and ``n_excluded``. When fewer
    than three valid pairs remain, ``rho``/``p_value`` are ``NaN`` and a
    ``message`` explains why.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    valid = np.isfinite(x) & np.isfinite(y)
    n_excluded = int((~valid).sum())
    xv, yv = x[valid], y[valid]
    n = int(xv.size)
    if n < _MIN_PAIRS:
        return {
            "rho": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "n_excluded": n_excluded,
            "message": f"Too few valid pairs ({n}) for a Spearman correlation "
                       f"(need ≥ {_MIN_PAIRS}).",
        }
    rho, p_value = spearmanr(xv, yv)
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n": n,
        "n_excluded": n_excluded,
        "message": None,
    }


def prepare_paxdb_correlation_data(
    df: pd.DataFrame,
    id_col: str,
    paxdb_ppm: pd.Series,
    expression_columns: list[str] | None = None,
    log_transform: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Align dataset mean abundance with PaxDb abundance, per accession.

    Parameters
    ----------
    df:
        Dataset table.
    id_col:
        Accession column.
    paxdb_ppm:
        Accession → ppm Series (see :func:`load_paxdb_ppm`).
    expression_columns:
        Optional explicit LFQ columns; defaults to the same detection used by
        :func:`biastracker.analysis.ranking.compute_mean_expression`.
    log_transform:
        When ``True`` (default) both abundances are log10-transformed for the
        scatter; non-positive values are dropped.

    Returns
    -------
    (matched, meta)
        ``matched`` has columns ``primary_id``, ``dataset_abundance``,
        ``paxdb_abundance`` (one row per matched, valid accession). ``meta``
        carries counts and axis labels.
    """
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in table.")

    dataset_abundance = compute_mean_expression(df, expression_columns)
    ds = pd.Series(
        dataset_abundance.to_numpy(dtype=float),
        index=df[id_col],
        name="dataset_abundance",
    )
    ds = ds[ds.index.notna()]
    ds.index = ds.index.astype(str)
    ds = ds[np.isfinite(ds.to_numpy(dtype=float))]
    # One value per accession (mean of duplicates).
    if ds.index.has_duplicates:
        ds = ds.groupby(level=0).mean()
    n_input = int(ds.size)

    paxdb_ppm = paxdb_ppm.astype(float)
    common = ds.index.intersection(paxdb_ppm.index)
    n_matched = int(len(common))

    matched = pd.DataFrame({
        "primary_id": common,
        "dataset_abundance": ds.reindex(common).to_numpy(dtype=float),
        "paxdb_abundance": paxdb_ppm.reindex(common).to_numpy(dtype=float),
    })

    if log_transform:
        x_label = "Dataset mean abundance (log₁₀ LFQ)"
        y_label = "PaxDb abundance (log₁₀ ppm)"
        for col in ("dataset_abundance", "paxdb_abundance"):
            vals = matched[col].to_numpy(dtype=float)
            # log10 only defined for positive values; others → NaN (dropped below).
            with np.errstate(invalid="ignore", divide="ignore"):
                matched[col] = np.where(vals > 0, np.log10(vals), np.nan)
    else:
        x_label = "Dataset mean abundance (LFQ)"
        y_label = "PaxDb abundance (ppm)"

    before = len(matched)
    matched = matched[
        np.isfinite(matched["dataset_abundance"].to_numpy(dtype=float))
        & np.isfinite(matched["paxdb_abundance"].to_numpy(dtype=float))
    ].reset_index(drop=True)
    n_excluded = before - len(matched)

    meta = {
        "n_input": n_input,
        "n_matched": n_matched,
        "n_used": len(matched),
        "n_excluded": n_excluded,
        "matched_fraction": (n_matched / n_input) if n_input else 0.0,
        "x_label": x_label,
        "y_label": y_label,
        "log_transform": log_transform,
    }
    return matched, meta


def paxdb_abundance_agreement(
    df: pd.DataFrame,
    id_col: str,
    paxdb_ppm: pd.Series,
    expression_columns: list[str] | None = None,
    log_transform: bool = True,
) -> PaxDbCorrelationResult:
    """High-level convenience: prepare data and compute the Spearman correlation."""
    matched, meta = prepare_paxdb_correlation_data(
        df, id_col=id_col, paxdb_ppm=paxdb_ppm,
        expression_columns=expression_columns, log_transform=log_transform,
    )
    stats = compute_spearman_correlation(
        matched["dataset_abundance"], matched["paxdb_abundance"]
    )
    return PaxDbCorrelationResult(
        matched=matched,
        rho=stats["rho"],
        p_value=stats["p_value"],
        n_used=stats["n"],
        n_matched=meta["n_matched"],
        n_input=meta["n_input"],
        n_excluded=meta["n_excluded"],
        x_label=meta["x_label"],
        y_label=meta["y_label"],
        message=stats["message"],
    )

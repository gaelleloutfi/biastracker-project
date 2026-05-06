"""
IO utilities for manual CSV/TSV input tables.

This module provides a generic loader that accepts arbitrary CSV/TSV tables
containing peptide or protein sequences and optional expression data.

The main entry point is:
    from_manual_table(...)

which:
    - reads a CSV/TSV file with flexible separator detection,
    - locates the sequence column from a list of candidate names,
    - locates or creates an expression column,
    - computes physicochemical properties using basic_props(),
    - computes missed cleavages for peptides,
    - standardises the result with ensure_sequence_table().
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .features_basic import basic_props
from .features_digest import missed_cleavages_in_peptide as mc_pep
from .schema import ensure_sequence_table
from ._compute_utils import compute_props_on_series


def _compute_props_on_series(
    seq: pd.Series,
    include_missed_cleavages: bool = False,
) -> pd.DataFrame:
    """
    Compute physicochemical properties for each sequence in a Series.

    This is a thin wrapper around compute_props_on_series() from _compute_utils.
    Kept for backward compatibility within this module.

    Parameters
    ----------
    seq : pandas.Series
        Series of amino-acid sequences.
    include_missed_cleavages : bool, default False
        When True, also compute and include ``missed_cleavages``.

    Returns
    -------
    pandas.DataFrame
        One row per sequence with computed property columns.
    """
    return compute_props_on_series(seq, include_missed_cleavages=include_missed_cleavages)


def from_manual_table(
    path: str | Path,
    level: Literal["peptide", "protein"],
    sep: str | None = None,
    seq_col_candidates: tuple[str, ...] = (
        "sequence",
        "Sequence",
        "Stripped.Sequence",
    ),
    expr_col_candidates: tuple[str, ...] = (
        "expression",
        "Expression",
        "intensity",
        "Intensity",
        "Precursor.Normalised",
    ),
    fill_expression: float = 0.0,
) -> pd.DataFrame:
    """
    Load an arbitrary CSV/TSV table with sequence and expression data.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input CSV or TSV file.
    level : {"peptide", "protein"}
        Molecular level of the sequences.
    sep : str or None
        Column separator.  If *None*, the separator is inferred from the
        first line of the file (tab, comma, or semicolon).
    seq_col_candidates : tuple of str
        Ordered list of column names to search for the sequence column.
        The first match is used.
    expr_col_candidates : tuple of str
        Ordered list of column names to search for the expression column.
        The first match is used.  If none is found, a new column named
        ``'expression'`` is added and filled with *fill_expression*.
    fill_expression : float
        Value used to fill a missing expression column (default ``0.0``).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with computed physicochemical properties and
        canonical columns added by :func:`ensure_sequence_table`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no sequence column can be found using *seq_col_candidates*.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Infer separator if not provided
    if sep is None:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        elif ";" in first_line:
            sep = ";"
        else:
            sep = ","

    df = pd.read_csv(path, sep=sep)

    # --- Resolve and rename sequence column ---
    seq_col = next((c for c in seq_col_candidates if c in df.columns), None)
    if seq_col is None:
        raise ValueError(
            f"Could not find a sequence column. "
            f"Tried: {list(seq_col_candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    if seq_col != "sequence":
        df = df.rename(columns={seq_col: "sequence"})

    # --- Resolve, rename, and cast expression column ---
    expr_col = next((c for c in expr_col_candidates if c in df.columns), None)
    if expr_col is None:
        df["expression"] = fill_expression
    elif expr_col != "expression":
        df = df.rename(columns={expr_col: "expression"})
    df["expression"] = pd.to_numeric(df["expression"], errors="coerce").fillna(fill_expression)

    # --- Handle empty dataframe (columns are now in canonical form) ---
    if df.empty:
        return ensure_sequence_table(df, level=level, id_col=None)

    # --- Compute physicochemical properties on the canonical sequence column ---
    props_df = _compute_props_on_series(
        df["sequence"],
        include_missed_cleavages=(level == "peptide"),
    )
    df = pd.concat(
        [df.reset_index(drop=True), props_df.reset_index(drop=True)],
        axis=1,
    )

    df = ensure_sequence_table(df, level=level, id_col=None)
    return df

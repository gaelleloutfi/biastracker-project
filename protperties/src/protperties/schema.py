"""
Schema utilities for standardizing annotated sequence tables.

Provides :func:`ensure_sequence_table` to add canonical columns
(``level``, ``sequence``, ``primary_id``) to peptide- or protein-level
DataFrames returned by the various loaders.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd


def ensure_sequence_table(
    df: pd.DataFrame,
    level: Literal["peptide", "protein"],
    id_col: str | None = None,
) -> pd.DataFrame:
    """Return a copy of *df* with canonical columns added.

    Parameters
    ----------
    df:
        Input DataFrame (not modified in-place).
    level:
        Either ``"peptide"`` or ``"protein"``.  Written verbatim into the
        new ``level`` column.
    id_col:
        Optional name of a column in *df* to copy as ``primary_id``.
        If *None*, no ``primary_id`` column is added.
        Raises :exc:`ValueError` if the column does not exist in *df*.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added:

        * ``level`` – the value passed as *level*
        * ``sequence`` – amino-acid sequence (copied from
          ``Stripped.Sequence`` when that column is present and
          ``sequence`` is not; kept as-is if ``sequence`` already exists).
          For ``level="protein"``, if no sequence column exists, a column
          of ``pd.NA`` is created. For ``level="peptide"``, it is required.
        * ``primary_id`` – only when *id_col* is not *None*

    Raises
    ------
    ValueError
        If no sequence column can be located, or if *id_col* is specified
        but does not exist in *df*.
    """
    out = df.copy()

    # --- level ---
    out["level"] = level

    # --- sequence ---
    if "sequence" not in out.columns:
        if "Stripped.Sequence" in out.columns:
            out["sequence"] = out["Stripped.Sequence"]
        elif level == "protein":
            out["sequence"] = pd.NA
        else:
            raise ValueError(
                "No sequence column found. "
                "Expected 'sequence' or 'Stripped.Sequence' in the DataFrame."
            )

    # --- primary_id ---
    if id_col is not None:
        if id_col not in out.columns:
            raise ValueError(
                f"id_col '{id_col}' not found in the DataFrame. "
                f"Available columns: {list(out.columns)}"
            )
        out["primary_id"] = out[id_col]

    return out

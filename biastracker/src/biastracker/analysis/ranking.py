"""Ranking-metric preparation for pre-ranked fGSEA.

The supervisor review narrowed the meaningful fGSEA ranking metrics for this kind
of proteomics data to two choices:

* **Mean expression** — the row-wise mean of the dataset's LFQ / expression
  columns (or the precomputed ``mean_lfq`` / ``expression`` column when the raw
  sample columns are not detectable, e.g. DIA-NN pg_matrix).
* **Custom metric** — a numeric column already provided by the user, which is
  the natural choice for a "homemade" dataset that ships its own ranking value.

The set of allowed metrics and their user-facing labels lives in the single
:data:`FGSEA_RANKING_METHODS` registry so it can be adjusted from one place once
the wording is finalised scientifically.

Handling conventions (shared with :func:`biastracker.analysis.enrichment.run_fgsea`):

* **Missing / non-numeric values** are coerced to ``NaN`` and dropped.
* **Infinite values** are dropped (they produce undefined enrichment scores).
* **Duplicate accessions** are collapsed to their **maximum** score — consistent
  with ``run_fgsea``, which keeps the first row after a descending sort.
* **Ties** keep input order (stable ``mergesort``) so the ranking is
  deterministic.
* The returned :class:`pandas.Series` is named, numeric, finite, indexed by
  accession, and sorted by **descending** score (top of the ranked list first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Central registry — adjust allowed metrics / labels here only.
# ---------------------------------------------------------------------------
MEAN_EXPRESSION = "mean_expression"
CUSTOM = "custom"

FGSEA_RANKING_METHODS: dict[str, str] = {
    MEAN_EXPRESSION: "Mean expression (row-wise mean of LFQ columns)",
    CUSTOM: "Custom metric (numeric column)",
}

DEFAULT_RANKING_METHOD = MEAN_EXPRESSION

# Column prefixes / names used to auto-detect per-sample expression columns.
# MaxQuant proteinGroups store one ``LFQ intensity <sample>`` column per run.
_LFQ_PREFIXES: tuple[str, ...] = ("LFQ intensity ",)
# Precomputed row-mean columns emitted by the protperties loaders, used as a
# fallback when the raw per-sample columns are not present (e.g. DIA-NN).
_PRECOMPUTED_MEAN_COLS: tuple[str, ...] = ("mean_lfq", "expression")

# Minimum number of finite ranked values fGSEA needs to run.
_MIN_RANKED = 2


def detect_expression_columns(df: pd.DataFrame) -> list[str]:
    """Return the per-sample LFQ/expression columns detected in *df*.

    Detection is prefix-based (``"LFQ intensity "``), matching the MaxQuant
    convention used by the protperties loaders. DIA-NN sample columns are not
    tagged and are therefore not detected here; callers fall back to the
    precomputed mean column (see :func:`compute_mean_expression`).
    """
    return [
        c for c in df.columns
        if any(str(c).startswith(p) for p in _LFQ_PREFIXES)
    ]


def compute_mean_expression(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.Series:
    """Row-wise mean expression across the selected LFQ/expression columns.

    Parameters
    ----------
    df:
        Source table.
    columns:
        Explicit list of expression columns to average. When ``None``, columns
        are auto-detected with :func:`detect_expression_columns`; if none are
        found, the precomputed ``mean_lfq`` / ``expression`` column is used.

    Returns
    -------
    pandas.Series
        Row-aligned mean expression (``NaN`` where a row has no finite value).

    Raises
    ------
    ValueError
        If *columns* is an empty list, if any requested column is missing, or if
        no expression columns can be found and no precomputed mean column exists
        (we never silently rank from zero valid expression columns).
    """
    if columns is None:
        columns = detect_expression_columns(df)
        if not columns:
            for fallback in _PRECOMPUTED_MEAN_COLS:
                if fallback in df.columns:
                    return pd.to_numeric(df[fallback], errors="coerce")
            raise ValueError(
                "No LFQ/expression columns detected and no precomputed "
                f"{' / '.join(_PRECOMPUTED_MEAN_COLS)} column is present."
            )
    else:
        if len(columns) == 0:
            raise ValueError("No expression columns provided to average.")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Expression columns not found in table: {missing}")

    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    # mean(axis=1) skips NaN and yields NaN only for all-NaN rows.
    return numeric.mean(axis=1)


def prepare_fgsea_ranking(
    df: pd.DataFrame,
    id_col: str,
    method: str = DEFAULT_RANKING_METHOD,
    expression_columns: list[str] | None = None,
    custom_col: str | None = None,
    exclude_contaminants: bool = True,
) -> pd.Series:
    """Build a clean, named, descending ranking Series for :func:`run_fgsea`.

    Parameters
    ----------
    df:
        Source dataset table.
    id_col:
        Accession column (values are stringified for the index).
    method:
        One of :data:`FGSEA_RANKING_METHODS` — ``"mean_expression"`` or
        ``"custom"``.
    expression_columns:
        Optional explicit LFQ columns for ``method="mean_expression"``.
    custom_col:
        Numeric column to rank by for ``method="custom"``.
    exclude_contaminants:
        When ``True`` (default) rows flagged ``is_contaminant`` are dropped
        before ranking, so contaminants (kept ID-only in the dataset for the
        contaminant ORA) do not distort the fGSEA ranking. Ignored when the
        table has no ``is_contaminant`` column.

    Returns
    -------
    pandas.Series
        Finite scores indexed by accession, sorted descending, with duplicate
        accessions collapsed to their maximum. See the module docstring for the
        full handling contract.

    Raises
    ------
    ValueError
        For an unknown *method*, a missing ``id_col``/``custom_col``, a
        non-numeric custom metric, or fewer than two finite ranked values.
    """
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in table.")

    if exclude_contaminants and "is_contaminant" in df.columns:
        df = df[~df["is_contaminant"].fillna(False).astype(bool)]

    if method == MEAN_EXPRESSION:
        values = compute_mean_expression(df, expression_columns)
        name = MEAN_EXPRESSION
    elif method == CUSTOM:
        if not custom_col:
            raise ValueError("method='custom' requires a custom_col.")
        if custom_col not in df.columns:
            raise ValueError(f"custom_col '{custom_col}' not found in table.")
        values = pd.to_numeric(df[custom_col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).any():
            raise ValueError(
                f"Custom metric column '{custom_col}' has no finite numeric "
                "values; pick a numeric column."
            )
        name = custom_col
    else:
        raise ValueError(
            f"Unknown ranking method '{method}'. "
            f"Valid methods: {sorted(FGSEA_RANKING_METHODS)}"
        )

    scores = pd.Series(values.to_numpy(dtype=float), index=df[id_col], name=name)
    # Drop rows whose accession is missing, then normalise the index to str.
    scores = scores[scores.index.notna()]
    scores.index = scores.index.astype(str)
    # Keep only finite scores (drops NaN and ±inf).
    scores = scores[np.isfinite(scores.to_numpy(dtype=float))]
    # Collapse duplicate accessions to their maximum (consistent with run_fgsea).
    if scores.index.has_duplicates:
        scores = scores.groupby(level=0).max()
    scores = scores.sort_values(ascending=False, kind="mergesort")

    if len(scores) < _MIN_RANKED:
        raise ValueError(
            "Ranking has fewer than two finite values after cleaning; "
            "fGSEA needs at least two ranked proteins."
        )
    scores.name = name
    return scores

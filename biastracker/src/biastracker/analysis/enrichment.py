"""Over-representation analysis (ORA) for BiasTracker.

This module provides term-by-term Fisher's exact test enrichment analysis.
The main entry-points are:

* :func:`run_ora`  — core function that accepts explicit ID sets.
* :func:`run_group_ora` — convenience wrapper that extracts IDs from a
  :class:`~biastracker.dataset.BiasDataset` group column.
"""
from __future__ import annotations

from typing import Literal, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats

from biastracker.dataset import AnnotationSet, BiasDataset
from biastracker.stats import adjust_pvalues

# ---------------------------------------------------------------------------
# Column order for the returned DataFrame
# ---------------------------------------------------------------------------
_ORA_COLS = [
    "term_id",
    "term_name",
    "source",
    "category",
    "query_count",
    "query_size",
    "background_count",
    "background_size",
    "odds_ratio",
    "p_value",
    "fdr",
    "direction",
    "query_hits",
]

_GROUP_ORA_COLS = ["dataset", "group_col", "query_group"] + _ORA_COLS


# ---------------------------------------------------------------------------
# run_ora
# ---------------------------------------------------------------------------

def run_ora(
    query_ids: Set[str],
    background_ids: Set[str],
    annotations: AnnotationSet,
    min_term_size: int = 3,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run over-representation analysis (ORA) using Fisher's exact test.

    For every term in *annotations* the function builds a 2×2 contingency
    table and tests whether the query set is enriched or depleted.

    Parameters
    ----------
    query_ids:
        Set of entity identifiers in the query group.
    background_ids:
        Universe of all observable identifiers.  Query IDs not present in the
        background are silently excluded.
    annotations:
        :class:`~biastracker.dataset.AnnotationSet` providing term→ID mappings.
    min_term_size:
        Minimum number of background proteins annotated with a term to include
        it in the analysis.  Terms below this threshold are skipped.
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues` (default ``"fdr_bh"``).

    Returns
    -------
    pd.DataFrame
        One row per tested term with columns defined in ``_ORA_COLS``.
    """
    # Restrict query to background universe
    query_ids = set(str(q) for q in query_ids) & set(str(b) for b in background_ids)
    background_ids = set(str(b) for b in background_ids)

    n_query = len(query_ids)
    n_background = len(background_ids)
    n_not_query = n_background - n_query

    ann_df = annotations.table
    id_col = annotations.id_col
    term_col = annotations.term_col
    term_id_col = annotations.term_id_col
    cat_col = annotations.category_col
    source = annotations.source

    # Group annotation table by term once for efficiency
    term_groups = ann_df.groupby(term_col, sort=False)

    rows: list[dict] = []
    for term_name, term_df in term_groups:
        # IDs in background annotated with this term
        annotated_ids = set(term_df[id_col].astype(str)) & background_ids
        background_count = len(annotated_ids)

        if background_count < min_term_size:
            continue

        # IDs in query annotated with this term
        query_hits_ids = annotated_ids & query_ids
        query_count = len(query_hits_ids)

        # 2×2 contingency table
        a = query_count                          # query ∩ term
        b = n_query - query_count                # query ∩ ¬term
        c = background_count - query_count       # ¬query ∩ term
        d = n_not_query - c                      # ¬query ∩ ¬term

        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")

        if odds_ratio > 1:
            direction = "enriched"
        elif odds_ratio < 1:
            direction = "depleted"
        else:
            direction = "neutral"

        # Representative term_id and category from annotation table
        term_id = term_df[term_id_col].iloc[0] if term_id_col in term_df.columns else term_name
        category = term_df[cat_col].iloc[0] if cat_col in term_df.columns else "unknown"

        rows.append({
            "term_id": term_id,
            "term_name": term_name,
            "source": source,
            "category": category,
            "query_count": query_count,
            "query_size": n_query,
            "background_count": background_count,
            "background_size": n_background,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "fdr": np.nan,          # filled below
            "direction": direction,
            "query_hits": ";".join(sorted(query_hits_ids)),
        })

    if not rows:
        return pd.DataFrame(columns=_ORA_COLS)

    result = pd.DataFrame(rows)
    result = adjust_pvalues(result, p_col="p_value", method=correction, out_col="fdr")

    return result[_ORA_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# run_group_ora
# ---------------------------------------------------------------------------

def run_group_ora(
    dataset: BiasDataset,
    group_col: str,
    query_group: str,
    annotations: AnnotationSet,
    background: Literal["all", "other"] = "all",
    id_col: str = "primary_id",
    min_term_size: int = 3,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Convenience wrapper around :func:`run_ora` that extracts IDs from a dataset group.

    Parameters
    ----------
    dataset:
        Source :class:`~biastracker.dataset.BiasDataset`.
    group_col:
        Column in ``dataset.table`` that identifies groups.
    query_group:
        Value in *group_col* that defines the query set.
    annotations:
        :class:`~biastracker.dataset.AnnotationSet` providing term→ID mappings.
    background:
        How to define the background universe:

        * ``"all"`` — all IDs in the dataset (default).
        * ``"other"`` — all IDs in the dataset that do *not* belong to
          *query_group* (i.e. the complement within the dataset).
    id_col:
        Column containing entity identifiers (default ``"primary_id"``).
    min_term_size:
        Forwarded to :func:`run_ora`.
    correction:
        Forwarded to :func:`run_ora`.

    Returns
    -------
    pd.DataFrame
        Result of :func:`run_ora` with three extra leading columns:
        ``dataset``, ``group_col``, and ``query_group``.

    Raises
    ------
    ValueError
        If *group_col* is not found in ``dataset.table``.
    ValueError
        If *id_col* is not found in ``dataset.table``.
    """
    df = dataset.table

    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )
    if id_col not in df.columns:
        raise ValueError(
            f"id_col '{id_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )

    query_ids = set(
        df.loc[df[group_col] == query_group, id_col]
        .dropna()
        .astype(str)
    )

    if background == "other":
        background_ids = set(
            df.loc[df[group_col] != query_group, id_col]
            .dropna()
            .astype(str)
        )
        # Include query IDs in the background universe too so the Fisher table
        # is built over the full dataset (consistent with standard ORA practice
        # when background="other" means "complement within dataset").
        background_ids = background_ids | query_ids
    else:  # "all"
        background_ids = set(df[id_col].dropna().astype(str))

    result = run_ora(
        query_ids=query_ids,
        background_ids=background_ids,
        annotations=annotations,
        min_term_size=min_term_size,
        correction=correction,
    )

    if result.empty:
        for col in ("dataset", "group_col", "query_group"):
            result[col] = pd.Series(dtype=str)
        return result[_GROUP_ORA_COLS]

    result = result.copy()
    result.insert(0, "query_group", query_group)
    result.insert(0, "group_col", group_col)
    result.insert(0, "dataset", dataset.name)

    return result[_GROUP_ORA_COLS].reset_index(drop=True)

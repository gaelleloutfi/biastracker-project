"""Term-level property analysis for BiasTracker.

This module provides two complementary views of how physicochemical features
differ across biological annotation terms:

* :func:`summarize_term_properties` — descriptive statistics (mean, median,
  std, min, max) per term × feature.
* :func:`compare_term_vs_rest` — pairwise statistical comparison (Mann-Whitney
  U + KS test + FDR) of "term proteins" against "all other proteins" for each
  term × feature combination.
"""
from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

from biastracker.dataset import AnnotationSet, BiasDataset
from biastracker.preprocessing import select_numeric_features
from biastracker.stats import mannwhitney_u, ks_test, adjust_pvalues, effect_direction

# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "dataset",
    "term_id",
    "term_name",
    "source",
    "category",
    "feature",
    "n",
    "mean",
    "median",
    "std",
    "min",
    "max",
]

_COMPARE_COLS = [
    "dataset",
    "term_id",
    "term_name",
    "source",
    "category",
    "feature",
    "n_term",
    "n_rest",
    "mean_term",
    "mean_rest",
    "median_term",
    "median_rest",
    "delta_median",
    "direction",
    "mannwhitney_statistic",
    "mannwhitney_p",
    "mannwhitney_fdr",
    "ks_statistic",
    "ks_p",
    "ks_fdr",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_features(df: pd.DataFrame, features: Optional[List[str]]) -> List[str]:
    """Return numeric feature columns to analyse.

    When *features* is ``None``, all numeric columns are used.  Otherwise the
    list is filtered to those that exist and are numeric.
    """
    if features is None:
        return df.select_dtypes(include="number").columns.tolist()
    return select_numeric_features(df, features=features)


def _iter_terms(
    dataset: BiasDataset,
    annotations: AnnotationSet,
    min_term_size: int,
    id_col: str,
):
    """Yield ``(term_meta, term_df, rest_df)`` for each qualifying term.

    *term_meta* is a dict with ``term_id``, ``term_name``, ``source``,
    ``category``.  *term_df* is the subset of ``dataset.table`` whose
    *id_col* appears in the term.  *rest_df* is the complementary subset.

    Terms with fewer than *min_term_size* unique IDs (after intersecting with
    the dataset) are skipped.
    """
    df = dataset.table
    ann_df = annotations.table

    if id_col not in df.columns:
        raise ValueError(
            f"id_col '{id_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )

    dataset_ids = set(df[id_col].dropna().astype(str))

    term_col = annotations.term_col
    term_id_col = annotations.term_id_col
    cat_col = annotations.category_col

    for term_name, term_ann in ann_df.groupby(term_col, sort=False):
        annotated_ids = set(term_ann[annotations.id_col].astype(str)) & dataset_ids

        if len(annotated_ids) < min_term_size:
            continue

        term_id = term_ann[term_id_col].iloc[0] if term_id_col in term_ann.columns else term_name
        category = term_ann[cat_col].iloc[0] if cat_col in term_ann.columns else "unknown"

        meta = {
            "term_id": term_id,
            "term_name": term_name,
            "source": annotations.source,
            "category": category,
        }

        in_term_mask = df[id_col].astype(str).isin(annotated_ids)
        yield meta, df[in_term_mask], df[~in_term_mask]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_term_properties(
    dataset: BiasDataset,
    annotations: AnnotationSet,
    features: Optional[List[str]] = None,
    min_term_size: int = 3,
    id_col: str = "primary_id",
) -> pd.DataFrame:
    """Summarize physicochemical properties of proteins belonging to each term.

    For each annotation term × numeric feature, descriptive statistics are
    computed over the proteins (rows in *dataset*) annotated with that term.

    Parameters
    ----------
    dataset:
        Source :class:`~biastracker.dataset.BiasDataset`.
    annotations:
        Term→protein mapping as an :class:`~biastracker.dataset.AnnotationSet`.
    features:
        Numeric columns to summarize.  ``None`` uses all numeric columns.
    min_term_size:
        Minimum number of dataset proteins that must be annotated with a term
        for it to be included in the output.
    id_col:
        Column in ``dataset.table`` containing entity identifiers.

    Returns
    -------
    pd.DataFrame
        One row per (term, feature) pair with columns defined in
        ``_SUMMARY_COLS``.
    """
    df = dataset.table
    numeric_feats = _resolve_features(df, features)

    rows: list[dict] = []

    for meta, term_df, _ in _iter_terms(dataset, annotations, min_term_size, id_col):
        for feat in numeric_feats:
            s = term_df[feat].dropna()
            n = len(s)
            rows.append({
                **meta,
                "dataset": dataset.name,
                "feature": feat,
                "n": n,
                "mean": s.mean() if n > 0 else np.nan,
                "median": s.median() if n > 0 else np.nan,
                "std": s.std() if n > 0 else np.nan,
                "min": s.min() if n > 0 else np.nan,
                "max": s.max() if n > 0 else np.nan,
            })

    if not rows:
        return pd.DataFrame(columns=_SUMMARY_COLS)

    return pd.DataFrame(rows)[_SUMMARY_COLS].reset_index(drop=True)


def compare_term_vs_rest(
    dataset: BiasDataset,
    annotations: AnnotationSet,
    features: Optional[List[str]] = None,
    min_term_size: int = 3,
    id_col: str = "primary_id",
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare term proteins against all non-term proteins for each feature.

    For every (term, feature) pair the function runs a Mann-Whitney U test and
    a KS test, then applies FDR correction *across all tests simultaneously*.

    Parameters
    ----------
    dataset:
        Source :class:`~biastracker.dataset.BiasDataset`.
    annotations:
        Term→protein mapping as an :class:`~biastracker.dataset.AnnotationSet`.
    features:
        Numeric columns to test.  ``None`` uses all numeric columns.
    min_term_size:
        Minimum term size (after intersecting with dataset IDs).
    id_col:
        Column in ``dataset.table`` containing entity identifiers.
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues`.

    Returns
    -------
    pd.DataFrame
        One row per (term, feature) pair with columns defined in
        ``_COMPARE_COLS``.  FDR is computed jointly across all rows.
    """
    df = dataset.table
    numeric_feats = _resolve_features(df, features)

    rows: list[dict] = []

    for meta, term_df, rest_df in _iter_terms(dataset, annotations, min_term_size, id_col):
        for feat in numeric_feats:
            a = term_df[feat].dropna().values
            b = rest_df[feat].dropna().values

            n_term = len(a)
            n_rest = len(b)

            mean_term  = float(np.mean(a))  if n_term > 0 else np.nan
            mean_rest  = float(np.mean(b))  if n_rest > 0 else np.nan
            median_term = float(np.median(a)) if n_term > 0 else np.nan
            median_rest = float(np.median(b)) if n_rest > 0 else np.nan

            delta = (
                (median_term - median_rest)
                if not (np.isnan(median_term) or np.isnan(median_rest))
                else np.nan
            )

            direction = effect_direction(median_term, median_rest, "term", "rest")

            mw = mannwhitney_u(a, b)
            ks = ks_test(a, b)

            rows.append({
                **meta,
                "dataset": dataset.name,
                "feature": feat,
                "n_term": n_term,
                "n_rest": n_rest,
                "mean_term": mean_term,
                "mean_rest": mean_rest,
                "median_term": median_term,
                "median_rest": median_rest,
                "delta_median": delta,
                "direction": direction,
                "mannwhitney_statistic": mw["statistic"],
                "mannwhitney_p": mw["p_value"],
                "mannwhitney_fdr": np.nan,  # filled below
                "ks_statistic": ks["statistic"],
                "ks_p": ks["p_value"],
                "ks_fdr": np.nan,  # filled below
            })

    if not rows:
        return pd.DataFrame(columns=_COMPARE_COLS)

    result = pd.DataFrame(rows)

    # FDR correction across all (term × feature) pairs simultaneously
    result = adjust_pvalues(result, p_col="mannwhitney_p", method=correction, out_col="mannwhitney_fdr")
    result = adjust_pvalues(result, p_col="ks_p",          method=correction, out_col="ks_fdr")

    return result[_COMPARE_COLS].reset_index(drop=True)

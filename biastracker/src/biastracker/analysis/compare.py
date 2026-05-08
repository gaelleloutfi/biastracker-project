import numpy as np
import pandas as pd
from typing import List, Optional

from biastracker.dataset import BiasDataset
from biastracker.preprocessing import select_numeric_features
from biastracker.stats import (
    mannwhitney_u,
    ks_test,
    kruskal_test,
    adjust_pvalues,
    effect_direction,
)

_COMPARE_COLS = [
    "dataset",
    "group_col",
    "group_a",
    "group_b",
    "feature",
    "n_a",
    "n_b",
    "mean_a",
    "mean_b",
    "median_a",
    "median_b",
    "delta_median",
    "direction",
    "mannwhitney_statistic",
    "mannwhitney_p",
    "mannwhitney_fdr",
    "ks_statistic",
    "ks_p",
    "ks_fdr",
]

_MULTI_COLS = [
    "dataset",
    "group_col",
    "feature",
    "n_groups",
    "groups",
    "kruskal_statistic",
    "kruskal_p",
    "kruskal_fdr",
]

# Extra column appended by compare_datasets on top of _COMPARE_COLS
_DATASET_COMPARE_COLS = _COMPARE_COLS + ["comparison_type"]

_DATASET_GROUP_COL = "__dataset_group"


def compare_groups(
    dataset: BiasDataset,
    group_col: str,
    group_a: str,
    group_b: str,
    features: Optional[List[str]] = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare two groups across numeric features using Mann-Whitney U and KS tests.

    Parameters
    ----------
    dataset:
        The BiasDataset to analyse.
    group_col:
        Column in ``dataset.table`` that identifies groups.
    group_a, group_b:
        Labels of the two groups to compare.
    features:
        Numeric feature columns to test. ``None`` uses all numeric columns
        (via :func:`~biastracker.preprocessing.select_numeric_features`).
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues`.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns defined in ``_COMPARE_COLS``.
    """
    df = dataset.table

    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )

    # When features=None use all numeric columns; otherwise restrict to the
    # requested subset that is present and numeric.
    if features is None:
        numeric_feats = dataset.available_features()
    else:
        numeric_feats = select_numeric_features(df, features=features)
    mask = df[group_col].isin([group_a, group_b])
    sub = df[mask]

    vals_a = sub[sub[group_col] == group_a]
    vals_b = sub[sub[group_col] == group_b]

    rows = []
    for feat in numeric_feats:
        a = vals_a[feat].dropna().values
        b = vals_b[feat].dropna().values

        n_a = len(a)
        n_b = len(b)

        mean_a = float(np.mean(a)) if n_a > 0 else np.nan
        mean_b = float(np.mean(b)) if n_b > 0 else np.nan
        median_a = float(np.median(a)) if n_a > 0 else np.nan
        median_b = float(np.median(b)) if n_b > 0 else np.nan
        delta_median = (median_a - median_b) if not (np.isnan(median_a) or np.isnan(median_b)) else np.nan

        direction = effect_direction(median_a, median_b, group_a, group_b)

        mw = mannwhitney_u(a, b)
        ks = ks_test(a, b)

        rows.append({
            "dataset": dataset.name,
            "group_col": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "feature": feat,
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "median_a": median_a,
            "median_b": median_b,
            "delta_median": delta_median,
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

    # Apply FDR correction independently for MW and KS p-values
    result = adjust_pvalues(result, p_col="mannwhitney_p", method=correction, out_col="mannwhitney_fdr")
    result = adjust_pvalues(result, p_col="ks_p", method=correction, out_col="ks_fdr")

    return result[_COMPARE_COLS].reset_index(drop=True)


def compare_multiple_groups(
    dataset: BiasDataset,
    group_col: str,
    features: Optional[List[str]] = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run Kruskal-Wallis across all groups in *group_col* for each numeric feature.

    Parameters
    ----------
    dataset:
        The BiasDataset to analyse.
    group_col:
        Column in ``dataset.table`` that identifies groups.
    features:
        Numeric feature columns to test. ``None`` uses all numeric columns.
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues`.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns defined in ``_MULTI_COLS``.
    """
    df = dataset.table

    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )

    if features is None:
        numeric_feats = dataset.available_features()
    else:
        numeric_feats = select_numeric_features(df, features=features)

    # Build per-group value arrays once, reused for every feature
    group_labels = sorted(df[group_col].dropna().unique().tolist(), key=str)

    rows = []
    for feat in numeric_feats:
        groups_dict = {
            g: df.loc[df[group_col] == g, feat].dropna().values
            for g in group_labels
        }
        # Only count groups with >= 2 valid values (mirrors kruskal_test internals)
        valid_groups = [g for g, v in groups_dict.items() if len(v) >= 2]
        n_groups = len(valid_groups)

        kw = kruskal_test({g: groups_dict[g] for g in valid_groups})

        rows.append({
            "dataset": dataset.name,
            "group_col": group_col,
            "feature": feat,
            "n_groups": n_groups,
            "groups": ",".join(str(g) for g in valid_groups),
            "kruskal_statistic": kw["statistic"],
            "kruskal_p": kw["p_value"],
            "kruskal_fdr": np.nan,  # filled below
        })

    if not rows:
        return pd.DataFrame(columns=_MULTI_COLS)

    result = pd.DataFrame(rows)
    result = adjust_pvalues(result, p_col="kruskal_p", method=correction, out_col="kruskal_fdr")

    return result[_MULTI_COLS].reset_index(drop=True)


def compare_datasets(
    dataset_a: BiasDataset,
    dataset_b: BiasDataset,
    features: Optional[List[str]] = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare distributions of two separate :class:`~biastracker.dataset.BiasDataset` objects.

    The two datasets do **not** need to share the same rows or IDs.  The
    function merges their tables under a temporary grouping column and
    delegates statistical testing to :func:`compare_groups`.

    Parameters
    ----------
    dataset_a, dataset_b:
        The two datasets to compare.  Their ``.name`` attributes are used as
        group labels and must be distinct.
    features:
        Numeric feature columns to test.  ``None`` uses every numeric column
        found in **both** datasets.  Columns absent from either dataset are
        silently ignored.
    correction:
        Multiple-testing correction method (passed through to
        :func:`~biastracker.stats.adjust_pvalues`).

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`compare_groups` plus a ``comparison_type``
        column set to ``"between_datasets"``.

    Raises
    ------
    ValueError
        If the two dataset names are identical, or if no shared numeric
        features are found after resolving the ``features`` argument.
    """
    if dataset_a.name == dataset_b.name:
        raise ValueError(
            f"dataset_a.name and dataset_b.name are both '{dataset_a.name}'. "
            "They must be distinct so they can be used as group labels."
        )

    # --- resolve the feature intersection -----------------------------------
    feats_a = set(dataset_a.available_features())
    feats_b = set(dataset_b.available_features())

    if features is None:
        shared = sorted(feats_a & feats_b)
    else:
        shared = sorted(
            f for f in features
            if f in feats_a and f in feats_b
        )

    if not shared:
        raise ValueError(
            "No shared numeric features found between the two datasets. "
            f"Dataset A features: {sorted(feats_a)}. "
            f"Dataset B features: {sorted(feats_b)}."
        )

    # --- build combined table -----------------------------------------------
    df_a = dataset_a.table.copy()
    df_b = dataset_b.table.copy()

    df_a[_DATASET_GROUP_COL] = dataset_a.name
    df_b[_DATASET_GROUP_COL] = dataset_b.name

    combined_df = pd.concat([df_a, df_b], ignore_index=True)

    # The combined table needs the minimal schema expected by BiasDataset.
    # "level" and "sequence" are mandatory; carry them forward from both halves.
    if "level" not in combined_df.columns:
        combined_df["level"] = dataset_a.level
    if "sequence" not in combined_df.columns:
        combined_df["sequence"] = ""

    combined_dataset = BiasDataset(
        name=f"{dataset_a.name}_vs_{dataset_b.name}",
        table=combined_df,
        level=dataset_a.level,
        id_col=dataset_a.id_col,
    )

    # --- delegate to compare_groups -----------------------------------------
    result = compare_groups(
        combined_dataset,
        group_col=_DATASET_GROUP_COL,
        group_a=dataset_a.name,
        group_b=dataset_b.name,
        features=shared,
        correction=correction,
    )

    # Drop the internal sentinel column name from the output
    result = result.copy()
    result["group_col"] = "dataset"
    result["comparison_type"] = "between_datasets"

    return result[_DATASET_COMPARE_COLS].reset_index(drop=True)


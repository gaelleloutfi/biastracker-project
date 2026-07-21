import numpy as np
import pandas as pd
from typing import List, Optional

from biastracker.dataset import BiasDataset
from biastracker.preprocessing import DEFAULT_FEATURES, select_numeric_features
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

# Extra column appended by compare_multiple_datasets on top of _MULTI_COLS
_MULTI_DATASET_COLS = _MULTI_COLS + ["comparison_type"]

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
        Numeric feature columns to test. ``None`` uses the curated BiasTracker
        default feature list.
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
    groups = set(df[group_col].dropna().astype(str))
    missing_groups = [g for g in (group_a, group_b) if str(g) not in groups]
    if missing_groups:
        raise ValueError(
            f"group value(s) {missing_groups} not found in group_col '{group_col}'. "
            f"Available groups: {sorted(groups)}"
        )

    numeric_feats = select_numeric_features(df, features=features)
    group_values = df[group_col].astype(str)
    group_a_str = str(group_a)
    group_b_str = str(group_b)
    mask = group_values.isin([group_a_str, group_b_str])
    sub = df[mask]
    sub_groups = group_values[mask]

    vals_a = sub[sub_groups == group_a_str]
    vals_b = sub[sub_groups == group_b_str]

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
            "group_a": group_a_str,
            "group_b": group_b_str,
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
        Numeric feature columns to test. ``None`` uses the curated BiasTracker
        default feature list.
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

    numeric_feats = select_numeric_features(df, features=features)

    # Build per-group value arrays once, reused for every feature
    group_values = df[group_col].astype(str)
    group_labels = sorted(group_values[df[group_col].notna()].unique().tolist(), key=str)

    rows = []
    for feat in numeric_feats:
        groups_dict = {
            g: df.loc[group_values == g, feat].dropna().values
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
        Numeric feature columns to test.  ``None`` uses the curated BiasTracker
        default feature list found in **both** datasets. Columns absent from
        either dataset are silently ignored.
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
    if dataset_a.level != dataset_b.level:
        raise ValueError(
            f"dataset_a.level '{dataset_a.level}' does not match "
            f"dataset_b.level '{dataset_b.level}'"
        )

    # --- resolve the feature intersection -----------------------------------
    resolved_a = select_numeric_features(dataset_a.table, features=features)
    resolved_b = select_numeric_features(dataset_b.table, features=features)
    feats_a = set(resolved_a)
    feats_b = set(resolved_b)

    if features is None:
        shared = [f for f in resolved_a if f in feats_b]
    else:
        shared = [
            f for f in features
            if f in feats_a and f in feats_b
        ]

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


def compare_multiple_datasets(
    datasets: List[BiasDataset],
    features: Optional[List[str]] = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Kruskal-Wallis omnibus comparison across three or more datasets.

    This is the many-dataset analogue of :func:`compare_datasets`. The datasets
    are merged under a temporary grouping column and each numeric feature is
    tested with a Kruskal-Wallis H-test (does *any* dataset differ?), delegating
    to :func:`compare_multiple_groups`.

    Parameters
    ----------
    datasets:
        Two or more datasets to compare. Their ``.name`` attributes are used as
        group labels and must all be distinct, and their ``.level`` must match.
    features:
        Numeric feature columns to test. ``None`` uses the curated BiasTracker
        default feature list restricted to features present in **all** datasets.
    correction:
        Multiple-testing correction method (passed through to
        :func:`~biastracker.stats.adjust_pvalues`).

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`compare_multiple_groups` plus a ``comparison_type``
        column set to ``"between_datasets"``.

    Raises
    ------
    ValueError
        If fewer than two datasets are given, names are not distinct, levels
        differ, or no shared numeric features are found.
    """
    if len(datasets) < 2:
        raise ValueError("compare_multiple_datasets requires at least two datasets.")

    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError(
            f"dataset names must be distinct so they can be used as group labels; got {names}."
        )
    levels = {d.level for d in datasets}
    if len(levels) > 1:
        raise ValueError(f"all datasets must share the same level; got {sorted(levels)}.")

    # --- resolve the feature intersection across every dataset ---------------
    resolved = [select_numeric_features(d.table, features=features) for d in datasets]
    resolved_sets = [set(r) for r in resolved]
    if features is None:
        shared = [f for f in resolved[0] if all(f in s for s in resolved_sets[1:])]
    else:
        shared = [f for f in features if all(f in s for s in resolved_sets)]

    if not shared:
        raise ValueError(
            "No shared numeric features found across the datasets. "
            f"Per-dataset features: {[sorted(s) for s in resolved_sets]}."
        )

    # --- build combined table ------------------------------------------------
    parts = []
    for d in datasets:
        t = d.table.copy()
        t[_DATASET_GROUP_COL] = d.name
        parts.append(t)
    combined_df = pd.concat(parts, ignore_index=True)

    level = datasets[0].level
    if "level" not in combined_df.columns:
        combined_df["level"] = level
    if "sequence" not in combined_df.columns:
        combined_df["sequence"] = ""

    combined_dataset = BiasDataset(
        name="_vs_".join(names),
        table=combined_df,
        level=level,
        id_col=datasets[0].id_col,
    )

    # --- delegate to compare_multiple_groups ---------------------------------
    result = compare_multiple_groups(
        combined_dataset,
        group_col=_DATASET_GROUP_COL,
        features=shared,
        correction=correction,
    )

    result = result.copy()
    result["group_col"] = "dataset"
    result["comparison_type"] = "between_datasets"

    return result[_MULTI_DATASET_COLS].reset_index(drop=True)


def feature_significance(
    datasets: List[BiasDataset],
    feature: str,
    correction: str = "fdr_bh",
) -> Optional[dict]:
    """Significance of a single *feature* across datasets, for on-plot annotation.

    Returns the test result for one feature so a distribution plot can be
    annotated with its significance. Two datasets → Mann-Whitney U; three or more
    → a Kruskal-Wallis omnibus test.

    The FDR is only meaningful relative to a family of tests: when *feature* is
    part of the standard BiasTracker feature panel, the whole panel is tested so
    the returned ``fdr`` matches the Compare tab exactly. For a feature outside
    the panel (e.g. ``paxdb_log10_ppm``) only the nominal ``p_value`` is returned
    and ``fdr`` is ``None``.

    Returns ``None`` (no annotation) when fewer than two datasets carry the
    feature, when dataset names are not distinct, or when their levels differ.
    """
    ds_list = [d for d in datasets if feature in d.table.columns]
    names = [d.name for d in ds_list]
    if len(ds_list) < 2 or len(set(names)) != len(names):
        return None
    if len({d.level for d in ds_list}) > 1:
        return None

    in_panel = feature in DEFAULT_FEATURES
    features_arg = None if in_panel else [feature]

    try:
        if len(ds_list) == 2:
            res = compare_datasets(ds_list[0], ds_list[1], features=features_arg, correction=correction)
            row = res[res["feature"] == feature]
            if row.empty:
                return None
            r = row.iloc[0]
            return {
                "test": "Mann-Whitney U",
                "p_value": float(r["mannwhitney_p"]),
                "fdr": float(r["mannwhitney_fdr"]) if in_panel else None,
                "n_groups": 2,
                "groups": names,
            }
        res = compare_multiple_datasets(ds_list, features=features_arg, correction=correction)
        row = res[res["feature"] == feature]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "test": "Kruskal-Wallis",
            "p_value": float(r["kruskal_p"]),
            "fdr": float(r["kruskal_fdr"]) if in_panel else None,
            "n_groups": len(ds_list),
            "groups": names,
        }
    except ValueError:
        return None

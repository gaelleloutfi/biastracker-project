import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Any

def mannwhitney_u(x, y) -> Dict[str, Any]:
    x_valid = np.asarray(x, dtype=float)
    x_valid = x_valid[~np.isnan(x_valid)]
    y_valid = np.asarray(y, dtype=float)
    y_valid = y_valid[~np.isnan(y_valid)]
    
    if len(x_valid) < 2 or len(y_valid) < 2:
        return {"test": "mannwhitney", "statistic": np.nan, "p_value": np.nan}
        
    stat, p_val = stats.mannwhitneyu(x_valid, y_valid, alternative="two-sided")
    return {"test": "mannwhitney", "statistic": stat, "p_value": p_val}

def ks_test(x, y) -> Dict[str, Any]:
    x_valid = np.asarray(x, dtype=float)
    x_valid = x_valid[~np.isnan(x_valid)]
    y_valid = np.asarray(y, dtype=float)
    y_valid = y_valid[~np.isnan(y_valid)]
    
    if len(x_valid) < 2 or len(y_valid) < 2:
        return {"test": "ks", "statistic": np.nan, "p_value": np.nan}
        
    stat, p_val = stats.ks_2samp(x_valid, y_valid)
    return {"test": "ks", "statistic": stat, "p_value": p_val}

def kruskal_test(groups: Dict[str, Any]) -> Dict[str, Any]:
    valid_groups = []
    for k, v in groups.items():
        v_arr = np.asarray(v, dtype=float)
        v_arr = v_arr[~np.isnan(v_arr)]
        if len(v_arr) >= 2:
            valid_groups.append(v_arr)
            
    if len(valid_groups) < 2:
        return {"test": "kruskal", "statistic": np.nan, "p_value": np.nan}
        
    stat, p_val = stats.kruskal(*valid_groups)
    return {"test": "kruskal", "statistic": stat, "p_value": p_val}

def dunn_test(groups: Dict[str, Any], correction: str = "holm") -> pd.DataFrame:
    """Dunn's post-hoc test: all pairwise comparisons after a Kruskal-Wallis.

    Dunn's test re-uses the *pooled* mid-rank of every observation across all
    groups (the same ranking the Kruskal-Wallis omnibus uses), so it is the
    rank-based post-hoc that stays consistent with the omnibus rather than
    re-ranking each pair independently. Ties are handled with the standard
    tie-correction term. The raw two-sided p-values are then adjusted across the
    whole family of ``K·(K-1)/2`` pairs with ``correction``.

    Parameters
    ----------
    groups:
        Mapping of group label → 1-D values (NaNs dropped per group).
    correction:
        Multiple-testing method for the pairwise family, passed to
        :func:`~statsmodels.stats.multitest.multipletests` (e.g. ``"holm"``,
        ``"bonferroni"``, ``"fdr_bh"``).

    Returns
    -------
    pd.DataFrame
        One row per unordered pair with columns ``group_a``, ``group_b``,
        ``n_a``, ``n_b``, ``mean_rank_a``, ``mean_rank_b``, ``z``, ``p_value``
        (raw) and ``p_adj`` (corrected). Empty if fewer than two groups have
        data.
    """
    cols = ["group_a", "group_b", "n_a", "n_b",
            "mean_rank_a", "mean_rank_b", "z", "p_value", "p_adj"]

    clean: Dict[str, np.ndarray] = {}
    for k, v in groups.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 1:
            clean[k] = arr

    labels = list(clean.keys())
    if len(labels) < 2:
        return pd.DataFrame(columns=cols)

    pooled = np.concatenate([clean[k] for k in labels])
    N = len(pooled)
    ranks = stats.rankdata(pooled)  # average ranks for ties

    mean_rank: Dict[str, float] = {}
    n: Dict[str, int] = {}
    offset = 0
    for k in labels:
        m = len(clean[k])
        mean_rank[k] = float(ranks[offset:offset + m].mean())
        n[k] = m
        offset += m

    # Tie correction: sum over tie-groups of (t^3 - t).
    _, counts = np.unique(pooled, return_counts=True)
    ties = counts[counts > 1].astype(float)
    tie_term = float(np.sum(ties ** 3 - ties))
    sigma_base = (N * (N + 1) / 12.0) - tie_term / (12.0 * (N - 1)) if N > 1 else np.nan

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            se = np.sqrt(sigma_base * (1.0 / n[a] + 1.0 / n[b])) if sigma_base and sigma_base > 0 else np.nan
            if not np.isfinite(se) or se == 0:
                z = np.nan
                p = np.nan
            else:
                z = (mean_rank[a] - mean_rank[b]) / se
                p = float(2.0 * stats.norm.sf(abs(z)))
            rows.append({
                "group_a": a, "group_b": b, "n_a": n[a], "n_b": n[b],
                "mean_rank_a": mean_rank[a], "mean_rank_b": mean_rank[b],
                "z": z, "p_value": p,
            })

    out = pd.DataFrame(rows, columns=[c for c in cols if c != "p_adj"])
    out = adjust_pvalues(out, p_col="p_value", method=correction, out_col="p_adj")
    return out


def adjust_pvalues(df: pd.DataFrame, p_col: str = "p_value", method: str = "fdr_bh", out_col: str = "fdr") -> pd.DataFrame:
    out_df = df.copy()
    if p_col not in out_df.columns:
        return out_df
        
    mask = out_df[p_col].notna()
    if mask.sum() == 0:
        out_df[out_col] = np.nan
        return out_df
        
    pvals = out_df.loc[mask, p_col]
    reject, pvals_corrected, _, _ = multipletests(pvals, method=method)
    
    out_df[out_col] = np.nan
    out_df.loc[mask, out_col] = pvals_corrected
    return out_df

def effect_direction(median_a: float, median_b: float, label_a: str, label_b: str) -> str:
    if np.isnan(median_a) or np.isnan(median_b):
        return "unknown"
    if median_a > median_b:
        return f"higher_in_{label_a}"
    elif median_a < median_b:
        return f"higher_in_{label_b}"
    else:
        return "no_difference"

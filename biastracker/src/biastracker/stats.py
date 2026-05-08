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

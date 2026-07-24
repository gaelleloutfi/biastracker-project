import numpy as np
import pandas as pd
import pytest
from biastracker.stats import (
    mannwhitney_u,
    ks_test,
    kruskal_test,
    dunn_test,
    adjust_pvalues,
    effect_direction
)

def test_mannwhitney_u():
    # Simple valid arrays
    x = [1, 2, 3, 4, 5]
    y = [10, 11, 12, 13, 14]
    
    res = mannwhitney_u(x, y)
    assert res["test"] == "mannwhitney"
    assert not np.isnan(res["statistic"])
    assert not np.isnan(res["p_value"])
    assert res["p_value"] < 0.05
    
    # NaN handling
    x_nan = [1, 2, 3, np.nan, np.nan]
    res_nan = mannwhitney_u(x_nan, y)
    assert not np.isnan(res_nan["p_value"])
    
    # Too small groups
    x_small = [1]
    res_small = mannwhitney_u(x_small, y)
    assert np.isnan(res_small["p_value"])
    assert np.isnan(res_small["statistic"])
    
    x_nan_small = [1, np.nan, np.nan]
    res_nan_small = mannwhitney_u(x_nan_small, y)
    assert np.isnan(res_nan_small["p_value"])

def test_ks_test():
    x = [1, 2, 3, 4, 5]
    y = [10, 11, 12, 13, 14]
    
    res = ks_test(x, y)
    assert res["test"] == "ks"
    assert not np.isnan(res["p_value"])
    
    res_small = ks_test([1], y)
    assert np.isnan(res_small["p_value"])

def test_kruskal_test():
    groups = {
        "A": [1, 2, 3],
        "B": [10, 11, 12],
        "C": [20, 21, 22]
    }
    res = kruskal_test(groups)
    assert res["test"] == "kruskal"
    assert not np.isnan(res["p_value"])
    
    # With NaNs
    groups_nan = {
        "A": [1, 2, np.nan],
        "B": [10, 11, np.nan]
    }
    res_nan = kruskal_test(groups_nan)
    assert not np.isnan(res_nan["p_value"])
    
    # Too small valid groups
    groups_small = {
        "A": [1, 2, 3],
        "B": [10],
        "C": [np.nan, np.nan]
    }
    # Only "A" is valid, so < 2 valid groups
    res_small = kruskal_test(groups_small)
    assert np.isnan(res_small["p_value"])

def test_dunn_test_matches_reference():
    """Dunn's z / raw p reproduce an independently computed reference."""
    groups = {
        "A": [1, 2, 3, 4, 5, 6, 7],
        "B": [10, 11, 12, 13, 14, 15, 16],
        "C": [1, 3, 5, 7, 9, 11, 13],
    }
    out = dunn_test(groups, correction="bonferroni")

    assert len(out) == 3  # 3 unordered pairs
    row = out[(out["group_a"] == "A") & (out["group_b"] == "B")].iloc[0]
    # Reference values (pooled mid-rank Dunn's with tie correction).
    assert row["z"] == pytest.approx(-3.409419, abs=1e-4)
    assert row["p_value"] == pytest.approx(0.000651, abs=1e-5)
    # Bonferroni over 3 pairs = raw * 3.
    assert row["p_adj"] == pytest.approx(min(1.0, 0.000651 * 3), abs=1e-5)

    ac = out[(out["group_a"] == "A") & (out["group_b"] == "C")].iloc[0]
    assert ac["p_adj"] > 0.05  # A vs C not separated


def test_dunn_test_correction_is_monotone_and_ge_raw():
    groups = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]}
    out = dunn_test(groups, correction="holm")
    # Adjusted p is never smaller than the raw p.
    assert (out["p_adj"] >= out["p_value"] - 1e-12).all()


def test_dunn_test_too_few_groups_returns_empty():
    assert dunn_test({"A": [1, 2, 3]}).empty
    assert dunn_test({"A": [1, 2], "B": [np.nan, np.nan]}).empty


def test_adjust_pvalues():
    df = pd.DataFrame({
        "id": ["A", "B", "C", "D", "E"],
        "p_value": [0.01, np.nan, 0.05, 0.001, np.nan]
    })
    
    out_df = adjust_pvalues(df)
    
    # Output should not mutate input
    assert "fdr" not in df.columns
    
    # FDR is computed only for valid ones
    assert "fdr" in out_df.columns
    assert len(out_df) == 5
    
    # Row order preserved, NaNs preserved
    assert np.isnan(out_df.loc[1, "fdr"])
    assert np.isnan(out_df.loc[4, "fdr"])
    
    assert not np.isnan(out_df.loc[0, "fdr"])
    assert not np.isnan(out_df.loc[2, "fdr"])
    assert not np.isnan(out_df.loc[3, "fdr"])
    
    # Smallest p-value should remain the smallest fdr
    assert out_df.loc[3, "fdr"] <= out_df.loc[0, "fdr"]

def test_effect_direction():
    assert effect_direction(10, 5, "A", "B") == "higher_in_A"
    assert effect_direction(5, 10, "A", "B") == "higher_in_B"
    assert effect_direction(10, 10, "A", "B") == "no_difference"
    assert effect_direction(np.nan, 5, "A", "B") == "unknown"
    assert effect_direction(10, np.nan, "A", "B") == "unknown"
    assert effect_direction(np.nan, np.nan, "A", "B") == "unknown"

from biastracker.analysis.summary import summarize_dataset, summarize_groups

def test_summarize_dataset(toy_bias_dataset):
    summary = summarize_dataset(toy_bias_dataset, features=["length", "mw"])
    
    assert list(summary.columns) == ["dataset", "level", "source_type", "metric", "value"]
    
    metrics = dict(zip(summary["metric"], summary["value"]))
    
    assert metrics["n_rows"] == 6
    assert metrics["n_unique_ids"] == 6
    assert metrics["n_with_sequence"] == 6
    assert metrics.get("n_with_expression", 0) == 0 # no expression column in toy data
    
    assert metrics["length_n_non_missing"] == 6
    assert metrics["length_median"] == 16.5 # [10, 12, 15, 18, 20, 22] -> med(15,18)=16.5
    
    # Check basic metadata
    assert all(summary["dataset"] == "toy_proteins")
    assert all(summary["level"] == "protein")
    assert all(summary["source_type"] == "standard")

def test_summarize_dataset_missing_feature(toy_bias_dataset):
    # Should not crash if we ask for a feature that doesn't exist
    summary = summarize_dataset(toy_bias_dataset, features=["length", "missing_feat"])
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert "length_n_non_missing" in metrics
    assert "missing_feat_n_non_missing" not in metrics

def test_summarize_groups(toy_bias_dataset):
    summary = summarize_groups(toy_bias_dataset, group_col="group", features=["length", "mw"])
    
    assert list(summary.columns) == ["dataset", "group_col", "group", "feature", "n", "mean", "median", "std", "min", "max"]
    
    assert len(summary) == 4 # 2 groups * 2 features
    
    g_A_len = summary[(summary["group"] == "A") & (summary["feature"] == "length")].iloc[0]
    
    assert g_A_len["n"] == 3
    assert g_A_len["mean"] == 15.0 # (10+15+20)/3
    assert g_A_len["median"] == 15.0
    assert g_A_len["min"] == 10.0
    
    assert all(summary["dataset"] == "toy_proteins")
    assert all(summary["group_col"] == "group")

def test_summarize_groups_missing_feature(toy_bias_dataset):
    summary = summarize_groups(toy_bias_dataset, group_col="group", features=["length", "missing_feat"])
    assert len(summary) == 2 # 2 groups * 1 feature (length)

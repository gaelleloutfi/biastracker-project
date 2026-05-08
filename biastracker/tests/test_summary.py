import pandas as pd
import numpy as np
import pytest
from biastracker.dataset import BiasDataset
from biastracker.analysis.summary import summarize_dataset, summarize_groups

def test_summarize_dataset():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "level": ["protein", "protein", "protein"],
        "sequence": ["SEQ1", "SEQ2", ""],
        "expression": [10.0, np.nan, 30.0],
        "mw": [100.0, 200.0, 300.0]
    })
    
    ds = BiasDataset(name="test_ds", table=df, level="protein", source_type="manual")
    
    summary = summarize_dataset(ds, features=["mw"])
    
    assert list(summary.columns) == ["dataset", "level", "source_type", "metric", "value"]
    
    metrics = dict(zip(summary["metric"], summary["value"]))
    
    assert metrics["n_rows"] == 3
    assert metrics["n_unique_ids"] == 3
    assert metrics["n_with_sequence"] == 2
    assert metrics["n_with_expression"] == 2
    
    assert metrics["mw_n_non_missing"] == 3
    assert metrics["mw_mean"] == 200.0
    assert metrics["mw_min"] == 100.0
    assert metrics["mw_max"] == 300.0
    
    # Check columns
    assert all(summary["dataset"] == "test_ds")
    assert all(summary["level"] == "protein")
    assert all(summary["source_type"] == "manual")

def test_summarize_groups():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C", "D"],
        "level": ["protein", "protein", "protein", "protein"],
        "sequence": ["S", "S", "S", "S"],
        "group_col": ["G1", "G1", "G2", "G2"],
        "mw": [10.0, 20.0, 100.0, 200.0]
    })
    
    ds = BiasDataset(name="test_ds", table=df, level="protein")
    
    summary = summarize_groups(ds, group_col="group_col", features=["mw"])
    
    assert list(summary.columns) == ["dataset", "group_col", "group", "feature", "n", "mean", "median", "std", "min", "max"]
    
    assert len(summary) == 2
    g1 = summary[summary["group"] == "G1"].iloc[0]
    g2 = summary[summary["group"] == "G2"].iloc[0]
    
    assert g1["feature"] == "mw"
    assert g1["n"] == 2
    assert g1["mean"] == 15.0
    assert g1["min"] == 10.0
    
    assert g2["n"] == 2
    assert g2["mean"] == 150.0
    assert g2["max"] == 200.0
    
    assert all(summary["dataset"] == "test_ds")
    assert all(summary["group_col"] == "group_col")

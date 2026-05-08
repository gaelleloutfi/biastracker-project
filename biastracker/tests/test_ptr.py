import pandas as pd
import pytest
from biastracker.dataset import BiasDataset
from biastracker.analysis.ptr import (
    add_ptr_to_dataset,
    summarize_ptr,
    correlate_ptr_with_features
)

def test_add_ptr_to_dataset(tmp_path):
    # Fake PTR excel file
    ptr_df = pd.DataFrame({
        "Entry": ["P12345", "Q9XXX1"],
        "Entry Name": ["GENE1_HUMAN", "GENE2_HUMAN"],
        "PTR AML": [10.5, 22.0]
    })
    ptr_path = tmp_path / "fake_ptr.xlsx"
    ptr_df.to_excel(ptr_path, index=False)
    
    # Fake dataset
    df = pd.DataFrame({
        "primary_id": ["P12345", "Q9XXX1", "UNKNOWN"],
        "sequence": ["A", "B", "C"],
        "level": ["protein"] * 3
    })
    ds = BiasDataset(name="test", table=df, level="protein")
    
    # Add PTR
    new_ds = add_ptr_to_dataset(ds, str(ptr_path), species="human")
    
    assert "PTR_AML" in new_ds.table.columns
    assert new_ds.metadata.get("ptr_added") is True
    
    # Check values
    assert new_ds.table.loc[new_ds.table["primary_id"] == "P12345", "PTR_AML"].iloc[0] == 10.5
    assert new_ds.table.loc[new_ds.table["primary_id"] == "Q9XXX1", "PTR_AML"].iloc[0] == 22.0
    assert pd.isna(new_ds.table.loc[new_ds.table["primary_id"] == "UNKNOWN", "PTR_AML"].iloc[0])

def test_ptr_summary_and_correlation(tmp_path):
    df = pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4"],
        "sequence": ["A", "B", "C", "D"],
        "level": ["protein"] * 4,
        "PTR_AML": [10.5, 20.0, 30.5, 40.0],
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "group": ["A", "A", "B", "B"]
    })
    
    ds = BiasDataset(name="test", table=df, level="protein", group_col="group")
    
    # Global summary
    summ = summarize_ptr(ds)
    assert len(summ) == 1
    assert summ.iloc[0]["n"] == 4
    assert summ.iloc[0]["mean"] == 25.25
    
    # Group summary
    summ_group = summarize_ptr(ds, group_col="group")
    assert len(summ_group) == 2
    assert summ_group.iloc[0]["group"] == "A"
    assert summ_group.iloc[0]["n"] == 2
    
    # Correlation
    corr_df = correlate_ptr_with_features(ds)
    assert len(corr_df) == 1
    assert corr_df.iloc[0]["feature"] == "feature_1"
    assert corr_df.iloc[0]["n"] == 4
    assert corr_df.iloc[0]["correlation"] > 0.99
    
def test_ptr_missing_error():
    df = pd.DataFrame({
        "primary_id": ["P1", "P2"],
        "sequence": ["A", "B"],
        "level": ["protein"] * 2
    })
    ds = BiasDataset(name="test", table=df, level="protein")
    
    with pytest.raises(ValueError, match="PTR_AML column is missing"):
        summarize_ptr(ds)
        
    with pytest.raises(ValueError, match="PTR_AML column is missing"):
        correlate_ptr_with_features(ds)

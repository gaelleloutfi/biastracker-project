import pandas as pd
import numpy as np
import pytest
from biastracker.preprocessing import (
    select_numeric_features,
    filter_valid_ids,
    filter_valid_sequences,
    make_binary_group,
    add_group_from_mapping,
    clip_feature_quantiles,
    DEFAULT_FEATURES
)

def test_select_numeric_features():
    df = pd.DataFrame({
        "length": [10, 20],
        "mw": [100.0, 200.0],
        "str_col": ["a", "b"],
        "not_in_default": [1, 2]
    })
    
    # default behavior
    feats = select_numeric_features(df)
    assert feats == ["length", "mw"]
    
    # specific features
    feats = select_numeric_features(df, features=["mw", "not_in_default", "str_col", "missing"])
    assert feats == ["mw", "not_in_default"]
    
    # exclude
    feats = select_numeric_features(df, exclude=["mw"])
    assert feats == ["length"]

def test_filter_valid_ids():
    df = pd.DataFrame({
        "primary_id": ["A", "", " ", None, np.nan, "B"],
        "val": [1, 2, 3, 4, 5, 6]
    })
    df_filtered = filter_valid_ids(df)
    
    assert len(df_filtered) == 2
    assert list(df_filtered["primary_id"]) == ["A", "B"]
    
    # does not mutate
    assert len(df) == 6

def test_filter_valid_sequences():
    df = pd.DataFrame({
        "sequence": ["SEQ1", "", " ", None, np.nan, "SEQ2"],
        "val": [1, 2, 3, 4, 5, 6]
    })
    df_filtered = filter_valid_sequences(df)
    
    assert len(df_filtered) == 2
    assert list(df_filtered["sequence"]) == ["SEQ1", "SEQ2"]
    
    assert len(df) == 6

def test_make_binary_group():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "val": [1, 2, 3]
    })
    out_df = make_binary_group(df, id_set={"A", "C"})
    
    assert "group" in out_df.columns
    assert list(out_df["group"]) == ["query", "background", "query"]

    # does not mutate
    assert "group" not in df.columns
    
    # custom labels/cols
    out_df2 = make_binary_group(df, id_set={"B"}, group_col="my_group", positive_label="pos", negative_label="neg")
    assert list(out_df2["my_group"]) == ["neg", "pos", "neg"]

def test_add_group_from_mapping():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C", "D"]
    })
    mapping = {"A": "group1", "B": "group2"}
    
    out_df = add_group_from_mapping(df, mapping)
    
    assert "group" in out_df.columns
    assert list(out_df["group"]) == ["group1", "group2", "other", "other"]
    
    assert "group" not in df.columns

def test_clip_feature_quantiles():
    df = pd.DataFrame({
        "val1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is an outlier
        "val2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "str_col": ["a"] * 10
    })
    
    # 0.1 and 0.9 quantiles for val1 [1..100] are roughly 1.9 and ~90
    out_df = clip_feature_quantiles(df, features=["val1", "val2", "missing", "str_col"], lower=0.1, upper=0.9)
    
    assert out_df["val1"].max() < 100
    assert out_df["val1"].min() > 1
    
    # val2 is clipped slightly at boundaries depending on pd.quantile
    assert out_df["val2"].max() <= 10
    
    assert out_df["str_col"].iloc[0] == "a"
    
    assert df["val1"].max() == 100

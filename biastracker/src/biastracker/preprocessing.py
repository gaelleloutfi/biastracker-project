import pandas as pd
from typing import Set, Dict, List, Optional

DEFAULT_FEATURES = [
    "length", "mw", "pi", "gravy", "instability", "aromaticity", "aliphatic_index",
    "ext_reduced", "ext_cystine", "charge_at_pH", "trypsin_sites",
    "missed_cleavages", "expression"
]

def select_numeric_features(df: pd.DataFrame, features: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> List[str]:
    """Select present numeric features from df, based on a default list or provided features, excluding some."""
    exclude_set = set(exclude) if exclude else set()
    
    if features is None:
        features = DEFAULT_FEATURES
        
    numeric_cols = set(df.select_dtypes(include='number').columns)
    
    selected = []
    for f in features:
        if f in df.columns and f in numeric_cols and f not in exclude_set:
            selected.append(f)
            
    return selected

def filter_valid_ids(df: pd.DataFrame, id_col: str = "primary_id") -> pd.DataFrame:
    """Return rows where id_col is not null and not empty."""
    if id_col not in df.columns:
        return df.copy()
        
    mask = df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")
    return df[mask].copy()

def filter_valid_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where sequence is not null and not empty."""
    if "sequence" not in df.columns:
        return df.copy()
        
    mask = df["sequence"].notna() & (df["sequence"].astype(str).str.strip() != "")
    return df[mask].copy()

def make_binary_group(df: pd.DataFrame, id_set: Set[str], group_col: str = "group", positive_label: str = "query", negative_label: str = "background", id_col: str = "primary_id") -> pd.DataFrame:
    """Add a group column classifying rows as query or background based on id_set."""
    out_df = df.copy()
    if id_col not in out_df.columns:
        raise ValueError(f"Column '{id_col}' not found in dataframe.")
        
    is_positive = out_df[id_col].astype(str).isin(id_set)
    out_df[group_col] = is_positive.map({True: positive_label, False: negative_label})
    return out_df

def add_group_from_mapping(df: pd.DataFrame, mapping: Dict[str, str], id_col: str = "primary_id", group_col: str = "group", default: str = "other") -> pd.DataFrame:
    """Add a group column mapped from id_col using the mapping dictionary."""
    out_df = df.copy()
    if id_col not in out_df.columns:
        raise ValueError(f"Column '{id_col}' not found in dataframe.")
        
    out_df[group_col] = out_df[id_col].astype(str).map(mapping).fillna(default)
    return out_df

def clip_feature_quantiles(df: pd.DataFrame, features: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip values of specific features to given quantiles."""
    out_df = df.copy()
    
    for f in features:
        if f in out_df.columns and pd.api.types.is_numeric_dtype(out_df[f]):
            lower_val = out_df[f].quantile(lower)
            upper_val = out_df[f].quantile(upper)
            out_df[f] = out_df[f].clip(lower=lower_val, upper=upper_val)
            
    return out_df

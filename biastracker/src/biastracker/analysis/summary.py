import pandas as pd
from biastracker.dataset import BiasDataset
from biastracker.preprocessing import select_numeric_features

def summarize_dataset(dataset: BiasDataset, features: list[str] | None = None) -> pd.DataFrame:
    df = dataset.table
    numeric_feats = select_numeric_features(df, features=features)
    
    rows = []
    
    # Counts
    rows.append({"metric": "n_rows", "value": len(df)})
    
    if dataset.id_col and dataset.id_col in df.columns:
        rows.append({"metric": "n_unique_ids", "value": df[dataset.id_col].nunique()})
        
    if "sequence" in df.columns:
        n_seq = df["sequence"].notna() & (df["sequence"].astype(str).str.strip() != "")
        rows.append({"metric": "n_with_sequence", "value": n_seq.sum()})
        
    if "expression" in df.columns:
        rows.append({"metric": "n_with_expression", "value": df["expression"].notna().sum()})
        
    for f in numeric_feats:
        s = df[f]
        rows.append({"metric": f"{f}_n_non_missing", "value": s.notna().sum()})
        rows.append({"metric": f"{f}_mean", "value": s.mean()})
        rows.append({"metric": f"{f}_median", "value": s.median()})
        rows.append({"metric": f"{f}_std", "value": s.std()})
        rows.append({"metric": f"{f}_min", "value": s.min()})
        rows.append({"metric": f"{f}_max", "value": s.max()})
        
    out_df = pd.DataFrame(rows)
    out_df["dataset"] = dataset.name
    out_df["level"] = dataset.level
    out_df["source_type"] = dataset.source_type
    
    return out_df[["dataset", "level", "source_type", "metric", "value"]]

def summarize_groups(dataset: BiasDataset, group_col: str, features: list[str] | None = None) -> pd.DataFrame:
    df = dataset.table
    
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in dataset.table")
        
    numeric_feats = select_numeric_features(df, features=features)
    
    rows = []
    for group_name, group_df in df.groupby(group_col, dropna=False):
        for f in numeric_feats:
            s = group_df[f]
            rows.append({
                "group": group_name,
                "feature": f,
                "n": s.notna().sum(),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max()
            })
            
    if not rows:
        return pd.DataFrame(columns=["dataset", "group_col", "group", "feature", "n", "mean", "median", "std", "min", "max"])
        
    out_df = pd.DataFrame(rows)
    out_df["dataset"] = dataset.name
    out_df["group_col"] = group_col
    return out_df[["dataset", "group_col", "group", "feature", "n", "mean", "median", "std", "min", "max"]]

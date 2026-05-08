import pandas as pd
from typing import Optional, List
from biastracker.dataset import BiasDataset
from protperties import add_ptr_annotation
from scipy.stats import spearmanr, pearsonr

def add_ptr_to_dataset(
    dataset: BiasDataset,
    ptr_table_path: str,
    species: str = "human",
    accession_col: str = "primary_id",
    ptr_out_col: str = "PTR_AML",
) -> BiasDataset:
    """Adds PTR values to a BiasDataset using protperties.add_ptr_annotation.
    
    Parameters
    ----------
    dataset : BiasDataset
        The dataset to annotate.
    ptr_table_path : str
        Path to the PTR reference Excel file.
    species : str
        Species identifier.
    accession_col : str
        Column name containing UniProt accessions.
    ptr_out_col : str
        Output column name for PTR values.
        
    Returns
    -------
    BiasDataset
        A new BiasDataset with the annotated table.
    """
    new_table = add_ptr_annotation(
        dataset.table, 
        ptr_table_path, 
        accession_col=accession_col,
        species=species,
        ptr_out_col=ptr_out_col
    )
    
    return BiasDataset(
        name=dataset.name,
        table=new_table,
        level=dataset.level,
        source_type=dataset.source_type,
        id_col=dataset.id_col,
        group_col=dataset.group_col,
        metadata={**dataset.metadata, "ptr_added": True}
    )

def summarize_ptr(dataset: BiasDataset, group_col: Optional[str] = None) -> pd.DataFrame:
    """Summarizes PTR values globally or by group.
    
    Parameters
    ----------
    dataset : BiasDataset
        The dataset containing PTR values.
    group_col : str, optional
        If provided, summarize by this group column.
        
    Returns
    -------
    pd.DataFrame
        A dataframe with count (n), mean, median, std, min, and max of PTR values.
    """
    if "PTR_AML" not in dataset.table.columns:
        raise ValueError("PTR_AML column is missing from the dataset.")
        
    df = dataset.table.dropna(subset=["PTR_AML"])
    
    if group_col is None:
        stats = df["PTR_AML"].agg(["count", "mean", "median", "std", "min", "max"])
        return stats.to_frame().T.rename(columns={"count": "n"})
    else:
        if group_col not in dataset.table.columns:
            raise ValueError(f"group_col '{group_col}' not found in dataset.")
        stats = df.groupby(group_col)["PTR_AML"].agg(["count", "mean", "median", "std", "min", "max"])
        return stats.rename(columns={"count": "n"}).reset_index()

def correlate_ptr_with_features(
    dataset: BiasDataset,
    features: Optional[List[str]] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Tests whether PTR correlates with numeric features.
    
    Parameters
    ----------
    dataset : BiasDataset
        The dataset containing PTR values and features.
    features : list of str, optional
        List of specific numeric features to correlate with PTR.
    method : str
        Correlation method: "spearman" or "pearson".
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature, n, correlation, and p_value.
    """
    if "PTR_AML" not in dataset.table.columns:
        raise ValueError("PTR_AML column is missing from the dataset.")
        
    available = dataset.available_features(features)
    if "PTR_AML" in available:
        available.remove("PTR_AML")
        
    results = []
    for feat in available:
        df = dataset.table[["PTR_AML", feat]].dropna()
        n = len(df)
        if n < 3:
            continue
            
        if method == "spearman":
            res = spearmanr(df["PTR_AML"], df[feat])
            corr = res.statistic
            pval = res.pvalue
        elif method == "pearson":
            res = pearsonr(df["PTR_AML"], df[feat])
            corr = res[0]
            pval = res[1]
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
            
        results.append({
            "feature": feat,
            "n": n,
            "correlation": corr,
            "p_value": pval
        })
        
    return pd.DataFrame(results)

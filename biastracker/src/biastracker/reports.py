import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict

from biastracker.dataset import BiasDataset
from biastracker.analysis.summary import summarize_dataset

def prepare_output_dirs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Creates the standard result folders for a BiasTracker run.
    
    Args:
        output_dir: The root output directory.
        
    Returns:
        A dictionary containing the paths for 'root', 'tables', and 'figures'.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "root": output_dir,
        "tables": tables_dir,
        "figures": figures_dir,
    }

def save_table(df: pd.DataFrame, output_dir: Union[str, Path], filename: str) -> Path:
    """
    Generic helper to save any result table as CSV in the tables folder.
    
    Args:
        df: The DataFrame to save.
        output_dir: The root output directory.
        filename: The name of the file to save (will append .csv if not present).
        
    Returns:
        The path to the saved CSV file.
    """
    dirs = prepare_output_dirs(output_dir)
    if not filename.endswith(".csv"):
        filename += ".csv"
        
    out_path = dirs["tables"] / filename
    df.to_csv(out_path, index=False)
    return out_path

def save_dataset_summary(
    dataset: BiasDataset, 
    output_dir: Union[str, Path], 
    features: Optional[list[str]] = None,
    filename: str = "dataset_summary.csv"
) -> Path:
    """
    Runs summarize_dataset and saves the dataset summary table.
    
    Args:
        dataset: The BiasDataset to summarize.
        output_dir: The root output directory.
        features: Optional list of features to summarize.
        filename: The filename to save as. Defaults to "dataset_summary.csv".
        
    Returns:
        The path to the saved summary CSV file.
    """
    summary_df = summarize_dataset(dataset, features=features)
    return save_table(summary_df, output_dir, filename)

def save_group_comparison_results(
    results_df: pd.DataFrame, 
    output_dir: Union[str, Path], 
    filename: str = "feature_statistics.csv"
) -> Path:
    """
    Saves the feature comparison table from compare_groups.
    
    Args:
        results_df: The comparison results DataFrame.
        output_dir: The root output directory.
        filename: The filename to save as. Defaults to "feature_statistics.csv".
        
    Returns:
        The path to the saved CSV file.
    """
    return save_table(results_df, output_dir, filename)

def save_enrichment_results(
    enrichment_df: pd.DataFrame, 
    output_dir: Union[str, Path], 
    filename: str = "enrichment_results.csv"
) -> Path:
    """
    Saves ORA/enrichment results.
    
    Args:
        enrichment_df: The enrichment results DataFrame.
        output_dir: The root output directory.
        filename: The filename to save as. Defaults to "enrichment_results.csv".
        
    Returns:
        The path to the saved CSV file.
    """
    return save_table(enrichment_df, output_dir, filename)

import json
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict

from biastracker.dataset import AnnotationSet, BiasDataset
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

def save_annotation_set(
    annotation_set: AnnotationSet,
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """
    Saves a standardized AnnotationSet table to CSV in the annotations folder.
    """
    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    filename = filename or f"{annotation_set.name}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    out_path = annotations_dir / filename
    annotation_set.table.to_csv(out_path, index=False)
    return out_path

def save_annotation_metadata(
    annotation_set: AnnotationSet,
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """
    Saves AnnotationSet metadata to JSON in the annotations folder.
    """
    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    filename = filename or f"{annotation_set.name}.metadata.json"
    if not filename.endswith(".json"):
        filename += ".json"

    metadata = {
        "name": annotation_set.name,
        "source": annotation_set.source,
        "n_rows": int(len(annotation_set.table)),
        "n_unique_ids": int(annotation_set.table[annotation_set.id_col].nunique()),
        "n_terms": int(annotation_set.table[annotation_set.term_col].nunique()),
        "metadata": annotation_set.metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = annotations_dir / filename
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return out_path

def export_annotations_for_workflow(
    loaded_annotations: dict[str, AnnotationSet],
    output_dir: Union[str, Path],
) -> list[Path]:
    """
    Exports all annotation sets loaded for a workflow.
    """
    paths: list[Path] = []
    for annotation_set in loaded_annotations.values():
        paths.append(save_annotation_set(annotation_set, output_dir))
        paths.append(save_annotation_metadata(annotation_set, output_dir))
    return paths

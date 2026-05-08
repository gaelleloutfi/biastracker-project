import pytest
from pathlib import Path
import pandas as pd

from biastracker.dataset import BiasDataset
from biastracker.reports import (
    prepare_output_dirs,
    save_table,
    save_dataset_summary,
    save_group_comparison_results,
    save_enrichment_results
)

def test_prepare_output_dirs(tmp_path):
    output_dir = tmp_path / "results"
    dirs = prepare_output_dirs(output_dir)
    
    assert dirs["root"] == output_dir
    assert dirs["tables"] == output_dir / "tables"
    assert dirs["figures"] == output_dir / "figures"
    
    assert output_dir.exists()
    assert (output_dir / "tables").exists()
    assert (output_dir / "figures").exists()

def test_save_table(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_dir = tmp_path / "results"
    
    # Check that it appends .csv
    path1 = save_table(df, output_dir, "my_table")
    assert path1.name == "my_table.csv"
    assert path1.parent == output_dir / "tables"
    assert path1.exists()
    
    saved_df = pd.read_csv(path1)
    assert "A" in saved_df.columns
    assert list(saved_df["A"]) == [1, 2]
    
    # Check that it doesn't append .csv if already there
    path2 = save_table(df, output_dir, "my_table2.csv")
    assert path2.name == "my_table2.csv"
    assert path2.exists()

def test_save_dataset_summary(tmp_path):
    # Create a small BiasDataset
    df = pd.DataFrame({
        "primary_id": ["P1", "P2", "P3"],
        "level": ["protein", "protein", "protein"],
        "sequence": ["SEQ1", "SEQ2", "SEQ3"],
        "Feat1": [1.0, 2.0, 3.0], 
        "Feat2": [3.0, 4.0, 5.0]
    })
    dataset = BiasDataset(name="test", table=df, level="protein")
    
    output_dir = tmp_path / "results"
    path = save_dataset_summary(dataset, output_dir)
    
    assert path.name == "dataset_summary.csv"
    assert path.exists()
    
    saved_df = pd.read_csv(path)
    # The summarize_dataset returns columns like metric, value, etc.
    assert "metric" in saved_df.columns
    assert "value" in saved_df.columns
    assert len(saved_df) > 0

def test_save_group_comparison_results(tmp_path):
    df = pd.DataFrame({"Feature": ["F1"], "pvalue": [0.05]})
    output_dir = tmp_path / "results"
    
    path = save_group_comparison_results(df, output_dir)
    assert path.name == "feature_statistics.csv"
    assert path.exists()
    assert path.parent == output_dir / "tables"
    
    path2 = save_group_comparison_results(df, output_dir, "custom_stats")
    assert path2.name == "custom_stats.csv"
    assert path2.exists()

def test_save_enrichment_results(tmp_path):
    df = pd.DataFrame({"Term": ["T1"], "pvalue": [0.01]})
    output_dir = tmp_path / "results"
    
    path = save_enrichment_results(df, output_dir)
    assert path.name == "enrichment_results.csv"
    assert path.exists()
    assert path.parent == output_dir / "tables"
    
    path2 = save_enrichment_results(df, output_dir, "custom_enrichment")
    assert path2.name == "custom_enrichment.csv"
    assert path2.exists()

import pytest
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
from biastracker.cli import app

runner = CliRunner()

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "BiasTracker version:" in result.stdout

def test_check():
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "Success" in result.stdout

def test_analyze(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"primary_id": ["P1", "P2"], "sequence": ["", ""], "level": ["protein", "protein"], "feat1": [1.0, 2.0]}).to_csv(input_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "analyze",
        "--input", str(input_csv),
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "dataset_summary.csv").exists()

def test_compare_groups(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4"], 
        "sequence": ["", "", "", ""],
        "level": ["protein", "protein", "protein", "protein"],
        "group": ["A", "A", "B", "B"],
        "feat1": [1.0, 1.2, 5.0, 4.8]
    }).to_csv(input_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "compare-groups",
        "--input", str(input_csv),
        "--group-col", "group",
        "--group-a", "A",
        "--group-b", "B",
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "feature_statistics.csv").exists()
    assert (out_dir / "figures" / "violin_feat1.png").exists()
    assert (out_dir / "figures" / "cdf_feat1.png").exists()

def test_enrich(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4"], 
        "sequence": ["", "", "", ""],
        "level": ["protein", "protein", "protein", "protein"],
        "group": ["A", "A", "B", "B"],
        "feat1": [1.0, 1.2, 5.0, 4.8]
    }).to_csv(input_csv, index=False)
    
    ann_csv = tmp_path / "ann.csv"
    pd.DataFrame({
        "primary_id": ["P1", "P1", "P2", "P3"],
        "term_name": ["T1", "T2", "T1", "T2"]
    }).to_csv(ann_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "enrich",
        "--input", str(input_csv),
        "--group-col", "group",
        "--query-group", "A",
        "--annotations", str(ann_csv),
        "--min-term-size", "1",
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "enrichment_results.csv").exists()

def test_run(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("datasets: []\nanalyses: []\n")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 0
    assert "config-driven execution is recognized but not fully implemented yet" in result.stdout

def test_run_invalid_config(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("invalid_yaml: [}")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    assert "Failed to load config" in result.stdout

def test_run_missing_keys(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("datasets: []\n")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    assert "Error: Config must contain 'datasets' and 'analyses'" in result.stdout

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

def test_analyze(tmp_path, toy_protein_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "analyze",
        "--input", str(input_csv),
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "dataset_summary.csv").exists()

def test_compare_groups(tmp_path, toy_protein_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
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
    assert (out_dir / "figures" / "violin_length.png").exists()
    assert (out_dir / "figures" / "cdf_length.png").exists()

def test_enrich(tmp_path, toy_protein_df, toy_annotation_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
    ann_csv = tmp_path / "ann.csv"
    toy_annotation_df.to_csv(ann_csv, index=False)
    
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

def test_run_valid_config(tmp_yaml_config):
    result = runner.invoke(app, ["run", str(tmp_yaml_config)])
    assert result.exit_code == 0
    assert "Workflow completed successfully" in result.stdout

def test_run_invalid_config(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("invalid_yaml: [}")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    assert "Error loading or validating config" in result.stdout

def test_run_missing_keys(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("datasets: []\n")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    assert "missing required field: 'project_name'" in result.stdout

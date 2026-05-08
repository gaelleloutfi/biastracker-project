import pytest
from pathlib import Path
from typer.testing import CliRunner
from biastracker.cli import app
import yaml
import matplotlib
matplotlib.use('Agg')

runner = CliRunner()

def test_full_workflow(tmp_yaml_config):
    # tmp_yaml_config sets up the environment and returns the path to the config
    result = runner.invoke(app, ["run", str(tmp_yaml_config)])
    
    assert result.exit_code == 0
    
    # Check outputs
    out_dir = tmp_yaml_config.parent / "results"
    
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    
    assert tables_dir.exists()
    assert figures_dir.exists()
    
    # At least one dataset summary
    assert list(tables_dir.glob("dataset_summary*.csv"))
    
    # At least one feature statistics
    assert list(tables_dir.glob("feature_statistics*.csv"))
    
    # At least one enrichment CSV
    assert list(tables_dir.glob("enrichment*.csv"))
    
    # At least one violin plot
    assert list(figures_dir.glob("violin*.png"))
    
    # At least one CDF plot
    assert list(figures_dir.glob("cdf*.png"))
    
    # Enrichment dotplot
    assert list(figures_dir.glob("enrichment_dotplot*.png"))


def test_example_config_exists_and_valid():
    config_path = Path("examples/ghost_proteome_config.yaml")
    if not config_path.exists():
        pytest.skip("Example config not found")
        
    # Check if valid YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    assert isinstance(config, dict)
    assert "project_name" in config

def test_example_config_run():
    config_path = Path("examples/ghost_proteome_config.yaml")
    data_path = Path("examples/data/toy_proteins.csv")
    ann_path = Path("examples/annotations/toy_terms.csv")
    
    if not (config_path.exists() and data_path.exists() and ann_path.exists()):
        pytest.skip("Example data files not found")
        
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0

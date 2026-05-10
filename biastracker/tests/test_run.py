import pytest
import yaml
from typer.testing import CliRunner
from biastracker.cli import app

runner = CliRunner()

def _combined_output(result):
    stderr = result.stderr if result.stderr_bytes is not None else ""
    return (result.stdout or "") + (stderr or "") + (result.output or "")

@pytest.fixture
def run_env(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "test_data.csv"
    data_file.write_text("primary_id,sequence,group,val1\nP1,A,A,1\nP2,B,A,2\nP3,C,B,3\nP4,D,B,4\n")

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann_file = ann_dir / "test_ann.csv"
    ann_file.write_text("primary_id,term_id,term_name,source,category\nP1,T1,Term1,S,C\nP2,T1,Term1,S,C\nP3,T2,Term2,S,C\nP4,T2,Term2,S,C\n")

    config_file = tmp_path / "config.yaml"
    config_data = {
        "project_name": "test_project",
        "datasets": [
            {
                "name": "ds1",
                "type": "standard_csv",
                "path": "data/test_data.csv",
                "level": "protein",
                "group_col": "group"
            }
        ],
        "annotations": [
            {
                "name": "ann1",
                "type": "long",
                "path": "annotations/test_ann.csv",
                "source": "S"
            }
        ],
        "analysis": {
            "summary": True,
            "comparisons": [
                {
                    "dataset": "ds1",
                    "group_col": "group",
                    "group_a": "A",
                    "group_b": "B",
                    "features": ["val1"]
                }
            ],
            "enrichment": [
                {
                    "dataset": "ds1",
                    "annotation": "ann1",
                    "group_col": "group",
                    "query_group": "A",
                    "min_term_size": 1
                }
            ]
        },
        "output": {
            "directory": "results"
        }
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return tmp_path, config_file

def test_run_command_success(run_env):
    tmp_path, config_file = run_env
    
    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Workflow completed successfully" in result.stdout

    results_dir = tmp_path / "results"
    assert results_dir.exists()
    assert (results_dir / "tables").exists()
    assert (results_dir / "figures").exists()

    # Check summary
    assert (results_dir / "tables" / "dataset_summary__ds1.csv").exists()

    # Check comparison
    assert (results_dir / "tables" / "feature_statistics__ds1__A_vs_B.csv").exists()
    assert (results_dir / "figures" / "violin__ds1__val1__A_vs_B.png").exists()
    assert (results_dir / "figures" / "cdf__ds1__val1__A_vs_B.png").exists()

    # Check enrichment
    assert (results_dir / "tables" / "enrichment__ds1__A__ann1.csv").exists()
    assert (results_dir / "figures" / "enrichment_dotplot__ds1__A__ann1.png").exists()

def test_run_unsupported_dataset(run_env):
    tmp_path, config_file = run_env
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    config["datasets"][0]["type"] = "unknown_type"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    output = _combined_output(result)
    assert "Unsupported dataset type 'unknown_type'" in output

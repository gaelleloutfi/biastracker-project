import pytest
from pathlib import Path
import yaml
from biastracker.config import load_config, validate_minimal_config

def test_load_config_valid(tmp_path):
    config_data = {"project_name": "Test", "output": {"directory": "./out"}}
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    loaded = load_config(config_file)
    assert loaded == config_data

def test_load_config_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_file)

def test_load_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("this is not: valid yaml: [")
    
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(config_file)

def test_load_config_not_dict(tmp_path):
    config_file = tmp_path / "not_dict.yaml"
    config_file.write_text("- just\n- a\n- list")
    
    with pytest.raises(ValueError, match="expected dictionary"):
        load_config(config_file)

def test_validate_minimal_config_valid():
    valid_config = {
        "project_name": "MyProject",
        "output": {"directory": "/tmp/out"}
    }
    # Should not raise
    validate_minimal_config(valid_config)

def test_validate_minimal_config_missing_project():
    invalid_config = {
        "output": {"directory": "/tmp/out"}
    }
    with pytest.raises(ValueError, match="project_name"):
        validate_minimal_config(invalid_config)

def test_validate_minimal_config_missing_output():
    invalid_config = {
        "project_name": "MyProject"
    }
    with pytest.raises(ValueError, match="output.directory"):
        validate_minimal_config(invalid_config)

def test_validate_minimal_config_invalid_datasets():
    invalid_config = {
        "project_name": "MyProject",
        "output": {"directory": "/tmp/out"},
        "datasets": "not a list"
    }
    with pytest.raises(ValueError, match="'datasets' field must be a list"):
        validate_minimal_config(invalid_config)

def test_validate_minimal_config_invalid_annotations():
    invalid_config = {
        "project_name": "MyProject",
        "output": {"directory": "/tmp/out"},
        "annotations": "not a list"
    }
    with pytest.raises(ValueError, match="'annotations' field must be a list"):
        validate_minimal_config(invalid_config)

def test_validate_minimal_config_invalid_analysis():
    invalid_config = {
        "project_name": "MyProject",
        "output": {"directory": "/tmp/out"},
        "analysis": []
    }
    with pytest.raises(ValueError, match="'analysis' field must be a dict"):
        validate_minimal_config(invalid_config)

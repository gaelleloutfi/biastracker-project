import yaml
from pathlib import Path
from typing import Dict, Any, Union

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads a YAML config file and returns it as a Python dictionary.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        The parsed configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or does not contain a dictionary at its root.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise ValueError(f"Invalid YAML config: expected dictionary at root, got {type(config).__name__}")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {path}: {e}")

def validate_minimal_config(config: Dict[str, Any]) -> None:
    """
    Checks that the config has the minimum fields needed for a BiasTracker run.
    
    Args:
        config: The configuration dictionary to validate.
        
    Raises:
        ValueError: If the config is invalid or missing required fields.
    """
    if "project_name" not in config:
        raise ValueError("Config is missing required field: 'project_name'")
        
    if "output" not in config or not isinstance(config["output"], dict) or "directory" not in config["output"]:
        raise ValueError("Config is missing required field: 'output.directory'")
        
    if "datasets" in config and config["datasets"] is not None:
        if not isinstance(config["datasets"], list):
            raise ValueError("'datasets' field must be a list if present")
            
    if "annotations" in config and config["annotations"] is not None:
        if not isinstance(config["annotations"], list):
            raise ValueError("'annotations' field must be a list if present")
            
    if "analysis" in config and config["analysis"] is not None:
        if not isinstance(config["analysis"], dict):
            raise ValueError("'analysis' field must be a dict if present")

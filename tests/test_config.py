# tests/test_config.py
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch
from src.config.config_loader import ConfigLoader

# Helper to clear the Singleton state between tests
def reset_config_loader():
    """Resets the Singleton ConfigLoader to ensure test isolation."""
    ConfigLoader._config = None
    ConfigLoader._instance = None

@pytest.fixture(autouse=True)
def run_before_and_after():
    """Fixture to ensure the ConfigLoader is reset before and after every test."""
    reset_config_loader()
    yield
    reset_config_loader()

def test_config_loading_success(tmp_path):
    """
    Verify that a valid config.yaml loads correctly into Dataclasses.
    Uses a sandbox directory to ensure zero side effects on production files.
    """
    # 1. Create a sandbox structure mirroring the project root
    # tmp_path is a unique directory created by pytest for this test
    sandbox_root = tmp_path / "project_root"
    sandbox_configs_dir = sandbox_root / "configs"
    sandbox_configs_dir.mkdir(parents=True, exist_ok=True)
    
    test_config_path = sandbox_configs_dir / "config.yaml"
    
    valid_content = {
        "project_name": "Test_Project",
        "env": {"mode": "dev", "compute": {"gpu": False}},
        "data": {
            "target_column": "target",
            "split_config": {"train": 0.8},
            "preprocessing": {},
            "schema": []
        },
        "paths": {
            "raw_data": "raw", "processed_data": "proc",
            "external_data": "ext", "models": "mod", "logs": "log"
        },
        "model": {"name": "test", "type": "rf", "params": {}},
        "training": {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        },
        "artifacts": {"input_file": "in", "output_file": "out", "base_path": "base"},
        "logging": {"file_path": "f", "level": "INFO", "rotation": "r", "retention": "ret"},
        "env_mapping": {
            "BATCH_SIZE": ["training", "batch_size"],
            "LEARNING_RATE": ["training", "learning_rate"]
        }
    }
    
    with open(test_config_path, "w") as f:
        yaml.dump(valid_content, f)

    # 2. Use Monkey Patching to redirect the ConfigLoader's project_root
    # This is a form of Dependency Injection where we swap the production path
    # for our sandbox path during the execution of this specific test.
    with patch.object(ConfigLoader, 'project_root', sandbox_root):
        config = ConfigLoader().get_config()
        
        # Assertions
        assert config.project_name == "Test_Project"
        assert config.training.learning_rate == 0.01
        assert config.training.batch_size == 32
        assert config.data.target_column == "target"

def test_config_loading_failure(tmp_path):
    """
    Verify that the Gatekeeper raises a ValueError for semantically invalid 
    hyperparameters (e.g., learning_rate > 1.0).
    """
    reset_config_loader()
    
    sandbox_root = tmp_path / "project_root"
    sandbox_configs_dir = sandbox_root / "configs"
    sandbox_configs_dir.mkdir(parents=True, exist_ok=True)
    test_config_path = sandbox_configs_dir / "config.yaml"
    
    # This config is COMPLETE (no KeyErrors), but INVALID (triggers ValueError)
    invalid_content = {
        "project_name": "Fail_Project",
        "env": {"mode": "dev", "compute": {"gpu": False}},
        "data": {
            "target_column": "target",
            "split_config": {"train": 0.8},
            "preprocessing": {},
            "schema": []
        },
        "paths": {
            "raw_data": "r", "processed_data": "p",
            "external_data": "e", "models": "m", "logs": "l"
        },
        "model": {"name": "test", "type": "rf", "params": {}},
        "training": {
            "learning_rate": 99.0,  # <--- THIS IS THE INVALID VALUE (must be < 1)
            "batch_size": 32,
            "epochs": 10
        },
        "artifacts": {"input_file": "in", "output_file": "out", "base_path": "base"},
        "logging": {"file_path": "f", "level": "INFO", "rotation": "r", "retention": "ret"},
        "env_mapping": {
            "BATCH_SIZE": ["training", "batch_size"],
            "LEARNING_RATE": ["training", "learning_rate"]
        }
    }
    
    with open(test_config_path, "w") as f:
        yaml.dump(invalid_content, f)

    with patch.object(ConfigLoader, 'project_root', sandbox_root):
        # Now the Mapping Phase succeeds, and the Validation Phase will catch the error
        with pytest.raises(ValueError) as excinfo:
            ConfigLoader().get_config()
        
        # Verify that our specific validation message is the one caught
        assert "Invalid learning_rate" in str(excinfo.value)


# /src/config/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

# --- Configuration Schema (The Contracts) ---

@dataclass(frozen=True)
class EnvConfig:
    mode: str
    compute: Dict[str, Any]

@dataclass(frozen=True)
class DataConfig:
    target_column: str
    split_config: Dict[str, Any]
    preprocessing: Dict[str, Any]
    schema: list

@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str
    params: Dict[str, Any]

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

@dataclass(frozen=True)
class PathsConfig:
    external_data: str
    logs: str
    models: str
    processed_data: str
    raw_data: str

@dataclass(frozen=True)
class ArtifactConfig:
    input_file: str
    output_file: str
    base_path: str

@dataclass(frozen=True)
class LoggingConfig:
    file_path: str
    level: str
    rotation: str
    retention: str

@dataclass(frozen=True)
class AppConfig:
    """
    The Root Configuration Object.
    Every top-level key in config.yaml maps to a nested Dataclass.
    """
    project_name: str
    env: EnvConfig
    data: DataConfig
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    logging: LoggingConfig
    project_root: str


# --- Singleton Loader ---

class ConfigLoader:
    _instance: Optional["ConfigLoader"] = None
    _config: Optional[AppConfig] = None
    project_root: Optional[Path] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._initialize_config()
        return cls._instance

    @classmethod
    def _initialize_config(cls) -> None:
        # Law #1 & #3: Dynamically calculate project root for Portability.
        cls.project_root = Path(__file__).resolve().parent.parent.parent
        config_path = cls.project_root / "configs" / "config.yaml"
        
        logger.info(f"System: Loading configuration from {config_path}")

        if not config_path.exists():
            logger.error(f"Configuration file missing: {config_path}")
            raise FileNotFoundError(f"Config file missing: {config_path}")

        try:
            with open(config_path, "r") as f:
                raw_data = yaml.safe_load(f)
            
            # Inject the project_root into the raw data for use in paths
            raw_data["project_root"] = str(cls.project_root)

            # Manual Mapping to Dataclasses
            # This ensures that our Config object matches our YAML structure exactly.
            cls._config = AppConfig(
                project_name=raw_data["project_name"],
                env=EnvConfig(**raw_data["env"]),
                data=DataConfig(**raw_data["data"]),
                paths=PathsConfig(**raw_data["paths"]),
                model=ModelConfig(**raw_data["model"]),
                training=TrainingConfig(**raw_data["training"]),
                artifacts=ArtifactConfig(**raw_data["artifacts"]),
                logging=LoggingConfig(**raw_data["logging"]),
                project_root=raw_data["project_root"]
            )
            
            logger.info("System: Configuration successfully mapped to Dataclasses.")
        except Exception as e:
            logger.error(f"System: Failed to map configuration: {e}")
            raise

    @classmethod
    def get_config(cls) -> AppConfig:
        if cls._config is None:
            ConfigLoader() 
        assert cls._config is not None, "Configuration failed to load."
        return cls._config

# Global instance for the Software Stack.
# This provides the Single Source of Truth.
config = ConfigLoader().get_config()

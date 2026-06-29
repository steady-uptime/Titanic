# /src/config/config_loader.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
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

            # Apply environment variable overrides here
            raw_data = cls._apply_env_overrides(raw_data)

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
            
            logger.info("System: Configuration successfully mapped to Dataclasses with Env Overrides applied.")
        except Exception as e:
            logger.error(f"System: Failed to map configuration: {e}")
            raise

    @staticmethod
    def _apply_env_overrides(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements Configuration Precedence.
        Checks for specific Environment Variables to override YAML keys. This allows runtime
        overrides including such things as dev, stage, prod environments; DATA_PATH locations;
        database URLs, API endpoints, API secrets to keep them out of the code and flexible.
        """
        # Define the mapping: { ENV_VAR_NAME: ("YAML_SECTION", "YAML_KEY") }
        # This is the "Source of Truth" for our external configuration interface.
        env_mapping = {
            "DATA_PATH": ("paths", "raw_data"),
            "MODEL_PATH": ("paths", "models"),
            "LOG_LEVEL": ("logging", "level"),
            "BATCH_SIZE": ("training", "batch_size"),
            "LEARNING_RATE": ("training", "learning_rate"),
            "ENV_MODE": ("env", "mode")
        }

        for env_var, (section, key) in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type casting logic: Ensure EnvVars (always strings) match Dataclass types
                target_type = type(raw_data[section][key])
                
                # Cast to float/int if necessary, otherwise keep as string
                if target_type == bool:
                    overridden_value = env_value.lower() in ("true", "1", "yes")
                elif target_type in (int, float):
                    overridden_value = target_type(env_value)
                else:
                    overridden_value = env_value
                
                raw_data[section][key] = overridden_value
                logger.info(f"System: Overriding {section}.{key} with EnvVar {env_var}")

        return raw_data

    @classmethod
    def get_config(cls) -> AppConfig:
        if cls._config is None:
            ConfigLoader() 
        assert cls._config is not None, "Configuration failed to load."
        return cls._config

# Global instance for the Software Stack.
config = ConfigLoader().get_config()

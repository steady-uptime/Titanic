# src/utils/persistence.py
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class ArtifactManager:
    """
    Persistence Engine: Handles the 'How' and 'Where' of storage.
    Decoupled from the ML Model logic.
    """
    def __init__(self, base_path: Path):
        # Law 2: Config-Driven. base_path is passed from the config slice.
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ArtifactManager initialized. Base path: {self.base_path}")

    def save_model(self, model: Any, model_name: str) -> Path:
        """
        Constructs the path and handles the serialization.
        """
        # The logic for filename extension and naming lives here ONLY.
        save_path = self.base_path / f"{model_name}.joblib"
        
        joblib.dump(model, save_path)
        logger.info(f"Model artifact saved to: {save_path}")
        return save_path

    def save_metrics(self, metrics: Dict[str, float], model_name: str, model_uri: Optional[str] = None) -> Path:
        """
        Saves evaluation metrics as a JSON artifact.
        Includes the 'model_uri' to create a traceable link between metrics and weights.
        """
        save_path = self.base_path / f"{model_name}_metrics.json"
        
        # Construct the metadata object (The "Handshake")
        metadata = {
            "model_name": model_name,
            "metrics": metrics,
            "model_uri": model_uri  # The pointer to the actual weights
        }

        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metrics artifact saved to: {save_path}")
        return save_path
import joblib
from pathlib import Path
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model.base import ModelWorker

# Law: Use a named logger for module-specific traceability
logger = logging.getLogger(__name__)

class SklearnRandomForestWorker(ModelWorker):
    def __init__(self, model_cfg: dict):
        """
        Architect Level: Dependency Injection.
        We only pass the 'model' slice of the config.
        """
        super().__init__(model_cfg)
        
        # Extract hyperparameters from the config slice
        params = self.config.get("params", {})
        self.model = RandomForestClassifier(**params)
        
        # Law 3: Portability. 
        # Convert string path from YAML into a Path object.
        self.save_dir = Path(self.config["save_path"])
        logger.info(f"SklearnRandomForestWorker initialized. Save path: {self.save_dir}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info(f"Starting training for {self.config['name']}...")
        self.model.fit(X, y)
        logger.info("Training complete.")

    def save(self) -> None:
        if self.model is None:
            logger.error("Attempted to save a null model.")
            return

        # Ensure the directory exists before attempting to write.
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Law 1: Zero Hardcoding. 
        # The filename is derived from the config name.
        file_path = self.save_dir / f"{self.config['name']}.joblib"
        
        try:
            joblib.dump(self.model, file_path)
            logger.info(f"Model artifact saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}", exc_info=True)
            raise

    def load(self) -> None:
        # Use the same Path logic for loading
        file_path = self.save_dir / f"{self.config['name']}.joblib"
        
        if file_path.exists():
            self.model = joblib.load(file_path)
            logger.info(f"Model loaded from {file_path}")
        else:
            logger.warning(f"No model found at {file_path}. Proceeding with uninitialized model.")
            
# src/model/sklearn_worker.py
from src.config.config_loader import ModelConfig
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.model.base import ModelWorker
from typing import Any
from loguru import logger

class SklearnRandomForestWorker(ModelWorker):
    def __init__(self, model_cfg: ModelConfig):
        # The parent class now accepts the Dataclass
        super().__init__(model_cfg)
        
        params = self.config.params
        self.model = RandomForestClassifier(**params)
        logger.info(f"SklearnRandomForestWorker initialized with name: {self.config.name}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info(f"Starting training for {self.config.name}...")
        self.model.fit(X, y)
        logger.success("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Implementation of the predict contract.
        """
        logger.info(f"Performing inference for model: {self.config.name}")
        predictions = self.model.predict(X)
        return pd.Series(predictions)

    def save(self, persistence_engine: Any) -> None:
        """
        The Worker knows the 'What' (the model object and its name).
        The ArtifactManager knows the 'How' (the filesystem path).
        """
        if self.model is None:
            logger.error("Attempted to save a null model.")
            raise ValueError("Model must be trained before saving.")

        persistence_engine.save_model(
            model=self.model, 
            model_name=self.config.name
        )

    def load(self, persistence_engine: Any) -> None:
        # Implementation would call persistence_engine.load_model(...)
        pass
          
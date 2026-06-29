# src/model/base.py
from typing import Any, TypeVar, Generic, Optional
from src.config.config_loader import ModelConfig
from abc import ABC, abstractmethod
import pandas as pd

# Define a TypeVar for the model to allow for better type inference
T = TypeVar('T')

class ModelWorker(ABC, Generic[T]):
    """
    Abstract Base Class for all model workers.
    Enforces a consistent interface for Training and Inference.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        # By using T, we tell the IDE that 'model' will be a specific 
        # type defined by the subclass (e.g., RandomForestClassifier)
        self.model: Optional[T] = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def save(self, persistence_engine: Any) -> None:
        pass

    @abstractmethod
    def load(self, persistence_engine: Any) -> None:
        pass

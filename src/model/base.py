# src/model/base.py
from dataclasses import dataclass
from typing import Any
from src.config.config_loader import ModelConfig
from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger

class ModelWorker(ABC):
    """
    Abstract Base Class for all model workers.
    Enforces a consistent interface for Training and Inference.
    """
    def __init__(self, config: ModelConfig):
        # This config is a slice (e.g., model_cfg)
        self.config = config
        self.model: Any = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def save(self, persistence_engine: Any) -> None:
        """
        Dependency Injection: The worker receives the engine 
        as a dependency during execution.
        """
        pass

    @abstractmethod
    def load(self, persistence_engine: Any) -> None:
        pass

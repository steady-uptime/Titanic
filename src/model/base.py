from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelWorker(ABC):
    """
    Abstract Base Class for all model workers.
    Enforces a consistent interface across different ML models.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.save_dir = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the provided features and target."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist the trained model to the directory defined in config."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load a persisted model from the directory defined in config."""
        pass
    

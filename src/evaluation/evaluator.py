# src/evaluation/evaluator.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
from typing import Dict
from loguru import logger
from src.model.base import ModelWorker # Assuming your ABC is here

class ModelEvaluator:
    """
    Worker responsible for calculating performance metrics on test datasets.
    """
    @staticmethod
    def evaluate(model: ModelWorker, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Executes the evaluation contract.
        Returns a dictionary of metrics.
        """
        logger.info("Starting model evaluation...")
        
        # Ensure we are working on a copy to maintain immutability of test data
        predictions = model.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions, average='weighted')),
            "precision": float(precision_score(y_test, predictions, average='weighted'))
        }
        
        logger.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics
    
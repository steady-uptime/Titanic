# src/data_processing/feature_engineering.py
import pandas as pd
from typing import Tuple
from loguru import logger

class FeatureEngineer:
    """
    Worker: Handles Data Representation and Feature Engineering.
    Responsibilities: Extraction, Encoding, and X/y Splitting.
    """
    def __init__(self, rules: dict, target_column: str):
        self.rules = rules
        self.target_column = target_column
        logger.info("FeatureEngineer initialized.")

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        logger.info("Starting feature engineering (Representation) phase...")

        # Feature Extraction
        if 'Cabin' in df.columns:
            logger.debug("Extracting 'Deck' from 'Cabin' column...")
            df['Cabin_Deck'] = df['Cabin'].str.extract('([A-Z])')[0]
            df['Cabin_Deck'] = df['Cabin_Deck'].fillna('U')
            df = df.drop(columns=['Cabin'])

        # Encoding
        encode_rules = self.rules.get("encoding_rules", {})
        for col, mapping in encode_rules.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logger.debug(f"Encoded {col} using mapping.")

        # X/y Splitting
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        logger.info(f"Feature engineering complete. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
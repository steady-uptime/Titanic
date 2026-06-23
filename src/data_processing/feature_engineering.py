# src/data_processing/feature_engineering.py
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from loguru import logger

class FeatureEngineer:
    """
    Worker: Handles Data Representation and Feature Engineering.
    Responsibilities: Extraction, Encoding, and X/y Splitting.
    """
    def __init__(self, rules: dict, target_column: str, split_config: dict):
        self.rules = rules
        self.target_column = target_column
        self.split_config = split_config

        logger.info("FeatureEngineer initialized.")

    def transform_and_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Performs feature engineering and returns four distinct artifacts.
        """
        df = df.copy()
        logger.info("Starting feature engineering (Representation) phase...")

        # --- STATELESS TRANSFORMATIONS ---
        # These are safe to do before the split.

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

        # 1. Feature Engineering Logic (Your existing logic)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Law 1 & 2: Use config for test_size and random_state
        # Stratify ensures that the proportion of survivors vs. non-survivors is 
        # the same in both the training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.split_config["test_size"],
            random_state=self.split_config["random_state"],
            stratify=y # Ensures class balance (important for Titanic)
        )
        
        logger.info(f"Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, y_train, X_test, y_test
    
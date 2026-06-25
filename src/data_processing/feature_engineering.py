# src/data_processing/feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series
from typing import Tuple, Dict, Any
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

    def transform_and_split(self, df: pd.DataFrame) -> Tuple[DataFrame, Series, DataFrame, Series, Dict[str, str]]:
        """
        Performs feature engineering and returns five distinct artifacts.
        """
        df = df.copy()
        logger.info("Starting feature engineering (Representation) phase...")

        # Stateless Transformations
        if 'Cabin' in df.columns:
            logger.debug("Extracting 'Deck' from 'Cabin' column...")
            df['Cabin_Deck'] = df['Cabin'].str.extract('([A-Z])')[0]
            df['Cabin_Deck'] = df['Cabin_Deck'].fillna('U')
            df = df.drop(columns=['Cabin'])

        encode_rules = self.rules.get("encoding_rules", {})
        for col, mapping in encode_rules.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logger.debug(f"Encoded {col} using mapping.")

        # Target Validation (Fail-Fast)
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found.")
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

        # Separate the features from the target column
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # The Split (explicit code)
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.split_config["test_size"],
            random_state=self.split_config["random_state"],
            stratify=y 
        )
        
        logger.info(f"Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Dynamic Schema Generation
        # Explicitly cast the keys to str 
        # to satisfy the Dict[str, str] type requirement.
        dynamic_schema: Dict[str, str] = {
            str(col): str(dtype) for col, dtype in X_train.dtypes.items()
        }

        # Final Return
        return X_train, y_train, X_test, y_test, dynamic_schema
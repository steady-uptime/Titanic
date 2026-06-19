# src/data_processing/preprocessing.py
import pandas as pd
from pathlib import Path
from loguru import logger

class DataPreprocessor:
    """
    Worker: Handles Data Sanitization.
    Responsibilities: Dropping columns and Imputation.
    """
    def __init__(self, rules: dict, processed_path: Path):
        self.rules = rules
        self.processed_path = processed_path
        logger.info(f"DataPreprocessor initialized. Target: {processed_path}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy to maintain Immutability
        df = df.copy()
        logger.info("Starting data cleaning (Sanitization) phase...")

        # Drop Columns
        drop_cols = self.rules.get("drop_columns", [])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        logger.debug(f"Dropped columns: {drop_cols}") 

        # Imputation
        impute_rules = self.rules.get("imputation_rules", {})
        for col, strategy in impute_rules.items():
            if col in df.columns:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                logger.debug(f"Imputed {col} using {strategy}")

        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        self.processed_path.mkdir(parents=True, exist_ok=True)
        save_path = self.processed_path / filename
        df.to_csv(save_path, index=False)
        logger.info(f"Successfully saved sanitized data to: {save_path}")
        
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, rules: dict, processed_path: Path):
        self.rules = rules
        self.processed_path = processed_path
        logger.info(f"DataPreprocessor initialized. Target: {processed_path}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy to maintain Immutability (Software Engineering Principle)
        df = df.copy()
        logger.info("Starting data cleaning pipeline...")

        # 1. Drop Columns
        drop_cols = self.rules.get("drop_columns", [])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        logger.info(f"Dropped columns: {drop_cols}")

        # 2. Feature Engineering: Extract 'Deck' from 'Cabin'
        if 'Cabin' in df.columns:
            logger.info("Extracting 'Deck' from 'Cabin' column...")
            # Extract first letter, e.g., 'C85' -> 'C'
            df['Cabin_Deck'] = df['Cabin'].str.extract('([A-Z])')[0]
            # Fill NaNs with 'U' for Unknown
            df['Cabin_Deck'] = df['Cabin_Deck'].fillna('U')
            df = df.drop(columns=['Cabin'])

        # 3. Imputation
        impute_rules = self.rules.get("imputation_rules", {})
        for col, strategy in impute_rules.items():
            if col in df.columns:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    # .mode() returns a Series, we take the first element
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                logger.info(f"Imputed {col} using {strategy}")

        # 4. Encoding
        encode_rules = self.rules.get("encoding_rules", {})
        for col, mapping in encode_rules.items():
            if col in df.columns:
                # Using .map() ensures that values not in the dict 
                # will become NaN, which we can then handle.
                df[col] = df[col].map(mapping)
                logger.info(f"Encoded {col} using mapping.")

        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        self.processed_path.mkdir(parents=True, exist_ok=True)
        save_path = self.processed_path / filename
        df.to_csv(save_path, index=False)
        logger.info(f"Successfully saved processed data to: {save_path}")

"""
ARCHITECTURAL NOTE: 
Currently, 'Cabin_Deck' extraction and 'Encoding' are both inside DataPreprocessor. 
From a Software Engineering perspective, this violates the 'Single Responsibility Principle' 
as the project scales. 

'DataPreprocessor' should ideally handle Sanitization (dropping cols, handling NaNs, 
extracting raw strings). 

'FeatureTransformer' (a separate worker) should handle Representation (mapping 
strings to integers, One-Hot Encoding, Scaling). 

For this Integration Test, we are keeping them coupled for simplicity, but in 
a production system, we would decouple them so we could swap different 
Encoding strategies (e.g., LabelEncoder vs OneHotEncoder) without 
modifying the cleaning logic.
"""

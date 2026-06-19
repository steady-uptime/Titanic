# src/data_processing/validator.py
import pandas as pd
from loguru import logger 
from typing import List, Dict

class DataValidator:
    """
    Performs schema validation on DataFrames to ensure contract adherence 
    between preprocessing and model training.
    """
    
    # Type Normalization Map: Maps logical config types to acceptable physical dtypes
    TYPE_NORM_MAP = {
        "object": ["object", "string", "str", "O"],
        "int64": ["int64", "int32", "int128"],
        "float64": ["float64", "float32"]
    }

    def __init__(self, schema: List[Dict[str, str]]):
        """
        Dependency Injection: The schema is injected via configuration.
        :param schema: List of dicts containing 'column' and 'type'
        """
        self.schema = schema
        logger.debug("DataValidator initialized.") # DEBUG level for internal state

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validates the DataFrame against the configured schema.
        Raises ValueError on contract violation.
        """
        logger.info("Starting schema validation...")
        
        # Check for missing columns
        expected_columns = [item['column'] for item in self.schema]
        actual_columns = df.columns.tolist()
        
        missing_cols = set(expected_columns) - set(actual_columns)
        if missing_cols:
            logger.error(f"Schema violation: Missing columns: {missing_cols}")
            logger.error(f"DataFrame Head at point of failure:\n{df.head()}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Check for type mismatches using Normalization
        for item in self.schema:
            col = item['column']
            expected_type = item['type']
            actual_dtype_str = str(df[col].dtype)

            #  --- This is strict checking which pandas is too strict?
            # Causes -> TypeError: string indices must be integers, not 'str'
            # # Check if the column type matches the expected pandas dtype
            # if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
            #     # Note: This is a strict check. 
            #     # If you need fuzzy matching (e.g., int32 vs int64), 
            #     # logic would be adjusted here.

            # Retrieve acceptable physical types for the logical type
            acceptable_types = self.TYPE_NORM_MAP.get(expected_type, [expected_type])
            
            # Check if the actual dtype falls into the acceptable list
            if not any(t in actual_dtype_str for t in acceptable_types):
                logger.error(f"Schema violation: Column '{col}' expected {expected_type}, got {actual_dtype_str}")
                logger.error(f"DataFrame Head at point of failure:\n{df.head()}")
                raise ValueError(f"Type mismatch on column '{col}': Expected {expected_type}, got {actual_dtype_str}")

        logger.info("Schema validation passed.")

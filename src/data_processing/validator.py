# src/data_processing/validator.py
from loguru import logger
import pandas as pd
from typing import Dict, List, Union

class DataValidator:
    """
    Gatekeeper for Contract Validation.
    Ensures that the processed DataFrame matches the expected
    schema. This class follows a Fail-Fast pattern.
    """
    # Define common type aliases to make the validator robust
    # This handles differences between Pandas versions and SQL types
    TYPE_MAPPINGS = {
        "object": ["object", "str"],
        "str": ["object", "str"],
        "float": ["float64", "float32", "float"],
        "int": ["int64", "int32", "int"]
    }
    
    def __init__(self, expected_schema: Union[Dict[str, str], List[Dict[str, str]]]):
        """
        Accepts either:
        1. A dictionary: {"column_name": "dtype"}
        2. A list of dictionaries: [{"column": "column_name", "type": "dtype"}]
        """
        if isinstance(expected_schema, list):
            # Convert List of Dicts to a flat Dictionary for internal use
            logger.debug("Converting list-based schema to dictionary format.")
            self.expected_schema = {item['column']: item['type'] for item in expected_schema}
        else:
            self.expected_schema = expected_schema

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validates the DataFrame against the provided schema.
        Raises RuntimeError if the contract is violated.
        """
        logger.info("Starting contract validation...")
        
        # Check for missing columns
        missing_cols = set(self.expected_schema.keys()) - set(df.columns)
        if missing_cols:
            logger.error(f"DataFrame Head at point of failure:\n{df.head()}")
            error_msg = f"Contract Validation Failed: Missing columns: {missing_cols}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Check for type mismatches
        for col, expected_type in self.expected_schema.items():
            if col not in df.columns:
                continue
            
            actual_type = str(df[col].dtype)
            
            # Normalization Logic:
            # Check if the expected_type is a known alias for the actual_type
            is_valid = False
            
            # Check if actual_type is in the mapping for the expected_type
            if expected_type in self.TYPE_MAPPINGS and actual_type in self.TYPE_MAPPINGS[expected_type]:
                is_valid = True
            # Check if expected_type is in the mapping for the actual_type (Inverse check)
            elif actual_type in self.TYPE_MAPPINGS and expected_type in self.TYPE_MAPPINGS[actual_type]:
                is_valid = True
            # Exact match fallback
            elif expected_type in actual_type:
                is_valid = True

            if not is_valid:
                logger.error(f"DataFrame Head at point of failure:\n{df.head()}")
                error_msg = f"Contract Validation Failed: Column '{col}' expected {expected_type}, got {actual_type}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        logger.success("Contract validation passed.")

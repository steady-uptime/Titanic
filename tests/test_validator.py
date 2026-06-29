import pytest
import pandas as pd
import numpy as np
from src.data_processing.validator import DataValidator

# --- Mock Schema for Testing ---
TEST_SCHEMA = {
    'Pclass': 'int',
    'Sex': 'int',
    'Age': 'float',
    'SibSp': 'int',
    'Parch': 'int',
    'Fare': 'float',
    'Embarked': 'int',
    'Cabin_Deck': 'float'
}

# --- Test Fixtures ---
@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': [0, 1, 0],
        'Age': [22.0, 38.0, 26.0],
        'SibSp': [1, 0, 0],
        'Parch': [0, 0, 1],
        'Fare': [7.25, 55.8, 26.15],
        'Embarked': [0, 1, 0],
        'Cabin_Deck': [8.0, 8.0, 3.0]
    })

@pytest.fixture
def type_mismatch_df():
    return pd.DataFrame({
        'Pclass': [1, 2],
        'Sex': [0, 1],
        'Age': [22.0, "Unknown"],  # <--- Type Mismatch
        'SibSp': [1, 0],
        'Parch': [0, 0],
        'Fare': [7.25, 55.8],
        'Embarked': [0, 1],
        'Cabin_Deck': [8.0, 8.0]
    })

@pytest.fixture
def missing_column_df():
    return pd.DataFrame({
        'Sex': [0, 1],
        'Age': [22.0, 38.0],
        'SibSp': [1, 0],
        'Parch': [0, 0],
        'Fare': [7.25, 55.8],
        'Embarked': [0, 1],
        'Cabin_Deck': [8.0, 8.0]
    })

# --- Unit Tests ---

def test_validator_success(valid_df):
    """Verify that valid data passes the contract (No exception raised)."""
    validator = DataValidator(expected_schema=TEST_SCHEMA)
    # SUCCESS ARCHITECTURE: 
    # The method returns None. If it reaches the next line, it PASSED.
    validator.validate(valid_df)

def test_validator_type_mismatch(type_mismatch_df):
    """Verify that the validator catches type mismatches (Must raise RuntimeError)."""
    validator = DataValidator(expected_schema=TEST_SCHEMA)
    # NEGATIVE TESTING:
    # We WANT it to crash. pytest.raises catches the crash and marks it as a PASS.
    with pytest.raises(RuntimeError) as excinfo:
        validator.validate(type_mismatch_df)
    
    assert "expected float, got object" in str(excinfo.value)

def test_validator_missing_column(missing_column_df):
    """Verify that the validator catches missing columns (Must raise RuntimeError)."""
    validator = DataValidator(expected_schema=TEST_SCHEMA)
    # NEGATIVE TESTING:
    # Again, we want to verify that the system correctly rejects the bad input.
    with pytest.raises(RuntimeError) as excinfo:
        validator.validate(missing_column_df)
    
    assert "Missing columns" in str(excinfo.value)

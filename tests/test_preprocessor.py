import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_processing.preprocessing import DataPreprocessor

# --- Mock Configuration for Testing ---
# We define exactly what the class needs to function, 
# decoupled from the real config.yaml file.
MOCK_RULES = {
    "drop_columns": ["PassengerId", "Name"],
    "imputation_rules": {
        "Age": "median"
    }
}

MOCK_PATH = Path("data/processed_test")

# --- Test Fixtures ---
@pytest.fixture
def raw_dirty_data():
    return pd.DataFrame({
        'Pclass': [1, 2, 3, 1],
        'Sex': [0, 1, 0, 1],
        'Age': [22.0, np.nan, 38.0, 28.0],  # One NaN to test imputation
        'SibSp': [1, 0, 0, 1],
        'Parch': [0, 0, 1, 0],
        'Fare': [7.25, 55.8, 26.15, 8.0],
        'Embarked': [0, 1, 0, 1],
        'Cabin_Deck': [8.0, 8.0, 3.0, 8.0],
        'PassengerId': [101, 102, 103, 104], # Should be dropped
        'Name': ['Name1', 'Name2', 'Name3', 'Name4'] # Should be dropped
    })

# --- Unit Tests ---

def test_preprocessor_drops_columns(raw_dirty_data):
    """Verify that non-feature columns are removed (Sanitization)."""
    # Injecting the mock dependencies
    preprocessor = DataPreprocessor(rules=MOCK_RULES, processed_path=MOCK_PATH)
    
    processed_df = preprocessor.clean_data(raw_dirty_data)
    
    assert 'PassengerId' not in processed_df.columns
    assert 'Name' not in processed_df.columns
    assert 'Pclass' in processed_df.columns
    assert 'Sex' in processed_df.columns

def test_preprocessor_imputes_age(raw_dirty_data):
    """Verify that Age NaN values are filled with the median (Determinism)."""
    # Injecting the mock dependencies
    preprocessor = DataPreprocessor(rules=MOCK_RULES, processed_path=MOCK_PATH)
    
    processed_df = preprocessor.clean_data(raw_dirty_data)
    
    assert processed_df['Age'].isnull().sum() == 0
    # Median of [22.0, 38.0, 28.0] is 28.0
    assert processed_df.loc[1, 'Age'] == 28.0

def test_preprocessor_immutability(raw_dirty_data):
    """Verify that the input DataFrame is not modified in-place."""
    # Injecting the mock dependencies
    preprocessor = DataPreprocessor(rules=MOCK_RULES, processed_path=MOCK_PATH)
    
    _ = preprocessor.clean_data(raw_dirty_data)
    
    # The original dataframe must remain untouched
    assert 'Name' in raw_dirty_data.columns
    assert raw_dirty_data['Age'].isnull().sum() == 1

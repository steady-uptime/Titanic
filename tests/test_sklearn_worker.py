# tests/test_sklearn_worker.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.config.config_loader import ModelConfig
from src.model.sklearn_worker import SklearnRandomForestWorker
from sklearn.ensemble import RandomForestClassifier

# --- Test Fixtures ---

@pytest.fixture
def mock_model_config():
    """
    Creates a test slice of the ModelConfig.
    This simulates the 'model_cfg' slice extracted in train.py.
    """
    return ModelConfig(
        name="test_rf_worker",
        type="rf",
        params={
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42
        }
    )

@pytest.fixture
def dummy_data():
    """
    Creates minimal synthetic data for training and prediction.
    We use small arrays to ensure tests run in milliseconds.
    """
    # 4 samples, 3 features
    X = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
    y = pd.Series([0, 1, 0, 1])
    return X, y

# --- Unit Tests ---

def test_worker_initialization(mock_model_config):
    """
    Verify that the worker correctly instantiates the Scikit-Learn model
    using the parameters provided in the config slice.
    """
    worker = SklearnRandomForestWorker(mock_model_config)
    
    # Assert that the model object was created
    assert isinstance(worker.model, RandomForestClassifier)

    # Use .get_params() to verify hyperparameters
    # This is the standard Scikit-Learn API for inspecting parameters
    params = worker.model.get_params()
    
    assert params["n_estimators"] == 10
    assert params["max_depth"] == 5
    assert params["random_state"] == 42

def test_worker_train(mock_model_config, dummy_data):
    """
    Verify that the train method executes without error.
    In unit testing, a successful execution without exception is a 'Pass'.
    """
    X, y = dummy_data
    worker = SklearnRandomForestWorker(mock_model_config)
    
    # If this raises any error, pytest will catch it and fail the test.
    worker.train(X, y)

def test_worker_predict(mock_model_config, dummy_data):
    """
    Verify that the predict method returns a pandas Series 
    with the correct dimensions.
    """
    X, y = dummy_data
    worker = SklearnRandomForestWorker(mock_model_config)
    worker.train(X, y)
    
    # Perform prediction
    predictions = worker.predict(X)
    
    # Assertions
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X)

def test_worker_save_contract(mock_model_config, dummy_data):
    """
    Verify the Persistence Contract: The worker must call the persistence 
    engine's save_model method with the correct arguments.
    
    Systems Engineering Context: We use a 'Mock' object here. 
    We don't want to actually save a file to disk; we just want to 
    verify that the worker *attempted* to call the correct method 
    on the dependency provided to it.
    """
    X, y = dummy_data
    worker = SklearnRandomForestWorker(mock_model_config)
    worker.train(X, y)
    
    # Create a Mock object to act as the ArtifactManager
    # This is a 'Test Double' - it mimics the interface of the real class.
    mock_persistence_engine = Mock()
    
    # Execute the save method
    worker.save(mock_persistence_engine)
    
    # Assertions: Verify the 'Contract' was fulfilled.
    # Did the worker call save_model?
    mock_persistence_engine.save_model.assert_called_once()
    
    # Extract the arguments passed to save_model
    args, kwargs = mock_persistence_engine.save_model.call_args
    
    # Verify the model object and name were passed correctly
    assert kwargs['model'] is not None
    assert kwargs['model_name'] == "test_rf_worker"

def test_worker_save_raises_error_if_not_trained(mock_model_config):
    """
    Verify that the worker raises a ValueError if save is called 
    before train. This ensures the state machine integrity.
    """
    worker = SklearnRandomForestWorker(mock_model_config)
    mock_persistence_engine = Mock()
    
    with pytest.raises(ValueError) as excinfo:
        worker.save(mock_persistence_engine)
    
    assert "Model must be trained before saving" in str(excinfo.value)

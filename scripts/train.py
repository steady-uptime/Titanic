# scripts/train.py
from pathlib import Path
from src.config.config_loader import ConfigLoader
from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessing import DataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.model.sklearn_worker import SklearnRandomForestWorker
from src.data_processing.validator import DataValidator
from src.utils.persistence import ArtifactManager
from src.evaluation.evaluator import ModelEvaluator 
from src.utils.logger import setup_logger 
from loguru import logger

def main():
    config = ConfigLoader.get_config()
    setup_logger(config["logging"])
    logger.info("Configuration loaded successfully.")

    # Configuration slicing (Separation of Concerns)
    # Only give the workers the data they need to function
    data_cfg = config["data"]
    artifacts_cfg = config["artifacts"]
    model_cfg = config["model"]
    # Slicing sub-domains for specific workers
    pre_cfg = data_cfg["preprocessing"]
    schema = data_cfg["schema"]
    processed_schema = data_cfg["processed_schema"]

    # --- Data Engineering Phase ---
    logger.info("--- Starting Pipeline: Data Engineering Phase ---")
        
    # Load Raw Data
    loader = DataLoader(data_cfg)
    raw_data = loader.load_csv(artifacts_cfg["input_file"])
    
    #  Contract Validation (Raw)
    validator_raw = DataValidator(schema)
    validator_raw.validate(raw_data)

    # Data Sanitization (Cleaning)
    processed_path_obj = Path(data_cfg['processed_path']) 
    preprocessor = DataPreprocessor(
        rules=pre_cfg, 
        processed_path=processed_path_obj
    )
    sanitized_data = preprocessor.clean_data(raw_data)
    
    # Save intermediate artifact for auditability
    preprocessor.save_processed_data(
        sanitized_data, 
        filename=artifacts_cfg["output_file"]
    )

    engineer = FeatureEngineer(
        rules=pre_cfg, 
        target_column=data_cfg["target_column"], 
        split_config=data_cfg["split_config"]
    )
    
    # IMPORTANT: We transform the data. 
    # If FeatureEngineer returns a DataFrame, we validate it immediately.
    X_train, y_train, X_test, y_test = engineer.transform_and_split(sanitized_data)

    # Contract Validation (Final Gate)
    # We replace the manual "select_dtypes" check with the Validator.
    # This is the "Contract" for the Model Worker.
    logger.info("--- Starting Contract Validation Phase - post-engineering ---")
    validator_final = DataValidator(processed_schema)
    
    # We validate X (the features). 
    # Note: If your processed_schema includes 'Survived', 
    # you may need to pass the full dataframe to the validator 
    # or adjust the schema to only include feature columns.
    validator_final.validate(X_train) 

    # --- STATE INSPECTION ---
    # We use the logger to output the first 5 rows of our feature matrix.
    # This confirms that 'Cabin' is gone and 'Cabin_Deck' is present.
    logger.info(f"Final Feature Matrix (X) Head:\n{X_train.head()}")

    # --- Model Engineering Phase ---
    logger.info("--- Starting Model Engineering Phase ---")
    
    # Initialize Persistence Engine
    # We pass the base_path from the artifacts config slice
    artifact_manager = ArtifactManager(base_path=Path(artifacts_cfg["base_path"]))

    # Initialize Worker (The "How")
    # This uses the model_cfg slice
    worker = SklearnRandomForestWorker(model_cfg)
    
    # Execute Training
    worker.train(X_train, y_train)
    
    # Dependency Injection: The Worker uses the Manager to save
    worker.save(artifact_manager)

    # --- Evaluation Phase ---
    # NOTE: Ensure X_test and y_test are loaded before this point
    # Example: X_test, y_test = loader.load_test_data(...)
    logger.info("--- Starting Evaluation Phase ---")
    metrics = ModelEvaluator.evaluate(worker, X_test, y_test)
    
    # Persistence of metadata
    artifact_manager.save_metrics(metrics, model_cfg["name"])

    logger.success("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
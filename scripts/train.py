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
    pre_cfg = data_cfg["preprocessing"]
    source_schema = data_cfg["schema"]


    # --- Data Engineering Phase ---
    logger.info("--- Starting Pipeline: Data Engineering Phase ---")
        
    # Load Raw Data
    loader = DataLoader(data_cfg)
    raw_data = loader.load_csv(artifacts_cfg["input_file"])
    
    #  Contract Validation (Raw)
    validator_raw = DataValidator(source_schema)
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
    
    # Transform the data. 
    # FeatureEngineer returns a 5-tuple including the dynamic_schema
    X_train, y_train, X_test, y_test, dynamic_schema = engineer.transform_and_split(sanitized_data)

    # Contract Validation (Final Gate)
    # Instead of using schema from config, inject  the dynamic_schema
    logger.info("--- Starting Contract Validation Phase - post-engineering ---")
    validator_final = DataValidator(dynamic_schema)
    
    # Validate X (the features). 
    validator_final.validate(X_train) 

    # --- STATE INSPECTION ---
    # Use the logger to output the first 5 rows of our feature matrix.
    # This confirms that 'Cabin' is gone and 'Cabin_Deck' is present.
    logger.info(f"Final Feature Matrix (X) Head:\n{X_train.head()}")

    # --- Model Engineering Phase ---
    logger.info("--- Starting Model Engineering Phase ---")
    
    # Initialize Persistence Engine
    # Pass the base_path from the artifacts config slice
    artifact_manager = ArtifactManager(base_path=artifacts_cfg["base_path"])
    worker = SklearnRandomForestWorker(model_cfg)
    # Execute Training and save afterwards
    worker.train(X_train, y_train)
    model_path = worker.save(artifact_manager)

    logger.info("--- Starting Evaluation Phase ---")
    metrics = ModelEvaluator.evaluate(worker, X_test, y_test)
    
    # Persistence of metadata
    # Pass the path as a string for JSON serialization
    artifact_manager.save_metrics(
        metrics, 
        model_cfg["name"], 
        model_uri=str(model_path)
    )

    logger.success("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
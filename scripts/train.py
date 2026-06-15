import logging
from pathlib import Path
from src.config.config_loader import ConfigLoader
from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessing import DataPreprocessor
from src.model.sklearn_worker import SklearnRandomForestWorker
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def main():
    config = ConfigLoader.get_config()
    
    # Logic for "Scripted" execution: 
    # We might only want to run the model training part with a specific data slice.
    # Here, we still use the slices to maintain consistency with the pipeline.
    data_cfg = config["data"]
    model_cfg = config["model"]
    artifacts_cfg = config["artifacts"]

    # We extract the specific sub-domain for preprocessing from the data_cfg object.
    # This maintains the "Single Source of Truth" while keeping the Preprocessor
    # decoupled from the overall configuration structure.
    pre_cfg = data_cfg["preprocessing"]

    # --- Explicit Type Casting ---
    # The config.yaml provides a string. The DataPreprocessor requires a Path object.
    # We cast it here at the orchestration level to ensure the worker remains agnostic 
    # of the configuration source.
    processed_path_obj = Path(data_cfg['processed_path']) 

    logger.info("Running manual training script...")

    loader = DataLoader(data_cfg)
    raw_data = loader.load_csv(artifacts_cfg["input_file"])

    preprocessor = DataPreprocessor(
        rules=pre_cfg, 
        processed_path=processed_path_obj
    )
    
    processed_data = preprocessor.clean_data(raw_data)
    preprocessor.save_processed_data(
        processed_data, 
        filename=artifacts_cfg["output_file"]
    )

    # Explicitly showing the "Scripting" flow: 
    # You can easily add print statements or manual checks here for debugging.
    target_col = data_cfg["target_column"]
    X = processed_data.drop(target_col, axis=1)
    y = processed_data[target_col]

    # CONTRACT VALIDATION: Ensure no strings remain in the features
    # This prevents the 'ValueError: could not convert string to float'
    # Extract the filtered columns to a local variable to avoid redundant computation
    # This follows the principle of DRY (Don't Repeat Yourself)
    obj_cols = X.select_dtypes(include=['object']).columns

    # Explicitly check if the resulting Index is non-empty
    if not obj_cols.empty:
        logger.error(f"Data Integrity Error: Non-numeric columns found: {obj_cols.tolist()}")
        raise ValueError("Preprocessing failed to convert all categorical features to numeric.")

    logger.info("--- Starting Model Engineering Phase ---")

    worker = SklearnRandomForestWorker(model_cfg)
    worker.train(X, y)
    worker.save()

    logger.info("Manual training complete.")

if __name__ == "__main__":
    main()
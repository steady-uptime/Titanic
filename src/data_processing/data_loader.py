import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_config: dict):
        """
        Worker: Fetches raw data.
        Decoupled: Receives ONLY the data portion of the config.
        """
        # Law #2: Config-Driven Architecture
        # We use the dictionary passed in from the Orchestrator.
        # Note: We use the keys directly because config["data"] was already passed in.
        self.raw_path = Path(data_config['raw_path'])
        self.processed_path = Path(data_config['processed_path'])
        self.external_path = Path(data_config['external_path'])
        
        logger.info(f"Initialized DataLoader with raw_path: {self.raw_path}")

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Loads a CSV file from the RAW data path.
        :param filename: The name of the file (e.g., 'titanic.csv')
        :return: Loaded pandas DataFrame
        """
        file_path = self.raw_path / filename
        
        logger.info(f"Attempting to load data from: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {filename} was not found in {self.raw_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {filename}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"An error occurred while loading the CSV: {e}")
            raise e

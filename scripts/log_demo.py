# scripts/log_demo.py
from src.config.config_loaderv1 import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger

def main():
    # 1. Load the Singleton Configuration
    cfg = ConfigLoader.get_config()
    
    # Use bracket notation because cfg is a dictionary, not an object.
    # This adheres to the principle of proper Data Structure usage.
    logging_config = cfg.get("logging", {})
    
    # 2. Initialize Logging using the Config Injection
    setup_logger(logging_config)
        
    logger.info("--- Loguru Level Demonstration ---")
    
    # loguru supports the following standard levels:
    logger.debug("DEBUG: Fine-grained information for debugging.")
    logger.info("INFO: Confirmation that things are working as expected.")
    logger.success("SUCCESS: The operation completed successfully.")
    logger.warning("WARNING: Something unexpected happened, but the app is running.")
    logger.error("ERROR: A serious problem occurred; the operation failed.")
    logger.critical("CRITICAL: A failure so severe that the application may stop.")
    
    # Demonstration of structured logging (Contextual Data)
    logger.info("Structured log example", user_id=42, status="active")

if __name__ == "__main__":
    main()

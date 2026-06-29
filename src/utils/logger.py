# src/utils/logger.py
import sys
from loguru import logger

def setup_logger(config_slice: dict):
    """
    Configures the loguru logger based on the provided configuration slice.
    """
    # DEBUG: This will tell us exactly what dictionary is being passed
    print(f"DEBUG: setup_logger received config_slice: {config_slice}")

    # Remove the default loguru handler
    logger.remove()

    # Extract variables with defaults
    log_level = config_slice.get("level", "INFO").upper()
    file_path = config_slice.get("file_path", "logs/pipeline.log")
    rotation = config_slice.get("rotation", "10 MB")
    retention = config_slice.get("retention", "10 days")

    # Add a standard output handler (Stream)
    # We add a nice format for the console so it's easy to read.
    logger.add(
        sys.stdout, 
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    # Add a file handler (Sink)
    # We use a try/except here to catch exactly what loguru complains about
    try:
        # We use logger.debug here because the logger is now active
        logger.debug(f"Attempting to add file sink to: {file_path}")

        logger.add(
            file_path, 
            rotation=rotation,
            retention=retention,
            level=log_level,
            serialize=False
        )
        logger.debug("File sink successfully attached.")
    except Exception as e:
        # If loguru fails, we use the standard library to report the error
        # This prevents the app from crashing silently
        import logging
        logging.error(f"Failed to initialize loguru file sink: {e}")
        raise e

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config.config_loader import ConfigLoader

def setup_logging():
    # Law 5: Use Singleton Config
    config = ConfigLoader.get_config()
    log_cfg = config.get("logging", {})
    
    # Law 3: Portability - Resolve the project root correctly
    # If "project_root" is missing, this will raise a KeyError immediately.
    # This is the correct behavior for a production system.
    project_root_str = config["project_root"] 
    project_root = Path(project_root_str) 
    
    
    # Resolve log directory and file path
    log_dir = project_root / log_cfg.get("path", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    # Parse Rotation (e.g., "10 MB")
    # We need to convert "10 MB" into bytes for the handler
    rotation_str = log_cfg.get("rotation", "10 MB")
    size_mb = int(rotation_str.split()[0])
    max_bytes = size_mb * 1024 * 1024

    # Parse Retention (e.g., "10 days")
    # RotatingFileHandler uses 'backupCount' (how many files to keep)
    # For simplicity in this architecture, we treat "10 days" as "10 files"
    retention_str = log_cfg.get("retention", "10 days")
    backup_count = int(retention_str.split()[0])

    # Configure the Handler
    # This handles the "Writing to File" logic
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    
    # This handles the "Console Output" logic
    console_handler = logging.StreamHandler(sys.stdout)

    # Create a unified formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_cfg.get("level", "INFO"))
    
    # Clear existing handlers to avoid duplicate logs if setup is called twice
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

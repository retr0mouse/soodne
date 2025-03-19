import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

# Create a more detailed format that includes the function name
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
LOG_LEVEL = logging.INFO

# Get the project root directory (assuming we're in app/core)
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, level: int = None) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Use provided level or default
    logger.setLevel(level or LOG_LEVEL)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler with INFO level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler with DEBUG level for more detailed logging
    log_file = LOG_DIR / f"{name.replace('.', '_')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Error file handler for ERROR and above
    error_file = LOG_DIR / f"{name.replace('.', '_')}_error.log"
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    return logger

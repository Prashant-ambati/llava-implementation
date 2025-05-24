"""
Logging utilities for the LLaVA implementation.
"""

import logging
import sys
from pathlib import Path

from ..configs.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE

def setup_logging(name: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        name: Optional name for the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Optional name for the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name) 
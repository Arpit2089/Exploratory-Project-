# utils/logger.py
import logging
import os
import sys

def get_logger(name, logfile=None, level=logging.INFO):
    """
    Creates and returns a logger that outputs to both console and a specified file.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    # Standardize the log format for easy debugging
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (if logfile is provided)
    if logfile:
        log_dir = os.path.dirname(logfile)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(logfile, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Prevent log messages from propagating to the root logger (avoids double printing)
    logger.propagate = False
    
    return logger
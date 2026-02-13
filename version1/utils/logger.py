# utils/logger.py
import logging
import sys
from pathlib import Path


def get_logger(name=__name__, level=logging.INFO, logfile=None):
    """
    Production-ready logger.

    - Console prints INFO and above
    - File logs (if provided) store DEBUG and above
    - Prevents duplicate handlers
    """

    logger = logging.getLogger(name)

    # Avoid re-configuring the same logger
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # master level (handlers control actual output)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # -------- Console Handler (clean output) --------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)  # usually INFO
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # -------- File Handler (full debug logs) --------
    if logfile:
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)  # always store full detail
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False  # prevent duplicate prints
    return logger

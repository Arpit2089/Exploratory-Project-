# utils/logger.py
import logging
import sys
from pathlib import Path

def get_logger(name=__name__, level=logging.DEBUG, logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

"""
logger.py

Provides a single, consistently-configured logger factory used across
every module in the pipeline (Spark session, loaders, scraper, validator,
main orchestrator).

Usage
-----
from utils.logger import get_logger
logger = get_logger(__name__)
logger.info("message")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.config import LOG_FILE, LOG_FORMAT, LOG_LEVEL, LOG_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured to write to both stdout and a shared
    pipeline log file. Safe to call multiple times for the same name;
    handlers are only attached once.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. re-imported) - avoid duplicate handlers.
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(Path(LOG_FILE), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(LOG_LEVEL)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

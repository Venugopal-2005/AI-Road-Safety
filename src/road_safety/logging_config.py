"""Logging setup for the Road Safety System."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "road_safety",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_level: Log severity string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to write log files.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Convert level string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if logger was already initialized
    if logger.handlers:
        return logger

    # Formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream handler (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File handler (if requested)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning("Failed to initialize file logger at %s: %s", log_file, e)

    return logger

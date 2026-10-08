"""Logging utilities."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional


def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Get a module-level logger with optional file output.

    Args:
        name: Logger name (typically __name__).
        log_dir: Directory for log files. If None, only console output.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

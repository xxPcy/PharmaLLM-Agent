from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    name: str | None = None,
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a project logger."""

    logger = logging.getLogger(name)
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(numeric_level)
    logger.propagate = False

    if getattr(logger, "_pharma_llm_configured", False):
        return logger

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    setattr(logger, "_pharma_llm_configured", True)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger without mutating global logging state."""

    return logging.getLogger(name)


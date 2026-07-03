"""Logging configuration using loguru.

- Console: colored output, INFO level
- File: no color, DEBUG level, 10MB rotation, 5 backups
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

# Remove default handler
_logger.remove()

# Console handler: colored, INFO level
_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File handler: no color, DEBUG level, rotation
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

_logger.add(
    str(_log_dir / "inferx.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    colorize=False,
    rotation="10 MB",
    retention=5,
    compression="gz",
    encoding="utf-8",
)

# Re-export for use across the codebase
logger = _logger

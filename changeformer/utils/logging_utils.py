"""
Lightweight logging utilities for ChangeFormer.

Provides a simple `get_logger` helper used by entry scripts.
"""

import logging
from typing import Optional


_is_basic_configured = False


def _ensure_basic_config(level: int = logging.INFO) -> None:
    """Ensure basic logging configuration is applied once.

    Args:
        level: Default logging level for root logger.
    """
    global _is_basic_configured
    if not _is_basic_configured:
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _is_basic_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        A logging.Logger instance configured with a basic formatter.
    """
    _ensure_basic_config()
    return logging.getLogger(name)



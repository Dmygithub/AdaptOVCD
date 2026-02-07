"""
Identification module for change detection.

This module contains DGTRS semantic identification implementation.
"""

from .base import BaseIdentifier
from .dgtrs import DGTRSIdentifier

__all__ = [
    'BaseIdentifier',
    'DGTRSIdentifier'
]
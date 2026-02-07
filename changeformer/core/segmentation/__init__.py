"""
Segmentation module for change detection.

This module contains SAM segmentation implementation.
"""

from .base import BaseSegmentor
from .sam import SAMSegmentor

__all__ = [
    'BaseSegmentor',
    'SAMSegmentor'
]
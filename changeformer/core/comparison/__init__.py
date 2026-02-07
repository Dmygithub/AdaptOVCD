"""
Comparison module for change detection.

This module contains various feature comparison implementations including DINO, DINOv2, and DINOv3.
"""

from .base import BaseComparator
from .dino import DINOComparator
from .dinov2 import DINOv2Comparator
from .dinov3 import DINOv3Comparator

__all__ = [
    'BaseComparator',
    'DINOComparator',
    'DINOv2Comparator',
    'DINOv3Comparator'
]
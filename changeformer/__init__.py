"""
ChangeFormer: A modular change detection framework.

This package provides a unified interface for various change detection models
combining different segmentors, comparators, and identifiers.
"""

import os
import sys

__version__ = "1.0.0"

# Centralized path management - add all necessary paths once
def _setup_paths():
    """Setup all necessary paths for the ChangeFormer framework."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Add project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add third party libraries
    third_party_paths = [
        os.path.join(project_root, 'third_party'),
        os.path.join(project_root, 'third_party', 'APE'),
        os.path.join(project_root, 'third_party', 'CLIP'),
        os.path.join(project_root, 'third_party', 'detectron2'),
        os.path.join(project_root, 'third_party', 'detrex'),
        os.path.join(project_root, 'third_party', 'DGTRS'),
        os.path.join(project_root, 'third_party', 'sam2'),
        os.path.join(project_root, 'third_party', 'SegEarth_OV'),
        os.path.join(project_root, 'third_party', 'segment_anything'),
        os.path.join(project_root, 'third_party', 'SimFeatUp'),
    ]
    
    for path in third_party_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

# Setup paths when the package is imported
_setup_paths()

from .models import build_pipeline

__all__ = [
    "build_pipeline",
] 
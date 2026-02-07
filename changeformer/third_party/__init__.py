"""
Third-party library adapters for ChangeFormer.

This module provides unified interfaces to third-party libraries, 
managing their import paths and dependencies.
"""

import os
import sys

def _setup_third_party_paths():
    """Setup paths for all third-party libraries."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Third-party library paths
    libs_root = os.path.join(project_root, 'third_party')
    
    third_party_paths = [
        libs_root,
        os.path.join(libs_root, 'APE'),
        os.path.join(libs_root, 'CLIP'),
        os.path.join(libs_root, 'detectron2'),
        os.path.join(libs_root, 'detrex'),
        os.path.join(libs_root, 'DGTRS'),
        os.path.join(libs_root, 'sam2'),
        os.path.join(libs_root, 'SegEarth_OV'),
        os.path.join(libs_root, 'segment_anything'),
        os.path.join(libs_root, 'SimFeatUp'),
    ]
    
    for path in third_party_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

# Setup paths when imported
_setup_third_party_paths()

__all__ = [
    "ape_adapter",
    "sam_adapter", 
    "clip_adapter"
] 
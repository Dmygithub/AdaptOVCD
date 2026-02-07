"""
SAM (Segment Anything Model) adapter.

This module provides a clean interface to the SAM and SAM2 libraries,
handling imports and providing unified access patterns.
"""

import sys
import os
from typing import Any, Tuple


def get_sam_model_registry():
    """Get SAM and SAM-HQ model registry functions."""
    try:
        from segment_anything import sam_model_registry
        try:
            from segment_anything import sam_hq_model_registry
            return sam_model_registry, sam_hq_model_registry
        except ImportError:
            # SAM-HQ not available, return standard SAM registry twice
            return sam_model_registry, sam_model_registry
    except ImportError as e:
        raise ImportError(f"Failed to import SAM model registry: {e}")


def get_sam_mask_utils():
    """Get SAM mask utility functions."""
    try:
        from segment_anything.utils.amg import rle_to_mask, MaskData
        return rle_to_mask, MaskData
    except ImportError as e:
        raise ImportError(f"Failed to import SAM mask utilities: {e}")


def get_sam2_builder():
    """Get SAM2 model builder."""
    try:
        # Return a builder object with the build_sam2 function
        class SAM2Builder:
            @staticmethod
            def build_sam2(config_file, checkpoint, device='cuda'):
                from sam2.build_sam import build_sam2
                return build_sam2(config_file, checkpoint, device=device)
        
        return SAM2Builder()
    except ImportError as e:
        raise ImportError(f"Failed to import SAM2 builder: {e}")


def get_sam_mask_proposal():
    """Get SAM mask proposal functionality."""
    try:
        # This will be imported via the core segmentation module
        return None  # Placeholder for now
    except ImportError as e:
        raise ImportError(f"Failed to import SAM mask proposal: {e}")


def get_sam2_mask_proposal():
    """Get SAM2 mask proposal functionality."""
    try:
        # This will be imported via the core segmentation module
        return None  # Placeholder for now  
    except ImportError as e:
        raise ImportError(f"Failed to import SAM2 mask proposal: {e}")


__all__ = [
    "get_sam_model_registry",
    "get_sam_mask_utils",
    "get_sam2_builder",
    "get_sam_mask_proposal", 
    "get_sam2_mask_proposal"
] 
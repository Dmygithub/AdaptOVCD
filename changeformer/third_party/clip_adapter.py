"""
CLIP and Vision Transformer models adapter.

This module provides a clean interface to DINO, DINOv2, and other CLIP-based models.
"""

import sys
import os
from typing import Any, Tuple


def get_dino_model(device: str = 'cpu'):
    """
    Get DINO model and processor.
    
    Args:
        device: Device to load the model on.
        
    Returns:
        Tuple of (model, processor).
    """
    try:
        from changeformer.core.utils.model_utils import get_dino_model_and_processor
        return get_dino_model_and_processor(device)
        
    except Exception as e:
        raise ImportError(f"Failed to load DINO model: {e}")


def get_dinov2_model(device: str = 'cpu'):
    """
    Get DINOv2 model and processor.
    
    Args:
        device: Device to load the model on.
        
    Returns:
        Tuple of (model, processor).
    """
    try:
        from changeformer.core.utils.model_utils import get_dinov2_model_and_processor
        return get_dinov2_model_and_processor(device)
        
    except Exception as e:
        raise ImportError(f"Failed to load DINOv2 model: {e}")


def get_segearth_model(device: str = 'cpu', name_list: list = None):
    """
    Get SegEarth-OV model.
    
    Args:
        device: Device to load the model on.
        name_list: List of class names for segmentation.
        
    Returns:
        Tuple of (model, processor).
    """
    try:
        # This is a placeholder - actual implementation would load SegEarth-OV
        # from the third_party/SegEarth_OV directory
        return None, None
        
    except Exception as e:
        raise ImportError(f"Failed to load SegEarth-OV model: {e}")


def get_dgtrs_model(device: str = 'cpu', confidence_threshold: float = 0.2):
    """
    Get DGTRS-CLIP model.
    
    Args:
        device: Device to load the model on.
        confidence_threshold: Confidence threshold for predictions.
        
    Returns:
        Tuple of (model, processor).
    """
    try:
        # This is a placeholder - actual implementation would load DGTRS
        # from the third_party/DGTRS directory
        return None, None
        
    except Exception as e:
        raise ImportError(f"Failed to load DGTRS model: {e}")


__all__ = [
    "get_dino_model",
    "get_dinov2_model", 
    "get_segearth_model",
    "get_dgtrs_model"
] 
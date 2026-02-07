"""
DGTRS-CLIP identifier implementation.

This module implements the DGTRS-CLIP-based semantic identification for change detection.
Based on the original sam_dinov2_dgtrs_pipeline.py implementation.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

from .base import BaseIdentifier


class DGTRSIdentifier(BaseIdentifier):
    """
    DGTRS-CLIP-based semantic identification for change detection.
    
    Uses DGTRS-CLIP to perform semantic classification of detected changes.
    Note: This implementation uses the existing changeformer.core.identification.identifiers
    functions to avoid dynamic_earth import conflicts.
    """
    
    def __init__(self):
        """Initialize DGTRS identifier."""
        super().__init__()
        self.confidence_threshold = None
        self.name_list = None
    
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup DGTRS identifier with configuration.
        
        Args:
            config: DGTRS configuration dictionary
            device: Device to run on
        """
        self.validate_config(config)
        self.config = config
        self.device = device
        
        # Configuration parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.name_list = config.get('name_list', ['background', 'building'])
        self.model_path = config.get('model_path', None)
        
        # Extract target class information for identification logic
        self.target_class = self._extract_target_class(self.name_list)
        
        # Import and setup DGTRS identifier from existing core module
        try:
            from changeformer.core.identification.identifiers import get_dgtrs_identifier
            
            self.model, self.processor = get_dgtrs_identifier(
                device=device,
                name_list=self.name_list,
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold
            )
            
            # Attach target class information to model for identification logic
            self.model.target_class = self.target_class
            
            self._is_setup = True
            print(f"DGTRS identifier initialized on {device}")
            
        except ImportError as e:
            print(f"Warning: DGTRS components not available: {e}")
            raise ImportError("DGTRS components are not available")
    
    def identify(self, img1: np.ndarray, img2: np.ndarray, change_masks: List[np.ndarray], 
                 img1_mask_num: int) -> Tuple[List[np.ndarray], int]:
        """
        Perform semantic identification using DGTRS-CLIP.
        
        Args:
            img1: First input image
            img2: Second input image
            change_masks: List of change masks to identify
            img1_mask_num: Number of masks from first image
            
        Returns:
            Tuple of (identified_masks, img1_mask_num)
        """
        if not self._is_setup:
            raise RuntimeError("DGTRS identifier not setup. Call setup() first.")
        
        if len(change_masks) == 0:
            return [], img1_mask_num
        
        # Use the existing identify function from core module
        try:
            from changeformer.core.identification.identifiers import identify_with_dgtrs
            
            identified_masks, updated_img1_mask_num = identify_with_dgtrs(
                img1, img2, change_masks, img1_mask_num,
                self.model, self.processor,
                model_type='DGTRS-CLIP',
                device=self.device
            )
            
            return identified_masks, updated_img1_mask_num
            
        except ImportError:
            # Fallback: return all change masks without identification
            print("Warning: DGTRS identification not available, returning all change masks")
            return change_masks, img1_mask_num
    
    def _extract_target_class(self, name_list: List[str]) -> str:
        """Extracts target class from name list.
        
        Args:
            name_list: List of class names.
            
        Returns:
            Target class name.
        """
        if len(name_list) >= 2:
            # Typically the second item is the target class, the first is background
            target_class = name_list[1] if isinstance(name_list[1], str) else str(name_list[1])
            # If comma-separated, extract the primary class
            if ',' in target_class:
                # Take the first primary class
                target_class = target_class.split(',')[0].strip()
            return target_class.lower()
        return 'unknown'
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate DGTRS configuration."""
        super().validate_config(config)
        
        # Check name_list
        name_list = config.get('name_list', [])
        if not isinstance(name_list, list) or len(name_list) == 0:
            raise ValueError("name_list must be a non-empty list")
        
        # Check confidence threshold
        confidence_threshold = config.get('confidence_threshold', 0.5)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get DGTRS configuration template."""
        return {
            "type": "DGTRS-CLIP",
            "confidence_threshold": 0.5,
            "name_list": ['background', 'building'],
            "model_path": None
        }
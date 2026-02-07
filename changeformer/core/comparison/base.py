"""
Base classes for comparison components.

This module defines the abstract interfaces that all comparators must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class BaseComparator(ABC):
    """
    Abstract base class for all comparators.
    
    A comparator takes two images and their masks, then determines which masks
    represent actual changes between the images using feature comparison.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.device = None
        self.config = None
        self.model = None
        self.processor = None
        self._is_setup = False
    
    @abstractmethod
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup the comparator with configuration.
        
        Args:
            config: Comparator-specific configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        pass
    
    @abstractmethod
    def compare(self, img1: np.ndarray, img2: np.ndarray, masks: List[np.ndarray], 
                img1_mask_num: int) -> Tuple[List[np.ndarray], int]:
        """
        Compare features between two images to identify change masks.
        
        Args:
            img1: First input image as numpy array (H, W, C)
            img2: Second input image as numpy array (H, W, C)
            masks: List of binary masks to compare
            img1_mask_num: Number of masks from the first image
            
        Returns:
            Tuple of:
            - List of change masks (masks that represent actual changes)
            - Updated number of masks from the first image
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for this comparator.
        
        Supports two modes:
        1. Legacy mode: requires 'type', 'feature_dim', 'patch_size'
        2. New mode: requires 'variant' (auto-configures feature_dim and patch_size)
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for either 'type' or 'variant'
        if 'type' not in config and 'variant' not in config:
            raise ValueError("Comparator config must specify 'type' or 'variant'")
        
        # If using variant mode, skip feature_dim and patch_size validation
        # (they will be auto-configured by factory)
        if 'variant' in config and 'type' not in config:
            return
        
        # Legacy mode: validate required fields
        required_fields = ['feature_dim', 'patch_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Comparator config must specify '{field}'")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get a template configuration for this comparator.
        
        Returns:
            Template configuration dictionary with default values
        """
        return {
            "type": self.__class__.__name__.replace('Comparator', '').upper(),
            "feature_dim": 768,
            "patch_size": 14,
            "change_confidence_threshold": 145
        }
    
    def __str__(self) -> str:
        """String representation of the comparator."""
        return f"{self.__class__.__name__}(setup={self._is_setup})"
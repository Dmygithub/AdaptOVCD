"""
Base classes for identification components.

This module defines the abstract interfaces that all identifiers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class BaseIdentifier(ABC):
    """
    Abstract base class for all identifiers.
    
    An identifier takes change masks and performs semantic classification
    to filter or categorize the detected changes.
    """
    
    def __init__(self):
        """Initialize the identifier."""
        self.device = None
        self.config = None
        self.model = None
        self.processor = None
        self._is_setup = False
    
    @abstractmethod
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup the identifier with configuration.
        
        Args:
            config: Identifier-specific configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        pass
    
    @abstractmethod
    def identify(self, img1: np.ndarray, img2: np.ndarray, change_masks: List[np.ndarray], 
                 img1_mask_num: int) -> Tuple[List[np.ndarray], int]:
        """
        Perform semantic identification on change masks.
        
        Args:
            img1: First input image as numpy array (H, W, C)
            img2: Second input image as numpy array (H, W, C)
            change_masks: List of change masks to identify
            img1_mask_num: Number of masks from the first image
            
        Returns:
            Tuple of:
            - List of identified/filtered change masks
            - Updated number of masks from the first image
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for this identifier.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'type' not in config:
            raise ValueError("Identifier config must specify 'type'")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get a template configuration for this identifier.
        
        Returns:
            Template configuration dictionary with default values
        """
        return {
            "type": self.__class__.__name__.replace('Identifier', '').upper(),
            "name_list": ['background', 'building']
        }
    
    def __str__(self) -> str:
        """String representation of the identifier."""
        return f"{self.__class__.__name__}(setup={self._is_setup})"
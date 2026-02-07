"""
Base classes for segmentation components.

This module defines the abstract interfaces that all segmentors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class BaseSegmentor(ABC):
    """
    Abstract base class for all segmentors.
    
    A segmentor takes two images and generates masks for change detection.
    Different segmentors (SAM, SAM2, etc.) implement different segmentation strategies.
    """
    
    def __init__(self):
        """Initialize the segmentor."""
        self.device = None
        self.config = None
        self._is_setup = False
    
    @abstractmethod
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup the segmentor with configuration.
        
        Args:
            config: Segmentor-specific configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        pass
    
    @abstractmethod
    def segment(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """
        Generate segmentation masks for two input images.
        
        Args:
            img1: First input image as numpy array (H, W, C)
            img2: Second input image as numpy array (H, W, C)
            
        Returns:
            Tuple of:
            - List of binary masks as numpy arrays
            - Number of masks from the first image
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for this segmentor.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'type' not in config:
            raise ValueError("Segmentor config must specify 'type'")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get a template configuration for this segmentor.
        
        Returns:
            Template configuration dictionary with default values
        """
        return {
            "type": self.__class__.__name__.replace('Segmentor', '').upper(),
            "device": "cuda"
        }
    
    def __str__(self) -> str:
        """String representation of the segmentor."""
        return f"{self.__class__.__name__}(setup={self._is_setup})"
"""
Base classes for OVCD enhancement modules.

This module defines the interface that all enhancement modules must implement
to ensure compatibility with the OVCD pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import numpy as np


class BaseEnhancementModule(ABC):
    """
    Base class for all OVCD enhancement modules.
    
    All enhancement modules should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhancement module.
        
        Args:
            config: Configuration dictionary for the module
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the module with its configuration.
        This method is called once when the module is first used.
        """
        pass
    
    @abstractmethod
    def get_module_type(self) -> str:
        """
        Get the type of this enhancement module.
        
        Returns:
            Module type string (e.g., 'threshold_adjustment', 'feature_enhancement')
        """
        pass
    
    @abstractmethod
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get the configuration template for this module.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the provided configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation - can be overridden
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this module.
        
        Returns:
            Dictionary containing module information
        """
        return {
            'name': self.name,
            'type': self.get_module_type(),
            'initialized': self.is_initialized,
            'config': self.config
        }


class ThresholdAdjustmentModule(BaseEnhancementModule):
    """
    Base class for threshold adjustment modules.
    
    These modules adjust detection thresholds based on image characteristics.
    """
    
    def get_module_type(self) -> str:
        """Get module type."""
        return 'threshold_adjustment'
    
    @abstractmethod
    def compute_adaptive_threshold(
        self,
        img1_embed: torch.Tensor,
        img2_embed: torch.Tensor,
        base_threshold: float,
        **kwargs
    ) -> float:
        """
        Compute adaptive threshold based on image features.
        
        Args:
            img1_embed: Feature embeddings of first image [C, H, W]
            img2_embed: Feature embeddings of second image [C, H, W]
            base_threshold: Base threshold value
            **kwargs: Additional parameters
            
        Returns:
            Adaptive threshold value
        """
        pass


class FeatureEnhancementModule(BaseEnhancementModule):
    """
    Base class for feature enhancement modules.
    
    These modules enhance or modify feature representations.
    """
    
    def get_module_type(self) -> str:
        """Get module type."""
        return 'feature_enhancement'
    
    @abstractmethod
    def enhance_features(
        self,
        features: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Enhance feature representations.
        
        Args:
            features: Input feature tensor
            **kwargs: Additional parameters
            
        Returns:
            Enhanced feature tensor
        """
        pass


class PostProcessingModule(BaseEnhancementModule):
    """
    Base class for post-processing modules.
    
    These modules refine detection results after initial processing.
    """
    
    def get_module_type(self) -> str:
        """Get module type."""
        return 'post_processing'
    
    @abstractmethod
    def post_process(
        self,
        masks: np.ndarray,
        confidences: np.ndarray,
        **kwargs
    ) -> tuple:
        """
        Post-process detection results.
        
        Args:
            masks: Detection masks
            confidences: Confidence scores
            **kwargs: Additional parameters
            
        Returns:
            Tuple of processed masks and confidences
        """
        pass
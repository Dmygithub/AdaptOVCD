"""
OVCD Enhancement Modules

This package contains plug-and-play enhancement modules for OVCD (Open-Vocabulary Change Detection).
Each module can be easily integrated into the OVCD pipeline to improve performance.

Available Modules:
- ACT (Adaptive Change Thresholding): Dynamic threshold optimization using global + edge-guided Otsu
- ARA (Adaptive Radiometric Alignment): Cross-temporal radiometric harmonization for multi-temporal images
- ACF (Adaptive Confidence Filtering): Lightweight confidence-based filtering with simple parameter control

Features:
- Modular design with standardized interfaces
- Easy configuration management
- Plug-and-play integration
- Automatic module registration and discovery
"""

from .base import (
    BaseEnhancementModule, 
    ThresholdAdjustmentModule, 
    FeatureEnhancementModule, 
    PostProcessingModule
)
from .registry import ModuleRegistry, register_module, get_module, list_available_modules, get_module_info
from .adaptive_change_thresholding import AdaptiveChangeThresholdingModule  # ACT
from .adaptive_illumination_alignment import AdaptiveRadiometricAlignmentModule  # ARA
from .confidence_filtering import ConfidenceFilteringModule

# Global module registry
registry = ModuleRegistry()

# Register built-in modules
register_module('adaptive_change_thresholding', AdaptiveChangeThresholdingModule)  # ACT
register_module('adaptive_radiometric_alignment', AdaptiveRadiometricAlignmentModule)  # ARA
register_module('confidence_filtering', ConfidenceFilteringModule)  # ACF

__all__ = [
    # Base classes
    'BaseEnhancementModule',
    'ThresholdAdjustmentModule',
    'FeatureEnhancementModule', 
    'PostProcessingModule',
    
    # Registry system
    'ModuleRegistry', 
    'register_module',
    'get_module',
    'list_available_modules',
    'get_module_info',
    'registry',
    
    # Built-in modules
    'AdaptiveChangeThresholdingModule',  # ACT
    'AdaptiveRadiometricAlignmentModule',  # ARA
    'ConfidenceFilteringModule',  # ACF
]
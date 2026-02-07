"""
Core module for the modular change detection framework.

This module provides the foundational components for building change detection pipelines
through a modular architecture with segmentation, comparison, and identification stages.
"""

# Import factory functions
from .factory import (
    create_pipeline,
    create_segmentor,
    create_comparator,
    create_identifier,
    get_supported_components,
    validate_model_combination,
    get_model_name_from_config
)

# Import base classes
from .segmentation.base import BaseSegmentor
from .comparison.base import BaseComparator
from .identification.base import BaseIdentifier

# Import concrete implementations
from .segmentation import SAMSegmentor
from .comparison import DINOComparator, DINOv2Comparator, DINOv3Comparator
from .identification import DGTRSIdentifier

# Import unified pipeline
from .pipeline.unified import UnifiedPipeline

__all__ = [
    # Factory functions
    'create_pipeline',
    'create_segmentor', 
    'create_comparator',
    'create_identifier',
    'get_supported_components',
    'validate_model_combination',
    'get_model_name_from_config',
    
    # Base classes
    'BaseSegmentor',
    'BaseComparator', 
    'BaseIdentifier',
    
    # Concrete implementations
    'SAMSegmentor',
    'DINOComparator',
    'DINOv2Comparator',
    'DINOv3Comparator',
    'DGTRSIdentifier',
    
    # Unified pipeline
    'UnifiedPipeline'
]
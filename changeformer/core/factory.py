"""
Component factory for creating segmentation, comparison, and identification modules.

This module provides factory functions to create components based on configuration.
"""

from typing import Dict, Any, Tuple

from .segmentation import BaseSegmentor, SAMSegmentor
from .comparison import BaseComparator, DINOComparator, DINOv2Comparator, DINOv3Comparator
from .comparison.dino_config import get_variant_spec, list_variants
from .identification import BaseIdentifier, DGTRSIdentifier
from .pipeline.unified import UnifiedPipeline


def create_segmentor(config: Dict[str, Any]) -> BaseSegmentor:
    """
    Create a segmentor based on configuration.
    
    Args:
        config: Segmentor configuration dictionary
        
    Returns:
        Initialized segmentor instance
        
    Raises:
        ValueError: If segmentor type is not supported
    """
    segmentor_type = config.get('type', '').upper()
    
    if segmentor_type == 'SAM':
        return SAMSegmentor()
    else:
        raise ValueError(f"Unsupported segmentor type: {segmentor_type}")


def create_comparator(config: Dict[str, Any]) -> BaseComparator:
    """
    Create a comparator based on configuration.
    
    Supports two configuration modes:
    1. New mode (recommended): Use 'variant' to auto-configure feature_dim and patch_size
       Example: {variant: 'dinov3_vitl16', change_confidence_threshold: 130}
    
    2. Legacy mode: Use 'type' explicitly (DINOv2, DINOv3)
       Example: {type: 'DINOv3', variant: 'dinov3_vitl16', weights_path: '...'}
    
    Args:
        config: Comparator configuration dictionary
        
    Returns:
        Initialized comparator instance
        
    Raises:
        ValueError: If comparator type/variant is not supported
    """
    # New mode: Auto-configure from variant
    if 'variant' in config and 'type' not in config:
        variant = config['variant']
        variant_spec = get_variant_spec(variant)
        
        # Directly modify the config dict (not a copy) so changes propagate
        config['type'] = variant_spec['model_type']
        config['feature_dim'] = variant_spec['feature_dim']
        config['patch_size'] = variant_spec['patch_size']
        
        # Set weights_path for DINOv3 if not specified
        if variant_spec['model_type'] == 'DINOv3' and 'weights_path' not in config:
            config['weights_path'] = variant_spec['weights_path']
        
        print(f"Auto-configured comparator: {variant} -> {variant_spec['description']}")
    
    # Create comparator based on type
    comparator_type = config.get('type', '').upper()
    
    if comparator_type == 'DINO':
        return DINOComparator()
    elif comparator_type == 'DINOV2':
        return DINOv2Comparator()
    elif comparator_type == 'DINOV3':
        return DINOv3Comparator()
    else:
        raise ValueError(f"Unsupported comparator type: {comparator_type}")


def create_identifier(config: Dict[str, Any]) -> BaseIdentifier:
    """
    Create an identifier based on configuration.
    
    Args:
        config: Identifier configuration dictionary
        
    Returns:
        Initialized identifier instance
        
    Raises:
        ValueError: If identifier type is not supported
    """
    identifier_type = config.get('type', '').upper()
    
    if identifier_type == 'DGTRS-CLIP' or identifier_type == 'DGTRS':
        return DGTRSIdentifier()
    else:
        raise ValueError(f"Unsupported identifier type: {identifier_type}")


def create_pipeline(config: Dict[str, Any]) -> UnifiedPipeline:
    """
    Create a complete pipeline based on configuration.
    
    Args:
        config: Full pipeline configuration dictionary
        
    Returns:
        Initialized UnifiedPipeline instance
        
    Raises:
        ValueError: If any component type is not supported
        KeyError: If required configuration keys are missing
    """
    # Validate required configuration sections
    required_sections = ['segmentor', 'comparator', 'identifier']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")
    
    # Create individual components
    segmentor = create_segmentor(config['segmentor'])
    comparator = create_comparator(config['comparator'])
    identifier = create_identifier(config['identifier'])
    
    # Create and return unified pipeline
    pipeline = UnifiedPipeline(
        segmentor=segmentor,
        comparator=comparator,
        identifier=identifier,
        config=config
    )
    
    return pipeline


def get_supported_components() -> Dict[str, list]:
    """
    Get lists of supported component types.
    
    Returns:
        Dictionary with lists of supported types for each component category
    """
    return {
        'segmentors': ['SAM'],
        'comparators': ['DINO', 'DINOv2', 'DINOv3'],
        'identifiers': ['DGTRS-CLIP']
    }


def validate_model_combination(segmentor_type: str, comparator_type: str, identifier_type: str) -> bool:
    """
    Validate if a model combination is supported.
    
    Args:
        segmentor_type: Type of segmentor
        comparator_type: Type of comparator  
        identifier_type: Type of identifier
        
    Returns:
        True if combination is valid, False otherwise
    """
    supported = get_supported_components()
    
    return (
        segmentor_type.upper() in [s.upper() for s in supported['segmentors']] and
        comparator_type.upper() in [c.upper() for c in supported['comparators']] and
        identifier_type.upper() in [i.upper() for i in supported['identifiers']]
    )


def get_model_name_from_config(config: Dict[str, Any]) -> str:
    """
    Generate a model name from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Generated model name string
    """
    segmentor_type = config.get('segmentor', {}).get('type', 'unknown').lower()
    comparator_type = config.get('comparator', {}).get('type', 'unknown').lower()
    identifier_type = config.get('identifier', {}).get('type', 'unknown').lower()
    
    # Clean up identifier type name
    if identifier_type == 'dgtrs-clip':
        identifier_type = 'dgtrs'
    
    return f"{segmentor_type}_{comparator_type}_{identifier_type}"
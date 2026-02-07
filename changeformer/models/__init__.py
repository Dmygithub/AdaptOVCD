"""
Models module for ChangeFormer.

This module provides the unified build_pipeline function using the new modular architecture.
"""

from changeformer.core.factory import create_pipeline, get_supported_components, validate_model_combination


def build_pipeline(config):
    """
    Build the appropriate pipeline based on the configuration using the new modular architecture.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        UnifiedPipeline instance
        
    Raises:
        ValueError: If the model combination is not supported
        KeyError: If required configuration keys are missing
    """
    try:
        # Validate configuration structure
        required_sections = ['segmentor', 'comparator', 'identifier']
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Missing required configuration section: {section}")
            # For comparator, allow 'variant' instead of 'type' (new mode)
            if section == 'comparator':
                if 'type' not in config[section] and 'variant' not in config[section]:
                    raise KeyError(f"Missing 'type' or 'variant' in comparator configuration")
            else:
                if 'type' not in config[section]:
                    raise KeyError(f"Missing 'type' in {section} configuration")
        
        # Extract component types
        segmentor_type = config['segmentor']['type']
        # For comparator, infer type from variant if using new mode
        if 'variant' in config['comparator'] and 'type' not in config['comparator']:
            from changeformer.core.comparison.dino_config import get_variant_spec
            variant_spec = get_variant_spec(config['comparator']['variant'])
            comparator_type = variant_spec['model_type']
        else:
            comparator_type = config['comparator']['type']
        identifier_type = config['identifier']['type']
        
        # Validate model combination
        if not validate_model_combination(segmentor_type, comparator_type, identifier_type):
            supported = get_supported_components()
            raise ValueError(
                f"Unsupported model combination: {segmentor_type}+{comparator_type}+{identifier_type}\n"
                f"Supported combinations:\n"
                f"  Segmentors: {supported['segmentors']}\n"
                f"  Comparators: {supported['comparators']}\n"
                f"  Identifiers: {supported['identifiers']}"
            )
        
        # Create unified pipeline using factory
        pipeline = create_pipeline(config)
        
        print(f"Created unified pipeline: {segmentor_type}+{comparator_type}+{identifier_type}")
        return pipeline
        
    except Exception as e:
        print(f"Error building pipeline: {e}")
        raise


__all__ = [
    "build_pipeline"
] 
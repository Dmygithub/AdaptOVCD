"""
Configuration management for ChangeFormer.

This module handles loading and parsing of model and dataset configurations.
"""

import os
import yaml
import torch
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Auto-detect device if set to "auto"
    if config.get('device') == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config


def get_available_models() -> List[str]:
    """
    Get list of available model configurations.
    
    Returns:
        List of available model names
    """
    # Define supported model combinations
    supported_models = [
        'sam_dinov2_dgtrs',
        'sam2_dinov2_dgtrs', 
        'sam_dinov2_segearth',
        'sam_dino_segearth',
        'sam_dino_dgtrs',
        'sam2_dino_dgtrs',
        'sam2_dinov2_segearth',
        'sam2_dino_segearth',
        'sam_dinov2_dgtrs_modular'
    ]
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'models')
    if not os.path.exists(models_dir):
        return supported_models
    
    available_models = []
    for file in os.listdir(models_dir):
        if file.endswith('.yaml'):
            model_name = file[:-5]  # Remove .yaml extension
            available_models.append(model_name)
    
    return sorted(available_models)


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model configuration by name.
    
    Args:
        model_name: Name of the model (without .yaml extension)
        
    Returns:
        Model configuration dictionary
        
    Raises:
        FileNotFoundError: If model configuration doesn't exist
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', '..', 'configs', 'models', f'{model_name}.yaml')
    config_path = os.path.normpath(config_path)
    
    return load_config(config_path)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files are missing
    """
    required_keys = ['segmentor', 'comparator', 'identifier']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate segmentor config
    segmentor_config = config['segmentor']
    if 'type' not in segmentor_config:
        raise ValueError("Segmentor configuration missing 'type' field")
    
    # Check for segmentor-specific requirements
    _validate_component_config(segmentor_config, 'segmentor')
    
    # Validate comparator config  
    comparator_config = config['comparator']
    if 'type' not in comparator_config:
        raise ValueError("Comparator configuration missing 'type' field")
    
    # Check for comparator-specific requirements
    _validate_component_config(comparator_config, 'comparator')
    
    # Validate identifier config
    identifier_config = config['identifier']
    if 'type' not in identifier_config:
        raise ValueError("Identifier configuration missing 'type' field")
    
    # Check for identifier-specific requirements
    _validate_component_config(identifier_config, 'identifier')
    
    # Validate device configuration
    if 'device' in config:
        device = config['device']
        if device not in ['cpu', 'cuda', 'auto'] and not device.startswith('cuda:'):
            raise ValueError(f"Invalid device configuration: {device}")


def _validate_component_config(component_config: Dict[str, Any], component_type: str) -> None:
    """
    Validate individual component configuration.
    
    Args:
        component_config: Component configuration dictionary
        component_type: Type of component ('segmentor', 'comparator', 'identifier')
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files are missing
    """
    # Only validate critical file paths that are explicitly specified
    critical_paths = ['model_path', 'checkpoint']
    for path_key in critical_paths:
        if path_key in component_config:
            path_value = component_config[path_key]
            if path_value and not os.path.exists(path_value):
                # Only raise error for explicitly specified non-existent paths
                raise FileNotFoundError(
                    f"{component_type.capitalize()} {path_key} not found: {path_value}"
                )
    
    # Lightweight parameter validation - only check critical parameters
    component_name = component_config['type'].upper()
    
    if component_type == 'segmentor' and component_name in ['SAM', 'SAM2']:
        # Set default model type if not specified (don't validate)
        if 'model_type' not in component_config:
            component_config['model_type'] = 'default'
    
    elif component_type == 'comparator' and component_name in ['DINO', 'DINOV2']:
        # Only validate patch_size if it's clearly invalid
        if 'patch_size' in component_config:
            patch_size = component_config['patch_size']
            if not isinstance(patch_size, (int, float)) or patch_size <= 0:
                raise ValueError(f"Invalid patch_size for {component_name}: {patch_size}")
    
    # Skip other validations to improve performance


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged 


def apply_overrides(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Apply command-line style overrides to a loaded configuration.

    This utility consolidates common override behaviors used by demo/evaluate,
    keeping the entry scripts concise and consistent.

    Args:
      config: Configuration dictionary loaded from YAML.
      args: Namespace-like object (e.g., argparse.Namespace) that may contain
        optional override attributes such as device, points_per_side, etc.

    Returns:
      The updated configuration dictionary (same object, for convenience).
    """
    # Device override
    device_override = getattr(args, "device", None)
    if device_override:
        config["device"] = device_override

    # Ensure component keys exist
    if "segmentor" not in config:
        config["segmentor"] = {}
    if "params" not in config["segmentor"]:
        config["segmentor"]["params"] = {}
    if "comparator" not in config:
        config["comparator"] = {}
    if "identifier" not in config:
        config["identifier"] = {}

    # Segmentor params overrides
    seg_params = config["segmentor"]["params"]
    if getattr(args, "points_per_side", None) is not None:
        seg_params["points_per_side"] = args.points_per_side
    if getattr(args, "pred_iou_thresh", None) is not None:
        seg_params["pred_iou_thresh"] = args.pred_iou_thresh
    if getattr(args, "stability_score_thresh", None) is not None:
        seg_params["stability_score_thresh"] = args.stability_score_thresh
    if getattr(args, "stability_score_offset", None) is not None:
        seg_params["stability_score_offset"] = args.stability_score_offset
    if getattr(args, "mask_threshold", None) is not None:
        seg_params["mask_threshold"] = args.mask_threshold

    # Comparator overrides
    if getattr(args, "change_confidence_threshold", None) is not None:
        config["comparator"]["change_confidence_threshold"] = args.change_confidence_threshold

    # Identifier overrides
    if getattr(args, "confidence_threshold", None) is not None:
        config["identifier"]["confidence_threshold"] = args.confidence_threshold

    return config
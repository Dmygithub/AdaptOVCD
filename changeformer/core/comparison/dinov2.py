"""
DINOv2 comparator implementation.

This module implements the DINOv2-based feature comparison for change detection.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

from .base import BaseComparator

# Calculate path to DynamicEarth directory for dynamic_earth imports
current_file = os.path.abspath(__file__)
ovcd_core_dir = os.path.dirname(os.path.dirname(current_file))  # changeformer/core/
ovcd_changeformer_dir = os.path.dirname(ovcd_core_dir)  # changeformer/
ovcd_root = os.path.dirname(ovcd_changeformer_dir)  # OVCD/
parent_dir = os.path.dirname(ovcd_root)  # parent of OVCD

# Try to find DynamicEarth directory
possible_names = ['DynamicEarth', 'reproduction']
reproduction_path = None

for name in possible_names:
    test_path = os.path.join(parent_dir, name)
    if os.path.exists(test_path):
        dynamic_earth_path = os.path.join(test_path, 'dynamic_earth')
        if os.path.exists(dynamic_earth_path):
            reproduction_path = test_path
            break

if reproduction_path and reproduction_path not in sys.path:
    sys.path.insert(0, reproduction_path)

try:
    # Use local model_utils instead of dynamic_earth
    from ..utils.model_utils import get_model_and_processor
    # Import from our modified matching module
    from .matching import bitemporal_feature_matching as bitemporal_match
    DINOV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DINOv2 components not available: {e}")
    DINOV2_AVAILABLE = False
    get_model_and_processor = None
    bitemporal_match = None


class DINOv2Comparator(BaseComparator):
    """
    DINOv2-based feature comparison for change detection.
    
    Uses DINOv2 features to compare masks between two images
    and identify actual changes.
    """
    
    def __init__(self):
        """Initialize DINOv2 comparator."""
        super().__init__()
        self.change_threshold = None
        self.enhancement_modules = {}
    
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup DINOv2/DINOv3 comparator with configuration.
        
        Args:
            config: DINOv2/DINOv3 configuration dictionary
            device: Device to run on
        """
        if not DINOV2_AVAILABLE:
            raise ImportError("DINOv2 components are not available")
        
        self.validate_config(config)
        self.config = config
        self.device = device
        
        # Load DINOv2 or DINOv3 model
        model_type = config.get('type', 'DINOv2')
        model_config = {
            'variant': config.get('variant', 'dinov2_vitb14'),
            'weights_path': config.get('weights_path', None)
        }
        self.model, self.processor = get_model_and_processor(model_type, device, model_config=model_config)
        
        # Configuration parameters
        self.feature_dim = config.get('feature_dim', 768)
        self.patch_size = config.get('patch_size', 14)
        self.change_threshold = config.get('change_confidence_threshold', 145)
        
        # Initialize enhancement modules
        self._load_enhancement_modules(config)
        
        self._is_setup = True
        print(f"{model_type} comparator initialized on {device}")
    
    def compare(self, img1: np.ndarray, img2: np.ndarray, masks: List[np.ndarray], 
                img1_mask_num: int) -> Tuple[List[np.ndarray], int]:
        """
        Compare features between two images to identify change masks.
        
        Args:
            img1: First input image
            img2: Second input image
            masks: List of masks to compare
            img1_mask_num: Number of masks from first image
            
        Returns:
            Tuple of (change_masks, img1_mask_num)
        """
        if not self._is_setup:
            raise RuntimeError("DINOv2 comparator not setup. Call setup() first.")
        
        if len(masks) == 0:
            return [], img1_mask_num
        
        # Convert masks to numpy array if needed
        if isinstance(masks, list):
            masks = np.array(masks)
        
        # Model configuration for bitemporal matching
        model_config = {
            'model_type': self.config.get('type', 'DINOv2'),
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size
        }
        
        # Perform bitemporal matching using DINOv2 features
        change_masks, updated_img1_mask_num = bitemporal_match(
            img1, img2, masks,
            self.model, self.processor,
            img1_mask_num,
            change_confidence_threshold=self.change_threshold,
            device=self.device,
            model_config=model_config,
            enhancement_modules=self.enhancement_modules
        )
        
        return change_masks, updated_img1_mask_num
    
    def _load_enhancement_modules(self, config: Dict[str, Any]) -> None:
        """
        Load enhancement modules from configuration.
        
        Args:
            config: Configuration dictionary
        """
        try:
            # Import Module package
            import sys
            import os
            
            # Add OVCD root to path if not already there
            ovcd_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if ovcd_root not in sys.path:
                sys.path.insert(0, ovcd_root)
            
            from Module import get_module
            
            # Load adaptive change thresholding module (new module name)
            if config.get('enable_adaptive_change_thresholding', False):
                act_config = config.get('adaptive_change_thresholding_config', {})
                act_module = get_module('adaptive_change_thresholding', act_config)
                self.enhancement_modules['adaptive_change_thresholding'] = act_module
                print(f"Loaded enhancement module: adaptive_change_thresholding")
            
            # Load other enhancement modules if specified
            enhancement_config = config.get('enhancement_modules', {})
            if enhancement_config is not None:
                for module_name, module_config in enhancement_config.items():
                    if module_config.get('enabled', True):
                        module = get_module(module_name, module_config.get('config', {}))
                        self.enhancement_modules[module_name] = module
                        print(f"Loaded enhancement module: {module_name}")
                    
        except ImportError as e:
            print(f"Warning: Could not load Module package: {e}")
        except Exception as e:
            print(f"Warning: Failed to load enhancement modules: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate DINOv2 configuration."""
        super().validate_config(config)
        
        # DINOv2 specific validation
        patch_size = config.get('patch_size', 14)
        if patch_size not in [14, 16]:
            print(f"Warning: DINOv2 typically uses patch_size 14 or 16, got {patch_size}")
        
        feature_dim = config.get('feature_dim', 768)
        valid_dims = [768, 1024]
        if feature_dim not in valid_dims:
            print(f"Warning: DINOv2 typically uses feature_dim in {valid_dims}, got {feature_dim}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get DINOv2 configuration template."""
        return {
            "type": "DINOv2",
            "feature_dim": 768,
            "patch_size": 14,
            "change_confidence_threshold": 145
        }
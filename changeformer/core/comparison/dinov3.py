"""
DINOv3 comparator implementation.

This module implements the DINOv3-based feature comparison for change detection.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

from .base import BaseComparator

# Add DINOv3 to path
current_file = os.path.abspath(__file__)
ovcd_core_dir = os.path.dirname(os.path.dirname(current_file))
ovcd_changeformer_dir = os.path.dirname(ovcd_core_dir)
ovcd_root = os.path.dirname(ovcd_changeformer_dir)
dinov3_path = os.path.join(ovcd_root, 'third_party', 'dinov3')

if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

try:
    from ..utils.model_utils import get_model_and_processor
    from .matching import bitemporal_feature_matching as bitemporal_match
    from dinov3.hub.backbones import dinov3_vitl16, dinov3_vith16plus, dinov3_vitb16
    DINOV3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DINOv3 components not available: {e}")
    DINOV3_AVAILABLE = False
    get_model_and_processor = None
    bitemporal_match = None


class DINOv3Comparator(BaseComparator):
    """
    DINOv3-based feature comparison for change detection.
    
    Uses DINOv3 features to compare masks between two images
    and identify actual changes.
    """
    
    def __init__(self):
        """Initialize DINOv3 comparator."""
        super().__init__()
        self.change_threshold = None
        self.enhancement_modules = {}
    
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup DINOv3 comparator with configuration.
        
        Args:
            config: DINOv3 configuration dictionary
            device: Device to run on
        """
        if not DINOV3_AVAILABLE:
            raise ImportError("DINOv3 components are not available")
        
        self.validate_config(config)
        self.config = config
        self.device = device
        
        # Load DINOv3 model from local weights
        variant = config.get('variant', 'dinov3_vitl16')
        weights_path = config.get('weights_path', None)
        
        # Variant function mapping
        variant_map = {
            'dinov3_vitl16': dinov3_vitl16,
            'dinov3_vith16': dinov3_vith16plus,
            'dinov3_vitb16': dinov3_vitb16,
        }
        
        model_fn = variant_map.get(variant)
        if model_fn is None:
            raise ValueError(f"Unsupported DINOv3 variant: {variant}")
        
        # Load from local weights
        if weights_path and os.path.exists(weights_path):
            # Use absolute path for DINOv3 official loader
            abs_weights_path = os.path.abspath(weights_path)
            self.model = model_fn(pretrained=True, weights=abs_weights_path).to(device)
            print(f"DINOv3 {variant} loaded from {abs_weights_path}")
        else:
            raise FileNotFoundError(f"DINOv3 weights not found at {weights_path}")
        
        # Use DINOv2 processor (compatible)
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', 
            do_resize=False, do_center_crop=False)
        
        # Configuration parameters
        self.feature_dim = config.get('feature_dim', 1024)
        self.patch_size = config.get('patch_size', 16)
        self.change_threshold = config.get('change_confidence_threshold', 140)
        
        # Initialize enhancement modules
        self._load_enhancement_modules(config)
        
        self._is_setup = True
        print(f"DINOv3 comparator initialized on {device}")
    
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
            raise RuntimeError("DINOv3 comparator not setup. Call setup() first.")
        
        if len(masks) == 0:
            return [], img1_mask_num
        
        # Convert masks to numpy array if needed
        if isinstance(masks, list):
            masks = np.array(masks)
        
        # Model configuration for bitemporal matching
        model_config = {
            'model_type': 'DINOv3',
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size
        }
        
        # Perform bitemporal matching using DINOv3 features
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
            import sys
            import os
            
            ovcd_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if ovcd_root not in sys.path:
                sys.path.insert(0, ovcd_root)
            
            from Module import get_module
            
            # Load adaptive change thresholding module
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
        """Validate DINOv3 configuration."""
        super().validate_config(config)
        
        # Check weights_path
        weights_path = config.get('weights_path', None)
        if not weights_path:
            raise ValueError("DINOv3 requires 'weights_path' in config")
        
        # DINOv3 specific validation
        patch_size = config.get('patch_size', 16)
        if patch_size != 16:
            print(f"Warning: DINOv3 uses patch_size 16, got {patch_size}")
        
        feature_dim = config.get('feature_dim', 1024)
        valid_dims = [768, 1024, 1280, 1536]
        if feature_dim not in valid_dims:
            print(f"Warning: DINOv3 typically uses feature_dim in {valid_dims}, got {feature_dim}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get DINOv3 configuration template."""
        return {
            "type": "DINOv3",
            "variant": "dinov3_vitl16",
            "weights_path": "models/dinov3/dinov3_vitl16_distilled.pth",
            "feature_dim": 1024,
            "patch_size": 16,
            "change_confidence_threshold": 140
        }


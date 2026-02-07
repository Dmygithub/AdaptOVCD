"""
DINO comparator implementation.

This module implements the DINO-based feature comparison for change detection.
Based on the original sam_dino_segearth_pipeline.py implementation.
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
    DINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DINO components not available: {e}")
    DINO_AVAILABLE = False
    get_model_and_processor = None
    bitemporal_match = None


class DINOComparator(BaseComparator):
    """
    DINO-based feature comparison for change detection.
    
    Uses DINO features to compare masks between two images
    and identify actual changes.
    """
    
    def __init__(self):
        """Initialize DINO comparator."""
        super().__init__()
        self.change_threshold = None
    
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup DINO comparator with configuration.
        
        Args:
            config: DINO configuration dictionary
            device: Device to run on
        """
        if not DINO_AVAILABLE:
            raise ImportError("DINO components are not available")
        
        self.validate_config(config)
        self.config = config
        self.device = device
        
        # Load DINO model
        model_type = config.get('type', 'DINO')
        self.model, self.processor = get_model_and_processor(model_type, device)
        
        # Configuration parameters
        self.feature_dim = config.get('feature_dim', 768)
        self.patch_size = config.get('patch_size', 16)
        self.change_threshold = config.get('change_confidence_threshold', 145)
        
        self._is_setup = True
        print(f"DINO comparator initialized on {device}")
    
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
            raise RuntimeError("DINO comparator not setup. Call setup() first.")
        
        if len(masks) == 0:
            return [], img1_mask_num
        
        # Convert masks to numpy array if needed
        if isinstance(masks, list):
            masks = np.array(masks)
        
        # Model configuration for bitemporal matching
        model_config = {
            'model_type': 'DINO',
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size
        }
        
        # Perform bitemporal matching using DINO features
        change_masks, updated_img1_mask_num = bitemporal_match(
            img1, img2, masks,
            self.model, self.processor,
            img1_mask_num,
            change_confidence_threshold=self.change_threshold,
            device=self.device,
            model_config=model_config
        )
        
        return change_masks, updated_img1_mask_num
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate DINO configuration."""
        super().validate_config(config)
        
        # DINO specific validation
        patch_size = config.get('patch_size', 16)
        if patch_size != 16:
            print(f"Warning: DINO typically uses patch_size=16, got {patch_size}")
        
        feature_dim = config.get('feature_dim', 768)
        if feature_dim != 768:
            print(f"Warning: DINO typically uses feature_dim=768, got {feature_dim}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get DINO configuration template."""
        return {
            "type": "DINO",
            "feature_dim": 768,
            "patch_size": 16,
            "change_confidence_threshold": 145
        }
"""
SAM (Segment Anything Model) segmentor implementation.

This module implements the SAM segmentor for automatic mask generation.
"""

import os
import sys
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List

from .base import BaseSegmentor

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
    from segment_anything.utils.amg import rle_to_mask
    from segment_anything import sam_model_registry, sam_hq_model_registry
    from .sam_maskgen import MaskProposal
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM components not available: {e}")
    SAM_AVAILABLE = False
    rle_to_mask = None
    sam_model_registry = None
    sam_hq_model_registry = None
    MaskProposal = None


class SAMSegmentor(BaseSegmentor):
    """
    SAM-based automatic mask generation segmentor.
    
    Uses the Segment Anything Model to generate high-quality masks
    for change detection between two images.
    """
    
    def __init__(self):
        """Initialize SAM segmentor."""
        super().__init__()
        self.mp = None
        self.sam_model = None
    
    def setup(self, config: Dict[str, Any], device: str = 'cuda') -> None:
        """
        Setup SAM segmentor with configuration.
        
        Args:
            config: SAM configuration dictionary
            device: Device to run on
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM components are not available")
        
        self.validate_config(config)
        self.config = config
        self.device = device
        
        # SAM model configuration
        sam_version = config.get('sam_version', 'vit_h')
        checkpoint = config.get('checkpoint', 'models/sam_vit_h_4b8939.pth')
        use_sam_hq = config.get('use_sam_hq', False)
        sam_hq_checkpoint = config.get('sam_hq_checkpoint', None)
        
        # Initialize SAM model
        model_registry = sam_hq_model_registry if use_sam_hq else sam_model_registry
        checkpoint_path = sam_hq_checkpoint if use_sam_hq else checkpoint
        
        # Print SAM type being used
        if use_sam_hq:
            print(f"Loading SAM-HQ model: {checkpoint_path}")
        else:
            print(f"Loading Standard SAM model: {checkpoint_path}")
        
        self.sam_model = model_registry[sam_version](checkpoint=checkpoint_path).to(device)
        
        # Setup MaskProposal
        self.mp = MaskProposal()
        
        # Get parameters from config
        params = config.get('params', {})
        self.mp.make_mask_generator(
            model=self.sam_model,
            points_per_side=params.get('points_per_side', 32),
            points_per_batch=params.get('points_per_batch', 64),
            pred_iou_thresh=params.get('pred_iou_thresh', 0.5),
            stability_score_thresh=params.get('stability_score_thresh', 0.95),
            stability_score_offset=params.get('stability_score_offset', 0.9),
            box_nms_thresh=params.get('box_nms_thresh', 0.7),
            min_mask_region_area=params.get('min_mask_region_area', 0),
            mask_threshold=params.get('mask_threshold', 0.0)
        )
        
        # Set hyperparameters for mask proposal
        area_thresh_val = params.get('area_thresh', 0.8)
        min_mask_region_area_val = params.get('min_mask_region_area', 0)
        
        self.mp.set_hyperparameters(
            match_hist=params.get('match_hist', False),
            area_thresh=area_thresh_val
        )
        
        # Load preprocessing module if enabled
        self.preprocessing_module = self._load_preprocessing_module(config)
        
        self._is_setup = True
        print(f"SAM segmentor initialized with {sam_version} on {device}")
    
    def _load_preprocessing_module(self, config: Dict[str, Any]):
        """Load radiometric preprocessing module if enabled."""
        # Check for adaptive radiometric alignment first (new module name)
        if config.get('enable_adaptive_radiometric_alignment', False):
            try:
                from Module import get_module
                
                module_config = config.get('adaptive_radiometric_alignment_config', {})
                module = get_module('adaptive_radiometric_alignment', module_config)
                
                print(f"Adaptive radiometric alignment initialized: {module_config.get('method', 'simple_histogram')}")
                print(f"Loaded adaptive radiometric alignment module")
                return module
            except Exception as e:
                print(f"Warning: Failed to load adaptive radiometric alignment module: {e}")
        
        # Check for legacy illumination alignment (old module name for backward compatibility)
        # Note: adaptive_illumination_alignment has been renamed to adaptive_radiometric_alignment
        if config.get('enable_adaptive_illumination_alignment', False):
            try:
                from Module import get_module
                
                module_config = config.get('adaptive_illumination_alignment_config', {})
                module = get_module('adaptive_radiometric_alignment', module_config)
                
                print(f"Adaptive radiometric alignment initialized (legacy config): {module_config.get('method', 'simple_histogram')}")
                print(f"Loaded adaptive radiometric alignment module")
                return module
            except Exception as e:
                print(f"Warning: Failed to load adaptive radiometric alignment module: {e}")
        
        return None
    
    def segment(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """Generate segmentation masks using SAM with optional radiometric preprocessing.
        
        Applies radiometric preprocessing if enabled to align radiometric conditions
        between image pairs, which is crucial for accurate change detection.
        
        Args:
            img1: First input image (H, W, C).
            img2: Second input image (H, W, C).
            
        Returns:
            Tuple of (masks, img1_mask_num).
            
        Raises:
            RuntimeError: If segmentor is not properly setup.
        """
        if not self._is_setup:
            raise RuntimeError("SAM segmentor not setup. Call setup() first.")
        
        # Apply preprocessing if available
        if self.preprocessing_module is not None:
            processed_img1, processed_img2 = self.preprocessing_module.process(img1, img2)
        else:
            processed_img1, processed_img2 = img1, img2
        
        # Generate masks using MaskProposal
        mask_data, img1_mask_num = self.mp.forward(processed_img1, processed_img2)
        
        # Convert RLE masks to binary numpy arrays
        if len(mask_data['rles']) == 0:
            return [], 0
        
        # Convert RLE masks to binary masks
        masks = [rle_to_mask(rle).astype(bool) for rle in mask_data['rles']]
        
        return masks, img1_mask_num
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate SAM configuration."""
        super().validate_config(config)
        
        # Check SAM version
        sam_version = config.get('sam_version', 'vit_h')
        valid_versions = ['vit_b', 'vit_l', 'vit_h']
        if sam_version not in valid_versions:
            raise ValueError(f"sam_version must be one of {valid_versions}, got {sam_version}")
        
        # Check checkpoint exists
        checkpoint = config.get('checkpoint')
        if checkpoint and not os.path.exists(checkpoint):
            print(f"Warning: SAM checkpoint not found: {checkpoint}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get SAM configuration template."""
        return {
            "type": "SAM",
            "sam_version": "vit_h",
            "checkpoint": "models/sam_vit_h_4b8939.pth",
            "use_sam_hq": False,
            "sam_hq_checkpoint": None,
            "params": {
                "points_per_side": 32,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.95,
                "stability_score_offset": 0.9,
                "box_nms_thresh": 0.7,
                "min_mask_region_area": 0,
                "match_hist": False,
                "area_thresh": 0.8
            }
        }
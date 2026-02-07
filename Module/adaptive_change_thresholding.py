"""
ACT (Adaptive Change Thresholding) Module for OVCD Change Detection.

This module implements dynamic threshold optimization using global and edge-guided 
Otsu thresholding on DINOv2 feature difference maps. ACT adaptively adjusts detection 
thresholds based on both global statistics and edge-region characteristics.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from skimage.filters import threshold_otsu

from .base import ThresholdAdjustmentModule

# Constants
DEFAULT_GLOBAL_WEIGHT = 0.6
DEFAULT_EDGE_WEIGHT = 0.4
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_EDGE_DILATION_KERNEL = 3
DEFAULT_MIN_EDGE_PIXELS = 100
DEFAULT_MIN_ANGLE = 100
DEFAULT_MAX_ANGLE = 180
EDGE_DETECTION_METHODS = ('canny', 'sobel')
NUMERICAL_EPSILON = 1e-8


class AdaptiveChangeThresholdingModule(ThresholdAdjustmentModule):
    """
    ACT (Adaptive Change Thresholding) module for dynamic threshold optimization.
    
    Uses global + edge-guided Otsu thresholding on DINOv2 feature difference maps 
    to compute optimal detection thresholds for open-vocabulary change detection.
    
    Features:
    - Global Otsu thresholding on entire feature difference map
    - Local Otsu thresholding on edge regions (buildings boundaries)
    - Weighted combination of global and local thresholds
    - Lightweight computation with no neural network inference
    - Configurable edge detection methods (Canny, Sobel)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Otsu adaptive threshold module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Configuration parameters
        self.global_weight = self.config.get('global_weight', DEFAULT_GLOBAL_WEIGHT)
        self.edge_weight = self.config.get('edge_weight', DEFAULT_EDGE_WEIGHT)
        self.edge_detection_method = self.config.get('edge_detection_method', 'canny')
        self.canny_low = self.config.get('canny_low', DEFAULT_CANNY_LOW)
        self.canny_high = self.config.get('canny_high', DEFAULT_CANNY_HIGH)
        self.edge_dilation_kernel = self.config.get('edge_dilation_kernel', DEFAULT_EDGE_DILATION_KERNEL)
        self.min_edge_pixels = self.config.get('min_edge_pixels', DEFAULT_MIN_EDGE_PIXELS)
        self.debug = self.config.get('debug', False)
        
        # Threshold mapping parameters
        self.min_angle = self.config.get('min_angle', DEFAULT_MIN_ANGLE)
        self.max_angle = self.config.get('max_angle', DEFAULT_MAX_ANGLE)
        
        # Validate edge detection method
        if self.edge_detection_method not in EDGE_DETECTION_METHODS:
            raise ValueError(f"Invalid edge detection method: {self.edge_detection_method}. "
                           f"Supported methods: {EDGE_DETECTION_METHODS}")
    
    def initialize(self) -> None:
        """Initialize the module."""
        if self.debug:
            print(f"OtsuAdaptiveThresholdModule initialized:")
            print(f"  Global weight: {self.global_weight}")
            print(f"  Edge weight: {self.edge_weight}")
            print(f"  Edge detection: {self.edge_detection_method}")
            print(f"  Angle range: [{self.min_angle}, {self.max_angle}]")
        
        self.is_initialized = True
    
    def compute_adaptive_threshold(
        self,
        img1_embed: torch.Tensor,
        img2_embed: torch.Tensor,
        base_threshold: float,
        **kwargs
    ) -> float:
        """
        Compute adaptive threshold using global + local Otsu on feature difference map.
        
        Args:
            img1_embed: Feature embeddings of first image [C, H, W]
            img2_embed: Feature embeddings of second image [C, H, W]
            base_threshold: Base threshold (used as fallback)
            
        Returns:
            Adaptive threshold value in degrees
        """
        try:
            # Step 1: Compute feature difference map
            diff_map = self._compute_feature_difference(img1_embed, img2_embed)
            
            # Step 2: Compute global Otsu threshold
            global_thresh = self._compute_global_otsu(diff_map)
            
            # Step 3: Detect edge regions and compute local Otsu
            edge_thresh = self._compute_edge_otsu(diff_map, img1_embed)
            
            # Step 4: Combine thresholds
            if edge_thresh is not None:
                # Weighted combination
                final_thresh = (self.global_weight * global_thresh + 
                               self.edge_weight * edge_thresh)
                combination_type = "global+edge"
            else:
                # Fallback to global only
                final_thresh = global_thresh
                combination_type = "global_only"
            
            # Convert to angle degrees (similar to original threshold format)
            final_thresh_angle = self._similarity_to_angle(final_thresh)
            
            if self.debug:
                print(f"  Global Otsu: {self._similarity_to_angle(global_thresh):.1f}°")
                if edge_thresh is not None:
                    print(f"  Edge Otsu: {self._similarity_to_angle(edge_thresh):.1f}°")
                print(f"  Combination: {combination_type}")
                print(f"  Final threshold: {final_thresh_angle:.1f}° (base: {base_threshold:.1f}°)")
            
            return final_thresh_angle
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Otsu adaptive threshold failed: {e}")
            return base_threshold
    
    def _compute_feature_difference(
        self, 
        img1_embed: torch.Tensor, 
        img2_embed: torch.Tensor
    ) -> np.ndarray:
        """
        Compute feature difference map ||feat_t1 - feat_t2||.
        
        Args:
            img1_embed: Feature embeddings of first image [C, H, W]
            img2_embed: Feature embeddings of second image [C, H, W]
            
        Returns:
            Difference map as numpy array [H, W]
        """
        # Compute L2 norm of feature differences
        diff = img1_embed - img2_embed
        diff_norm = torch.norm(diff, dim=0)  # [H, W]
        
        # Convert to numpy and normalize to [0, 1]
        diff_map = diff_norm.cpu().numpy()
        
        # Fast normalization with minimal checks
        diff_min, diff_max = diff_map.min(), diff_map.max()
        if diff_max > diff_min + NUMERICAL_EPSILON:  # Avoid division by zero
            diff_map = (diff_map - diff_min) / (diff_max - diff_min)
        else:
            # If all values are essentially the same, return zeros
            diff_map = np.zeros_like(diff_map)
        
        return diff_map
    
    def _compute_global_otsu(self, diff_map: np.ndarray) -> float:
        """
        Compute global Otsu threshold on the entire difference map.
        
        Args:
            diff_map: Feature difference map [H, W] in range [0, 1]
            
        Returns:
            Global Otsu threshold value in range [0, 1]
        """
        # Convert to uint8 for Otsu (required by skimage) - optimized
        diff_uint8 = (diff_map * 255).astype(np.uint8)
        
        # Check if the image has sufficient variation for Otsu
        if diff_uint8.max() == diff_uint8.min():
            # If all pixels have the same value, return a default threshold
            return 0.5
        
        try:
            # Compute Otsu threshold
            otsu_thresh = threshold_otsu(diff_uint8)
            
            # Convert back to [0, 1] range
            global_thresh = otsu_thresh / 255.0
            
            return global_thresh
            
        except Exception as e:
            if self.debug:
                print(f"Global Otsu computation failed: {e}")
            # Return a reasonable default threshold
            return 0.5
    
    def _compute_edge_otsu(
        self, 
        diff_map: np.ndarray, 
        img1_embed: torch.Tensor
    ) -> Optional[float]:
        """
        Compute local Otsu threshold on edge regions of the difference map.
        
        IMPORTANT: Edge detection is performed on diff_map (feature difference), 
        not on img1_embed (semantic features). This is because:
        - diff_map edges represent change region boundaries (e.g., building edges)
        - img1_embed features are smooth and lack clear edges for detection
        
        Args:
            diff_map: Feature difference map [H, W] in range [0, 1]
            img1_embed: Feature embeddings of first image [C, H, W] 
                       (kept for API compatibility, not used)
            
        Returns:
            Edge Otsu threshold value or None if insufficient edge pixels
        """
        # Step 1: Convert diff_map to uint8 for edge detection
        diff_uint8 = (diff_map * 255).astype(np.uint8)
        
        # Step 2: Detect edges on the difference map
        try:
            if self.edge_detection_method == 'canny':
                edges = cv2.Canny(diff_uint8, self.canny_low, self.canny_high)
            elif self.edge_detection_method == 'sobel':
                # Sobel edge detection on difference map
                grad_x = cv2.Sobel(diff_uint8, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(diff_uint8, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(grad_x**2 + grad_y**2)
                edges = (edges > np.percentile(edges, 80)).astype(np.uint8) * 255
            else:
                raise ValueError(f"Unknown edge detection method: {self.edge_detection_method}")
        except Exception as e:
            if self.debug:
                print(f"  Edge detection failed: {e}")
            return None
        
        # Step 3: Dilate edge regions to get more context
        if self.edge_dilation_kernel > 0:
            kernel = np.ones((self.edge_dilation_kernel, self.edge_dilation_kernel), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Step 4: Extract edge regions from difference map
        edge_mask = edges > 0
        edge_diff_values = diff_map[edge_mask]
        
        # Step 5: Check if we have enough edge pixels
        if len(edge_diff_values) < self.min_edge_pixels:
            if self.debug:
                print(f"  Insufficient edge pixels: {len(edge_diff_values)} < {self.min_edge_pixels}")
            return None
        
        # Step 6: Compute Otsu on edge regions
        edge_uint8 = (edge_diff_values * 255).astype(np.uint8)
        
        try:
            edge_otsu_thresh = threshold_otsu(edge_uint8)
            edge_thresh = edge_otsu_thresh / 255.0
            
            if self.debug:
                print(f"  Edge pixels: {len(edge_diff_values)}")
                
            return edge_thresh
        except Exception as e:
            if self.debug:
                print(f"  Edge Otsu failed: {e}")
            return None
    
    def _similarity_to_angle(self, similarity_value: float) -> float:
        """
        Convert normalized similarity value to angle in degrees.
        
        This function maps a normalized similarity value [0, 1] to an angle
        in the range [min_angle, max_angle] degrees for threshold computation.
        
        Args:
            similarity_value: Normalized similarity value in range [0, 1]
            
        Returns:
            Angle in degrees in range [min_angle, max_angle]
        """
        # Fast clipping and mapping without numpy overhead
        if similarity_value < 0.0:
            similarity_value = 0.0
        elif similarity_value > 1.0:
            similarity_value = 1.0
        
        # Direct calculation - avoid numpy for scalar operations
        angle = self.min_angle + similarity_value * (self.max_angle - self.min_angle)
        
        return angle
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get configuration template for this module."""
        return {
            "global_weight": DEFAULT_GLOBAL_WEIGHT,
            "edge_weight": DEFAULT_EDGE_WEIGHT,
            "edge_detection_method": "canny",  # Options: "canny" or "sobel"
            "canny_low": DEFAULT_CANNY_LOW,
            "canny_high": DEFAULT_CANNY_HIGH,
            "edge_dilation_kernel": DEFAULT_EDGE_DILATION_KERNEL,
            "min_edge_pixels": DEFAULT_MIN_EDGE_PIXELS,
            "min_angle": DEFAULT_MIN_ANGLE,
            "max_angle": DEFAULT_MAX_ANGLE,
            "debug": False
        }
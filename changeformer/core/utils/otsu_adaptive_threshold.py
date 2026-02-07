"""
Otsu-based Adaptive Threshold Module for OVCD Change Detection.

This module implements a lightweight adaptive threshold adjustment using
global and local Otsu thresholding on DINOv2 feature difference maps.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from skimage.filters import threshold_otsu


class OtsuAdaptiveThresholdModule:
    """
    Otsu-based adaptive threshold module for change detection.
    
    Uses global + local (edge-focused) Otsu thresholding on DINOv2 feature
    difference maps to compute optimal thresholds for building change detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Otsu adaptive threshold module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configuration parameters
        self.global_weight = self.config.get('global_weight', 0.5)  # Weight for global Otsu
        self.edge_weight = self.config.get('edge_weight', 0.5)      # Weight for edge Otsu
        self.edge_detection_method = self.config.get('edge_detection_method', 'canny')  # 'canny' or 'sobel'
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.edge_dilation_kernel = self.config.get('edge_dilation_kernel', 3)  # Dilate edge regions
        self.min_edge_pixels = self.config.get('min_edge_pixels', 100)  # Minimum pixels for edge Otsu
        self.debug = self.config.get('debug', False)
        
        print(f"OtsuAdaptiveThresholdModule initialized (global_w: {self.global_weight}, edge_w: {self.edge_weight})")
    
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
                print(f"  Global Otsu: {self._cosine_to_angle(global_thresh):.1f} | Edge Otsu: {self._cosine_to_angle(edge_thresh):.1f}")
            else:
                # Fallback to global only
                final_thresh = global_thresh
                combination_type = "global_only"
                print(f"  Global Otsu: {self._cosine_to_angle(global_thresh):.1f} | Edge: insufficient pixels")
            
            # Convert to angle degrees (similar to original threshold format)
            final_thresh_angle = self._cosine_to_angle(final_thresh)
            
            if self.debug:
                print(f"  Global Otsu: {self._cosine_to_angle(global_thresh):.1f}°")
                if edge_thresh is not None:
                    print(f"  Edge Otsu: {self._cosine_to_angle(edge_thresh):.1f}°")
                print(f"  Combination: {combination_type}")
                print(f"  Final threshold: {final_thresh_angle:.1f}° (base: {base_threshold:.1f}°)")
            
            return final_thresh_angle
            
        except Exception as e:
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
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        
        return diff_map
    
    def _compute_global_otsu(self, diff_map: np.ndarray) -> float:
        """
        Compute global Otsu threshold on the entire difference map.
        
        Args:
            diff_map: Feature difference map [H, W]
            
        Returns:
            Global Otsu threshold value
        """
        # Convert to uint8 for Otsu (required by skimage)
        diff_uint8 = (diff_map * 255).astype(np.uint8)
        
        # Compute Otsu threshold
        otsu_thresh = threshold_otsu(diff_uint8)
        
        # Convert back to [0, 1] range
        global_thresh = otsu_thresh / 255.0
        
        return global_thresh
    
    def _compute_edge_otsu(
        self, 
        diff_map: np.ndarray, 
        img1_embed: torch.Tensor
    ) -> Optional[float]:
        """
        Compute local Otsu threshold on edge regions.
        
        Args:
            diff_map: Feature difference map [H, W]
            img1_embed: Feature embeddings for edge detection [C, H, W]
            
        Returns:
            Edge Otsu threshold value or None if insufficient edge pixels
        """
        # Step 1: Create intensity image for edge detection
        # Use the first few channels or compute intensity
        if img1_embed.shape[0] >= 3:
            # If we have at least 3 channels, use them as RGB-like
            intensity = torch.mean(img1_embed[:3], dim=0)
        else:
            # Otherwise use mean of all channels
            intensity = torch.mean(img1_embed, dim=0)
        
        intensity_np = intensity.cpu().numpy()
        intensity_uint8 = ((intensity_np - intensity_np.min()) / 
                          (intensity_np.max() - intensity_np.min() + 1e-8) * 255).astype(np.uint8)
        
        # Step 2: Detect edges
        if self.edge_detection_method == 'canny':
            edges = cv2.Canny(intensity_uint8, self.canny_low, self.canny_high)
        elif self.edge_detection_method == 'sobel':
            # Sobel edge detection
            grad_x = cv2.Sobel(intensity_uint8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(intensity_uint8, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            edges = (edges > np.percentile(edges, 80)).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unknown edge detection method: {self.edge_detection_method}")
        
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
    
    def _cosine_to_angle(self, cosine_sim: float) -> float:
        """
        Convert cosine similarity to angle in degrees.
        
        Note: This is a simplified conversion. In practice, you might need
        to adjust this based on how your similarity values map to angles.
        
        Args:
            cosine_sim: Cosine similarity value [0, 1]
            
        Returns:
            Angle in degrees
        """
        # Map [0, 1] similarity to reasonable angle range [100, 180]
        # Higher similarity -> higher angle (more restrictive threshold)
        angle = 100 + cosine_sim * 80
        return angle
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get configuration template for this module."""
        return {
            "global_weight": 0.5,
            "edge_weight": 0.5,
            "edge_detection_method": "canny",  # or "sobel"
            "canny_low": 50,
            "canny_high": 150,
            "edge_dilation_kernel": 3,
            "min_edge_pixels": 100,
            "debug": False
        }
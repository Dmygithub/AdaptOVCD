"""
ARA (Adaptive Radiometric Alignment) Module

Cross-temporal radiometric harmonization for multi-temporal remote sensing images.
ARA adaptively aligns radiometric conditions between temporal image pairs while 
preserving original image characteristics for optimal SAM performance.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any
from skimage import exposure

from .base import BaseEnhancementModule


class AdaptiveRadiometricAlignmentModule(BaseEnhancementModule):
    """
    ARA (Adaptive Radiometric Alignment) Module
    
    Design Principles:
    - Adaptive radiometric harmonization between temporal pairs
    - Preserve original image characteristics for SAM
    - Minimal processing complexity
    - Cross-temporal consistency optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.module_name = "adaptive_radiometric_alignment"
        
    def get_module_type(self) -> str:
        """Returns module type."""
        return "preprocessing"
    
    def get_config_template(self) -> Dict[str, Any]:
        """Returns configuration template."""
        return {
            "method": "simple_histogram",
            "preserve_original": True,
            "gentle_processing": True,
            "clip_extreme": True,
            "max_change_ratio": 0.15
        }
    
    def initialize(self) -> None:
        """Initializes the module."""
        config = self.config
        self.method = config.get("method", "simple_histogram")
        self.preserve_original = config.get("preserve_original", True)
        self.gentle_processing = config.get("gentle_processing", True)
        self.clip_extreme = config.get("clip_extreme", True)
        self.max_change_ratio = config.get("max_change_ratio", 0.15)
        self.is_initialized = True
        print(f"Adaptive radiometric alignment initialized: {self.method}")
    
    
    def process(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs lightweight radiometric alignment.
        
        Args:
            img1: Reference image (H, W, C).
            img2: Image to align (H, W, C).
            
        Returns:
            Tuple of (img1, aligned_img2).
        """
        if self.method == "disabled":
            return img1, img2
        elif self.method == "simple_histogram":
            return self._simple_histogram_match(img1, img2)
        elif self.method == "luminance_only":
            return self._luminance_alignment(img1, img2)
        else:
            return img1, img2
    
    def _simple_histogram_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple histogram matching with minimal processing overhead."""
        try:
            # Use same method as DynamicEarth project
            img2_aligned = exposure.match_histograms(
                image=img2, 
                reference=img1, 
                channel_axis=-1
            ).astype(np.uint8)
            
            # Gentle processing: limit change magnitude
            if self.gentle_processing:
                img2_aligned = self._gentle_blend(img2, img2_aligned)
            
            # Clip extreme values
            if self.clip_extreme:
                img2_aligned = self._clip_extremes(img2_aligned)
            
            return img1, img2_aligned
            
        except Exception as e:
            print(f"Warning: Histogram matching failed: {e}")
            return img1, img2
    
    def _luminance_alignment(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Luminance-only alignment (most conservative approach)."""
        try:
            # Convert to YUV space (faster than LAB)
            yuv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YUV)
            yuv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)
            
            # Align Y channel (luminance) only
            y1, y2 = yuv1[:, :, 0], yuv2[:, :, 0]
            y2_aligned = exposure.match_histograms(y2, y1)
            
            # Reconstruct image
            yuv2_aligned = yuv2.copy()
            yuv2_aligned[:, :, 0] = y2_aligned
            
            img2_aligned = cv2.cvtColor(yuv2_aligned, cv2.COLOR_YUV2RGB)
            
            # Gentle processing
            if self.gentle_processing:
                img2_aligned = self._gentle_blend(img2, img2_aligned)
            
            return img1, img2_aligned.astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Luminance alignment failed: {e}")
            return img1, img2
    
    def _gentle_blend(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Gentle blending to prevent over-processing."""
        # Calculate change magnitude
        diff = np.abs(processed.astype(np.float32) - original.astype(np.float32))
        max_diff = np.max(diff)
        
        if max_diff == 0:
            return processed
        
        # Reduce processing strength if change is too large
        change_ratio = max_diff / 255.0
        if change_ratio > self.max_change_ratio:
            # Calculate blend weight
            blend_weight = self.max_change_ratio / change_ratio
            result = (blend_weight * processed.astype(np.float32) + 
                     (1 - blend_weight) * original.astype(np.float32))
            return result.astype(np.uint8)
        
        return processed
    
    def _clip_extremes(self, img: np.ndarray) -> np.ndarray:
        """Clips extreme values to avoid pure black/white."""
        # Clip to [5, 250] range
        return np.clip(img, 5, 250)

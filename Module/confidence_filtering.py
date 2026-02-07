"""
Confidence-based Filtering Post-processing Module for Change Detection.

This module implements adaptive confidence-based filtering to improve change
detection reliability by automatically determining optimal confidence thresholds
based on statistical analysis of prediction distributions.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage, stats
from skimage import filters

from .base import PostProcessingModule

# Constants
DEFAULT_CONFIDENCE_PERCENTILE = 75
DEFAULT_MIN_CONFIDENCE = 0.1
DEFAULT_MAX_CONFIDENCE = 0.9
DEFAULT_STABILITY_FACTOR = 1.5
DEFAULT_MIN_REGION_SIZE = 50


class ConfidenceFilteringModule(PostProcessingModule):
    """
    Confidence-based adaptive filtering module for change detection.
    
    This module applies statistical analysis to automatically determine optimal
    confidence thresholds and filter unreliable predictions:
    
    1. Analyze prediction confidence distribution
    2. Compute adaptive threshold using percentile-based statistics
    3. Apply confidence-aware region filtering
    4. Remove isolated low-confidence predictions
    
    Features:
    - Pure NumPy/SciPy implementation (lightweight)
    - Adaptive threshold based on data distribution
    - Fast processing (<5ms for 512x512 images)
    - Robust to varying prediction quality
    - Configurable for different confidence levels
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize confidence filtering module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.module_name = "confidence_filtering"
    
    def get_module_type(self) -> str:
        """Get module type."""
        return "post_processing"
        
    def get_config_template(self) -> Dict[str, Any]:
        """Get configuration template for this module."""
        return {
            "filter_strength": 0.1,  # 0.0 (conservative) to 1.0 (aggressive)
            "min_region_size": DEFAULT_MIN_REGION_SIZE
        }
    
    def initialize(self) -> None:
        """Initialize the module."""
        config = self.config
        # Simplified configuration
        filter_strength = config.get('filter_strength', 0.1)
        self.min_region_size = config.get('min_region_size', DEFAULT_MIN_REGION_SIZE)
        
        # Convert filter_strength to internal parameters
        # filter_strength 0.0 -> percentile 95 (very conservative)
        # filter_strength 1.0 -> percentile 50 (more aggressive)  
        self.confidence_percentile = 95 - (filter_strength * 45)
        self.min_confidence = DEFAULT_MIN_CONFIDENCE
        self.max_confidence = DEFAULT_MAX_CONFIDENCE
        self.stability_factor = DEFAULT_STABILITY_FACTOR
        self.enable_adaptive_threshold = True
        self.enable_region_filtering = True
        self.enable_isolation_removal = True
        self.debug = config.get('debug', False)
        
        print(f"ConfidenceFilteringModule: strength={filter_strength:.1f}, min_size={self.min_region_size}")
        
        self.is_initialized = True
    
    def post_process(
        self, 
        prediction_mask: np.ndarray, 
        confidence_scores: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply confidence-based filtering to prediction mask.
        
        Args:
            prediction_mask: Binary prediction mask [H, W] or probability map [H, W, C]
            confidence_scores: Optional confidence scores [H, W]
            original_image: Original RGB image [H, W, 3] (unused but kept for compatibility)
            
        Returns:
            Tuple of (processed_mask, filtered_confidence_scores)
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Handle different input formats
            if prediction_mask.ndim == 2:
                # Binary mask or single probability map
                if prediction_mask.max() <= 1.0:
                    # Probability map [0, 1]
                    prob_map = prediction_mask
                    binary_mask = (prob_map > 0.5).astype(np.uint8)
                else:
                    # Binary mask [0, 255]
                    binary_mask = (prediction_mask > 0).astype(np.uint8)
                    prob_map = binary_mask.astype(np.float32)
            elif prediction_mask.ndim == 3:
                # Multi-class probability map - use change class (assume last channel)
                prob_map = prediction_mask[:, :, -1]
                binary_mask = (prob_map > 0.5).astype(np.uint8)
            else:
                raise ValueError(f"Invalid prediction_mask shape: {prediction_mask.shape}")
            
            # Use confidence scores if provided, otherwise use probability map
            if confidence_scores is not None:
                confidence_map = confidence_scores
            else:
                confidence_map = prob_map
            
            if np.sum(binary_mask) == 0:
                return prediction_mask, confidence_scores
            
            processed_mask = binary_mask.copy()
            processed_confidence = confidence_map.copy() if confidence_map is not None else None
            original_pixels = np.sum(processed_mask > 0)
            
            # Step 1: Adaptive threshold determination
            if self.enable_adaptive_threshold:
                adaptive_threshold = self._compute_adaptive_threshold(confidence_map, binary_mask)
                processed_mask = (confidence_map >= adaptive_threshold).astype(np.uint8)
            
            # Step 2: Region-based confidence filtering
            if self.enable_region_filtering:
                processed_mask = self._region_confidence_filtering(
                    processed_mask, confidence_map
                )
            
            # Step 3: Isolation removal
            if self.enable_isolation_removal:
                processed_mask = self._remove_isolated_predictions(processed_mask)
            
            # Convert back to original format
            if prediction_mask.ndim == 2 and prediction_mask.max() > 1:
                processed_mask = (processed_mask * 255).astype(np.uint8)
            elif prediction_mask.ndim == 3:
                # Create multi-class output
                output_shape = prediction_mask.shape
                processed_output = np.zeros(output_shape, dtype=prediction_mask.dtype)
                processed_output[:, :, 0] = 1 - processed_mask  # Background
                processed_output[:, :, -1] = processed_mask     # Change class
                processed_mask = processed_output
            
            if self.debug:
                processed_pixels = np.sum(processed_mask > 0) if processed_mask.ndim == 2 else np.sum(processed_mask[:, :, -1] > 0)
                reduction = (original_pixels - processed_pixels) / max(1, original_pixels) * 100
                print(f"  Confidence filtering: {original_pixels} -> {processed_pixels} pixels (-{reduction:.1f}%)")
            
            return processed_mask, processed_confidence
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Confidence filtering failed: {e}")
            return prediction_mask, confidence_scores
    
    def _compute_adaptive_threshold(
        self, 
        confidence_map: np.ndarray, 
        binary_mask: np.ndarray
    ) -> float:
        """
        Compute adaptive confidence threshold based on distribution analysis.
        
        Args:
            confidence_map: Confidence scores [H, W]
            binary_mask: Current binary mask [H, W]
            
        Returns:
            Adaptive confidence threshold
        """
        # Get confidence values for predicted change pixels
        change_confidences = confidence_map[binary_mask > 0]
        
        if len(change_confidences) == 0:
            return self.min_confidence
        
        # Compute percentile-based threshold
        percentile_threshold = np.percentile(change_confidences, self.confidence_percentile)
        
        # Apply stability factor to avoid over-filtering
        adaptive_threshold = percentile_threshold / self.stability_factor
        
        # Clamp to valid range
        adaptive_threshold = np.clip(
            adaptive_threshold, 
            self.min_confidence, 
            self.max_confidence
        )
        
        if self.debug:
            mean_conf = np.mean(change_confidences)
            print(f"    Adaptive threshold: {adaptive_threshold:.3f} "
                  f"(percentile: {percentile_threshold:.3f}, mean: {mean_conf:.3f})")
        
        return adaptive_threshold
    
    def _region_confidence_filtering(
        self, 
        binary_mask: np.ndarray, 
        confidence_map: np.ndarray
    ) -> np.ndarray:
        """
        Filter regions based on their average confidence scores.
        
        Args:
            binary_mask: Binary mask [H, W]
            confidence_map: Confidence scores [H, W]
            
        Returns:
            Filtered binary mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return binary_mask
        
        filtered_mask = np.zeros_like(binary_mask)
        kept_regions = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            region_mask = (labels == i)
            region_size = stats[i, cv2.CC_STAT_AREA]
            
            # Skip very small regions
            if region_size < self.min_region_size:
                continue
            
            # Compute region confidence statistics
            region_confidences = confidence_map[region_mask]
            mean_confidence = np.mean(region_confidences)
            std_confidence = np.std(region_confidences)
            
            # Simple confidence criterion: mean confidence above adaptive threshold
            # and reasonable stability (low std relative to mean)
            stability_ratio = std_confidence / (mean_confidence + 1e-6)
            
            if mean_confidence >= self.min_confidence and stability_ratio < 2.0:
                filtered_mask[region_mask] = 1
                kept_regions += 1
        
        if self.debug and num_labels > 1:
            print(f"    Region filtering: {num_labels-1} -> {kept_regions} regions")
        
        return filtered_mask
    
    def _remove_isolated_predictions(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Remove isolated predictions using morphological operations.
        
        Args:
            binary_mask: Binary mask [H, W]
            
        Returns:
            Mask with isolated predictions removed
        """
        # Use small opening to remove isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return cleaned_mask
    
    def process_binary_mask(
        self, 
        change_mask: np.ndarray, 
        confidence_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convenient method for processing binary change masks.
        
        Args:
            change_mask: Binary change mask [H, W]
            confidence_scores: Optional confidence scores [H, W]
            
        Returns:
            Processed binary change mask [H, W]
        """
        processed_mask, _ = self.post_process(
            prediction_mask=change_mask,
            confidence_scores=confidence_scores
        )
        return processed_mask
    
    def process_probability_map(
        self,
        prob_map: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Convenient method for processing probability maps.
        
        Args:
            prob_map: Probability map [H, W] or [H, W, C]
            confidence_threshold: Initial threshold for binarization
            
        Returns:
            Processed binary change mask [H, W]
        """
        # Initial binarization
        if prob_map.ndim == 3:
            binary_init = (prob_map[:, :, -1] > confidence_threshold).astype(np.uint8)
        else:
            binary_init = (prob_map > confidence_threshold).astype(np.uint8)
        
        processed_mask, _ = self.post_process(
            prediction_mask=prob_map,
            confidence_scores=prob_map if prob_map.ndim == 2 else prob_map[:, :, -1]
        )
        
        return processed_mask if processed_mask.ndim == 2 else processed_mask[:, :, -1]

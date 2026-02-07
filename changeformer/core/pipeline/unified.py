"""
Unified pipeline for change detection.

This module implements a unified pipeline that combines segmentation, comparison,
and identification stages through a configuration-driven approach.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from skimage.io import imread

from ..segmentation.base import BaseSegmentor
from ..comparison.base import BaseComparator  
from ..identification.base import BaseIdentifier


class UnifiedPipeline:
    """
    Unified change detection pipeline.
    
    This pipeline combines four stages:
    1. Segmentation: Generate masks from input images
    2. Comparison: Compare features to identify potential changes
    3. Identification: Perform semantic classification of changes
    4. Post-processing: Apply enhancement modules to refine results
    """
    
    def __init__(self, segmentor: BaseSegmentor, comparator: BaseComparator, 
                 identifier: BaseIdentifier, config: Dict[str, Any]):
        """
        Initialize the unified pipeline.
        
        Args:
            segmentor: Segmentation component
            comparator: Comparison component
            identifier: Identification component
            config: Full pipeline configuration
        """
        self.segmentor = segmentor
        self.comparator = comparator
        self.identifier = identifier
        self.config = config
        
        # Handle auto device detection
        device = config.get('device', 'cuda')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Setup all components
        self._setup_components()
        
        print(f"Unified pipeline initialized: {segmentor} -> {comparator} -> {identifier}")
    
    def _get_comparison_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get appropriate images for comparison stage.
        
        Uses preprocessed images from segmentor if available to ensure 
        resolution consistency between segmentation and comparison stages.
        
        Args:
            img1, img2: Original input images
            
        Returns:
            Tuple of images to use for comparison
        """
        # Check if segmentor has preprocessing module and get processed images
        if hasattr(self.segmentor, 'preprocessing_module') and self.segmentor.preprocessing_module is not None:
            # Use the same preprocessing as segmentor for consistency
            processed_img1, processed_img2 = self.segmentor.preprocessing_module.process(img1, img2)
            # Removed debug print: Using radiometric-aligned images for comparison
            return processed_img1, processed_img2
        
        # Fallback to original images if no preprocessing was applied
        return img1, img2
    
    def _setup_components(self):
        """Setup all pipeline components with their respective configurations."""
        # Setup segmentor
        if not self.segmentor._is_setup:
            self.segmentor.setup(self.config['segmentor'], self.device)
        
        # Setup comparator
        if not self.comparator._is_setup:
            self.comparator.setup(self.config['comparator'], self.device)
        
        # Setup identifier
        if not self.identifier._is_setup:
            self.identifier.setup(self.config['identifier'], self.device)
    
    def predict(self, img1_path: str, img2_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict change detection between two images.
        
        Args:
            img1_path: Path to the first image
            img2_path: Path to the second image
            
        Returns:
            Tuple of (change_mask, metadata)
        """
        # Load images
        img1 = imread(img1_path)
        img2 = imread(img2_path)
        
        # Stage 1: Segmentation
        masks, img1_mask_num = self.segmentor.segment(img1, img2)
        
        if len(masks) == 0:
            # No masks generated
            change_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
            metadata = {
                'total_masks': 0,
                'change_masks': 0,
                'identified_masks': 0,
                'img1_mask_num': 0
            }
            return change_mask, metadata
        
        # Stage 2: Comparison
        # Use processed images if available from segmentor to ensure resolution consistency
        comparison_img1, comparison_img2 = self._get_comparison_images(img1, img2)
        change_masks, img1_mask_num = self.comparator.compare(comparison_img1, comparison_img2, masks, img1_mask_num)
        
        if len(change_masks) == 0:
            # No changes detected
            change_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
            metadata = {
                'total_masks': len(masks),
                'change_masks': 0,
                'identified_masks': 0,
                'img1_mask_num': img1_mask_num
            }
            return change_mask, metadata
        
        # Stage 3: Identification
        identified_masks, img1_mask_num = self.identifier.identify(img1, img2, change_masks, img1_mask_num)
        
        # Merge final masks
        change_mask = self._merge_masks(identified_masks, img1.shape[:2])
        
        # Stage 4: Post-processing (if available)
        change_mask = self._apply_postprocessing(change_mask, img1)
        
        # Prepare metadata
        metadata = {
            'total_masks': len(masks),
            'change_masks': len(change_masks),
            'identified_masks': len(identified_masks),
            'img1_mask_num': img1_mask_num,
            'device': self.device
        }
        
        return change_mask, metadata
    
    def _apply_postprocessing(self, change_mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Apply post-processing modules to the change mask.
        
        Args:
            change_mask: Binary change mask [H, W] (0-255 format)
            original_image: Original RGB image [H, W, 3] for context
            
        Returns:
            Post-processed change mask
        """
        try:
            # Get post-processing modules from comparator's enhancement modules
            if hasattr(self.comparator, 'enhancement_modules') and self.comparator.enhancement_modules:
                # Convert to binary format for processing
                binary_mask = (change_mask > 0).astype(np.uint8)
                original_pixels = np.sum(binary_mask > 0)
                
                # Apply each post-processing module
                for module_name, module in self.comparator.enhancement_modules.items():
                    if hasattr(module, 'get_module_type') and module.get_module_type() == 'post_processing':
                        try:
                            # Apply post-processing
                            processed_mask, _ = module.post_process(
                                prediction_mask=binary_mask,
                                confidence_scores=None,
                                original_image=original_image
                            )
                            
                            binary_mask = processed_mask if processed_mask.ndim == 2 else processed_mask[:, :, -1]
                            binary_mask = (binary_mask > 0).astype(np.uint8)
                            
                            # Debug output
                            processed_pixels = np.sum(binary_mask > 0)
                            change_pct = (processed_pixels - original_pixels) / max(1, original_pixels) * 100
                            print(f"  {module_name} post-processing: {original_pixels} -> {processed_pixels} pixels ({change_pct:+.1f}%)")
                            original_pixels = processed_pixels  # Update for next module
                            
                        except Exception as e:
                            print(f"Warning: {module_name} post-processing failed: {e}")
                            continue
                
                # Convert back to 0-255 format
                final_mask = (binary_mask * 255).astype(np.uint8)
                return final_mask
            
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
        
        # Return original mask if post-processing fails or no modules available
        return change_mask
    
    def _merge_masks(self, masks: list, shape: Tuple[int, int]) -> np.ndarray:
        """
        Merge individual masks into a single binary mask.
        
        Args:
            masks: List of masks to merge
            shape: Output shape (height, width)
            
        Returns:
            Merged binary change mask
        """
        if len(masks) == 0:
            return np.zeros(shape, dtype=np.uint8)
        
        # Sum the masks and convert to binary (255 for changed areas)
        change_mask = np.sum(masks, axis=0).astype(np.uint8)
        change_mask[change_mask > 0] = 255
        
        return change_mask
    
    def batch_detect(self, image_pairs, output_dir, verbose=False):
        """
        Batch process multiple image pairs for evaluation.
        
        Args:
            image_pairs: List of (img1_path, img2_path) tuples
            output_dir: Directory to save prediction results
            verbose: Whether to print verbose output
            
        Returns:
            List of result dictionaries with success status
        """
        import os
        from skimage.io import imsave
        
        results = []
        
        for i, (img1_path, img2_path) in enumerate(image_pairs):
            result = {'success': False, 'error': None}
            
            try:
                if verbose:
                    print(f"Processing pair {i+1}/{len(image_pairs)}: {os.path.basename(img1_path)}")
                
                # Run prediction
                change_mask, metadata = self.predict(img1_path, img2_path)
                
                # Save result
                output_filename = os.path.basename(img1_path)
                output_path = os.path.join(output_dir, output_filename)
                imsave(output_path, change_mask)
                
                result['success'] = True
                result['metadata'] = metadata
                
            except Exception as e:
                result['error'] = str(e)
                if verbose:
                    print(f"  Error: {e}")
            
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, str]:
        """
        Get information about the current pipeline configuration.
        
        Returns:
            Dictionary with pipeline component information
        """
        return {
            'segmentor': str(self.segmentor),
            'comparator': str(self.comparator),
            'identifier': str(self.identifier),
            'device': self.device
        }
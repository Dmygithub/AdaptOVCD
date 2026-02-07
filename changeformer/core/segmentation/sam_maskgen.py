"""
SAM mask generation utilities.

This module contains the mask generators.
"""

import os
import sys
import torch
import numpy as np
from skimage.exposure import match_histograms
from typing import Dict, Any, Tuple

# Import from torchvision
from torchvision.ops.boxes import batched_nms, box_area

# Ensure third_party paths are added
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sam_path = os.path.join(project_root, 'third_party', 'segment_anything')
if sam_path not in sys.path:
    sys.path.insert(0, sam_path)

try:
    from segment_anything.utils.amg import (
        MaskData,
        area_from_rle,
        generate_crop_boxes
    )
    from segment_anything import SamAutomaticMaskGenerator
    
    SAM_AVAILABLE = True
    print("SAM successfully imported in sam_maskgen")
except ImportError as e:
    print(f"Warning: SAM not available in sam_maskgen: {e}")
    SAM_AVAILABLE = False
    MaskData = None
    SamAutomaticMaskGenerator = None
    generate_crop_boxes = None
    area_from_rle = None



if SAM_AVAILABLE:
    class SimpleMaskGenerator:
        """
        Simplified SAM mask generator wrapper - handles mask_threshold properly.
        """
        
        def __init__(self, *args, **kwargs):
            # mask_threshold is now handled in MaskProposal.make_mask_generator
            # Just create the SamAutomaticMaskGenerator directly
            self._generator = SamAutomaticMaskGenerator(*args, **kwargs)
        
        def __getattr__(self, name):
            """Delegate all other attributes to the wrapped generator."""
            return getattr(self._generator, name)
        
        @torch.no_grad()
        def simple_generate(self, image: np.ndarray) -> MaskData:
            """
            Generates masks for the given image.

            Arguments:
              image (np.ndarray): The image to generate masks for, in HWC uint8 format.

            Returns:
               MaskData: A MaskData object containing mask information.
            """
            # Generate masks
            mask_data = self._simple_generate_masks(image)
            initial_count = len(mask_data['rles'])
            # Generated initial masks

            # Filter small disconnected regions and holes in masks
            if self.min_mask_region_area > 0:
                # Apply postprocess_small_regions filtering
                mask_data = self.postprocess_small_regions(
                    mask_data,
                    self.min_mask_region_area,
                    max(self.box_nms_thresh, self.crop_nms_thresh),
                )
                final_count = len(mask_data['rles'])
                # Applied small regions filtering

            mask_data['areas'] = np.asarray([area_from_rle(rle) for rle in mask_data['rles']])
            if isinstance(mask_data['boxes'], torch.Tensor):
                mask_data['areas'] = torch.from_numpy(mask_data['areas'])
                
            return mask_data
        
        def _simple_generate_masks(self, image: np.ndarray) -> MaskData:
        """Generate masks using crop-based approach."""
            orig_size = image.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )

            # Iterate over image crops
            data = MaskData()
            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
                data.cat(crop_data)

            # Remove duplicate masks between crops
            if len(crop_boxes) > 1:
                # Prefer masks from smaller crops
                scores = 1 / box_area(data["crop_boxes"])
                scores = scores.to(data["boxes"].device)
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    scores,
                    torch.zeros(len(data["boxes"])),  # categories
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)

            return data
else:
    # Fallback if SAM not available
    class SimpleMaskGenerator:
        def __init__(self, *args, **kwargs):
            # Remove mask_threshold to avoid parameter errors
            kwargs.pop('mask_threshold', 0.0)
            raise ImportError("SAM not available")
        
        def simple_generate(self, image):
            raise ImportError("SAM not available")


class MaskProposal:
    """
    Mask proposal generator for SAM-based segmentation.
    
    """
    
    def __init__(self):
        """Initialize mask proposal generator."""
        self.set_hyperparameters()
        self.maskgen = None

    def set_hyperparameters(self, **kwargs):
        """
        Set hyperparameters for mask proposal generation.
        
        Args:
            match_hist (bool): Whether to match histograms between images
            area_thresh (float): Area threshold for filtering masks
            points_per_side (int): Number of points to sample per side
            points_per_batch (int): Number of points to process per batch
            pred_iou_thresh (float): Prediction IoU threshold
            stability_score_thresh (float): Stability score threshold
            stability_score_offset (float): Stability score offset
            box_nms_thresh (float): Box NMS threshold
            min_mask_region_area (int): Minimum mask region area
        """
        # SAM mask generation parameters
        self.points_per_side = kwargs.get('points_per_side', 32)
        self.points_per_batch = kwargs.get('points_per_batch', 64)
        self.pred_iou_thresh = kwargs.get('pred_iou_thresh', 0.5)
        self.stability_score_thresh = kwargs.get('stability_score_thresh', 0.95)
        self.stability_score_offset = kwargs.get('stability_score_offset', 0.9)
        self.box_nms_thresh = kwargs.get('box_nms_thresh', 0.7)
        self.min_mask_region_area = kwargs.get('min_mask_region_area', 0)
        
        # Mask proposal parameters
        self.match_hist = kwargs.get('match_hist', False)
        self.area_thresh = kwargs.get('area_thresh', 0.8)
        # MaskProposal hyperparameters set

    def make_mask_generator(self, generator_class=SimpleMaskGenerator, **kwargs):
        """
        Create mask generator instance.
        
        Args:
            generator_class: Mask generator class (SimpleMaskGenerator or SAM2_SimpleMaskGenerator)
            **kwargs: Arguments for mask generator initialization including model, points_per_side, etc.
        """
        # Extract mask_threshold before creating generator (SAM doesn't accept it as constructor param)
        mask_threshold = kwargs.pop('mask_threshold', 0.0)
        
        # Create mask generator without mask_threshold
        self.maskgen = generator_class(**kwargs)
        
        # Set mask_threshold directly on the SAM model if provided
        if mask_threshold != 0.0 and hasattr(self.maskgen, 'predictor') and hasattr(self.maskgen.predictor, 'model'):
            self.maskgen.predictor.model.mask_threshold = mask_threshold
        
        # Store mask_threshold for reference
        self.mask_threshold = mask_threshold
    
    def proposal(self, img: np.ndarray) -> Tuple[MaskData, int]:
        """
        Generate mask proposals for a single image.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            Tuple of (mask_data, num_masks)
        """
        h, w = img.shape[:2]
        # Generate mask proposals
        mask_data = self.maskgen.simple_generate(img)
        
        # mask_threshold is now handled natively by SAM model
        
        # Apply area-based filtering
        original_count = len(mask_data['rles'])
        
        # Filter masks that are too large
        mask_data.filter((mask_data['areas'] / (h * w)) < self.area_thresh)
        
        # Area filtering completed
        filtered_count = len(mask_data['rles'])
        
        return mask_data, filtered_count
    


    def forward(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[MaskData, int]:
        """
        Generate mask proposals for two images.
        
        Args:
            img1 (np.ndarray): First image
            img2 (np.ndarray): Second image
            
        Returns:
            Tuple of (combined_mask_data, img1_mask_count)
        """
        if self.match_hist:
            img2 = match_histograms(image=img2, reference=img1, channel_axis=-1).astype(np.uint8)

        data = MaskData()
        mask_num = []
        for im in [img1, img2]:
            d, l = self.proposal(im)
            data.cat(d)
            mask_num.append(l)

        # Apply NMS to remove duplicate masks
        keep = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        keep = keep.sort()[0]  # Sort for subsequent separate processing
        data.filter(keep)

        # Return mask data and mask number of img1
        return data, len(keep[keep < mask_num[0]])
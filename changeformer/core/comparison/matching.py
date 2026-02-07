"""
Bitemporal feature matching and instance-level change detection.

This module contains algorithms for comparing features between two temporal images
and identifying changed instances based on semantic similarity.

"""

import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Union, Dict, Any
from skimage.filters import threshold_otsu

# Ensure third_party paths are added
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sam_path = os.path.join(project_root, 'third_party', 'segment_anything')
if sam_path not in sys.path:
    sys.path.insert(0, sam_path)

try:
    import pycocotools.mask as maskUtils
    from segment_anything.utils.amg import rle_to_mask, MaskData
    COCO_AVAILABLE = True
    print("SAM successfully imported in matching")
except ImportError as e:
    print(f"Warning: pycocotools or SAM not available in matching: {e}")
    COCO_AVAILABLE = False
    maskUtils = None
    rle_to_mask = None
    MaskData = None


def angle_to_cosine(angle: float) -> float:
    """
    Convert an angle in degrees to its cosine value.
    
    Args:
        angle (float): Angle in degrees (0-180)
        
    Returns:
        float: Cosine value (-1 to 1)
    """
    assert 0 <= angle <= 180, "Angle must be between 0 and 180 degrees."
    return math.cos(math.radians(angle))


def cosine_to_angle(cosine: float) -> float:
    """
    Convert a cosine value to its corresponding angle in degrees.
    
    Args:
        cosine (float): Cosine value (-1 to 1)
        
    Returns:
        float: Angle in degrees (0-180)
    """
    assert -1 <= cosine <= 1, "Cosine value must be between -1 and 1."
    return math.degrees(math.acos(cosine))


def extract_features(image: np.ndarray, model, processor, model_config: Dict[str, Any], device: str) -> torch.Tensor:
    """
    Process the input image and extract features using the specified model.

    Args:
        image (np.ndarray): Input image as a NumPy array
        model: Pre-trained model for feature extraction
        processor: Pre-processing function for images
        model_config (dict): Configuration dictionary containing model type, feature dimension, and patch size
        device (str): Device to run the model on ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Extracted feature tensor
    """
    model_type = model_config['model_type']
    feature_dim = model_config['feature_dim']
    patch_size = model_config['patch_size']

    if model_type == 'DINO':
        processed_image = processor(image).unsqueeze(0).to(device)
        h, w = processed_image.shape[-2:]
        features = model.get_intermediate_layers(processed_image)[0]
        features = features[:, 1:].permute(0, 2, 1).view(1, feature_dim, h // patch_size, w // patch_size)
    elif model_type == 'DINOv2':
        processed_image = processor(images=image, return_tensors="pt")
        pixel_values = processed_image['pixel_values'].to(device)
        features = model(pixel_values)
        
        # Get actual patch tokens (exclude CLS token)
        patch_tokens = features.last_hidden_state[:, 1:]  # Shape: [1, num_patches, feature_dim]
        num_patches = patch_tokens.shape[1]
        
        # Calculate actual grid size from number of patches
        grid_size = int(num_patches ** 0.5)
        features = patch_tokens.permute(0, 2, 1).view(1, feature_dim, grid_size, grid_size)
    elif model_type == 'DINOv3':
        # DINOv3 uses similar processing to DINOv2
        processed_image = processor(images=image, return_tensors="pt")
        pixel_values = processed_image['pixel_values'].to(device)
        
        # DINOv3 forward_features returns a dict with patch tokens
        with torch.no_grad():
            output = model.forward_features(pixel_values)
        
        # Get patch tokens from the output dict
        # Shape: [batch, num_patches, feature_dim]
        patch_tokens = output['x_norm_patchtokens']
        num_patches = patch_tokens.shape[1]
        
        # Calculate grid size from number of patches
        grid_size = int(num_patches ** 0.5)
        features = patch_tokens.permute(0, 2, 1).view(1, feature_dim, grid_size, grid_size)
    else:
        raise ValueError("Unsupported model type. Use 'DINO', 'DINOv2', or 'DINOv3'.")

    return features


@torch.no_grad()
def bitemporal_feature_matching(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_data: np.ndarray,
    model,
    processor,
    img1_mask_num: int,
    model_config: Dict[str, Any] = None,
    device: str = 'cpu',
    auto_threshold: bool = False,
    change_confidence_threshold: float = 145,
    enhancement_modules: Dict[str, Any] = None,
) -> Tuple[np.ndarray, int]:
    """
    Perform bitemporal change detection between two images using feature matching.
    
    This function is an extension of AnyChange: "Segment any change. arXiv:2402.01188."
    
    Args:
        img1 (np.ndarray): T1 image
        img2 (np.ndarray): T2 image  
        mask_data (np.ndarray): Mask data for change detection (boolean array)
        model: Pre-trained model for feature extraction
        processor: Pre-processing function for images
        img1_mask_num (int): Number of masks from img1 to consider
        model_config (dict, optional): Configuration dictionary for model parameters
        device (str): Device to run the model on ('cpu' or 'cuda')
        auto_threshold (bool): Whether to compute threshold automatically
        change_confidence_threshold (float): Threshold for change confidence
        enhancement_modules: Dictionary of enhancement modules
    
    Returns:
        Tuple[np.ndarray, int]: Filtered mask data and number of masks retained from img1
    """

    # Set default model configuration if none provided
    if model_config is None:
        model_config = {
            'model_type': 'DINO',
            'feature_dim': 768,
            'patch_size': 16
        }
    
    # Extract features for both images
    img1_embed = extract_features(img1, model, processor, model_config, device)
    img2_embed = extract_features(img2, model, processor, model_config, device)

    # Resize embeddings to match the original image dimensions
    H, W = img1.shape[:2]
    img1_embed = F.interpolate(img1_embed, size=(H, W), mode='bilinear', align_corners=True).squeeze_(0)
    img2_embed = F.interpolate(img2_embed, size=(H, W), mode='bilinear', align_corners=True).squeeze_(0)

    # Apply enhancement modules for threshold adjustment
    if enhancement_modules:
        original_threshold = change_confidence_threshold
        
        # Apply threshold adjustment modules
        for module_name, module in enhancement_modules.items():
            if hasattr(module, 'compute_adaptive_threshold'):
                change_confidence_threshold = module.compute_adaptive_threshold(
                    img1_embed=img1_embed,
                    img2_embed=img2_embed,
                    base_threshold=change_confidence_threshold
                )
                delta = change_confidence_threshold - original_threshold
                break  # Use the first threshold adjustment module for now
    
    # Fallback to automatic threshold computation if required
    if auto_threshold and not enhancement_modules:
        cos_similarity = -F.cosine_similarity(img1_embed, img2_embed, dim=0)
        cos_similarity_flat = cos_similarity.reshape(-1).cpu().numpy()
        threshold = threshold_otsu(cos_similarity_flat)
        change_confidence_threshold = cosine_to_angle(threshold)

    def _latent_match(mask_data: np.ndarray, img1_embed: torch.Tensor, img2_embed: torch.Tensor):
        """Match latent features of images based on the provided mask."""
        change_confidence = torch.zeros(len(mask_data), dtype=torch.float32, device=device)

        for i, mask in enumerate(mask_data):
            binary_mask = torch.from_numpy(mask).to(device, dtype=torch.bool)
            t1_mask_embed = torch.mean(img1_embed[:, binary_mask], dim=-1)
            t2_mask_embed = torch.mean(img2_embed[:, binary_mask], dim=-1)
            score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
            change_confidence[i] += score

        # Keep masks where confidence exceeds the threshold
        keep_indices = change_confidence > angle_to_cosine(change_confidence_threshold)
        keep_indices = keep_indices.cpu().numpy()
        retained_mask_data = mask_data[keep_indices]
        retained_count_img1 = len(np.where(keep_indices[:img1_mask_num])[0])

        return retained_mask_data, retained_count_img1

    # Perform latent matching
    filtered_mask_data, filtered_img1_mask_num = _latent_match(mask_data, img1_embed, img2_embed)
    
    
    return filtered_mask_data, filtered_img1_mask_num


def instance_level_change_detection(
    mask_data: Union[MaskData, np.ndarray],
    img1_mask_num: int,
    iou_threshold: float = 1e-2
) -> Tuple[np.ndarray, int]:
    """
    Identify and return the changed instance masks and the updated mask count.
    
    Identifying and returning the changed instance masks and the updated mask count.
    

    Parameters:
        mask_data (Union[MaskData, np.ndarray]): Input mask data
        img1_mask_num (int): The number of masks in the first image
        iou_threshold (float, optional): IoU threshold for determining instance changes

    Returns:
        Tuple[np.ndarray, int]: Array of changed instances and updated count of img1 masks
    """
    
    if not COCO_AVAILABLE:
        raise ImportError("pycocotools is required for instance-level change detection")
    
    # Convert MaskData to a NumPy array if necessary
    if isinstance(mask_data, MaskData):
        mask_data = np.array([rle_to_mask(rle) for rle in mask_data['rles']], dtype=np.uint8)
    elif isinstance(mask_data, np.ndarray):
        mask_data = mask_data.astype(np.uint8)
    
    img1_masks = mask_data[:img1_mask_num]
    img2_masks = mask_data[img1_mask_num:]
    
    # Determine changed instances based on the presence of masks
    if len(img1_masks) == 0:
        change_instances = img2_masks if len(img2_masks) > 0 else np.array([])
        img1_mask_num = 0
    elif len(img2_masks) == 0:
        change_instances = img1_masks
        img1_mask_num = len(img1_masks)
    else:
        img1_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in img1_masks]
        img2_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in img2_masks]
        
        ious = maskUtils.iou(img2_rles, img1_rles, [0] * len(img1_rles))
        ious_img1 = ious.sum(axis=0)
        ious_img2 = ious.sum(axis=1)
        
        img1_change_idx = np.where(ious_img1 <= iou_threshold)[0]
        img2_change_idx = np.where(ious_img2 <= iou_threshold)[0]
        
        img1_mask_num = len(img1_change_idx)
        change_instances = np.concatenate(
            [img1_masks[img1_change_idx], img2_masks[img2_change_idx]]
        )
    
    return change_instances, img1_mask_num


# Legacy function alias for backward compatibility
def bitemporal_match(*args, **kwargs):
    """Legacy alias for bitemporal_feature_matching."""
    return bitemporal_feature_matching(*args, **kwargs)


def instance_ceg(*args, **kwargs):
    """Legacy alias for instance_level_change_detection."""
    return instance_level_change_detection(*args, **kwargs)
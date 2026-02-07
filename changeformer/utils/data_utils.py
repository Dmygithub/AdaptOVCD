"""
Data utilities for ChangeFormer.

This module provides functions for loading and preprocessing image data.
"""

import os
import re
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from skimage.io import imread

# Import dataset adapters
from .dataset_adapters import get_unified_image_pairs


def natural_sort_key(filename: str) -> list:
    """
    Create a key for natural sorting that handles numbers correctly.
    
    For example: test_1.png, test_2.png, ..., test_10.png, test_11.png
    instead of: test_1.png, test_10.png, test_11.png, ..., test_2.png
    
    Args:
        filename: Filename to create sort key for
        
    Returns:
        List of strings and integers for proper sorting
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', filename)]


def load_image_pair(img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pair of images for change detection.
    
    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        
    Returns:
        Tuple of (img1, img2) as numpy arrays
        
    Raises:
        FileNotFoundError: If either image file doesn't exist
        IOError: If images cannot be loaded
    """
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"First image not found: {img1_path}")
    
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Second image not found: {img2_path}")
    
    try:
        img1 = imread(img1_path)
        img2 = imread(img2_path)
    except Exception as e:
        raise IOError(f"Failed to load images: {e}")
    
    return img1, img2


def validate_image_pair(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    Validate that two images are compatible for change detection.
    
    Args:
        img1: First image array
        img2: Second image array
        
    Raises:
        ValueError: If images are not compatible
    """
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError(
            f"Image dimensions don't match: {img1.shape[:2]} vs {img2.shape[:2]}"
        )
    
    if len(img1.shape) != len(img2.shape):
        raise ValueError(
            f"Image channel dimensions don't match: {len(img1.shape)} vs {len(img2.shape)}"
        )


def ensure_output_directory(output_path: str) -> str:
    """
    Ensure output directory exists, create if necessary.
    
    Args:
        output_path: Path to output file or directory
        
    Returns:
        Normalized output path
    """
    # Get directory path
    if os.path.isfile(output_path) or output_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = output_path
    
    # Create directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    return os.path.normpath(output_path)


def get_image_files(directory: str, extensions: Optional[list] = None) -> list:
    """
    Get list of image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: common image formats)
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for file in sorted(os.listdir(directory), key=natural_sort_key):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, file))
    
    return image_files


def match_image_pairs(dir1: str, dir2: str) -> list:
    """
    Match image files between two directories by filename.
    
    Args:
        dir1: First image directory
        dir2: Second image directory
        
    Returns:
        List of tuples (img1_path, img2_path) for matched pairs
    """
    files1 = {os.path.basename(f): f for f in get_image_files(dir1)}
    files2 = {os.path.basename(f): f for f in get_image_files(dir2)}
    
    pairs = []
    for filename in sorted(files1.keys(), key=natural_sort_key):
        if filename in files2:
            pairs.append((files1[filename], files2[filename]))
    
    return pairs


def get_dataset_image_pairs(dataset_name: str, data_root: str, 
                          dataset_config: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Unified interface: Get image pairs using dataset adapters.
    
    Args:
        dataset_name: Dataset name
        data_root: Dataset root path
        dataset_config: Dataset configuration
        
    Returns:
        (image_pairs, evaluation_config) tuple
    """
    return get_unified_image_pairs(dataset_name, data_root, dataset_config) 
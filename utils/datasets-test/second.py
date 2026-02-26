#!/usr/bin/env python3
"""
SECOND Dataset Preprocessing Script

This script processes the SECOND dataset by converting semantic segmentation labels
to class-specific change detection labels. It analyzes temporal semantic labels 
and generates binary change masks for each of the 6 classes.

Classes and their RGB color codes:
- Non-change: RGB(255, 255, 255) - White (background)
- Low vegetation: RGB(0, 128, 0) - Dark green  
- Tree: RGB(0, 255, 0) - Bright green
- Building: RGB(128, 0, 0) - Dark red
- Water: RGB(0, 0, 255) - Blue
- Non-vegetated ground surface: RGB(128, 128, 128) - Gray
- Playground: RGB(255, 0, 0) - Red (if exists)

Usage:
    conda activate ovcd
    python utils/second.py
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List


# Class definitions with RGB color mapping
CLASS_COLOR_MAP = {
    'building': (128, 0, 0),           # Dark red
    'water': (0, 0, 255),             # Blue  
    'tree': (0, 255, 0),              # Bright green
    'low_vegetation': (0, 128, 0),    # Dark green
    'non_veg_ground_surface': (128, 128, 128),  # Gray
    'playground': (255, 0, 0),        # Red
}

# Background color (non-change)
BACKGROUND_COLOR = (255, 255, 255)    # White

# Default paths  
DEFAULT_INPUT_DIR = "data/second"
DEFAULT_OUTPUT_DIR = "data/second"


def create_class_mask(label1: np.ndarray, label2: np.ndarray, 
                     target_color: Tuple[int, int, int]) -> np.ndarray:
    """
    Create binary change mask for a specific class.
    
    Args:
        label1: First temporal semantic label (H, W, 3)
        label2: Second temporal semantic label (H, W, 3)  
        target_color: RGB color tuple for the target class
        
    Returns:
        Binary change mask where 255 indicates change, 0 indicates no change
    """
    # Create masks for target class in both images
    mask1 = np.all(label1 == target_color, axis=-1)
    mask2 = np.all(label2 == target_color, axis=-1)
    
    # Change detection: XOR operation (appears in one but not both)
    change_mask = np.logical_xor(mask1, mask2)
    
    # Convert to uint8 binary mask
    binary_mask = change_mask.astype(np.uint8) * 255
    
    return binary_mask


def process_image_pair(img1_path: str, img2_path: str, 
                      label1_path: str, label2_path: str,
                      output_base_dir: str, verbose: bool = False) -> Dict[str, bool]:
    """
    Process a single image pair and generate class-specific change masks.
    
    Args:
        img1_path: Path to first temporal image
        img2_path: Path to second temporal image  
        label1_path: Path to first temporal label
        label2_path: Path to second temporal label
        output_base_dir: Base output directory
        verbose: Enable verbose output
        
    Returns:
        Dictionary with success status for each class
    """
    results = {}
    
    try:
        # Load semantic labels
        label1 = np.array(Image.open(label1_path))
        label2 = np.array(Image.open(label2_path))
        
        if verbose:
            print(f"  Label shapes: {label1.shape}, {label2.shape}")
        
        # Get filename for output
        filename = os.path.basename(img1_path)
        
        # Process each class
        for class_name, target_color in CLASS_COLOR_MAP.items():
            try:
                # Create class-specific change mask
                change_mask = create_class_mask(label1, label2, target_color)
                
                # Create output directory for this class
                class_output_dir = os.path.join(output_base_dir, f"label_{class_name}")
                os.makedirs(class_output_dir, exist_ok=True)
                
                # Save change mask
                output_path = os.path.join(class_output_dir, filename)
                Image.fromarray(change_mask).save(output_path)
                
                results[class_name] = True
                
            except Exception as e:
                print(f"    Error processing {class_name}: {e}")
                results[class_name] = False
                
    except Exception as e:
        print(f"  Error processing image pair: {e}")
        for class_name in CLASS_COLOR_MAP.keys():
            results[class_name] = False
    
    return results


def copy_original_structure(input_dir: str, output_dir: str) -> bool:
    """
    Copy the original image and label structure to output directory.
    
    Args:
        input_dir: Input directory containing im1, im2, label1, label2
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import shutil
        
        # Copy image directories
        for subdir in ['im1', 'im2', 'label1', 'label2']:
            src_dir = os.path.join(input_dir, subdir)
            dst_dir = os.path.join(output_dir, subdir)
            
            if os.path.exists(src_dir):
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"Copied {subdir} directory")
            else:
                print(f"Warning: {subdir} directory not found in input")
                
        return True
        
    except Exception as e:
        print(f"Error copying original structure: {e}")
        return False




def main():
    """Main preprocessing function."""
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    
    print("SECOND Dataset Preprocessing")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        print("Please make sure your data is in the 'data/second' directory")
        sys.exit(1)
    
    required_subdirs = ['im1', 'im2', 'label1', 'label2']
    for subdir in required_subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Error: Required subdirectory '{subdir}' not found in input directory")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy original structure only if input and output directories are different
    if input_dir != output_dir:
        print("\nCopying original directory structure...")
        copy_original_structure(input_dir, output_dir)
    else:
        print("\nInput and output directories are the same, skipping copy step...")
    
    # Get image pairs
    im1_dir = os.path.join(input_dir, 'im1') 
    im2_dir = os.path.join(input_dir, 'im2')
    label1_dir = os.path.join(input_dir, 'label1')
    label2_dir = os.path.join(input_dir, 'label2')
    
    # Get list of image files
    image_files = [f for f in os.listdir(im1_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    print(f"\nProcessing {len(image_files)} image pairs...")
    
    # Statistics tracking
    success_stats = {class_name: 0 for class_name in CLASS_COLOR_MAP.keys()}
    total_processed = 0
    
    # Process each image pair
    for i, filename in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        # Construct file paths
        img1_path = os.path.join(im1_dir, filename)
        img2_path = os.path.join(im2_dir, filename)
        label1_path = os.path.join(label1_dir, filename)
        label2_path = os.path.join(label2_dir, filename)
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [img1_path, img2_path, label1_path, label2_path]):
            print(f"  Skipping {filename}: missing files")
            continue
        
        # Process image pair
        results = process_image_pair(
            img1_path, img2_path, label1_path, label2_path,
            output_dir, verbose=False
        )
        
        # Update statistics
        total_processed += 1
        for class_name, success in results.items():
            if success:
                success_stats[class_name] += 1
    
    # Print final statistics
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)
    print(f"Total image pairs processed: {total_processed}")
    print(f"Output directory: {output_dir}")
    print("\nClass-specific success rates:")
    
    for class_name, success_count in success_stats.items():
        rate = success_count / total_processed * 100 if total_processed > 0 else 0
        class_output_dir = os.path.join(output_dir, f"label_{class_name}")
        file_count = len([f for f in os.listdir(class_output_dir) if f.endswith('.png')]) if os.path.exists(class_output_dir) else 0
        print(f"  {class_name:25}: {success_count:4d}/{total_processed} ({rate:5.1f}%) - {file_count} files")
    
    print(f"\nClass-specific label directories created:")
    for class_name in CLASS_COLOR_MAP.keys():
        class_dir = os.path.join(output_dir, f"label_{class_name}")
        print(f"  {class_dir}")


if __name__ == "__main__":
    main()

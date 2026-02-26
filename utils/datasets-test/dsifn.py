#!/usr/bin/env python3
"""
DSIFN Dataset Preprocessing Script.
Reorganizes DSIFN dataset from test folder to standard format (A/B/label).
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def preprocess_dsifn(source_dir, target_dir, convert_tif=True):
def preprocess_dsifn(source_dir, target_dir, convert_tif=True):
    """
    Preprocesses DSIFN dataset.
    
    Args:
        source_dir: Source directory path (DSIFN/test).
        target_dir: Target directory path (dsifn).
        convert_tif: Whether to convert tif to jpg.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Check source directory
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    
    t1_dir = source_path / "t1"
    t2_dir = source_path / "t2"
    mask_dir = source_path / "mask"
    
    for dir_path in [t1_dir, t2_dir, mask_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Subdirectory does not exist: {dir_path}")
    
    # Create target directory structure
    print(f"\n📁 Creating target directory: {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    
    a_dir = target_path / "A"
    b_dir = target_path / "B"
    label_dir = target_path / "label"
    
    for dir_path in [a_dir, b_dir, label_dir]:
        dir_path.mkdir(exist_ok=True)
        print(f"  ✓ {dir_path.name}/")
    
    # Copy t1 -> A
    print(f"\n📋 Copying t1 -> A...")
    t1_files = sorted(list(t1_dir.glob("*.jpg")))
    for i, src_file in enumerate(t1_files, 1):
        dst_file = a_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        if i % 50 == 0 or i == len(t1_files):
            print(f"  Processing... {i}/{len(t1_files)}")
    print(f"  ✓ Completed: {len(t1_files)} files")
    
    # Copy t2 -> B
    print(f"\n📋 Copying t2 -> B...")
    t2_files = sorted(list(t2_dir.glob("*.jpg")))
    for i, src_file in enumerate(t2_files, 1):
        dst_file = b_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        if i % 50 == 0 or i == len(t2_files):
            print(f"  Processing... {i}/{len(t2_files)}")
    print(f"  ✓ Completed: {len(t2_files)} files")
    
    # Process mask -> label (tif to jpg)
    print(f"\n🖼️  Processing mask -> label...")
    mask_files = sorted(list(mask_dir.glob("*.tif")))
    
    if convert_tif:
        print("  Converting .tif -> .jpg")
        
        for i, src_file in enumerate(mask_files, 1):
            # Read tif file
            img = Image.open(src_file)
            
            # Convert to numpy array
            arr = np.array(img)
            
            # Normalize to 0-255 range
            # If float (0.0-1.0), convert to 0-255
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 255).astype(np.uint8)
            # If int 0-1, convert to 0-255
            elif arr.max() <= 1:
                arr = (arr * 255).astype(np.uint8)
            
            # Create grayscale image
            img_gray = Image.fromarray(arr, mode='L')
            
            # Convert to RGB (for viewing)
            img_rgb = img_gray.convert('RGB')
            
            # Save as jpg
            dst_file = label_dir / f"{src_file.stem}.jpg"
            img_rgb.save(dst_file, 'JPEG', quality=100)
            
            img.close()
            img_rgb.close()
            
            if i % 50 == 0 or i == len(mask_files):
                print(f"  Processing... {i}/{len(mask_files)}")
        print(f"  ✓ Completed: {len(mask_files)} files (tif->jpg)")
    else:
        # Copy tif file directly
        for i, src_file in enumerate(mask_files, 1):
            dst_file = label_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            if i % 50 == 0 or i == len(mask_files):
                print(f"  Processing... {i}/{len(mask_files)}")
        print(f"  ✓ Completed: {len(mask_files)} files")
    
    # Verify file count
    print(f"\n✅ Preprocessing completed!")
    print(f"\n📊 Statistics:")
    print(f"  A/     : {len(list(a_dir.glob('*')))} files")
    print(f"  B/     : {len(list(b_dir.glob('*')))} files")
    print(f"  label/ : {len(list(label_dir.glob('*')))} files")
    
    # Check if counts match
    a_count = len(list(a_dir.glob('*')))
    b_count = len(list(b_dir.glob('*')))
    label_count = len(list(label_dir.glob('*')))
    
    if a_count == b_count == label_count:
        print(f"\n✓ Count verification passed: {a_count} image pairs")
    else:
        print(f"\n⚠️  Warning: File count mismatch!")
        print(f"  A: {a_count}, B: {b_count}, label: {label_count}")
    
    return target_path


def main():
    parser = argparse.ArgumentParser(description='DSIFN Dataset Preprocessing')
    parser.add_argument('--source', type=str, 
                        default='/home/dmy/dmy/OVCD/data/DSIFN/test',
                        help='Source directory path (default: /home/dmy/dmy/OVCD/data/DSIFN/test)')
    parser.add_argument('--target', type=str,
                        default='/home/dmy/dmy/OVCD/data/dsifn',
                        help='Target directory path (default: /home/dmy/dmy/OVCD/data/dsifn)')
    parser.add_argument('--no-convert', action='store_true',
                        help='Do not convert tif to jpg, copy directly')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("=" * 60)
    print("DSIFN Dataset Preprocessing")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Convert format: {'Yes (tif->jpg)' if not args.no_convert else 'No (Keep tif)'}")
    print("=" * 60)
    
    try:
        preprocess_dsifn(
            source_dir=args.source,
            target_dir=args.target,
            convert_tif=not args.no_convert
        )
        print(f"\n🎉 Preprocessing success! Dataset saved to: {args.target}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


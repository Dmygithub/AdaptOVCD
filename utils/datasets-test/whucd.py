#!/usr/bin/env python3
"""
WHUCD Dataset Preprocessing Script

This script preprocesses the WHU Building Change Detection dataset to match
the expected structure for evaluation, similar to LEVIRCD dataset.

Tasks:
1. Reorganize directory structure from Building change detection dataset_add
2. Convert TIF format to PNG format
3. Create whucd/A, whucd/B, whucd/label structure

Author: Assistant
Date: 2025-10-04
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import argparse
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Required packages not found. Please install:")
    print("pip install Pillow numpy")
    exit(1)

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
    print("GDAL found - will use for TIF reading")
except ImportError:
    GDAL_AVAILABLE = False
    print("GDAL not found - will use PIL for TIF reading")


class WHUCDPreprocessor:
    """WHUCD Dataset Preprocessor"""
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize preprocessor
        
        Args:
            source_dir: Path to 'Building change detection dataset_add' directory
            output_dir: Path to output 'whucd' directory
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Define source paths
        self.image_data_dir = self.source_dir / "1. The two-period image data"
        self.a_dir = self.image_data_dir / "2012" / "splited_images" / "test" / "image"  # First time phase
        self.b_dir = self.image_data_dir / "2016" / "splited_images" / "test" / "image"  # Second time phase
        self.label_file = self.image_data_dir / "change_label" / "test" / "change_label.tif"  # Change label
        
        # Define output paths
        self.output_a_dir = self.output_dir / "A"
        self.output_b_dir = self.output_dir / "B" 
        self.output_label_dir = self.output_dir / "label"
        
        self.validate_structure()
    
    def validate_structure(self):
        """Validate source directory structure"""
        print("Validating source directory structure...")
        
        required_paths = [
            self.a_dir,
            self.b_dir,
            self.label_file
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            print("Error: Missing required paths:")
            for path in missing_paths:
                print(f"  - {path}")
            raise FileNotFoundError("Required paths not found")
        
        print("✓ Source directory structure is valid")
    
    def setup_output_dirs(self):
        """Create output directory structure"""
        print("Setting up output directories...")
        
        # Remove existing output directory if it exists
        if self.output_dir.exists():
            print(f"Removing existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create output directories
        self.output_a_dir.mkdir(parents=True, exist_ok=True)
        self.output_b_dir.mkdir(parents=True, exist_ok=True)
        self.output_label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created output directory structure at: {self.output_dir}")
    
    def tif_to_png(self, tif_path: Path, png_path: Path) -> bool:
        """
        Convert TIF to PNG format
        
        Args:
            tif_path: Input TIF file path
            png_path: Output PNG file path
            
        Returns:
            Success status
        """
        try:
            if GDAL_AVAILABLE:
                # Use GDAL for better TIF support
                dataset = gdal.Open(str(tif_path))
                if dataset is None:
                    raise Exception("Failed to open with GDAL")
                
                # Read image data
                band_count = dataset.RasterCount
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                
                if band_count == 1:
                    # Grayscale or single band
                    band = dataset.GetRasterBand(1)
                    image_array = band.ReadAsArray()
                    image = Image.fromarray(image_array)
                else:
                    # Multi-band (RGB)
                    bands = []
                    for i in range(min(3, band_count)):  # Take first 3 bands for RGB
                        band = dataset.GetRasterBand(i + 1)
                        bands.append(band.ReadAsArray())
                    
                    if len(bands) == 3:
                        image_array = np.stack(bands, axis=-1)
                        image = Image.fromarray(image_array.astype(np.uint8))
                    else:
                        image = Image.fromarray(bands[0])
                
            else:
                # Fallback to PIL
                image = Image.open(tif_path)
            
            # Save as PNG
            image.save(png_path, 'PNG')
            return True
            
        except Exception as e:
            print(f"Error converting {tif_path}: {e}")
            return False
    
    def process_image_directory(self, source_dir: Path, output_dir: Path, desc: str):
        """
        Process a directory of images (convert TIF to PNG)
        
        Args:
            source_dir: Source directory with TIF files
            output_dir: Output directory for PNG files
            desc: Description for progress bar
        """
        print(f"Processing {desc}...")
        
        # Get list of TIF files
        tif_files = list(source_dir.glob("*.tif"))
        if not tif_files:
            print(f"Warning: No TIF files found in {source_dir}")
            return
        
        success_count = 0
        
        # Process each TIF file
        for tif_file in tqdm(tif_files, desc=desc):
            # Create output PNG filename
            png_filename = tif_file.stem + ".png"
            png_path = output_dir / png_filename
            
            # Convert TIF to PNG
            if self.tif_to_png(tif_file, png_path):
                success_count += 1
            else:
                print(f"Failed to convert: {tif_file}")
        
        print(f"✓ {desc}: {success_count}/{len(tif_files)} files converted successfully")
    
    def process_label_file(self):
        """
        Process the large change label TIF file
        
        Note: This assumes the label file covers the same area as the image tiles.
        If the label needs to be split into tiles matching the image tiles,
        additional logic would be needed.
        """
        print("Processing change label...")
        
        if not self.label_file.exists():
            print(f"Warning: Label file not found: {self.label_file}")
            return
        
        # For now, convert the single label file
        # In practice, you might need to split this into tiles matching your images
        output_label_path = self.output_label_dir / "change_label.png"
        
        if self.tif_to_png(self.label_file, output_label_path):
            print("✓ Change label converted successfully")
            
            # Create individual label files matching image names
            print("Creating individual label files...")
            self.create_individual_labels()
        else:
            print("Failed to convert change label")
    
    def create_individual_labels(self):
        """
        Create individual label PNG files by cropping the large label image
        based on the discovered grid pattern (23 columns x 30 rows).
        """
        # Get list of images from A directory to match naming
        a_images = list(self.output_a_dir.glob("*.png"))
        
        if not a_images:
            print("Warning: No images found in A directory for label matching")
            return
        
        # Load the main label image
        main_label_path = self.output_label_dir / "change_label.png"
        if not main_label_path.exists():
            print("Warning: Main label file not found")
            return
        
        try:
            main_label = Image.open(main_label_path)
            print(f"Main label size: {main_label.size}")
        except Exception as e:
            print(f"Error loading main label: {e}")
            return
        
        print("Creating individual label files by cropping...")
        
        # WHUCD grid parameters (discovered from analysis)
        GRID_COLS = 23  # X direction  
        GRID_ROWS = 30  # Y direction
        TILE_SIZE = 512  # Standard tile size
        
        label_width, label_height = main_label.size
        
        print(f"Grid: {GRID_COLS}x{GRID_ROWS}, Label size: {label_width}x{label_height}")
        print(f"Standard tile size: {TILE_SIZE}x{TILE_SIZE}")
        
        # Parse tile coordinates and crop corresponding regions
        successful_crops = 0
        
        for img_path in tqdm(a_images, desc="Cropping labels"):
            try:
                filename = img_path.stem  # Remove .png extension
                if '_' not in filename:
                    continue
                    
                parts = filename.split('_')
                if len(parts) != 2 or not all(p.isdigit() for p in parts):
                    continue
                
                x_idx, linear_idx = int(parts[0]), int(parts[1])
                
                # Convert linear index to 2D grid coordinates
                # Row-major order: linear_idx = row * cols + col
                grid_row = linear_idx // GRID_COLS
                grid_col = linear_idx % GRID_COLS
                
                # Calculate pixel coordinates for cropping (using standard tile size)
                x_pixel = grid_col * TILE_SIZE
                y_pixel = grid_row * TILE_SIZE
                
                # Define crop region with proper boundary handling
                x_end = min(x_pixel + TILE_SIZE, label_width)
                y_end = min(y_pixel + TILE_SIZE, label_height)
                
                crop_box = (x_pixel, y_pixel, x_end, y_end)
                
                # Crop the region from the large label image
                cropped_label = main_label.crop(crop_box)
                
                # Pad to standard tile size if needed (for boundary tiles)
                if cropped_label.size != (TILE_SIZE, TILE_SIZE):
                    # Create a standard size image filled with background (0)
                    padded = Image.new('L', (TILE_SIZE, TILE_SIZE), 0)
                    # Paste the cropped region at the top-left
                    padded.paste(cropped_label, (0, 0))
                    cropped_label = padded
                
                # Convert to binary mask (0 or 255)
                import numpy as np
                label_array = np.array(cropped_label)
                label_array = (label_array > 0) * 255
                
                # Save individual label
                individual_label_path = self.output_label_dir / f"{filename}.png"
                Image.fromarray(label_array.astype(np.uint8), mode='L').save(individual_label_path, 'PNG')
                successful_crops += 1
                
            except Exception as e:
                print(f"Error processing label for {img_path.name}: {e}")
                continue
        
        print(f"Successfully cropped {successful_crops} labels")
        
        # Remove the main label file as we now have individual ones
        try:
            main_label_path.unlink()
            print("Removed main label file")
        except:
            pass
    
    def run(self):
        """Run the complete preprocessing pipeline"""
        print("=" * 60)
        print("WHUCD Dataset Preprocessing")
        print("=" * 60)
        
        # Setup
        self.setup_output_dirs()
        
        # Process images
        self.process_image_directory(
            self.a_dir, 
            self.output_a_dir, 
            "Converting 2012 images (A)"
        )
        
        self.process_image_directory(
            self.b_dir, 
            self.output_b_dir, 
            "Converting 2016 images (B)"
        )
        
        # Process labels
        self.process_label_file()
        
        # Summary
        self.print_summary()
        
        print("=" * 60)
        print("✓ WHUCD preprocessing completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print("You can now use this with evaluate.py")
        print("=" * 60)
    
    def print_summary(self):
        """Print processing summary"""
        print("\nProcessing Summary:")
        print("-" * 40)
        
        # Count files in each directory
        a_count = len(list(self.output_a_dir.glob("*.png")))
        b_count = len(list(self.output_b_dir.glob("*.png")))
        label_count = len(list(self.output_label_dir.glob("*.png")))
        
        print(f"A directory (2012):     {a_count} files")
        print(f"B directory (2016):     {b_count} files")
        print(f"Label directory:        {label_count} files")
        
        if a_count != b_count:
            print("⚠️  Warning: Image count mismatch between A and B directories")
        
        if label_count == 0:
            print("⚠️  Warning: No label files created")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Preprocess WHUCD dataset")
    parser.add_argument(
        "--source", 
        type=str, 
        default="data/Building change detection dataset_add",
        help="Path to source 'Building change detection dataset_add' directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/whucd", 
        help="Path to output 'whucd' directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output directory"
    )
    
    args = parser.parse_args()
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory not found: {args.source}")
        print("Please ensure you have the 'Building change detection dataset_add' directory")
        return
    
    # Check if output exists and handle accordingly
    if os.path.exists(args.output) and not args.force:
        response = input(f"Output directory '{args.output}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    try:
        # Create and run preprocessor
        preprocessor = WHUCDPreprocessor(args.source, args.output)
        preprocessor.run()
        
        print(f"\n🎉 Success! You can now run:")
        print(f"python evaluate.py --dataset whucd --data_root {args.output}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

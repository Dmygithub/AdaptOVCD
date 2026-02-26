#!/usr/bin/env python3
"""
BANDON Dataset Preprocessor.

Reorganizes BANDON dataset into standard A/B/label structure, similar to LEVIR-CD.
Extracts necessary three folders from complex directory structure.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import argparse

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Warning: PIL not available, image validation will be skipped")
    Image = None
    np = None


class BANDONPreprocessor:
    """BANDON Dataset Preprocessor."""
    
    def __init__(self, source_dir: str, output_dir: str, overwrite: bool = False):
        """
        Initializes preprocessor.
        
        Args:
            source_dir: Source data directory (BANDON_Index/BANDON_Index/test).
            output_dir: Output directory (data/bandon).
            overwrite: Whether to overwrite existing directory.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        
        # Source paths
        self.source_imgs_a = self.source_dir / "imgs" / "A"
        self.source_imgs_b = self.source_dir / "imgs" / "B"
        self.source_labels = self.source_dir / "labels_unch0ch1ig255_unch0ch255"  # Use standard format
        
        # Output paths
        self.output_a_dir = self.output_dir / "A"
        self.output_b_dir = self.output_dir / "B"
        self.output_label_dir = self.output_dir / "label"
        
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def validate_source_structure(self) -> bool:
        """Validates source directory structure."""
        print("Validating source directory structure...")
        
        required_dirs = [
            self.source_imgs_a,
            self.source_imgs_b,
            self.source_labels
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"❌ Missing directory: {dir_path}")
                return False
            
            # Check if there are files
            files = list(dir_path.glob("*"))
            if not files:
                print(f"❌ Empty directory: {dir_path}")
                return False
            
            print(f"✓ Found directory: {dir_path} ({len(files)} files)")
        
        print("✓ Source directory structure is valid")
        return True
    
    def setup_output_directories(self):
        """Creates output directory structure."""
        print("Setting up output directories...")
        
        # If output directory exists, decide whether to overwrite based on overwrite parameter
        if self.output_dir.exists():
            if self.overwrite:
                print(f"Output directory {self.output_dir} already exists, overwriting...")
                # Remove existing directory
                shutil.rmtree(self.output_dir)
            else:
                print(f"Output directory {self.output_dir} already exists")
                print("Use --overwrite/-f flag to overwrite existing directory")
                return False
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_a_dir.mkdir(exist_ok=True)
        self.output_b_dir.mkdir(exist_ok=True)
        self.output_label_dir.mkdir(exist_ok=True)
        
        print(f"✓ Created output directory structure at: {self.output_dir}")
        return True
    
    def copy_images(self, source_dir: Path, target_dir: Path, desc: str) -> int:
        """
        Copies and converts image files to PNG format.
        
        Args:
            source_dir: Source directory.
            target_dir: Target directory.
            desc: Description.
            
        Returns:
            Number of converted files.
        """
        print(f"Converting {desc} from JPG to PNG...")
        
        # Get all JPG files
        jpg_files = list(source_dir.glob("*.jpg"))
        
        if not jpg_files:
            print(f"❌ No JPG files found in {source_dir}")
            return 0
        
        if not Image:
            print("❌ PIL not available, cannot convert images")
            return 0
        
        converted_count = 0
        
        # Add progress display
        try:
            from tqdm import tqdm
            jpg_files_iter = tqdm(jpg_files, desc=f"Converting {desc}")
        except ImportError:
            jpg_files_iter = jpg_files
            print(f"Processing {len(jpg_files)} files...")
        
        for jpg_file in jpg_files_iter:
            # Generate PNG filename
            png_name = jpg_file.stem + ".png"
            target_file = target_dir / png_name
            
            # Skip existing files
            if target_file.exists():
                converted_count += 1
                continue
            
            try:
                # Open JPG image and save as PNG
                with Image.open(jpg_file) as img:
                    # Ensure RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(target_file, 'PNG', optimize=True)
                converted_count += 1
                
                # Show progress every 10 files (if no tqdm)
                if not hasattr(jpg_files_iter, 'update') and converted_count % 10 == 0:
                    print(f"  Converted {converted_count}/{len(jpg_files)} files...")
                    
            except Exception as e:
                print(f"Error converting {jpg_file.name}: {e}")
                # Continue processing other files instead of stopping
        
        print(f"✓ {desc}: {converted_count}/{len(jpg_files)} files converted to PNG successfully")
        return converted_count
    
    def copy_labels(self) -> int:
        """
        Copies label files.
        
        Returns:
            Number of copied files.
        """
        print("Copying labels...")
        
        # Get all PNG files
        png_files = list(self.source_labels.glob("*.png"))
        
        if not png_files:
            print(f"❌ No PNG files found in {self.source_labels}")
            return 0
        
        copied_count = 0
        for png_file in png_files:
            target_file = self.output_label_dir / png_file.name
            
            try:
                shutil.copy2(png_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {png_file.name}: {e}")
        
        print(f"✓ Labels: {copied_count}/{len(png_files)} files copied successfully")
        return copied_count
    
    def validate_output(self) -> bool:
        """Validates output results."""
        print("Validating output...")
        
        # Check if file counts match (all files are now PNG)
        a_files = list(self.output_a_dir.glob("*.png"))
        b_files = list(self.output_b_dir.glob("*.png"))
        label_files = list(self.output_label_dir.glob("*.png"))
        
        print(f"Output file counts:")
        print(f"  A directory: {len(a_files)} PNG files")
        print(f"  B directory: {len(b_files)} PNG files")
        print(f"  Label directory: {len(label_files)} PNG files")
        
        # Check consistency
        if len(a_files) != len(b_files) or len(a_files) != len(label_files):
            print("❌ File count mismatch between A, B, and label directories")
            return False
        
        # Check if filenames match
        a_names = {f.stem for f in a_files}
        b_names = {f.stem for f in b_files}
        label_names = {f.stem for f in label_files}
        
        if a_names != b_names or a_names != label_names:
            print("❌ File name mismatch between directories")
            missing_in_b = a_names - b_names
            missing_in_label = a_names - label_names
            if missing_in_b:
                print(f"  Missing in B: {list(missing_in_b)[:5]}...")
            if missing_in_label:
                print(f"  Missing in label: {list(missing_in_label)[:5]}...")
            return False
        
        print(f"✓ All directories have matching {len(a_files)} files")
        
        # If PIL is available, validate images
        if Image and np:
            self._validate_sample_images()
        
        return True
    
    def _validate_sample_images(self):
        """Validates sample images."""
        print("Validating sample images...")
        
        # Check a few sample files (now all PNG)
        sample_files = ["test_1.png", "test_9.png", "test_22.png"]
        
        for sample_name in sample_files:
            base_name = sample_name.replace('.png', '')
            
            a_path = self.output_a_dir / sample_name
            b_path = self.output_b_dir / sample_name
            label_path = self.output_label_dir / sample_name
            
            if all(p.exists() for p in [a_path, b_path, label_path]):
                try:
                    # Check image size
                    img_a = Image.open(a_path)
                    img_b = Image.open(b_path)
                    label = Image.open(label_path)
                    
                    print(f"  {base_name}:")
                    print(f"    A: {img_a.size}, B: {img_b.size}, Label: {label.size}")
                    
                    # Check label values
                    label_array = np.array(label)
                    unique_values = np.unique(label_array)
                    print(f"    Label values: {unique_values}")
                    
                    if img_a.size != img_b.size or img_a.size != label.size:
                        print(f"    ⚠️  Size mismatch for {base_name}")
                    
                except Exception as e:
                    print(f"    Error validating {base_name}: {e}")
    
    def run(self):
        """Runs the complete preprocessing pipeline."""
        print("=" * 60)
        print("BANDON Dataset Preprocessing")
        print("=" * 60)
        
        # Validate source directory
        if not self.validate_source_structure():
            print("❌ Source validation failed")
            return False
        
        # Set output directory
        if not self.setup_output_directories():
            return False
        
        # Copy files
        a_count = self.copy_images(self.source_imgs_a, self.output_a_dir, "A images (2012)")
        b_count = self.copy_images(self.source_imgs_b, self.output_b_dir, "B images (2016)")
        label_count = self.copy_labels()
        
        # Validate results
        if not self.validate_output():
            print("❌ Output validation failed")
            return False
        
        # Print summary
        print("\nProcessing Summary:")
        print("-" * 40)
        print(f"A directory (2012):     {a_count} files")
        print(f"B directory (2016):     {b_count} files")
        print(f"Label directory:        {label_count} files")
        print("=" * 60)
        print("✓ BANDON preprocessing completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print("You can now use this with evaluate.py")
        print("=" * 60)
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess BANDON dataset to standard A/B/label structure"
    )
    
    parser.add_argument(
        "--source_dir", "-s",
        type=str,
        default="data/BANDON_Index/BANDON_Index/test",
        help="Source directory containing the raw BANDON dataset"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data/bandon",
        help="Output directory for processed dataset"
    )
    
    parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="Overwrite existing output directory without asking"
    )
    
    args = parser.parse_args()
    
    # Create and run preprocessor
    preprocessor = BANDONPreprocessor(args.source_dir, args.output_dir, args.overwrite)
    
    try:
        success = preprocessor.run()
        if success:
            print("\n🎉 Preprocessing completed successfully!")
            print("\nNext steps:")
            print("1. Run evaluation:")
            print("   python evaluate.py --dataset bandon --output_dir outputs/bandon/base --save_predictions --verbose")
            print("2. Or run in background:")
            print("   nohup python evaluate.py --dataset bandon --output_dir outputs/bandon/base --save_predictions --verbose > log/bandon/base.log 2>&1 &")
        else:
            print("\n❌ Preprocessing failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Preprocessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

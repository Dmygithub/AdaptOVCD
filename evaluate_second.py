import os

# For single GPU, 4-6 threads is recommended
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
# 

import sys
import argparse
import time
import random
import numpy as np
import torch

# Limit PyTorch CPU threads
torch.set_num_threads(4)
#

from typing import Dict, Any, List, Tuple, Optional


# Add changeformer to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set seeds for reproducibility
def set_seeds(seed: int = 42):
    """Set seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA settings for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from changeformer.utils.config import load_model_config, load_config, get_available_models
from changeformer.utils.data_utils import match_image_pairs, get_image_files
from changeformer.utils.eval_utils import (
    evaluate_predictions, print_metrics, 
    evaluate_second_dataset, print_second_metrics
)
from changeformer.utils.registry_utils import ensure_clean_registry
from changeformer.models import build_pipeline

# Class to model configuration mapping
# Using OVCD configuration with DINOv3
CLASS_TO_MODEL_MAP = {
    'building': 'OVCD_second_building',
    'water': 'OVCD_second_water',
    'non_veg_ground_surface': 'OVCD_second_non_veg_ground_surface',
    'low_vegetation': 'OVCD_second_low_vegetation',
    'tree': 'OVCD_second_tree',
    'playground': 'OVCD_second_playground'
}

# Constants
DEFAULT_DATASET = 'second'
DEFAULT_DATA_ROOT = 'data/second'
DEFAULT_OUTPUT_DIR = 'outputs/second'
DEFAULT_THRESHOLD = 0.5


def get_available_datasets() -> List[str]:
    """Get list of available dataset configurations.
    
    Returns:
        List of available dataset names.
    """
    datasets_dir = os.path.join(os.path.dirname(__file__), 'configs', 'datasets')
    if not os.path.exists(datasets_dir):
        return []
    
    datasets = []
    for file in os.listdir(datasets_dir):
        if file.endswith('.yaml'):
            datasets.append(file[:-5])  # Remove .yaml extension
    
    return sorted(datasets)


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load dataset configuration by name."""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'datasets', f'{dataset_name}.yaml')
    return load_config(config_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SECOND Dataset Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available classes:
{chr(10).join(f"  - {class_name}" for class_name in CLASS_TO_MODEL_MAP.keys())}

Examples:
  # Single class evaluation
  python evaluate_second.py --class building --output_dir outputs/test_HQ/second/building --save_predictions --verbose
  
  # Evaluate all 6 classes
  python evaluate_second.py --class all --output_dir outputs/test_HQ/second/all_classes --save_predictions --verbose
  
  # With custom parameters
  python evaluate_second.py --class water --points_per_side 16 --pred_iou_thresh 0.5 --output_dir outputs/test_HQ/second/water
  
  # ACF parameter tuning examples
  python evaluate_second.py --class water --filter_strength 0.8 --min_region_size 100 --apr_debug
        """
    )
    
    parser.add_argument(
        "--class", "-c", dest='class_name',
        type=str,
        required=True,
        choices=list(CLASS_TO_MODEL_MAP.keys()) + ['all'],
        help="Class to evaluate or 'all' for all classes"
    )
    
    
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of the SECOND dataset"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save predictions "
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Evaluation threshold (default: {DEFAULT_THRESHOLD})"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on (cuda/cpu, overrides config)"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction masks (for analysis)"
    )
    
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Only run inference without evaluation (useful for generating predictions)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Model parameter overrides
    parser.add_argument(
        "--points_per_side",
        type=int,
        help="SAM points per side (overrides config, e.g., 32)"
    )
    
    parser.add_argument(
        "--pred_iou_thresh",
        type=float,
        help="SAM prediction IoU threshold (overrides config, e.g., 0.5)"
    )
    
    parser.add_argument(
        "--mask_threshold",
        type=float,
        help="SAM mask binarization threshold (overrides config, e.g., 0.7)"
    )
    
    parser.add_argument(
        "--stability_score_thresh",
        type=float,
        help="SAM stability score threshold (overrides config, e.g., 0.95)"
    )
    
    parser.add_argument(
        "--stability_score_offset",
        type=float,
        help="SAM stability score offset (overrides config, e.g., 0.9)"
    )
    
    parser.add_argument(
        "--change_confidence_threshold",
        type=int,
        help="DINOv2 change confidence threshold (overrides config, e.g., 145)"
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Identifier confidence threshold (overrides config, e.g., 0.5)"
    )
    
    
    parser.add_argument(
        "--max_change_ratio",
        type=float,
        help="Maximum change ratio for lightweight illumination preprocessing (0.0-1.0, overrides config, e.g., 0.06)"
    )
    
    parser.add_argument(
        "--box_nms_thresh",
        type=float,
        help="SAM box NMS threshold (overrides config, e.g., 0.7)"
    )
    
    parser.add_argument(
        "--debug_adaptive_threshold",
        action="store_true",
        help="Enable debug output for adaptive threshold adjustment"
    )
    
    # Additional SAM parameters
    parser.add_argument(
        "--min_mask_region_area",
        type=int,
        help="SAM minimum mask region area (overrides config, e.g., 500)"
    )
    
    parser.add_argument(
        "--points_per_batch",
        type=int,
        help="SAM points per batch (overrides config, e.g., 64)"
    )
    
    parser.add_argument(
        "--area_thresh",
        type=float,
        help="SAM area threshold (overrides config, e.g., 0.75)"
    )
    
    # Otsu adaptive threshold parameters
    parser.add_argument(
        "--global_weight",
        type=float,
        help="Otsu global weight (overrides config, e.g., 0.70)"
    )
    
    parser.add_argument(
        "--edge_weight",
        type=float,
        help="Otsu edge weight (overrides config, e.g., 0.30)"
    )
    
    parser.add_argument(
        "--min_angle",
        type=int,
        help="Otsu minimum angle threshold (overrides config, e.g., 120)"
    )
    
    parser.add_argument(
        "--max_angle",
        type=int,
        help="Otsu maximum angle threshold (overrides config, e.g., 160)"
    )
    
    # Canny edge detection parameters
    parser.add_argument(
        "--canny_low",
        type=int,
        help="Canny edge detection low threshold (overrides config, e.g., 30-70)"
    )
    
    parser.add_argument(
        "--canny_high",
        type=int,
        help="Canny edge detection high threshold (overrides config, e.g., 100-150)"
    )
    
    parser.add_argument(
        "--min_edge_pixels",
        type=int,
        help="Minimum edge pixels required for edge-based threshold (overrides config, e.g., 50-100)"
    )
    
    # Adaptive radiometric alignment parameters (backward compatible)
    parser.add_argument(
        "--enable_adaptive_radiometric_alignment",
        action="store_true",
        help="Enable adaptive radiometric alignment preprocessing"
    )
    
    parser.add_argument(
        "--disable_adaptive_radiometric_alignment",
        action="store_true",
        help="Disable adaptive radiometric alignment preprocessing"
    )
    
    # Legacy parameters for backward compatibility
    parser.add_argument(
        "--enable_adaptive_illumination_alignment",
        action="store_true",
        help="Enable adaptive radiometric alignment preprocessing (legacy parameter)"
    )
    
    parser.add_argument(
        "--disable_adaptive_illumination_alignment",
        action="store_true",
        help="Disable adaptive radiometric alignment preprocessing (legacy parameter)"
    )
    
    # APR (Adaptive Post-processing Refinement) parameters
    parser.add_argument(
        "--filter_strength",
        type=float,
        help="APR confidence filtering strength (0.0-1.0, overrides config, e.g., 0.8)"
    )
    
    parser.add_argument(
        "--min_region_size",
        type=int,
        help="APR minimum region size for confidence filtering (overrides config, e.g., 100)"
    )
    

    
    parser.add_argument(
        "--apr_debug",
        action="store_true",
        help="Enable APR debug output for detailed post-processing information"
    )
    
    
    return parser.parse_args()


def get_image_pairs(dataset_config: Dict[str, Any], data_root: str):
    """Get image pairs based on dataset configuration."""
    data_paths = dataset_config['data_paths']
    
    if dataset_config['dataset_name'] == 'second':
        # Special handling for SECOND dataset
        img1_dir = os.path.join(data_root, data_paths['test_im1'])
        img2_dir = os.path.join(data_root, data_paths['test_im2'])
        label1_dir = os.path.join(data_root, data_paths['test_label1'])
        label2_dir = os.path.join(data_root, data_paths['test_label2'])
        
        pairs = match_image_pairs(img1_dir, img2_dir)
        return pairs, label1_dir, label2_dir  # Return both label dirs for SECOND
        
    else:
        # Standard dataset structure
        img1_dir = os.path.join(data_root, data_paths['test_A'])
        img2_dir = os.path.join(data_root, data_paths['test_B']) 
        gt_dir = os.path.join(data_root, data_paths['test_label'])
        
        pairs = match_image_pairs(img1_dir, img2_dir)
        return pairs, gt_dir, None


def run_second_evaluation(args, model_config, dataset_config, image_pairs):
    """Run evaluation specifically for SECOND dataset."""
    print("Running SECOND dataset multi-class evaluation...")
    
    # Get paths
    gt1_dir = os.path.join(args.data_root, dataset_config['data_paths']['test_label1'])
    gt2_dir = os.path.join(args.data_root, dataset_config['data_paths']['test_label2'])
    
    # Create output directory for multi-class predictions
    pred_base_dir = os.path.join(args.output_dir, f"{args.model}_{args.dataset}")
    os.makedirs(pred_base_dir, exist_ok=True)
    
    # Clear any previous model registrations to avoid conflicts
    ensure_clean_registry()
    
    # Initialize pipeline
    print("Initializing model...")
    start_time = time.time()
    pipeline = build_pipeline(model_config)
    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f}s")
    
    # Run multi-class inference
    print("Running multi-class inference...")
    start_time = time.time()
    
    success_count = 0
    for i, (img1_path, img2_path) in enumerate(image_pairs):
        if args.verbose:
            print(f"Processing pair {i+1}/{len(image_pairs)}: {os.path.basename(img1_path)}")
        
        try:
            # Generate class-specific predictions
            class_results = pipeline.predict_second_multiclass(
                img1_path, img2_path, pred_base_dir, verbose=False
            )
            success_count += 1
            
        except Exception as e:
            if args.verbose:
                print(f"  Error: {e}")
            # Clear GPU memory if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore if torch not available
            continue
    
    inference_time = time.time() - start_time
    print(f"Inference completed: {success_count}/{len(image_pairs)} successful")
    print(f"Inference time: {inference_time:.2f}s ({inference_time/len(image_pairs):.2f}s per pair)")
    
    if args.inference_only:
        print(f"Inference-only mode. Predictions saved to: {pred_base_dir}")
        return
    
    # Evaluate using SECOND-specific evaluation
    print("Evaluating multi-class predictions...")
    eval_results = evaluate_second_dataset(
        pred_base_dir=pred_base_dir,
        gt1_dir=gt1_dir,
        gt2_dir=gt2_dir,
        threshold=args.threshold
    )
    
    # Print results
    print_second_metrics(eval_results, f"{args.model} on {args.dataset}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset} (multi-class)\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Initialization time: {init_time:.2f}s\n")
        f.write(f"Inference time: {inference_time:.2f}s\n")
        f.write(f"Time per image: {inference_time/len(image_pairs):.2f}s\n")
        f.write("-" * 50 + "\n")
        
        # Write individual class results
        f.write("Individual Class Results:\n")
        for class_name, results in eval_results.items():
            if class_name != 'overall':
                f.write(f"\n{class_name.upper()}:\n")
                for metric, value in results.items():
                    if metric != 'processed_images':
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write(f"  processed_images: {results.get('processed_images', 0)}\n")
        
        # Write overall results
        if 'overall' in eval_results:
            f.write(f"\nOverall Metrics (Average):\n")
            for metric, value in eval_results['overall'].items():
                f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    
    # Cleanup predictions if not requested to save
    if not args.save_predictions:
        try:
            import shutil
            if os.path.exists(pred_base_dir):
                shutil.rmtree(pred_base_dir)
                print("Temporary predictions cleaned up")
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to cleanup temporary predictions: {e}")
        except Exception as e:
            print(f"Unexpected error during cleanup: {e}")
    else:
        print(f"Predictions saved to: {pred_base_dir}")


def run_standard_evaluation(args, model_config, dataset_config, image_pairs, gt_dir):
    """Run evaluation for standard datasets (binary change detection)."""
    print("Running standard binary change detection evaluation...")
    
    # Create output directory for predictions
    pred_dir = os.path.join(args.output_dir, f"{args.model}_{args.dataset}")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Clear any previous model registrations to avoid conflicts
    ensure_clean_registry()
    
    # Initialize pipeline
    print("Initializing model...")
    start_time = time.time()
    pipeline = build_pipeline(model_config)
    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f}s")
    
    # Run inference on all image pairs
    print("Running inference...")
    start_time = time.time()
    
    # Add progress bar wrapper around batch_detect
    from tqdm import tqdm
    
    # Monkey patch the batch_detect to add progress bar
    original_batch_detect = pipeline.batch_detect
    
    def batch_detect_with_progress(image_pairs, output_dir, verbose=False):
        import os
        from skimage.io import imsave
        
        results = []
        progress_bar = tqdm(image_pairs, desc="Running inference", unit="pair")
        
        for i, (img1_path, img2_path) in enumerate(progress_bar):
            result = {'success': False, 'error': None}
            
            try:
                # Update progress bar
                progress_bar.set_postfix({
                    'current': os.path.basename(img1_path)[:15] + '...' if len(os.path.basename(img1_path)) > 15 else os.path.basename(img1_path),
                    'success': sum(1 for r in results if r['success'])
                })
                
                # Run prediction
                change_mask, metadata = pipeline.predict(img1_path, img2_path)
                
                # Save result
                output_filename = os.path.basename(img1_path)
                output_path = os.path.join(output_dir, output_filename)
                imsave(output_path, change_mask)
                
                result['success'] = True
                result['metadata'] = metadata
                
            except Exception as e:
                result['error'] = str(e)
                if verbose:
                    progress_bar.write(f"Error processing {os.path.basename(img1_path)}: {e}")
                # Clear GPU memory if using CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass  # Ignore if torch not available
            
            results.append(result)
        
        progress_bar.close()
        return results
    
    # Use the progress bar version
    results = batch_detect_with_progress(image_pairs, pred_dir, args.verbose)
    
    inference_time = time.time() - start_time
    
    # Count successful predictions
    successful = sum(1 for r in results if r['success'])
    print(f"Inference completed: {successful}/{len(image_pairs)} successful")
    print(f"Inference time: {inference_time:.2f}s ({inference_time/len(image_pairs):.2f}s per pair)")
    
    if successful == 0:
        raise ValueError("No successful predictions generated")
    
    if args.inference_only:
        print(f"Inference-only mode. Predictions saved to: {pred_dir}")
        return
    
    # Evaluate predictions
    print("Evaluating predictions...")
    eval_results = evaluate_predictions(
        pred_dir, 
        gt_dir, 
        threshold=args.threshold
    )
    
    # Print results
    print_metrics(eval_results, f"{args.model} on {args.dataset}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Number of images: {eval_results['num_images']}\n")
        f.write(f"Initialization time: {init_time:.2f}s\n")
        f.write(f"Inference time: {inference_time:.2f}s\n")
        f.write(f"Time per image: {inference_time/len(image_pairs):.2f}s\n")
        f.write("-" * 50 + "\n")
        for metric, value in eval_results.items():
            if metric != 'num_images':
                f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    
    # Cleanup predictions if not requested to save
    if not args.save_predictions:
        try:
            import shutil
            if os.path.exists(pred_dir):
                shutil.rmtree(pred_dir)
                print("Temporary predictions cleaned up")
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to cleanup temporary predictions: {e}")
        except Exception as e:
            print(f"Unexpected error during cleanup: {e}")
    else:
        print(f"Predictions saved to: {pred_dir}")


def apply_parameter_overrides(model_config, args):
    """Apply command line parameter overrides to model config."""
    try:
        if args.points_per_side is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['points_per_side'] = args.points_per_side
            print(f"Override: points_per_side = {args.points_per_side}")
        
        if args.pred_iou_thresh is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['pred_iou_thresh'] = args.pred_iou_thresh
            print(f"Override: pred_iou_thresh = {args.pred_iou_thresh}")
        
        if args.change_confidence_threshold is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            model_config['comparator']['change_confidence_threshold'] = args.change_confidence_threshold
            print(f"Override: change_confidence_threshold = {args.change_confidence_threshold}")
        
        if args.confidence_threshold is not None:
            if 'identifier' not in model_config:
                model_config['identifier'] = {}
            model_config['identifier']['confidence_threshold'] = args.confidence_threshold
            print(f"Override: confidence_threshold = {args.confidence_threshold}")
        
        if args.max_change_ratio is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'adaptive_radiometric_alignment_config' not in model_config['segmentor']:
                model_config['segmentor']['adaptive_radiometric_alignment_config'] = {}
            model_config['segmentor']['adaptive_radiometric_alignment_config']['max_change_ratio'] = args.max_change_ratio
            # Enable adaptive radiometric alignment when max_change_ratio is specified
            model_config['segmentor']['enable_adaptive_radiometric_alignment'] = True
            print(f"Override: max_change_ratio = {args.max_change_ratio}")
        
        # SAM parameters
        if args.stability_score_thresh is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['stability_score_thresh'] = args.stability_score_thresh
            print(f"Override: stability_score_thresh = {args.stability_score_thresh}")
        
        if args.stability_score_offset is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['stability_score_offset'] = args.stability_score_offset
            print(f"Override: stability_score_offset = {args.stability_score_offset}")
        
        if args.mask_threshold is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['mask_threshold'] = args.mask_threshold
            print(f"Override: mask_threshold = {args.mask_threshold}")
        
        if args.box_nms_thresh is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['box_nms_thresh'] = args.box_nms_thresh
            print(f"Override: box_nms_thresh = {args.box_nms_thresh}")
        
        if args.min_mask_region_area is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['min_mask_region_area'] = args.min_mask_region_area
            print(f"Override: min_mask_region_area = {args.min_mask_region_area}")
        
        if args.points_per_batch is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['points_per_batch'] = args.points_per_batch
            print(f"Override: points_per_batch = {args.points_per_batch}")
        
        if args.area_thresh is not None:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            if 'params' not in model_config['segmentor']:
                model_config['segmentor']['params'] = {}
            model_config['segmentor']['params']['area_thresh'] = args.area_thresh
            print(f"Override: area_thresh = {args.area_thresh}")
        
        # Adaptive Change Thresholding (ACT) parameters
        if args.global_weight is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['global_weight'] = args.global_weight
            print(f"Override: global_weight = {args.global_weight}")
        
        if args.edge_weight is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['edge_weight'] = args.edge_weight
            print(f"Override: edge_weight = {args.edge_weight}")
        
        if args.min_angle is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['min_angle'] = args.min_angle
            print(f"Override: min_angle = {args.min_angle}")
        
        if args.max_angle is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['max_angle'] = args.max_angle
            print(f"Override: max_angle = {args.max_angle}")
        
        # Canny edge detection parameters
        if args.canny_low is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['canny_low'] = args.canny_low
            print(f"Override: canny_low = {args.canny_low}")
        
        if args.canny_high is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['canny_high'] = args.canny_high
            print(f"Override: canny_high = {args.canny_high}")
        
        if args.min_edge_pixels is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['min_edge_pixels'] = args.min_edge_pixels
            print(f"Override: min_edge_pixels = {args.min_edge_pixels}")
        
        # Additional parameters
        if args.debug_adaptive_threshold:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                model_config['comparator']['adaptive_change_thresholding_config'] = {}
            model_config['comparator']['adaptive_change_thresholding_config']['debug'] = True
            print(f"Override: debug_adaptive_threshold = True")
        
        if args.enable_adaptive_radiometric_alignment or args.enable_adaptive_illumination_alignment:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            model_config['segmentor']['enable_adaptive_radiometric_alignment'] = True
            print(f"Override: enable_adaptive_radiometric_alignment = True")
        
        if args.disable_adaptive_radiometric_alignment or args.disable_adaptive_illumination_alignment:
            if 'segmentor' not in model_config:
                model_config['segmentor'] = {}
            model_config['segmentor']['enable_adaptive_radiometric_alignment'] = False
            print(f"Override: enable_adaptive_radiometric_alignment = False")
        
        # APR (Adaptive Post-processing Refinement) parameters
        if args.filter_strength is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'enhancement_modules' not in model_config['comparator']:
                model_config['comparator']['enhancement_modules'] = {}
            if 'confidence_filtering' not in model_config['comparator']['enhancement_modules']:
                model_config['comparator']['enhancement_modules']['confidence_filtering'] = {'enabled': True, 'config': {}}
            
            model_config['comparator']['enhancement_modules']['confidence_filtering']['config']['filter_strength'] = args.filter_strength
            print(f"Override: filter_strength = {args.filter_strength}")
        
        if args.min_region_size is not None:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'enhancement_modules' not in model_config['comparator']:
                model_config['comparator']['enhancement_modules'] = {}
            if 'confidence_filtering' not in model_config['comparator']['enhancement_modules']:
                model_config['comparator']['enhancement_modules']['confidence_filtering'] = {'enabled': True, 'config': {}}
            
            model_config['comparator']['enhancement_modules']['confidence_filtering']['config']['min_region_size'] = args.min_region_size
            print(f"Override: min_region_size = {args.min_region_size}")
        
        # Note: kernel_size and min_area parameters are deprecated since morphological_postprocessing was removed
        # if args.kernel_size is not None:
        #     print("Warning: --kernel_size is deprecated since morphological_postprocessing was removed")
        
        # if args.min_area is not None:
        #     print("Warning: --min_area is deprecated since morphological_postprocessing was removed")
        
        if args.apr_debug:
            if 'comparator' not in model_config:
                model_config['comparator'] = {}
            if 'enhancement_modules' not in model_config['comparator']:
                model_config['comparator']['enhancement_modules'] = {}
            
            # Enable debug for both modules
            if 'confidence_filtering' not in model_config['comparator']['enhancement_modules']:
                model_config['comparator']['enhancement_modules']['confidence_filtering'] = {'enabled': True, 'config': {}}
            # Note: morphological_postprocessing has been removed due to negative impact
            
            model_config['comparator']['enhancement_modules']['confidence_filtering']['config']['debug'] = True
            print(f"Override: apr_debug = True (confidence_filtering only)")
            
    except Exception as e:
        print(f"Error applying parameter overrides: {e}")
        raise


def run_single_class_evaluation(args, class_name):
    """Run evaluation for a single class."""
    try:
        # Get model configuration for this class
        model_name = CLASS_TO_MODEL_MAP[class_name]
        model_config = load_model_config(model_name)
        
        # Override device if specified
        if args.device:
            model_config['device'] = args.device
        
        # Apply parameter overrides
        apply_parameter_overrides(model_config, args)
        
        # Load dataset config
        dataset_config = load_dataset_config('second')
        
        # Get image pairs
        image_pairs, _, _ = get_image_pairs(dataset_config, args.data_root)
        
        if not image_pairs:
            raise ValueError(f"No image pairs found")
        
        # Use class-specific change labels instead of original semantic labels
        gt1_dir = os.path.join(args.data_root, f"label_{class_name}")
        gt2_dir = None  # For SECOND change detection, we only need the change labels
        
        print(f"Found {len(image_pairs)} image pairs")
        
        # Create output directory
        class_output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Clear registry and initialize pipeline
        ensure_clean_registry()
        print("Initializing model...")
        start_time = time.time()
        pipeline = build_pipeline(model_config)
        init_time = time.time() - start_time
        print(f"Model initialized in {init_time:.2f}s")
        
        # Run inference
        print("Running inference...")
        pred_dir = os.path.join(class_output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        start_time = time.time()
        from tqdm import tqdm
        
        success_count = 0
        for i, (img1_path, img2_path) in enumerate(tqdm(image_pairs, desc=f"Processing {class_name}")):
            try:
                change_mask, metadata = pipeline.predict(img1_path, img2_path)
                
                # Save prediction
                output_filename = os.path.basename(img1_path)
                output_path = os.path.join(pred_dir, output_filename)
                from skimage.io import imsave
                imsave(output_path, change_mask)
                
                success_count += 1
                
            except Exception as e:
                if args.verbose:
                    print(f"Error processing {os.path.basename(img1_path)}: {e}")
                # Clear GPU memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                continue
        
        inference_time = time.time() - start_time
        print(f"Inference completed: {success_count}/{len(image_pairs)} successful")
        
        if success_count == 0:
            print(f"ERROR: No successful predictions for {class_name}")
            return None
        
        if args.inference_only:
            print(f"Inference-only mode. Predictions saved to: {pred_dir}")
            return {'processing_stats': {'successful': success_count, 'processing_time': inference_time}}
        
        # Evaluate predictions for this specific class
        print(f"Evaluating {class_name} predictions...")
        from changeformer.utils.eval_utils import evaluate_second_class_simple
        
        results = evaluate_second_class_simple(
            pred_dir=pred_dir,
            change_label_dir=gt1_dir,  # gt1_dir points to label_building, label_water, etc.
            class_name=class_name,
            threshold=args.threshold
        )
        
        # Add processing stats
        results['processing_stats'] = {
            'successful': success_count,
            'processing_time': inference_time,
            'init_time': init_time
        }
        
        # Print results using standard format (consistent with evaluate.py)
        from changeformer.utils.eval_utils import print_metrics
        
        # Add missing fields for compatibility with print_metrics
        results['num_images'] = success_count
        results['threshold'] = args.threshold
        
        # Use standard print format
        print_metrics(results, f"{model_name} on {class_name}")
        
        # Save individual class results
        results_file = os.path.join(class_output_dir, f"{class_name}_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Class: {class_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write(f"Processed images: {success_count}\n")
            f.write(f"Processing time: {inference_time:.2f}s\n")
            f.write("-" * 50 + "\n")
            for metric, value in results.items():
                if metric != 'processing_stats':
                    f.write(f"{metric}: {value:.4f}\n")
        
        # Cleanup predictions if not requested to save
        if not args.save_predictions:
            try:
                import shutil
                if os.path.exists(pred_dir):
                    shutil.rmtree(pred_dir)
            except:
                pass
        
        return results
        
    except Exception as e:
        print(f"ERROR: Error evaluating {class_name}: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        return None


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Set seeds for reproducibility
    set_seeds(seed=42)
    
    print(f"ChangeFormer SECOND Evaluation - Class: {args.class_name}")
    
    try:
        if args.class_name == 'all':
            # Multi-class evaluation
            print(f"🎯 Multi-class evaluation mode: evaluating all {len(CLASS_TO_MODEL_MAP)} classes")
            print(f"Classes: {', '.join(CLASS_TO_MODEL_MAP.keys())}")
            
            all_results = {}
            
            for class_name in CLASS_TO_MODEL_MAP.keys():
                print(f"\n{'='*60}")
                print(f"🔍 Evaluating class: {class_name}")
                print(f"{'='*60}")
                
                try:
                    results = run_single_class_evaluation(args, class_name)
                    if results:
                        all_results[class_name] = results
                        print(f"SUCCESS: {class_name} evaluation completed successfully")
                    else:
                        print(f"ERROR: {class_name} evaluation failed")
                        
                except Exception as e:
                    print(f"ERROR: Error evaluating {class_name}: {str(e)}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # Print overall summary
            if all_results:
                print("\n📊 Overall Results Summary:")
                print("=" * 80)
                print(f"{'Class':<20} | {'F1':<8} | {'IoU_C':<8} | {'Precision':<10} | {'Recall':<8}")
                print("-" * 80)
                
                for class_name, results in all_results.items():
                    if 'f1_score_change' in results:
                        f1 = results.get('f1_score_change', 0)
                        iou_c = results.get('iou_change', 0)
                        precision = results.get('precision_change', 0)
                        recall = results.get('recall_change', 0)
                        
                        print(f"{class_name:<20} | {f1:<8.4f} | {iou_c:<8.4f} | {precision:<10.4f} | {recall:<8.4f}")
                
                # Calculate average metrics
                if len(all_results) > 1:
                    avg_metrics = {}
                    metric_names = ['f1_score_change', 'precision_change', 'recall_change', 'iou_change']
                    
                    for metric in metric_names:
                        values = [results[metric] for results in all_results.values() if metric in results]
                        if values:
                            avg_metrics[metric] = np.mean(values)
                    
                    print("-" * 80)
                    print(f"{'AVERAGE':<20} | {avg_metrics.get('f1_score_change', 0):<8.4f} | "
                          f"{avg_metrics.get('iou_change', 0):<8.4f} | "
                          f"{avg_metrics.get('precision_change', 0):<10.4f} | "
                          f"{avg_metrics.get('recall_change', 0):<8.4f}")
                
                # Save multi-class results
                results_file = os.path.join(args.output_dir, f"second_all_classes_results.txt")
                with open(results_file, 'w') as f:
                    f.write(f"Dataset: second (all classes)\n")
                    f.write(f"Threshold: {args.threshold}\n")
                    f.write(f"Number of classes: {len(CLASS_TO_MODEL_MAP)}\n")
                    f.write("-" * 50 + "\n")
                    
                    for class_name, results in all_results.items():
                        f.write(f"\n{class_name.upper()}:\n")
                        f.write(f"  model: {CLASS_TO_MODEL_MAP[class_name]}\n")
                        for metric, value in results.items():
                            if metric != 'processing_stats':
                                f.write(f"  {metric}: {value:.4f}\n")
                
                print(f"Results saved to: {results_file}")
            else:
                print("ERROR: No successful evaluations completed")
        
        else:
            # Single class evaluation
            print(f"Single class evaluation mode: {args.class_name}")
            results = run_single_class_evaluation(args, args.class_name)
            
            if results:
                print("SUCCESS: Evaluation completed successfully")
            else:
                print("ERROR: Evaluation failed")
                sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
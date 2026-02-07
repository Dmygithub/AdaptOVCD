import os
import sys
import argparse
import time
import random
import numpy as np
import torch
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
from changeformer.utils.auto_dataset_detector import auto_find_dataset
from changeformer.models import build_pipeline

# Constants
DEFAULT_MODEL = 'OVCD_levircd'  # Default OVCD config with DINOv3
DEFAULT_DATASET = 'levircd'
DEFAULT_DATA_ROOT = None  # Auto-detected based on dataset
DEFAULT_OUTPUT_DIR = 'outputs/test'
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


def get_dataset_root(dataset_name: str) -> str:
    """
    Get dataset root path, with automatic detection fallback.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset root path
    """
    # Try automatic detection first
    detected_path = auto_find_dataset(dataset_name)
    if detected_path:
        print(f"Auto-detected {dataset_name} dataset at: {detected_path}")
        return detected_path
    
    # Fallback to conventional path structure
    fallback_path = f"data/{dataset_name}"
    if os.path.exists(fallback_path):
        print(f"Using conventional path for {dataset_name}: {fallback_path}")
        return fallback_path
    
    # If nothing found, still return the conventional path (let it fail later with a clear error)
    print(f"Warning: Dataset {dataset_name} not found. Using fallback path: {fallback_path}")
    return fallback_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="ChangeFormer: Unified evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join(f"  - {model}" for model in get_available_models())}

Available datasets:
{chr(10).join(f"  - {dataset}" for dataset in get_available_datasets())}

Examples:
  python evaluate.py                                                    # Use defaults
  python evaluate.py --model sam2_dinov2_dgtrs --dataset levircd       # Override model
  python evaluate.py --dataset second --data_root /path/to/second_dataset --output_dir results/
  
  # ACF parameter tuning examples
  python evaluate.py --filter_strength 0.8 --min_region_size 100 --apr_debug
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        choices=get_available_models(),
        help="Model name to evaluate"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        choices=get_available_datasets(),
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of the dataset (auto-detected if not specified)"
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
        "--histogram_blend_ratio",
        type=float,
        help="Histogram blending ratio for image preprocessing (0.0-1.0, overrides config, e.g., 0.9)"
    )
    
    parser.add_argument(
        "--max_change_ratio",
        type=float,
        help="Maximum change ratio for radiometric preprocessing (0.0-1.0, overrides config, e.g., 0.06)"
    )
    
    parser.add_argument(
        "--box_nms_thresh",
        type=float,
        help="SAM box NMS threshold (overrides config, e.g., 0.7)"
    )
    
    parser.add_argument(
        "--debug_adaptive_threshold",
        action="store_true",
        default=None,  # None means don't override config file
        help="Enable debug output for adaptive threshold adjustment (overrides config file if specified)"
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
    
    # Adaptive radiometric alignment parameters (backward compatible with illumination_alignment)
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
    
    # DGTRS identifier parameters
    parser.add_argument(
        "--dgtrs_confidence_threshold",
        type=float,
        help="DGTRS identifier confidence threshold (overrides config, e.g., 0.5)"
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


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Set seeds for reproducibility
    set_seeds(seed=42)
    
    print(f"ChangeFormer Evaluation - Model: {args.model}, Dataset: {args.dataset}")
    
    try:
        # Load configurations
        model_config = load_model_config(args.model)
        dataset_config = load_dataset_config(args.dataset)
        
        # Auto-detect dataset root if not specified
        if args.data_root is None:
            args.data_root = get_dataset_root(args.dataset)
        
        # Override device if specified
        if args.device:
            model_config['device'] = args.device
        
        # Override model parameters if specified
        try:
            if args.points_per_side is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    print("Warning: 'segmentor' not found in config")
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    print("Warning: 'params' not found in segmentor config, creating it")
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['points_per_side'] = args.points_per_side
                print(f"Override: points_per_side = {args.points_per_side}")
            
            if args.pred_iou_thresh is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['pred_iou_thresh'] = args.pred_iou_thresh
                print(f"Override: pred_iou_thresh = {args.pred_iou_thresh}")
            
            if args.mask_threshold is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['mask_threshold'] = args.mask_threshold
                print(f"Override: mask_threshold = {args.mask_threshold}")
            
            if args.stability_score_thresh is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['stability_score_thresh'] = args.stability_score_thresh
                print(f"Override: stability_score_thresh = {args.stability_score_thresh}")
            
            if args.stability_score_offset is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['stability_score_offset'] = args.stability_score_offset
                print(f"Override: stability_score_offset = {args.stability_score_offset}")
            
            if args.change_confidence_threshold is not None:
                if 'comparator' not in model_config:
                    print("Warning: 'comparator' not found in config")
                    model_config['comparator'] = {}
                
                model_config['comparator']['change_confidence_threshold'] = args.change_confidence_threshold
                print(f"Override: change_confidence_threshold = {args.change_confidence_threshold}")
            
            if args.confidence_threshold is not None:
                if 'identifier' not in model_config:
                    print("Warning: 'identifier' not found in config")
                    model_config['identifier'] = {}
                
                model_config['identifier']['confidence_threshold'] = args.confidence_threshold
                print(f"Override: confidence_threshold = {args.confidence_threshold}")
            
            if args.histogram_blend_ratio is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['histogram_blend_ratio'] = args.histogram_blend_ratio
                # Enable histogram matching when blend ratio is specified
                model_config['segmentor']['params']['match_hist'] = True
                print(f"Override: histogram_blend_ratio = {args.histogram_blend_ratio}")
                print("Override: match_hist = True (automatically enabled)")
            
            if args.max_change_ratio is not None:
                # Apply to adaptive radiometric alignment config
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'adaptive_radiometric_alignment_config' not in model_config['segmentor']:
                    model_config['segmentor']['adaptive_radiometric_alignment_config'] = {}
                
                model_config['segmentor']['adaptive_radiometric_alignment_config']['max_change_ratio'] = args.max_change_ratio
                # Enable adaptive radiometric alignment when max_change_ratio is specified
                model_config['segmentor']['enable_adaptive_radiometric_alignment'] = True
                print(f"Override: max_change_ratio = {args.max_change_ratio}")
                print("Override: enable_adaptive_radiometric_alignment = True (automatically enabled)")
            
            if args.box_nms_thresh is not None:
                # Ensure segmentor and params exist
                if 'segmentor' not in model_config:
                    model_config['segmentor'] = {}
                if 'params' not in model_config['segmentor']:
                    model_config['segmentor']['params'] = {}
                
                model_config['segmentor']['params']['box_nms_thresh'] = args.box_nms_thresh
                print(f"Override: box_nms_thresh = {args.box_nms_thresh}")
            
            # Only override config if explicitly specified via command line
            # Note: store_true args default to False when not specified
            if hasattr(args, 'debug_adaptive_threshold') and args.debug_adaptive_threshold:
                # Enable debug output for adaptive threshold module
                if 'comparator' not in model_config:
                    model_config['comparator'] = {}
                if 'adaptive_change_thresholding_config' not in model_config['comparator']:
                    model_config['comparator']['adaptive_change_thresholding_config'] = {}
                
                model_config['comparator']['adaptive_change_thresholding_config']['debug'] = True
                print(f"Override: debug_adaptive_threshold = True")
            
            # Additional SAM parameters
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
            
            # Otsu adaptive threshold parameters
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
            
            # Adaptive radiometric alignment control (with backward compatibility)
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
            
            # DGTRS identifier parameters
            if args.dgtrs_confidence_threshold is not None:
                if 'identifier' not in model_config:
                    model_config['identifier'] = {}
                
                model_config['identifier']['confidence_threshold'] = args.dgtrs_confidence_threshold
                print(f"Override: dgtrs_confidence_threshold = {args.dgtrs_confidence_threshold}")
            
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
            
            # Multi-scale attention control removed
                
        except Exception as e:
            print(f"Error applying parameter overrides: {e}")
            print("Config structure:")
            print(f"  model_config keys: {list(model_config.keys()) if isinstance(model_config, dict) else 'Not a dict'}")
            if isinstance(model_config, dict) and 'segmentor' in model_config:
                print(f"  segmentor keys: {list(model_config['segmentor'].keys())}")
            raise
        
        if args.verbose:
            print(f"Model: {model_config['description']}")
            print(f"Dataset: {dataset_config['description']}")
            print(f"Device: {model_config['device']}")
            print(f"Final config - points_per_side: {model_config['segmentor']['params']['points_per_side']}")
            print(f"Final config - change_confidence_threshold: {model_config['comparator']['change_confidence_threshold']}")
        
        # Get image pairs
        print("Loading dataset...")
        image_pairs, gt_dir_or_label1, gt_dir_label2 = get_image_pairs(dataset_config, args.data_root)
        
        if not image_pairs:
            raise ValueError(f"No image pairs found in dataset {args.dataset}")
        
        print(f"Found {len(image_pairs)} image pairs")
        
        # Route to appropriate evaluation method
        if args.dataset == 'second':
            run_second_evaluation(args, model_config, dataset_config, image_pairs)
        else:
            run_standard_evaluation(args, model_config, dataset_config, image_pairs, gt_dir_or_label1)
            
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
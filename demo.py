# python demo.py --model OVCD_levircd --input1 demo_images/A/test_115.png --input2 demo_images/B/test_115.png --output outputs/demo
import os
import sys
import argparse
import time
from typing import Optional, Dict, Any

from skimage.io import imsave

# Add changeformer to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from changeformer.utils.config import load_model_config, get_available_models
from changeformer.utils.data_utils import ensure_output_directory, load_image_pair, validate_image_pair
from changeformer.utils.registry_utils import ensure_clean_registry
from changeformer.models import build_pipeline

# Constants
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
OUTPUT_DIR = "outputs/demo"
DEFAULT_MODEL = 'demo'
INPUT1='demo_images/A/test_115.png'
INPUT2='demo_images/B/test_115.png'

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="ChangeFormer: Unified change detection demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join(f"  - {model}" for model in get_available_models())}

Examples:
  python demo.py --model sam2_dinov2_dgtrs --input1 /path/to/image1.png --input2 /path/to/image2.png
  python demo.py --model sam_dinov2_segearth --input1 img1.jpg --input2 img2.jpg --output results/
        """
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default=DEFAULT_MODEL,
        choices=get_available_models(),
        help="Model name to use for change detection"
    )

    
    parser.add_argument(
        "--input1", 
        type=str, 
        default=INPUT1,
        help="Path to the first input image"
    )
    
    parser.add_argument(
        "--input2", 
        type=str, 
        default=INPUT2,
        help="Path to the second input image"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str, 
        default=OUTPUT_DIR,
        help=f"Output directory or file path "
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Custom configuration file (optional, overrides model defaults)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on (cuda/cpu, overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # SAM parameter overrides
    parser.add_argument(
        "--mask_threshold",
        type=float,
        help="SAM mask binarization threshold (e.g., 0.7 for tighter boundaries)"
    )
    
    parser.add_argument(
        "--stability_score_thresh",
        type=float,
        help="SAM stability score threshold (e.g., 0.95)"
    )
    
    parser.add_argument(
        "--stability_score_offset",
        type=float,
        help="SAM stability score offset (e.g., 0.9)"
    )
    
    parser.add_argument(
        "--debug_adaptive_threshold",
        action="store_true",
        help="Enable debug output for adaptive threshold adjustment"
    )
    
    # Additional parameters for fine-tuning
    parser.add_argument(
        "--change_confidence_threshold",
        type=int,
        help="DINOv2 change confidence threshold (e.g., 140)"
    )
    
    parser.add_argument(
        "--max_change_ratio",
        type=float,
        help="Maximum change ratio for lightweight illumination (e.g., 0.055)"
    )
    
    parser.add_argument(
        "--points_per_side",
        type=int,
        help="SAM points per side (e.g., 34)"
    )
    
    parser.add_argument(
        "--pred_iou_thresh",
        type=float,
        help="SAM prediction IoU threshold (e.g., 0.87)"
    )
    
    return parser.parse_args()


def load_and_validate_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and validate model configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If configuration is invalid
    """
    # Load configuration
    if args.config:
        print(f"Loading custom config: {args.config}")
        config = load_model_config(args.config)
    else:
        print(f"Loading model config: {args.model}")
        config = load_model_config(args.model)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Ensure component keys exist
    if 'segmentor' not in config:
        config['segmentor'] = {}
    if 'params' not in config['segmentor']:
        config['segmentor']['params'] = {}
    if 'comparator' not in config:
        config['comparator'] = {}
    if 'identifier' not in config:
        config['identifier'] = {}
    
    # Override SAM parameters if specified
    if args.mask_threshold is not None:
        config['segmentor']['params']['mask_threshold'] = args.mask_threshold
        print(f"Override: mask_threshold = {args.mask_threshold}")
    
    if args.stability_score_thresh is not None:
        config['segmentor']['params']['stability_score_thresh'] = args.stability_score_thresh
        print(f"Override: stability_score_thresh = {args.stability_score_thresh}")
    
    if args.stability_score_offset is not None:
        config['segmentor']['params']['stability_score_offset'] = args.stability_score_offset
        print(f"Override: stability_score_offset = {args.stability_score_offset}")
    
    # Only override debug flag when explicitly specified via command line
    if hasattr(args, 'debug_adaptive_threshold') and args.debug_adaptive_threshold:
        if 'adaptive_change_thresholding_config' not in config['comparator']:
            config['comparator']['adaptive_change_thresholding_config'] = {}
        config['comparator']['adaptive_change_thresholding_config']['debug'] = True
        print(f"Override: debug_adaptive_threshold = True")
    
    # Additional parameter overrides
    if args.change_confidence_threshold is not None:
        config['comparator']['change_confidence_threshold'] = args.change_confidence_threshold
        print(f"Override: change_confidence_threshold = {args.change_confidence_threshold}")
    
    if args.max_change_ratio is not None:
        if 'adaptive_radiometric_alignment_config' not in config['segmentor']:
            config['segmentor']['adaptive_radiometric_alignment_config'] = {}
        config['segmentor']['adaptive_radiometric_alignment_config']['max_change_ratio'] = args.max_change_ratio
        config['segmentor']['enable_adaptive_radiometric_alignment'] = True
        print(f"Override: max_change_ratio = {args.max_change_ratio}")
        print("Override: enable_adaptive_radiometric_alignment = True (automatically enabled)")
    
    if args.points_per_side is not None:
        config['segmentor']['params']['points_per_side'] = args.points_per_side
        print(f"Override: points_per_side = {args.points_per_side}")
    
    if args.pred_iou_thresh is not None:
        config['segmentor']['params']['pred_iou_thresh'] = args.pred_iou_thresh
        print(f"Override: pred_iou_thresh = {args.pred_iou_thresh}")
    
    return config


def prepare_output_path(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """Prepare output file path.
    
    Args:
        args: Command line arguments
        config: Model configuration
        
    Returns:
        Output file path
    """
    if os.path.isdir(args.output) or not args.output.endswith(SUPPORTED_IMAGE_EXTENSIONS):
        # Output is directory, generate filename
        input_name = os.path.splitext(os.path.basename(args.input1))[0]
        output_filename = f"{config.get('output_prefix', args.model)}_{input_name}.png"
        output_path = os.path.join(args.output, output_filename)
    else:
        # Output is file path
        output_path = args.output
        
    return ensure_output_directory(output_path)


def main() -> None:
    """Main demo function."""
    args = parse_arguments()
    
    print(f"ChangeFormer Demo - Model: {args.model}")
    print(f"Input images: {args.input1}, {args.input2}")
    
    try:
        # Load and validate configuration
        config = load_and_validate_config(args)
        
        if args.verbose:
            print(f"Using device: {config['device']}")
            print(f"Model components: {config['segmentor']['type']} + {config['comparator']['type']} + {config['identifier']['type']}")
        
        # Validate input images
        print("Validating input images...")
        img1, img2 = load_image_pair(args.input1, args.input2)
        validate_image_pair(img1, img2)
        print(f"Images loaded: {img1.shape} and {img2.shape}")
        
        # Clear any previous model registrations to avoid conflicts
        ensure_clean_registry()
        
        # Initialize pipeline
        print("Initializing model...")
        start_time = time.time()
        pipeline = build_pipeline(config)
        init_time = time.time() - start_time
        print(f"Model initialized in {init_time:.2f}s")
        
        # Run change detection
        print("Running change detection...")
        start_time = time.time()
        
        change_mask, metadata = pipeline.predict(args.input1, args.input2)
        detection_time = time.time() - start_time
        
        # Prepare output path and save result
        output_path = prepare_output_path(args, config)
        imsave(output_path, change_mask)
        
        # Print summary
        print(f"\n========== Detection Complete ==========")
        print(f"Detection time: {detection_time:.2f}s")
        print(f"Total time: {init_time + detection_time:.2f}s")
        print(f"Result saved to: {output_path}")
        
        if args.verbose and metadata:
            print(f"\nDetection metadata:")
            for key, value in metadata.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
                    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check that all required files exist and paths are correct.")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your model configuration and input parameters.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("Please check the error details above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
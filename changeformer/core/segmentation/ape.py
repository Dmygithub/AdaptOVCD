"""
APE (Any Pointing and Exploring) segmentation core module.

This module provides the core APE functionality.
"""

import sys
import os
from typing import List, Dict, Any
import multiprocessing as mp
from detectron2.config import LazyConfig
from detectron2.data.detection_utils import read_image

# Import third-party APE components
from changeformer.third_party.ape_adapter import get_ape_demo_class


def build_ape_model(
    config_file: str,
    confidence_threshold: float = 0.5,
    opt: List[str] = [],
):
    """
    Build APE model for inference.

    Args:
        config_file (str): Path to the configuration file.
        confidence_threshold (float): Minimum score for instance predictions.
        opt (List[str]): Additional command-line options for the configuration.

    Returns:
        APE demo instance for inference.
    """
    # Load config from file and command-line arguments
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opt)
    
    # Set confidence threshold
    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = confidence_threshold
    else:
        cfg.model.test_score_thresh = confidence_threshold
    
    # Initialize multiprocessing
    mp.set_start_method("spawn", force=True)
    
    # Create APE demo instance using adapter
    VisualizationDemo = get_ape_demo_class()
    demo = VisualizationDemo(cfg)
    return demo


def extract_ape_prediction(
    model,
    input_path: str,
    text_prompt: str = None,
    with_box: bool = False,
    with_mask: bool = False,
    with_sseg: bool = False,
) -> Dict[str, Any]:
    """
    Extract predictions from APE model for a single image.

    Args:
        model: APE model instance.
        input_path (str): Path to the input image.
        text_prompt (str): Text prompt for the model.
        with_box (bool): Whether to include bounding boxes in output.
        with_mask (bool): Whether to include masks in output.
        with_sseg (bool): Whether to include semantic segmentation in output.

    Returns:
        Dict containing model predictions.
    """
    
    # Read and process the input image
    try:
        img = read_image(input_path, format="BGR")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return {}

    # Run inference
    predictions, visualized_output, visualized_outputs, metadata = model.run_on_image(
        img,
        text_prompt=text_prompt,
        with_box=with_box,
        with_mask=with_mask,
        with_sseg=with_sseg,
    )

    return predictions 
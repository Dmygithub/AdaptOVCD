"""
Evaluation utilities for ChangeFormer.

This module provides metrics calculation and evaluation functionality.
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm


class ChangeDetectionMetrics:
    """
    Change Detection Evaluation Metric Calculator.
    
    This class maintains the exact same logic as the original evaluation
    code to ensure consistent results.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize metric calculator.
        
        Args:
            threshold: Binarization threshold (0-1 scale), default 0.5
        """
        self.threshold = threshold * 255.0  # Convert to pixel value
        self.eps = 1e-7  # Numerical stability constant
        
        # Initialize accumulators
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.results = {}

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.results = {}

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        """
        Update metric calculations with new prediction and ground truth pair.
        
        Args:
            pred_mask: Predicted change mask (H, W)
            gt_mask: Ground truth change mask (H, W)
        """
        # Ensure same dimensions
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
        
        # Binarize masks
        pred_binary = (pred_mask > self.threshold).astype(np.uint8)
        gt_binary = (gt_mask > self.threshold).astype(np.uint8)
        
        # Calculate confusion matrix components
        self.tp += np.sum((pred_binary == 1) & (gt_binary == 1))
        self.tn += np.sum((pred_binary == 0) & (gt_binary == 0))
        self.fp += np.sum((pred_binary == 1) & (gt_binary == 0))
        self.fn += np.sum((pred_binary == 0) & (gt_binary == 1))

    def compute(self) -> Dict[str, float]:
        """
        Compute and return all evaluation metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        # Overall Accuracy (OA)
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        
        # Precision
        precision = self.tp / (self.tp + self.fp + self.eps)
        
        # Recall (Sensitivity)
        recall = self.tp / (self.tp + self.fn + self.eps)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        
        # Specificity
        specificity = self.tn / (self.tn + self.fp + self.eps)
        
        # mIoU (mean Intersection over Union)
        # Background IoU
        bg_iou = self.tn / (self.tn + self.fp + self.fn + self.eps)
        # Change IoU  
        change_iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        miou = (bg_iou + change_iou) / 2.0
        
        self.results = {
            'oa': oa,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'miou': miou,
            'bg_iou': bg_iou,
            'change_iou': change_iou,
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn
        }
        
        return self.results


def evaluate_predictions(pred_dir: str, gt_dir: str, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        threshold: Binarization threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = ChangeDetectionMetrics(threshold)
    
    # Get all prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    
    if not pred_files:
        raise ValueError(f"No prediction files found in {pred_dir}")
    
    processed = 0
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        gt_file = pred_file  # Assume same filename
        
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {pred_file}")
            continue
        
        try:
            # Load masks
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if pred_mask is None or gt_mask is None:
                print(f"Warning: Failed to load {pred_file}")
                continue
            
            # Update metrics
            metrics.update(pred_mask, gt_mask)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            continue
    
    if processed == 0:
        raise ValueError("No valid image pairs were processed")
    
    # Compute final metrics
    results = metrics.compute()
    results['num_images'] = processed
    
    return results


def evaluate_second_class(pred_dir: str, gt1_dir: str, gt2_dir: str, 
                         class_name: str, class_id: int, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate change detection for a specific class in SECOND dataset.
    
    Args:
        pred_dir: Directory containing prediction masks for this class
        gt1_dir: Directory containing first time point labels
        gt2_dir: Directory containing second time point labels
        class_name: Name of the class being evaluated
        class_id: ID of the class in the label maps
        threshold: Binarization threshold
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if not os.path.exists(pred_dir):
        print(f"Warning: Prediction directory not found: {pred_dir}")
        return {}
    
    # Initialize metric calculator
    metrics = ChangeDetectionMetrics(threshold=threshold)
    
    # Get list of images
    image_list = sorted(os.listdir(pred_dir))
    processed_count = 0
    
    for filename in tqdm(image_list, desc=f'Evaluating {class_name}'):
        try:
            # Load data
            pred_path = os.path.join(pred_dir, filename)
            label1_path = os.path.join(gt1_dir, filename)
            label2_path = os.path.join(gt2_dir, filename)

            # Check if all files exist
            if not all(os.path.exists(path) for path in [pred_path, label1_path, label2_path]):
                continue

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            label1 = cv2.imread(label1_path, cv2.IMREAD_ANYCOLOR)
            label2 = cv2.imread(label2_path, cv2.IMREAD_ANYCOLOR)

            # Validate inputs
            if pred is None or label1 is None or label2 is None:
                continue
            
            # Handle different image dimensions (grayscale vs color)
            if len(label1.shape) == 3 and label1.shape[2] == 3:
                label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
            if len(label2.shape) == 3 and label2.shape[2] == 3:
                label2 = cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY)
                
            if pred.shape[:2] != label1.shape[:2] or pred.shape[:2] != label2.shape[:2]:
                print(f"Size mismatch in file: {filename} - Pred: {pred.shape}, Label1: {label1.shape}, Label2: {label2.shape}")
                continue

            # Generate change label - detect class disappearance (T1 has class, T2 doesn't)
            # For SECOND dataset, we need to use grayscale values, not class IDs
            # Map class_id to grayscale_value for SECOND dataset
            grayscale_mapping = {1: 29, 2: 128, 3: 75, 4: 149, 5: 38, 6: 76}
            grayscale_value = grayscale_mapping.get(class_id, class_id)
            
            change_label = ((label1 == grayscale_value) != (label2 == grayscale_value)).astype(np.uint8) * 255
            
            # Update metrics
            metrics.update(pred, change_label)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    if processed_count == 0:
        print(f"Warning: No valid images processed for class {class_name}")
        return {}
    
    # Compute and return results
    results = metrics.compute()
    results['processed_images'] = processed_count
    return results


def evaluate_second_class_simple(pred_dir: str, change_label_dir: str, 
                                class_name: str, threshold: float = 0.5) -> Dict[str, float]:
    """
    Simplified evaluation for SECOND dataset when change labels are pre-computed.
    
    Args:
        pred_dir: Directory containing prediction masks
        change_label_dir: Directory containing pre-computed change labels for this class
        class_name: Name of the class being evaluated
        threshold: Binarization threshold
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if not os.path.exists(pred_dir):
        print(f"Warning: Prediction directory not found: {pred_dir}")
        return {}
    
    if not os.path.exists(change_label_dir):
        print(f"Warning: Change label directory not found: {change_label_dir}")
        return {}
    
    # Initialize metric calculator
    metrics = ChangeDetectionMetrics(threshold=threshold)
    
    # Get list of images
    image_list = sorted(os.listdir(pred_dir))
    processed_count = 0
    
    for filename in tqdm(image_list, desc=f'Evaluating {class_name}'):
        try:
            # Load prediction and change label
            pred_path = os.path.join(pred_dir, filename)
            change_label_path = os.path.join(change_label_dir, filename)

            # Check if both files exist
            if not (os.path.exists(pred_path) and os.path.exists(change_label_path)):
                continue

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            change_label = cv2.imread(change_label_path, cv2.IMREAD_GRAYSCALE)

            # Validate inputs
            if pred is None or change_label is None:
                continue
                
            if pred.shape != change_label.shape:
                print(f"Size mismatch in file: {filename} - Pred: {pred.shape}, Label: {change_label.shape}")
                continue

            # Update metrics
            metrics.update(pred, change_label)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    if processed_count == 0:
        print(f"Warning: No valid images processed for class {class_name}")
        return {}
    
    # Compute and return results
    results = metrics.compute()
    results['processed_images'] = processed_count
    return results


def evaluate_second_dataset(pred_base_dir: str, gt1_dir: str, gt2_dir: str, 
                          threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all classes for SECOND dataset.
    
    Args:
        pred_base_dir: Base directory containing prediction subdirectories for each class
        gt1_dir: Directory containing first time point labels
        gt2_dir: Directory containing second time point labels
        threshold: Binarization threshold
        
    Returns:
        Dictionary containing results for all classes and overall metrics
    """
    # SECOND dataset class configuration
    CLASS_CONFIG = {
        'building': {'id': 5, 'pred_subdir': 'building'},
        'water': {'id': 1, 'pred_subdir': 'water'},
        'ground': {'id': 2, 'pred_subdir': 'non_veg_ground_surface'},
        'low_vegetation': {'id': 3, 'pred_subdir': 'low_vegetation'},
        'tree': {'id': 4, 'pred_subdir': 'tree'},
        'playground': {'id': 6, 'pred_subdir': 'playground'},
    }
    
    all_results = {}
    
    # Evaluate each class
    for class_name, config in CLASS_CONFIG.items():
        print(f"\nEvaluating class: {class_name}")
        
        pred_class_dir = os.path.join(pred_base_dir, config['pred_subdir'])
        results = evaluate_second_class(
            pred_dir=pred_class_dir,
            gt1_dir=gt1_dir,
            gt2_dir=gt2_dir,
            class_name=class_name,
            class_id=config['id'],
            threshold=threshold
        )
        
        if results:
            all_results[class_name] = results
    
    # Calculate overall metrics (average across all classes)
    if all_results:
        overall_metrics = {}
        metric_names = [key for key in next(iter(all_results.values())).keys() 
                       if key != 'processed_images']
        
        for metric_name in metric_names:
            values = [results[metric_name] for results in all_results.values() 
                     if metric_name in results]
            if values:
                overall_metrics[metric_name] = np.mean(values)
        
        all_results['overall'] = overall_metrics
    
    return all_results


def print_metrics(results: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print evaluation metrics in a formatted table.
    
    Args:
        results: Dictionary containing metrics
        model_name: Name of the model being evaluated
    """
    print(f"\n========== {model_name} Evaluation Results ==========")
    print(f"Number of images: {results.get('num_images', 'N/A')}")
    print(f"Threshold: {results.get('threshold', 'N/A')}")
    print("-" * 50)
    print(f"Overall Accuracy (OA):    {results['oa']:.4f}")
    print(f"Mean IoU (mIoU):          {results['miou']:.4f}")
    print(f"F1 Score:                 {results['f1']:.4f}")
    print(f"Precision:                {results['precision']:.4f}")
    print(f"Recall:                   {results['recall']:.4f}")
    print(f"Specificity:              {results['specificity']:.4f}")
    print("-" * 50)
    print(f"Background IoU:           {results['bg_iou']:.4f}")
    print(f"Change IoU:               {results['change_iou']:.4f}")
    print("-" * 50)
    print(f"True Positives:           {results['tp']:.0f}")
    print(f"True Negatives:           {results['tn']:.0f}")
    print(f"False Positives:          {results['fp']:.0f}")
    print(f"False Negatives:          {results['fn']:.0f}")
    print("=" * 50)


def print_second_metrics(all_results: Dict[str, Dict[str, float]], model_name: str = "Model") -> None:
    """
    Print SECOND dataset evaluation metrics in a formatted table.
    
    Args:
        all_results: Dictionary containing results for all classes
        model_name: Name of the model being evaluated
    """
    print(f"\n========== {model_name} SECOND Dataset Evaluation Results ==========")
    
    if 'overall' not in all_results:
        print("No results available")
        return
    
    # Print individual class results
    print("\nIndividual Class Results:")
    print("-" * 80)
    print(f"{'Class':<15} {'mIoU':<8} {'OA':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Images':<8}")
    print("-" * 80)
    
    for class_name, results in all_results.items():
        if class_name != 'overall':
            print(f"{class_name:<15} "
                  f"{results.get('miou', 0):<8.4f} "
                  f"{results.get('oa', 0):<8.4f} "
                  f"{results.get('f1', 0):<8.4f} "
                  f"{results.get('precision', 0):<10.4f} "
                  f"{results.get('recall', 0):<8.4f} "
                  f"{results.get('processed_images', 0):<8}")
    
    # Print overall results
    overall = all_results['overall']
    total_images = sum(r.get('processed_images', 0) for r in all_results.values() if r != all_results['overall'])
    
    print("-" * 80)
    print(f"{'AVERAGE':<15} "
          f"{overall.get('miou', 0):<8.4f} "
          f"{overall.get('oa', 0):<8.4f} "
          f"{overall.get('f1', 0):<8.4f} "
          f"{overall.get('precision', 0):<10.4f} "
          f"{overall.get('recall', 0):<8.4f} "
          f"{total_images:<8}")
    
    print("\nOverall Metrics (Average across all classes):")
    print("-" * 50)
    for metric_name, value in overall.items():
        if metric_name != 'processed_images':
            print(f"{metric_name:15}: {value:.4f}")
    print("=" * 80) 
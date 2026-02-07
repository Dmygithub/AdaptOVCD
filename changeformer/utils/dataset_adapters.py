"""
Dataset Adapter Module.

Provides a unified dataset interface supporting various change detection dataset formats and structures.
Address compatibility issues for special formats like SECOND multi-class dataset.
"""

import os
import cv2
import numpy as np
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .logging_utils import get_logger


class BaseDatasetAdapter(ABC):
    """Base class for dataset adapters."""
    
    def __init__(self, dataset_name: str, data_root: str):
        """Initializes the dataset adapter.
        
        Args:
            dataset_name: Name of the dataset.
            data_root: Root directory path of the dataset.
        """
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        if not self.data_root.exists():
            raise ValueError(f"Dataset root directory not found: {data_root}")
    
    @abstractmethod
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Gets the list of image pairs.
        
        Returns:
            List of (img1_path, img2_path) tuples.
        """
        pass
    
    @abstractmethod
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Gets evaluation configuration.
        
        Returns:
            Evaluation configuration dictionary.
        """
        pass
    
    def validate_dataset(self) -> bool:
        """Validates dataset integrity."""
        try:
            pairs = self.get_image_pairs()
            if not pairs:
                self.logger.error(f"No image pairs found in {self.data_root}")
                return False
            
            # Check if the first few files exist
            for i, (img1, img2) in enumerate(pairs[:5]):
                if not os.path.exists(img1):
                    self.logger.error(f"Image 1 not found: {img1}")
                    return False
                if not os.path.exists(img2):
                    self.logger.error(f"Image 2 not found: {img2}")
                    return False
            
            self.logger.info(f"Dataset validation passed: {len(pairs)} pairs found")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False


class StandardAdapter(BaseDatasetAdapter):
    """Adapter for standard formats (LEVIR-CD, WHU-CD, etc.)."""
    
    def __init__(self, dataset_name: str, data_root: str, config: Dict[str, Any]):
        """Initializes the standard adapter.

        Args:
            config: Dataset configuration containing data_paths, etc.
        """
        super().__init__(dataset_name, data_root)
        self.config = config
        
        # Get path configuration
        data_paths = config.get('data_paths', {})
        self.img1_dir = self.data_root / data_paths.get('test_A', 'A')
        self.img2_dir = self.data_root / data_paths.get('test_B', 'B')
        self.gt_dir = self.data_root / data_paths.get('test_label', 'label')
        
        # File format
        patterns = config.get('patterns', {})
        self.image_format = patterns.get('image_format', '*.png')
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Gets image pairs for standard format."""
        pairs = []
        
        if not self.img1_dir.exists() or not self.img2_dir.exists():
            self.logger.error(f"Image directories not found: {self.img1_dir}, {self.img2_dir}")
            return pairs
        
        # Get list of image files
        img1_files = sorted([f for f in os.listdir(self.img1_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        for img1_file in img1_files:
            img1_path = str(self.img1_dir / img1_file)
            img2_path = str(self.img2_dir / img1_file)
            
            if os.path.exists(img2_path):
                pairs.append((img1_path, img2_path))
            else:
                self.logger.warning(f"Missing pair for {img1_file}")
        
        return pairs
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Gets standard evaluation configuration."""
        return {
            'type': 'standard',
            'gt_dir': str(self.gt_dir),
            'metrics': self.config.get('evaluation', {}).get('metrics', 
                                     ['miou', 'f1', 'oa', 'precision', 'recall']),
            'threshold': self.config.get('evaluation', {}).get('threshold', 0.5)
        }


class SECONDAdapter(BaseDatasetAdapter):
    """Adapter for SECOND dataset - Supports multi-class semantic change detection."""
    
    # Class configuration for SECOND dataset
    CLASS_CONFIG = {
        'building': {'id': 5, 'pred_subdir': 'building'},
        'water': {'id': 1, 'pred_subdir': 'water'},
        'ground': {'id': 2, 'pred_subdir': 'non_veg_ground_surface'},
        'low_vegetation': {'id': 3, 'pred_subdir': 'low_vegetation'},
        'tree': {'id': 4, 'pred_subdir': 'tree'},
        'playground': {'id': 6, 'pred_subdir': 'playground'},
    }
    
    def __init__(self, dataset_name: str, data_root: str, config: Dict[str, Any]):
        super().__init__(dataset_name, data_root)
        self.config = config
        
        # Special path structure for SECOND dataset
        data_paths = config.get('data_paths', {})
        self.img1_dir = self.data_root / data_paths.get('test_im1', 'test/im1')
        self.img2_dir = self.data_root / data_paths.get('test_im2', 'test/im2')
        self.gt1_dir = self.data_root / data_paths.get('test_label1', 'test/label1')
        self.gt2_dir = self.data_root / data_paths.get('test_label2', 'test/label2')
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Gets image pairs for SECOND dataset."""
        pairs = []
        
        if not all(d.exists() for d in [self.img1_dir, self.img2_dir]):
            self.logger.error(f"Image directories not found")
            return pairs
        
        # Get list of image files
        img1_files = sorted([f for f in os.listdir(self.img1_dir) 
                           if f.lower().endswith('.png')])
        
        for img1_file in img1_files:
            img1_path = str(self.img1_dir / img1_file)
            img2_path = str(self.img2_dir / img1_file)
            
            if os.path.exists(img2_path):
                pairs.append((img1_path, img2_path))
            else:
                self.logger.warning(f"Missing pair for {img1_file}")
        
        return pairs
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Gets multi-class evaluation configuration for SECOND."""
        return {
            'type': 'second_multiclass',
            'gt1_dir': str(self.gt1_dir),
            'gt2_dir': str(self.gt2_dir),
            'class_config': self.CLASS_CONFIG,
            'metrics': ['miou', 'f1', 'oa', 'precision', 'recall'],
            'threshold': 0.5
        }
    
    def prepare_class_predictions(self, base_output_dir: str, class_name: str) -> str:
        """Prepares prediction directory for a specific class.
        
        Args:
            base_output_dir: Base output directory.
            class_name: Name of the class.
            
        Returns:
            Path to the prediction directory for the class.
        """
        if class_name not in self.CLASS_CONFIG:
            raise ValueError(f"Unknown class: {class_name}")
        
        class_dir = os.path.join(base_output_dir, self.CLASS_CONFIG[class_name]['pred_subdir'])
        os.makedirs(class_dir, exist_ok=True)
        return class_dir


class WHUCDAdapter(StandardAdapter):
    """Adapter for WHU-CD dataset - Handles TIF format."""
    
    def __init__(self, dataset_name: str, data_root: str, config: Dict[str, Any]):
        super().__init__(dataset_name, data_root, config)
        self.needs_format_conversion = True
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Gets WHU-CD image pairs, handling TIF format."""
        pairs = []
        
        if not self.img1_dir.exists() or not self.img2_dir.exists():
            self.logger.error(f"Image directories not found: {self.img1_dir}, {self.img2_dir}")
            return pairs
        
        # Specifically handle TIF files
        img1_files = sorted([f for f in os.listdir(self.img1_dir) 
                           if f.lower().endswith(('.tif', '.tiff'))])
        
        for img1_file in img1_files:
            img1_path = str(self.img1_dir / img1_file)
            img2_path = str(self.img2_dir / img1_file)
            
            if os.path.exists(img2_path):
                pairs.append((img1_path, img2_path))
            else:
                self.logger.warning(f"Missing pair for {img1_file}")
        
        return pairs


def create_dataset_adapter(dataset_name: str, data_root: str, 
                         config: Dict[str, Any]) -> BaseDatasetAdapter:
    """Factory function to create a dataset adapter.
    
    Args:
        dataset_name: Name of the dataset.
        data_root: Root path of the dataset.
        config: Dataset configuration.
        
    Returns:
        Instance of the corresponding dataset adapter.
    """
    logger = get_logger('DatasetAdapterFactory')
    
    # Select adapter based on dataset name
    if dataset_name.lower() == 'second':
        adapter = SECONDAdapter(dataset_name, data_root, config)
    elif dataset_name.lower() == 'whucd':
        adapter = WHUCDAdapter(dataset_name, data_root, config)
    else:
        # Use standard adapter by default (LEVIR-CD, S2Looking, BANDON, etc.)
        adapter = StandardAdapter(dataset_name, data_root, config)
    
    # Validate dataset
    if not adapter.validate_dataset():
        logger.warning(f"Dataset validation failed for {dataset_name}")
    
    logger.info(f"Created {adapter.__class__.__name__} for {dataset_name}")
    return adapter


def get_unified_image_pairs(dataset_name: str, data_root: str, 
                          dataset_config: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Unified interface: Gets image pairs and evaluation config for any dataset.
    
    Args:
        dataset_name: Name of the dataset.
        data_root: Root path of the dataset.
        dataset_config: Dataset configuration.
        
    Returns:
        Tuple of (image_pairs, evaluation_config).
    """
    adapter = create_dataset_adapter(dataset_name, data_root, dataset_config)
    
    image_pairs = adapter.get_image_pairs()
    eval_config = adapter.get_evaluation_config()
    
    return image_pairs, eval_config

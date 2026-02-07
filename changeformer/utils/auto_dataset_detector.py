"""
Automatic Dataset Detector.

Intelligently detects and adapts local dataset paths, requiring only the dataset name from the user.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .logging_utils import get_logger


class DatasetDetector:
    """Automatic dataset detector."""
    
    def __init__(self, project_root: str = None):
        """Initializes the detector.
        
        Args:
            project_root: Project root directory. Detected automatically if not provided.
        """
        self.logger = get_logger(self.__class__.__name__)
        
        if project_root is None:
            # Automatically detect project root
            current_file = Path(__file__).resolve()
            # Find project root from changeformer/utils/auto_dataset_detector.py
            self.project_root = current_file.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.logger.info(f"Project root: {self.project_root}")
        
        # Possible data directory locations
        self.search_paths = [
            self.project_root / "data",
            self.project_root / "dataset",
            self.project_root / "datasets",
            self.project_root.parent / "data",  # Parent directory
        ]
    
    def detect_all_datasets(self) -> Dict[str, str]:
        """Detects all available datasets.
        
        Returns:
            Mapping of dataset names to their paths.
        """
        detected = {}
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
            
            self.logger.info(f"Scanning: {search_path}")
            
            for item in search_path.iterdir():
                if not item.is_dir():
                    continue
                
                dataset_name = self._identify_dataset(item)
                if dataset_name:
                    # Prefer exact directory name match
                    if dataset_name not in detected:
                        detected[dataset_name] = str(item)
                        self.logger.info(f"  Found {dataset_name}: {item}")
                    else:
                        # If exists, check if current is exact match
                        current_dir_name = item.name.lower()
                        existing_dir_name = Path(detected[dataset_name]).name.lower()
                        
                        # Exact match priority (e.g., 's2looking' over 's2looking copy')
                        if current_dir_name == dataset_name and existing_dir_name != dataset_name:
                            detected[dataset_name] = str(item)
                            self.logger.info(f"  Updated {dataset_name}: {item} (exact match)")
        
        return detected
    
    def _identify_dataset(self, path: Path) -> Optional[str]:
        """Identifies dataset type based on directory structure.
        
        Args:
            path: Dataset directory path.
            
        Returns:
            Dataset name or None.
        """
        path_name = path.name.lower()
        
        # Direct name match
        direct_matches = {
            'levircd': 'levircd',
            'levir-cd': 'levircd',
            'levir_cd': 'levircd',
            'whucd': 'whucd',
            'whu-cd': 'whucd',
            'whu_cd': 'whucd',
            'second': 'second',
            'second_dataset': 'second',
            's2looking': 's2looking',
            's2-looking': 's2looking',
            'bandon': 'bandon',
            'bandon_index': 'bandon',
            'building change detection dataset_add': 'building_change',  # Newly detected dataset
        }
        
        for pattern, dataset_name in direct_matches.items():
            if pattern in path_name:
                # Validate directory structure
                if self._validate_dataset_structure(path, dataset_name):
                    return dataset_name
        
        # Intelligent identification based on structure
        return self._identify_by_structure(path)
    
    def _validate_dataset_structure(self, path: Path, dataset_name: str) -> bool:
        """Validates if dataset directory structure meets expectations.
        
        Args:
            path: Dataset path.
            dataset_name: Name of the dataset.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            if dataset_name == 'levircd':
                # LEVIRCD: A/, B/, label/
                return all((path / d).exists() for d in ['A', 'B', 'label'])
            
            elif dataset_name == 'whucd':
                # WHUCD: A/, B/, label/ (TIF format)
                a_dir = path / 'A'
                return (a_dir.exists() and 
                       any(f.suffix.lower() in ['.tif', '.tiff'] for f in a_dir.iterdir()))
            
            elif dataset_name == 'second':
                # SECOND: test/im1, test/im2, test/label1, test/label2
                test_dir = path / 'test'
                return (test_dir.exists() and 
                       all((test_dir / d).exists() for d in ['im1', 'im2', 'label1', 'label2']))
            
            elif dataset_name == 's2looking':
                # S2Looking: Image1/, Image2/, label/
                return all((path / d).exists() for d in ['Image1', 'Image2', 'label'])
            
            elif dataset_name == 'bandon':
                # BANDON: A/, B/, label/ (preprocessed structure)
                return all((path / d).exists() for d in ['A', 'B', 'label'])
            
            elif dataset_name == 'building_change':
                # Building Change: Contains year directories
                return any(d.name in ['2012', '2016'] for d in path.iterdir() if d.is_dir())
            
            return True  # Default to valid
            
        except Exception:
            return False
    
    def _identify_by_structure(self, path: Path) -> Optional[str]:
        """Identifies dataset type based on directory structure.
        
        Args:
            path: Dataset path.
            
        Returns:
            Identified dataset type.
        """
        try:
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            
            # SECOND features: test/im1, im2, label1, label2
            if 'test' in subdirs:
                test_dir = path / 'test'
                test_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
                if all(d in test_subdirs for d in ['im1', 'im2', 'label1', 'label2']):
                    return 'second'
            
            # Standard A/B/label structure
            if all(d in subdirs for d in ['A', 'B', 'label']):
                # Check file format
                a_dir = path / 'A'
                if a_dir.exists():
                    files = list(a_dir.iterdir())
                    if files:
                        first_file = files[0]
                        if first_file.suffix.lower() in ['.tif', '.tiff']:
                            return 'whucd'
                        else:
                            return 'levircd'
            
            # S2Looking features
            if all(d in subdirs for d in ['Image1', 'Image2', 'label']):
                return 's2looking'
            
            # BANDON features: Standard A/B/label, but check for JPG
            if all(d in subdirs for d in ['A', 'B', 'label']):
                # Check if A directory has JPG files (distinguish from others)
                a_dir = path / 'A'
                if a_dir.exists():
                    files = list(a_dir.iterdir())
                    if files and any(f.suffix.lower() == '.jpg' for f in files):
                        return 'bandon'
            
            # Time series dataset features
            year_dirs = [d for d in subdirs if d.isdigit() and len(d) == 4]
            if len(year_dirs) >= 2:
                return 'building_change'
            
            return None
            
        except Exception:
            return None
    
    def find_dataset(self, dataset_name: str) -> Optional[str]:
        """Finds path for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dataset path or None.
        """
        all_datasets = self.detect_all_datasets()
        return all_datasets.get(dataset_name)
    
    def list_available_datasets(self) -> List[Tuple[str, str]]:
        """Lists all available datasets.
        
        Returns:
            List of (dataset_name, path) tuples.
        """
        all_datasets = self.detect_all_datasets()
        return [(name, path) for name, path in all_datasets.items()]


# Global detector instance
_detector = None


def get_dataset_detector() -> DatasetDetector:
    """Gets the global dataset detector instance."""
    global _detector
    if _detector is None:
        _detector = DatasetDetector()
    return _detector


def auto_find_dataset(dataset_name: str) -> Optional[str]:
    """Automatically finds dataset path.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Dataset path or None.
    """
    detector = get_dataset_detector()
    return detector.find_dataset(dataset_name)


def list_detected_datasets() -> Dict[str, str]:
    """Lists all detected datasets.
    
    Returns:
        Mapping of dataset names to paths.
    """
    detector = get_dataset_detector()
    return detector.detect_all_datasets()

"""
Semantic identification modules for change detection.

This module contains various identifier implementations including SegEarth-OV and DGTRS-CLIP
for semantic classification of detected changes.

"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Dict, Any, Tuple, Optional
import cv2
from PIL import Image

# Try importing DGTRS dependencies
try:
    sys.path.append("third_party/DGTRS")
    from model import longclip
    DGTRS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DGTRS dependencies not available: {e}")
    DGTRS_AVAILABLE = False
    longclip = None








class DGTRSWrapper(nn.Module):
    """
    Wrapper for DGTRS-CLIP model to match DynamicEarth's identifier interface.
    
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 confidence_threshold: float = 0.2,
                 name_list: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize DGTRS-CLIP wrapper.
        
        Args:
            model_path (str): Path to DGTRS-CLIP checkpoint
            device (str): Device to run inference on
            confidence_threshold (float): Confidence threshold for predictions
            name_list (List[str]): List of class names
        """
        super().__init__()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.name_list = name_list or ['background', 'building']
        
        # Initialize DGTRS-CLIP model
        self._load_dgtrs_model(model_path)
    def _load_dgtrs_model(self, model_path: Optional[str]):
        """Load DGTRS-CLIP model using their official API."""
        try:
            if not DGTRS_AVAILABLE:
                raise ImportError("DGTRS not available")
                
            # Load DGTRS-CLIP model
            if model_path is None:
                model_path = "models/DGTRS/LRSCLIP_ViT-L-14.pt"
            
            self.model, self.preprocess = longclip.load(model_path, device=self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading DGTRS-CLIP model: {e}")
            print("Using dummy model for testing...")
            self.model = self._create_dummy_model()
            self.preprocess = self._create_dummy_preprocess()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when DGTRS is not available."""
        class DummyDGTRS(nn.Module):
            def encode_image(self, image):
                return torch.rand(image.shape[0], 512).to(image.device)
            
            def encode_text(self, text):
                return torch.rand(text.shape[0], 512).to(text.device)
        
        return DummyDGTRS().to(self.device)
    
    def _create_dummy_preprocess(self):
        """Create dummy preprocessing when DGTRS is not available."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def predict(self, 
                image_tensor: torch.Tensor,
                data_samples=None,
                proposal_masks: Optional[np.ndarray] = None) -> List[int]:
        """
        Predict semantic classes for given masks using DGTRS-CLIP.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor
            data_samples: Unused, kept for interface compatibility
            proposal_masks (np.ndarray): Proposal masks to classify
            
        Returns:
            List[int]: Predicted class indices for each mask
        """
        if proposal_masks is None:
            return []
            
        predictions = []
        
        # Fully flexible soft-coded prompt generation
        # Users define foreground (target) and background (exclusion) freely via name_list
        # No hardcoded rules to avoid semantic conflicts (e.g., "road" in both classes)
        text_prompts = []
        for name in self.name_list:
            # Check if name already includes "satellite image" prefix
            if 'satellite image' in name.lower():
                # Use as-is if complete prompt provided
                text_prompts.append(name)
            else:
                # Add standard prefix only
                text_prompts.append(f"satellite image of {name}")
        
        try:
            # Batch process masks for better speed
            batch_size = 8
            for i in range(0, len(proposal_masks), batch_size):
                batch_masks = proposal_masks[i:i+batch_size]
                for mask in batch_masks:
                    pred_class = self._classify_mask_region(image_tensor, mask, text_prompts)
                    predictions.append(pred_class)
                
        except Exception as e:
            print(f"Error in DGTRS-CLIP prediction: {e}")
            # Fallback to background class
            predictions.extend([0] * len(proposal_masks))
                
        return predictions
    
    def _classify_mask_region(self, image_tensor: torch.Tensor, mask: np.ndarray, 
                             text_prompts: List[str]) -> int:
        """Classify a masked region using DGTRS-CLIP."""
        try:
            # Convert tensor to PIL Image
            image_pil = self._tensor_to_pil(image_tensor)
            
            # Apply mask to image
            masked_image = self._apply_mask_to_image(image_pil, mask)
            
            # Preprocess image for DGTRS-CLIP
            image_input = self.preprocess(masked_image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompts
            if DGTRS_AVAILABLE and longclip is not None:
                text_tokens = longclip.tokenize(text_prompts).to(self.device)
            else:
                # Dummy tokenization for testing
                text_tokens = torch.randint(0, 1000, (len(text_prompts), 77)).to(self.device)
            
            with torch.no_grad():
                # Get features from DGTRS-CLIP
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)
                
                # Calculate similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).squeeze(0)
                
                # Get predicted class
                pred_class = torch.argmax(similarity).item()
                raw_confidence = torch.softmax(similarity, dim=0)[pred_class].item()
                
                # Apply confidence threshold
                if raw_confidence > self.confidence_threshold:
                    return pred_class
                else:
                    return 0  # Background class
                    
        except Exception as e:
            print(f"Error in mask classification: {e}")
            return 0  # Default to background
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if normalized
        if tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
            tensor = tensor * std + mean
        
        # Clamp and convert to numpy
        tensor = torch.clamp(tensor, 0, 1)
        numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
        numpy_image = (numpy_image * 255).astype(np.uint8)
        
        return Image.fromarray(numpy_image)
    
    def _apply_mask_to_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply mask to image by cropping the masked region."""
        # Resize mask to match image size
        mask_resized = cv2.resize(
            mask.astype(np.uint8), 
            image.size, 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Find bounding box of the mask
        coords = np.column_stack(np.where(mask_resized))
        if len(coords) == 0:
            return image  # Return original if empty mask
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add some padding to provide context
        padding = max(10, min(20, (y_max - y_min) // 4))
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(image.height, y_max + padding)
        x_max = min(image.width, x_max + padding)
        
        # Crop the image to the bounding box
        cropped = image.crop((x_min, y_min, x_max+1, y_max+1))
        
        # Resize to standard size for DGTRS-CLIP
        cropped = cropped.resize((224, 224), Image.LANCZOS)
        
        return cropped





def get_dgtrs_identifier(device: str = 'cuda', 
                        name_list: Optional[List[str]] = None,
                        model_path: Optional[str] = None,
                        **kwargs) -> Tuple[DGTRSWrapper, Any]:
    """
    Factory function to create DGTRS-CLIP identifier.
    
    Args:
        device (str): Device to run on
        name_list (List[str]): List of class names
        model_path (str): Path to DGTRS-CLIP checkpoint
        
    Returns:
        Tuple: (model, processor) pair
    """
    
    # Create processor following DGTRS-CLIP's preprocessing requirements
    # This matches the original dynamic_earth processor format
    processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create model wrapper
    model = DGTRSWrapper(
        model_path=model_path,
        device=device,
        name_list=name_list,
        **kwargs
    )
    
    return model, processor





def identify_with_dgtrs(img1, img2, cmasks, img1_mask_num, identifier_model, identifier_processor, model_type='DGTRS-CLIP', device='cuda'):
    """
    Identify specific classes of change masks using DGTRS-CLIP.
    
    This function follows the exact logic from the original identify() function.
    """
    if len(cmasks) == 0:
        return cmasks, img1_mask_num
    
    try:
        # Process img1 - convert to tensor format expected by DGTRS
        img1_tensor = identifier_processor(img1).unsqueeze(0).to(device)
        img1_mask_classes = identifier_model.predict(img1_tensor, data_samples=None, proposal_masks=cmasks)
        
        # Process img2 - convert to tensor format expected by DGTRS  
        img2_tensor = identifier_processor(img2).unsqueeze(0).to(device)
        img2_mask_classes = identifier_model.predict(img2_tensor, data_samples=None, proposal_masks=cmasks)
        # Auto-detect building category from name_list for strict change logic
        # Building detection requires instance matching (both time points must have target class)
        # Other categories use simple logic (any classification difference)
        is_building_detection = False
        
        # Check if foreground category (name_list[1]) contains "building" or "buildings"
        if hasattr(identifier_model, 'name_list') and len(identifier_model.name_list) >= 2:
            foreground_name = identifier_model.name_list[1].lower()
            if 'building' in foreground_name or 'buildings' in foreground_name:
                is_building_detection = True
        
        # Fallback: check target_class attribute (for backward compatibility)
        if not is_building_detection:
            target_class = getattr(identifier_model, 'target_class', None)
            if target_class and 'building' in str(target_class).lower():
                is_building_detection = True
        
        if is_building_detection:
            # 🏢 Building detection: Use DynamicEarth's strict logic (is_instance_class=True)
            # Requires both time points to have building class for change confirmation
            change_instance_match = img1_mask_classes[:img1_mask_num] + img2_mask_classes[img1_mask_num:]
            change_idx = np.where((np.array(img1_mask_classes) != np.array(img2_mask_classes)) & 
                                np.array(change_instance_match).astype(bool))
        else:
            # 🌳 Other categories: Use simple logic (is_instance_class=False)
            # Any classification difference is considered as change
            change_idx = np.where(np.array(img1_mask_classes) != np.array(img2_mask_classes))
        

        ####################################################
        # Filter masks based on change detection
        cmasks_filtered = np.array(cmasks)[change_idx]
        img1_mask_num_filtered = np.sum(change_idx[0] < img1_mask_num)
        
        return cmasks_filtered, img1_mask_num_filtered
        
    except Exception as e:
        print(f"Warning: DGTRS identification failed: {e}")
        import traceback
        traceback.print_exc()
        # Return original masks if identification fails
        return cmasks, img1_mask_num


# ==================== APE Related Functions ====================

def build_ape_identifier(config_file: str, confidence_threshold: float, opt: list):
    """
    Build APE identifier model.
    
    Args:
        config_file: Path to APE config file
        confidence_threshold: Confidence threshold for predictions
        opt: Additional options for APE
        
    Returns:
        APE model instance
    """
    try:
        from changeformer.core.segmentation.ape import build_ape_model
        return build_ape_model(config_file, confidence_threshold, opt)
    except ImportError as e:
        raise ImportError(f"APE dependencies not available: {e}")


def extract_prediction_from_ape(model, input_path: str, text_prompt: str):
    """
    Extract predictions from APE model.
    
    Args:
        model: APE model instance
        input_path: Path to input image
        text_prompt: Text prompt for APE
        
    Returns:
        APE predictions
    """
    try:
        from changeformer.core.segmentation.ape import extract_ape_prediction
        return extract_ape_prediction(model, input_path, text_prompt, with_mask=True)
    except ImportError as e:
        raise ImportError(f"APE dependencies not available: {e}")


# End of identifiers.py
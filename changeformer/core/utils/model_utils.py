"""
Model utilities for loading DINO and DINOv2 models.

This module contains the exact model loading utilities from dynamic_earth/utils/model.py.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, Dinov2Model, Dinov2Config


# DINOv2 variant configurations
DINOV2_VARIANTS = {
    'dinov2_vits14': {
        'hf_name': 'facebook/dinov2-small',
        'hidden_size': 384,
        'patch_size': 14,
    },
    'dinov2_vitb14': {
        'hf_name': 'facebook/dinov2-base',
        'hidden_size': 768,
        'patch_size': 14,
    },
    'dinov2_vitl14': {
        'hf_name': 'facebook/dinov2-large',
        'hidden_size': 1024,
        'patch_size': 14,
    },
    'dinov2_vitg14': {
        'hf_name': 'facebook/dinov2-giant',
        'hidden_size': 1536,
        'patch_size': 14,
    },
}


@torch.no_grad()
def get_model_and_processor(
    model_type,
    device,
    model_config=None,
    processor_config=None,
    ):
    """
    Get model and processor for the specified model type.
    
    Args:
        model_type: 'DINO', 'DINOv2', or 'DINOv3'
        device: Device to load model on
        model_config: Optional config dict with 'variant' and 'weights_path'
    """
    
    if model_type == 'DINO':
        processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
    
    elif model_type == 'DINOv2':
        # Get variant and weights_path from config
        variant = 'dinov2_vitb14'
        weights_path = None
        
        if model_config:
            variant = model_config.get('variant', 'dinov2_vitb14')
            weights_path = model_config.get('weights_path', None)
        
        # Get variant configuration
        variant_config = DINOV2_VARIANTS.get(variant)
        if variant_config is None:
            print(f"Warning: Unknown DINOv2 variant '{variant}', using dinov2_vitb14")
            variant = 'dinov2_vitb14'
            variant_config = DINOV2_VARIANTS[variant]
        
        hf_name = variant_config['hf_name']
        
        # Load processor
        processor = AutoImageProcessor.from_pretrained(hf_name, do_resize=False, do_center_crop=False)
        
        # Load model - always from HuggingFace for correct architecture
        print(f"Loading DINOv2 variant: {variant} ({hf_name})")
        model = AutoModel.from_pretrained(hf_name).to(device)
    
    return model, processor


@torch.no_grad()
def get_dino_model_and_processor(device: str = 'cpu'):
    return get_model_and_processor('DINO', device)


@torch.no_grad()
def get_dinov2_model_and_processor(device: str = 'cpu', variant: str = 'dinov2_vitb14'):
    model_config = {'variant': variant}
    return get_model_and_processor('DINOv2', device, model_config=model_config)

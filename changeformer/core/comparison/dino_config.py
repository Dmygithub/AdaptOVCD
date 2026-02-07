"""
DINO model variant specifications.

This module defines the specifications for different DINO model variants.
Feature dimensions and patch sizes are determined by model architecture and should not be modified.
"""

# DINO model variant specifications
# Format: {variant_name: (feature_dim, patch_size, weights_path)}
DINO_VARIANTS = {
    # DINO models (torch.hub.load, auto-download)
    'dino_vitb16': {
        'feature_dim': 768,
        'patch_size': 16,
        'model_type': 'DINO',
        'weights_path': None,  # Auto-download via torch.hub
        'description': 'DINO ViT-B/16 (768-dim)'
    },
    
    # DINOv2 models (HuggingFace, auto-download)
    'dinov2_vitl14': {
        'feature_dim': 1024,
        'patch_size': 14,
        'model_type': 'DINOv2',
        'weights_path': None,  # Auto-download via HuggingFace
        'description': 'DINOv2 ViT-L/14 (1024-dim)'
    },
    
    # DINOv3 models (local weights required)
    'dinov3_vitb16': {
        'feature_dim': 768,
        'patch_size': 16,
        'model_type': 'DINOv3',
        'weights_path': 'models/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth',
        'description': 'DINOv3 ViT-B/16 (768-dim, 100M params)'
    },
    'dinov3_vitl16': {
        'feature_dim': 1024,
        'patch_size': 16,
        'model_type': 'DINOv3',
        'weights_path': 'models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'description': 'DINOv3 ViT-L/16 (1024-dim, 300M params)'
    },
    'dinov3_vith16': {
        'feature_dim': 1280,
        'patch_size': 16,
        'model_type': 'DINOv3',
        'weights_path': 'models/dinov3/dinov3_vith16_pretrain_lvd1689m.pth',
        'description': 'DINOv3 ViT-H/16+ (1280-dim, 840M params)'
    },
}


def get_variant_spec(variant: str) -> dict:
    """
    Get specification for a DINO variant.
    
    Args:
        variant: Variant name (e.g., 'dino_vitb16', 'dinov2_vitl14', 'dinov3_vitl16')
        
    Returns:
        Dictionary with 'feature_dim', 'patch_size', 'model_type', 'weights_path', 'description'
        
    Raises:
        ValueError: If variant is not supported
    """
    if variant not in DINO_VARIANTS:
        supported = ', '.join(DINO_VARIANTS.keys())
        raise ValueError(f"Unsupported DINO variant: {variant}. Supported variants: {supported}")
    
    return DINO_VARIANTS[variant].copy()


def list_variants() -> list:
    """Get list of all supported DINO variants."""
    return list(DINO_VARIANTS.keys())


def get_variant_description(variant: str) -> str:
    """Get description for a variant."""
    spec = get_variant_spec(variant)
    return spec['description']


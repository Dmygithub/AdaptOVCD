"""
Registry utilities for ChangeFormer.

This module provides utilities for managing model registries and resolving conflicts.
"""

import warnings
from typing import Optional


def clear_mmseg_registry(model_name: Optional[str] = 'SegEarth_OV') -> None:
    """
    Clear MMSeg registry conflicts to avoid model registration issues.
    
    This function safely removes a specific model from the MMSeg registry
    to prevent conflicts when re-importing or re-registering models.
    
    Args:
        model_name: Name of the model to remove from registry. 
                   Defaults to 'SegEarth_OV'.
    """
    try:
        from mmseg.registry import MODELS
        if model_name in MODELS._module_dict:
            del MODELS._module_dict[model_name]
            warnings.warn(
                f"Cleared {model_name} from MMSeg registry to avoid conflicts",
                UserWarning,
                stacklevel=2
            )
    except ImportError:
        # MMSeg not available, no need to clear registry
        pass
    except AttributeError:
        # Registry structure may have changed, ignore silently
        pass
    except Exception as e:
        # Log unexpected errors but don't fail the program
        warnings.warn(
            f"Failed to clear MMSeg registry for {model_name}: {e}",
            UserWarning,
            stacklevel=2
        )


def ensure_clean_registry() -> None:
    """
    Ensure all known problematic models are cleared from registries.
    
    This function clears all models that are known to cause registration
    conflicts in the ChangeFormer pipeline.
    """
    problematic_models = ['SegEarth_OV']
    
    for model_name in problematic_models:
        clear_mmseg_registry(model_name)
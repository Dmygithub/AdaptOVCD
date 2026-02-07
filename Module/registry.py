"""
Module registry system for OVCD enhancement modules.

This system allows for dynamic registration and retrieval of enhancement modules,
enabling plug-and-play functionality with thread-safe operations.
"""

import threading
from typing import Dict, Type, Any, Optional, List
from .base import BaseEnhancementModule


class ModuleRegistry:
    """
    Thread-safe registry for OVCD enhancement modules.
    
    Manages registration, instantiation, and retrieval of enhancement modules
    with thread-safe operations using RLock for nested locking support.
    """
    
    def __init__(self):
        """Initialize the thread-safe module registry."""
        self._modules: Dict[str, Type[BaseEnhancementModule]] = {}
        self._instances: Dict[str, BaseEnhancementModule] = {}
        self._lock = threading.RLock()  # Use RLock for nested lock support
    
    def register(self, name: str, module_class: Type[BaseEnhancementModule]) -> None:
        """
        Register an enhancement module (thread-safe).
        
        Args:
            name: Unique name for the module
            module_class: Module class to register
            
        Raises:
            ValueError: If module name already exists or class is invalid
        """
        with self._lock:
            if name in self._modules:
                raise ValueError(f"Module '{name}' is already registered")
            
            if not issubclass(module_class, BaseEnhancementModule):
                raise ValueError(f"Module class must inherit from BaseEnhancementModule")
            
            self._modules[name] = module_class
            print(f"Registered enhancement module: {name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister an enhancement module (thread-safe).
        
        Args:
            name: Name of the module to unregister
        """
        with self._lock:
            if name in self._modules:
                del self._modules[name]
            if name in self._instances:
                del self._instances[name]
            print(f"Unregistered enhancement module: {name}")
    
    def get_module_class(self, name: str) -> Type[BaseEnhancementModule]:
        """
        Get a registered module class (thread-safe).
        
        Args:
            name: Name of the module
            
        Returns:
            Module class
            
        Raises:
            KeyError: If module is not registered
        """
        with self._lock:
            if name not in self._modules:
                available = list(self._modules.keys())
                raise KeyError(f"Module '{name}' is not registered. Available modules: {available}")
            
            return self._modules[name]
    
    def create_module(self, name: str, config: Dict[str, Any] = None) -> BaseEnhancementModule:
        """
        Create an instance of a registered module (thread-safe).
        
        Args:
            name: Name of the module
            config: Configuration for the module
            
        Returns:
            Module instance
        """
        # Get module class (thread-safe)
        with self._lock:
            if name not in self._modules:
                available = list(self._modules.keys())
                raise KeyError(f"Module '{name}' is not registered. Available modules: {available}")
            module_class = self._modules[name]
        
        # Create and initialize instance (no lock needed for instance operations)
        instance = module_class(config)
        if not instance.is_initialized:
            instance.initialize()
            instance.is_initialized = True
        
        return instance
    
    def get_or_create_module(self, name: str, config: Dict[str, Any] = None) -> BaseEnhancementModule:
        """
        Get existing module instance or create a new one (thread-safe).
        
        Args:
            name: Name of the module
            config: Configuration for the module (only used for new instances)
            
        Returns:
            Module instance
        """
        with self._lock:
            cache_key = f"{name}_{hash(str(sorted(config.items())) if config else '')}"
            
            if cache_key not in self._instances:
                self._instances[cache_key] = self.create_module(name, config)
            
            return self._instances[cache_key]
    
    def list_modules(self) -> List[str]:
        """
        List all registered module names (thread-safe).
        
        Returns:
            List of module names
        """
        with self._lock:
            return list(self._modules.keys())
    
    def get_module_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered module.
        
        Args:
            name: Name of the module
            
        Returns:
            Module information dictionary
        """
        module_class = self.get_module_class(name)
        
        # Create a temporary instance to get template info
        temp_instance = module_class()
        
        return {
            'name': name,
            'class': module_class.__name__,
            'type': temp_instance.get_module_type(),
            'config_template': temp_instance.get_config_template(),
            'docstring': module_class.__doc__
        }
    
    def list_modules_by_type(self, module_type: str) -> List[str]:
        """
        List modules of a specific type.
        
        Args:
            module_type: Type of modules to list
            
        Returns:
            List of module names of the specified type
        """
        matching_modules = []
        for name in self._modules:
            try:
                info = self.get_module_info(name)
                if info['type'] == module_type:
                    matching_modules.append(name)
            except Exception:
                continue
        
        return matching_modules
    
    def clear_cache(self) -> None:
        """Clear all cached module instances."""
        self._instances.clear()
        print("Cleared module instance cache")


# Global registry instance
_global_registry = ModuleRegistry()


def register_module(name: str, module_class: Type[BaseEnhancementModule]) -> None:
    """
    Register a module in the global registry.
    
    Args:
        name: Unique name for the module
        module_class: Module class to register
    """
    _global_registry.register(name, module_class)


def get_module(name: str, config: Dict[str, Any] = None) -> BaseEnhancementModule:
    """
    Get a module instance from the global registry.
    
    Args:
        name: Name of the module
        config: Configuration for the module
        
    Returns:
        Module instance
    """
    return _global_registry.get_or_create_module(name, config)


def list_available_modules() -> List[str]:
    """
    List all available modules in the global registry.
    
    Returns:
        List of module names
    """
    return _global_registry.list_modules()


def get_module_info(name: str) -> Dict[str, Any]:
    """
    Get information about a module from the global registry.
    
    Args:
        name: Name of the module
        
    Returns:
        Module information dictionary
    """
    return _global_registry.get_module_info(name)
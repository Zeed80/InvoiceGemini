"""
Dependency injection container for the application.
"""
from typing import Dict, Any, Callable, Type, Optional
import logging
from functools import lru_cache


class ServiceNotFoundError(Exception):
    """Exception raised when a service is not found in the container."""
    pass


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_service(self, name: str, service: Any, singleton: bool = False):
        """Register a service instance."""
        if singleton:
            self._singletons[name] = service
        else:
            self._services[name] = service
        self.logger.debug(f"Registered service: {name} (singleton={singleton})")
        
    def register_factory(self, name: str, factory: Callable, singleton: bool = False):
        """Register a factory function for creating services."""
        self._factories[name] = (factory, singleton)
        self.logger.debug(f"Registered factory: {name} (singleton={singleton})")
        
    def get(self, name: str) -> Any:
        """Get a service by name."""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
            
        # Check regular services
        if name in self._services:
            return self._services[name]
            
        # Check factories
        if name in self._factories:
            factory, is_singleton = self._factories[name]
            instance = factory(self)  # Pass container to factory
            
            if is_singleton:
                self._singletons[name] = instance
                
            return instance
            
        raise ServiceNotFoundError(f"Service '{name}' not found in container")
        
    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return (name in self._services or 
                name in self._singletons or 
                name in self._factories)
                
    def resolve(self, cls: Type) -> Any:
        """Resolve dependencies for a class and create an instance."""
        # Get constructor parameters
        import inspect
        signature = inspect.signature(cls.__init__)
        params = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to resolve by type annotation
            if param.annotation != param.empty:
                service_name = param.annotation.__name__.lower()
                if self.has(service_name):
                    params[param_name] = self.get(service_name)
                elif param.default != param.empty:
                    params[param_name] = param.default
                else:
                    raise ServiceNotFoundError(
                        f"Cannot resolve parameter '{param_name}' for {cls.__name__}"
                    )
                    
        return cls(**params)
        
    def clear(self):
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    return _container


def register_core_services(container: DIContainer):
    """Register core application services."""
    from app.settings_manager import settings_manager
    from app.processing_engine import ModelManager
    from app.field_manager import field_manager
    from app.plugins.plugin_manager import PluginManager
    from app.plugins.universal_plugin_manager import UniversalPluginManager
    
    # Register singletons
    container.register_service('settings_manager', settings_manager, singleton=True)
    container.register_service('field_manager', field_manager, singleton=True)
    
    # Register factories
    container.register_factory(
        'model_manager',
        lambda c: ModelManager(),
        singleton=True
    )
    
    container.register_factory(
        'plugin_manager',
        lambda c: PluginManager(),
        singleton=True
    )
    
    container.register_factory(
        'universal_plugin_manager',
        lambda c: UniversalPluginManager(),
        singleton=True
    )
    
    # Register security services if available
    try:
        from app.security.crypto_manager import CryptoManager
        from app.security.secrets_manager import SecretsManager
        
        container.register_factory(
            'crypto_manager',
            lambda c: CryptoManager(),
            singleton=True
        )
        
        container.register_factory(
            'secrets_manager',
            lambda c: SecretsManager(c.get('crypto_manager')),
            singleton=True
        )
    except ImportError:
        container.logger.warning("Security modules not available")
        
    # Register memory manager if available
    try:
        from app.core.memory_manager import MemoryManager
        
        container.register_factory(
            'memory_manager',
            lambda c: MemoryManager(),
            singleton=True
        )
    except ImportError:
        container.logger.warning("Memory manager not available")
        
    # Register resource manager if available
    try:
        from app.core.resource_manager import ResourceManager
        
        container.register_factory(
            'resource_manager',
            lambda c: ResourceManager(),
            singleton=True
        )
    except ImportError:
        container.logger.warning("Resource manager not available") 
"""
Factory pattern for creating ML models and processors.
"""
from typing import Dict, Any, Optional, Protocol, Tuple
from abc import ABC, abstractmethod
import logging

from app.core.di_container import DIContainer


class ModelProcessor(Protocol):
    """Protocol for model processors."""
    
    def process(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Process an image and return results."""
        ...
        
    def is_available(self) -> bool:
        """Check if the processor is available."""
        ...
        
    def cleanup(self):
        """Cleanup resources."""
        ...


class BaseModelFactory(ABC):
    """Abstract base class for model factories."""
    
    def __init__(self, container: DIContainer):
        self.container = container
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create a processor for the given model type."""
        pass
        
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get list of supported model types."""
        pass


class LayoutLMFactory(BaseModelFactory):
    """Factory for LayoutLM models."""
    
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create LayoutLM processor."""
        try:
            from app.processing.layoutlm_processor import LayoutLMProcessor
            
            # Check memory if available
            memory_mgr = self._get_service('memory_manager')
            if memory_mgr and not memory_mgr.check_memory_available(2000):  # 2GB
                self.logger.error("Not enough memory for LayoutLM")
                return None
                
            return LayoutLMProcessor(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to create LayoutLM processor: {e}")
            return None
            
    def get_supported_models(self) -> list[str]:
        """Get supported models."""
        return ['layoutlm', 'layoutlmv3']
        
    def _get_service(self, name: str) -> Optional[Any]:
        """Get service from container if available."""
        try:
            return self.container.get(name)
        except (KeyError, AttributeError, Exception) as e:
            # Сервис недоступен в контейнере - возвращаем None
            return None


class DonutFactory(BaseModelFactory):
    """Factory for Donut models."""
    
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create Donut processor."""
        try:
            from app.processing.donut_processor import DonutProcessor
            
            # Check memory if available
            memory_mgr = self._get_service('memory_manager')
            if memory_mgr and not memory_mgr.check_memory_available(3000):  # 3GB
                self.logger.error("Not enough memory for Donut")
                return None
                
            return DonutProcessor(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to create Donut processor: {e}")
            return None
            
    def get_supported_models(self) -> list[str]:
        """Get supported models."""
        return ['donut']
        
    def _get_service(self, name: str) -> Optional[Any]:
        """Get service from container if available."""
        try:
            return self.container.get(name)
        except (KeyError, AttributeError, Exception) as e:
            # Сервис недоступен в контейнере - возвращаем None для Donut
            return None


class GeminiFactory(BaseModelFactory):
    """Factory for Gemini models."""
    
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create Gemini processor."""
        try:
            from app.gemini_processor import GeminiProcessor
            
            # Get secrets manager if available
            secrets_mgr = self._get_service('secrets_manager')
            if secrets_mgr:
                api_key = secrets_mgr.get_secret('google_api_key')
                kwargs['api_key'] = api_key
                
            return GeminiProcessor(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to create Gemini processor: {e}")
            return None
            
    def get_supported_models(self) -> list[str]:
        """Get supported models."""
        return ['gemini', 'gemini-pro', 'gemini-flash']
        
    def _get_service(self, name: str) -> Optional[Any]:
        """Get service from container if available."""
        try:
            return self.container.get(name)
        except (KeyError, AttributeError, Exception) as e:
            # Сервис недоступен в контейнере - возвращаем None для Gemini
            return None


class LLMPluginFactory(BaseModelFactory):
    """Factory for LLM plugin models."""
    
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create LLM plugin processor."""
        try:
            plugin_manager = self.container.get('plugin_manager')
            plugin_id = kwargs.get('plugin_id')
            
            if not plugin_id:
                self.logger.error("No plugin_id provided")
                return None
                
            # Load plugin
            plugin = plugin_manager.get_plugin(plugin_id)
            if not plugin:
                self.logger.error(f"Plugin not found: {plugin_id}")
                return None
                
            # Check if plugin has required methods
            if not hasattr(plugin, 'extract_invoice_data'):
                self.logger.error(f"Plugin {plugin_id} doesn't have extract_invoice_data method")
                return None
                
            return plugin
            
        except Exception as e:
            self.logger.error(f"Failed to create LLM plugin processor: {e}")
            return None
            
    def get_supported_models(self) -> list[str]:
        """Get supported models."""
        return ['cloud_llm', 'local_llm']
        
        
class ModelFactoryRegistry:
    """Registry for model factories."""
    
    def __init__(self, container: DIContainer):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self._factories: Dict[str, BaseModelFactory] = {}
        
        # Register default factories
        self._register_default_factories()
        
    def _register_default_factories(self):
        """Register default model factories."""
        self.register_factory('layoutlm', LayoutLMFactory(self.container))
        self.register_factory('donut', DonutFactory(self.container))
        self.register_factory('gemini', GeminiFactory(self.container))
        self.register_factory('cloud_llm', LLMPluginFactory(self.container))
        self.register_factory('local_llm', LLMPluginFactory(self.container))
        
    def register_factory(self, model_type: str, factory: BaseModelFactory):
        """Register a model factory."""
        self._factories[model_type] = factory
        self.logger.debug(f"Registered factory for model type: {model_type}")
        
    def create_processor(self, model_type: str, **kwargs) -> Optional[ModelProcessor]:
        """Create a processor for the given model type."""
        if model_type not in self._factories:
            self.logger.error(f"No factory registered for model type: {model_type}")
            return None
            
        factory = self._factories[model_type]
        return factory.create_processor(model_type, **kwargs)
        
    def get_supported_models(self) -> list[str]:
        """Get all supported model types."""
        models = []
        for factory in self._factories.values():
            models.extend(factory.get_supported_models())
        return list(set(models))  # Remove duplicates
        
    def has_factory(self, model_type: str) -> bool:
        """Check if a factory is registered for model type."""
        return model_type in self._factories 
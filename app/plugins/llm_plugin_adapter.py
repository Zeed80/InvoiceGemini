"""
Адаптер для интеграции существующих LLM плагинов с новой универсальной системой плагинов
"""
from typing import Dict, Any, Optional
from .base_plugin import LLMPlugin as NewLLMPlugin, PluginMetadata, PluginType, PluginCapability
from .base_llm_plugin import BaseLLMPlugin

class LLMPluginAdapter(NewLLMPlugin):
    """
    Адаптер для интеграции существующих BaseLLMPlugin с новой системой плагинов
    """
    
    def __init__(self, legacy_plugin: BaseLLMPlugin, **kwargs):
        """
        Инициализация адаптера
        
        Args:
            legacy_plugin: Экземпляр старого LLM плагина
            **kwargs: Дополнительные параметры
        """
        # Инициализируем новый базовый класс
        super().__init__(
            model_name=legacy_plugin.model_name,
            api_key=legacy_plugin.api_key,
            config=kwargs.get('config', {})
        )
        
        # Сохраняем ссылку на старый плагин
        self.legacy_plugin = legacy_plugin
        self.provider_name = legacy_plugin.provider_name
        self.provider_config = legacy_plugin.provider_config
        
        # Копируем состояние
        self.is_loaded = legacy_plugin.is_loaded
        self.client = legacy_plugin.client
        self.generation_config = getattr(legacy_plugin, 'generation_config', {})
    
    @property
    def metadata(self) -> PluginMetadata:
        """Возвращает метаданные плагина"""
        capabilities = [PluginCapability.TEXT]
        
        # Проверяем поддержку изображений
        if hasattr(self.legacy_plugin, 'provider_config') and \
           self.legacy_plugin.provider_config and \
           self.legacy_plugin.provider_config.supports_vision:
            capabilities.append(PluginCapability.VISION)
        
        # Проверяем поддержку обучения
        if self.legacy_plugin.supports_training():
            capabilities.append(PluginCapability.TRAINING)
        
        return PluginMetadata(
            name=f"{self.provider_config.display_name} Plugin",
            version="1.0.0",
            description=f"LLM плагин для {self.provider_config.display_name}",
            author="InvoiceGemini Team",
            plugin_type=PluginType.LLM,
            capabilities=capabilities,
            dependencies=[]
        )
    
    def initialize(self) -> bool:
        """Инициализирует плагин"""
        try:
            if not self.legacy_plugin.is_loaded:
                success = self.legacy_plugin.load_model()
                if success:
                    self.is_loaded = True
                    self.client = self.legacy_plugin.client
                return success
            else:
                self.is_loaded = True
                return True
        except Exception as e:
            print(f"Ошибка инициализации LLM плагина: {e}")
            return False
    
    def cleanup(self):
        """Очищает ресурсы плагина"""
        try:
            if hasattr(self.legacy_plugin, 'cleanup'):
                self.legacy_plugin.cleanup()
            self.is_loaded = False
            self.client = None
        except Exception as e:
            print(f"Ошибка очистки LLM плагина: {e}")
    
    def load_model(self) -> bool:
        """Загружает LLM модель"""
        return self.legacy_plugin.load_model()
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ на промпт"""
        return self.legacy_plugin.generate_response(
            prompt=prompt,
            image_path=kwargs.get('image_path'),
            image_context=kwargs.get('image_context', '')
        )
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Обрабатывает данные через LLM"""
        if isinstance(input_data, str):
            # Если входные данные - это путь к изображению
            if input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                return self.legacy_plugin.process_image(
                    image_path=input_data,
                    ocr_lang=kwargs.get('ocr_lang'),
                    custom_prompt=kwargs.get('custom_prompt')
                )
            else:
                # Если это текстовый промпт
                return self.generate_response(input_data, **kwargs)
        elif isinstance(input_data, dict):
            if 'image_path' in input_data:
                return self.legacy_plugin.process_image(
                    image_path=input_data['image_path'],
                    ocr_lang=input_data.get('ocr_lang'),
                    custom_prompt=input_data.get('custom_prompt')
                )
            elif 'prompt' in input_data:
                return self.generate_response(input_data['prompt'], **kwargs)
        
        raise ValueError("Неподдерживаемый тип входных данных для LLM")
    
    def extract_invoice_data(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Извлекает данные из счета-фактуры
        Совместимый метод для интеграции с основным приложением
        """
        try:
            return self.legacy_plugin.process_image(
                image_path=image_path,
                custom_prompt=prompt
            )
        except Exception as e:
            print(f"Ошибка извлечения данных из счета: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        base_info = self.legacy_plugin.get_model_info()
        
        # Добавляем информацию о новой системе плагинов
        base_info.update({
            "plugin_system": "universal",
            "adapter_version": "1.0.0",
            "capabilities": [cap.value for cap in self.metadata.capabilities]
        })
        
        return base_info
    
    def get_training_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию для обучения"""
        if hasattr(self.legacy_plugin, 'get_training_config'):
            return self.legacy_plugin.get_training_config()
        
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "supports_fine_tuning": False,
            "generation_config": self.generation_config
        }

def create_llm_adapter(legacy_plugin: BaseLLMPlugin, **kwargs) -> LLMPluginAdapter:
    """
    Создает адаптер для старого LLM плагина
    
    Args:
        legacy_plugin: Экземпляр старого LLM плагина
        **kwargs: Дополнительные параметры
        
    Returns:
        LLMPluginAdapter: Адаптер для интеграции с новой системой
    """
    return LLMPluginAdapter(legacy_plugin, **kwargs)

def adapt_all_llm_plugins(plugin_manager) -> Dict[str, LLMPluginAdapter]:
    """
    Создает адаптеры для всех существующих LLM плагинов
    
    Args:
        plugin_manager: Экземпляр старого PluginManager
        
    Returns:
        Dict[str, LLMPluginAdapter]: Словарь адаптеров
    """
    adapters = {}
    
    try:
        # Получаем все доступные плагины из старого менеджера
        available_plugins = plugin_manager.get_available_plugins()
        
        for plugin_id in available_plugins:
            try:
                # Создаем экземпляр старого плагина
                legacy_instance = plugin_manager.create_plugin_instance(plugin_id)
                
                if legacy_instance:
                    # Создаем адаптер
                    adapter = create_llm_adapter(legacy_instance)
                    adapters[plugin_id] = adapter
                    print(f"✅ Создан адаптер для плагина: {plugin_id}")
                
            except Exception as e:
                print(f"❌ Ошибка создания адаптера для плагина {plugin_id}: {e}")
    
    except Exception as e:
        print(f"❌ Ошибка адаптации плагинов: {e}")
    
    return adapters 
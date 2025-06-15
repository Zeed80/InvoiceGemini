"""
Базовые классы для системы плагинов InvoiceGemini
Поддержка различных типов плагинов: LLM, обработка, просмотр, экспорт
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
from pathlib import Path

class PluginType(Enum):
    """Типы плагинов в системе"""
    LLM = "llm"                    # LLM модели для извлечения данных
    PROCESSOR = "processor"        # Обработка документов (OCR, предобработка)
    VIEWER = "viewer"             # Просмотр и редактирование результатов
    EXPORTER = "exporter"         # Экспорт в различные форматы
    VALIDATOR = "validator"       # Валидация данных
    TRANSFORMER = "transformer"   # Трансформация данных

class PluginCapability(Enum):
    """Возможности плагинов"""
    VISION = "vision"             # Работа с изображениями
    TEXT = "text"                 # Работа с текстом
    TRAINING = "training"         # Поддержка обучения
    STREAMING = "streaming"       # Потоковая обработка
    BATCH = "batch"              # Пакетная обработка
    ASYNC = "async"              # Асинхронная обработка

class PluginMetadata:
    """Метаданные плагина"""
    
    def __init__(self, 
                 name: str,
                 version: str,
                 description: str,
                 author: str,
                 plugin_type: PluginType,
                 capabilities: List[PluginCapability] = None,
                 dependencies: List[str] = None,
                 config_schema: Dict[str, Any] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.plugin_type = plugin_type
        self.capabilities = capabilities or []
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метаданные в словарь"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "type": self.plugin_type.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "dependencies": self.dependencies,
            "config_schema": self.config_schema
        }

class BasePlugin(ABC):
    """
    Базовый класс для всех плагинов в системе InvoiceGemini
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация базового плагина
        
        Args:
            config: Конфигурация плагина
        """
        self.config = config or {}
        self.is_loaded = False
        self.is_enabled = True
        self._metadata = None
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Возвращает метаданные плагина"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Инициализирует плагин
        
        Returns:
            bool: True если инициализация успешна
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Очищает ресурсы плагина"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Валидирует конфигурацию плагина
        
        Args:
            config: Конфигурация для валидации
            
        Returns:
            bool: True если конфигурация валидна
        """
        # Базовая валидация по схеме
        schema = self.metadata.config_schema
        if not schema:
            return True
        
        # Простая валидация обязательных полей
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in config:
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус плагина"""
        return {
            "name": self.metadata.name,
            "type": self.metadata.plugin_type.value,
            "is_loaded": self.is_loaded,
            "is_enabled": self.is_enabled,
            "version": self.metadata.version
        }

class ProcessorPlugin(BasePlugin):
    """Базовый класс для плагинов обработки"""
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Обрабатывает входные данные
        
        Args:
            input_data: Входные данные
            **kwargs: Дополнительные параметры
            
        Returns:
            Any: Обработанные данные
        """
        pass
    
    def supports_format(self, format_type: str) -> bool:
        """
        Проверяет поддержку формата
        
        Args:
            format_type: Тип формата
            
        Returns:
            bool: True если формат поддерживается
        """
        return True

class LLMPlugin(ProcessorPlugin):
    """Базовый класс для LLM плагинов"""
    
    def __init__(self, model_name: str = None, api_key: str = None, **kwargs):
        super().__init__(kwargs.get('config', {}))
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """Загружает LLM модель"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ на промпт"""
        pass
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Обрабатывает данные через LLM"""
        if isinstance(input_data, str):
            return self.generate_response(input_data, **kwargs)
        elif isinstance(input_data, dict) and 'prompt' in input_data:
            return self.generate_response(input_data['prompt'], **kwargs)
        else:
            raise ValueError("Неподдерживаемый тип входных данных для LLM")

class ViewerPlugin(BasePlugin):
    """Базовый класс для плагинов просмотра"""
    
    @abstractmethod
    def create_viewer(self, data: Any, parent=None) -> Any:
        """
        Создает виджет просмотра
        
        Args:
            data: Данные для просмотра
            parent: Родительский виджет
            
        Returns:
            Any: Виджет просмотра
        """
        pass
    
    @abstractmethod
    def update_view(self, viewer: Any, data: Any):
        """
        Обновляет представление данных
        
        Args:
            viewer: Виджет просмотра
            data: Новые данные
        """
        pass

class ExporterPlugin(BasePlugin):
    """Базовый класс для плагинов экспорта"""
    
    @abstractmethod
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """
        Экспортирует данные в файл
        
        Args:
            data: Данные для экспорта
            output_path: Путь к выходному файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если экспорт успешен
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Возвращает список поддерживаемых форматов"""
        pass
    
    def get_file_extension(self) -> str:
        """Возвращает расширение файла по умолчанию"""
        formats = self.get_supported_formats()
        return formats[0] if formats else "txt"

class ValidatorPlugin(BasePlugin):
    """Базовый класс для плагинов валидации"""
    
    @abstractmethod
    def validate(self, data: Any) -> Dict[str, Any]:
        """
        Валидирует данные
        
        Args:
            data: Данные для валидации
            
        Returns:
            Dict[str, Any]: Результат валидации с ошибками и предупреждениями
        """
        pass
    
    def is_valid(self, data: Any) -> bool:
        """
        Проверяет валидность данных
        
        Args:
            data: Данные для проверки
            
        Returns:
            bool: True если данные валидны
        """
        result = self.validate(data)
        return len(result.get('errors', [])) == 0

class TransformerPlugin(BasePlugin):
    """Базовый класс для плагинов трансформации данных"""
    
    @abstractmethod
    def transform(self, input_data: Any, target_format: str, **kwargs) -> Any:
        """
        Трансформирует данные из одного формата в другой
        
        Args:
            input_data: Входные данные
            target_format: Целевой формат
            **kwargs: Дополнительные параметры
            
        Returns:
            Any: Трансформированные данные
        """
        pass
    
    @abstractmethod
    def get_supported_transformations(self) -> Dict[str, List[str]]:
        """
        Возвращает поддерживаемые трансформации
        
        Returns:
            Dict[str, List[str]]: Словарь {входной_формат: [выходные_форматы]}
        """
        pass

# Утилитарные функции для работы с плагинами
def create_plugin_metadata(name: str, version: str, description: str, 
                          author: str, plugin_type: PluginType,
                          **kwargs) -> PluginMetadata:
    """Создает метаданные плагина"""
    return PluginMetadata(
        name=name,
        version=version,
        description=description,
        author=author,
        plugin_type=plugin_type,
        **kwargs
    )

def load_plugin_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Загружает конфигурацию плагина из файла"""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации плагина {config_path}: {e}")
        return {}

def save_plugin_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Сохраняет конфигурацию плагина в файл"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения конфигурации плагина {config_path}: {e}") 
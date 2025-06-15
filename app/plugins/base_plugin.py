"""
Базовые классы для системы плагинов InvoiceGemini
Поддержка различных типов плагинов: LLM, обработка, просмотр, экспорт, импорт, интеграции
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import json
from pathlib import Path

class PluginType(Enum):
    """Типы плагинов в системе"""
    LLM = "llm"                    # LLM модели для извлечения данных
    PROCESSOR = "processor"        # Обработка документов (OCR, предобработка)
    VIEWER = "viewer"             # Просмотр и редактирование результатов
    EXPORTER = "exporter"         # Экспорт в различные форматы
    IMPORTER = "importer"         # Импорт из различных источников
    VALIDATOR = "validator"       # Валидация данных
    TRANSFORMER = "transformer"   # Трансформация данных
    INTEGRATION = "integration"   # Интеграции с внешними системами
    WORKFLOW = "workflow"         # Обработка рабочих процессов
    NOTIFICATION = "notification" # Уведомления и алерты

class PluginCapability(Enum):
    """Возможности плагинов"""
    VISION = "vision"             # Работа с изображениями
    TEXT = "text"                 # Работа с текстом
    TRAINING = "training"         # Поддержка обучения
    STREAMING = "streaming"       # Потоковая обработка
    BATCH = "batch"              # Пакетная обработка
    ASYNC = "async"              # Асинхронная обработка
    REALTIME = "realtime"        # Реального времени
    API = "api"                  # API интеграции
    DATABASE = "database"        # Работа с БД
    CLOUD = "cloud"              # Облачные возможности

class PluginPriority(Enum):
    """Приоритеты плагинов"""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5

class PluginStatus(Enum):
    """Статусы плагинов"""
    UNLOADED = "unloaded"
    LOADING = "loading" 
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

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
                 config_schema: Dict[str, Any] = None,
                 priority: PluginPriority = PluginPriority.NORMAL,
                 supported_formats: List[str] = None,
                 min_python_version: str = "3.8",
                 website: str = None,
                 license: str = "MIT",
                 keywords: List[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.plugin_type = plugin_type
        self.capabilities = capabilities or []
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}
        self.priority = priority
        self.supported_formats = supported_formats or []
        self.min_python_version = min_python_version
        self.website = website
        self.license = license
        self.keywords = keywords or []
    
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
            "config_schema": self.config_schema,
            "priority": self.priority.value,
            "supported_formats": self.supported_formats,
            "min_python_version": self.min_python_version,
            "website": self.website,
            "license": self.license,
            "keywords": self.keywords
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
        self.status = PluginStatus.UNLOADED
        self.is_enabled = True
        self._metadata = None
        self._progress_callback: Optional[Callable] = None
        self._last_error: Optional[str] = None
    
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
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Устанавливает callback для отслеживания прогресса"""
        self._progress_callback = callback
    
    def report_progress(self, progress: int, message: str = ""):
        """Сообщает о прогрессе выполнения"""
        if self._progress_callback:
            self._progress_callback(progress, message)
    
    def get_last_error(self) -> Optional[str]:
        """Возвращает последнюю ошибку"""
        return self._last_error
    
    def set_error(self, error: str):
        """Устанавливает ошибку"""
        self._last_error = error
        self.status = PluginStatus.ERROR
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Валидирует конфигурацию плагина
        
        Args:
            config: Конфигурация для валидации
            
        Returns:
            bool: True если конфигурация валидна
        """
        schema = self.metadata.config_schema
        if not schema:
            return True
        
        # Простая валидация обязательных полей
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in config:
                self.set_error(f"Отсутствует обязательное поле: {field}")
                return False
        
        # Валидация типов
        field_types = schema.get("types", {})
        for field, expected_type in field_types.items():
            if field in config:
                if not isinstance(config[field], expected_type):
                    self.set_error(f"Неверный тип поля {field}")
                    return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус плагина"""
        return {
            "name": self.metadata.name,
            "type": self.metadata.plugin_type.value,
            "status": self.status.value,
            "is_enabled": self.is_enabled,
            "version": self.metadata.version,
            "priority": self.metadata.priority.value,
            "last_error": self._last_error
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

class ImporterPlugin(BasePlugin):
    """Базовый класс для плагинов импорта"""
    
    @abstractmethod
    def import_data(self, source: str, **kwargs) -> Dict[str, Any]:
        """
        Импортирует данные из источника
        
        Args:
            source: Источник данных
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Импортированные данные
        """
        pass
    
    @abstractmethod
    def get_supported_sources(self) -> List[str]:
        """
        Возвращает поддерживаемые источники данных
        
        Returns:
            List[str]: Список поддерживаемых источников
        """
        pass
    
    def validate_source(self, source: str) -> bool:
        """
        Проверяет доступность источника
        
        Args:
            source: Источник для проверки
            
        Returns:
            bool: True если источник доступен
        """
        return True

class IntegrationPlugin(BasePlugin):
    """Базовый класс для плагинов интеграции с внешними системами"""
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """
        Устанавливает соединение с внешней системой
        
        Returns:
            bool: True если соединение установлено
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Разрывает соединение с внешней системой"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Тестирует соединение
        
        Returns:
            bool: True если соединение работает
        """
        pass
    
    @abstractmethod
    def sync_data(self, data: Any, direction: str = "export") -> Dict[str, Any]:
        """
        Синхронизирует данные с внешней системой
        
        Args:
            data: Данные для синхронизации
            direction: Направление синхронизации ("export", "import", "both")
            
        Returns:
            Dict[str, Any]: Результат синхронизации
        """
        pass
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Возвращает статус соединения"""
        return {
            "connected": self.test_connection(),
            "last_sync": getattr(self, '_last_sync', None),
            "sync_status": getattr(self, '_sync_status', 'unknown')
        }

class WorkflowPlugin(BasePlugin):
    """Базовый класс для плагинов рабочих процессов"""
    
    @abstractmethod
    def execute_workflow(self, workflow_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Выполняет рабочий процесс
        
        Args:
            workflow_data: Данные для обработки
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результат выполнения
        """
        pass
    
    @abstractmethod
    def get_workflow_steps(self) -> List[Dict[str, Any]]:
        """
        Возвращает шаги рабочего процесса
        
        Returns:
            List[Dict[str, Any]]: Список шагов
        """
        pass
    
    def validate_workflow_data(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Валидирует данные рабочего процесса
        
        Args:
            workflow_data: Данные для валидации
            
        Returns:
            bool: True если данные валидны
        """
        return True

class NotificationPlugin(BasePlugin):
    """Базовый класс для плагинов уведомлений"""
    
    @abstractmethod
    def send_notification(self, message: str, **kwargs) -> bool:
        """
        Отправляет уведомление
        
        Args:
            message: Сообщение для отправки
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если уведомление отправлено
        """
        pass
    
    @abstractmethod
    def get_notification_channels(self) -> List[str]:
        """
        Возвращает доступные каналы уведомлений
        
        Returns:
            List[str]: Список каналов
        """
        pass
    
    def format_message(self, template: str, data: Dict[str, Any]) -> str:
        """
        Форматирует сообщение по шаблону
        
        Args:
            template: Шаблон сообщения
            data: Данные для подстановки
            
        Returns:
            str: Отформатированное сообщение
        """
        try:
            return template.format(**data)
        except KeyError as e:
            self.set_error(f"Ошибка форматирования: отсутствует ключ {e}")
            return template

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
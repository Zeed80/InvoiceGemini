#!/usr/bin/env python3
"""
Unified Plugin Management System for InvoiceGemini
Унифицированная система управления плагинами с динамической загрузкой
"""

import os
import sys
import json
import importlib
import importlib.util
import inspect
import threading
from typing import Dict, List, Type, Optional, Any, Union, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import weakref

from PyQt6.QtCore import QObject, pyqtSignal

from .base_plugin import BasePlugin, PluginType, PluginStatus, PluginMetadata
from .base_llm_plugin import BaseLLMPlugin
from ..settings_manager import settings_manager


class PluginEventType(Enum):
    """Типы событий плагинов"""
    LOADED = "loaded"
    UNLOADED = "unloaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATED = "updated"


@dataclass
class PluginEvent:
    """Событие плагина"""
    plugin_id: str
    event_type: PluginEventType
    timestamp: datetime
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class PluginRegistry:
    """Реестр плагинов с метаданными"""
    
    def __init__(self):
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self._by_type: Dict[PluginType, Set[str]] = {pt: set() for pt in PluginType}
        self._dependencies: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
    
    def register(self, plugin_id: str, plugin_class: Type[BasePlugin], 
                metadata: PluginMetadata, file_path: str = None):
        """Регистрировать плагин"""
        with self._lock:
            self._plugins[plugin_id] = {
                'class': plugin_class,
                'metadata': metadata,
                'file_path': file_path,
                'instance': None,
                'enabled': True,
                'registered_at': datetime.now()
            }
            
            # Добавляем в индекс по типу
            self._by_type[metadata.plugin_type].add(plugin_id)
            
            # Обрабатываем зависимости
            deps = set(metadata.dependencies)
            self._dependencies[plugin_id] = deps
            
            for dep in deps:
                if dep not in self._dependents:
                    self._dependents[dep] = set()
                self._dependents[dep].add(plugin_id)
    
    def unregister(self, plugin_id: str):
        """Разрегистрировать плагин"""
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            
            # Удаляем из индексов
            plugin_info = self._plugins[plugin_id]
            plugin_type = plugin_info['metadata'].plugin_type
            self._by_type[plugin_type].discard(plugin_id)
            
            # Очищаем зависимости
            if plugin_id in self._dependencies:
                deps = self._dependencies[plugin_id]
                for dep in deps:
                    if dep in self._dependents:
                        self._dependents[dep].discard(plugin_id)
                del self._dependencies[plugin_id]
            
            if plugin_id in self._dependents:
                del self._dependents[plugin_id]
            
            # Удаляем плагин
            del self._plugins[plugin_id]
            return True
    
    def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Получить информацию о плагине"""
        with self._lock:
            return self._plugins.get(plugin_id)
    
    def get_by_type(self, plugin_type: PluginType) -> List[str]:
        """Получить плагины по типу"""
        with self._lock:
            return list(self._by_type[plugin_type])
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Получить все плагины"""
        with self._lock:
            return self._plugins.copy()
    
    def get_dependencies(self, plugin_id: str) -> Set[str]:
        """Получить зависимости плагина"""
        with self._lock:
            return self._dependencies.get(plugin_id, set()).copy()
    
    def get_dependents(self, plugin_id: str) -> Set[str]:
        """Получить зависящие плагины"""
        with self._lock:
            return self._dependents.get(plugin_id, set()).copy()
    
    def check_dependencies(self, plugin_id: str) -> bool:
        """Проверить доступность зависимостей"""
        with self._lock:
            deps = self._dependencies.get(plugin_id, set())
            for dep in deps:
                if dep not in self._plugins or not self._plugins[dep]['enabled']:
                    return False
            return True


class PluginLoader:
    """Загрузчик плагинов с динамической загрузкой"""
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._file_timestamps: Dict[str, float] = {}
    
    def load_plugin_from_file(self, file_path: str) -> List[Type[BasePlugin]]:
        """Загрузить плагин из файла"""
        try:
            file_path = Path(file_path)
            
            # Проверяем изменения файла
            current_time = file_path.stat().st_mtime
            if str(file_path) in self._file_timestamps:
                if current_time <= self._file_timestamps[str(file_path)]:
                    # Файл не изменился, возвращаем кэшированный результат
                    return self._get_cached_classes(str(file_path))
            
            # Загружаем модуль с правильным контекстом
            module_name = f"app.plugins.models.{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if not spec or not spec.loader:
                logging.warning(f"Cannot load plugin spec: {file_path}")
                return []
            
            # Если модуль уже загружен, перезагружаем
            if str(file_path) in self._loaded_modules:
                module = self._loaded_modules[str(file_path)]
                importlib.reload(module)
            else:
                module = importlib.util.module_from_spec(spec)
                # Добавляем модуль в sys.modules для правильной работы relative imports
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    self._loaded_modules[str(file_path)] = module
                except Exception as e:
                    # Удаляем из sys.modules если загрузка не удалась
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    raise e
            
            # Обновляем timestamp
            self._file_timestamps[str(file_path)] = current_time
            
            # Ищем классы плагинов
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and 
                    obj.__module__ == module.__name__):
                    plugin_classes.append(obj)
            
            return plugin_classes
            
        except Exception as e:
            logging.error(f"Error loading plugin from {file_path}: {e}")
            return []
    
    def _get_cached_classes(self, file_path: str) -> List[Type[BasePlugin]]:
        """Получить кэшированные классы плагинов"""
        if file_path not in self._loaded_modules:
            return []
        
        module = self._loaded_modules[file_path]
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BasePlugin) and 
                obj != BasePlugin and 
                obj.__module__ == module.__name__):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    def unload_plugin_file(self, file_path: str):
        """Выгрузить плагин из файла"""
        file_path_str = str(file_path)
        if file_path_str in self._loaded_modules:
            del self._loaded_modules[file_path_str]
        if file_path_str in self._file_timestamps:
            del self._file_timestamps[file_path_str]


class UnifiedPluginManager(QObject):
    """
    Унифицированный менеджер плагинов
    
    Объединяет функциональность всех существующих менеджеров:
    - PluginManager (LLM плагины)
    - UniversalPluginManager (все типы плагинов)
    - AdvancedPluginManager (установка/обновление)
    
    Новые возможности:
    - Динамическая загрузка/выгрузка
    - Управление зависимостями
    - Событийная система
    - Автоматическое обновление
    - Производительный реестр
    """
    
    # Сигналы
    plugin_loaded = pyqtSignal(str, dict)  # plugin_id, metadata
    plugin_unloaded = pyqtSignal(str)  # plugin_id
    plugin_enabled = pyqtSignal(str)  # plugin_id
    plugin_disabled = pyqtSignal(str)  # plugin_id
    plugin_error = pyqtSignal(str, str)  # plugin_id, error
    
    def __init__(self, plugins_dir: str = None, auto_scan: bool = True):
        super().__init__()
        
        # Директории
        self.builtin_dir = Path(__file__).parent / "models"
        self.user_dir = Path(plugins_dir) if plugins_dir else Path("plugins/user")
        self.user_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты
        self.registry = PluginRegistry()
        self.loader = PluginLoader()
        
        # Экземпляры плагинов
        self._instances: Dict[str, BasePlugin] = {}
        self._instance_refs: Dict[str, weakref.ref] = {}
        
        # События
        self._event_handlers: Dict[PluginEventType, List[Callable]] = {
            event_type: [] for event_type in PluginEventType
        }
        self._events_history: List[PluginEvent] = []
        
        # Настройки
        self.auto_enable = True
        self.max_events_history = 1000
        
        # Thread safety
        self._lock = threading.RLock()
        
        logging.info(f"🔧 UnifiedPluginManager initialized")
        logging.info(f"   Builtin dir: {self.builtin_dir}")
        logging.info(f"   User dir: {self.user_dir}")
        
        # Автоматическое сканирование
        if auto_scan:
            self.scan_plugins()
    
    def scan_plugins(self, force_reload: bool = False):
        """Сканировать и загрузить плагины"""
        logging.info("🔍 Scanning for plugins...")
        
        loaded_count = 0
        
        # Сканируем встроенные плагины
        if self.builtin_dir.exists():
            loaded_count += self._scan_directory(self.builtin_dir, force_reload)
        
        # Сканируем пользовательские плагины
        if self.user_dir.exists():
            loaded_count += self._scan_directory(self.user_dir, force_reload)
        
        logging.info(f"✅ Plugin scan completed: {loaded_count} plugins loaded")
        return loaded_count
    
    def _scan_directory(self, directory: Path, force_reload: bool = False) -> int:
        """Сканировать директорию на предмет плагинов"""
        loaded_count = 0
        
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            try:
                plugin_classes = self.loader.load_plugin_from_file(file_path)
                
                for plugin_class in plugin_classes:
                    plugin_id = self._generate_plugin_id(plugin_class)
                    
                    # Проверяем, не загружен ли уже
                    if plugin_id in self.registry.get_all() and not force_reload:
                        continue
                    
                    # Получаем метаданные
                    metadata = self._extract_metadata(plugin_class)
                    
                    # Регистрируем
                    self.registry.register(plugin_id, plugin_class, metadata, str(file_path))
                    
                    # Автоматически включаем если настроено
                    if self.auto_enable:
                        self.enable_plugin(plugin_id)
                    
                    loaded_count += 1
                    self._emit_event(plugin_id, PluginEventType.LOADED)
                    
                    logging.info(f"📦 Loaded plugin: {plugin_id} ({metadata.plugin_type.value})")
                
            except Exception as e:
                logging.error(f"Failed to load plugin from {file_path}: {e}")
        
        return loaded_count
    
    def _generate_plugin_id(self, plugin_class: Type[BasePlugin]) -> str:
        """Генерировать ID плагина"""
        # Пытаемся получить ID из класса
        if hasattr(plugin_class, 'PLUGIN_ID'):
            return plugin_class.PLUGIN_ID
        
        # Иначе используем имя класса
        class_name = plugin_class.__name__
        if class_name.endswith('Plugin'):
            class_name = class_name[:-6]  # Убираем 'Plugin'
        
        return class_name.lower()
    
    def _extract_metadata(self, plugin_class: Type[BasePlugin]) -> PluginMetadata:
        """Извлечь метаданные из класса плагина"""
        try:
            # Создаём временный экземпляр для получения метаданных
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            # Очищаем временный экземпляр
            if hasattr(temp_instance, 'cleanup'):
                temp_instance.cleanup()
            
            return metadata
            
        except Exception as e:
            logging.warning(f"Failed to extract metadata from {plugin_class}: {e}")
            
            # Возвращаем базовые метаданные
            return PluginMetadata(
                name=plugin_class.__name__,
                version="1.0.0",
                description="No description available",
                author="Unknown",
                plugin_type=PluginType.PROCESSOR  # По умолчанию
            )
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Включить плагин"""
        with self._lock:
            plugin_info = self.registry.get(plugin_id)
            if not plugin_info:
                logging.error(f"Plugin not found: {plugin_id}")
                return False
            
            # Проверяем зависимости
            if not self.registry.check_dependencies(plugin_id):
                missing_deps = []
                for dep in self.registry.get_dependencies(plugin_id):
                    if dep not in self.registry.get_all():
                        missing_deps.append(dep)
                
                error_msg = f"Missing dependencies for {plugin_id}: {missing_deps}"
                logging.error(error_msg)
                self._emit_event(plugin_id, PluginEventType.ERROR, {"error": error_msg})
                return False
            
            # Создаём экземпляр если нужно
            if plugin_id not in self._instances:
                try:
                    plugin_class = plugin_info['class']
                    instance = plugin_class()
                    
                    # Инициализируем
                    if hasattr(instance, 'initialize'):
                        init_success = instance.initialize({})
                        if not init_success:
                            logging.error(f"Failed to initialize plugin: {plugin_id}")
                            return False
                    
                    self._instances[plugin_id] = instance
                    
                    # Создаём weak reference для автоочистки
                    def cleanup_callback(ref):
                        if plugin_id in self._instance_refs:
                            del self._instance_refs[plugin_id]
                    
                    self._instance_refs[plugin_id] = weakref.ref(instance, cleanup_callback)
                    
                except Exception as e:
                    error_msg = f"Failed to create instance for {plugin_id}: {e}"
                    logging.error(error_msg)
                    self._emit_event(plugin_id, PluginEventType.ERROR, {"error": error_msg})
                    return False
            
            # Включаем
            plugin_info['enabled'] = True
            self._emit_event(plugin_id, PluginEventType.ENABLED)
            self.plugin_enabled.emit(plugin_id)
            
            logging.info(f"✅ Plugin enabled: {plugin_id}")
            return True
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Отключить плагин"""
        with self._lock:
            plugin_info = self.registry.get(plugin_id)
            if not plugin_info:
                return False
            
            # Проверяем зависящие плагины
            dependents = self.registry.get_dependents(plugin_id)
            enabled_dependents = [dep for dep in dependents 
                                if self.registry.get(dep) and self.registry.get(dep)['enabled']]
            
            if enabled_dependents:
                logging.warning(f"Cannot disable {plugin_id}: has enabled dependents {enabled_dependents}")
                return False
            
            # Отключаем
            plugin_info['enabled'] = False
            
            # Удаляем экземпляр
            if plugin_id in self._instances:
                instance = self._instances[plugin_id]
                if hasattr(instance, 'cleanup'):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        logging.error(f"Error during plugin cleanup {plugin_id}: {e}")
                
                del self._instances[plugin_id]
            
            if plugin_id in self._instance_refs:
                del self._instance_refs[plugin_id]
            
            self._emit_event(plugin_id, PluginEventType.DISABLED)
            self.plugin_disabled.emit(plugin_id)
            
            logging.info(f"⏹️ Plugin disabled: {plugin_id}")
            return True
    
    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Получить экземпляр плагина"""
        with self._lock:
            return self._instances.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, BasePlugin]:
        """Получить плагины по типу"""
        with self._lock:
            plugin_ids = self.registry.get_by_type(plugin_type)
            return {pid: self._instances[pid] for pid in plugin_ids 
                   if pid in self._instances}
    
    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Получить информацию о всех доступных плагинах"""
        with self._lock:
            result = {}
            for plugin_id, plugin_info in self.registry.get_all().items():
                result[plugin_id] = {
                    'metadata': asdict(plugin_info['metadata']),
                    'enabled': plugin_info['enabled'],
                    'file_path': plugin_info.get('file_path'),
                    'has_instance': plugin_id in self._instances,
                    'dependencies': list(self.registry.get_dependencies(plugin_id)),
                    'dependents': list(self.registry.get_dependents(plugin_id))
                }
            return result
    
    def reload_plugin(self, plugin_id: str) -> bool:
        """Перезагрузить плагин"""
        with self._lock:
            plugin_info = self.registry.get(plugin_id)
            if not plugin_info:
                return False
            
            file_path = plugin_info.get('file_path')
            if not file_path:
                logging.error(f"No file path for plugin: {plugin_id}")
                return False
            
            # Отключаем плагин
            was_enabled = plugin_info['enabled']
            if was_enabled:
                self.disable_plugin(plugin_id)
            
            # Разрегистрируем
            self.registry.unregister(plugin_id)
            
            # Выгружаем модуль
            self.loader.unload_plugin_file(file_path)
            
            # Перезагружаем
            try:
                plugin_classes = self.loader.load_plugin_from_file(file_path)
                
                for plugin_class in plugin_classes:
                    new_plugin_id = self._generate_plugin_id(plugin_class)
                    if new_plugin_id == plugin_id:
                        metadata = self._extract_metadata(plugin_class)
                        self.registry.register(plugin_id, plugin_class, metadata, file_path)
                        
                        if was_enabled:
                            self.enable_plugin(plugin_id)
                        
                        self._emit_event(plugin_id, PluginEventType.UPDATED)
                        logging.info(f"🔄 Plugin reloaded: {plugin_id}")
                        return True
                
                logging.error(f"Plugin class not found after reload: {plugin_id}")
                return False
                
            except Exception as e:
                logging.error(f"Failed to reload plugin {plugin_id}: {e}")
                return False
    
    def add_event_handler(self, event_type: PluginEventType, handler: Callable):
        """Добавить обработчик событий"""
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: PluginEventType, handler: Callable):
        """Удалить обработчик событий"""
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    def _emit_event(self, plugin_id: str, event_type: PluginEventType, data: Dict[str, Any] = None):
        """Испустить событие плагина"""
        event = PluginEvent(plugin_id, event_type, datetime.now(), data)
        
        # Добавляем в историю
        self._events_history.append(event)
        if len(self._events_history) > self.max_events_history:
            self._events_history.pop(0)
        
        # Вызываем обработчики
        for handler in self._event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Error in event handler: {e}")
    
    def get_events_history(self, limit: int = None) -> List[PluginEvent]:
        """Получить историю событий"""
        if limit:
            return self._events_history[-limit:]
        return self._events_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику плагинов"""
        with self._lock:
            all_plugins = self.registry.get_all()
            
            by_type = {}
            for plugin_type in PluginType:
                by_type[plugin_type.value] = len(self.registry.get_by_type(plugin_type))
            
            return {
                'total_plugins': len(all_plugins),
                'enabled_plugins': sum(1 for p in all_plugins.values() if p['enabled']),
                'active_instances': len(self._instances),
                'by_type': by_type,
                'events_count': len(self._events_history),
                'builtin_dir': str(self.builtin_dir),
                'user_dir': str(self.user_dir)
            }
    
    def cleanup(self):
        """Очистка ресурсов"""
        logging.info("🧹 Cleaning up UnifiedPluginManager...")
        
        with self._lock:
            # Отключаем все плагины
            for plugin_id in list(self._instances.keys()):
                self.disable_plugin(plugin_id)
            
            # Очищаем реестр
            self._instances.clear()
            self._instance_refs.clear()
            
        logging.info("✅ UnifiedPluginManager cleanup completed")


# Глобальный экземпляр для совместимости
_unified_manager = None

def get_unified_plugin_manager() -> UnifiedPluginManager:
    """Получить глобальный экземпляр менеджера плагинов"""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedPluginManager()
    return _unified_manager 
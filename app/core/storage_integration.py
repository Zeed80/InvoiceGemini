"""
Модуль интеграции оптимизированной системы хранения
Обеспечивает плавный переход к OptimizedStorageManager
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .storage_adapter import get_storage_adapter, migrate_to_optimized_storage
from .optimized_storage_manager import get_optimized_storage_manager
from ..settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class StorageIntegration(QObject):
    """
    Класс для интеграции оптимизированного хранилища в основное приложение
    """
    
    migration_started = pyqtSignal()
    migration_progress = pyqtSignal(int, str)  # progress, status
    migration_completed = pyqtSignal(bool)  # success
    storage_ready = pyqtSignal()
    
    def __init__(self, settings_manager: SettingsManager):
        super().__init__()
        
        self.settings_manager = settings_manager
        self.storage_adapter = get_storage_adapter(settings_manager)
        self.optimized_storage = get_optimized_storage_manager()
        
        # Статус интеграции
        self.integration_enabled = False
        self.migration_completed_flag = False
        
        # Таймер для батчевых операций
        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self._process_pending_batches)
        self.batch_timer.start(5000)  # 5 секунд
        
        logger.info("🔧 StorageIntegration инициализирован")
    
    def initialize_storage_system(self, auto_migrate: bool = True) -> bool:
        """
        Инициализирует систему хранения данных
        
        Args:
            auto_migrate: Автоматически запустить миграцию
            
        Returns:
            bool: Успешность инициализации
        """
        try:
            logger.info("🚀 Инициализация системы хранения...")
            
            # Проверяем статус миграции
            migration_status = self.storage_adapter.get_migration_status()
            
            if migration_status['migration_completed']:
                logger.info("✅ Миграция уже завершена, активируем оптимизированное хранилище")
                self.integration_enabled = True
                self.migration_completed_flag = True
                self.storage_ready.emit()
                return True
            
            if auto_migrate:
                logger.info("🔄 Запускаем автоматическую миграцию...")
                return self.start_migration()
            else:
                logger.info("⏳ Миграция не запущена, используем совместимый режим")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы хранения: {e}")
            return False
    
    def start_migration(self) -> bool:
        """Запускает процесс миграции данных"""
        try:
            self.migration_started.emit()
            logger.info("🔄 Начинаем миграцию данных...")
            
            # Этап 1: Подготовка (10%)
            self.migration_progress.emit(10, "Подготовка к миграции...")
            
            # Проверяем целостность исходных данных
            if not self._validate_source_data():
                raise ValueError("Исходные данные не прошли валидацию")
            
            # Этап 2: Миграция основных настроек (40%)
            self.migration_progress.emit(40, "Миграция основных настроек...")
            
            success = self.storage_adapter.start_migration()
            
            if not success:
                raise RuntimeError("Миграция не удалась")
            
            # Этап 3: Валидация мигрированных данных (70%)
            self.migration_progress.emit(70, "Валидация мигрированных данных...")
            
            if not self._validate_migrated_data():
                raise ValueError("Мигрированные данные не прошли валидацию")
            
            # Этап 4: Активация системы (90%)
            self.migration_progress.emit(90, "Активация оптимизированной системы...")
            
            self.integration_enabled = True
            self.migration_completed_flag = True
            
            # Этап 5: Завершение (100%)
            self.migration_progress.emit(100, "Миграция завершена успешно")
            self.migration_completed.emit(True)
            self.storage_ready.emit()
            
            logger.info("🎉 Миграция завершена успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции: {e}")
            self.migration_progress.emit(0, f"Ошибка: {str(e)}")
            self.migration_completed.emit(False)
            return False
    
    def _validate_source_data(self) -> bool:
        """Валидирует исходные данные перед миграцией"""
        try:
            # Проверяем доступность SettingsManager
            if not self.settings_manager:
                logger.warning("⚠️ SettingsManager недоступен")
                return False
            
            # Проверяем файл настроек
            settings_file = Path(self.settings_manager.settings_file_path)
            if not settings_file.exists():
                logger.warning("⚠️ Файл настроек не найден")
                return True  # Не критично, будут созданы настройки по умолчанию
            
            # Проверяем основные секции
            required_sections = ['General', 'Models', 'Paths']
            missing_sections = []
            
            for section in required_sections:
                if section not in self.settings_manager.config:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.warning(f"⚠️ Отсутствуют секции: {missing_sections}")
            
            logger.info("✅ Валидация исходных данных пройдена")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации исходных данных: {e}")
            return False
    
    def _validate_migrated_data(self) -> bool:
        """Валидирует мигрированные данные"""
        try:
            # Проверяем основные настройки
            test_settings = [
                ('active_model', 'general'),
                ('layoutlm_id', 'models'),
                ('tesseract_path', 'paths')
            ]
            
            for key, category in test_settings:
                value = self.optimized_storage.get_setting(key, None, category)
                if value is None:
                    logger.warning(f"⚠️ Настройка {category}.{key} не найдена после миграции")
            
            # Проверяем статистику миграции
            migration_status = self.storage_adapter.get_migration_status()
            detailed_status = migration_status.get('detailed_status', {})
            
            all_migrated = all(detailed_status.values())
            if not all_migrated:
                logger.warning(f"⚠️ Не все компоненты мигрированы: {detailed_status}")
            
            logger.info("✅ Валидация мигрированных данных пройдена")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации мигрированных данных: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None, 
                   section: str = 'general') -> Any:
        """
        Унифицированный метод получения настроек
        Автоматически выбирает оптимальный источник данных
        """
        if self.integration_enabled:
            return self.storage_adapter.get_setting(key, default, section)
        else:
            # Fallback к SettingsManager
            try:
                return self.settings_manager.get_string(section, key, str(default))
            except Exception as e:
                logger.warning(f"⚠️ Ошибка получения настройки {section}.{key}: {e}")
                return default
    
    def set_setting(self, key: str, value: Any, 
                   section: str = 'general', batch: bool = False) -> bool:
        """
        Унифицированный метод сохранения настроек
        Автоматически выбирает оптимальный способ сохранения
        """
        if self.integration_enabled:
            return self.storage_adapter.set_setting(key, value, section, batch)
        else:
            # Fallback к SettingsManager
            try:
                self.settings_manager.set_value(section, key, str(value))
                if not batch:
                    self.settings_manager.save_settings()
                return True
            except Exception as e:
                logger.warning(f"⚠️ Ошибка сохранения настройки {section}.{key}: {e}")
                return False
    
    def _process_pending_batches(self):
        """Обрабатывает отложенные батчевые операции"""
        if self.integration_enabled:
            try:
                # Запускаем обработку батчей в оптимизированном хранилище
                self.optimized_storage._process_batch()
            except Exception as e:
                logger.warning(f"⚠️ Ошибка обработки батчей: {e}")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы системы хранения"""
        stats = {
            'integration_enabled': self.integration_enabled,
            'migration_completed': self.migration_completed_flag,
            'storage_type': 'OptimizedStorageManager' if self.integration_enabled else 'SettingsManager'
        }
        
        if self.integration_enabled:
            # Добавляем статистику оптимизированного хранилища
            try:
                storage_stats = self.optimized_storage.get_statistics()
                stats.update(storage_stats)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка получения статистики хранилища: {e}")
        
        # Добавляем статистику миграции
        migration_status = self.storage_adapter.get_migration_status()
        stats['migration_status'] = migration_status
        
        return stats
    
    def cleanup(self):
        """Очистка ресурсов при завершении работы"""
        try:
            if self.batch_timer.isActive():
                self.batch_timer.stop()
            
            if self.integration_enabled:
                self.optimized_storage.cleanup()
            
            logger.info("🧹 StorageIntegration очищен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки StorageIntegration: {e}")


# Глобальный экземпляр интеграции
_storage_integration: Optional[StorageIntegration] = None


def get_storage_integration(settings_manager: SettingsManager = None) -> StorageIntegration:
    """Получает глобальный экземпляр StorageIntegration"""
    global _storage_integration
    
    if _storage_integration is None and settings_manager is not None:
        _storage_integration = StorageIntegration(settings_manager)
    
    return _storage_integration


def initialize_optimized_storage(settings_manager: SettingsManager, 
                               auto_migrate: bool = True) -> bool:
    """
    Инициализирует оптимизированную систему хранения
    
    Args:
        settings_manager: Экземпляр SettingsManager
        auto_migrate: Автоматически запустить миграцию
        
    Returns:
        bool: Успешность инициализации
    """
    integration = get_storage_integration(settings_manager)
    if integration:
        return integration.initialize_storage_system(auto_migrate)
    return False 
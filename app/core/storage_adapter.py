"""
Адаптер для интеграции OptimizedStorageManager с существующим SettingsManager API
Обеспечивает постепенную миграцию без нарушения совместимости
"""

import json
import logging
from typing import Any, Optional, Dict, List, Union
from pathlib import Path

from .optimized_storage_manager import get_optimized_storage_manager
from ..settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class StorageAdapter:
    """
    Адаптер для плавной миграции от SettingsManager к OptimizedStorageManager
    Предоставляет совместимый API и выполняет миграцию данных
    """
    
    def __init__(self, settings_manager: SettingsManager = None):
        self.settings_manager = settings_manager
        self.optimized_storage = get_optimized_storage_manager()
        
        # Маппинг секций SettingsManager к категориям OptimizedStorageManager
        self.section_mapping = {
            'General': 'general',
            'Models': 'models',
            'Gemini': 'gemini',
            'Network': 'network',
            'Paths': 'paths', 
            'Training': 'training',
            'Interface': 'interface',
            'Processing': 'processing',
            'OCR': 'ocr',
            'Prompts': 'prompts'
        }
        
        # Флаги миграции
        self._migration_status = {
            'settings_migrated': False,
            'preferences_migrated': False,
            'api_keys_migrated': False
        }
        
        logger.info("🔄 StorageAdapter инициализирован")
    
    def start_migration(self, force: bool = False) -> bool:
        """
        Запускает миграцию данных от SettingsManager к OptimizedStorageManager
        
        Args:
            force: Принудительная миграция даже если уже выполнена
            
        Returns:
            bool: Успешность миграции
        """
        try:
            if not self.settings_manager:
                logger.warning("SettingsManager не предоставлен, пропускаем миграцию")
                return True
                
            # Проверяем статус миграции
            migration_status = self.optimized_storage.get_setting(
                'migration_completed', False, 'system'
            )
            
            if migration_status and not force:
                logger.info("✅ Миграция уже выполнена")
                return True
            
            logger.info("🔄 Начинаем миграцию настроек...")
            
            # Миграция основных настроек
            success = self._migrate_settings()
            if success:
                self._migration_status['settings_migrated'] = True
                logger.info("✅ Основные настройки мигрированы")
            
            # Миграция зашифрованных данных  
            success = self._migrate_encrypted_settings()
            if success:
                self._migration_status['api_keys_migrated'] = True
                logger.info("✅ Зашифрованные настройки мигрированы")
            
            # Миграция пользовательских предпочтений
            success = self._migrate_user_preferences()
            if success:
                self._migration_status['preferences_migrated'] = True
                logger.info("✅ Пользовательские предпочтения мигрированы")
            
            # Отмечаем миграцию как завершенную
            all_migrated = all(self._migration_status.values())
            if all_migrated:
                self.optimized_storage.set_setting(
                    'migration_completed', True, 'system'
                )
                self.optimized_storage.set_setting(
                    'migration_timestamp', str(Path(__file__).stat().st_mtime), 'system'
                )
                logger.info("🎉 Миграция полностью завершена")
            
            return all_migrated
            
        except Exception as e:
            logger.error(f"❌ Ошибка при миграции: {e}")
            return False
    
    def _migrate_settings(self) -> bool:
        """Миграция основных настроек"""
        try:
            migrated_count = 0
            
            for section_name, category in self.section_mapping.items():
                if section_name in self.settings_manager.config:
                    section = self.settings_manager.config[section_name]
                    
                    for key, value in section.items():
                        # Конвертируем значения в соответствующие типы
                        converted_value = self._convert_setting_value(value)
                        
                        self.optimized_storage.set_setting(
                            key, converted_value, category, batch=True
                        )
                        migrated_count += 1
            
            logger.info(f"📊 Мигрировано {migrated_count} настроек")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции настроек: {e}")
            return False
    
    def _migrate_encrypted_settings(self) -> bool:
        """Миграция зашифрованных настроек (API ключи)"""
        try:
            # Список известных зашифрованных настроек
            encrypted_keys = [
                'google_api_key',
                'openai_api_key', 
                'anthropic_api_key',
                'gemini_api_key',
                'huggingface_token',
                'deepseek_api_key',
                'mistral_api_key',
                'xai_api_key'
            ]
            
            migrated_count = 0
            
            for key in encrypted_keys:
                try:
                    # Получаем зашифрованное значение из SettingsManager
                    encrypted_value = self.settings_manager.get_encrypted_setting(key)
                    
                    if encrypted_value:
                        # Сохраняем в OptimizedStorageManager как зашифрованную настройку
                        self.optimized_storage.set_setting(
                            f"encrypted_{key}", encrypted_value, 'security', batch=True
                        )
                        migrated_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось мигрировать ключ {key}: {e}")
            
            logger.info(f"🔐 Мигрировано {migrated_count} зашифрованных настроек")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции зашифрованных настроек: {e}")
            return False
    
    def _migrate_user_preferences(self) -> bool:
        """Миграция пользовательских предпочтений UI"""
        try:
            # Пользовательские настройки интерфейса
            ui_settings = [
                ('Interface', 'active_model'),
                ('Interface', 'last_export_path'),
                ('Interface', 'last_open_path'), 
                ('Interface', 'show_preview'),
                ('Processing', 'preprocess_images'),
                ('Processing', 'denoise_level'),
                ('Processing', 'contrast_enhance'),
                ('Processing', 'image_resize'),
                ('OCR', 'use_osd'),
                ('OCR', 'psm_mode'),
                ('OCR', 'oem_mode')
            ]
            
            migrated_count = 0
            
            for section, key in ui_settings:
                try:
                    value = self.settings_manager.get_string(section, key)
                    if value:
                        converted_value = self._convert_setting_value(value)
                        category = self.section_mapping.get(section, 'general')
                        
                        self.optimized_storage.set_setting(
                            key, converted_value, category, batch=True
                        )
                        migrated_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось мигрировать {section}.{key}: {e}")
            
            logger.info(f"👤 Мигрировано {migrated_count} пользовательских настроек")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции пользовательских настроек: {e}")
            return False
    
    def _convert_setting_value(self, value: str) -> Any:
        """Конвертирует строковые значения из INI в соответствующие типы"""
        if not isinstance(value, str):
            return value
            
        # Булевы значения
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Числовые значения
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON массивы/объекты
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Обычная строка
        return value
    
    def get_setting(self, key: str, default: Any = None, 
                   section: str = 'general') -> Any:
        """
        Унифицированный метод получения настроек
        Сначала пытается получить из OptimizedStorageManager, затем из SettingsManager
        """
        # Маппинг секции
        category = self.section_mapping.get(section, section.lower())
        
        # Пытаемся получить из оптимизированного хранилища
        value = self.optimized_storage.get_setting(key, None, category)
        
        if value is not None:
            return value
        
        # Fallback к SettingsManager
        if self.settings_manager:
            try:
                return self.settings_manager.get_string(section, key, str(default))
            except Exception as e:
                logger.warning(f"⚠️ Ошибка получения настройки {section}.{key}: {e}")
        
        return default
    
    def set_setting(self, key: str, value: Any, 
                   section: str = 'general', batch: bool = False) -> bool:
        """
        Унифицированный метод сохранения настроек
        Сохраняет в OptimizedStorageManager и синхронизирует с SettingsManager
        """
        category = self.section_mapping.get(section, section.lower())
        
        # Сохраняем в оптимизированном хранилище
        success = self.optimized_storage.set_setting(key, value, category, batch)
        
        # Синхронизируем с SettingsManager для обратной совместимости
        if success and self.settings_manager:
            try:
                self.settings_manager.set_value(section, key, str(value))
                self.settings_manager.save_settings()
            except Exception as e:
                logger.warning(f"⚠️ Ошибка синхронизации с SettingsManager: {e}")
        
        return success
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Возвращает статус миграции"""
        return {
            'migration_completed': self.optimized_storage.get_setting(
                'migration_completed', False, 'system'
            ),
            'migration_timestamp': self.optimized_storage.get_setting(
                'migration_timestamp', None, 'system'
            ),
            'detailed_status': self._migration_status.copy()
        }


# Глобальный экземпляр адаптера
_storage_adapter: Optional[StorageAdapter] = None


def get_storage_adapter(settings_manager: SettingsManager = None) -> StorageAdapter:
    """Получает глобальный экземпляр StorageAdapter"""
    global _storage_adapter
    
    if _storage_adapter is None:
        _storage_adapter = StorageAdapter(settings_manager)
    
    return _storage_adapter


def migrate_to_optimized_storage(settings_manager: SettingsManager, 
                               force: bool = False) -> bool:
    """
    Вспомогательная функция для запуска миграции
    
    Args:
        settings_manager: Экземпляр SettingsManager для миграции
        force: Принудительная миграция
        
    Returns:
        bool: Успешность миграции
    """
    adapter = get_storage_adapter(settings_manager)
    return adapter.start_migration(force) 
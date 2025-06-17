"""
Менеджер секретов для безопасного хранения и доступа к чувствительным данным.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from .crypto_manager import CryptoManager

logger = logging.getLogger(__name__)


class SecretsManager:
    """Менеджер для безопасного управления секретами приложения."""
    
    # Список известных секретов для валидации
    KNOWN_SECRETS = [
        'google_api_key',
        'hf_token',
        'openai_api_key',
        'anthropic_api_key',
        'mistral_api_key',
        'deepseek_api_key',
        'xai_api_key'
    ]
    
    def __init__(self, secrets_file: Optional[str] = None):
        """
        Инициализация менеджера секретов.
        
        Args:
            secrets_file: Путь к файлу с секретами. 
                         Если не указан, используется путь по умолчанию.
        """
        self.crypto_manager = CryptoManager()
        self.secrets_file = Path(secrets_file) if secrets_file else self._get_default_secrets_path()
        self._secrets_cache: Dict[str, str] = {}
        self._load_secrets()
        
    def _get_default_secrets_path(self) -> Path:
        """Возвращает путь по умолчанию для хранения секретов."""
        from .. import config
        secrets_dir = Path(config.APP_DATA_PATH) / "security"
        secrets_dir.mkdir(parents=True, exist_ok=True)
        return secrets_dir / ".secrets.enc"
    
    def _load_secrets(self) -> None:
        """Загружает секреты из файла."""
        if not self.secrets_file.exists():
            logger.info("Файл секретов не найден, создается новый")
            self._save_secrets()
            return
            
        try:
            # Читаем зашифрованный файл
            encrypted_data = self.secrets_file.read_text(encoding='utf-8')
            
            if encrypted_data:
                # Расшифровываем
                decrypted_data = self.crypto_manager.decrypt(encrypted_data)
                # Парсим JSON
                self._secrets_cache = json.loads(decrypted_data)
                logger.info(f"Загружено {len(self._secrets_cache)} секретов")
            else:
                self._secrets_cache = {}
                
        except Exception as e:
            logger.error(f"Ошибка загрузки секретов: {e}")
            self._secrets_cache = {}
    
    def _save_secrets(self) -> None:
        """Сохраняет секреты в файл."""
        try:
            # Сериализуем в JSON
            json_data = json.dumps(self._secrets_cache, indent=2)
            
            # Шифруем
            encrypted_data = self.crypto_manager.encrypt(json_data)
            
            # Сохраняем
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            self.secrets_file.write_text(encrypted_data, encoding='utf-8')
            
            # Устанавливаем права доступа
            if os.name != 'nt':
                os.chmod(self.secrets_file, 0o600)
                
            logger.info("Секреты успешно сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения секретов: {e}")
            raise
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Получает секрет по ключу.
        
        Args:
            key: Ключ секрета
            default: Значение по умолчанию если секрет не найден
            
        Returns:
            Значение секрета или default
        """
        # Сначала проверяем переменные окружения
        env_value = os.environ.get(key.upper())
        if env_value:
            return env_value
            
        # Затем проверяем кэш
        return self._secrets_cache.get(key, default)
    
    def set_secret(self, key: str, value: str) -> None:
        """
        Устанавливает секрет.
        
        Args:
            key: Ключ секрета
            value: Значение секрета
        """
        if not value:
            logger.warning(f"Попытка установить пустой секрет для ключа: {key}")
            return
            
        # Добавляем в кэш
        self._secrets_cache[key] = value
        
        # Сохраняем
        self._save_secrets()
        
        logger.info(f"Секрет '{key}' успешно установлен")
    
    def delete_secret(self, key: str) -> bool:
        """
        Удаляет секрет.
        
        Args:
            key: Ключ секрета
            
        Returns:
            True если секрет был удален, False если не найден
        """
        if key in self._secrets_cache:
            del self._secrets_cache[key]
            self._save_secrets()
            logger.info(f"Секрет '{key}' удален")
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """
        Возвращает список всех ключей секретов.
        
        Returns:
            Список ключей
        """
        return list(self._secrets_cache.keys())
    
    def validate_secret(self, key: str, value: str) -> bool:
        """
        Валидирует секрет.
        
        Args:
            key: Ключ секрета
            value: Значение для проверки
            
        Returns:
            True если секрет валиден
        """
        if not value or not value.strip():
            return False
            
        # Специфичная валидация для разных типов секретов
        if key == 'google_api_key':
            # Google API ключи обычно начинаются с 'AIza'
            return value.startswith('AIza') and len(value) == 39
        elif key == 'hf_token':
            # Hugging Face токены начинаются с 'hf_'
            return value.startswith('hf_') and len(value) > 10
        elif key == 'openai_api_key':
            # OpenAI ключи начинаются с 'sk-'
            return value.startswith('sk-') and len(value) > 20
        elif key == 'anthropic_api_key':
            # Anthropic ключи начинаются с 'sk-ant-'
            return value.startswith('sk-ant-') and len(value) > 20
        
        # Общая валидация - минимальная длина
        return len(value) >= 10
    
    def migrate_from_settings(self, settings_manager) -> Dict[str, bool]:
        """
        Мигрирует секреты из старой системы настроек.
        
        Args:
            settings_manager: Экземпляр SettingsManager
            
        Returns:
            Словарь с результатами миграции {key: success}
        """
        results = {}
        
        # Список полей для миграции
        migration_map = {
            'google_api_key': ('Gemini', 'api_key'),
            'hf_token': ('Network', 'hf_token'),
            'openai_api_key': ('CloudLLM', 'openai_api_key'),
            'anthropic_api_key': ('CloudLLM', 'anthropic_api_key'),
        }
        
        for secret_key, (section, setting_key) in migration_map.items():
            try:
                # Получаем значение из старых настроек
                old_value = settings_manager.get_string(section, setting_key, '')
                
                if old_value and self.validate_secret(secret_key, old_value):
                    # Сохраняем в новую систему
                    self.set_secret(secret_key, old_value)
                    
                    # Удаляем из старых настроек
                    settings_manager.set_value(section, setting_key, '')
                    
                    results[secret_key] = True
                    logger.info(f"Успешно мигрирован секрет: {secret_key}")
                else:
                    results[secret_key] = False
                    
            except Exception as e:
                logger.error(f"Ошибка миграции секрета {secret_key}: {e}")
                results[secret_key] = False
        
        return results
    
    def create_backup(self, backup_path: Optional[str] = None) -> Path:
        """
        Создает резервную копию секретов.
        
        Args:
            backup_path: Путь для сохранения резервной копии
            
        Returns:
            Путь к созданной резервной копии
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.secrets_file.parent / f"secrets_backup_{timestamp}.enc"
        else:
            backup_file = Path(backup_path)
            
        # Копируем файл
        if self.secrets_file.exists():
            import shutil
            shutil.copy2(self.secrets_file, backup_file)
            logger.info(f"Создана резервная копия секретов: {backup_file}")
        
        return backup_file
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Восстанавливает секреты из резервной копии.
        
        Args:
            backup_path: Путь к файлу резервной копии
            
        Returns:
            True если восстановление успешно
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            logger.error(f"Файл резервной копии не найден: {backup_path}")
            return False
            
        try:
            # Создаем резервную копию текущих секретов
            self.create_backup()
            
            # Копируем файл резервной копии
            import shutil
            shutil.copy2(backup_file, self.secrets_file)
            
            # Перезагружаем секреты
            self._load_secrets()
            
            logger.info("Секреты успешно восстановлены из резервной копии")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления из резервной копии: {e}")
            return False
    
    def export_env_template(self, output_path: Optional[str] = None) -> Path:
        """
        Экспортирует шаблон .env файла со всеми доступными секретами.
        
        Args:
            output_path: Путь для сохранения шаблона
            
        Returns:
            Путь к созданному файлу
        """
        if output_path is None:
            from .. import config
            template_path = Path(config.APP_DATA_PATH).parent / ".env.template"
        else:
            template_path = Path(output_path)
            
        # Создаем шаблон
        template_lines = [
            "# InvoiceGemini Environment Variables Template",
            "# Copy this file to .env and fill in your actual values",
            "",
        ]
        
        # Добавляем все известные секреты
        for secret in self.KNOWN_SECRETS:
            current_value = self.get_secret(secret)
            if current_value:
                # Маскируем значение
                masked = current_value[:4] + "..." + current_value[-4:] if len(current_value) > 8 else "***"
                template_lines.append(f"{secret.upper()}={masked} # Current value is set")
            else:
                template_lines.append(f"{secret.upper()}= # Not set")
        
        # Сохраняем
        template_path.write_text("\n".join(template_lines))
        logger.info(f"Шаблон .env файла создан: {template_path}")
        
        return template_path


# Глобальный экземпляр менеджера секретов
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Возвращает глобальный экземпляр менеджера секретов."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
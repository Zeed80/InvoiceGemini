"""
Модуль для шифрования и дешифрования чувствительных данных.
Использует Fernet для симметричного шифрования.
"""

import os
import base64
import base64.binascii
from pathlib import Path
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class CryptoManager:
    """Менеджер для шифрования и дешифрования данных."""
    
    def __init__(self, key_file: Optional[str] = None):
        """
        Инициализация менеджера шифрования.
        
        Args:
            key_file: Путь к файлу с ключом шифрования. 
                     Если не указан, используется путь по умолчанию.
        """
        self.key_file = Path(key_file) if key_file else self._get_default_key_path()
        self._cipher: Optional[Fernet] = None
        self._ensure_key_exists()
        
    def _get_default_key_path(self) -> Path:
        """Возвращает путь по умолчанию для хранения ключа шифрования."""
        # Получаем директорию данных приложения
        from .. import config
        key_dir = Path(config.APP_DATA_PATH) / "security"
        key_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл ключа в защищенной директории
        key_file = key_dir / ".encryption.key"
        
        # Устанавливаем права доступа только для владельца (только на Unix)
        if os.name != 'nt':  # Unix-like системы
            try:
                os.chmod(key_dir, 0o700)
            except (OSError, PermissionError, Exception) as e:
                # Ошибка установки прав доступа - не критично
                pass
                
        return key_file
    
    def _ensure_key_exists(self) -> None:
        """Убеждается, что ключ шифрования существует, или создает новый."""
        if not self.key_file.exists():
            logger.info("Создание нового ключа шифрования...")
            self._generate_and_save_key()
        else:
            # Проверяем права доступа к файлу ключа
            if os.name != 'nt':
                try:
                    os.chmod(self.key_file, 0o600)
                except (OSError, PermissionError, Exception) as e:
                    # Ошибка установки прав доступа к ключу - не критично
                    pass
    
    def _generate_and_save_key(self) -> None:
        """Генерирует новый ключ шифрования и сохраняет его."""
        # Генерируем случайный ключ
        key = Fernet.generate_key()
        
        # Сохраняем ключ в файл
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.key_file.write_bytes(key)
        
        # Устанавливаем права доступа только для владельца
        if os.name != 'nt':
            os.chmod(self.key_file, 0o600)
            
        logger.info(f"Новый ключ шифрования сохранен: {self.key_file}")
    
    def _load_cipher(self) -> Fernet:
        """Загружает или создает объект шифрования."""
        if self._cipher is None:
            try:
                key = self.key_file.read_bytes()
                self._cipher = Fernet(key)
            except Exception as e:
                logger.error(f"Ошибка загрузки ключа шифрования: {e}")
                raise ValueError("Не удалось загрузить ключ шифрования") from e
        return self._cipher
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Шифрует данные и возвращает зашифрованную строку.
        
        Args:
            data: Данные для шифрования (строка или байты)
            
        Returns:
            Зашифрованные данные в виде base64 строки
        """
        cipher = self._load_cipher()
        
        # Преобразуем строку в байты если нужно
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Шифруем данные
        encrypted = cipher.encrypt(data)
        
        # Возвращаем как строку для удобства хранения
        return encrypted.decode('ascii')
    
    def decrypt(self, encrypted_data: Union[str, bytes]) -> str:
        """
        Дешифрует данные и возвращает исходную строку.
        
        Args:
            encrypted_data: Зашифрованные данные
            
        Returns:
            Расшифрованная строка
        """
        cipher = self._load_cipher()
        
        # Преобразуем строку в байты если нужно
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode('ascii')
            
        try:
            # Дешифруем данные
            decrypted = cipher.decrypt(encrypted_data)
            
            # Возвращаем как строку
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Ошибка дешифрования: {e}")
            raise ValueError("Не удалось расшифровать данные") from e
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Создает ключ шифрования из пароля используя PBKDF2.
        
        Args:
            password: Пароль для генерации ключа
            salt: Соль для усиления безопасности (если None, генерируется случайная)
            
        Returns:
            Tuple из (ключ, соль)
        """
        if salt is None:
            salt = os.urandom(16)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def rotate_key(self) -> None:
        """
        Ротация ключа шифрования.
        Создает новый ключ и перешифровывает все существующие данные.
        """
        # Сохраняем старый cipher
        old_cipher = self._cipher
        
        # Генерируем новый ключ
        self._generate_and_save_key()
        self._cipher = None  # Сбрасываем, чтобы загрузить новый ключ
        
        logger.info("Ключ шифрования успешно обновлен")
        
        # TODO: Здесь должна быть логика перешифровки существующих данных
        # Это требует интеграции с SettingsManager
    
    def is_encrypted(self, data: str) -> bool:
        """
        Проверяет, являются ли данные зашифрованными.
        
        Args:
            data: Строка для проверки
            
        Returns:
            True если данные выглядят как зашифрованные
        """
        try:
            # Пытаемся декодировать как base64
            base64.b64decode(data, validate=True)
            # Проверяем, что строка имеет правильный формат Fernet
            return data.startswith('gAAAAA') and len(data) > 100
        except (ValueError, TypeError, base64.binascii.Error, Exception) as e:
            # Ошибка проверки шифрования - данные не зашифрованы
            return False
    
    def clear_key(self) -> None:
        """Удаляет ключ шифрования (ОПАСНО! Используйте с осторожностью)."""
        if self.key_file.exists():
            self.key_file.unlink()
            logger.warning("Ключ шифрования удален!")
        self._cipher = None
"""
Менеджер ресурсов для управления временными файлами и предотвращения утечек.
"""

import os
import shutil
import tempfile
import logging
import atexit
from pathlib import Path
from typing import Set, Optional, Dict
from contextlib import contextmanager
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResourceManager:
    """Менеджер для управления временными ресурсами приложения."""
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        """
        Инициализация менеджера ресурсов.
        
        Args:
            base_temp_dir: Базовая директория для временных файлов
        """
        self.base_temp_dir = Path(base_temp_dir) if base_temp_dir else self._get_default_temp_dir()
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = Lock()
        self._temp_dirs: Set[Path] = set()
        self._temp_files: Set[Path] = set()
        self._session_dirs: Dict[str, Path] = {}  # session_id -> temp_dir
        
        # Регистрируем cleanup при завершении программы
        atexit.register(self.cleanup_all)
        
        # Очищаем старые временные файлы при запуске
        self._cleanup_old_temp_files()
        
        logger.info(f"ResourceManager инициализирован с temp_dir: {self.base_temp_dir}")
    
    def _get_default_temp_dir(self) -> Path:
        """Возвращает путь по умолчанию для временных файлов."""
        from .. import config
        temp_dir = Path(config.TEMP_PATH)
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def _cleanup_old_temp_files(self, max_age_hours: int = 24):
        """Удаляет старые временные файлы."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for item in self.base_temp_dir.iterdir():
                try:
                    # Проверяем время модификации
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                            logger.info(f"Удалена старая временная директория: {item}")
                        else:
                            item.unlink()
                            logger.info(f"Удален старый временный файл: {item}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить {item}: {e}")
                    
        except Exception as e:
            logger.warning(f"Ошибка при очистке старых временных файлов: {e}")
    
    @contextmanager
    def temp_directory(self, prefix: str = "invoice_", cleanup: bool = True):
        """
        Контекстный менеджер для создания временной директории.
        
        Args:
            prefix: Префикс для имени директории
            cleanup: Удалять ли директорию при выходе
            
        Yields:
            Path: Путь к временной директории
        """
        temp_dir = None
        try:
            # Создаем временную директорию
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_temp_dir))
            
            with self._lock:
                self._temp_dirs.add(temp_dir)
            
            logger.debug(f"Создана временная директория: {temp_dir}")
            
            yield temp_dir
            
        finally:
            if temp_dir and cleanup:
                self._remove_temp_dir(temp_dir)
    
    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "temp_", cleanup: bool = True):
        """
        Контекстный менеджер для создания временного файла.
        
        Args:
            suffix: Суффикс (расширение) файла
            prefix: Префикс имени файла
            cleanup: Удалять ли файл при выходе
            
        Yields:
            Path: Путь к временному файлу
        """
        temp_file = None
        try:
            # Создаем временный файл
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.base_temp_dir
            )
            os.close(fd)  # Закрываем дескриптор
            
            temp_file = Path(temp_path)
            
            with self._lock:
                self._temp_files.add(temp_file)
            
            logger.debug(f"Создан временный файл: {temp_file}")
            
            yield temp_file
            
        finally:
            if temp_file and cleanup:
                self._remove_temp_file(temp_file)
    
    def create_session_directory(self, session_id: str) -> Path:
        """
        Создает директорию для сессии обработки.
        
        Args:
            session_id: Уникальный идентификатор сессии
            
        Returns:
            Path: Путь к директории сессии
        """
        with self._lock:
            if session_id in self._session_dirs:
                return self._session_dirs[session_id]
            
            session_dir = self.base_temp_dir / f"session_{session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            self._session_dirs[session_id] = session_dir
            self._temp_dirs.add(session_dir)
            
            logger.info(f"Создана директория сессии: {session_dir}")
            return session_dir
    
    def cleanup_session(self, session_id: str):
        """Очищает ресурсы сессии."""
        with self._lock:
            if session_id in self._session_dirs:
                session_dir = self._session_dirs.pop(session_id)
                self._remove_temp_dir(session_dir)
                logger.info(f"Очищена сессия: {session_id}")
    
    def _remove_temp_dir(self, temp_dir: Path):
        """Безопасно удаляет временную директорию."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Удалена временная директория: {temp_dir}")
            
            with self._lock:
                self._temp_dirs.discard(temp_dir)
                
        except Exception as e:
            logger.warning(f"Ошибка при удалении директории {temp_dir}: {e}")
    
    def _remove_temp_file(self, temp_file: Path):
        """Безопасно удаляет временный файл."""
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Удален временный файл: {temp_file}")
            
            with self._lock:
                self._temp_files.discard(temp_file)
                
        except Exception as e:
            logger.warning(f"Ошибка при удалении файла {temp_file}: {e}")
    
    def cleanup_all(self):
        """Очищает все временные ресурсы."""
        logger.info("Начало полной очистки временных ресурсов...")
        
        with self._lock:
            # Копируем множества, чтобы избежать изменения во время итерации
            temp_dirs = self._temp_dirs.copy()
            temp_files = self._temp_files.copy()
        
        # Удаляем все временные файлы
        for temp_file in temp_files:
            self._remove_temp_file(temp_file)
        
        # Удаляем все временные директории
        for temp_dir in temp_dirs:
            self._remove_temp_dir(temp_dir)
        
        # Очищаем сессии
        with self._lock:
            self._session_dirs.clear()
        
        logger.info("Очистка временных ресурсов завершена")
    
    def get_temp_space_usage(self) -> Dict[str, float]:
        """
        Возвращает информацию об использовании дискового пространства.
        
        Returns:
            Dict с информацией о размерах
        """
        total_size = 0
        file_count = 0
        dir_count = 0
        
        try:
            for item in self.base_temp_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1
        except Exception as e:
            logger.warning(f"Ошибка при подсчете размера temp: {e}")
        
        return {
            "total_size_bytes": float(total_size),
            "total_size_mb": total_size / 1024 / 1024,
            "file_count": float(file_count),
            "dir_count": float(dir_count),
            "active_sessions": float(len(self._session_dirs))
        }
    
    def cleanup_if_needed(self, threshold_mb: int = 1000):
        """
        Выполняет очистку если использование превышает порог.
        
        Args:
            threshold_mb: Порог в мегабайтах
        """
        usage = self.get_temp_space_usage()
        
        if usage["total_size_mb"] > threshold_mb:
            logger.warning(
                f"Использование temp превышает порог: "
                f"{usage['total_size_mb']:.1f}MB > {threshold_mb}MB"
            )
            
            # Сначала пробуем удалить старые файлы
            self._cleanup_old_temp_files(max_age_hours=1)
            
            # Проверяем снова
            usage = self.get_temp_space_usage()
            if usage["total_size_mb"] > threshold_mb:
                logger.warning("Выполняется полная очистка временных файлов...")
                self.cleanup_all()


# Глобальный экземпляр менеджера ресурсов
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Возвращает глобальный экземпляр менеджера ресурсов."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
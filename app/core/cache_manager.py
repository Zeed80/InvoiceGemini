"""
Менеджер кэширования результатов обработки документов.
"""

import os
import json
import hashlib
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from threading import Lock
import pickle

logger = logging.getLogger(__name__)


class CacheEntry:
    """Запись в кэше"""
    def __init__(self, file_hash: str, model_type: str, result: Dict[str, Any], 
                 file_path: str = None, file_size: int = 0):
        self.file_hash = file_hash
        self.model_type = model_type
        self.result = result
        self.file_path = file_path
        self.file_size = file_size
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует запись в словарь"""
        return {
            'file_hash': self.file_hash,
            'model_type': self.model_type,
            'result': self.result,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Создает запись из словаря"""
        entry = cls(
            file_hash=data['file_hash'],
            model_type=data['model_type'],
            result=data['result'],
            file_path=data.get('file_path'),
            file_size=data.get('file_size', 0)
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        entry.last_accessed = datetime.fromisoformat(data['last_accessed'])
        entry.access_count = data.get('access_count', 1)
        return entry


class CacheManager:
    """Менеджер кэширования результатов обработки"""
    
    def __init__(self, cache_dir: str = None, max_cache_size_gb: float = 2.0,
                 max_age_days: int = 30):
        """
        Инициализация менеджера кэша.
        
        Args:
            cache_dir: Директория для хранения кэша
            max_cache_size_gb: Максимальный размер кэша в ГБ
            max_age_days: Максимальный возраст записей в днях
        """
        self.cache_dir = Path(cache_dir or "data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.max_age = timedelta(days=max_age_days)
        
        self._lock = Lock()
        self._index_file = self.cache_dir / "cache_index.json"
        self._cache_index: Dict[str, CacheEntry] = {}
        
        # Загружаем индекс кэша
        self._load_index()
        
        # Очищаем устаревшие записи
        self._cleanup_old_entries()
        
        logger.info(f"Менеджер кэша инициализирован. Директория: {self.cache_dir}")
        
    def _load_index(self):
        """Загружает индекс кэша из файла"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._cache_index[key] = CacheEntry.from_dict(entry_data)
                logger.info(f"Загружено {len(self._cache_index)} записей из индекса кэша")
            except Exception as e:
                logger.error(f"Ошибка загрузки индекса кэша: {e}")
                self._cache_index = {}
                
    def _save_index(self):
        """Сохраняет индекс кэша в файл"""
        try:
            data = {key: entry.to_dict() for key, entry in self._cache_index.items()}
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса кэша: {e}")
            
    def _get_cache_key(self, file_hash: str, model_type: str) -> str:
        """Генерирует ключ кэша"""
        return f"{file_hash}_{model_type}"
        
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Возвращает путь к файлу кэша"""
        return self.cache_dir / f"{cache_key}.cache"
        
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Вычисляет хэш файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            str: SHA256 хэш файла
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Читаем файл блоками для экономии памяти
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Ошибка вычисления хэша файла {file_path}: {e}")
            # Используем путь и время модификации как fallback
            stat = os.stat(file_path)
            fallback = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.sha256(fallback.encode()).hexdigest()
            
    def get_cached_result(self, file_hash: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Получает кэшированный результат.
        
        Args:
            file_hash: Хэш файла
            model_type: Тип модели
            
        Returns:
            Optional[Dict[str, Any]]: Кэшированный результат или None
        """
        with self._lock:
            cache_key = self._get_cache_key(file_hash, model_type)
            
            if cache_key not in self._cache_index:
                return None
                
            entry = self._cache_index[cache_key]
            
            # Проверяем возраст записи
            if datetime.now() - entry.created_at > self.max_age:
                logger.info(f"Кэш устарел для {cache_key}")
                self._remove_entry(cache_key)
                return None
                
            # Загружаем результат из файла
            cache_file = self._get_cache_file_path(cache_key)
            if not cache_file.exists():
                logger.warning(f"Файл кэша не найден: {cache_file}")
                del self._cache_index[cache_key]
                return None
                
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    
                # Обновляем статистику
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._save_index()
                
                logger.info(f"Результат получен из кэша: {cache_key}")
                return result
                
            except Exception as e:
                logger.error(f"Ошибка чтения кэша {cache_key}: {e}")
                self._remove_entry(cache_key)
                return None
                
    def cache_result(self, file_hash: str, model_type: str, result: Dict[str, Any],
                    file_path: str = None) -> bool:
        """
        Сохраняет результат в кэш.
        
        Args:
            file_hash: Хэш файла
            model_type: Тип модели
            result: Результат обработки
            file_path: Путь к исходному файлу (опционально)
            
        Returns:
            bool: True если успешно сохранено
        """
        with self._lock:
            try:
                cache_key = self._get_cache_key(file_hash, model_type)
                cache_file = self._get_cache_file_path(cache_key)
                
                # Сохраняем результат
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                    
                # Определяем размер файла
                file_size = 0
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    
                # Создаем запись в индексе
                entry = CacheEntry(
                    file_hash=file_hash,
                    model_type=model_type,
                    result={},  # Не храним результат в индексе для экономии памяти
                    file_path=file_path,
                    file_size=file_size
                )
                
                self._cache_index[cache_key] = entry
                self._save_index()
                
                # Проверяем размер кэша
                self._check_cache_size()
                
                logger.info(f"Результат сохранен в кэш: {cache_key}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка сохранения в кэш: {e}")
                return False
                
    def _check_cache_size(self):
        """Проверяет размер кэша и удаляет старые записи при необходимости"""
        total_size = sum(
            self._get_cache_file_path(key).stat().st_size
            for key in self._cache_index
            if self._get_cache_file_path(key).exists()
        )
        
        if total_size <= self.max_cache_size_bytes:
            return
            
        logger.info(f"Размер кэша превышен: {total_size / (1024**3):.2f} GB")
        
        # Сортируем записи по последнему доступу
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Удаляем старые записи
        for cache_key, entry in sorted_entries:
            if total_size <= self.max_cache_size_bytes * 0.9:  # Оставляем 10% запас
                break
                
            file_size = self._get_entry_size(cache_key)
            self._remove_entry(cache_key)
            total_size -= file_size
            
    def _cleanup_old_entries(self):
        """Удаляет устаревшие записи"""
        now = datetime.now()
        expired_keys = []
        
        for cache_key, entry in self._cache_index.items():
            if now - entry.created_at > self.max_age:
                expired_keys.append(cache_key)
                
        for key in expired_keys:
            self._remove_entry(key)
            
        if expired_keys:
            logger.info(f"Удалено {len(expired_keys)} устаревших записей из кэша")
            
    def _remove_entry(self, cache_key: str):
        """Удаляет запись из кэша"""
        try:
            # Удаляем файл
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
                
            # Удаляем из индекса
            if cache_key in self._cache_index:
                del self._cache_index[cache_key]
                
        except Exception as e:
            logger.error(f"Ошибка удаления записи {cache_key}: {e}")
            
    def _get_entry_size(self, cache_key: str) -> int:
        """Возвращает размер записи в байтах"""
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            return cache_file.stat().st_size
        return 0
        
    def clear_cache(self):
        """Полностью очищает кэш"""
        with self._lock:
            try:
                # Удаляем все файлы
                for cache_key in list(self._cache_index.keys()):
                    self._remove_entry(cache_key)
                    
                # Очищаем индекс
                self._cache_index.clear()
                self._save_index()
                
                logger.info("Кэш полностью очищен")
                
            except Exception as e:
                logger.error(f"Ошибка очистки кэша: {e}")
    
    def clear_expired(self):
        """Очищает устаревшие записи из кэша (публичный метод)"""
        with self._lock:
            try:
                self._cleanup_old_entries()
                self._save_index()
                logger.info("Очистка устаревших записей завершена")
            except Exception as e:
                logger.error(f"Ошибка очистки устаревших записей: {e}")
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        with self._lock:
            total_size = sum(
                self._get_entry_size(key)
                for key in self._cache_index
            )
            
            total_hits = sum(
                entry.access_count - 1  # -1 т.к. первый доступ это сохранение
                for entry in self._cache_index.values()
            )
            
            return {
                'total_entries': len(self._cache_index),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_cache_size_bytes / (1024 * 1024),
                'total_hits': total_hits,
                'oldest_entry': min(
                    (entry.created_at for entry in self._cache_index.values()),
                    default=None
                ),
                'most_accessed': max(
                    self._cache_index.items(),
                    key=lambda x: x[1].access_count,
                    default=(None, None)
                )[0]
            }
            
    def invalidate_file(self, file_path: str):
        """
        Инвалидирует все кэшированные результаты для файла.
        
        Args:
            file_path: Путь к файлу
        """
        with self._lock:
            # Вычисляем хэш файла
            file_hash = self.calculate_file_hash(file_path)
            
            # Находим все записи с этим хэшем
            keys_to_remove = [
                key for key, entry in self._cache_index.items()
                if entry.file_hash == file_hash
            ]
            
            # Удаляем записи
            for key in keys_to_remove:
                self._remove_entry(key)
                
            if keys_to_remove:
                logger.info(f"Инвалидировано {len(keys_to_remove)} записей для файла {file_path}")
                

# Глобальный экземпляр менеджера кэша
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Возвращает глобальный экземпляр менеджера кэша"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager 
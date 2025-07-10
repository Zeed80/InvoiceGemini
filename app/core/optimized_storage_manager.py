"""
Оптимизированная система хранения данных с пулом соединений и кэшированием
"""

import sqlite3
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor
import time
from queue import Queue, Empty
import threading

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Статистика запросов"""
    query_type: str
    execution_time: float
    rows_affected: int
    timestamp: float
    query_hash: str


class ConnectionPool:
    """Пул соединений с SQLite"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.connections: Queue = Queue(maxsize=pool_size)
        self.lock = Lock()
        
        # Настройки SQLite для производительности
        self._sqlite_optimizations = {
            'PRAGMA journal_mode': 'WAL',  # Write-Ahead Logging
            'PRAGMA synchronous': 'NORMAL',  # Балансируем безопасность/скорость
            'PRAGMA cache_size': '10000',  # 10MB кэш
            'PRAGMA temp_store': 'memory',  # Временные данные в памяти
            'PRAGMA mmap_size': '268435456',  # 256MB memory mapping
            'PRAGMA optimize': None  # Оптимизация статистики
        }
        
        # Создаем соединения
        self._create_connections()
    
    def _create_connections(self):
        """Создает пул соединений"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(
                str(self.db_path), 
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row  # Доступ по имени колонки
            
            # Применяем оптимизации
            for pragma, value in self._sqlite_optimizations.items():
                if value is not None:
                    conn.execute(f"{pragma} = {value}")
                else:
                    conn.execute(pragma)
            
            conn.commit()
            self.connections.put(conn)
    
    def get_connection(self, timeout: float = 5.0) -> sqlite3.Connection:
        """Получает соединение из пула"""
        try:
            return self.connections.get(timeout=timeout)
        except Empty:
            raise RuntimeError("Не удалось получить соединение из пула")
    
    def return_connection(self, conn: sqlite3.Connection):
        """Возвращает соединение в пул"""
        try:
            self.connections.put_nowait(conn)
        except:
            # Пул полный, закрываем соединение
            conn.close()
    
    def close_all(self):
        """Закрывает все соединения"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except Empty:
                break


class QueryCache:
    """Кэш запросов с TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}  # query_hash -> (result, expire_time)
        self.access_times: Dict[str, float] = {}
        self.lock = RLock()
    
    def get(self, query_hash: str) -> Optional[Any]:
        """Получает результат из кэша"""
        with self.lock:
            if query_hash in self.cache:
                result, expire_time = self.cache[query_hash]
                
                if time.time() < expire_time:
                    self.access_times[query_hash] = time.time()
                    return result
                else:
                    # Удаляем устаревшую запись
                    del self.cache[query_hash]
                    if query_hash in self.access_times:
                        del self.access_times[query_hash]
            
            return None
    
    def put(self, query_hash: str, result: Any, ttl: Optional[float] = None):
        """Сохраняет результат в кэш"""
        if ttl is None:
            ttl = self.default_ttl
            
        expire_time = time.time() + ttl
        
        with self.lock:
            # Проверяем размер кэша
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[query_hash] = (result, expire_time)
            self.access_times[query_hash] = time.time()
    
    def _evict_lru(self):
        """Вытесняет наименее используемые записи"""
        if not self.access_times:
            return
            
        # Находим 10% старых записей для удаления
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        to_remove = sorted_items[:len(sorted_items) // 10 + 1]
        
        for query_hash, _ in to_remove:
            if query_hash in self.cache:
                del self.cache[query_hash]
            if query_hash in self.access_times:
                del self.access_times[query_hash]
    
    def clear(self):
        """Очищает кэш"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class OptimizedStorageManager(QObject):
    """
    Оптимизированный менеджер хранения с:
    - Пулом соединений
    - Кэшированием запросов
    - Асинхронными операциями
    - Батчевыми обновлениями
    - Автоматической оптимизацией
    """
    
    query_executed = pyqtSignal(str, float)  # query_type, execution_time
    cache_hit = pyqtSignal(str)  # query_hash
    cache_miss = pyqtSignal(str)  # query_hash
    
    def __init__(self, db_path: str = "data/invoices.db", 
                 pool_size: int = 5, cache_size: int = 1000):
        super().__init__()
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Пул соединений
        self.connection_pool = ConnectionPool(str(self.db_path), pool_size)
        
        # Кэш запросов
        self.query_cache = QueryCache(max_size=cache_size)
        
        # Асинхронный executor
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="storage")
        
        # Батчевые операции
        self.batch_queue: Queue = Queue()
        self.batch_size = 100
        self.batch_timeout = 5.0  # секунды
        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self._process_batch)
        self.batch_timer.start(int(self.batch_timeout * 1000))
        
        # Статистика
        self.query_stats: List[QueryStats] = []
        self.stats_lock = Lock()
        
        # Инициализация схемы
        self._initialize_schema()
        
        # Оптимизация по расписанию
        self.optimization_timer = QTimer()
        self.optimization_timer.timeout.connect(self._periodic_optimization)
        self.optimization_timer.start(3600000)  # Каждый час
        
        logger.info(f"🗄️ OptimizedStorageManager инициализирован:")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Pool size: {pool_size}")
        logger.info(f"   Cache size: {cache_size}")
    
    def _initialize_schema(self):
        """Инициализирует схему базы данных"""
        schema_sql = """
        -- Таблица настроек
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Индексы для настроек
        CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category);
        CREATE INDEX IF NOT EXISTS idx_settings_updated ON settings(updated_at);
        
        -- Таблица результатов обработки
        CREATE TABLE IF NOT EXISTS processing_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            model_type TEXT NOT NULL,
            result_data TEXT NOT NULL,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_hash, model_type)
        );
        
        -- Индексы для результатов
        CREATE INDEX IF NOT EXISTS idx_results_hash ON processing_results(file_hash);
        CREATE INDEX IF NOT EXISTS idx_results_model ON processing_results(model_type);
        CREATE INDEX IF NOT EXISTS idx_results_created ON processing_results(created_at);
        
        -- Таблица статистики использования
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Индекс для статистики
        CREATE INDEX IF NOT EXISTS idx_stats_type ON usage_stats(event_type);
        CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON usage_stats(timestamp);
        
        -- Триггеры для обновления updated_at
        CREATE TRIGGER IF NOT EXISTS update_settings_timestamp 
        AFTER UPDATE ON settings 
        BEGIN 
            UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
        END;
        """
        
        conn = self.connection_pool.get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            self.connection_pool.return_connection(conn)
    
    def get_setting(self, key: str, default: Any = None, 
                   category: str = 'general', use_cache: bool = True) -> Any:
        """Получает настройку с кэшированием"""
        query_hash = f"setting_{key}_{category}"
        
        # Проверяем кэш
        if use_cache:
            cached_result = self.query_cache.get(query_hash)
            if cached_result is not None:
                self.cache_hit.emit(query_hash)
                return cached_result
        
        self.cache_miss.emit(query_hash)
        
        # Выполняем запрос
        start_time = time.time()
        conn = self.connection_pool.get_connection()
        
        try:
            cursor = conn.execute(
                "SELECT value FROM settings WHERE key = ? AND category = ?",
                (key, category)
            )
            row = cursor.fetchone()
            
            if row:
                try:
                    result = json.loads(row['value'])
                except json.JSONDecodeError:
                    result = row['value']
            else:
                result = default
            
            # Кэшируем результат
            if use_cache:
                self.query_cache.put(query_hash, result, ttl=600)  # 10 минут
            
            execution_time = time.time() - start_time
            self._record_query_stats('SELECT', execution_time, 1, query_hash)
            
            return result
            
        finally:
            self.connection_pool.return_connection(conn)
    
    def set_setting(self, key: str, value: Any, category: str = 'general', 
                   batch: bool = False) -> bool:
        """Устанавливает настройку"""
        # Сериализуем значение
        if isinstance(value, (dict, list, tuple)):
            serialized_value = json.dumps(value, ensure_ascii=False)
        else:
            serialized_value = str(value)
        
        query_data = {
            'type': 'upsert_setting',
            'key': key,
            'value': serialized_value,
            'category': category
        }
        
        if batch:
            # Добавляем в батч
            self.batch_queue.put(query_data)
            return True
        else:
            # Выполняем немедленно
            return self._execute_setting_update(query_data)
    
    def _execute_setting_update(self, query_data: Dict) -> bool:
        """Выполняет обновление настройки"""
        start_time = time.time()
        conn = self.connection_pool.get_connection()
        
        try:
            conn.execute("""
                INSERT OR REPLACE INTO settings (key, value, category) 
                VALUES (?, ?, ?)
            """, (query_data['key'], query_data['value'], query_data['category']))
            
            conn.commit()
            
            # Инвалидируем кэш
            query_hash = f"setting_{query_data['key']}_{query_data['category']}"
            self.query_cache.cache.pop(query_hash, None)
            
            execution_time = time.time() - start_time
            self._record_query_stats('INSERT/UPDATE', execution_time, 1, query_hash)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения настройки {query_data['key']}: {e}")
            return False
        finally:
            self.connection_pool.return_connection(conn)
    
    def get_processing_result(self, file_hash: str, model_type: str, 
                             use_cache: bool = True) -> Optional[Dict]:
        """Получает результат обработки"""
        query_hash = f"result_{file_hash}_{model_type}"
        
        # Проверяем кэш
        if use_cache:
            cached_result = self.query_cache.get(query_hash)
            if cached_result is not None:
                self.cache_hit.emit(query_hash)
                return cached_result
        
        self.cache_miss.emit(query_hash)
        
        start_time = time.time()
        conn = self.connection_pool.get_connection()
        
        try:
            cursor = conn.execute("""
                SELECT result_data, processing_time, created_at 
                FROM processing_results 
                WHERE file_hash = ? AND model_type = ?
            """, (file_hash, model_type))
            
            row = cursor.fetchone()
            if row:
                try:
                    result = {
                        'data': json.loads(row['result_data']),
                        'processing_time': row['processing_time'],
                        'created_at': row['created_at']
                    }
                    
                    # Кэшируем на 1 час
                    if use_cache:
                        self.query_cache.put(query_hash, result, ttl=3600)
                    
                    execution_time = time.time() - start_time
                    self._record_query_stats('SELECT', execution_time, 1, query_hash)
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка десериализации результата: {e}")
                    return None
            
            return None
            
        finally:
            self.connection_pool.return_connection(conn)
    
    def save_processing_result(self, file_hash: str, file_path: str, 
                              model_type: str, result_data: Dict, 
                              processing_time: float, batch: bool = False) -> bool:
        """Сохраняет результат обработки"""
        query_data = {
            'type': 'save_result',
            'file_hash': file_hash,
            'file_path': file_path,
            'model_type': model_type,
            'result_data': json.dumps(result_data, ensure_ascii=False),
            'processing_time': processing_time
        }
        
        if batch:
            self.batch_queue.put(query_data)
            return True
        else:
            return self._execute_result_save(query_data)
    
    def _execute_result_save(self, query_data: Dict) -> bool:
        """Выполняет сохранение результата"""
        start_time = time.time()
        conn = self.connection_pool.get_connection()
        
        try:
            conn.execute("""
                INSERT OR REPLACE INTO processing_results 
                (file_hash, file_path, model_type, result_data, processing_time)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_data['file_hash'],
                query_data['file_path'],
                query_data['model_type'],
                query_data['result_data'],
                query_data['processing_time']
            ))
            
            conn.commit()
            
            # Инвалидируем кэш
            query_hash = f"result_{query_data['file_hash']}_{query_data['model_type']}"
            self.query_cache.cache.pop(query_hash, None)
            
            execution_time = time.time() - start_time
            self._record_query_stats('INSERT/UPDATE', execution_time, 1, query_hash)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения результата: {e}")
            return False
        finally:
            self.connection_pool.return_connection(conn)
    
    def _process_batch(self):
        """Обрабатывает батч операций"""
        if self.batch_queue.empty():
            return
        
        operations = []
        try:
            # Собираем операции из очереди
            while len(operations) < self.batch_size:
                try:
                    operation = self.batch_queue.get_nowait()
                    operations.append(operation)
                except Empty:
                    break
            
            if not operations:
                return
            
            # Выполняем батч
            self._execute_batch_operations(operations)
            
        except Exception as e:
            logger.error(f"Ошибка выполнения батча: {e}")
    
    def _execute_batch_operations(self, operations: List[Dict]):
        """Выполняет батч операций"""
        if not operations:
            return
        
        start_time = time.time()
        conn = self.connection_pool.get_connection()
        
        try:
            conn.execute("BEGIN TRANSACTION")
            
            for operation in operations:
                if operation['type'] == 'upsert_setting':
                    conn.execute("""
                        INSERT OR REPLACE INTO settings (key, value, category) 
                        VALUES (?, ?, ?)
                    """, (operation['key'], operation['value'], operation['category']))
                    
                elif operation['type'] == 'save_result':
                    conn.execute("""
                        INSERT OR REPLACE INTO processing_results 
                        (file_hash, file_path, model_type, result_data, processing_time)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        operation['file_hash'],
                        operation['file_path'],
                        operation['model_type'],
                        operation['result_data'],
                        operation['processing_time']
                    ))
            
            conn.execute("COMMIT")
            
            execution_time = time.time() - start_time
            self._record_query_stats('BATCH', execution_time, len(operations), 'batch')
            
            logger.info(f"✅ Выполнен батч из {len(operations)} операций за {execution_time:.3f}с")
            
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Ошибка выполнения батча: {e}")
        finally:
            self.connection_pool.return_connection(conn)
    
    def _record_query_stats(self, query_type: str, execution_time: float, 
                           rows_affected: int, query_hash: str):
        """Записывает статистику запроса"""
        with self.stats_lock:
            stats = QueryStats(
                query_type=query_type,
                execution_time=execution_time,
                rows_affected=rows_affected,
                timestamp=time.time(),
                query_hash=query_hash
            )
            
            self.query_stats.append(stats)
            
            # Ограничиваем размер статистики
            if len(self.query_stats) > 10000:
                self.query_stats = self.query_stats[-5000:]
            
            self.query_executed.emit(query_type, execution_time)
    
    def _periodic_optimization(self):
        """Периодическая оптимизация базы данных"""
        logger.info("🔧 Запуск периодической оптимизации БД")
        
        conn = self.connection_pool.get_connection()
        try:
            # VACUUM для освобождения места
            conn.execute("VACUUM")
            
            # ANALYZE для обновления статистики
            conn.execute("ANALYZE")
            
            # Очистка старых записей статистики (старше 30 дней)
            conn.execute("""
                DELETE FROM usage_stats 
                WHERE timestamp < datetime('now', '-30 days')
            """)
            
            conn.commit()
            
            logger.info("✅ Оптимизация БД завершена")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации БД: {e}")
        finally:
            self.connection_pool.return_connection(conn)
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику работы"""
        with self.stats_lock:
            if not self.query_stats:
                return {}
            
            total_queries = len(self.query_stats)
            avg_time = sum(s.execution_time for s in self.query_stats) / total_queries
            
            query_types = {}
            for stats in self.query_stats:
                if stats.query_type not in query_types:
                    query_types[stats.query_type] = {
                        'count': 0,
                        'total_time': 0,
                        'avg_time': 0
                    }
                
                query_types[stats.query_type]['count'] += 1
                query_types[stats.query_type]['total_time'] += stats.execution_time
                
            # Вычисляем средние времена
            for query_type, data in query_types.items():
                data['avg_time'] = data['total_time'] / data['count']
            
            return {
                'total_queries': total_queries,
                'average_execution_time': avg_time,
                'query_types': query_types,
                'cache_size': len(self.query_cache.cache),
                'connection_pool_size': self.connection_pool.pool_size,
                'batch_queue_size': self.batch_queue.qsize()
            }
    
    def cleanup(self):
        """Очистка ресурсов"""
        # Обрабатываем оставшиеся батчи
        self._process_batch()
        
        # Останавливаем таймеры
        self.batch_timer.stop()
        self.optimization_timer.stop()
        
        # Завершаем executor
        self.executor.shutdown(wait=True)
        
        # Закрываем пул соединений
        self.connection_pool.close_all()
        
        # Очищаем кэш
        self.query_cache.clear()
        
        logger.info("🧹 OptimizedStorageManager очищен")


# Глобальный экземпляр
_storage_manager: Optional[OptimizedStorageManager] = None


def get_optimized_storage_manager() -> OptimizedStorageManager:
    """Возвращает глобальный экземпляр оптимизированного менеджера хранения"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = OptimizedStorageManager()
    return _storage_manager 
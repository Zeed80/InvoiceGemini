#!/usr/bin/env python3
"""
Optimized File Processing System for InvoiceGemini
Высокопроизводительная обработка файлов с параллельной обработкой и оптимизациями
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
import logging

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QTimer
import torch

from ..settings_manager import settings_manager
from ..core.cache_manager import get_cache_manager
from .. import utils


@dataclass
class ProcessingTask:
    """Задача обработки файла"""
    file_path: str
    task_id: str
    priority: int = 1
    model_type: str = "gemini"
    custom_prompt: str = ""
    ocr_lang: str = "rus+eng"
    
    def __post_init__(self):
        self.created_at = datetime.now()
        self.status = "pending"
        self.result = None
        self.error = None
        self.processing_time = 0.0


@dataclass
class ProcessingResult:
    """Результат обработки файла"""
    task_id: str
    file_path: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    model_type: str = "unknown"
    from_cache: bool = False


class GPUResourceManager:
    """Менеджер GPU ресурсов для оптимальной утилизации"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.memory_usage = {}
        self.device_locks = [threading.Lock() for _ in range(self.device_count)]
        
        # Мониторинг GPU памяти
        if self.gpu_available:
            self._monitor_gpu_memory()
    
    def _monitor_gpu_memory(self):
        """Мониторинг использования GPU памяти"""
        for device_id in range(self.device_count):
            try:
                memory_info = torch.cuda.mem_get_info(device_id)
                free_memory = memory_info[0] / 1024**3  # GB
                total_memory = memory_info[1] / 1024**3  # GB
                used_memory = total_memory - free_memory
                
                self.memory_usage[device_id] = {
                    'free': free_memory,
                    'used': used_memory,
                    'total': total_memory,
                    'utilization': used_memory / total_memory
                }
            except Exception as e:
                logging.warning(f"Failed to get GPU {device_id} memory info: {e}")
    
    def get_optimal_device(self) -> int:
        """Получить оптимальное GPU устройство"""
        if not self.gpu_available:
            return -1  # CPU
            
        self._monitor_gpu_memory()
        
        # Выбираем GPU с наименьшим использованием памяти
        best_device = 0
        min_utilization = float('inf')
        
        for device_id, memory_info in self.memory_usage.items():
            if memory_info['utilization'] < min_utilization:
                min_utilization = memory_info['utilization']
                best_device = device_id
        
        return best_device
    
    def acquire_device(self, device_id: int) -> bool:
        """Заблокировать GPU устройство"""
        if device_id == -1 or device_id >= len(self.device_locks):
            return True  # CPU или невалидный ID
            
        return self.device_locks[device_id].acquire(blocking=False)
    
    def release_device(self, device_id: int):
        """Освободить GPU устройство"""
        if device_id == -1 or device_id >= len(self.device_locks):
            return
            
        try:
            self.device_locks[device_id].release()
        except (IndexError, RuntimeError, AttributeError, Exception) as e:
            # Устройство уже освобождено или недоступно
            pass


class SmartCache:
    """Умный кэш с предиктивной загрузкой"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager() if hasattr(get_cache_manager, '__call__') else None
        self.prediction_cache = {}
        self.access_patterns = {}
        
    def get_cached_result(self, file_path: str, model_type: str) -> Optional[Dict]:
        """Получить результат из кэша"""
        if not self.cache_manager:
            return None
            
        try:
            file_hash = self.cache_manager.calculate_file_hash(file_path)
            result = self.cache_manager.get_cached_result(file_hash, model_type)
            
            if result:
                # Обновляем паттерны доступа
                self._update_access_pattern(file_path, model_type)
                
            return result
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None
    
    def cache_result(self, file_path: str, model_type: str, result: Dict):
        """Сохранить результат в кэш"""
        if not self.cache_manager or not result:
            return
            
        try:
            file_hash = self.cache_manager.calculate_file_hash(file_path)
            self.cache_manager.cache_result(file_hash, model_type, result, file_path)
            
            # Обновляем паттерны доступа
            self._update_access_pattern(file_path, model_type)
            
        except Exception as e:
            logging.error(f"Cache save error: {e}")
    
    def _update_access_pattern(self, file_path: str, model_type: str):
        """Обновить паттерны доступа для предиктивного кэширования"""
        key = f"{model_type}:{Path(file_path).suffix.lower()}"
        if key not in self.access_patterns:
            self.access_patterns[key] = 0
        self.access_patterns[key] += 1
    
    def predict_next_files(self, current_file: str, file_list: List[str]) -> List[str]:
        """Предсказать следующие файлы для предзагрузки"""
        current_path = Path(current_file)
        current_dir = current_path.parent
        current_suffix = current_path.suffix.lower()
        
        # Файлы из той же директории с тем же расширением
        candidates = []
        for file_path in file_list:
            file_path_obj = Path(file_path)
            if (file_path_obj.parent == current_dir and 
                file_path_obj.suffix.lower() == current_suffix):
                candidates.append(file_path)
        
        # Сортируем по имени для предсказуемости
        candidates.sort()
        
        # Возвращаем следующие 3 файла
        try:
            current_index = candidates.index(current_file)
            return candidates[current_index + 1:current_index + 4]
        except ValueError:
            return candidates[:3]


class ParallelFileProcessor(QObject):
    """
    Высокопроизводительный процессор файлов с параллельной обработкой
    
    Основные функции:
    - Параллельная обработка на CPU и GPU
    - Умный кэш с предиктивной загрузкой
    - Динамическое управление ресурсами
    - Приоритезация задач
    - Мониторинг производительности
    """
    
    # Сигналы для прогресса
    task_started = pyqtSignal(str)  # task_id
    task_completed = pyqtSignal(str, dict)  # task_id, result
    task_failed = pyqtSignal(str, str)  # task_id, error
    progress_updated = pyqtSignal(int, int)  # current, total
    batch_completed = pyqtSignal(list)  # all results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Конфигурация
        self.max_workers = self._calculate_optimal_workers()
        self.max_gpu_tasks = min(torch.cuda.device_count(), 2) if torch.cuda.is_available() else 0
        self.batch_size = 4  # Размер пакета для обработки
        
        # Менеджеры ресурсов
        self.gpu_manager = GPUResourceManager()
        self.smart_cache = SmartCache()
        
        # Очереди и результаты
        self.task_queue: List[ProcessingTask] = []
        self.processing_tasks: Dict[str, ProcessingTask] = {}
        self.results: Dict[str, ProcessingResult] = {}
        
        # Thread pools
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gpu_executor = ThreadPoolExecutor(max_workers=self.max_gpu_tasks) if self.max_gpu_tasks > 0 else None
        
        # Статус
        self.is_processing = False
        self.stop_requested = False
        
        # Статистика
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'throughput': 0.0  # файлов/сек
        }
        
        logging.info(f"🚀 ParallelFileProcessor initialized:")
        logging.info(f"   CPU workers: {self.max_workers}")
        logging.info(f"   GPU workers: {self.max_gpu_tasks}")
        logging.info(f"   Batch size: {self.batch_size}")
    
    def _calculate_optimal_workers(self) -> int:
        """Вычислить оптимальное количество рабочих потоков"""
        cpu_count = mp.cpu_count()
        
        # Для I/O-bound задач используем больше потоков
        # Для CPU-bound задач ограничиваем количество
        if torch.cuda.is_available():
            # Если есть GPU, CPU используется меньше
            return min(cpu_count, 4)
        else:
            # Только CPU - используем больше потоков но не все
            return min(cpu_count * 2, 8)
    
    def process_files(self, file_paths: List[str], model_type: str = "gemini", 
                     custom_prompt: str = "", ocr_lang: str = "rus+eng",
                     priority_callback: Callable[[str], int] = None) -> bool:
        """
        Начать параллельную обработку файлов
        
        Args:
            file_paths: Список путей к файлам
            model_type: Тип модели для обработки
            custom_prompt: Пользовательский промпт
            ocr_lang: Язык OCR
            priority_callback: Функция для определения приоритета файла
        """
        if self.is_processing:
            logging.warning("Processing already in progress")
            return False
        
        try:
            # Создаём задачи
            self.task_queue.clear()
            self.processing_tasks.clear()
            self.results.clear()
            
            for i, file_path in enumerate(file_paths):
                priority = priority_callback(file_path) if priority_callback else 1
                
                task = ProcessingTask(
                    file_path=file_path,
                    task_id=f"task_{i}_{int(time.time())}",
                    priority=priority,
                    model_type=model_type,
                    custom_prompt=custom_prompt,
                    ocr_lang=ocr_lang
                )
                self.task_queue.append(task)
            
            # Сортируем по приоритету
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)
            
            # Обновляем статистику
            self.stats['total_tasks'] = len(self.task_queue)
            self.stats['completed_tasks'] = 0
            self.stats['failed_tasks'] = 0
            self.stats['cache_hits'] = 0
            
            # Запускаем обработку
            self.is_processing = True
            self.stop_requested = False
            
            # Обработка в отдельном потоке
            processing_thread = threading.Thread(target=self._process_queue)
            processing_thread.daemon = True
            processing_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start file processing: {e}")
            return False
    
    def _process_queue(self):
        """Основной цикл обработки очереди"""
        start_time = time.time()
        
        try:
            while self.task_queue and not self.stop_requested:
                # Получаем пакет задач
                batch = self._get_next_batch()
                if not batch:
                    break
                
                # Обрабатываем пакет параллельно
                self._process_batch(batch)
                
                # Обновляем прогресс
                completed = self.stats['completed_tasks'] + self.stats['failed_tasks']
                self.progress_updated.emit(completed, self.stats['total_tasks'])
            
            # Завершение обработки
            self.is_processing = False
            
            # Статистика
            total_time = time.time() - start_time
            self.stats['processing_time'] = total_time
            self.stats['throughput'] = self.stats['completed_tasks'] / total_time if total_time > 0 else 0
            
            # Собираем все результаты
            all_results = list(self.results.values())
            self.batch_completed.emit(all_results)
            
            logging.info(f"✅ Batch processing completed:")
            logging.info(f"   Total: {self.stats['total_tasks']}")
            logging.info(f"   Completed: {self.stats['completed_tasks']}")
            logging.info(f"   Failed: {self.stats['failed_tasks']}")
            logging.info(f"   Cache hits: {self.stats['cache_hits']}")
            logging.info(f"   Throughput: {self.stats['throughput']:.2f} files/sec")
            
        except Exception as e:
            logging.error(f"Error in processing queue: {e}")
            self.is_processing = False
    
    def _get_next_batch(self) -> List[ProcessingTask]:
        """Получить следующий пакет задач для обработки"""
        batch = []
        
        while len(batch) < self.batch_size and self.task_queue:
            task = self.task_queue.pop(0)
            batch.append(task)
            self.processing_tasks[task.task_id] = task
        
        return batch
    
    def _process_batch(self, batch: List[ProcessingTask]):
        """Обработать пакет задач параллельно"""
        # Разделяем задачи на CPU и GPU
        cpu_tasks = []
        gpu_tasks = []
        
        for task in batch:
            if (self.gpu_executor and 
                self.max_gpu_tasks > 0 and 
                self._should_use_gpu(task)):
                gpu_tasks.append(task)
            else:
                cpu_tasks.append(task)
        
        # Запускаем обработку параллельно
        futures = []
        
        # CPU задачи
        for task in cpu_tasks:
            future = self.cpu_executor.submit(self._process_single_task, task, use_gpu=False)
            futures.append(future)
        
        # GPU задачи
        if self.gpu_executor:
            for task in gpu_tasks:
                future = self.gpu_executor.submit(self._process_single_task, task, use_gpu=True)
                futures.append(future)
        
        # Ждём завершения всех задач в пакете
        for future in as_completed(futures):
            try:
                result = future.result()
                self._handle_task_result(result)
            except Exception as e:
                logging.error(f"Task execution error: {e}")
    
    def _should_use_gpu(self, task: ProcessingTask) -> bool:
        """Определить, стоит ли использовать GPU для задачи"""
        # GPU эффективен для тяжёлых моделей
        gpu_models = ['layoutlm', 'donut']
        return task.model_type.lower() in gpu_models
    
    def _process_single_task(self, task: ProcessingTask, use_gpu: bool = False) -> ProcessingResult:
        """Обработать одну задачу"""
        start_time = time.time()
        device_id = -1
        
        try:
            self.task_started.emit(task.task_id)
            
            # Проверяем кэш
            cached_result = self.smart_cache.get_cached_result(task.file_path, task.model_type)
            if cached_result:
                self.stats['cache_hits'] += 1
                return ProcessingResult(
                    task_id=task.task_id,
                    file_path=task.file_path,
                    success=True,
                    result=cached_result,
                    processing_time=time.time() - start_time,
                    model_type=task.model_type,
                    from_cache=True
                )
            
            # Получаем GPU устройство если нужно
            if use_gpu:
                device_id = self.gpu_manager.get_optimal_device()
                if not self.gpu_manager.acquire_device(device_id):
                    # Если GPU занято, используем CPU
                    use_gpu = False
                    device_id = -1
            
            # Получаем процессор модели
            processor = self._get_model_processor(task.model_type, device_id)
            if not processor:
                raise RuntimeError(f"Model processor not available: {task.model_type}")
            
            # Обрабатываем файл
            result = processor.process_image(
                task.file_path, 
                task.ocr_lang, 
                custom_prompt=task.custom_prompt
            )
            
            if result:
                # Сохраняем в кэш
                self.smart_cache.cache_result(task.file_path, task.model_type, result)
                
                return ProcessingResult(
                    task_id=task.task_id,
                    file_path=task.file_path,
                    success=True,
                    result=result,
                    processing_time=time.time() - start_time,
                    model_type=task.model_type,
                    from_cache=False
                )
            else:
                raise RuntimeError("Processing returned empty result")
                
        except Exception as e:
            return ProcessingResult(
                task_id=task.task_id,
                file_path=task.file_path,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                model_type=task.model_type
            )
        finally:
            # Освобождаем GPU
            if device_id != -1:
                self.gpu_manager.release_device(device_id)
    
    def _get_model_processor(self, model_type: str, device_id: int = -1):
        """Получить процессор модели"""
        # Здесь должна быть интеграция с ModelManager
        # Пока возвращаем None для демонстрации
        return None
    
    def _handle_task_result(self, result: ProcessingResult):
        """Обработать результат задачи"""
        self.results[result.task_id] = result
        
        if result.success:
            self.stats['completed_tasks'] += 1
            self.task_completed.emit(result.task_id, result.result or {})
        else:
            self.stats['failed_tasks'] += 1
            self.task_failed.emit(result.task_id, result.error or "Unknown error")
        
        # Удаляем из активных задач
        if result.task_id in self.processing_tasks:
            del self.processing_tasks[result.task_id]
    
    def stop_processing(self):
        """Остановить обработку"""
        self.stop_requested = True
        
        # Очищаем очередь
        self.task_queue.clear()
        
        logging.info("⏹️ Processing stop requested")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику обработки"""
        return self.stats.copy()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.stop_processing()
        
        # Закрываем thread pools
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.gpu_executor:
            self.gpu_executor.shutdown(wait=True)
        
        logging.info("🧹 ParallelFileProcessor cleaned up")


class OptimizedProcessingThread(QThread):
    """
    Оптимизированный поток обработки для интеграции с UI
    Использует ParallelFileProcessor для высокой производительности
    """
    
    # Сигналы для совместимости с существующим кодом
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    partial_result_signal = pyqtSignal(dict)
    
    def __init__(self, file_paths: Union[str, List[str]], model_type: str, 
                 ocr_lang: str = "rus+eng", is_folder: bool = False, 
                 model_manager=None, parent=None):
        super().__init__(parent)
        
        # Параметры обработки
        if isinstance(file_paths, str):
            if is_folder:
                # Получаем все поддерживаемые файлы из папки
                self.file_paths = self._get_supported_files(file_paths)
            else:
                self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
            
        self.model_type = model_type
        self.ocr_lang = ocr_lang
        self.is_folder = is_folder
        self.model_manager = model_manager
        
        # Процессор
        self.processor = ParallelFileProcessor()
        
        # Подключаем сигналы
        self.processor.task_completed.connect(self._on_task_completed)
        self.processor.task_failed.connect(self._on_task_failed)
        self.processor.progress_updated.connect(self._on_progress_updated)
        self.processor.batch_completed.connect(self._on_batch_completed)
        
        # Результаты
        self.results = []
        self.errors = []
    
    def _get_supported_files(self, folder_path: str) -> List[str]:
        """Получить поддерживаемые файлы из папки"""
        supported_files = []
        
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and utils.is_supported_format(file_path):
                    supported_files.append(file_path)
        except OSError as e:
            logging.error(f"Error reading directory {folder_path}: {e}")
        
        return sorted(supported_files)  # Сортируем для предсказуемости
    
    def run(self):
        """Основной метод выполнения"""
        try:
            if not self.file_paths:
                self.finished_signal.emit(None)
                return
            
            # Получаем пользовательский промпт
            custom_prompt = settings_manager.get_string(
                'Prompts', 
                f'{self.model_type}_prompt', 
                ''
            )
            
            # Запускаем параллельную обработку
            success = self.processor.process_files(
                file_paths=self.file_paths,
                model_type=self.model_type,
                custom_prompt=custom_prompt,
                ocr_lang=self.ocr_lang
            )
            
            if not success:
                self.error_signal.emit("Failed to start parallel processing")
                return
            
            # Ждём завершения (processor работает в отдельных потоках)
            while self.processor.is_processing and not self.isInterruptionRequested():
                self.msleep(100)
            
        except Exception as e:
            logging.error(f"Error in OptimizedProcessingThread: {e}")
            self.error_signal.emit(str(e))
    
    def _on_task_completed(self, task_id: str, result: dict):
        """Обработка завершения задачи"""
        self.results.append(result)
        
        if self.is_folder:
            # Для пакетной обработки отправляем частичный результат
            self.partial_result_signal.emit(result)
        
    def _on_task_failed(self, task_id: str, error: str):
        """Обработка ошибки задачи"""
        self.errors.append(error)
        logging.error(f"Task {task_id} failed: {error}")
    
    def _on_progress_updated(self, current: int, total: int):
        """Обновление прогресса"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_signal.emit(progress)
    
    def _on_batch_completed(self, all_results: List):
        """Завершение пакетной обработки"""
        if self.is_folder:
            # Для пакетной обработки возвращаем None (результаты уже отправлены)
            self.finished_signal.emit(None)
        else:
            # Для одиночного файла возвращаем первый результат
            if all_results and all_results[0].success:
                self.finished_signal.emit(all_results[0].result)
            else:
                self.finished_signal.emit(None)
    
    def stop(self):
        """Остановить обработку"""
        self.requestInterruption()
        self.processor.stop_processing()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.processor.cleanup()


# Псевдоним для совместимости с существующим кодом
ProcessingThread = OptimizedProcessingThread 
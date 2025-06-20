#!/usr/bin/env python3
"""
Optimized File Processing Core for InvoiceGemini
Высокопроизводительная обработка с параллельными вычислениями
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
import logging

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
import torch

from ..settings_manager import settings_manager
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
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass 
class ProcessingResult:
    """Результат обработки"""
    task_id: str
    file_path: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    from_cache: bool = False


class OptimizedFileProcessor(QObject):
    """
    Оптимизированный процессор файлов с параллельной обработкой
    """
    
    # Сигналы
    task_started = pyqtSignal(str)  # task_id
    task_completed = pyqtSignal(str, dict)  # task_id, result
    task_failed = pyqtSignal(str, str)  # task_id, error
    progress_updated = pyqtSignal(int, int)  # current, total
    batch_completed = pyqtSignal(list)  # all results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Конфигурация
        self.max_workers = self._get_optimal_workers()
        self.batch_size = 3
        
        # Ресурсы
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Состояние
        self.is_processing = False
        self.stop_requested = False
        
        # Данные
        self.task_queue: List[ProcessingTask] = []
        self.results: Dict[str, ProcessingResult] = {}
        
        # Статистика
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'processing_time': 0.0
        }
        
        logging.info(f"🚀 OptimizedFileProcessor: {self.max_workers} workers")
    
    def _get_optimal_workers(self) -> int:
        """Оптимальное количество потоков"""
        cpu_count = mp.cpu_count()
        # Для I/O операций используем больше потоков
        return min(cpu_count * 2, 8)
    
    def process_files(self, file_paths: List[str], model_type: str = "gemini",
                     custom_prompt: str = "", ocr_lang: str = "rus+eng") -> bool:
        """Запуск параллельной обработки файлов"""
        if self.is_processing:
            return False
        
        try:
            # Создаём задачи
            self.task_queue.clear()
            self.results.clear()
            
            for i, file_path in enumerate(file_paths):
                task = ProcessingTask(
                    file_path=file_path,
                    task_id=f"task_{i}_{int(time.time())}",
                    model_type=model_type,
                    custom_prompt=custom_prompt,
                    ocr_lang=ocr_lang
                )
                self.task_queue.append(task)
            
            # Статистика
            self.stats['total_tasks'] = len(self.task_queue)
            self.stats['completed_tasks'] = 0
            self.stats['failed_tasks'] = 0
            
            # Запуск
            self.is_processing = True
            self.stop_requested = False
            
            # Обработка в отдельном потоке
            processing_thread = threading.Thread(target=self._process_queue)
            processing_thread.daemon = True
            processing_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start processing: {e}")
            return False
    
    def _process_queue(self):
        """Основной цикл обработки"""
        start_time = time.time()
        
        try:
            while self.task_queue and not self.stop_requested:
                # Получаем пакет
                batch = self._get_next_batch()
                if not batch:
                    break
                
                # Обрабатываем параллельно
                self._process_batch(batch)
                
                # Прогресс
                completed = self.stats['completed_tasks'] + self.stats['failed_tasks']
                self.progress_updated.emit(completed, self.stats['total_tasks'])
            
            # Завершение
            self.is_processing = False
            total_time = time.time() - start_time
            self.stats['processing_time'] = total_time
            
            # Результаты
            all_results = list(self.results.values())
            self.batch_completed.emit(all_results)
            
            logging.info(f"✅ Processing completed: {self.stats['completed_tasks']}/{self.stats['total_tasks']}")
            
        except Exception as e:
            logging.error(f"Processing error: {e}")
            self.is_processing = False
    
    def _get_next_batch(self) -> List[ProcessingTask]:
        """Получить следующий пакет"""
        batch = []
        while len(batch) < self.batch_size and self.task_queue:
            batch.append(self.task_queue.pop(0))
        return batch
    
    def _process_batch(self, batch: List[ProcessingTask]):
        """Обработать пакет параллельно"""
        futures = []
        
        for task in batch:
            future = self.executor.submit(self._process_single_task, task)
            futures.append(future)
        
        # Ждём результаты
        for future in as_completed(futures):
            try:
                result = future.result()
                self._handle_result(result)
            except Exception as e:
                logging.error(f"Task execution error: {e}")
    
    def _process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """Обработать одну задачу"""
        start_time = time.time()
        
        try:
            self.task_started.emit(task.task_id)
            
            # Здесь должна быть реальная обработка
            # Пока имитируем
            time.sleep(0.1)  # Имитация обработки
            
            # Заглушка результата
            result = {
                "Поставщик": f"Компания {task.task_id}",
                "Номер счета": f"#{task.task_id[-4:]}",
                "Дата": datetime.now().strftime("%d.%m.%Y"),
                "Сумма": "10000.00"
            }
            
            return ProcessingResult(
                task_id=task.task_id,
                file_path=task.file_path,
                success=True,
                result=result,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                task_id=task.task_id,
                file_path=task.file_path,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _handle_result(self, result: ProcessingResult):
        """Обработать результат"""
        self.results[result.task_id] = result
        
        if result.success:
            self.stats['completed_tasks'] += 1
            self.task_completed.emit(result.task_id, result.result or {})
        else:
            self.stats['failed_tasks'] += 1
            self.task_failed.emit(result.task_id, result.error or "Unknown error")
    
    def stop_processing(self):
        """Остановить обработку"""
        self.stop_requested = True
        self.task_queue.clear()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.stop_processing()
        self.executor.shutdown(wait=True)


class OptimizedProcessingThread(QThread):
    """Оптимизированный поток для UI интеграции"""
    
    # Сигналы совместимости
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    partial_result_signal = pyqtSignal(dict)
    
    def __init__(self, file_paths: Union[str, List[str]], model_type: str,
                 ocr_lang: str = "rus+eng", is_folder: bool = False,
                 model_manager=None, parent=None):
        super().__init__(parent)
        
        # Подготовка путей
        if isinstance(file_paths, str):
            if is_folder:
                self.file_paths = self._get_files_from_folder(file_paths)
            else:
                self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
        
        self.model_type = model_type
        self.ocr_lang = ocr_lang
        self.is_folder = is_folder
        
        # Процессор
        self.processor = OptimizedFileProcessor()
        
        # Подключения
        self.processor.task_completed.connect(self._on_task_completed)
        self.processor.task_failed.connect(self._on_task_failed)
        self.processor.progress_updated.connect(self._on_progress_updated)
        self.processor.batch_completed.connect(self._on_batch_completed)
        
        # Результаты
        self.results = []
    
    def _get_files_from_folder(self, folder_path: str) -> List[str]:
        """Получить файлы из папки"""
        files = []
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and utils.is_supported_format(file_path):
                    files.append(file_path)
        except OSError as e:
            logging.error(f"Error reading folder {folder_path}: {e}")
        return sorted(files)
    
    def run(self):
        """Выполнение"""
        try:
            if not self.file_paths:
                self.finished_signal.emit(None)
                return
            
            # Промпт
            custom_prompt = settings_manager.get_string(
                'Prompts',
                f'{self.model_type}_prompt',
                ''
            )
            
            # Запуск
            success = self.processor.process_files(
                file_paths=self.file_paths,
                model_type=self.model_type,
                custom_prompt=custom_prompt,
                ocr_lang=self.ocr_lang
            )
            
            if not success:
                self.error_signal.emit("Failed to start processing")
                return
            
            # Ожидание
            while self.processor.is_processing and not self.isInterruptionRequested():
                self.msleep(100)
                
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def _on_task_completed(self, task_id: str, result: dict):
        """Завершение задачи"""
        self.results.append(result)
        if self.is_folder:
            self.partial_result_signal.emit(result)
    
    def _on_task_failed(self, task_id: str, error: str):
        """Ошибка задачи"""
        logging.error(f"Task {task_id} failed: {error}")
    
    def _on_progress_updated(self, current: int, total: int):
        """Прогресс"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_signal.emit(progress)
    
    def _on_batch_completed(self, all_results: List):
        """Завершение пакета"""
        if self.is_folder:
            self.finished_signal.emit(None)
        else:
            if all_results and all_results[0].success:
                self.finished_signal.emit(all_results[0].result)
            else:
                self.finished_signal.emit(None)
    
    def stop(self):
        """Остановка"""
        self.requestInterruption()
        self.processor.stop_processing()


# Псевдоним для замены
ProcessingThreadOptimized = OptimizedProcessingThread 
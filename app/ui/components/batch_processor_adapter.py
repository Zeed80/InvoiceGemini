"""
Адаптер для BatchProcessor для совместимости с MainWindow.
"""

from typing import Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal
from .batch_processor import BatchProcessingWidget, BatchItem, BatchProcessingThread


class BatchProcessorAdapter(QObject):
    """Адаптер для BatchProcessingWidget с упрощенным API для MainWindow"""
    
    # Signals
    processing_started = pyqtSignal(int)  # total files
    file_processed = pyqtSignal(str, dict, int, int)  # file_path, result, index, total
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # percentage
    status_updated = pyqtSignal(str)  # status message
    
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.widget = BatchProcessingWidget()
        self._current_batch = []
        self._processing_thread = None
        self._model_type = None
        self._ocr_lang = None
        self._model_settings = None
        
        # Не подключаем сигналы виджета, так как мы будем работать напрямую с потоком
        
    def process_folder(self, folder_path: str, model_type: str, 
                      ocr_lang: Optional[str] = None, 
                      model_settings: Optional[Dict] = None):
        """Обработка всех файлов в папке"""
        # Загружаем файлы в виджет
        self.widget.load_files_from_folder(folder_path)
        
        if not self.widget.items:
            self.error_occurred.emit("Не найдено файлов для обработки")
            return
            
        # Сохраняем настройки
        self._model_type = model_type
        self._ocr_lang = ocr_lang
        self._model_settings = model_settings or {}
        
        # Получаем процессор
        processor = self.model_manager.get_model(model_type)
        if not processor:
            self.error_occurred.emit(f"Процессор {model_type} не доступен")
            return
            
        # Настраиваем процессор для Gemini
        if model_type == 'gemini' and 'sub_model_id' in model_settings:
            processor.model_id = model_settings['sub_model_id']
            
        # Создаем и запускаем поток обработки
        self._processing_thread = BatchProcessingThread(
            self.widget.items,
            processor,
            model_type,
            ocr_lang,
            delay_between_files=0.1  # Небольшая задержка между файлами
        )
        
        # Подключаем сигналы потока
        self._processing_thread.item_started.connect(self._on_item_started)
        self._processing_thread.item_completed.connect(self._on_item_completed)
        self._processing_thread.item_error.connect(self._on_item_error)
        self._processing_thread.progress_updated.connect(self._on_progress_updated)
        self._processing_thread.batch_completed.connect(self._on_batch_completed)
        
        # Сигнал о начале
        total_files = len(self.widget.items)
        self.processing_started.emit(total_files)
        
        # Запускаем поток
        self._processing_thread.start()
        
    def _on_item_started(self, index: int):
        """Обработчик начала обработки элемента"""
        item = self.widget.items[index]
        self.status_updated.emit(f"Обработка файла: {item.file_name}")
        
    def _on_item_completed(self, index: int, result: dict):
        """Обработчик завершения обработки элемента"""
        item = self.widget.items[index]
        total = len(self.widget.items)
        
        # Отправляем сигнал с результатом
        self.file_processed.emit(item.file_path, result, index, total)
        
    def _on_item_error(self, index: int, error_message: str):
        """Обработчик ошибки обработки элемента"""
        item = self.widget.items[index]
        self.error_occurred.emit(f"Ошибка при обработке {item.file_name}: {error_message}")
        
    def _on_progress_updated(self, current: int, total: int):
        """Обновление прогресса"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_updated.emit(progress)
            
    def _on_batch_completed(self):
        """Завершение пакетной обработки"""
        self.processing_finished.emit()
        
    def cancel_processing(self):
        """Отмена текущей обработки"""
        if self._processing_thread and self._processing_thread.isRunning():
            self._processing_thread.cancel()
            self._processing_thread.wait()


# Alias for backward compatibility
BatchProcessor = BatchProcessorAdapter 
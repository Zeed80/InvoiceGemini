"""
Адаптер для BatchProcessor для совместимости с MainWindow.
"""

from typing import Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal
from .batch_processor import BatchProcessingWidget, BatchItem


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
        
    def process_folder(self, folder_path: str, model_type: str, 
                      ocr_lang: Optional[str] = None, 
                      model_settings: Optional[Dict] = None):
        """Обработка всех файлов в папке"""
        # Загружаем файлы
        self.widget.load_files_from_folder(folder_path)
        
        if not self.widget.items:
            self.error_occurred.emit("Не найдено файлов для обработки")
            return
            
        # Сохраняем текущие настройки
        self._current_model_type = model_type
        self._current_ocr_lang = ocr_lang
        self._current_model_settings = model_settings or {}
        
        # Сигнал о начале
        total_files = len(self.widget.items)
        self.processing_started.emit(total_files)
        
        # Обрабатываем файлы последовательно
        self._process_next_file(0)
        
    def _process_next_file(self, index: int):
        """Обработка следующего файла"""
        if index >= len(self.widget.items):
            # Все файлы обработаны
            self.processing_finished.emit()
            return
            
        item = self.widget.items[index]
        total = len(self.widget.items)
        
        # Обновляем статус
        self.status_updated.emit(f"Обработка файла {index + 1} из {total}: {item.file_name}")
        
        # Получаем процессор
        processor = self.model_manager.get_model(self._current_model_type)
        if not processor:
            self.error_occurred.emit(f"Процессор {self._current_model_type} не доступен")
            return
            
        try:
            # Обработка файла
            if self._current_model_type == 'gemini' and 'sub_model_id' in self._current_model_settings:
                # Для Gemini передаем sub_model_id
                processor.model_id = self._current_model_settings['sub_model_id']
                
            result = processor.process_image(item.file_path, self._current_ocr_lang)
            
            if result:
                # Успешная обработка
                self.file_processed.emit(item.file_path, result, index, total)
                
                # Обновляем прогресс
                progress = int((index + 1) / total * 100)
                self.progress_updated.emit(progress)
                
                # Обрабатываем следующий файл
                self._process_next_file(index + 1)
            else:
                # Ошибка обработки
                self.error_occurred.emit(f"Не удалось обработать файл: {item.file_name}")
                
        except Exception as e:
            self.error_occurred.emit(f"Ошибка при обработке {item.file_name}: {str(e)}")


# Alias for backward compatibility
BatchProcessor = BatchProcessorAdapter 
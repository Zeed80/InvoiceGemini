"""
Processing controller for coordinating document processing.
"""
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging
import tempfile

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QProgressBar, QPushButton, QMessageBox

from app.processing_engine import ModelManager
from app.threads import ProcessingThread
from app.settings_manager import settings_manager
from app.plugins.unified_plugin_manager import get_unified_plugin_manager


class ProcessingSignals(QObject):
    """Signals for processing controller."""
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(dict)  # results
    processing_error = pyqtSignal(str)  # error message
    progress_updated = pyqtSignal(int)  # progress percentage
    status_updated = pyqtSignal(str)  # status message


class ProcessingController(QObject):
    """Controller for document processing operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.signals = ProcessingSignals()
        
        # Core components
        self.model_manager = ModelManager()
        self.plugin_manager = get_unified_plugin_manager()
        self.universal_plugin_manager = self.plugin_manager  # Алиас для обратной совместимости
        
        # Processing state
        self._processing_thread: Optional[ProcessingThread] = None
        self._current_model: Optional[str] = None
        self._current_plugin = None
        self._temp_dir = tempfile.TemporaryDirectory()
        
        # UI references (set by parent)
        self.progress_bar: Optional[QProgressBar] = None
        self.process_button: Optional[QPushButton] = None
        
    def set_ui_components(self, progress_bar: QProgressBar, process_button: QPushButton):
        """Set UI component references."""
        self.progress_bar = progress_bar
        self.process_button = process_button
        
    def set_current_model(self, model_name: str):
        """Set the current model for processing."""
        self._current_model = model_name
        self.logger.info(f"Current model set to: {model_name}")
        
    def set_current_plugin(self, plugin):
        """Set the current LLM plugin."""
        self._current_plugin = plugin
        
    def process_file(self, file_path: Union[str, Path]) -> bool:
        """Process a single file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Check model selection
            if not self._current_model:
                self._show_error(self.tr("Модель не выбрана"))
                return False
                
            # Start processing
            self._start_processing([str(file_path)], is_folder=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start file processing: {e}")
            self._show_error(str(e))
            return False
            
    def process_folder(self, folder_path: Union[str, Path]) -> bool:
        """Process all files in a folder."""
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"Invalid folder path: {folder_path}")
                
            # Find all supported files
            supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
            files = []
            
            for ext in supported_extensions:
                files.extend(folder_path.glob(f"*{ext}"))
                files.extend(folder_path.glob(f"*{ext.upper()}"))
                
            if not files:
                self._show_error(self.tr("В папке не найдено поддерживаемых файлов"))
                return False
                
            # Convert to string paths
            file_paths = [str(f) for f in files]
            
            # Start processing
            self._start_processing(file_paths, is_folder=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start folder processing: {e}")
            self._show_error(str(e))
            return False
            
    def _start_processing(self, file_paths: List[str], is_folder: bool):
        """Start processing thread."""
        try:
            # Disable UI
            if self.process_button:
                self.process_button.setEnabled(False)
                self.process_button.setText(self.tr("Обработка..."))
                
            # Show progress bar
            if self.progress_bar:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
            # Emit signal
            self.signals.processing_started.emit()
            
            # Get settings
            ocr_lang = settings_manager.get_string('OCR', 'language', 'rus+eng')
            
            # Create processing thread based on model type
            if self._current_model in ['cloud_llm', 'local_llm'] and self._current_plugin:
                # Use plugin for processing
                self._processing_thread = self._create_plugin_thread(
                    file_paths, is_folder
                )
            else:
                # Use standard processing
                self._processing_thread = ProcessingThread(
                    file_paths,
                    self._current_model,
                    ocr_lang,
                    is_folder,
                    self.model_manager
                )
                
            # Connect signals
            self._processing_thread.progress.connect(self._on_progress_updated)
            self._processing_thread.finished.connect(self._on_processing_finished)
            self._processing_thread.error.connect(self._on_processing_error)
            
            # Start processing
            self._processing_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self._reset_ui()
            self._show_error(str(e))
            
    def _create_plugin_thread(self, file_paths: List[str], is_folder: bool) -> QThread:
        """Create processing thread for LLM plugin."""
        # This is a simplified version - in real implementation
        # you would create a specialized thread for plugin processing
        
        class PluginProcessingThread(QThread):
            progress = pyqtSignal(int)
            finished = pyqtSignal(object)
            error = pyqtSignal(str)
            
            def __init__(self, paths, plugin, is_folder):
                super().__init__()
                self.paths = paths
                self.plugin = plugin
                self.is_folder = is_folder
                
            def run(self):
                try:
                    if self.is_folder:
                        # Process each file
                        results = []
                        for i, path in enumerate(self.paths):
                            result = self.plugin.extract_invoice_data(path)
                            results.append(result)
                            progress = int((i + 1) / len(self.paths) * 100)
                            self.progress.emit(progress)
                        self.finished.emit({'files': results})
                    else:
                        # Single file
                        result = self.plugin.extract_invoice_data(self.paths[0])
                        self.progress.emit(100)
                        self.finished.emit(result)
                except Exception as e:
                    self.error.emit(str(e))
                    
        return PluginProcessingThread(file_paths, self._current_plugin, is_folder)
        
    def _on_progress_updated(self, value: int):
        """Handle progress update."""
        if self.progress_bar:
            self.progress_bar.setValue(value)
        self.signals.progress_updated.emit(value)
        
    def _on_processing_finished(self, results):
        """Handle processing completion."""
        try:
            # Reset UI
            self._reset_ui()
            
            # Emit results
            if results:
                self.signals.processing_finished.emit(results)
            else:
                self.signals.processing_error.emit(
                    self.tr("Обработка завершена без результатов")
                )
                
        except Exception as e:
            self.logger.error(f"Error handling processing results: {e}")
            self.signals.processing_error.emit(str(e))
            
    def _on_processing_error(self, error_msg: str):
        """Handle processing error."""
        self._reset_ui()
        self.signals.processing_error.emit(error_msg)
        self._show_error(error_msg)
        
    def _reset_ui(self):
        """Reset UI components to initial state."""
        if self.process_button:
            self.process_button.setEnabled(True)
            self.process_button.setText(self.tr("Обработать"))
            
        if self.progress_bar:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            
    def _show_error(self, message: str):
        """Show error message."""
        QMessageBox.critical(
            None,
            self.tr("Ошибка обработки"),
            message
        )
        
    def stop_processing(self):
        """Stop current processing."""
        if self._processing_thread and self._processing_thread.isRunning():
            self._processing_thread.quit()
            self._processing_thread.wait()
            self._processing_thread = None
            self._reset_ui()
            
    def is_processing(self) -> bool:
        """Check if processing is in progress."""
        return (self._processing_thread is not None and 
                self._processing_thread.isRunning())
                
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ['layoutlm', 'donut', 'gemini', 'cloud_llm', 'local_llm']
        
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_processing()
            self._temp_dir.cleanup()
        except (OSError, AttributeError) as e:
            # Temporary directory может быть уже очищен или недоступен
            pass 
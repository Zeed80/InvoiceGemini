"""
File list widget with processing indicators.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QProgressBar, QPushButton, QGroupBox, QFrame, QSizePolicy,
    QScrollArea, QApplication, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QFont, QIcon

# Импортируем существующий модуль анализа PDF
try:
    from ...pdf_text_analyzer import has_text_layer
    PDF_ANALYZER_AVAILABLE = True
except ImportError:
    PDF_ANALYZER_AVAILABLE = False


class ProcessingStatus(Enum):
    """Статус обработки файла."""
    NOT_PROCESSED = "not_processed"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class FileProcessingInfo:
    """Информация о обработке файла."""
    file_path: str
    requires_ocr: bool = None  # Будет определено автоматически
    progress: int = 0  # 0-100
    status: ProcessingStatus = ProcessingStatus.NOT_PROCESSED
    fields_recognized: int = 0
    total_fields: int = 0
    error_message: str = ""
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Автоматическое определение требования OCR после создания объекта."""
        if self.requires_ocr is None:
            self.requires_ocr = self._determine_ocr_requirement()
    
    def _determine_ocr_requirement(self) -> bool:
        """Определяет, требуется ли OCR для файла."""
        ext = Path(self.file_path).suffix.lower()
        
        # Для изображений всегда требуется OCR
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
            return True
            
        # Для PDF проверяем наличие текстового слоя
        elif ext == '.pdf':
            return self._pdf_requires_ocr()
            
        # Для других типов файлов OCR не требуется
        return False
        
    def _pdf_requires_ocr(self) -> bool:
        """Проверяет, требуется ли OCR для PDF файла."""
        if not PDF_ANALYZER_AVAILABLE:
            # Fallback логика
            try:
                import PyPDF2
                with open(self.file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages_to_check = min(3, len(pdf_reader.pages))
                    total_text_length = 0
                    
                    for page_num in range(pages_to_check):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text().strip()
                        total_text_length += len(text)
                        
                        if total_text_length > 100:
                            return False
                            
                    return total_text_length < 50
                    
            except (ImportError, Exception):
                return True
        
        try:
            # Используем существующий анализатор PDF из проекта
            has_text = has_text_layer(self.file_path)
            return not has_text
            
        except Exception:
            return True


class FileItemWidget(QWidget):
    """Виджет элемента файла в списке."""
    
    file_selected = pyqtSignal(str)  # file_path
    process_requested = pyqtSignal(str)  # file_path
    filename_clicked = pyqtSignal(str, dict)  # file_path, processing_data
    
    def __init__(self, file_info: FileProcessingInfo, parent=None):
        super().__init__(parent)
        self.file_info = file_info
        self._init_ui()
        self._update_display()
        
    def _init_ui(self):
        """Инициализация UI элемента файла."""
        self.setMaximumHeight(60)  # Уменьшили высоту
        self.setMinimumHeight(60)
        
        # Основной layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 3, 6, 3)  # Уменьшили отступы
        main_layout.setSpacing(6)  # Уменьшили промежутки
        
        # Галочка OCR слева
        self.ocr_checkbox = QCheckBox()
        self.ocr_checkbox.setEnabled(False)  # Только для отображения
        self.ocr_checkbox.setToolTip(self.tr("OCR требуется для обработки"))
        self.ocr_checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 0px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #ff9800;
                border: 2px solid #f57c00;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #4CAF50;
                border: 2px solid #388e3c;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.ocr_checkbox)
        
        # Информация о файле
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)  # Уменьшили промежуток
        
        # Название файла (кликабельное)
        self.filename_label = QLabel()
        self.filename_label.setFont(QFont("Arial", 8, QFont.Weight.Bold))  # Уменьшили шрифт
        self.filename_label.setWordWrap(True)
        self.filename_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.filename_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                text-decoration: underline;
            }
            QLabel:hover {
                color: #1976D2;
                background-color: rgba(33, 150, 243, 0.1);
                border-radius: 2px;
                padding: 1px;
            }
        """)
        self.filename_label.mousePressEvent = self._on_filename_clicked
        info_layout.addWidget(self.filename_label)
        
        # Информация о полях
        self.fields_label = QLabel()
        self.fields_label.setFont(QFont("Arial", 7))  # Еще меньше шрифт
        info_layout.addWidget(self.fields_label)
        
        info_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(info_widget)
        
        # Правая часть - прогресс и управление
        controls_widget = QWidget()
        controls_widget.setFixedWidth(100)  # Уменьшили ширину
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)  # Уменьшили промежуток
        
        # Статус обработки
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Arial", 7))  # Уменьшили шрифт
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.status_label)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(10)  # Уменьшили высоту
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 2px;
                text-align: center;
                font-size: 7px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 1px;
            }
        """)
        controls_layout.addWidget(self.progress_bar)
        
        # Кнопка обработки
        self.process_button = QPushButton("📄")
        self.process_button.setFixedSize(20, 20)  # Уменьшили размер
        self.process_button.setToolTip(self.tr("Обработать файл"))
        self.process_button.clicked.connect(self._on_process_clicked)
        controls_layout.addWidget(self.process_button)
        
        main_layout.addWidget(controls_widget)
        
        # Стиль рамки
        self.setStyleSheet("""
            FileItemWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
                margin: 1px;
            }
            FileItemWidget:hover {
                background-color: #f0f8ff;
                border-color: #4CAF50;
            }
        """)
        
        # Клик по виджету для выбора файла
        self.mousePressEvent = self._on_mouse_press
        
    def _on_mouse_press(self, event):
        """Обработка клика мыши для выбора файла."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.file_selected.emit(self.file_info.file_path)
            
    def _on_process_clicked(self):
        """Обработка клика кнопки обработки."""
        self.process_requested.emit(self.file_info.file_path)
        
    def _on_filename_clicked(self, event):
        """Обработка клика по имени файла."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Подготавливаем данные для передачи
            processing_data = {
                'status': self.file_info.status.value,
                'progress': self.file_info.progress,
                'fields_recognized': self.file_info.fields_recognized,
                'total_fields': self.file_info.total_fields,
                'requires_ocr': self.file_info.requires_ocr,
                'error_message': self.file_info.error_message,
                'processing_time': self.file_info.processing_time
            }
            self.filename_clicked.emit(self.file_info.file_path, processing_data)
        
    def _update_display(self):
        """Обновление отображения информации о файле."""
        # Имя файла
        filename = Path(self.file_info.file_path).name
        self.filename_label.setText(filename)
        
        # Статус OCR через галочку
        self.ocr_checkbox.setChecked(self.file_info.requires_ocr)
        if self.file_info.requires_ocr:
            self.ocr_checkbox.setToolTip(self.tr("OCR требуется для обработки"))
        else:
            self.ocr_checkbox.setToolTip(self.tr("OCR не требуется - файл содержит текст"))
        
        # Информация о полях
        if self.file_info.total_fields > 0:
            fields_text = f"Поля: {self.file_info.fields_recognized}/{self.file_info.total_fields}"
            accuracy = (self.file_info.fields_recognized / self.file_info.total_fields) * 100
            if accuracy >= 80:
                fields_color = "#4CAF50"
            elif accuracy >= 50:
                fields_color = "#ff9800" 
            else:
                fields_color = "#f44336"
            self.fields_label.setText(fields_text)
            self.fields_label.setStyleSheet(f"color: {fields_color};")
        else:
            self.fields_label.setText("Поля: не обработано")
            self.fields_label.setStyleSheet("color: #666;")
        
        # Прогресс
        self.progress_bar.setValue(self.file_info.progress)
        
        # Статус обработки (более компактно)
        status_text = ""
        status_color = "#666"
        
        if self.file_info.status == ProcessingStatus.NOT_PROCESSED:
            status_text = "Не обработано"
            status_color = "#666"
        elif self.file_info.status == ProcessingStatus.PROCESSING:
            status_text = "Обработка..."
            status_color = "#2196F3"
        elif self.file_info.status == ProcessingStatus.COMPLETED:
            status_text = "Завершено"
            status_color = "#4CAF50"
        elif self.file_info.status == ProcessingStatus.ERROR:
            status_text = "Ошибка"
            status_color = "#f44336"
            
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        
        # Кнопка обработки
        self.process_button.setEnabled(
            self.file_info.status in [ProcessingStatus.NOT_PROCESSED, ProcessingStatus.ERROR]
        )
        
    def update_file_info(self, file_info: FileProcessingInfo):
        """Обновление информации о файле."""
        self.file_info = file_info
        self._update_display()
        
    def set_selected(self, selected: bool):
        """Выделение элемента файла."""
        if selected:
            self.setStyleSheet("""
                FileItemWidget {
                    background-color: #e3f2fd;
                    border: 2px solid #2196F3;
                    border-radius: 4px;
                    margin: 1px;
                }
            """)
        else:
            self.setStyleSheet("""
                FileItemWidget {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin: 1px;
                }
                FileItemWidget:hover {
                    background-color: #f0f8ff;
                    border-color: #4CAF50;
                }
            """)


class FileListWidget(QWidget):
    """Виджет списка файлов с индикаторами обработки."""
    
    file_selected = pyqtSignal(str)  # file_path
    process_file_requested = pyqtSignal(str)  # file_path
    process_all_requested = pyqtSignal()
    filename_clicked = pyqtSignal(str, dict)  # file_path, processing_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_widgets: Dict[str, FileItemWidget] = {}
        self.file_infos: Dict[str, FileProcessingInfo] = {}
        self.current_selected_path: Optional[str] = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Инициализация UI списка файлов."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Заголовок и кнопки управления
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        self.title_label = QLabel(self.tr("📂 Список файлов"))
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Кнопка "Обработать все"
        self.process_all_button = QPushButton(self.tr("🚀 Все"))
        self.process_all_button.setFixedSize(50, 24)
        self.process_all_button.setToolTip(self.tr("Обработать все файлы"))
        self.process_all_button.clicked.connect(self.process_all_requested.emit)
        self.process_all_button.setEnabled(False)
        header_layout.addWidget(self.process_all_button)
        
        layout.addWidget(header_widget)
        
        # Область прокрутки для списка файлов
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Контейнер для элементов файлов
        self.files_container = QWidget()
        self.files_layout = QVBoxLayout(self.files_container)
        self.files_layout.setContentsMargins(4, 4, 4, 4)
        self.files_layout.setSpacing(2)
        self.files_layout.addStretch()  # Растягиватель внизу
        
        self.scroll_area.setWidget(self.files_container)
        layout.addWidget(self.scroll_area)
        
        # Информационная панель
        self.info_panel = QWidget()
        info_layout = QHBoxLayout(self.info_panel)
        info_layout.setContentsMargins(8, 4, 8, 4)
        
        self.files_count_label = QLabel(self.tr("Файлов: 0"))
        self.files_count_label.setFont(QFont("Arial", 8))
        info_layout.addWidget(self.files_count_label)
        
        info_layout.addStretch()
        
        self.processed_count_label = QLabel(self.tr("Обработано: 0"))
        self.processed_count_label.setFont(QFont("Arial", 8))
        info_layout.addWidget(self.processed_count_label)
        
        layout.addWidget(self.info_panel)
        
        # Стиль
        self.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #fafafa;
            }
        """)
        
    def set_files(self, file_paths: List[str]):
        """Установка списка файлов."""
        # Очистка текущих файлов
        self.clear_files()
        
        # Добавление новых файлов
        for file_path in file_paths:
            file_info = FileProcessingInfo(
                file_path=file_path,
                requires_ocr=self._requires_ocr(file_path)
            )
            self.add_file(file_info)
            
        self._update_counters()
        
    def _requires_ocr(self, file_path: str) -> bool:
        """Определение, требуется ли OCR для файла."""
        # Создаем временный FileProcessingInfo для определения OCR
        temp_info = FileProcessingInfo(file_path=file_path)
        return temp_info.requires_ocr
        
    def add_file(self, file_info: FileProcessingInfo):
        """Добавление файла в список."""
        if file_info.file_path in self.file_widgets:
            return  # Файл уже добавлен
            
        # Создание виджета файла
        file_widget = FileItemWidget(file_info)
        file_widget.file_selected.connect(self._on_file_selected)
        file_widget.process_requested.connect(self.process_file_requested.emit)
        file_widget.filename_clicked.connect(self.filename_clicked.emit)
        
        # Добавление в layout (перед растягивателем)
        self.files_layout.insertWidget(self.files_layout.count() - 1, file_widget)
        
        # Сохранение ссылок
        self.file_widgets[file_info.file_path] = file_widget
        self.file_infos[file_info.file_path] = file_info
        
        self.process_all_button.setEnabled(len(self.file_widgets) > 0)
        
    def remove_file(self, file_path: str):
        """Удаление файла из списка."""
        if file_path in self.file_widgets:
            widget = self.file_widgets.pop(file_path)
            self.files_layout.removeWidget(widget)
            widget.deleteLater()
            
            if file_path in self.file_infos:
                del self.file_infos[file_path]
                
            if self.current_selected_path == file_path:
                self.current_selected_path = None
                
            self._update_counters()
            self.process_all_button.setEnabled(len(self.file_widgets) > 0)
            
    def clear_files(self):
        """Очистка всех файлов."""
        for widget in self.file_widgets.values():
            self.files_layout.removeWidget(widget)
            widget.deleteLater()
            
        self.file_widgets.clear()
        self.file_infos.clear()
        self.current_selected_path = None
        self._update_counters()
        self.process_all_button.setEnabled(False)
        
    def update_file_progress(self, file_path: str, progress: int, status: ProcessingStatus = None):
        """Обновление прогресса обработки файла."""
        if file_path in self.file_infos:
            self.file_infos[file_path].progress = progress
            if status:
                self.file_infos[file_path].status = status
                
            if file_path in self.file_widgets:
                self.file_widgets[file_path].update_file_info(self.file_infos[file_path])
                
            self._update_counters()
            
    def update_file_fields(self, file_path: str, recognized_fields: int, total_fields: int):
        """Обновление информации о распознанных полях."""
        if file_path in self.file_infos:
            self.file_infos[file_path].fields_recognized = recognized_fields
            self.file_infos[file_path].total_fields = total_fields
            
            if file_path in self.file_widgets:
                self.file_widgets[file_path].update_file_info(self.file_infos[file_path])
                
    def set_file_error(self, file_path: str, error_message: str):
        """Установка ошибки обработки файла."""
        if file_path in self.file_infos:
            self.file_infos[file_path].status = ProcessingStatus.ERROR
            self.file_infos[file_path].error_message = error_message
            self.file_infos[file_path].progress = 0
            
            if file_path in self.file_widgets:
                self.file_widgets[file_path].update_file_info(self.file_infos[file_path])
                
            self._update_counters()
            
    def _on_file_selected(self, file_path: str):
        """Обработка выбора файла."""
        # Снятие выделения с предыдущего
        if self.current_selected_path and self.current_selected_path in self.file_widgets:
            self.file_widgets[self.current_selected_path].set_selected(False)
            
        # Выделение нового
        self.current_selected_path = file_path
        if file_path in self.file_widgets:
            self.file_widgets[file_path].set_selected(True)
            
        self.file_selected.emit(file_path)
        
    def _update_counters(self):
        """Обновление счетчиков файлов."""
        total_files = len(self.file_infos)
        processed_files = sum(
            1 for info in self.file_infos.values() 
            if info.status == ProcessingStatus.COMPLETED
        )
        
        self.files_count_label.setText(self.tr(f"Файлов: {total_files}"))
        self.processed_count_label.setText(self.tr(f"Обработано: {processed_files}"))
        
    def get_selected_file(self) -> Optional[str]:
        """Получение выбранного файла."""
        return self.current_selected_path
        
    def get_all_files(self) -> List[str]:
        """Получение всех файлов."""
        return list(self.file_infos.keys())
        
    def get_unprocessed_files(self) -> List[str]:
        """Получение необработанных файлов."""
        return [
            path for path, info in self.file_infos.items()
            if info.status == ProcessingStatus.NOT_PROCESSED
        ] 
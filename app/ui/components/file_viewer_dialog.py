"""
Диалог просмотра файлов с результатами обработки.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget,
    QWidget, QGroupBox, QFrame, QSplitter, QTextEdit, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon
from app.ui.performance_optimized_widgets import OptimizedTableWidget

try:
    import fitz  # PyMuPDF для PDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FileViewerDialog(QDialog):
    """Диалог для просмотра файлов с результатами обработки."""
    
    def __init__(self, file_path: str, processing_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.processing_data = processing_data
        self.filename = Path(file_path).name
        
        self._init_ui()
        self._load_file_content()
        
    def _init_ui(self):
        """Инициализация UI диалога."""
        self.setWindowTitle(self.tr("Просмотр файла: {name}").format(name=self.filename))
        self.setMinimumSize(900, 700)
        self.resize(1200, 800)
        
        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Заголовок с информацией о файле
        self._create_header()
        main_layout.addWidget(self.header_widget)
        
        # Основная область с табами
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Таб просмотра файла
        self._create_file_view_tab()
        
        # Таб результатов обработки (если файл обработан)
        if self.processing_data.get('status') == 'completed':
            self._create_results_tab()
        
        # Кнопки управления
        self._create_buttons()
        main_layout.addWidget(self.buttons_widget)
        
        # Стили
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
        """)
        
    def _create_header(self):
        """Создание заголовка с информацией о файле."""
        self.header_widget = QGroupBox(self.tr("Информация о файле"))
        header_layout = QVBoxLayout(self.header_widget)
        
        # Первая строка - основная информация
        info_layout1 = QHBoxLayout()
        
        # Имя файла
        filename_label = QLabel(f"📄 {self.filename}")
        filename_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_layout1.addWidget(filename_label)
        
        info_layout1.addStretch()
        
        # Статус обработки
        status = self.processing_data.get('status', 'not_processed')
        status_text = {
            'not_processed': '⏳ ' + self.tr('Не обработано'),
            'processing': '🔄 ' + self.tr('Обработка...'),
            'completed': '✅ ' + self.tr('Завершено'),
            'error': '❌ ' + self.tr('Ошибка')
        }.get(status, self.tr('Неизвестно'))
        
        status_color = {
            'not_processed': '#666',
            'processing': '#2196F3',
            'completed': '#4CAF50',
            'error': '#f44336'
        }.get(status, '#666')
        
        status_label = QLabel(status_text)
        status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        status_label.setStyleSheet(f"color: {status_color}; padding: 4px 8px; background-color: rgba(0,0,0,0.1); border-radius: 4px;")
        info_layout1.addWidget(status_label)
        
        header_layout.addLayout(info_layout1)
        
        # Вторая строка - детали
        info_layout2 = QHBoxLayout()
        
        # Путь к файлу
        path_label = QLabel(f"📁 {self.file_path}")
        path_label.setFont(QFont("Arial", 9))
        path_label.setStyleSheet("color: #666;")
        info_layout2.addWidget(path_label)
        
        info_layout2.addStretch()
        
        # OCR статус
        ocr_status = ("🟠 " + self.tr("OCR требуется")) if self.processing_data.get('requires_ocr', True) else ("🟢 " + self.tr("OCR не требуется"))
        ocr_label = QLabel(ocr_status)
        ocr_label.setFont(QFont("Arial", 9))
        info_layout2.addWidget(ocr_label)
        
        # Прогресс и поля (если есть)
        if status == 'completed':
            progress = self.processing_data.get('progress', 0)
            recognized = self.processing_data.get('fields_recognized', 0)
            total = self.processing_data.get('total_fields', 0)
            
            if total > 0:
                accuracy = (recognized / total) * 100
                fields_text = f"🎯 Поля: {recognized}/{total} ({accuracy:.1f}%)"
                fields_color = "#4CAF50" if accuracy >= 80 else "#ff9800" if accuracy >= 50 else "#f44336"
                
                fields_label = QLabel(fields_text)
                fields_label.setFont(QFont("Arial", 9))
                fields_label.setStyleSheet(f"color: {fields_color}; font-weight: bold;")
                info_layout2.addWidget(fields_label)
        
        header_layout.addLayout(info_layout2)
        
    def _create_file_view_tab(self):
        """Создание таба просмотра файла."""
        file_view_widget = QWidget()
        file_layout = QVBoxLayout(file_view_widget)
        
        # Заголовок
        view_title = QLabel("📖 " + self.tr("Просмотр содержимого файла"))
        view_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        file_layout.addWidget(view_title)
        
        # Область просмотра
        self.file_content_scroll = QScrollArea()
        self.file_content_scroll.setWidgetResizable(True)
        self.file_content_scroll.setMinimumHeight(400)
        
        # Контейнер для содержимого
        self.file_content_widget = QLabel()
        self.file_content_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_content_widget.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 20px;
            }
        """)
        
        self.file_content_scroll.setWidget(self.file_content_widget)
        file_layout.addWidget(self.file_content_scroll)
        
        self.tab_widget.addTab(file_view_widget, "📖 " + self.tr("Просмотр файла"))
        
    def _create_results_tab(self):
        """Создание таба с результатами обработки."""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Заголовок
        results_title = QLabel("📊 " + self.tr("Результаты обработки"))
        results_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_layout.addWidget(results_title)
        
        # Статистика обработки
        stats_group = QGroupBox(self.tr("Статистика"))
        stats_layout = QHBoxLayout(stats_group)
        
        progress = self.processing_data.get('progress', 0)
        recognized = self.processing_data.get('fields_recognized', 0)
        total = self.processing_data.get('total_fields', 0)
        processing_time = self.processing_data.get('processing_time', 0)
        
        # Прогресс
        progress_label = QLabel(self.tr("📈 Прогресс: {p}%").format(p=progress))
        stats_layout.addWidget(progress_label)
        
        # Поля
        if total > 0:
            accuracy = (recognized / total) * 100
            fields_text = self.tr("🎯 Распознано полей: {r}/{t} ({a:.1f}%)").format(r=recognized, t=total, a=accuracy)
        else:
            fields_text = self.tr("🎯 Поля не обработаны")
        fields_label = QLabel(fields_text)
        stats_layout.addWidget(fields_label)
        
        # Время обработки
        if processing_time > 0:
            time_text = self.tr("⏱️ Время: {t:.2f} сек").format(t=processing_time)
            time_label = QLabel(time_text)
            stats_layout.addWidget(time_label)
        
        stats_layout.addStretch()
        results_layout.addWidget(stats_group)
        
        # Таблица извлеченных данных (заглушка - данные нужно передавать отдельно)
        data_group = QGroupBox(self.tr("Извлеченные данные"))
        data_layout = QVBoxLayout(data_group)
        
        self.results_table = OptimizedTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels([self.tr("Поле"), self.tr("Значение")])
        
        # Пример данных (в реальном приложении данные должны передаваться)
        sample_data = [
            ("Поставщик", "ООО 'Пример'"),
            ("№ счета", "INV-001"),
            ("Дата счета", "2025-01-20"),
            ("Сумма с НДС", "150 000 ₽"),
            ("НДС %", "20%"),
            ("Валюта", "RUB")
        ]
        
        self.results_table.setRowCount(len(sample_data))
        for row, (field, value) in enumerate(sample_data):
            self.results_table.setItem(row, 0, QTableWidgetItem(field))
            self.results_table.setItem(row, 1, QTableWidgetItem(value))
        
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setStretchLastSection(True)
        
        data_layout.addWidget(self.results_table)
        results_layout.addWidget(data_group)
        
        self.tab_widget.addTab(results_widget, "📊 " + self.tr("Результаты"))
        
    def _create_buttons(self):
        """Создание кнопок управления."""
        self.buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(self.buttons_widget)
        
        # Кнопка открыть в системном просмотрщике
        self.open_external_button = QPushButton("🔗 " + self.tr("Открыть в системе"))
        self.open_external_button.setToolTip(self.tr("Открыть файл в стандартном приложении"))
        self.open_external_button.clicked.connect(self._open_external)
        buttons_layout.addWidget(self.open_external_button)
        
        # Кнопка копировать путь
        self.copy_path_button = QPushButton("📋 " + self.tr("Копировать путь"))
        self.copy_path_button.setToolTip(self.tr("Копировать путь к файлу в буфер обмена"))
        self.copy_path_button.clicked.connect(self._copy_path)
        buttons_layout.addWidget(self.copy_path_button)
        
        buttons_layout.addStretch()
        
        # Кнопка закрыть
        self.close_button = QPushButton("❌ " + self.tr("Закрыть"))
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)
        
    def _load_file_content(self):
        """Загрузка содержимого файла для просмотра."""
        try:
            file_ext = Path(self.file_path).suffix.lower()
            
            if file_ext == '.pdf':
                self._load_pdf_content()
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
                self._load_image_content()
            else:
                self.file_content_widget.setText(self.tr("⚠️ Предварительный просмотр недоступен для файлов типа {ext}").format(ext=file_ext))
                
        except Exception as e:
            self.file_content_widget.setText(self.tr("❌ Ошибка загрузки файла: {err}").format(err=str(e)))
            
    def _load_pdf_content(self):
        """Загрузка содержимого PDF файла."""
        if not PYMUPDF_AVAILABLE:
            self.file_content_widget.setText(self.tr("📄 PDF файл\n\n⚠️ Для предварительного просмотра PDF установите PyMuPDF:\npip install PyMuPDF"))
            return
            
        try:
            doc = fitz.open(self.file_path)
            if len(doc) > 0:
                # Рендерим первую страницу
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Увеличиваем масштаб
                img_data = pix.tobytes("png")
                
                # Создаем QPixmap и устанавливаем в label
                pixmap = QPixmap()
                pixmap.loadFromData(img_data)
                
                # Масштабируем для отображения
                scaled_pixmap = pixmap.scaled(
                    800, 1000, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.file_content_widget.setPixmap(scaled_pixmap)
                
                # Если страниц больше одной, добавляем информацию
                if len(doc) > 1:
                    info_text = self.tr("📄 PDF документ ({pages} стр.)\nОтображается первая страница").format(pages=len(doc))
                    self.file_content_widget.setToolTip(info_text)
            else:
                self.file_content_widget.setText(self.tr("📄 PDF файл пуст"))
                
            doc.close()
            
        except Exception as e:
            self.file_content_widget.setText(self.tr("❌ Ошибка загрузки PDF: {err}").format(err=str(e)))
            
    def _load_image_content(self):
        """Загрузка изображения."""
        try:
            pixmap = QPixmap(self.file_path)
            if not pixmap.isNull():
                # Масштабируем для отображения
                scaled_pixmap = pixmap.scaled(
                    800, 600, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.file_content_widget.setPixmap(scaled_pixmap)
                
                # Информация о размере
                info_text = self.tr("🖼️ Изображение {w}x{h}").format(w=pixmap.width(), h=pixmap.height())
                self.file_content_widget.setToolTip(info_text)
            else:
                self.file_content_widget.setText(self.tr("❌ Не удалось загрузить изображение"))
                
        except Exception as e:
            self.file_content_widget.setText(self.tr("❌ Ошибка загрузки изображения: {err}").format(err=str(e)))
            
    def _open_external(self):
        """Открытие файла в стандартном приложении."""
        try:
            if sys.platform.startswith('win'):
                os.startfile(self.file_path)
            elif sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.file_path}"')
            else:  # Linux
                os.system(f'xdg-open "{self.file_path}"')
        except Exception as e:
            print(self.tr("Ошибка открытия файла: {err}").format(err=e))
            
    def _copy_path(self):
        """Копирование пути к файлу в буфер обмена."""
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.file_path)
        except Exception as e:
            print(self.tr("Ошибка копирования пути: {err}").format(err=e)) 
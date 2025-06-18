"""
Компонент для улучшенной пакетной обработки файлов.
"""

import os
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QComboBox, QSpinBox, QGroupBox, QTextEdit, QFileDialog,
    QMessageBox, QProgressBar, QSplitter, QTabWidget
)
from PyQt6.QtGui import QIcon, QColor, QFont

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Элемент пакетной обработки"""
    file_path: str
    file_name: str
    file_size: int
    status: str = "pending"  # pending, processing, completed, error, skipped
    result: Optional[Dict] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    selected: bool = True


class BatchProcessingThread(QThread):
    """Поток для пакетной обработки файлов"""
    
    # Сигналы
    item_started = pyqtSignal(int)  # индекс элемента
    item_completed = pyqtSignal(int, dict)  # индекс, результат
    item_error = pyqtSignal(int, str)  # индекс, сообщение об ошибке
    progress_updated = pyqtSignal(int, int)  # текущий, всего
    batch_completed = pyqtSignal()
    
    def __init__(self, items: List[BatchItem], processor, model_type: str, 
                 ocr_lang: str = None, delay_between_files: float = 0):
        super().__init__()
        self.items = items
        self.processor = processor
        self.model_type = model_type
        self.ocr_lang = ocr_lang
        self.delay_between_files = delay_between_files
        self._cancelled = False
        
    def run(self):
        """Выполнение пакетной обработки"""
        selected_items = [(i, item) for i, item in enumerate(self.items) if item.selected]
        total_items = len(selected_items)
        
        for idx, (item_index, item) in enumerate(selected_items):
            if self._cancelled:
                break
                
            # Сигнал о начале обработки
            self.item_started.emit(item_index)
            
            start_time = datetime.now()
            
            try:
                # Обработка файла
                result = self.processor.process_image(
                    item.file_path, 
                    self.ocr_lang
                )
                
                if result:
                    item.result = result
                    item.status = "completed"
                    item.processing_time = (datetime.now() - start_time).total_seconds()
                    self.item_completed.emit(item_index, result)
                else:
                    item.status = "error"
                    item.error_message = "Не удалось обработать файл"
                    self.item_error.emit(item_index, item.error_message)
                    
            except Exception as e:
                item.status = "error"
                item.error_message = str(e)
                self.item_error.emit(item_index, str(e))
                logger.error(f"Ошибка обработки {item.file_name}: {e}")
                
            # Обновление прогресса
            self.progress_updated.emit(idx + 1, total_items)
            
            # Задержка между файлами
            if idx < len(selected_items) - 1 and self.delay_between_files > 0:
                self.msleep(int(self.delay_between_files * 1000))
                
        self.batch_completed.emit()
        
    def cancel(self):
        """Отмена обработки"""
        self._cancelled = True


class BatchProcessingWidget(QWidget):
    """Виджет для пакетной обработки файлов"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[BatchItem] = []
        self.processing_thread: Optional[BatchProcessingThread] = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Настройка интерфейса"""
        layout = QVBoxLayout(self)
        
        # Панель управления
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Разделитель для таблицы и статистики
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Таблица файлов
        self.files_table = self._create_files_table()
        splitter.addWidget(self.files_table)
        
        # Панель статистики
        stats_panel = self._create_stats_panel()
        splitter.addWidget(stats_panel)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Начать обработку")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        buttons_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Пауза")
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setEnabled(False)
        buttons_layout.addWidget(self.pause_button)
        
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        buttons_layout.addWidget(self.cancel_button)
        
        buttons_layout.addStretch()
        
        self.export_button = QPushButton("Экспорт результатов")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        buttons_layout.addWidget(self.export_button)
        
        layout.addLayout(buttons_layout)
        
    def _create_control_panel(self) -> QGroupBox:
        """Создание панели управления"""
        group = QGroupBox("Настройки обработки")
        layout = QHBoxLayout(group)
        
        # Выбор папки
        self.folder_button = QPushButton("Выбрать папку...")
        self.folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_button)
        
        self.folder_label = QLabel("Папка не выбрана")
        layout.addWidget(self.folder_label)
        
        layout.addStretch()
        
        # Задержка между файлами
        layout.addWidget(QLabel("Задержка (сек):"))
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(0, 60)
        self.delay_spinbox.setValue(1)
        self.delay_spinbox.setToolTip("Задержка между обработкой файлов")
        layout.addWidget(self.delay_spinbox)
        
        # Фильтр по типам файлов
        layout.addWidget(QLabel("Тип файлов:"))
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["Все", "PDF", "Изображения", "PDF и изображения"])
        self.file_type_combo.currentTextChanged.connect(self.filter_files)
        layout.addWidget(self.file_type_combo)
        
        return group
        
    def _create_files_table(self) -> QTableWidget:
        """Создание таблицы файлов"""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "", "Файл", "Размер", "Статус", "Время", "Результат"
        ])
        
        # Настройка колонок
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        table.setColumnWidth(0, 30)
        
        # Контекстное меню
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._show_context_menu)
        
        return table
        
    def _create_stats_panel(self) -> QTabWidget:
        """Создание панели статистики"""
        tabs = QTabWidget()
        
        # Вкладка статистики
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        self.stats_label = QLabel("Статистика обработки:")
        font = QFont()
        font.setBold(True)
        self.stats_label.setFont(font)
        stats_layout.addWidget(self.stats_label)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        tabs.addTab(stats_widget, "Статистика")
        
        # Вкладка логов
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        tabs.addTab(self.log_text, "Логи")
        
        return tabs
        
    def select_folder(self):
        """Выбор папки для обработки"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с файлами",
            ""
        )
        
        if folder:
            self.load_files_from_folder(folder)
            
    def load_files_from_folder(self, folder: str):
        """Загрузка файлов из папки"""
        self.items.clear()
        self.files_table.setRowCount(0)
        
        # Поддерживаемые форматы
        supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Сканирование папки
        folder_path = Path(folder)
        files = []
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
                
        # Сортировка по имени
        files.sort(key=lambda x: x.name)
        
        # Создание элементов
        for file_path in files:
            item = BatchItem(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_path.stat().st_size
            )
            self.items.append(item)
            self._add_item_to_table(item)
            
        # Обновление UI
        self.folder_label.setText(f"Выбрано файлов: {len(self.items)}")
        self.start_button.setEnabled(len(self.items) > 0)
        self._update_stats()
        
        # Лог
        self._add_log(f"Загружено {len(self.items)} файлов из папки {folder}")
        
    def _add_item_to_table(self, item: BatchItem):
        """Добавление элемента в таблицу"""
        row = self.files_table.rowCount()
        self.files_table.insertRow(row)
        
        # Чекбокс
        checkbox = QCheckBox()
        checkbox.setChecked(item.selected)
        checkbox.stateChanged.connect(lambda state, i=item: self._on_item_selection_changed(i, state))
        self.files_table.setCellWidget(row, 0, checkbox)
        
        # Имя файла
        self.files_table.setItem(row, 1, QTableWidgetItem(item.file_name))
        
        # Размер
        size_mb = item.file_size / (1024 * 1024)
        self.files_table.setItem(row, 2, QTableWidgetItem(f"{size_mb:.2f} MB"))
        
        # Статус
        status_item = QTableWidgetItem(self._get_status_text(item.status))
        status_item.setForeground(self._get_status_color(item.status))
        self.files_table.setItem(row, 3, status_item)
        
        # Время обработки
        time_text = f"{item.processing_time:.1f}s" if item.processing_time else "-"
        self.files_table.setItem(row, 4, QTableWidgetItem(time_text))
        
        # Результат
        result_text = self._get_result_summary(item)
        self.files_table.setItem(row, 5, QTableWidgetItem(result_text))
        
    def _get_status_text(self, status: str) -> str:
        """Получение текста статуса"""
        status_map = {
            "pending": "Ожидание",
            "processing": "Обработка...",
            "completed": "Завершено",
            "error": "Ошибка",
            "skipped": "Пропущено"
        }
        return status_map.get(status, status)
        
    def _get_status_color(self, status: str) -> QColor:
        """Получение цвета статуса"""
        color_map = {
            "pending": QColor("#666666"),
            "processing": QColor("#2196F3"),
            "completed": QColor("#4CAF50"),
            "error": QColor("#f44336"),
            "skipped": QColor("#FFA500")
        }
        return color_map.get(status, QColor("#000000"))
        
    def _get_result_summary(self, item: BatchItem) -> str:
        """Получение краткого описания результата"""
        if item.status == "error":
            return item.error_message or "Неизвестная ошибка"
        elif item.result:
            # Подсчет извлеченных полей
            fields_count = len([v for v in item.result.values() if v])
            return f"Извлечено полей: {fields_count}"
        else:
            return "-"
            
    def _on_item_selection_changed(self, item: BatchItem, state: int):
        """Обработка изменения выбора элемента"""
        item.selected = state == Qt.CheckState.Checked.value
        self._update_stats()
        
    def filter_files(self, filter_type: str):
        """Фильтрация файлов по типу"""
        for i, item in enumerate(self.items):
            show = True
            
            if filter_type == "PDF":
                show = item.file_name.lower().endswith('.pdf')
            elif filter_type == "Изображения":
                show = any(item.file_name.lower().endswith(ext) 
                          for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])
                          
            self.files_table.setRowHidden(i, not show)
            
    def start_processing(self):
        """Начало обработки"""
        if self.processing_thread and self.processing_thread.isRunning():
            return
            
        # Получаем процессор из родительского окна
        parent = self.parent()
        while parent and not hasattr(parent, 'model_manager'):
            parent = parent.parent()
            
        if not parent:
            QMessageBox.warning(self, "Ошибка", "Не удалось получить доступ к обработчику")
            return
            
        # Получаем текущие настройки
        model_type = getattr(parent, 'current_model_type', 'gemini')
        processor = parent.model_manager.get_model(model_type)
        ocr_lang = getattr(parent, 'current_ocr_lang', 'rus')
        
        # Создаем поток обработки
        self.processing_thread = BatchProcessingThread(
            self.items,
            processor,
            model_type,
            ocr_lang,
            self.delay_spinbox.value()
        )
        
        # Подключаем сигналы
        self.processing_thread.item_started.connect(self._on_item_started)
        self.processing_thread.item_completed.connect(self._on_item_completed)
        self.processing_thread.item_error.connect(self._on_item_error)
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.batch_completed.connect(self._on_batch_completed)
        
        # Обновляем UI
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len([i for i in self.items if i.selected]))
        
        # Запускаем обработку
        self.processing_thread.start()
        self._add_log("Начата пакетная обработка файлов")
        
    def pause_processing(self):
        """Приостановка обработки"""
        # TODO: Реализовать паузу
        pass
        
    def cancel_processing(self):
        """Отмена обработки"""
        if self.processing_thread:
            self.processing_thread.cancel()
            self.processing_thread.wait()
            
        self._on_batch_completed()
        self._add_log("Обработка отменена пользователем")
        
    def _on_item_started(self, index: int):
        """Обработка начала обработки элемента"""
        item = self.items[index]
        item.status = "processing"
        
        # Обновляем таблицу
        status_item = self.files_table.item(index, 3)
        status_item.setText(self._get_status_text("processing"))
        status_item.setForeground(self._get_status_color("processing"))
        
        self._add_log(f"Обработка файла: {item.file_name}")
        
    def _on_item_completed(self, index: int, result: dict):
        """Обработка завершения обработки элемента"""
        item = self.items[index]
        
        # Обновляем таблицу
        self.files_table.item(index, 3).setText(self._get_status_text("completed"))
        self.files_table.item(index, 3).setForeground(self._get_status_color("completed"))
        self.files_table.item(index, 4).setText(f"{item.processing_time:.1f}s")
        self.files_table.item(index, 5).setText(self._get_result_summary(item))
        
        self._add_log(f"Файл обработан: {item.file_name}")
        self._update_stats()
        
    def _on_item_error(self, index: int, error_message: str):
        """Обработка ошибки обработки элемента"""
        item = self.items[index]
        
        # Обновляем таблицу
        self.files_table.item(index, 3).setText(self._get_status_text("error"))
        self.files_table.item(index, 3).setForeground(self._get_status_color("error"))
        self.files_table.item(index, 5).setText(error_message)
        
        self._add_log(f"Ошибка обработки {item.file_name}: {error_message}")
        self._update_stats()
        
    def _on_progress_updated(self, current: int, total: int):
        """Обновление прогресса"""
        self.progress_bar.setValue(current)
        
    def _on_batch_completed(self):
        """Завершение пакетной обработки"""
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(True)
        
        self._add_log("Пакетная обработка завершена")
        self._update_stats()
        
        # Показываем сообщение
        completed = len([i for i in self.items if i.status == "completed"])
        errors = len([i for i in self.items if i.status == "error"])
        
        QMessageBox.information(
            self,
            "Обработка завершена",
            f"Обработано файлов: {completed}\nОшибок: {errors}"
        )
        
    def _update_stats(self):
        """Обновление статистики"""
        total = len(self.items)
        selected = len([i for i in self.items if i.selected])
        completed = len([i for i in self.items if i.status == "completed"])
        errors = len([i for i in self.items if i.status == "error"])
        
        # Общее время обработки
        total_time = sum(i.processing_time or 0 for i in self.items if i.processing_time)
        avg_time = total_time / completed if completed > 0 else 0
        
        stats_text = f"""
Всего файлов: {total}
Выбрано: {selected}
Обработано: {completed}
Ошибок: {errors}

Общее время: {total_time:.1f} сек
Среднее время: {avg_time:.1f} сек/файл
        """
        
        self.stats_text.setText(stats_text.strip())
        
    def _add_log(self, message: str):
        """Добавление записи в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def _show_context_menu(self, position):
        """Показ контекстного меню"""
        # TODO: Реализовать контекстное меню
        pass
        
    def export_results(self):
        """Экспорт результатов"""
        # TODO: Реализовать экспорт в различные форматы
        pass


# Alias for backward compatibility
BatchProcessor = BatchProcessingWidget 
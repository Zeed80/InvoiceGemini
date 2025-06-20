#!/usr/bin/env python3
"""
Optimized Preview Dialog for InvoiceGemini
Модульная архитектура с lazy loading для улучшенной производительности
"""

import os
import json
import copy
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QTextEdit, QSplitter, QGroupBox, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QScrollArea, QMessageBox, QProgressBar, QStatusBar,
    QFormLayout, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QRunnable, QThreadPool, QMutex
from PyQt6.QtGui import QIcon, QPixmap, QFont

from ..settings_manager import settings_manager
from .. import utils


class PreviewDataModel:
    """Модель данных для предварительного просмотра"""
    
    def __init__(self, results=None, model_type=None, file_path=None):
        self.original_results = results or {}
        self.current_results = copy.deepcopy(self.original_results)
        self.model_type = model_type or "unknown"
        self.file_path = file_path or ""
        self.is_modified = False
        self.comparison_results = {}
        self._change_mutex = QMutex()
    
    def update_field(self, field_id: str, value: Any) -> bool:
        """Безопасное обновление поля с threading protection"""
        self._change_mutex.lock()
        try:
            old_value = self.current_results.get(field_id)
            if old_value != value:
                self.current_results[field_id] = value
                self.is_modified = True
                return True
            return False
        finally:
            self._change_mutex.unlock()
    
    def get_changes(self) -> Dict[str, Any]:
        """Получить только изменённые поля"""
        changes = {}
        for key, value in self.current_results.items():
            original_value = self.original_results.get(key)
            if original_value != value:
                changes[key] = {"old": original_value, "new": value}
        return changes
    
    def reset_to_original(self):
        """Сброс к оригинальным данным"""
        self._change_mutex.lock()
        try:
            self.current_results = copy.deepcopy(self.original_results)
            self.is_modified = False
        finally:
            self._change_mutex.unlock()


class LazyTabWidget(QTabWidget):
    """TabWidget с lazy loading для производительности"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tab_creators = {}
        self._loaded_tabs = set()
        self.currentChanged.connect(self._on_tab_changed)
    
    def add_lazy_tab(self, tab_id: str, title: str, creator_func):
        """Добавить вкладку с отложенным созданием"""
        placeholder = QWidget()
        placeholder.setObjectName(f"placeholder_{tab_id}")
        index = self.addTab(placeholder, title)
        self._tab_creators[index] = creator_func
        return index
    
    def _on_tab_changed(self, index: int):
        """Создание контента вкладки при первом обращении"""
        if index in self._tab_creators and index not in self._loaded_tabs:
            try:
                # Создаем контент
                content_widget = self._tab_creators[index]()
                
                # Заменяем placeholder
                old_widget = self.widget(index)
                self.removeTab(index)
                self.insertTab(index, content_widget, self.tabText(index))
                self.setCurrentIndex(index)
                
                # Помечаем как загруженную
                self._loaded_tabs.add(index)
                
                # Очищаем memory
                if old_widget:
                    old_widget.deleteLater()
                    
            except Exception as e:
                print(f"❌ Ошибка загрузки вкладки {index}: {e}")


class CompactFieldEditor(QWidget):
    """Компактный редактор полей с оптимизированным UI"""
    
    field_changed = pyqtSignal(str, object)  # field_id, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_widgets = {}
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        header = QLabel("📝 Редактирование полей")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Scroll area для полей
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.fields_layout = QFormLayout(scroll_widget)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(400)  # Ограничиваем высоту
        layout.addWidget(scroll_area)
        
        # Кнопки действий
        actions_layout = QHBoxLayout()
        
        validate_btn = QPushButton("✅ Валидировать")
        validate_btn.clicked.connect(self._validate_fields)
        actions_layout.addWidget(validate_btn)
        
        actions_layout.addStretch()
        
        reset_btn = QPushButton("🔄 Сброс")
        reset_btn.clicked.connect(self._reset_fields)
        actions_layout.addWidget(reset_btn)
        
        layout.addLayout(actions_layout)
    
    def create_field_widgets(self, data_model: PreviewDataModel):
        """Создание виджетов полей на основе данных"""
        # Очищаем существующие виджеты
        self._clear_fields()
        
        # Получаем поля из настроек
        table_fields = settings_manager.get_table_fields()
        
        for field in table_fields[:10]:  # Ограничиваем количество для производительности
            if not field.get("visible", True):
                continue
                
            field_id = field.get("id", "")
            field_name = field.get("name", field_id)
            field_type = field.get("type", "text")
            
            # Создаем подходящий виджет
            widget = self._create_field_widget(field_type)
            
            # Устанавливаем значение
            current_value = data_model.current_results.get(field_id, "")
            self._set_widget_value(widget, current_value)
            
            # Подключаем сигналы
            self._connect_widget_signals(widget, field_id)
            
            # Сохраняем виджет
            self.field_widgets[field_id] = widget
            
            # Добавляем в layout
            self.fields_layout.addRow(f"{field_name}:", widget)
    
    def _create_field_widget(self, field_type: str) -> QWidget:
        """Создание виджета по типу поля"""
        if field_type == "number":
            widget = QDoubleSpinBox()
            widget.setRange(-999999.99, 999999.99)
            widget.setDecimals(2)
            return widget
        elif field_type == "integer":
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            return widget
        elif field_type == "date":
            widget = QLineEdit()
            widget.setPlaceholderText("ДД.ММ.ГГГГ")
            return widget
        elif field_type == "multiline":
            widget = QTextEdit()
            widget.setMaximumHeight(60)  # Компактный размер
            return widget
        else:  # text
            return QLineEdit()
    
    def _set_widget_value(self, widget: QWidget, value: Any):
        """Установка значения виджета"""
        if isinstance(widget, QLineEdit):
            widget.setText(str(value) if value else "")
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value) if value else "")
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            try:
                widget.setValue(float(value) if value else 0)
            except (ValueError, TypeError):
                widget.setValue(0)
    
    def _connect_widget_signals(self, widget: QWidget, field_id: str):
        """Подключение сигналов виджета"""
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda text: self.field_changed.emit(field_id, text))
        elif isinstance(widget, QTextEdit):
            widget.textChanged.connect(lambda: self.field_changed.emit(field_id, widget.toPlainText()))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda value: self.field_changed.emit(field_id, value))
    
    def _clear_fields(self):
        """Очистка всех полей"""
        for widget in self.field_widgets.values():
            widget.deleteLater()
        self.field_widgets.clear()
        
        # Очищаем layout
        while self.fields_layout.count():
            child = self.fields_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _validate_fields(self):
        """Валидация полей"""
        errors = []
        
        for field_id, widget in self.field_widgets.items():
            value = self._get_widget_value(widget)
            if not value and field_id in ["Поставщик", "Номер счета", "Дата"]:
                errors.append(f"Пустое обязательное поле: {field_id}")
        
        if errors:
            QMessageBox.warning(self, "Ошибки валидации", "\n".join(errors))
        else:
            QMessageBox.information(self, "Валидация", "✅ Все поля корректны!")
    
    def _reset_fields(self):
        """Сброс полей"""
        # Этот метод будет вызван из родительского компонента
        pass
    
    def _get_widget_value(self, widget: QWidget) -> Any:
        """Получение значения виджета"""
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QTextEdit):
            return widget.toPlainText()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return ""


class QuickExportPanel(QWidget):
    """Быстрый экспорт с предустановками"""
    
    export_requested = pyqtSignal(dict, str)  # data, format
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        header = QLabel("💾 Быстрый экспорт")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Кнопки быстрого экспорта
        buttons_layout = QGridLayout()
        
        excel_btn = QPushButton("📊 Excel")
        excel_btn.clicked.connect(lambda: self.export_requested.emit({}, "excel"))
        buttons_layout.addWidget(excel_btn, 0, 0)
        
        json_btn = QPushButton("📄 JSON")
        json_btn.clicked.connect(lambda: self.export_requested.emit({}, "json"))
        buttons_layout.addWidget(json_btn, 0, 1)
        
        csv_btn = QPushButton("📋 CSV")
        csv_btn.clicked.connect(lambda: self.export_requested.emit({}, "csv"))
        buttons_layout.addWidget(csv_btn, 1, 0)
        
        pdf_btn = QPushButton("📑 PDF")
        pdf_btn.clicked.connect(lambda: self.export_requested.emit({}, "pdf"))
        buttons_layout.addWidget(pdf_btn, 1, 1)
        
        layout.addLayout(buttons_layout)
        
        # Опции экспорта
        options_group = QGroupBox("⚙️ Опции")
        options_layout = QVBoxLayout(options_group)
        
        self.include_metadata = QCheckBox("Включить метаданные")
        self.include_metadata.setChecked(True)
        options_layout.addWidget(self.include_metadata)
        
        self.include_timestamps = QCheckBox("Включить временные метки")
        self.include_timestamps.setChecked(True)
        options_layout.addWidget(self.include_timestamps)
        
        layout.addWidget(options_group)
        layout.addStretch()


class OptimizedPreviewDialog(QDialog):
    """
    Оптимизированный диалог предварительного просмотра
    Ключевые улучшения:
    - Модульная архитектура
    - Lazy loading вкладок
    - Оптимизированная модель данных
    - Компактный UI
    - Улучшенная производительность
    """
    
    # Сигналы
    results_edited = pyqtSignal(dict)
    export_requested = pyqtSignal(dict, str)
    
    def __init__(self, results=None, model_type=None, file_path=None, parent=None):
        super().__init__(parent)
        
        # Модель данных
        self.data_model = PreviewDataModel(results, model_type, file_path)
        
        # UI компоненты (создаются по требованию)
        self.field_editor = None
        self.export_panel = None
        
        # Настройка UI
        self._setup_ui()
        self._setup_connections()
        
        # Автосохранение
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save)
        self.auto_save_timer.start(30000)  # 30 секунд
    
    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("🔍 Предварительный просмотр - InvoiceGemini")
        self.setMinimumSize(900, 600)
        self.resize(1200, 700)
        
        layout = QVBoxLayout(self)
        
        # Информация о файле
        info_panel = self._create_info_panel()
        layout.addWidget(info_panel)
        
        # Основная область с lazy tabs
        self.tab_widget = LazyTabWidget()
        
        # Добавляем вкладки с отложенной загрузкой
        self.tab_widget.add_lazy_tab("edit", "📝 Редактирование", self._create_edit_tab)
        self.tab_widget.add_lazy_tab("compare", "⚖️ Сравнение", self._create_compare_tab)
        self.tab_widget.add_lazy_tab("export", "💾 Экспорт", self._create_export_tab)
        
        layout.addWidget(self.tab_widget)
        
        # Статус бар
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        
        # Кнопки управления
        buttons_panel = self._create_buttons_panel()
        layout.addWidget(buttons_panel)
        
        # Обновляем статус
        self._update_status()
    
    def _create_info_panel(self) -> QWidget:
        """Создание информационной панели"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(80)
        
        layout = QHBoxLayout(panel)
        
        # Иконка типа модели
        model_icon = QLabel("🤖")
        model_icon.setFont(QFont("Arial", 16))
        layout.addWidget(model_icon)
        
        # Информация о модели и файле
        info_layout = QVBoxLayout()
        
        model_label = QLabel(f"Модель: {self.data_model.model_type}")
        model_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        info_layout.addWidget(model_label)
        
        file_info = "Папка" if os.path.isdir(self.data_model.file_path) else "Файл"
        file_name = os.path.basename(self.data_model.file_path) or "Неизвестно"
        file_label = QLabel(f"{file_info}: {file_name}")
        info_layout.addWidget(file_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Индикатор изменений
        self.changes_indicator = QLabel("💾")
        self.changes_indicator.setToolTip("Есть несохранённые изменения")
        self.changes_indicator.setVisible(False)
        layout.addWidget(self.changes_indicator)
        
        return panel
    
    def _create_edit_tab(self) -> QWidget:
        """Создание вкладки редактирования"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Левая часть - редактор полей
        self.field_editor = CompactFieldEditor()
        self.field_editor.field_changed.connect(self._on_field_changed)
        self.field_editor.create_field_widgets(self.data_model)
        layout.addWidget(self.field_editor, 2)
        
        # Правая часть - быстрые действия
        actions_panel = QWidget()
        actions_layout = QVBoxLayout(actions_panel)
        
        # Быстрый экспорт
        self.export_panel = QuickExportPanel()
        self.export_panel.export_requested.connect(self.export_requested)
        actions_layout.addWidget(self.export_panel)
        
        # Статистика изменений
        changes_group = QGroupBox("📊 Изменения")
        changes_layout = QVBoxLayout(changes_group)
        
        self.changes_count_label = QLabel("Изменений: 0")
        changes_layout.addWidget(self.changes_count_label)
        
        actions_layout.addWidget(changes_group)
        actions_layout.addStretch()
        
        layout.addWidget(actions_panel, 1)
        
        return tab
    
    def _create_compare_tab(self) -> QWidget:
        """Создание вкладки сравнения"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Таблица сравнения
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["Поле", "Оригинал", "Текущий"])
        
        # Настраиваем таблицу для производительности
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.setMaximumHeight(400)
        
        layout.addWidget(self.comparison_table)
        
        # Обновляем данные сравнения
        self._update_comparison_table()
        
        layout.addStretch()
        return tab
    
    def _create_export_tab(self) -> QWidget:
        """Создание вкладки экспорта"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Расширенные настройки экспорта
        settings_group = QGroupBox("⚙️ Настройки экспорта")
        settings_layout = QFormLayout(settings_group)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel (.xlsx)", "JSON (.json)", "CSV (.csv)", "PDF (.pdf)"])
        settings_layout.addRow("Формат:", self.format_combo)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Стандартный", "Подробный", "Компактный"])
        settings_layout.addRow("Шаблон:", self.template_combo)
        
        layout.addWidget(settings_group)
        
        # Предварительный просмотр
        preview_group = QGroupBox("👁️ Предварительный просмотр")
        preview_layout = QVBoxLayout(preview_group)
        
        self.export_preview = QTextEdit()
        self.export_preview.setReadOnly(True)
        self.export_preview.setMaximumHeight(200)
        self._update_export_preview()
        preview_layout.addWidget(self.export_preview)
        
        layout.addWidget(preview_group)
        layout.addStretch()
        
        return tab
    
    def _create_buttons_panel(self) -> QWidget:
        """Создание панели кнопок"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Левые кнопки
        help_btn = QPushButton("❓ Справка")
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        
        layout.addStretch()
        
        # Правые кнопки
        apply_btn = QPushButton("✅ Применить")
        apply_btn.clicked.connect(self._apply_changes)
        apply_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("✅ OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("❌ Отмена")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        return panel
    
    def _setup_connections(self):
        """Настройка соединений сигналов"""
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
    
    def _on_field_changed(self, field_id: str, value: Any):
        """Обработка изменения поля"""
        if self.data_model.update_field(field_id, value):
            self._update_status()
            self._update_changes_indicator()
            self._update_comparison_table()
    
    def _on_tab_changed(self, index: int):
        """Обработка смены вкладки"""
        tab_names = ["Редактирование", "Сравнение", "Экспорт"]
        if index < len(tab_names):
            self.status_bar.showMessage(f"Активна вкладка: {tab_names[index]}")
    
    def _update_status(self):
        """Обновление строки состояния"""
        status = "Изменено" if self.data_model.is_modified else "Без изменений"
        field_count = len(self.data_model.current_results)
        self.status_bar.showMessage(f"Статус: {status} | Полей: {field_count}")
    
    def _update_changes_indicator(self):
        """Обновление индикатора изменений"""
        self.changes_indicator.setVisible(self.data_model.is_modified)
        
        if hasattr(self, 'changes_count_label'):
            changes = self.data_model.get_changes()
            self.changes_count_label.setText(f"Изменений: {len(changes)}")
    
    def _update_comparison_table(self):
        """Обновление таблицы сравнения"""
        if not hasattr(self, 'comparison_table'):
            return
            
        changes = self.data_model.get_changes()
        self.comparison_table.setRowCount(len(changes))
        
        for row, (field, change) in enumerate(changes.items()):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(field))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(str(change["old"] or "")))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(str(change["new"] or "")))
    
    def _update_export_preview(self):
        """Обновление предварительного просмотра экспорта"""
        if not hasattr(self, 'export_preview'):
            return
            
        preview_text = f"=== ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ===\n"
        preview_text += f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
        preview_text += f"Модель: {self.data_model.model_type}\n\n"
        
        preview_text += "=== ДАННЫЕ ===\n"
        for key, value in list(self.data_model.current_results.items())[:5]:  # Только первые 5 для производительности
            preview_text += f"{key}: {value}\n"
            
        if len(self.data_model.current_results) > 5:
            preview_text += f"... и ещё {len(self.data_model.current_results) - 5} полей\n"
        
        self.export_preview.setPlainText(preview_text)
    
    def _apply_changes(self):
        """Применение всех изменений"""
        self.results_edited.emit(self.data_model.current_results)
        self.data_model.original_results = copy.deepcopy(self.data_model.current_results)
        self.data_model.is_modified = False
        self._update_status()
        self._update_changes_indicator()
        self.status_bar.showMessage("✅ Изменения применены", 3000)
    
    def _auto_save(self):
        """Автоматическое сохранение"""
        if self.data_model.is_modified:
            self._apply_changes()
    
    def _show_help(self):
        """Показать справку"""
        help_text = """
        <h3>🔍 Оптимизированный предварительный просмотр</h3>
        
        <h4>📝 Редактирование:</h4>
        <p>• Быстрое редактирование полей<br>
        • Автоматическая валидация<br>
        • Автосохранение каждые 30 секунд</p>
        
        <h4>⚖️ Сравнение:</h4>
        <p>• Показывает только изменённые поля<br>
        • Сравнение оригинал ↔ текущий</p>
        
        <h4>💾 Экспорт:</h4>
        <p>• Быстрый экспорт в разные форматы<br>
        • Предварительный просмотр<br>
        • Настраиваемые шаблоны</p>
        """
        QMessageBox.information(self, "Справка", help_text)
    
    def closeEvent(self, event):
        """Обработка закрытия диалога"""
        if self.data_model.is_modified:
            reply = QMessageBox.question(
                self, "Несохранённые изменения",
                "У вас есть несохранённые изменения. Сохранить их?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self._apply_changes()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
                return
        
        # Очистка ресурсов
        self.auto_save_timer.stop()
        event.accept()


# Псевдоним для совместимости
PreviewDialog = OptimizedPreviewDialog 
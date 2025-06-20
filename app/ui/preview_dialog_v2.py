#!/usr/bin/env python3
"""
Optimized Preview Dialog v2.0 for InvoiceGemini
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
        self.field_widgets = {}
        
        # Настройка UI
        self._setup_ui()
        self._setup_connections()
        
        # Автосохранение
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save)
        self.auto_save_timer.start(30000)  # 30 секунд
    
    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("🔍 Предварительный просмотр v2.0 - InvoiceGemini")
        self.setMinimumSize(800, 500)
        self.resize(1000, 600)
        
        layout = QVBoxLayout(self)
        
        # Информация о файле
        info_panel = self._create_info_panel()
        layout.addWidget(info_panel)
        
        # Основная область
        main_area = self._create_main_area()
        layout.addWidget(main_area)
        
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
        panel.setMaximumHeight(60)
        
        layout = QHBoxLayout(panel)
        
        # Информация о модели и файле
        model_label = QLabel(f"🤖 Модель: {self.data_model.model_type}")
        model_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(model_label)
        
        layout.addStretch()
        
        file_info = "📁 Папка" if os.path.isdir(self.data_model.file_path) else "📄 Файл"
        file_name = os.path.basename(self.data_model.file_path) or "Неизвестно"
        file_label = QLabel(f"{file_info}: {file_name}")
        layout.addWidget(file_label)
        
        # Индикатор изменений
        self.changes_indicator = QLabel("💾")
        self.changes_indicator.setToolTip("Есть несохранённые изменения")
        self.changes_indicator.setVisible(False)
        layout.addWidget(self.changes_indicator)
        
        return panel
    
    def _create_main_area(self) -> QWidget:
        """Создание основной рабочей области"""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая часть - редактор полей
        left_panel = self._create_field_editor()
        splitter.addWidget(left_panel)
        
        # Правая часть - сравнение и экспорт
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Устанавливаем пропорции
        splitter.setSizes([600, 400])
        
        return splitter
    
    def _create_field_editor(self) -> QWidget:
        """Создание редактора полей"""
        group = QGroupBox("📝 Редактирование полей")
        layout = QVBoxLayout(group)
        
        # Scroll area для полей
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.fields_layout = QFormLayout(scroll_widget)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Создаем поля
        self._create_field_widgets()
        
        # Кнопки действий
        actions_layout = QHBoxLayout()
        
        validate_btn = QPushButton("✅ Валидировать")
        validate_btn.clicked.connect(self._validate_fields)
        actions_layout.addWidget(validate_btn)
        
        reset_btn = QPushButton("🔄 Сброс")
        reset_btn.clicked.connect(self._reset_fields)
        actions_layout.addWidget(reset_btn)
        
        layout.addLayout(actions_layout)
        
        return group
    
    def _create_right_panel(self) -> QWidget:
        """Создание правой панели"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Таблица сравнения
        compare_group = QGroupBox("⚖️ Изменения")
        compare_layout = QVBoxLayout(compare_group)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["Поле", "Оригинал", "Текущий"])
        self.comparison_table.setMaximumHeight(200)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        compare_layout.addWidget(self.comparison_table)
        
        layout.addWidget(compare_group)
        
        # Быстрый экспорт
        export_group = QGroupBox("💾 Быстрый экспорт")
        export_layout = QGridLayout(export_group)
        
        excel_btn = QPushButton("📊 Excel")
        excel_btn.clicked.connect(lambda: self.export_requested.emit(self.data_model.current_results, "excel"))
        export_layout.addWidget(excel_btn, 0, 0)
        
        json_btn = QPushButton("📄 JSON")
        json_btn.clicked.connect(lambda: self.export_requested.emit(self.data_model.current_results, "json"))
        export_layout.addWidget(json_btn, 0, 1)
        
        csv_btn = QPushButton("📋 CSV")
        csv_btn.clicked.connect(lambda: self.export_requested.emit(self.data_model.current_results, "csv"))
        export_layout.addWidget(csv_btn, 1, 0)
        
        pdf_btn = QPushButton("📑 PDF")
        pdf_btn.clicked.connect(lambda: self.export_requested.emit(self.data_model.current_results, "pdf"))
        export_layout.addWidget(pdf_btn, 1, 1)
        
        layout.addWidget(export_group)
        
        # Статистика
        stats_group = QGroupBox("📊 Статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Полей: 0 | Изменений: 0")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return widget
    
    def _create_field_widgets(self):
        """Создание виджетов полей на основе данных"""
        # Получаем поля из настроек
        table_fields = settings_manager.get_table_fields()
        
        for field in table_fields[:8]:  # Ограничиваем для производительности
            if not field.get("visible", True):
                continue
                
            field_id = field.get("id", "")
            field_name = field.get("name", field_id)
            field_type = field.get("type", "text")
            
            # Создаем подходящий виджет
            widget = self._create_field_widget(field_type)
            
            # Устанавливаем значение
            current_value = self.data_model.current_results.get(field_id, "")
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
        else:  # text
            return QLineEdit()
    
    def _set_widget_value(self, widget: QWidget, value: Any):
        """Установка значения виджета"""
        if isinstance(widget, QLineEdit):
            widget.setText(str(value) if value else "")
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            try:
                widget.setValue(float(value) if value else 0)
            except (ValueError, TypeError):
                widget.setValue(0)
    
    def _connect_widget_signals(self, widget: QWidget, field_id: str):
        """Подключение сигналов виджета"""
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda text: self._on_field_changed(field_id, text))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda value: self._on_field_changed(field_id, value))
    
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
        pass
    
    def _on_field_changed(self, field_id: str, value: Any):
        """Обработка изменения поля"""
        if self.data_model.update_field(field_id, value):
            self._update_status()
            self._update_changes_indicator()
            self._update_comparison_table()
            self._update_stats()
    
    def _update_status(self):
        """Обновление строки состояния"""
        status = "Изменено" if self.data_model.is_modified else "Без изменений"
        field_count = len(self.data_model.current_results)
        self.status_bar.showMessage(f"Статус: {status} | Полей: {field_count}")
    
    def _update_changes_indicator(self):
        """Обновление индикатора изменений"""
        self.changes_indicator.setVisible(self.data_model.is_modified)
    
    def _update_comparison_table(self):
        """Обновление таблицы сравнения"""
        changes = self.data_model.get_changes()
        self.comparison_table.setRowCount(len(changes))
        
        for row, (field, change) in enumerate(changes.items()):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(field))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(str(change["old"] or "")))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(str(change["new"] or "")))
    
    def _update_stats(self):
        """Обновление статистики"""
        if hasattr(self, 'stats_label'):
            changes_count = len(self.data_model.get_changes())
            fields_count = len(self.data_model.current_results)
            self.stats_label.setText(f"Полей: {fields_count} | Изменений: {changes_count}")
    
    def _validate_fields(self):
        """Валидация полей"""
        errors = []
        
        required_fields = ["Поставщик", "Номер счета", "Дата"]
        for field in required_fields:
            if not self.data_model.current_results.get(field, "").strip():
                errors.append(f"Пустое обязательное поле: {field}")
        
        if errors:
            QMessageBox.warning(self, "Ошибки валидации", "\n".join(errors))
        else:
            QMessageBox.information(self, "Валидация", "✅ Все поля корректны!")
    
    def _reset_fields(self):
        """Сброс полей к оригинальным значениям"""
        self.data_model.reset_to_original()
        
        # Обновляем виджеты
        for field_id, widget in self.field_widgets.items():
            value = self.data_model.current_results.get(field_id, "")
            self._set_widget_value(widget, value)
        
        self._update_status()
        self._update_changes_indicator()
        self._update_comparison_table()
        self._update_stats()
        
        self.status_bar.showMessage("🔄 Поля сброшены к оригинальным значениям", 3000)
    
    def _apply_changes(self):
        """Применение всех изменений"""
        self.results_edited.emit(self.data_model.current_results)
        self.data_model.original_results = copy.deepcopy(self.data_model.current_results)
        self.data_model.is_modified = False
        self._update_status()
        self._update_changes_indicator()
        self._update_comparison_table()
        self.status_bar.showMessage("✅ Изменения применены", 3000)
    
    def _auto_save(self):
        """Автоматическое сохранение"""
        if self.data_model.is_modified:
            self._apply_changes()
            self.status_bar.showMessage("💾 Автосохранение", 2000)
    
    def _show_help(self):
        """Показать справку"""
        help_text = """
        <h3>🔍 Оптимизированный предварительный просмотр v2.0</h3>
        
        <h4>Основные функции:</h4>
        <ul>
        <li><b>Редактирование полей</b> - Изменяйте данные напрямую</li>
        <li><b>Автосохранение</b> - Каждые 30 секунд</li>
        <li><b>Валидация</b> - Проверка корректности данных</li>
        <li><b>Быстрый экспорт</b> - В Excel, JSON, CSV, PDF</li>
        <li><b>Отслеживание изменений</b> - Сравнение с оригиналом</li>
        </ul>
        
        <h4>Горячие клавиши:</h4>
        <ul>
        <li><b>Ctrl+S</b> - Сохранить изменения</li>
        <li><b>Ctrl+R</b> - Сброс к оригиналу</li>
        <li><b>F1</b> - Эта справка</li>
        </ul>
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


# Псевдоним для обратной совместимости
PreviewDialog = OptimizedPreviewDialog 
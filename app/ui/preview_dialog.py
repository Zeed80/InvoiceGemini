#!/usr/bin/env python3
"""
Preview Dialog for InvoiceGemini - Result Preview and Editing System
Allows users to preview, edit, and compare results from different models before export.
"""

import os
import json
import copy
from typing import Dict, List, Optional, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QTextEdit, QSplitter, QGroupBox, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QScrollArea, QMessageBox, QProgressBar, QMenuBar, QMenu,
    QToolBar, QFileDialog, QStatusBar, QGridLayout,
    QFormLayout, QButtonGroup, QRadioButton, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QIcon, QPixmap, QFont, QColor, QPalette, QAction

from ..settings_manager import settings_manager
from .. import utils
from .. import config as app_config
from ..ui.performance_optimized_widgets import OptimizedTableWidget, SmartProgressBar


class PreviewDialog(QDialog):
    """
    Comprehensive preview dialog for invoice extraction results.
    Features:
    - Side-by-side model comparison
    - Interactive editing capabilities
    - Export preview and validation
    - Batch result management
    """
    
    # Signals
    results_edited = pyqtSignal(dict)  # Emitted when results are edited
    export_requested = pyqtSignal(dict, str)  # Emitted when export is requested
    comparison_requested = pyqtSignal(list)  # Emitted when comparison is requested
    
    def __init__(self, results=None, model_type=None, file_path=None, parent=None):
        super().__init__(parent)
        
        # Store initial data
        self.original_results = results or {}
        self.current_results = copy.deepcopy(self.original_results)
        self.model_type = model_type or "unknown"
        self.file_path = file_path or ""
        self.comparison_results = {}  # For storing results from multiple models
        
        # UI state
        self.is_modified = False
        self.editing_enabled = True
        self.auto_save_enabled = True
        
        self.init_ui()
        self.setup_connections()
        self.load_initial_data()
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(30000)  # Auto-save every 30 seconds
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🔍 " + self.tr("Предварительный просмотр результатов - InvoiceGemini"))
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main content area with tabs
        self.tab_widget = QTabWidget()
        
        # Tab 1: Single Result Preview
        self.single_preview_tab = self.create_single_preview_tab()
        self.tab_widget.addTab(self.single_preview_tab, "📄 Просмотр результата")
        
        # Tab 2: Model Comparison
        self.comparison_tab = self.create_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "⚖️ Сравнение моделей")
        
        # Tab 3: Batch Preview
        self.batch_tab = self.create_batch_preview_tab()
        self.tab_widget.addTab(self.batch_tab, "📂 Пакетный просмотр")
        
        # Tab 4: Export Preview
        self.export_tab = self.create_export_preview_tab()
        self.tab_widget.addTab(self.export_tab, "💾 Предварительный экспорт")
        
        main_layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.create_status_bar()
        
        # Create bottom button panel
        self.create_bottom_panel()
        
        # Apply styling
        self.apply_styling()
    
    def create_menu_bar(self):
        """Create menu bar"""
        self.menu_bar = QMenuBar(self)
        
        # File menu
        file_menu = self.menu_bar.addMenu("📁 " + self.tr("Файл"))
        
        self.save_action = QAction("💾 " + self.tr("Сохранить изменения"), self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_changes)
        file_menu.addAction(self.save_action)
        
        self.save_as_action = QAction("📄 " + self.tr("Сохранить как..."), self)
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(self.save_as_action)
        
        file_menu.addSeparator()
        
        self.export_action = QAction("📤 " + self.tr("Экспорт..."), self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_results)
        file_menu.addAction(self.export_action)
        
        # Edit menu
        edit_menu = self.menu_bar.addMenu("✏️ " + self.tr("Редактирование"))
        
        self.undo_action = QAction("↶ " + self.tr("Отменить"), self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_changes)
        edit_menu.addAction(self.undo_action)
        
        self.reset_action = QAction("🔄 " + self.tr("Сбросить к оригиналу"), self)
        self.reset_action.triggered.connect(self.reset_to_original)
        edit_menu.addAction(self.reset_action)
        
        edit_menu.addSeparator()
        
        self.toggle_editing_action = QAction("🔒 " + self.tr("Блокировать редактирование"), self)
        self.toggle_editing_action.setCheckable(True)
        self.toggle_editing_action.triggered.connect(self.toggle_editing)
        edit_menu.addAction(self.toggle_editing_action)
        
        # View menu
        view_menu = self.menu_bar.addMenu("👁️ " + self.tr("Вид"))
        
        self.show_original_action = QAction("📋 " + self.tr("Показать оригинал"), self)
        self.show_original_action.setCheckable(True)
        self.show_original_action.triggered.connect(self.toggle_original_view)
        view_menu.addAction(self.show_original_action)
        
        self.show_diff_action = QAction("🔍 " + self.tr("Показать изменения"), self)
        self.show_diff_action.setCheckable(True)
        self.show_diff_action.triggered.connect(self.toggle_diff_view)
        view_menu.addAction(self.show_diff_action)
        
        # Tools menu
        tools_menu = self.menu_bar.addMenu("🔧 " + self.tr("Инструменты"))
        
        self.validate_action = QAction("✅ " + self.tr("Валидация данных"), self)
        self.validate_action.triggered.connect(self.validate_results)
        tools_menu.addAction(self.validate_action)
        
        self.compare_models_action = QAction("⚖️ " + self.tr("Сравнить модели"), self)
        self.compare_models_action.triggered.connect(self.compare_models)
        tools_menu.addAction(self.compare_models_action)
        
        # Add menu bar to layout
        self.layout().setMenuBar(self.menu_bar)
    
    def create_toolbar(self):
        """Create toolbar with quick actions"""
        self.toolbar = QToolBar(self)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Save button
        save_btn = QPushButton("💾 " + self.tr("Сохранить"))
        save_btn.clicked.connect(self.save_changes)
        save_btn.setToolTip(self.tr("Сохранить изменения (Ctrl+S)"))
        self.toolbar.addWidget(save_btn)
        
        self.toolbar.addSeparator()
        
        # Edit toggle
        self.edit_toggle = QCheckBox("✏️ " + self.tr("Редактирование"))
        self.edit_toggle.setChecked(True)
        self.edit_toggle.toggled.connect(self.toggle_editing)
        self.toolbar.addWidget(self.edit_toggle)
        
        # Auto-save toggle
        self.auto_save_toggle = QCheckBox("🔄 " + self.tr("Авто-сохранение"))
        self.auto_save_toggle.setChecked(True)
        self.auto_save_toggle.toggled.connect(self.toggle_auto_save)
        self.toolbar.addWidget(self.auto_save_toggle)
        
        self.toolbar.addSeparator()
        
        # Export button
        export_btn = QPushButton("📤 " + self.tr("Экспорт"))
        export_btn.clicked.connect(self.export_results)
        export_btn.setToolTip(self.tr("Экспорт результатов (Ctrl+E)"))
        self.toolbar.addWidget(export_btn)
        
        # Add toolbar to layout
        self.layout().addWidget(self.toolbar)
    
    def create_single_preview_tab(self):
        """Create the single result preview tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left side: Editable fields
        left_group = QGroupBox("📝 " + self.tr("Редактируемые поля"))
        left_layout = QVBoxLayout(left_group)
        
        # Scroll area for fields
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.fields_layout = QFormLayout(scroll_widget)
        
        # Create input fields for each result field
        self.field_widgets = {}
        self.create_field_widgets()
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)
        
        # Right side: Original vs Current comparison
        right_group = QGroupBox("🔍 " + self.tr("Сравнение: Оригинал ↔ Текущий"))
        right_layout = QVBoxLayout(right_group)
        
        # Comparison table
        self.comparison_table = OptimizedTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels([self.tr("Поле"), self.tr("Оригинал"), self.tr("Текущий")])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.comparison_table)
        
        # Add to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_group)
        splitter.addWidget(right_group)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_comparison_tab(self):
        """Create the model comparison tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection controls
        controls_group = QGroupBox("🎯 " + self.tr("Управление сравнением"))
        controls_layout = QHBoxLayout(controls_group)
        
        # Available models
        models_label = QLabel(self.tr("Доступные модели:"))
        controls_layout.addWidget(models_label)
        
        self.models_combo = QComboBox()
        self.models_combo.addItems(["LayoutLMv3", "Donut", "Gemini 2.0", "LLM Plugin"])
        controls_layout.addWidget(self.models_combo)
        
        # Add model button
        add_model_btn = QPushButton("➕ " + self.tr("Добавить модель"))
        add_model_btn.clicked.connect(self.add_model_comparison)
        controls_layout.addWidget(add_model_btn)
        
        # Run comparison button
        run_comparison_btn = QPushButton("🚀 " + self.tr("Запустить сравнение"))
        run_comparison_btn.clicked.connect(self.run_model_comparison)
        controls_layout.addWidget(run_comparison_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Comparison results table
        comparison_group = QGroupBox("📊 " + self.tr("Результаты сравнения"))
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.model_comparison_table = OptimizedTableWidget()
        comparison_layout.addWidget(self.model_comparison_table)
        
        # Accuracy metrics
        metrics_group = QGroupBox("📈 " + self.tr("Метрики точности"))
        metrics_layout = QGridLayout(metrics_group)
        
        self.accuracy_labels = {}
        for i, metric in enumerate([self.tr("Точность полей"), self.tr("Полнота данных"), self.tr("Время обработки"), self.tr("Общая оценка")]):
            label = QLabel(f"{metric}:")
            value_label = QLabel("N/A")
            value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            metrics_layout.addWidget(label, i, 0)
            metrics_layout.addWidget(value_label, i, 1)
            self.accuracy_labels[metric] = value_label
        
        comparison_layout.addWidget(metrics_group)
        layout.addWidget(comparison_group)
        
        return tab
    
    def create_batch_preview_tab(self):
        """Create the batch preview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Batch controls
        controls_group = QGroupBox("📁 " + self.tr("Управление пакетом"))
        controls_layout = QHBoxLayout(controls_group)
        
        # Batch size info
        self.batch_info_label = QLabel(self.tr("Файлов в пакете: {n}").format(n=0))
        controls_layout.addWidget(self.batch_info_label)
        
        # Filter controls
        filter_label = QLabel(self.tr("Фильтр:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([self.tr("Все файлы"), self.tr("С ошибками"), self.tr("Успешные"), self.tr("Требующие проверки")])
        self.filter_combo.currentTextChanged.connect(self.filter_batch_results)
        
        controls_layout.addWidget(filter_label)
        controls_layout.addWidget(self.filter_combo)
        
        # Batch actions
        validate_all_btn = QPushButton("✅ " + self.tr("Валидировать все"))
        validate_all_btn.clicked.connect(self.validate_all_batch)
        controls_layout.addWidget(validate_all_btn)
        
        export_batch_btn = QPushButton("📤 " + self.tr("Экспорт пакета"))
        export_batch_btn.clicked.connect(self.export_batch)
        controls_layout.addWidget(export_batch_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Batch results table
        self.batch_table = OptimizedTableWidget()
        self.batch_table.itemSelectionChanged.connect(self.on_batch_item_selected)
        layout.addWidget(self.batch_table)
        
        # Batch statistics
        stats_group = QGroupBox("📊 " + self.tr("Статистика пакета"))
        stats_layout = QGridLayout(stats_group)
        
        self.batch_stats = {}
        stats = [self.tr("Обработано"), self.tr("Успешно"), self.tr("С ошибками"), self.tr("Требуют проверки"), self.tr("Средняя точность")]
        for i, stat in enumerate(stats):
            label = QLabel(f"{stat}:")
            value_label = QLabel("0")
            value_label.setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(label, i // 3, (i % 3) * 2)
            stats_layout.addWidget(value_label, i // 3, (i % 3) * 2 + 1)
            self.batch_stats[stat] = value_label
        
        layout.addWidget(stats_group)
        
        return tab
    
    def create_export_preview_tab(self):
        """Create the export preview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Export settings
        settings_group = QGroupBox("⚙️ " + self.tr("Настройки экспорта"))
        settings_layout = QFormLayout(settings_group)
        
        # Format selection
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([self.tr("Excel (.xlsx)"), "CSV (.csv)", "JSON (.json)", self.tr("PDF отчет (.pdf)"), self.tr("HTML отчет (.html)")])
        self.export_format_combo.currentTextChanged.connect(self.update_export_preview)
        settings_layout.addRow(self.tr("Формат:"), self.export_format_combo)
        
        # Template selection
        self.template_combo = QComboBox()
        self.template_combo.addItems([self.tr("Стандартный"), self.tr("Подробный"), self.tr("Сводка"), self.tr("Пользовательский")])
        self.template_combo.currentTextChanged.connect(self.update_export_preview)
        settings_layout.addRow(self.tr("Шаблон:"), self.template_combo)
        
        # Include options
        self.include_metadata_cb = QCheckBox(self.tr("Включить метаданные"))
        self.include_metadata_cb.setChecked(True)
        self.include_metadata_cb.toggled.connect(self.update_export_preview)
        settings_layout.addRow(self.include_metadata_cb)
        
        self.include_timestamps_cb = QCheckBox("Включить временные метки")
        self.include_timestamps_cb.setChecked(True)
        self.include_timestamps_cb.toggled.connect(self.update_export_preview)
        settings_layout.addRow("", self.include_timestamps_cb)
        
        layout.addWidget(settings_group)
        
        # Export preview
        preview_group = QGroupBox("👁️ Предварительный просмотр экспорта")
        preview_layout = QVBoxLayout(preview_group)
        
        self.export_preview_text = QTextEdit()
        self.export_preview_text.setReadOnly(True)
        self.export_preview_text.setMaximumHeight(300)
        preview_layout.addWidget(self.export_preview_text)
        
        layout.addWidget(preview_group)
        
        # Export actions
        actions_group = QGroupBox("🚀 Действия экспорта")
        actions_layout = QHBoxLayout(actions_group)
        
        preview_export_btn = QPushButton("👁️ Обновить предпросмотр")
        preview_export_btn.clicked.connect(self.update_export_preview)
        actions_layout.addWidget(preview_export_btn)
        
        test_export_btn = QPushButton("🧪 Тестовый экспорт")
        test_export_btn.clicked.connect(self.test_export)
        actions_layout.addWidget(test_export_btn)
        
        actions_layout.addStretch()
        
        final_export_btn = QPushButton("💾 Финальный экспорт")
        final_export_btn.clicked.connect(self.final_export)
        final_export_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        actions_layout.addWidget(final_export_btn)
        
        layout.addWidget(actions_group)
        
        return tab
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        
        # Status message
        self.status_label = QLabel("Готов")
        self.status_bar.addWidget(self.status_label)
        
        # Modification indicator
        self.modified_label = QLabel()
        self.status_bar.addPermanentWidget(self.modified_label)
        
        # File info
        self.file_info_label = QLabel()
        self.status_bar.addPermanentWidget(self.file_info_label)
        
        self.layout().addWidget(self.status_bar)
    
    def create_bottom_panel(self):
        """Create bottom button panel"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Left side buttons
        help_btn = QPushButton("❓ Справка")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)
        
        layout.addStretch()
        
        # Right side buttons
        apply_btn = QPushButton("✅ Применить изменения")
        apply_btn.clicked.connect(self.apply_changes)
        apply_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("✅ OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("❌ Отмена")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        self.layout().addWidget(panel)
    
    def create_field_widgets(self):
        """Create input widgets for all result fields"""
        # Get table fields from settings
        table_fields = settings_manager.get_table_fields()
        
        for field in table_fields:
            if not field.get("visible", True):
                continue
                
            field_id = field.get("id", "")
            field_name = field.get("name", field_id)
            field_type = field.get("type", "text")
            
            # Create appropriate widget based on field type
            if field_type == "number":
                widget = QDoubleSpinBox()
                widget.setRange(-999999.99, 999999.99)
                widget.setDecimals(2)
            elif field_type == "integer":
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
            elif field_type == "date":
                widget = QLineEdit()
                widget.setPlaceholderText("ДД.ММ.ГГГГ")
            elif field_type == "multiline":
                widget = QTextEdit()
                widget.setMaximumHeight(80)
            else:  # text
                widget = QLineEdit()
            
            # Connect change signals
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.on_field_changed)
            elif isinstance(widget, QTextEdit):
                widget.textChanged.connect(self.on_field_changed)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.on_field_changed)
            
            # Store widget
            self.field_widgets[field_id] = widget
            
            # Add to layout
            self.fields_layout.addRow(f"{field_name}:", widget)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Tab changes
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def load_initial_data(self):
        """Load initial data into the dialog"""
        # Check if we have batch results
        if isinstance(self.original_results, dict) and "batch_results" in self.original_results:
            # Load batch data
            self.load_batch_data()
            # Switch to batch tab by default for batch results
            self.tab_widget.setCurrentIndex(2)  # Batch tab is index 2
        else:
            # Update field widgets with current results for single file
            self.update_field_widgets()
            
            # Update comparison table
            self.update_comparison_table()
        
        # Update status
        self.update_status()
        
        # Update file info
        if self.file_path:
            if os.path.isdir(self.file_path):
                # For batch processing (folder)
                file_count = len([f for f in os.listdir(self.file_path) 
                                if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))])
                self.file_info_label.setText(f"📁 {os.path.basename(self.file_path)} ({file_count} файлов)")
            else:
                # For single file
                file_name = os.path.basename(self.file_path)
                file_size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
                self.file_info_label.setText(f"📄 {file_name} ({file_size} bytes)")
    
    def load_batch_data(self):
        """Load batch results into the batch table"""
        try:
            batch_results = self.original_results.get("batch_results", [])
            
            if not batch_results:
                self.batch_info_label.setText(self.tr("Файлов в пакете: 0"))
                return
            
            # Update batch info
            self.batch_info_label.setText(self.tr("Файлов в пакете: {n}").format(n=len(batch_results)))
            
            # Set up table columns based on first result
            if batch_results:
                headers = list(batch_results[0].keys())
                self.batch_table.setColumnCount(len(headers))
                self.batch_table.setHorizontalHeaderLabels(headers)
                
                # Fill table with data
                self.batch_table.setRowCount(len(batch_results))
                
                for row, result in enumerate(batch_results):
                    for col, header in enumerate(headers):
                        value = result.get(header, "")
                        item = QTableWidgetItem(str(value))
                        self.batch_table.setItem(row, col, item)
                
                # Resize columns to content
                self.batch_table.resizeColumnsToContents()
                
                # Update batch statistics
                self.update_batch_statistics(batch_results)
                
        except Exception as e:
            print(f"Ошибка загрузки пакетных данных: {e}")
            self.batch_info_label.setText(self.tr("Ошибка загрузки данных"))
    
    def update_batch_statistics(self, batch_results):
        """Update batch statistics"""
        try:
            total_files = len(batch_results)
            successful = 0
            with_errors = 0
            need_review = 0
            
            for result in batch_results:
                # Check for errors (look for error indicators in the data)
                has_error = any(
                    "ошибка" in str(value).lower() or "error" in str(value).lower() 
                    for value in result.values()
                )
                
                if has_error:
                    with_errors += 1
                else:
                    successful += 1
                    
                # Check if needs review (empty important fields)
                important_fields = ["Поставщик", "№ Счета", "Сумма с НДС", "Дата счета"]
                empty_important = any(
                    not result.get(field, "").strip() 
                    for field in important_fields 
                    if field in result
                )
                
                if empty_important and not has_error:
                    need_review += 1
            
            # Update statistics labels
            self.batch_stats["Обработано"].setText(str(total_files))
            self.batch_stats["Успешно"].setText(str(successful))
            self.batch_stats["С ошибками"].setText(str(with_errors))
            self.batch_stats["Требуют проверки"].setText(str(need_review))
            
            # Calculate average accuracy (simplified)
            if total_files > 0:
                accuracy = (successful / total_files) * 100
                self.batch_stats["Средняя точность"].setText(f"{accuracy:.1f}%")
            else:
                self.batch_stats["Средняя точность"].setText("0%")
                
        except Exception as e:
            print(f"Ошибка обновления статистики: {e}")
            # Set default values
            for stat in self.batch_stats.values():
                stat.setText("N/A")
    
    def update_field_widgets(self):
        """Update field widgets with current result values"""
        for field_id, widget in self.field_widgets.items():
            value = self.current_results.get(field_id, "")
            
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QDoubleSpinBox):
                try:
                    widget.setValue(float(value) if value else 0.0)
                except (ValueError, TypeError):
                    widget.setValue(0.0)
            elif isinstance(widget, QSpinBox):
                try:
                    widget.setValue(int(float(value)) if value else 0)
                except (ValueError, TypeError):
                    widget.setValue(0)
    
    def update_comparison_table(self):
        """Update the original vs current comparison table"""
        # Clear table
        self.comparison_table.setRowCount(0)
        
        # Get all unique keys
        all_keys = set(self.original_results.keys()) | set(self.current_results.keys())
        
        # Add rows
        for i, key in enumerate(sorted(all_keys)):
            self.comparison_table.insertRow(i)
            
            # Field name
            field_item = QTableWidgetItem(key)
            self.comparison_table.setItem(i, 0, field_item)
            
            # Original value
            original_value = str(self.original_results.get(key, ""))
            original_item = QTableWidgetItem(original_value)
            self.comparison_table.setItem(i, 1, original_item)
            
            # Current value
            current_value = str(self.current_results.get(key, ""))
            current_item = QTableWidgetItem(current_value)
            
            # Highlight changes
            if original_value != current_value:
                current_item.setBackground(QColor("#FFF3CD"))
                self.is_modified = True
            
            self.comparison_table.setItem(i, 2, current_item)
        
        self.comparison_table.resizeColumnsToContents()
    
    def update_status(self):
        """Update status indicators"""
        # Modified indicator
        if self.is_modified:
            self.modified_label.setText("✏️ Изменено")
            self.modified_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.modified_label.setText("✅ Сохранено")
            self.modified_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Status message
        field_count = len(self.current_results)
        self.status_label.setText(f"Полей: {field_count} | Модель: {self.model_type}")
    
    def apply_styling(self):
        """Apply custom styling to the dialog"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #F0F0F0;
                border: 1px solid #CCCCCC;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
            }
            QTableWidget {
                alternate-background-color: #F9F9F9;
                selection-background-color: #E3F2FD;
            }
            QToolBar {
                border: 1px solid #CCCCCC;
                padding: 5px;
            }
        """)
    
    # Event handlers
    def on_field_changed(self):
        """Handle field value changes"""
        sender = self.sender()
        
        # Find which field was changed
        for field_id, widget in self.field_widgets.items():
            if widget == sender:
                # Get new value
                if isinstance(widget, QLineEdit):
                    value = widget.text()
                elif isinstance(widget, QTextEdit):
                    value = widget.toPlainText()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()
                else:
                    value = ""
                
                # Update current results
                self.current_results[field_id] = value
                self.is_modified = True
                
                # Update comparison table
                self.update_comparison_table()
                self.update_status()
                break
    
    def on_tab_changed(self, index):
        """Handle tab changes"""
        tab_names = ["single", "comparison", "batch", "export"]
        if index < len(tab_names):
            self.status_label.setText(f"Активна вкладка: {tab_names[index]}")
    
    def on_batch_item_selected(self):
        """Handle batch item selection"""
        selected_items = self.batch_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            self.status_label.setText(f"Выбран файл: строка {row + 1}")
    
    # Action methods
    def save_changes(self):
        """Save current changes"""
        self.results_edited.emit(self.current_results)
        self.original_results = copy.deepcopy(self.current_results)
        self.is_modified = False
        self.update_status()
        self.status_label.setText("Изменения сохранены")
    
    def save_as(self):
        """Save results to a new file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Сохранить результаты как"),
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            self.tr("JSON files (*.json);;All files (*.*)")
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, ensure_ascii=False, indent=2)
                self.status_label.setText(self.tr("Результаты сохранены: {file}").format(file=os.path.basename(file_path)))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Ошибка"), self.tr("Не удалось сохранить файл:\n{error}").format(error=str(e)))
    
    def undo_changes(self):
        """Undo recent changes"""
        # For now, reset to original
        self.reset_to_original()
    
    def reset_to_original(self):
        """Reset to original results"""
        self.current_results = copy.deepcopy(self.original_results)
        self.is_modified = False
        self.update_field_widgets()
        self.update_comparison_table()
        self.update_status()
        self.status_label.setText("Сброшено к оригиналу")
    
    def toggle_editing(self, enabled=None):
        """Toggle editing mode"""
        if enabled is None:
            enabled = not self.editing_enabled
        
        self.editing_enabled = enabled
        
        # Update all field widgets
        for widget in self.field_widgets.values():
            widget.setEnabled(enabled)
        
        # Update action text
        if hasattr(self, 'toggle_editing_action'):
            self.toggle_editing_action.setText("🔓 Разрешить редактирование" if not enabled else "🔒 Блокировать редактирование")
        
        self.status_label.setText(f"Редактирование {'включено' if enabled else 'отключено'}")
    
    def toggle_auto_save(self, enabled):
        """Toggle auto-save functionality"""
        self.auto_save_enabled = enabled
        if enabled:
            self.auto_save_timer.start(30000)
        else:
            self.auto_save_timer.stop()
    
    def auto_save(self):
        """Perform auto-save if enabled and modified"""
        if self.auto_save_enabled and self.is_modified:
            self.save_changes()
    
    def toggle_original_view(self, checked):
        """Toggle original results view"""
        # Implementation for showing original view
        pass
    
    def toggle_diff_view(self, checked):
        """Toggle diff view"""
        # Implementation for showing differences
        pass
    
    def validate_results(self):
        """Validate current results"""
        validation_errors = []
        
        # Basic validation rules
        required_fields = ["Поставщик", "Номер счета", "Дата", "Сумма"]
        
        for field in required_fields:
            if not self.current_results.get(field, "").strip():
                validation_errors.append(f"Пустое обязательное поле: {field}")
        
        # Date validation
        date_field = self.current_results.get("Дата", "")
        if date_field and not self._validate_date_format(date_field):
            validation_errors.append("Неверный формат даты")
        
        # Amount validation
        amount_field = self.current_results.get("Сумма", "")
        if amount_field and not self._validate_amount_format(amount_field):
            validation_errors.append("Неверный формат суммы")
        
        # Show results
        if validation_errors:
            QMessageBox.warning(
                self,
                self.tr("Ошибки валидации"),
                self.tr("Найдены следующие ошибки:\n\n") + "\n".join(validation_errors)
            )
        else:
            QMessageBox.information(self, self.tr("Валидация"), self.tr("✅ Все данные корректны!"))
    
    def _validate_date_format(self, date_str):
        """Validate date format"""
        try:
            # Try common date formats
            from datetime import datetime
            for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except (ValueError, TypeError, AttributeError) as e:
            # Ошибка парсинга даты или неподдерживаемый тип
            return False
    
    def _validate_amount_format(self, amount_str):
        """Validate amount format"""
        try:
            # Remove common currency symbols and spaces
            cleaned = amount_str.replace("₽", "").replace("$", "").replace("€", "").replace(" ", "").replace(",", ".")
            float(cleaned)
            return True
        except (ValueError, TypeError) as e:
            # Ошибка парсинга числа
            return False
    
    def compare_models(self):
        """Open model comparison interface"""
        self.tab_widget.setCurrentIndex(1)  # Switch to comparison tab
        self.status_label.setText("Переключено на сравнение моделей")
    
    def add_model_comparison(self):
        """Add a model to comparison"""
        model_name = self.models_combo.currentText()
        # Implementation for adding model comparison
        self.status_label.setText(self.tr("Добавлена модель для сравнения: {model}").format(model=model_name))
    
    def run_model_comparison(self):
        """Run model comparison"""
        # Implementation for running comparison
        self.status_label.setText("Запущено сравнение моделей...")
    
    def filter_batch_results(self, filter_type):
        """Filter batch results based on type"""
        try:
            if not hasattr(self, 'batch_table') or self.batch_table.rowCount() == 0:
                return
            
            # Show all rows first
            for row in range(self.batch_table.rowCount()):
                self.batch_table.setRowHidden(row, False)
            
            # Apply filter
            if filter_type == "С ошибками":
                # Hide rows without errors
                for row in range(self.batch_table.rowCount()):
                    has_error = False
                    for col in range(self.batch_table.columnCount()):
                        item = self.batch_table.item(row, col)
                        if item and ("ошибка" in item.text().lower() or "error" in item.text().lower()):
                            has_error = True
                            break
                    self.batch_table.setRowHidden(row, not has_error)
                    
            elif filter_type == "Успешные":
                # Hide rows with errors
                for row in range(self.batch_table.rowCount()):
                    has_error = False
                    for col in range(self.batch_table.columnCount()):
                        item = self.batch_table.item(row, col)
                        if item and ("ошибка" in item.text().lower() or "error" in item.text().lower()):
                            has_error = True
                            break
                    self.batch_table.setRowHidden(row, has_error)
                    
            elif filter_type == "Требующие проверки":
                # Hide rows that don't need review
                important_columns = []
                for col in range(self.batch_table.columnCount()):
                    header = self.batch_table.horizontalHeaderItem(col)
                    if header and header.text() in ["Поставщик", "№ Счета", "Сумма с НДС", "Дата счета"]:
                        important_columns.append(col)
                
                for row in range(self.batch_table.rowCount()):
                    needs_review = False
                    for col in important_columns:
                        item = self.batch_table.item(row, col)
                        if not item or not item.text().strip():
                            needs_review = True
                            break
                    self.batch_table.setRowHidden(row, not needs_review)
            
            # Update status
            visible_rows = sum(1 for row in range(self.batch_table.rowCount()) 
                             if not self.batch_table.isRowHidden(row))
            self.status_label.setText(self.tr("Фильтр '{filter_type}': показано {visible_rows} из {total_rows}").format(filter_type=filter_type, visible_rows=visible_rows, total_rows=self.batch_table.rowCount()))
            
        except Exception as e:
            print(f"Ошибка фильтрации: {e}")
            self.status_label.setText(self.tr("Ошибка применения фильтра: {filter_type}").format(filter_type=filter_type))
    
    def validate_all_batch(self):
        """Validate all items in batch"""
        # Implementation for batch validation
        self.status_label.setText("Запущена валидация пакета...")
    
    def export_batch(self):
        """Export batch results"""
        # Implementation for batch export
        self.status_label.setText("Запущен экспорт пакета...")
    
    def update_export_preview(self):
        """Update the export preview"""
        format_type = self.export_format_combo.currentText()
        template = self.template_combo.currentText()
        
        # Generate preview based on format and template
        preview_text = f"=== ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ЭКСПОРТА ===\n"
        preview_text += f"Формат: {format_type}\n"
        preview_text += f"Шаблон: {template}\n"
        preview_text += f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
        
        # Add sample data
        if self.current_results:
            preview_text += "=== ДАННЫЕ ===\n"
            for key, value in self.current_results.items():
                preview_text += f"{key}: {value}\n"
        
        self.export_preview_text.setPlainText(preview_text)
        self.status_label.setText("Предварительный просмотр обновлен")
    
    def test_export(self):
        """Perform test export"""
        # Implementation for test export
        self.status_label.setText("Выполнен тестовый экспорт")
    
    def final_export(self):
        """Perform final export"""
        format_type = self.export_format_combo.currentText()
        self.export_requested.emit(self.current_results, format_type)
        self.status_label.setText("Запущен финальный экспорт")
    
    def export_results(self):
        """Export results through main export dialog"""
        self.export_requested.emit(self.current_results, "default")
    
    def apply_changes(self):
        """Apply all changes and update parent"""
        self.save_changes()
        self.accept()
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>📖 Справка по предварительному просмотру</h2>
        
        <h3>🔍 Просмотр результата</h3>
        <p>• Редактируйте поля напрямую<br>
        • Сравнивайте с оригинальными данными<br>
        • Автоматическое сохранение каждые 30 секунд</p>
        
        <h3>⚖️ Сравнение моделей</h3>
        <p>• Добавляйте результаты от разных моделей<br>
        • Сравнивайте точность и производительность<br>
        • Выбирайте лучшие результаты</p>
        
        <h3>📂 Пакетный просмотр</h3>
        <p>• Просматривайте результаты для всех файлов<br>
        • Фильтруйте по статусу обработки<br>
        • Валидируйте весь пакет</p>
        
        <h3>💾 Экспорт</h3>
        <p>• Выбирайте формат и шаблон<br>
        • Предварительный просмотр экспорта<br>
        • Тестовый экспорт перед финальным</p>
        
        <h3>⌨️ Горячие клавиши</h3>
        <p>• Ctrl+S: Сохранить изменения<br>
        • Ctrl+Z: Отменить<br>
        • Ctrl+E: Экспорт</p>
        """
        
        QMessageBox.information(self, self.tr("Справка"), help_text) 
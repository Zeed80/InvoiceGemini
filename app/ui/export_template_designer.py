#!/usr/bin/env python3
"""
Advanced Export Template Designer for InvoiceGemini
Allows users to create custom export templates with visual formatting and branding.
"""

import os
import json
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QTextEdit, QSplitter, QGroupBox, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QScrollArea, QMessageBox, QProgressBar, QMenuBar, QMenu,
    QToolBar, QFileDialog, QStatusBar, QGridLayout,
    QFormLayout, QButtonGroup, QRadioButton, QSpacerItem, QSizePolicy,
    QColorDialog, QFontDialog,
    QListWidget, QListWidgetItem, QTextBrowser
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QIcon, QPixmap, QFont, QColor, QPalette, QTextDocument, QAction

from ..settings_manager import settings_manager
from .. import utils
from .. import config as app_config


@dataclass
class FieldFormat:
    """Formatting options for a field"""
    font_family: str = "Arial"
    font_size: int = 12
    font_bold: bool = False
    font_italic: bool = False
    text_color: str = "#000000"
    background_color: str = "#FFFFFF"
    alignment: str = "left"  # left, center, right
    width: int = 100  # Column width in pixels
    visible: bool = True
    order: int = 0


@dataclass 
class ExportTemplate:
    """Complete export template definition"""
    name: str = "Новый шаблон"
    description: str = ""
    template_type: str = "table"  # table, report, invoice
    format: str = "excel"  # excel, pdf, html, csv
    
    # Header settings
    include_header: bool = True
    header_text: str = "Отчёт по обработке счетов"
    header_font: str = "Arial"
    header_size: int = 16
    header_bold: bool = True
    header_color: str = "#2196F3"
    
    # Footer settings
    include_footer: bool = True
    footer_text: str = "Создано InvoiceGemini"
    footer_font: str = "Arial"
    footer_size: int = 10
    footer_color: str = "#666666"
    
    # Company branding
    include_logo: bool = False
    logo_path: str = ""
    company_name: str = ""
    company_address: str = ""
    
    # Data formatting
    field_formats: Dict[str, FieldFormat] = None
    include_metadata: bool = True
    include_timestamps: bool = True
    include_summary: bool = False
    
    # Page settings (for PDF)
    page_orientation: str = "portrait"  # portrait, landscape
    page_size: str = "A4"  # A4, A3, Letter
    margin_top: int = 20
    margin_bottom: int = 20
    margin_left: int = 15
    margin_right: int = 15
    
    # Colors and styling
    table_border_color: str = "#CCCCCC"
    table_header_bg: str = "#F5F5F5"
    table_alt_row_bg: str = "#FAFAFA"
    
    def __post_init__(self):
        if self.field_formats is None:
            self.field_formats = {}


class ExportTemplateDesigner(QDialog):
    """
    Advanced export template designer with visual preview and formatting options.
    """
    
    # Signals
    template_saved = pyqtSignal(str)  # template_name
    template_applied = pyqtSignal(dict)  # template_data
    
    def __init__(self, current_results=None, parent=None):
        super().__init__(parent)
        
        # Data
        self.current_results = current_results or {}
        self.current_template = ExportTemplate()
        self.templates_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data", "templates")
        self.ensure_templates_directory()
        
        # UI state
        self.preview_document = QTextDocument()
        self.is_modified = False
        
        self.init_ui()
        self.load_builtin_templates()
        self.load_field_formats()
        self.update_preview()
        
        # Auto-update timer for preview
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.setSingleShot(True)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("📄 Дизайнер шаблонов экспорта - InvoiceGemini")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main content with splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Template settings
        left_panel = self.create_settings_panel()
        
        # Right panel: Preview and field formatting
        right_panel = self.create_preview_panel()
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([600, 800])
        
        main_layout.addWidget(main_splitter)
        
        # Create bottom panel
        self.create_bottom_panel()
        
        # Apply styling
        self.apply_styling()
    
    def create_toolbar(self):
        """Create toolbar with template actions"""
        self.toolbar = QToolBar(self)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Template management
        new_btn = QPushButton("📄 Новый")
        new_btn.clicked.connect(self.new_template)
        new_btn.setToolTip("Создать новый шаблон")
        self.toolbar.addWidget(new_btn)
        
        load_btn = QPushButton("📂 Загрузить")
        load_btn.clicked.connect(self.load_template)
        load_btn.setToolTip("Загрузить существующий шаблон")
        self.toolbar.addWidget(load_btn)
        
        save_btn = QPushButton("💾 Сохранить")
        save_btn.clicked.connect(self.save_template)
        save_btn.setToolTip("Сохранить текущий шаблон")
        self.toolbar.addWidget(save_btn)
        
        self.toolbar.addSeparator()
        
        # Quick actions
        preview_btn = QPushButton("👁️ Обновить предпросмотр")
        preview_btn.clicked.connect(self.update_preview)
        preview_btn.setToolTip("Обновить предварительный просмотр")
        self.toolbar.addWidget(preview_btn)
        
        export_btn = QPushButton("📤 Тестовый экспорт")
        export_btn.clicked.connect(self.test_export)
        export_btn.setToolTip("Экспорт с текущим шаблоном")
        self.toolbar.addWidget(export_btn)
        
        self.layout().addWidget(self.toolbar)
    
    def create_settings_panel(self):
        """Create left settings panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Template selector
        template_group = QGroupBox("📋 Управление шаблонами")
        template_layout = QVBoxLayout(template_group)
        
        # Template list
        template_list_layout = QHBoxLayout()
        template_list_layout.addWidget(QLabel("Шаблон:"))
        
        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(self.on_template_changed)
        template_list_layout.addWidget(self.template_combo)
        
        template_layout.addLayout(template_list_layout)
        
        # Template name and description
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Название:"))
        self.template_name_edit = QLineEdit()
        self.template_name_edit.textChanged.connect(self.on_template_modified)
        name_layout.addWidget(self.template_name_edit)
        template_layout.addLayout(name_layout)
        
        self.template_desc_edit = QTextEdit()
        self.template_desc_edit.setMaximumHeight(60)
        self.template_desc_edit.setPlaceholderText("Описание шаблона...")
        self.template_desc_edit.textChanged.connect(self.on_template_modified)
        template_layout.addWidget(self.template_desc_edit)
        
        layout.addWidget(template_group)
        
        # Settings tabs
        self.settings_tabs = QTabWidget()
        
        # General tab
        general_tab = self.create_general_settings_tab()
        self.settings_tabs.addTab(general_tab, "⚙️ Основные")
        
        # Header/Footer tab
        header_footer_tab = self.create_header_footer_tab()
        self.settings_tabs.addTab(header_footer_tab, "📄 Заголовок/Подвал")
        
        # Branding tab
        branding_tab = self.create_branding_tab()
        self.settings_tabs.addTab(branding_tab, "🏢 Брендинг")
        
        # Page settings tab
        page_tab = self.create_page_settings_tab()
        self.settings_tabs.addTab(page_tab, "📐 Страница")
        
        layout.addWidget(self.settings_tabs)
        
        return panel
    
    def create_general_settings_tab(self):
        """Create general settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Template type
        self.template_type_combo = QComboBox()
        self.template_type_combo.addItems(["Таблица", "Отчёт", "Счёт-фактура"])
        self.template_type_combo.currentTextChanged.connect(self.on_template_modified)
        layout.addRow("Тип шаблона:", self.template_type_combo)
        
        # Export format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel (.xlsx)", "PDF (.pdf)", "HTML (.html)", "CSV (.csv)"])
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        layout.addRow("Формат экспорта:", self.format_combo)
        
        # Include options
        layout.addRow(QLabel(""))  # Spacer
        
        self.include_metadata_cb = QCheckBox("Включить метаданные")
        self.include_metadata_cb.setChecked(True)
        self.include_metadata_cb.toggled.connect(self.on_template_modified)
        layout.addRow(self.include_metadata_cb)
        
        self.include_timestamps_cb = QCheckBox("Включить временные метки")
        self.include_timestamps_cb.setChecked(True)
        self.include_timestamps_cb.toggled.connect(self.on_template_modified)
        layout.addRow(self.include_timestamps_cb)
        
        self.include_summary_cb = QCheckBox("Включить сводку")
        self.include_summary_cb.toggled.connect(self.on_template_modified)
        layout.addRow(self.include_summary_cb)
        
        return tab
    
    def create_header_footer_tab(self):
        """Create header/footer settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header settings
        header_group = QGroupBox("📄 Заголовок")
        header_layout = QFormLayout(header_group)
        
        self.include_header_cb = QCheckBox("Включить заголовок")
        self.include_header_cb.setChecked(True)
        self.include_header_cb.toggled.connect(self.on_template_modified)
        header_layout.addRow(self.include_header_cb)
        
        self.header_text_edit = QLineEdit()
        self.header_text_edit.setText("Отчёт по обработке счетов")
        self.header_text_edit.textChanged.connect(self.on_template_modified)
        header_layout.addRow("Текст:", self.header_text_edit)
        
        # Header font controls
        header_font_layout = QHBoxLayout()
        
        self.header_font_btn = QPushButton("Шрифт")
        self.header_font_btn.clicked.connect(self.choose_header_font)
        header_font_layout.addWidget(self.header_font_btn)
        
        self.header_color_btn = QPushButton("Цвет")
        self.header_color_btn.clicked.connect(self.choose_header_color)
        self.header_color_btn.setStyleSheet("background-color: #2196F3; color: white;")
        header_font_layout.addWidget(self.header_color_btn)
        
        header_layout.addRow("Форматирование:", header_font_layout)
        
        layout.addWidget(header_group)
        
        # Footer settings
        footer_group = QGroupBox("📄 Подвал")
        footer_layout = QFormLayout(footer_group)
        
        self.include_footer_cb = QCheckBox("Включить подвал")
        self.include_footer_cb.setChecked(True)
        self.include_footer_cb.toggled.connect(self.on_template_modified)
        footer_layout.addRow(self.include_footer_cb)
        
        self.footer_text_edit = QLineEdit()
        self.footer_text_edit.setText("Создано InvoiceGemini")
        self.footer_text_edit.textChanged.connect(self.on_template_modified)
        footer_layout.addRow("Текст:", self.footer_text_edit)
        
        # Footer font controls
        footer_font_layout = QHBoxLayout()
        
        self.footer_font_btn = QPushButton("Шрифт")
        self.footer_font_btn.clicked.connect(self.choose_footer_font)
        footer_font_layout.addWidget(self.footer_font_btn)
        
        self.footer_color_btn = QPushButton("Цвет")
        self.footer_color_btn.clicked.connect(self.choose_footer_color)
        self.footer_color_btn.setStyleSheet("background-color: #666666; color: white;")
        footer_font_layout.addWidget(self.footer_color_btn)
        
        footer_layout.addRow("Форматирование:", footer_font_layout)
        
        layout.addWidget(footer_group)
        
        return tab
    
    def create_branding_tab(self):
        """Create branding settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Logo settings
        self.include_logo_cb = QCheckBox("Включить логотип")
        self.include_logo_cb.toggled.connect(self.on_template_modified)
        layout.addRow(self.include_logo_cb)
        
        logo_layout = QHBoxLayout()
        self.logo_path_edit = QLineEdit()
        self.logo_path_edit.textChanged.connect(self.on_template_modified)
        logo_layout.addWidget(self.logo_path_edit)
        
        logo_browse_btn = QPushButton("📂 Обзор")
        logo_browse_btn.clicked.connect(self.choose_logo)
        logo_layout.addWidget(logo_browse_btn)
        
        layout.addRow("Путь к логотипу:", logo_layout)
        
        # Company info
        layout.addRow(QLabel(""))  # Spacer
        
        self.company_name_edit = QLineEdit()
        self.company_name_edit.textChanged.connect(self.on_template_modified)
        layout.addRow("Название компании:", self.company_name_edit)
        
        self.company_address_edit = QTextEdit()
        self.company_address_edit.setMaximumHeight(80)
        self.company_address_edit.textChanged.connect(self.on_template_modified)
        layout.addRow("Адрес компании:", self.company_address_edit)
        
        return tab
    
    def create_page_settings_tab(self):
        """Create page settings tab (for PDF export)"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Page orientation
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Книжная", "Альбомная"])
        self.orientation_combo.currentTextChanged.connect(self.on_template_modified)
        layout.addRow("Ориентация:", self.orientation_combo)
        
        # Page size
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["A4", "A3", "Letter"])
        self.page_size_combo.currentTextChanged.connect(self.on_template_modified)
        layout.addRow("Размер страницы:", self.page_size_combo)
        
        # Margins
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("Поля (мм):"))
        
        margins_layout = QGridLayout()
        
        self.margin_top_spin = QSpinBox()
        self.margin_top_spin.setRange(0, 100)
        self.margin_top_spin.setValue(20)
        self.margin_top_spin.valueChanged.connect(self.on_template_modified)
        margins_layout.addWidget(QLabel("Верх:"), 0, 0)
        margins_layout.addWidget(self.margin_top_spin, 0, 1)
        
        self.margin_bottom_spin = QSpinBox()
        self.margin_bottom_spin.setRange(0, 100)
        self.margin_bottom_spin.setValue(20)
        self.margin_bottom_spin.valueChanged.connect(self.on_template_modified)
        margins_layout.addWidget(QLabel("Низ:"), 0, 2)
        margins_layout.addWidget(self.margin_bottom_spin, 0, 3)
        
        self.margin_left_spin = QSpinBox()
        self.margin_left_spin.setRange(0, 100)
        self.margin_left_spin.setValue(15)
        self.margin_left_spin.valueChanged.connect(self.on_template_modified)
        margins_layout.addWidget(QLabel("Лево:"), 1, 0)
        margins_layout.addWidget(self.margin_left_spin, 1, 1)
        
        self.margin_right_spin = QSpinBox()
        self.margin_right_spin.setRange(0, 100)
        self.margin_right_spin.setValue(15)
        self.margin_right_spin.valueChanged.connect(self.on_template_modified)
        margins_layout.addWidget(QLabel("Право:"), 1, 2)
        margins_layout.addWidget(self.margin_right_spin, 1, 3)
        
        layout.addRow(margins_layout)
        
        # Table colors
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("Цвета таблицы:"))
        
        colors_layout = QVBoxLayout()
        
        # Border color
        border_layout = QHBoxLayout()
        border_layout.addWidget(QLabel("Границы:"))
        self.border_color_btn = QPushButton("Цвет")
        self.border_color_btn.clicked.connect(self.choose_border_color)
        self.border_color_btn.setStyleSheet("background-color: #CCCCCC;")
        border_layout.addWidget(self.border_color_btn)
        border_layout.addStretch()
        colors_layout.addLayout(border_layout)
        
        # Header background
        header_bg_layout = QHBoxLayout()
        header_bg_layout.addWidget(QLabel("Заголовок таблицы:"))
        self.table_header_bg_btn = QPushButton("Цвет")
        self.table_header_bg_btn.clicked.connect(self.choose_table_header_bg)
        self.table_header_bg_btn.setStyleSheet("background-color: #F5F5F5;")
        header_bg_layout.addWidget(self.table_header_bg_btn)
        header_bg_layout.addStretch()
        colors_layout.addLayout(header_bg_layout)
        
        # Alternate row background
        alt_row_layout = QHBoxLayout()
        alt_row_layout.addWidget(QLabel("Чередующиеся строки:"))
        self.alt_row_bg_btn = QPushButton("Цвет")
        self.alt_row_bg_btn.clicked.connect(self.choose_alt_row_bg)
        self.alt_row_bg_btn.setStyleSheet("background-color: #FAFAFA;")
        alt_row_layout.addWidget(self.alt_row_bg_btn)
        alt_row_layout.addStretch()
        colors_layout.addLayout(alt_row_layout)
        
        layout.addRow(colors_layout)
        
        return tab
    
    def create_preview_panel(self):
        """Create right preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Preview tabs
        self.preview_tabs = QTabWidget()
        
        # Visual preview tab
        preview_tab = self.create_visual_preview_tab()
        self.preview_tabs.addTab(preview_tab, "👁️ Предпросмотр")
        
        # Field formatting tab
        fields_tab = self.create_field_formatting_tab()
        self.preview_tabs.addTab(fields_tab, "🎨 Поля")
        
        layout.addWidget(self.preview_tabs)
        
        return panel
    
    def create_visual_preview_tab(self):
        """Create visual preview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Preview controls
        controls_layout = QHBoxLayout()
        
        self.preview_format_combo = QComboBox()
        self.preview_format_combo.addItems(["HTML Preview", "PDF Preview", "Excel Preview"])
        self.preview_format_combo.currentTextChanged.connect(self.update_preview)
        controls_layout.addWidget(QLabel("Предпросмотр:"))
        controls_layout.addWidget(self.preview_format_combo)
        
        controls_layout.addStretch()
        
        auto_update_cb = QCheckBox("Автообновление")
        auto_update_cb.setChecked(True)
        auto_update_cb.toggled.connect(self.toggle_auto_update)
        controls_layout.addWidget(auto_update_cb)
        
        layout.addLayout(controls_layout)
        
        # Preview area
        self.preview_browser = QTextBrowser()
        self.preview_browser.setMinimumHeight(400)
        layout.addWidget(self.preview_browser)
        
        return tab
    
    def create_field_formatting_tab(self):
        """Create field formatting tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Field list
        fields_group = QGroupBox("🎨 Форматирование полей")
        fields_layout = QVBoxLayout(fields_group)
        
        # Field selector
        field_select_layout = QHBoxLayout()
        field_select_layout.addWidget(QLabel("Поле:"))
        
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self.on_field_selected)
        field_select_layout.addWidget(self.field_combo)
        
        fields_layout.addLayout(field_select_layout)
        
        # Field formatting controls
        format_scroll = QScrollArea()
        format_widget = QWidget()
        format_layout = QFormLayout(format_widget)
        
        # Visibility
        self.field_visible_cb = QCheckBox("Видимое")
        self.field_visible_cb.setChecked(True)
        self.field_visible_cb.toggled.connect(self.on_field_format_changed)
        format_layout.addRow(self.field_visible_cb)
        
        # Order
        self.field_order_spin = QSpinBox()
        self.field_order_spin.setRange(0, 100)
        self.field_order_spin.valueChanged.connect(self.on_field_format_changed)
        format_layout.addRow("Порядок:", self.field_order_spin)
        
        # Width
        self.field_width_spin = QSpinBox()
        self.field_width_spin.setRange(50, 500)
        self.field_width_spin.setValue(100)
        self.field_width_spin.valueChanged.connect(self.on_field_format_changed)
        format_layout.addRow("Ширина (px):", self.field_width_spin)
        
        # Font
        font_layout = QHBoxLayout()
        self.field_font_btn = QPushButton("Шрифт")
        self.field_font_btn.clicked.connect(self.choose_field_font)
        font_layout.addWidget(self.field_font_btn)
        
        self.field_bold_cb = QCheckBox("Жирный")
        self.field_bold_cb.toggled.connect(self.on_field_format_changed)
        font_layout.addWidget(self.field_bold_cb)
        
        self.field_italic_cb = QCheckBox("Курсив")
        self.field_italic_cb.toggled.connect(self.on_field_format_changed)
        font_layout.addWidget(self.field_italic_cb)
        
        format_layout.addRow("Шрифт:", font_layout)
        
        # Colors
        color_layout = QHBoxLayout()
        
        self.field_text_color_btn = QPushButton("Цвет текста")
        self.field_text_color_btn.clicked.connect(self.choose_field_text_color)
        color_layout.addWidget(self.field_text_color_btn)
        
        self.field_bg_color_btn = QPushButton("Цвет фона")
        self.field_bg_color_btn.clicked.connect(self.choose_field_bg_color)
        color_layout.addWidget(self.field_bg_color_btn)
        
        format_layout.addRow("Цвета:", color_layout)
        
        # Alignment
        self.field_alignment_combo = QComboBox()
        self.field_alignment_combo.addItems(["Слева", "По центру", "Справа"])
        self.field_alignment_combo.currentTextChanged.connect(self.on_field_format_changed)
        format_layout.addRow("Выравнивание:", self.field_alignment_combo)
        
        format_scroll.setWidget(format_widget)
        format_scroll.setWidgetResizable(True)
        fields_layout.addWidget(format_scroll)
        
        layout.addWidget(fields_group)
        
        return tab
    
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
        apply_btn = QPushButton("✅ Применить шаблон")
        apply_btn.clicked.connect(self.apply_template)
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
    
    def apply_styling(self):
        """Apply custom styling"""
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
            QToolBar {
                border: 1px solid #CCCCCC;
                padding: 5px;
            }
            QTextBrowser {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
        """)
    
    def ensure_templates_directory(self):
        """Ensure templates directory exists"""
        os.makedirs(self.templates_directory, exist_ok=True)
    
    def load_builtin_templates(self):
        """Load built-in templates into combo box"""
        self.template_combo.clear()
        
        # Add built-in templates
        builtin_templates = [
            "Стандартный",
            "Подробный отчёт", 
            "Сводка",
            "Счёт-фактура",
            "Финансовый отчёт"
        ]
        
        for template in builtin_templates:
            self.template_combo.addItem(template)
        
        # Add custom templates from directory
        try:
            for file_name in os.listdir(self.templates_directory):
                if file_name.endswith('.json'):
                    template_name = os.path.splitext(file_name)[0]
                    self.template_combo.addItem(f"📄 {template_name}")
        except OSError:
            pass
    
    def load_field_formats(self):
        """Load available fields into field combo"""
        self.field_combo.clear()
        
        # Get fields from settings
        table_fields = settings_manager.get_table_fields()
        
        for field in table_fields:
            if field.get("visible", True):
                field_id = field.get("id", "")
                field_name = field.get("name", field_id)
                self.field_combo.addItem(field_name, field_id)
        
        # Initialize field formats for current template
        if not self.current_template.field_formats:
            self.current_template.field_formats = {}
            
            for field in table_fields:
                field_id = field.get("id", "")
                if field_id and field.get("visible", True):
                    self.current_template.field_formats[field_id] = FieldFormat()
    
    def update_preview(self):
        """Update the visual preview"""
        try:
            preview_format = self.preview_format_combo.currentText()
            
            if "HTML" in preview_format:
                html_content = self.generate_html_preview()
                self.preview_browser.setHtml(html_content)
            elif "PDF" in preview_format:
                # For PDF preview, show HTML representation
                html_content = self.generate_pdf_preview_html()
                self.preview_browser.setHtml(html_content)
            elif "Excel" in preview_format:
                html_content = self.generate_excel_preview_html()
                self.preview_browser.setHtml(html_content)
                
        except Exception as e:
            error_html = f"""
            <html><body>
            <h3 style="color: red;">Ошибка предварительного просмотра</h3>
            <p>{str(e)}</p>
            </body></html>
            """
            self.preview_browser.setHtml(error_html)
    
    def generate_html_preview(self):
        """Generate HTML preview content"""
        # Collect current template settings
        self.update_template_from_ui()
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Export Preview</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: white;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .company-info {
            text-align: center;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid %(border_color)s;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: %(header_bg)s;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: %(alt_row_bg)s;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #666;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 12px;
        }
    </style>
</head>
<body>""" % {
            'border_color': self.current_template.table_border_color,
            'header_bg': self.current_template.table_header_bg,
            'alt_row_bg': self.current_template.table_alt_row_bg
        }
        
        # Add header
        if self.current_template.include_header:
            html += f"""
    <div class="header">
        <h1 style="color: {self.current_template.header_color}; 
                   font-family: {self.current_template.header_font}; 
                   font-size: {self.current_template.header_size}px;">
            {self.current_template.header_text}
        </h1>
    </div>"""
        
        # Add company info
        if self.current_template.company_name:
            html += f"""
    <div class="company-info">
        <h2>{self.current_template.company_name}</h2>"""
            if self.current_template.company_address:
                html += f"<p>{self.current_template.company_address.replace(chr(10), '<br>')}</p>"
            html += "</div>"
        
        # Add metadata
        if self.current_template.include_metadata:
            html += f"""
    <div class="metadata">
        <strong>Дата создания:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M')}<br>
        <strong>Создано:</strong> InvoiceGemini v{app_config.APP_VERSION}<br>
        <strong>Шаблон:</strong> {self.current_template.name}
    </div>"""
        
        # Add data table
        html += self.generate_data_table_html()
        
        # Add summary
        if self.current_template.include_summary:
            html += self.generate_summary_html()
        
        # Add footer
        if self.current_template.include_footer:
            html += f"""
    <div class="footer">
        <p style="color: {self.current_template.footer_color}; 
                  font-family: {self.current_template.footer_font}; 
                  font-size: {self.current_template.footer_size}px;">
            {self.current_template.footer_text}
        </p>
    </div>"""
        
        html += """
</body>
</html>"""
        
        return html
    
    def generate_data_table_html(self):
        """Generate HTML for data table"""
        if not self.current_results:
            return "<p><em>Нет данных для предварительного просмотра</em></p>"
        
        # Get visible fields in order
        visible_fields = []
        field_formats = self.current_template.field_formats or {}
        
        # Get fields from settings
        table_fields = settings_manager.get_table_fields()
        field_info = {f.get("id", ""): f for f in table_fields}
        
        for field_id, field_format in field_formats.items():
            if field_format.visible and field_id in field_info:
                visible_fields.append((field_id, field_format, field_info[field_id]))
        
        # Sort by order
        visible_fields.sort(key=lambda x: x[1].order)
        
        if not visible_fields:
            return "<p><em>Нет видимых полей для отображения</em></p>"
        
        # Generate table HTML
        html = "<table>\n<thead>\n<tr>"
        
        for field_id, field_format, field_info in visible_fields:
            field_name = field_info.get("name", field_id)
            html += f'<th style="width: {field_format.width}px;">{field_name}</th>'
        
        html += "</tr>\n</thead>\n<tbody>"
        
        # Handle both single result and batch results
        if isinstance(self.current_results, dict):
            if "batch_results" in self.current_results:
                # Batch results
                for result in self.current_results["batch_results"]:
                    html += "\n<tr>"
                    for field_id, field_format, field_info in visible_fields:
                        value = result.get(field_info.get("name", field_id), "")
                        cell_style = self.get_cell_style(field_format)
                        html += f'<td style="{cell_style}">{value}</td>'
                    html += "</tr>"
            else:
                # Single result
                html += "\n<tr>"
                for field_id, field_format, field_info in visible_fields:
                    value = self.current_results.get(field_id, "")
                    cell_style = self.get_cell_style(field_format)
                    html += f'<td style="{cell_style}">{value}</td>'
                html += "</tr>"
        
        html += "\n</tbody>\n</table>"
        
        return html
    
    def get_cell_style(self, field_format: FieldFormat) -> str:
        """Get CSS style for table cell"""
        style_parts = []
        
        if field_format.font_family:
            style_parts.append(f"font-family: {field_format.font_family}")
        
        if field_format.font_size:
            style_parts.append(f"font-size: {field_format.font_size}px")
        
        if field_format.font_bold:
            style_parts.append("font-weight: bold")
        
        if field_format.font_italic:
            style_parts.append("font-style: italic")
        
        if field_format.text_color:
            style_parts.append(f"color: {field_format.text_color}")
        
        if field_format.background_color and field_format.background_color != "#FFFFFF":
            style_parts.append(f"background-color: {field_format.background_color}")
        
        alignment_map = {"left": "left", "center": "center", "right": "right"}
        if field_format.alignment in alignment_map:
            style_parts.append(f"text-align: {alignment_map[field_format.alignment]}")
        
        return "; ".join(style_parts)
    
    def generate_summary_html(self):
        """Generate summary section HTML"""
        html = """
    <div class="summary">
        <h3>Сводная информация</h3>
        <ul>"""
        
        if isinstance(self.current_results, dict) and "batch_results" in self.current_results:
            batch_results = self.current_results["batch_results"]
            html += f"<li>Всего обработано файлов: {len(batch_results)}</li>"
            
            # Count successful/error results
            successful = sum(1 for r in batch_results if not r.get("Ошибка", ""))
            errors = len(batch_results) - successful
            
            html += f"<li>Успешно обработано: {successful}</li>"
            if errors > 0:
                html += f"<li>С ошибками: {errors}</li>"
        else:
            html += "<li>Обработан 1 файл</li>"
        
        html += f"<li>Дата создания отчёта: {datetime.now().strftime('%d.%m.%Y %H:%M')}</li>"
        html += """
        </ul>
    </div>"""
        
        return html
    
    def generate_pdf_preview_html(self):
        """Generate PDF-style preview HTML"""
        html = self.generate_html_preview()
        
        # Add PDF-specific styling
        pdf_style = """
        <style>
            @page {
                size: """ + self.current_template.page_size + """ """ + self.current_template.page_orientation + """;
                margin: """ + str(self.current_template.margin_top) + """mm """ + str(self.current_template.margin_right) + """mm """ + str(self.current_template.margin_bottom) + """mm """ + str(self.current_template.margin_left) + """mm;
            }
            body {
                max-width: 210mm;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 20px;
            }
        </style>
        """
        
        # Insert PDF style after existing styles
        html = html.replace("</head>", pdf_style + "</head>")
        
        return html
    
    def generate_excel_preview_html(self):
        """Generate Excel-style preview HTML"""
        html = self.generate_html_preview()
        
        # Add Excel-specific styling
        excel_style = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f0f0;
                padding: 20px;
            }
            table {
                background-color: white;
                border: 2px solid #d0d0d0;
            }
            th {
                background-color: #e6e6e6;
                border: 1px solid #c0c0c0;
                font-weight: bold;
            }
            td {
                border: 1px solid #d0d0d0;
            }
        </style>
        """
        
        # Insert Excel style after existing styles
        html = html.replace("</head>", excel_style + "</head>")
        
        return html
    
    # Event handlers
    def on_template_changed(self, template_name):
        """Handle template selection change"""
        if template_name.startswith("📄"):
            # Custom template
            custom_name = template_name[2:].strip()
            self.load_custom_template(custom_name)
        else:
            # Built-in template
            self.load_builtin_template(template_name)
        
        self.update_ui_from_template()
        self.schedule_preview_update()
    
    def on_template_modified(self):
        """Handle template modification"""
        self.is_modified = True
        self.schedule_preview_update()
    
    def on_format_changed(self):
        """Handle format change"""
        self.on_template_modified()
        
        # Update UI based on format
        format_text = self.format_combo.currentText()
        is_pdf = "PDF" in format_text
        
        # Enable/disable page settings based on format
        self.settings_tabs.setTabEnabled(3, is_pdf)  # Page settings tab
    
    def on_field_selected(self, field_name):
        """Handle field selection for formatting"""
        field_id = self.field_combo.currentData()
        if not field_id:
            return
        
        # Load field format into UI
        field_format = self.current_template.field_formats.get(field_id, FieldFormat())
        
        self.field_visible_cb.setChecked(field_format.visible)
        self.field_order_spin.setValue(field_format.order)
        self.field_width_spin.setValue(field_format.width)
        self.field_bold_cb.setChecked(field_format.font_bold)
        self.field_italic_cb.setChecked(field_format.font_italic)
        
        # Update color buttons
        self.field_text_color_btn.setStyleSheet(f"background-color: {field_format.text_color}; color: white;")
        self.field_bg_color_btn.setStyleSheet(f"background-color: {field_format.background_color};")
        
        # Set alignment
        alignment_map = {"left": 0, "center": 1, "right": 2}
        self.field_alignment_combo.setCurrentIndex(alignment_map.get(field_format.alignment, 0))
    
    def on_field_format_changed(self):
        """Handle field format change"""
        field_id = self.field_combo.currentData()
        if not field_id:
            return
        
        # Create or update field format
        if field_id not in self.current_template.field_formats:
            self.current_template.field_formats[field_id] = FieldFormat()
        
        field_format = self.current_template.field_formats[field_id]
        
        # Update field format from UI
        field_format.visible = self.field_visible_cb.isChecked()
        field_format.order = self.field_order_spin.value()
        field_format.width = self.field_width_spin.value()
        field_format.font_bold = self.field_bold_cb.isChecked()
        field_format.font_italic = self.field_italic_cb.isChecked()
        
        alignment_map = {0: "left", 1: "center", 2: "right"}
        field_format.alignment = alignment_map.get(self.field_alignment_combo.currentIndex(), "left")
        
        self.on_template_modified()
    
    def schedule_preview_update(self):
        """Schedule preview update with delay"""
        self.preview_timer.start(500)  # 500ms delay
    
    def toggle_auto_update(self, enabled):
        """Toggle auto-update of preview"""
        if not enabled:
            self.preview_timer.stop()
    
    # Color/font choosers
    def choose_header_color(self):
        """Choose header color"""
        color = QColorDialog.getColor(QColor(self.current_template.header_color), self)
        if color.isValid():
            self.current_template.header_color = color.name()
            self.header_color_btn.setStyleSheet(f"background-color: {color.name()}; color: white;")
            self.on_template_modified()
    
    def choose_footer_color(self):
        """Choose footer color"""
        color = QColorDialog.getColor(QColor(self.current_template.footer_color), self)
        if color.isValid():
            self.current_template.footer_color = color.name()
            self.footer_color_btn.setStyleSheet(f"background-color: {color.name()}; color: white;")
            self.on_template_modified()
    
    def choose_header_font(self):
        """Choose header font"""
        font, ok = QFontDialog.getFont()
        if ok:
            self.current_template.header_font = font.family()
            self.current_template.header_size = font.pointSize()
            self.current_template.header_bold = font.bold()
            self.on_template_modified()
    
    def choose_footer_font(self):
        """Choose footer font"""
        font, ok = QFontDialog.getFont()
        if ok:
            self.current_template.footer_font = font.family()
            self.current_template.footer_size = font.pointSize()
            self.on_template_modified()
    
    def choose_field_font(self):
        """Choose field font"""
        field_id = self.field_combo.currentData()
        if not field_id:
            return
        
        font, ok = QFontDialog.getFont()
        if ok:
            if field_id not in self.current_template.field_formats:
                self.current_template.field_formats[field_id] = FieldFormat()
            
            field_format = self.current_template.field_formats[field_id]
            field_format.font_family = font.family()
            field_format.font_size = font.pointSize()
            self.on_template_modified()
    
    def choose_field_text_color(self):
        """Choose field text color"""
        field_id = self.field_combo.currentData()
        if not field_id:
            return
        
        if field_id not in self.current_template.field_formats:
            self.current_template.field_formats[field_id] = FieldFormat()
        
        field_format = self.current_template.field_formats[field_id]
        color = QColorDialog.getColor(QColor(field_format.text_color), self)
        if color.isValid():
            field_format.text_color = color.name()
            self.field_text_color_btn.setStyleSheet(f"background-color: {color.name()}; color: white;")
            self.on_template_modified()
    
    def choose_field_bg_color(self):
        """Choose field background color"""
        field_id = self.field_combo.currentData()
        if not field_id:
            return
        
        if field_id not in self.current_template.field_formats:
            self.current_template.field_formats[field_id] = FieldFormat()
        
        field_format = self.current_template.field_formats[field_id]
        color = QColorDialog.getColor(QColor(field_format.background_color), self)
        if color.isValid():
            field_format.background_color = color.name()
            self.field_bg_color_btn.setStyleSheet(f"background-color: {color.name()};")
            self.on_template_modified()
    
    def choose_border_color(self):
        """Choose table border color"""
        color = QColorDialog.getColor(QColor(self.current_template.table_border_color), self)
        if color.isValid():
            self.current_template.table_border_color = color.name()
            self.border_color_btn.setStyleSheet(f"background-color: {color.name()};")
            self.on_template_modified()
    
    def choose_table_header_bg(self):
        """Choose table header background color"""
        color = QColorDialog.getColor(QColor(self.current_template.table_header_bg), self)
        if color.isValid():
            self.current_template.table_header_bg = color.name()
            self.table_header_bg_btn.setStyleSheet(f"background-color: {color.name()};")
            self.on_template_modified()
    
    def choose_alt_row_bg(self):
        """Choose alternate row background color"""
        color = QColorDialog.getColor(QColor(self.current_template.table_alt_row_bg), self)
        if color.isValid():
            self.current_template.table_alt_row_bg = color.name()
            self.alt_row_bg_btn.setStyleSheet(f"background-color: {color.name()};")
            self.on_template_modified()
    
    def choose_logo(self):
        """Choose logo file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать логотип",
            "",
            "Изображения (*.png *.jpg *.jpeg *.svg *.bmp);;Все файлы (*.*)"
        )
        
        if file_path:
            self.current_template.logo_path = file_path
            self.logo_path_edit.setText(file_path)
            self.on_template_modified()
    
    # Template management
    def new_template(self):
        """Create new template"""
        self.current_template = ExportTemplate()
        self.current_template.name = "Новый шаблон"
        self.update_ui_from_template()
        self.is_modified = True
    
    def load_template(self):
        """Load template from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить шаблон",
            self.templates_directory,
            "Шаблоны (*.json);;Все файлы (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                self.load_template_from_dict(template_data)
                self.update_ui_from_template()
                self.is_modified = False
                
                QMessageBox.information(self, "Успешно", f"Шаблон загружен из {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить шаблон:\n{str(e)}")
    
    def save_template(self):
        """Save current template"""
        if not self.current_template.name or self.current_template.name == "Новый шаблон":
            name, ok = QMessageBox.getText(self, "Сохранение шаблона", "Введите название шаблона:")
            if not ok or not name.strip():
                return
            self.current_template.name = name.strip()
        
        # Update template from UI
        self.update_template_from_ui()
        
        # Save to file
        file_name = f"{self.current_template.name}.json"
        file_path = os.path.join(self.templates_directory, file_name)
        
        try:
            template_dict = asdict(self.current_template)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, ensure_ascii=False, indent=2)
            
            self.is_modified = False
            self.template_saved.emit(self.current_template.name)
            
            QMessageBox.information(self, "Успешно", f"Шаблон сохранён как {file_name}")
            
            # Refresh template list
            self.load_builtin_templates()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить шаблон:\n{str(e)}")
    
    def load_builtin_template(self, template_name):
        """Load built-in template by name"""
        # Reset to default
        self.current_template = ExportTemplate()
        
        # Customize based on template name
        if template_name == "Подробный отчёт":
            self.current_template.name = template_name
            self.current_template.include_summary = True
            self.current_template.header_text = "Подробный отчёт по обработке счетов-фактур"
            self.current_template.header_size = 18
        elif template_name == "Сводка":
            self.current_template.name = template_name
            self.current_template.include_summary = True
            self.current_template.include_metadata = False
            self.current_template.header_text = "Сводный отчёт"
        elif template_name == "Счёт-фактура":
            self.current_template.name = template_name
            self.current_template.template_type = "invoice"
            self.current_template.header_text = "Счёт-фактура"
            self.current_template.include_logo = True
        elif template_name == "Финансовый отчёт":
            self.current_template.name = template_name
            self.current_template.header_text = "Финансовый отчёт"
            self.current_template.table_header_bg = "#E3F2FD"
            self.current_template.header_color = "#1976D2"
        else:
            # Standard template
            self.current_template.name = template_name
    
    def load_custom_template(self, template_name):
        """Load custom template from file"""
        file_path = os.path.join(self.templates_directory, f"{template_name}.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            self.load_template_from_dict(template_data)
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить шаблон {template_name}:\n{str(e)}")
            self.current_template = ExportTemplate()
    
    def load_template_from_dict(self, template_data):
        """Load template from dictionary data"""
        # Create new template instance
        self.current_template = ExportTemplate()
        
        # Set basic properties
        for key, value in template_data.items():
            if key == "field_formats":
                # Handle field formats specially
                self.current_template.field_formats = {}
                for field_id, format_data in value.items():
                    field_format = FieldFormat()
                    for fmt_key, fmt_value in format_data.items():
                        if hasattr(field_format, fmt_key):
                            setattr(field_format, fmt_key, fmt_value)
                    self.current_template.field_formats[field_id] = field_format
            elif hasattr(self.current_template, key):
                setattr(self.current_template, key, value)
    
    def update_template_from_ui(self):
        """Update template object from UI controls"""
        # Basic settings
        self.current_template.name = self.template_name_edit.text()
        self.current_template.description = self.template_desc_edit.toPlainText()
        
        type_map = {"Таблица": "table", "Отчёт": "report", "Счёт-фактура": "invoice"}
        self.current_template.template_type = type_map.get(self.template_type_combo.currentText(), "table")
        
        format_map = {"Excel (.xlsx)": "excel", "PDF (.pdf)": "pdf", "HTML (.html)": "html", "CSV (.csv)": "csv"}
        self.current_template.format = format_map.get(self.format_combo.currentText(), "excel")
        
        self.current_template.include_metadata = self.include_metadata_cb.isChecked()
        self.current_template.include_timestamps = self.include_timestamps_cb.isChecked()
        self.current_template.include_summary = self.include_summary_cb.isChecked()
        
        # Header/Footer
        self.current_template.include_header = self.include_header_cb.isChecked()
        self.current_template.header_text = self.header_text_edit.text()
        self.current_template.include_footer = self.include_footer_cb.isChecked()
        self.current_template.footer_text = self.footer_text_edit.text()
        
        # Branding
        self.current_template.include_logo = self.include_logo_cb.isChecked()
        self.current_template.logo_path = self.logo_path_edit.text()
        self.current_template.company_name = self.company_name_edit.text()
        self.current_template.company_address = self.company_address_edit.toPlainText()
        
        # Page settings
        orientation_map = {"Книжная": "portrait", "Альбомная": "landscape"}
        self.current_template.page_orientation = orientation_map.get(self.orientation_combo.currentText(), "portrait")
        self.current_template.page_size = self.page_size_combo.currentText()
        
        self.current_template.margin_top = self.margin_top_spin.value()
        self.current_template.margin_bottom = self.margin_bottom_spin.value()
        self.current_template.margin_left = self.margin_left_spin.value()
        self.current_template.margin_right = self.margin_right_spin.value()
    
    def update_ui_from_template(self):
        """Update UI controls from template object"""
        # Block signals to prevent recursion
        self.template_name_edit.blockSignals(True)
        self.template_desc_edit.blockSignals(True)
        
        # Basic settings
        self.template_name_edit.setText(self.current_template.name)
        self.template_desc_edit.setPlainText(self.current_template.description)
        
        type_map = {"table": "Таблица", "report": "Отчёт", "invoice": "Счёт-фактура"}
        type_text = type_map.get(self.current_template.template_type, "Таблица")
        self.template_type_combo.setCurrentText(type_text)
        
        format_map = {"excel": "Excel (.xlsx)", "pdf": "PDF (.pdf)", "html": "HTML (.html)", "csv": "CSV (.csv)"}
        format_text = format_map.get(self.current_template.format, "Excel (.xlsx)")
        self.format_combo.setCurrentText(format_text)
        
        self.include_metadata_cb.setChecked(self.current_template.include_metadata)
        self.include_timestamps_cb.setChecked(self.current_template.include_timestamps)
        self.include_summary_cb.setChecked(self.current_template.include_summary)
        
        # Header/Footer
        self.include_header_cb.setChecked(self.current_template.include_header)
        self.header_text_edit.setText(self.current_template.header_text)
        self.include_footer_cb.setChecked(self.current_template.include_footer)
        self.footer_text_edit.setText(self.current_template.footer_text)
        
        # Update color buttons
        self.header_color_btn.setStyleSheet(f"background-color: {self.current_template.header_color}; color: white;")
        self.footer_color_btn.setStyleSheet(f"background-color: {self.current_template.footer_color}; color: white;")
        self.border_color_btn.setStyleSheet(f"background-color: {self.current_template.table_border_color};")
        self.table_header_bg_btn.setStyleSheet(f"background-color: {self.current_template.table_header_bg};")
        self.alt_row_bg_btn.setStyleSheet(f"background-color: {self.current_template.table_alt_row_bg};")
        
        # Branding
        self.include_logo_cb.setChecked(self.current_template.include_logo)
        self.logo_path_edit.setText(self.current_template.logo_path)
        self.company_name_edit.setText(self.current_template.company_name)
        self.company_address_edit.setPlainText(self.current_template.company_address)
        
        # Page settings
        orientation_map = {"portrait": "Книжная", "landscape": "Альбомная"}
        orientation_text = orientation_map.get(self.current_template.page_orientation, "Книжная")
        self.orientation_combo.setCurrentText(orientation_text)
        self.page_size_combo.setCurrentText(self.current_template.page_size)
        
        self.margin_top_spin.setValue(self.current_template.margin_top)
        self.margin_bottom_spin.setValue(self.current_template.margin_bottom)
        self.margin_left_spin.setValue(self.current_template.margin_left)
        self.margin_right_spin.setValue(self.current_template.margin_right)
        
        # Re-enable signals
        self.template_name_edit.blockSignals(False)
        self.template_desc_edit.blockSignals(False)
        
        # Load field formats
        self.load_field_formats()
        
        # Update preview
        self.schedule_preview_update()
    
    def test_export(self):
        """Test export with current template"""
        if not self.current_results:
            QMessageBox.information(self, "Тестовый экспорт", "Нет данных для экспорта")
            return
        
        # Update template from UI
        self.update_template_from_ui()
        
        # Get export format
        format_map = {"Excel (.xlsx)": "xlsx", "PDF (.pdf)": "pdf", "HTML (.html)": "html", "CSV (.csv)": "csv"}
        export_format = format_map.get(self.format_combo.currentText(), "xlsx")
        
        # Choose save location
        default_name = f"test_export_{self.current_template.name}.{export_format}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить тестовый экспорт",
            default_name,
            f"Файлы {export_format.upper()} (*.{export_format});;Все файлы (*.*)"
        )
        
        if file_path:
            try:
                # Export using template
                success = self.export_with_template(file_path, export_format)
                
                if success:
                    QMessageBox.information(
                        self, 
                        "Тестовый экспорт", 
                        f"Тестовый экспорт успешно сохранён:\n{file_path}"
                    )
                else:
                    QMessageBox.warning(self, "Ошибка", "Не удалось выполнить тестовый экспорт")
                    
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка тестового экспорта:\n{str(e)}")
    
    def export_with_template(self, file_path: str, export_format: str) -> bool:
        """Export data using current template"""
        try:
            if export_format == "html":
                # HTML export
                html_content = self.generate_html_preview()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return True
            
            elif export_format == "pdf":
                # PDF export (requires additional implementation)
                # For now, save as HTML with PDF styling
                html_content = self.generate_pdf_preview_html()
                html_path = file_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                QMessageBox.information(
                    self, 
                    "PDF экспорт", 
                    f"PDF экспорт сохранён как HTML:\n{html_path}\n\nДля полной поддержки PDF требуется дополнительная библиотека."
                )
                return True
            
            elif export_format in ["xlsx", "csv"]:
                # Use existing export functionality from utils
                # This would need to be enhanced to support template formatting
                from .. import utils
                
                success, message = utils.export_results(self.current_results, file_path, export_format)
                return success
            
            return False
            
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def apply_template(self):
        """Apply current template"""
        self.update_template_from_ui()
        self.template_applied.emit(asdict(self.current_template))
        QMessageBox.information(self, "Шаблон применён", f"Шаблон '{self.current_template.name}' применён")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>📖 Справка по дизайнеру шаблонов</h2>
        
        <h3>🎨 Создание шаблонов</h3>
        <p>• Выберите базовый шаблон или создайте новый<br>
        • Настройте заголовок, подвал и брендинг<br>
        • Отформатируйте отдельные поля данных<br>
        • Просматривайте результат в реальном времени</p>
        
        <h3>📄 Типы шаблонов</h3>
        <p>• <b>Таблица</b>: Простая табличная форма<br>
        • <b>Отчёт</b>: Подробный отчёт с сводкой<br>
        • <b>Счёт-фактура</b>: Формат счёта-фактуры</p>
        
        <h3>🎯 Форматы экспорта</h3>
        <p>• <b>Excel</b>: Полное форматирование и стили<br>
        • <b>PDF</b>: Профессиональные документы<br>
        • <b>HTML</b>: Веб-совместимый формат<br>
        • <b>CSV</b>: Простые данные для анализа</p>
        
        <h3>🏢 Брендинг</h3>
        <p>• Добавьте логотип компании<br>
        • Укажите название и адрес<br>
        • Настройте корпоративные цвета</p>
        
        <h3>⚙️ Советы</h3>
        <p>• Используйте автообновление предпросмотра<br>
        • Сохраняйте шаблоны для повторного использования<br>
        • Тестируйте экспорт перед применением<br>
        • Настраивайте поля по важности</p>
        """
        
        QMessageBox.information(self, "Справка", help_text)
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                "Несохранённые изменения",
                "У вас есть несохранённые изменения в шаблоне. Сохранить перед закрытием?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_template()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept() 
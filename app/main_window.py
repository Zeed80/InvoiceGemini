"""
Главное окно приложения для извлечения данных из счетов-фактур.
"""
import os
import sys
import json
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QRadioButton, QPushButton, QComboBox, QTextEdit, QScrollArea,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy, QApplication,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QProgressBar,
    QStatusBar, QSpacerItem, QDialog,
    QTabWidget, QButtonGroup  # ИСПРАВЛЕНИЕ: Добавляем QButtonGroup
)

# ФАЗА 2: Импорт оптимизированных UI компонентов
from .ui.performance_optimized_widgets import (
    OptimizedTableWidget, SmartProgressBar, VirtualScrollArea,
    AnimatedButton, OptimizedFileListWidget
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QUrl, QTimer, QThread
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QFont
from PyQt6.QtCore import QTranslator
from PIL import Image, ImageQt
import pdf2image
import tempfile

from . import config as app_config
from . import utils
from .settings_dialog import ModelManagementDialog
from .threads import ProcessingThread
from .settings_manager import settings_manager
from .processing_engine import ModelManager
from .training_dialog import TrainingDialog

# NEW: Import optimized storage system for Phase 3
from .core.storage_integration import get_storage_integration, initialize_optimized_storage

# NEW: Import LLM Plugin Manager
from .plugins.plugin_manager import PluginManager
from .ui.preview_dialog import PreviewDialog
from .ui.export_template_designer import ExportTemplateDesigner
from .ui.field_manager_dialog import FieldManagerDialog
from .field_manager import field_manager
from .ui.llm_providers_dialog import LLMProvidersDialog
from .prompt_generator import PromptGenerator

# NEW: Import new components for Phase 1 improvements
from .core.cache_manager import CacheManager
from .core.retry_manager import RetryManager
from .core.backup_manager import BackupManager
from .ui.components.file_selector import FileSelector
from .ui.components.file_list_widget import FileListWidget, ProcessingStatus, FileProcessingInfo
from .ui.components.file_viewer_dialog import FileViewerDialog
# from .ui.components.progress_indicator import ProgressIndicator  # Не используется
from .ui.components.batch_processor_adapter import BatchProcessor
from .ui.components.export_manager import ExportManager

logger = logging.getLogger(__name__)


class CollapsibleGroup(QWidget):
    """Сворачиваемая группа виджетов для экономии места."""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.expanded = True
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(2)
        
        # Header with toggle button
        self.header_frame = QFrame()
        self.header_frame.setFrameStyle(QFrame.Shape.Box)
        self.header_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #ccc; }")
        
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        # Toggle button (arrow)
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-radius: 10px;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_expanded)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        
        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 4, 8, 4)
        self.content_layout.setSpacing(4)
        
        # Add to main layout
        self.main_layout.addWidget(self.header_frame)
        self.main_layout.addWidget(self.content_widget)
        
    def toggle_expanded(self):
        """Переключает состояние развернутости."""
        self.expanded = not self.expanded
        self.content_widget.setVisible(self.expanded)
        self.toggle_button.setText("▼" if self.expanded else "►")
        
    def add_widget(self, widget):
        """Добавляет виджет в контент."""
        self.content_layout.addWidget(widget)
        
    def add_layout(self, layout):
        """Добавляет layout в контент."""
        self.content_layout.addLayout(layout)
        
    def set_expanded(self, expanded):
        """Устанавливает состояние развернутости."""
        if self.expanded != expanded:
            self.toggle_expanded()


class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        self._translator = QTranslator(self)
        self._current_language = None
        # Устанавливаем язык из настроек до построения UI
        try:
            lang_code = settings_manager.get_string('Interface', 'language', 'ru')
            self._install_translator(lang_code)
        except Exception as e:
            logger.warning(f"Localization init failed: {e}")
        self.init_ui()
        self.current_image_path = None
        self.current_folder_path = None # NEW: Добавлено для хранения пути к папке
        self.current_invoice_data = None  # Данные текущего обработанного документа для Paperless
        self.current_file_path = None  # Путь к текущему обработанному файлу для Paperless
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processing_thread = None
        
        # NEW: Добавляем ссылку на диалог управления моделями
        self.model_management_dialog = None
        
        # Заполняем список моделей Gemini при инициализации
        self.populate_gemini_models()
        
        # NEW: Создаем единый экземпляр ModelManager
        self.model_manager = ModelManager()
        
        # NEW: Initialize LLM Plugin Manager
        # Инициализируем универсальный менеджер плагинов
        from app.plugins.universal_plugin_manager import UniversalPluginManager
        from app.plugins.llm_plugin_adapter import adapt_all_llm_plugins
        from app.plugins.base_plugin import PluginType
        
        self.universal_plugin_manager = UniversalPluginManager()
        
        # Для обратной совместимости также инициализируем старый менеджер
        self.plugin_manager = PluginManager()
        self.current_llm_plugin = None
        
        # Создаем адаптеры для существующих LLM плагинов
        self.llm_adapters = adapt_all_llm_plugins(self.plugin_manager)
        self.llm_loading_thread = None
        
        # NEW: Initialize new core components
        self.cache_manager = CacheManager()
        self.retry_manager = RetryManager()
        self.backup_manager = BackupManager(app_data_dir=app_config.APP_DATA_PATH)
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(settings_manager)
        
        # NEW: Initialize optimized storage system (Phase 3)
        self.storage_integration = get_storage_integration(settings_manager)
        if self.storage_integration:
            # Подключаем сигналы миграции
            self.storage_integration.migration_started.connect(
                lambda: self.set_processing_status("Миграция настроек...")
            )
            self.storage_integration.migration_progress.connect(
                lambda progress, status: self.set_processing_status(f"{status} ({progress}%)")
            )
            self.storage_integration.migration_completed.connect(
                self._on_storage_migration_completed
            )
            self.storage_integration.storage_ready.connect(
                lambda: self.set_processing_status("Система хранения готова")
            )
            
            # Запускаем инициализацию в фоне
            QTimer.singleShot(500, lambda: self.storage_integration.initialize_storage_system())
        
        # NEW: Initialize UI components
        self.file_selector = None  # Will be initialized in init_ui
        # self.progress_indicator = None  # Не используется, заменен на progress_bar
        self.batch_processor = None  # Will be initialized after UI
        self.export_manager = None  # Will be initialized after UI
        
        # Store batch processing results
        self.batch_results = []
        self.current_batch_index = 0
        
        # Populate LLM models after UI initialization
        QTimer.singleShot(100, self.populate_llm_models)
        
        self.model_selector = QComboBox()
        
        # Populate Gemini models
        self.populate_gemini_models()
        
        # Populate TrOCR models
        self.populate_trocr_models()
        
        # Populate LLM providers and models
        self.populate_cloud_providers()
        self.populate_local_providers()
        
        # NEW: Initialize components that need UI and model_manager to be ready
        # Отложенная инициализация после полной настройки UI
        QTimer.singleShot(200, self._init_post_ui_components)
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        self.setWindowTitle(f"{app_config.APP_NAME} v{app_config.APP_VERSION}")
        self.setMinimumSize(1024, 768)
        
        # Создаем центральный виджет и главную компоновку
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем сплиттер для разделения экрана
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая часть - для списка файлов
        self.files_widget = QWidget()
        files_layout = QVBoxLayout(self.files_widget)
        
        # NEW: Use FileSelector component for selection
        self.file_selector = FileSelector()
        self.file_selector.signals.file_selected.connect(self.on_file_selected)
        self.file_selector.signals.folder_selected.connect(self.on_folder_selected)
        
        # For backward compatibility
        self.select_file_button = self.file_selector.select_file_button
        self.select_folder_button = self.file_selector.select_folder_button
        self.selected_path_label = self.file_selector.selection_label
        
        # NEW: File list widget with processing indicators
        self.file_list_widget = FileListWidget()
        self.file_list_widget.file_selected.connect(self.on_file_list_selection_changed)
        self.file_list_widget.process_file_requested.connect(self.on_process_single_file_requested)
        self.file_list_widget.process_all_requested.connect(self.on_process_all_files_requested)
        self.file_list_widget.filename_clicked.connect(self.on_filename_clicked)
        
        # Добавляем виджеты в левую часть
        files_layout.addWidget(self.file_selector)
        files_layout.addWidget(self.file_list_widget, 1)  # Растягивать при изменении размера окна
        
        # Keep image display for backward compatibility (hidden by default)
        self.image_widget = QWidget()
        self.image_widget.setVisible(False)  # Скрываем по умолчанию
        image_layout = QVBoxLayout(self.image_widget)
        
        # Область отображения изображения
        image_group = QGroupBox("Изображение")
        scroll_layout = QVBoxLayout()
        
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        
        scroll_layout.addWidget(self.image_scroll)
        image_group.setLayout(scroll_layout)
        image_layout.addWidget(image_group, 1)
        
        # Правая часть - для выбора модели и результатов с поддержкой прокрутки
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        
        # Устанавливаем содержимое в scroll area
        self.controls_scroll.setWidget(self.controls_widget)
        
        # Компактная секция выбора модели
        model_group = QGroupBox("Выбор модели")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(4)  # Уменьшаем отступы
        
        # ИСПРАВЛЕНИЕ: Создаем группу для RadioButton'ов ПЕРЕД их использованием
        self.model_button_group = QButtonGroup()
        
        # === ОБЛАЧНЫЕ МОДЕЛИ - СВОРАЧИВАЕМАЯ ГРУППА ===
        self.cloud_models_group = CollapsibleGroup("☁️ Облачные модели")
        self.cloud_models_group.set_expanded(False)  # ИЗМЕНЕНИЕ: По умолчанию свернута, будет развернута при загрузке настроек
        
        # Gemini - компактная версия с индикатором состояния
        gemini_layout = QHBoxLayout()
        gemini_layout.setSpacing(4)
        self.gemini_radio = QRadioButton("Google Gemini")
        self.gemini_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.gemini_radio)
        
        # ИСПРАВЛЕНИЕ: Добавляем индикатор состояния API ключа
        self.gemini_status_indicator = QLabel("❓")
        self.gemini_status_indicator.setFixedSize(16, 16)
        self.gemini_status_indicator.setToolTip("Статус API ключа")
        
        self.gemini_prompt_button = QPushButton("📝")
        self.gemini_prompt_button.setFixedSize(24, 24)
        self.gemini_prompt_button.clicked.connect(lambda: self.show_model_prompt('gemini'))
        self.gemini_prompt_button.setToolTip("Показать промпт")
        
        gemini_layout.addWidget(self.gemini_radio)
        gemini_layout.addWidget(self.gemini_status_indicator)
        gemini_layout.addWidget(self.gemini_prompt_button)
        self.cloud_models_group.add_layout(gemini_layout)
        
        # Gemini sub-model selector - компактный
        gemini_sub_layout = QHBoxLayout()
        gemini_sub_layout.setContentsMargins(16, 0, 0, 0)
        gemini_sub_layout.setSpacing(4)
        self.gemini_sub_model_label = QLabel("Модель:")
        self.gemini_sub_model_label.setMinimumWidth(50)
        self.gemini_model_selector = QComboBox()
        self.gemini_model_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.gemini_model_selector.currentIndexChanged.connect(self.on_gemini_sub_model_changed)
        gemini_sub_layout.addWidget(self.gemini_sub_model_label)
        gemini_sub_layout.addWidget(self.gemini_model_selector, 1)
        self.cloud_models_group.add_layout(gemini_sub_layout)
        
        # Cloud LLM Provider Selection - компактная версия
        cloud_llm_layout = QHBoxLayout()
        cloud_llm_layout.setSpacing(4)
        self.cloud_llm_radio = QRadioButton("Другие облачные LLM")
        self.cloud_llm_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.cloud_llm_radio)
        self.cloud_llm_prompt_button = QPushButton("📝")
        self.cloud_llm_prompt_button.setFixedSize(24, 24)
        self.cloud_llm_prompt_button.clicked.connect(lambda: self.show_model_prompt('cloud_llm'))
        self.cloud_llm_prompt_button.setToolTip("Показать промпт")
        cloud_llm_layout.addWidget(self.cloud_llm_radio)
        cloud_llm_layout.addWidget(self.cloud_llm_prompt_button)
        self.cloud_models_group.add_layout(cloud_llm_layout)
        
        # Cloud provider and model selection - компактная версия
        cloud_selection_layout = QVBoxLayout()
        cloud_selection_layout.setContentsMargins(16, 0, 0, 0)
        cloud_selection_layout.setSpacing(2)
        
        cloud_provider_layout = QHBoxLayout()
        cloud_provider_layout.setSpacing(4)
        self.cloud_provider_label = QLabel("Провайдер:")
        self.cloud_provider_label.setMinimumWidth(60)
        self.cloud_provider_selector = QComboBox()
        self.cloud_provider_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cloud_provider_selector.currentIndexChanged.connect(self.on_cloud_provider_changed)
        cloud_provider_layout.addWidget(self.cloud_provider_label)
        cloud_provider_layout.addWidget(self.cloud_provider_selector, 1)
        cloud_selection_layout.addLayout(cloud_provider_layout)
        
        cloud_model_layout = QHBoxLayout()
        cloud_model_layout.setSpacing(4)
        self.cloud_model_label = QLabel("Модель:")
        self.cloud_model_label.setMinimumWidth(60)
        self.cloud_model_selector = QComboBox()
        self.cloud_model_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cloud_model_selector.currentIndexChanged.connect(self.on_cloud_model_changed)
        cloud_model_layout.addWidget(self.cloud_model_label)
        cloud_model_layout.addWidget(self.cloud_model_selector, 1)
        cloud_selection_layout.addLayout(cloud_model_layout)
        
        # Cloud status indicator - компактный
        self.cloud_llm_status_label = QLabel("Не настроено")
        self.cloud_llm_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 1px 0;")
        cloud_selection_layout.addWidget(self.cloud_llm_status_label)
        
        self.cloud_models_group.add_layout(cloud_selection_layout)
        model_layout.addWidget(self.cloud_models_group)
        
        # Populate Gemini models и проверяем статус API ключа
        self.populate_gemini_models()
        self.check_gemini_api_status()
        
        # === ЛОКАЛЬНЫЕ МОДЕЛИ - СВОРАЧИВАЕМАЯ ГРУППА ===
        self.local_models_group = CollapsibleGroup("🖥️ Локальные модели")
        self.local_models_group.set_expanded(False)  # ИЗМЕНЕНИЕ: По умолчанию свернута, будет развернута при загрузке настроек
        
        # LayoutLM - компактная версия
        layoutlm_layout = QHBoxLayout()
        layoutlm_layout.setSpacing(4)
        self.layoutlm_radio = QRadioButton("LayoutLMv3")
        # ИСПРАВЛЕНИЕ: Убираем принудительную установку, будет выбираться из настроек
        self.layoutlm_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.layoutlm_radio)
        self.layoutlm_prompt_button = QPushButton("📝")
        self.layoutlm_prompt_button.setFixedSize(24, 24)
        self.layoutlm_prompt_button.clicked.connect(lambda: self.show_model_prompt('layoutlm'))
        self.layoutlm_prompt_button.setToolTip("Показать промпт")
        layoutlm_layout.addWidget(self.layoutlm_radio)
        layoutlm_layout.addWidget(self.layoutlm_prompt_button)
        self.local_models_group.add_layout(layoutlm_layout)
        
        # Donut - компактная версия
        donut_layout = QHBoxLayout()
        donut_layout.setSpacing(4)
        self.donut_radio = QRadioButton("Donut")
        self.donut_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.donut_radio)
        self.donut_prompt_button = QPushButton("📝")
        self.donut_prompt_button.setFixedSize(24, 24)
        self.donut_prompt_button.clicked.connect(lambda: self.show_model_prompt('donut'))
        self.donut_prompt_button.setToolTip("Показать промпт")
        donut_layout.addWidget(self.donut_radio)
        donut_layout.addWidget(self.donut_prompt_button)
        self.local_models_group.add_layout(donut_layout)
        
        # TrOCR - компактная версия
        trocr_layout = QHBoxLayout()
        trocr_layout.setSpacing(4)
        self.trocr_radio = QRadioButton("TrOCR (Microsoft)")
        self.trocr_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.trocr_radio)
        self.trocr_prompt_button = QPushButton("📝")
        self.trocr_prompt_button.setFixedSize(24, 24)
        self.trocr_prompt_button.clicked.connect(lambda: self.show_model_prompt('trocr'))
        self.trocr_prompt_button.setToolTip("Показать промпт")
        trocr_layout.addWidget(self.trocr_radio)
        trocr_layout.addWidget(self.trocr_prompt_button)
        self.local_models_group.add_layout(trocr_layout)
        
        # TrOCR model selection - компактная версия
        trocr_selection_layout = QVBoxLayout()
        trocr_selection_layout.setContentsMargins(16, 0, 0, 0)
        trocr_selection_layout.setSpacing(2)
        
        trocr_model_layout = QHBoxLayout()
        trocr_model_layout.setSpacing(4)
        self.trocr_model_label = QLabel("Модель:")
        self.trocr_model_label.setMinimumWidth(50)
        self.trocr_model_selector = QComboBox()
        self.trocr_model_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.trocr_model_selector.currentIndexChanged.connect(self.on_trocr_model_changed)
        self.trocr_model_selector.setToolTip("Выберите модель TrOCR для использования")
        
        # Кнопка обновления списка моделей
        self.trocr_refresh_button = QPushButton("🔄")
        self.trocr_refresh_button.setFixedSize(24, 24)
        self.trocr_refresh_button.clicked.connect(self.refresh_trained_models)
        self.trocr_refresh_button.setToolTip("Обновить список дообученных моделей")
        self.trocr_refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        
        trocr_model_layout.addWidget(self.trocr_model_label)
        trocr_model_layout.addWidget(self.trocr_model_selector, 1)
        trocr_model_layout.addWidget(self.trocr_refresh_button)
        trocr_selection_layout.addLayout(trocr_model_layout)
        
        # TrOCR status indicator - компактный
        self.trocr_status_label = QLabel("Не загружено")
        self.trocr_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 1px 0;")
        trocr_selection_layout.addWidget(self.trocr_status_label)
        
        self.local_models_group.add_layout(trocr_selection_layout)
        
        # Local LLM Models Section - компактная версия
        local_llm_layout = QHBoxLayout()
        local_llm_layout.setSpacing(4)
        self.local_llm_radio = QRadioButton("Локальные LLM (Ollama)")
        self.local_llm_radio.toggled.connect(self.on_model_changed)
        self.model_button_group.addButton(self.local_llm_radio)
        self.local_llm_prompt_button = QPushButton("📝")
        self.local_llm_prompt_button.setFixedSize(24, 24)
        self.local_llm_prompt_button.clicked.connect(lambda: self.show_model_prompt('local_llm'))
        self.local_llm_prompt_button.setToolTip("Показать промпт")
        local_llm_layout.addWidget(self.local_llm_radio)
        local_llm_layout.addWidget(self.local_llm_prompt_button)
        self.local_models_group.add_layout(local_llm_layout)
        
        # Local provider and model selection - компактная версия
        local_selection_layout = QVBoxLayout()
        local_selection_layout.setContentsMargins(16, 0, 0, 0)
        local_selection_layout.setSpacing(2)
        
        local_provider_layout = QHBoxLayout()
        local_provider_layout.setSpacing(4)
        self.local_provider_label = QLabel("Провайдер:")
        self.local_provider_label.setMinimumWidth(60)
        self.local_provider_selector = QComboBox()
        self.local_provider_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.local_provider_selector.currentIndexChanged.connect(self.on_local_provider_changed)
        local_provider_layout.addWidget(self.local_provider_label)
        local_provider_layout.addWidget(self.local_provider_selector, 1)
        local_selection_layout.addLayout(local_provider_layout)
        
        local_model_layout = QHBoxLayout()
        local_model_layout.setSpacing(4)
        self.local_model_label = QLabel("Модель:")
        self.local_model_label.setMinimumWidth(60)
        self.local_model_selector = QComboBox()
        self.local_model_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.local_model_selector.currentIndexChanged.connect(self.on_local_model_changed)
        local_model_layout.addWidget(self.local_model_label)
        local_model_layout.addWidget(self.local_model_selector, 1)
        local_selection_layout.addLayout(local_model_layout)
        
        # Local status indicator - компактный
        self.local_llm_status_label = QLabel("Не настроено")
        self.local_llm_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 1px 0;")
        local_selection_layout.addWidget(self.local_llm_status_label)
        
        self.local_models_group.add_layout(local_selection_layout)
        model_layout.addWidget(self.local_models_group)

        model_group.setLayout(model_layout)
        
        # Компактная секция OCR
        ocr_lang_group = QGroupBox("Язык OCR")
        ocr_lang_layout = QVBoxLayout()
        ocr_lang_layout.setContentsMargins(8, 6, 8, 6)  # Уменьшенные отступы
        
        self.ocr_lang_combo = QComboBox()
        self.ocr_lang_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ocr_lang_combo.addItem("English", "eng")
        self.ocr_lang_combo.addItem("Русский", "rus")
        self.ocr_lang_combo.addItem("English + Русский", "eng+rus")
        self.ocr_lang_combo.currentIndexChanged.connect(self.on_ocr_language_changed)
        
        ocr_lang_layout.addWidget(self.ocr_lang_combo)
        ocr_lang_group.setLayout(ocr_lang_layout)
        
        # === КОМПАКТНАЯ СЕКЦИЯ ОБРАБОТКИ ===
        process_section = QWidget()
        process_layout = QVBoxLayout(process_section)
        process_layout.setContentsMargins(0, 0, 0, 0)
        process_layout.setSpacing(4)  # Минимальные отступы
        
        # ФАЗА 2: Анимированная кнопка Обработать/Отменить
        self.process_button = AnimatedButton("🚀 Обработать")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.on_process_button_clicked)
        self.process_button.setMinimumHeight(28)  # Еще более компактная высота
        
        # ФАЗА 2: Смарт прогресс-бар с ETA
        self.progress_bar = SmartProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(12)  # Очень узкий прогресс-бар
        
        # Статус обработки (компактный)
        self.process_status_label = QLabel("")
        self.process_status_label.setVisible(False)
        self.process_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px 0;")
        self.process_status_label.setWordWrap(True)
        
        # Добавляем в компактную секцию
        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.progress_bar)
        process_layout.addWidget(self.process_status_label)
        
        # Переменные состояния обработки
        self.is_processing = False
        self.processing_thread = None
        
        # Компактная область результатов
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(8, 6, 8, 6)  # Уменьшенные отступы
        results_layout.setSpacing(4)  # Уменьшенные промежутки
        
        # Компактный заголовок с кнопками
        table_header_layout = QHBoxLayout()
        table_header_layout.setSpacing(4)
        table_header_label = QLabel("Данные:")
        table_header_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        table_header_layout.addWidget(table_header_label)
        
        # Компактная кнопка управления полями
        self.edit_fields_button = QPushButton("🔧")
        self.edit_fields_button.setFixedSize(24, 24)
        self.edit_fields_button.clicked.connect(self.show_field_manager_dialog)
        self.edit_fields_button.setToolTip("Управление полями")
        self.edit_fields_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        table_header_layout.addWidget(self.edit_fields_button)
        table_header_layout.addStretch()
        
        results_layout.addLayout(table_header_layout)
        
        # ФАЗА 2: Создаем оптимизированную таблицу для отображения результатов
        self.results_table = OptimizedTableWidget()
        
        # Настраиваем таблицу исходя из полей в настройках
        self.setup_results_table()
        
        results_layout.addWidget(self.results_table)
        
        # Компактные кнопки сохранения в Grid Layout
        save_buttons_grid = QGridLayout()
        save_buttons_grid.setSpacing(4)
        
        # Общий стиль для кнопок
        button_style = """
            QPushButton {
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                padding: 4px 6px;
                min-height: 20px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        # Template Designer button - компактная версия
        self.template_designer_button = QPushButton("🎨 Шаблоны")
        self.template_designer_button.clicked.connect(self.show_template_designer)
        self.template_designer_button.setToolTip("Дизайнер шаблонов")
        self.template_designer_button.setStyleSheet(button_style + "QPushButton { background-color: #9C27B0; } QPushButton:hover { background-color: #8E24AA; }")
        save_buttons_grid.addWidget(self.template_designer_button, 0, 0)
        
        # Preview button - компактная версия
        self.preview_button = QPushButton("🔍 Просмотр")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.show_preview_dialog)
        self.preview_button.setStyleSheet(button_style + "QPushButton { background-color: #FF9800; } QPushButton:hover { background-color: #F57C00; }")
        save_buttons_grid.addWidget(self.preview_button, 0, 1)
        
        # Save JSON button - компактная версия
        self.save_button = QPushButton("💾 JSON")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setStyleSheet(button_style + "QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #43A047; }")
        save_buttons_grid.addWidget(self.save_button, 1, 0)
        
        # Save Excel button - компактная версия
        self.save_excel_button = QPushButton("📊 Excel")
        self.save_excel_button.setEnabled(False)
        self.save_excel_button.clicked.connect(self.save_excel)
        self.save_excel_button.setStyleSheet(button_style + "QPushButton { background-color: #2196F3; } QPushButton:hover { background-color: #1976D2; }")
        save_buttons_grid.addWidget(self.save_excel_button, 1, 1)
        
        results_layout.addLayout(save_buttons_grid)
        
        results_group.setLayout(results_layout)
        
        # Добавляем виджеты в правую часть с компактным spacing
        controls_layout.setSpacing(6)  # Уменьшенные отступы между секциями
        controls_layout.setContentsMargins(4, 4, 4, 4)  # Уменьшенные отступы по краям
        
        controls_layout.addWidget(model_group)
        controls_layout.addWidget(ocr_lang_group)
        controls_layout.addWidget(process_section)  # Компактная секция обработки
        controls_layout.addWidget(results_group, 1)  # Таблица результатов растягивается
        
        # Добавляем левую и правую части в сплиттер
        splitter.addWidget(self.files_widget)  # Новый список файлов вместо изображения
        splitter.addWidget(self.controls_scroll)
        
        # Улучшенная адаптивная настройка splitter
        splitter.setHandleWidth(8)  # Увеличиваем ширину handle для удобства
        splitter.setChildrenCollapsible(False)  # Запрещаем полное сжатие панелей
        
        # Настройка пропорций с учетом адаптивности
        splitter.setSizes([320, 680])  # Компактная левая панель, больше места для правой
        splitter.setStretchFactor(0, 0)  # Левая панель фиксированной ширины
        splitter.setStretchFactor(1, 1)  # Правая панель растягивается
        
        # Устанавливаем минимальные размеры панелей
        self.files_widget.setMinimumWidth(300)  # Уменьшили ширину для компактного списка файлов
        self.files_widget.setMaximumWidth(350)  # Ограничиваем максимальную ширину
        self.controls_scroll.setMinimumWidth(350)
        self.controls_scroll.setMaximumWidth(600)  # Ограничиваем максимальную ширину
        
        # Добавляем сплиттер в главную компоновку
        main_layout.addWidget(splitter)
        
        # Устанавливаем центральный виджет
        self.setCentralWidget(central_widget)
        
        # Создаем строку состояния
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")
        
        # Создаем меню
        self.create_menus()

        # NEW: Включаем сортировку для таблицы
        self.results_table.setSortingEnabled(True)
        
        # Загружаем и применяем сохраненные настройки
        self.load_saved_settings()
        
        # ИСПРАВЛЕНИЕ: Инициализируем провайдеры при запуске
        self.initialize_providers()
        
        # ИСПРАВЛЕНИЕ: Проверяем статус компонентов после загрузки настроек
        self.update_model_component_visibility()
    
    def _init_post_ui_components(self):
        """Инициализирует компоненты, которые требуют готового UI."""
        try:
            # Initialize ExportManager 
            self.export_manager = ExportManager()
            print("ExportManager инициализирован")
            
            # Initialize BatchProcessor только если model_manager готов
            if hasattr(self, 'model_manager') and self.model_manager:
                self.batch_processor = BatchProcessor(self.model_manager)
                self.batch_processor.processing_started.connect(self.on_batch_processing_started)
                self.batch_processor.file_processed.connect(self.on_batch_file_processed)
                self.batch_processor.processing_finished.connect(self.on_batch_processing_finished)
                self.batch_processor.error_occurred.connect(self.on_batch_error)
                
                # Connect progress indicator to batch processor if it exists
                if hasattr(self, 'progress_bar') and self.progress_bar:
                    self.batch_processor.progress_updated.connect(self.update_progress)
                    self.batch_processor.status_updated.connect(self.set_processing_status)
                    
                print("BatchProcessor инициализирован")
            else:
                print("model_manager еще не готов, отложим инициализацию BatchProcessor")
                # Повторяем попытку через 100ms
                QTimer.singleShot(100, self._init_batch_processor)
                
        except Exception as e:
            print(f"Ошибка в _init_post_ui_components: {e}")
            import traceback
            traceback.print_exc()
            
    def _init_batch_processor(self):
        """Отложенная инициализация BatchProcessor."""
        try:
            if hasattr(self, 'model_manager') and self.model_manager and not hasattr(self, 'batch_processor'):
                self.batch_processor = BatchProcessor(self.model_manager)
                self.batch_processor.processing_started.connect(self.on_batch_processing_started)
                self.batch_processor.file_processed.connect(self.on_batch_file_processed)
                self.batch_processor.processing_finished.connect(self.on_batch_processing_finished)
                self.batch_processor.error_occurred.connect(self.on_batch_error)
                
                if hasattr(self, 'progress_bar') and self.progress_bar:
                    self.batch_processor.progress_updated.connect(self.update_progress)
                    self.batch_processor.status_updated.connect(self.set_processing_status)
                    
                print("BatchProcessor инициализирован (отложенная инициализация)")
        except Exception as e:
            print(f"Ошибка отложенной инициализации BatchProcessor: {e}")
    
    def on_file_selected(self, file_path: str):
        """Обработчик выбора файла через FileSelector."""
        self.current_image_path = file_path
        self.current_folder_path = None
        
        # Обновляем список файлов
        self.update_files_from_selection(file_path, False)
        
        # ИСПРАВЛЕНИЕ: Активируем кнопку обработки при выборе файла
        self.process_button.setText("🚀 Обработать")
        self.process_button.setEnabled(True)
        
        # Обновляем статус
        filename = os.path.basename(file_path)
        self.status_bar.showMessage(f"Выбран файл: {filename}")
        
        # Старый код для загрузки изображения (для совместимости)
        if hasattr(self, 'image_widget') and self.image_widget.isVisible():
            self.load_image(file_path)
        
    def on_folder_selected(self, folder_path: str):
        """Обработчик выбора папки через FileSelector."""
        self.current_folder_path = folder_path
        self.current_image_path = None
        
        # Обновляем список файлов
        self.update_files_from_selection(folder_path, True)
        
        # Enable batch processing
        self.process_button.setText("🚀 Обработать папку")
        self.process_button.setEnabled(True)
        self.status_bar.showMessage(f"Выбрана папка: {folder_path}")
    def on_batch_processing_started(self, total_files: int):
        """Обработчик начала пакетной обработки."""
        self.batch_results = []
        self.start_processing_ui()
        self.set_processing_status(f"Пакетная обработка: {total_files} файлов...")
        
    def on_batch_file_processed(self, file_path: str, result: dict, index: int, total: int):
        """Обработчик обработки одного файла из пакета."""
        self.batch_results.append({
            'file_path': file_path,
            'result': result
        })
        # Обновляем прогресс
        progress_value = int((index + 1) / total * 100)
        self.update_progress(progress_value)
        self.set_processing_status(f"Обработано {index + 1} из {total} файлов")
        
    def on_batch_processing_finished(self):
        """Обработчик завершения пакетной обработки."""
        self.stop_processing_ui()
        
        if self.batch_results:
            # Show batch results
            self.show_batch_results()
            
    def on_batch_error(self, error_message: str):
        """Обработчик ошибки при пакетной обработке."""
        self.stop_processing_ui()
        QMessageBox.critical(self, "Ошибка обработки", error_message)
        
    def _process_folder_with_batch_processor(self, folder_path: str):
        """Обрабатывает папку с помощью BatchProcessor."""
        if not hasattr(self, 'batch_processor') or not self.batch_processor:
            utils.show_error_message(
                self, "Ошибка", 
                "Компонент пакетной обработки не инициализирован. Попробуйте перезапустить приложение."
            )
            return
            
        # Определяем тип модели
        model_type = "layoutlm" if self.layoutlm_radio.isChecked() else "donut"
        if self.trocr_radio.isChecked():
            model_type = "trocr"
        elif self.gemini_radio.isChecked():
            model_type = "gemini"
        elif self.cloud_llm_radio.isChecked():
            model_type = "cloud_llm"
        elif self.local_llm_radio.isChecked():
            model_type = "local_llm"
            
        ocr_lang = self.ocr_lang_combo.currentData() if model_type == "layoutlm" else None
        
        # Дополнительные настройки для моделей
        model_settings = {}
        if model_type == "gemini":
            # Передаем выбранную sub-модель
            model_settings['sub_model_id'] = self.gemini_model_selector.currentData()
            
        # Отключаем кнопку обработки на время
        self.process_button.setEnabled(False)
        
        # Запускаем пакетную обработку
        try:
            self.batch_processor.process_folder(
                folder_path,
                model_type,
                ocr_lang,
                model_settings
            )
        except Exception as e:
            print(f"Ошибка запуска пакетной обработки: {e}")
            utils.show_error_message(
                self, "Ошибка обработки",
                f"Не удалось начать обработку папки: {str(e)}"
            )
            self.process_button.setEnabled(True)
        
    def show_batch_results(self):
        """Показывает результаты пакетной обработки."""
        if not self.batch_results:
            return
            
        # Clear existing results
        self.results_table.setRowCount(0)
        
        # Add all results to table
        successful_count = 0
        for batch_item in self.batch_results:
            result = batch_item['result']
            file_name = os.path.basename(batch_item['file_path'])
            
            if result:
                # Добавляем имя файла к результату
                result_with_file = result.copy() if isinstance(result, dict) else {}
                result_with_file['_source_file'] = file_name
                
                # Если результат содержит вложенную структуру с 'data'
                if isinstance(result, dict) and 'data' in result:
                    result_data = result['data']
                    if isinstance(result_data, dict):
                        result_data['_source_file'] = file_name
                        self.append_result_to_table(result_data)
                    else:
                        self.append_result_to_table(result_with_file)
                else:
                    # Прямой результат
                    self.append_result_to_table(result_with_file)
                    
                successful_count += 1
                
        # Enable export buttons
        self.save_button.setEnabled(True)
        self.save_excel_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        
        # Обновляем статус
        self.status_bar.showMessage(f"Обработано файлов: {successful_count} из {len(self.batch_results)}")
        
        # Показываем сообщение о завершении
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Пакетная обработка завершена",
            f"Успешно обработано {successful_count} из {len(self.batch_results)} файлов.\n"
            f"Результаты отображены в таблице."
        )
    
    def load_saved_settings(self):
        """Загружает сохраненные пользовательские настройки и применяет их."""
        # ИСПРАВЛЕНИЕ: Сначала снимаем выбор со всех кнопок
        for button in self.model_button_group.buttons():
            button.setChecked(False)
            
        # Загрузка активной модели
        active_model = settings_manager.get_active_model()
        print(f"🔧 Загружаем активную модель из настроек: {active_model}")
        
        # НОВОЕ: Определяем, какая группа должна быть развернута
        is_cloud_model = active_model in ['gemini', 'cloud_llm']
        is_local_model = active_model in ['layoutlm', 'donut', 'trocr', 'local_llm']
        
        # Устанавливаем выбранную модель
        if active_model == 'layoutlm':
            self.layoutlm_radio.setChecked(True)
        elif active_model == 'donut':
            self.donut_radio.setChecked(True)
        elif active_model == 'trocr':
            self.trocr_radio.setChecked(True)
        elif active_model == 'gemini':
            print("🔍 Устанавливаем Gemini как активную модель")
            self.gemini_radio.setChecked(True)
        elif active_model == 'cloud_llm':
            self.cloud_llm_radio.setChecked(True)
        elif active_model == 'local_llm':
            self.local_llm_radio.setChecked(True)
        else:
            # По умолчанию выбираем LayoutLM только если никакая модель не сохранена
            print("⚠️ Неизвестная активная модель, устанавливаем LayoutLM по умолчанию")
            self.layoutlm_radio.setChecked(True)
            is_local_model = True
            is_cloud_model = False
        
        # НОВОЕ: Управляем видимостью групп в зависимости от выбранной модели
        self.update_group_visibility_based_on_model(is_cloud_model, is_local_model)
        
        print(f"📂 Группы: Облачные={'развернуты' if is_cloud_model else 'свернуты'}, "
              f"Локальные={'развернуты' if is_local_model else 'свернуты'}")
        
        # Загрузка языка OCR
        ocr_lang = settings_manager.get_string('OCR', 'language', 'rus+eng')
        for i in range(self.ocr_lang_combo.count()):
            if self.ocr_lang_combo.itemData(i) == ocr_lang:
                self.ocr_lang_combo.setCurrentIndex(i)
                break
        
        # Загрузка выбранной модели Gemini
        selected_gemini_model = settings_manager.get_string('Gemini', 'selected_model', 'models/gemini-2.0-flash-exp')
        for i in range(self.gemini_model_selector.count()):
            if self.gemini_model_selector.itemData(i) == selected_gemini_model:
                self.gemini_model_selector.setCurrentIndex(i)
                break
        
        print(f"Загружены настройки: модель={active_model}, OCR={ocr_lang}, Gemini={selected_gemini_model}")
    
    def initialize_providers(self):
        """Инициализирует провайдеров при запуске приложения."""
        try:
            print("🔄 Инициализация провайдеров...")
            
            # Загружаем облачных провайдеров
            if hasattr(self, 'cloud_provider_selector'):
                self.populate_cloud_providers()
                print("☁️ Облачные провайдеры инициализированы")
            
            # Загружаем локальных провайдеров
            if hasattr(self, 'local_provider_selector'):
                self.populate_local_providers()
                print("🖥️ Локальные провайдеры инициализированы")
                
            print("✅ Провайдеры успешно инициализированы")
            
        except Exception as e:
            print(f"❌ Ошибка при инициализации провайдеров: {e}")
    
    def _auto_select_cloud_provider(self):
        """Автоматически выбирает первого доступного облачного провайдера."""
        try:
            if hasattr(self, 'cloud_provider_selector') and self.cloud_provider_selector.currentIndex() == 0 and self.cloud_provider_selector.count() > 1:
                print("🔄 Автоматически выбираем первого облачного провайдера...")
                self.cloud_provider_selector.setCurrentIndex(1)  # Выбираем первого реального провайдера
                # Принудительно вызываем обработчик изменения провайдера
                self.on_cloud_provider_changed()
                # Обновляем видимость компонентов после выбора провайдера
                QTimer.singleShot(50, self.update_model_component_visibility)
        except Exception as e:
            print(f"❌ Ошибка автоматического выбора облачного провайдера: {e}")

    def _auto_select_local_provider(self):
        """Автоматически выбирает первого локального провайдера (даже если недоступен)."""
        try:
            if hasattr(self, 'local_provider_selector') and self.local_provider_selector.currentIndex() == 0 and self.local_provider_selector.count() > 1:
                print("🔄 Автоматически выбираем первого локального провайдера...")
                self.local_provider_selector.setCurrentIndex(1)  # Выбираем первого реального провайдера
                # Принудительно вызываем обработчик изменения провайдера
                self.on_local_provider_changed()
                # Обновляем видимость компонентов после выбора провайдера
                QTimer.singleShot(50, self.update_model_component_visibility)
        except Exception as e:
            print(f"❌ Ошибка автоматического выбора локального провайдера: {e}")
    
    def on_model_changed(self, checked):
        """Обработчик события изменения модели."""
        if checked:
            # Получаем выбранную модель
            model_name = 'layoutlm'  # По умолчанию
            if self.layoutlm_radio.isChecked():
                model_name = 'layoutlm'
            elif self.donut_radio.isChecked():
                model_name = 'donut'
            elif self.trocr_radio.isChecked():
                model_name = 'trocr'
            elif self.gemini_radio.isChecked():
                model_name = 'gemini'
            elif self.cloud_llm_radio.isChecked():
                model_name = 'cloud_llm'
            elif self.local_llm_radio.isChecked():
                model_name = 'local_llm'
                
            # НОВОЕ: Сохраняем выбор и обновляем UI групп
            self.save_model_selection_and_update_ui(model_name)
            
            # Обновляем доступность выбора языка OCR (только для LayoutLM)
            self.ocr_lang_combo.setEnabled(model_name == 'layoutlm')
            
            # Обновляем статусы LLM, если выбраны
            if model_name == 'cloud_llm':
                self.update_cloud_llm_status()
            elif model_name == 'local_llm':
                self.update_local_llm_status()
            elif model_name == 'gemini':
                self.check_gemini_api_status()
            
            print(f"Выбрана модель: {model_name}")
    
    def on_ocr_language_changed(self, index):
        """Обработчик изменения языка OCR."""
        selected_lang = self.ocr_lang_combo.currentData()
        print(f"Выбран язык OCR: {selected_lang}")
        # Сохраняем выбор в настройках
        settings_manager.set_value('OCR', 'language', selected_lang)
        # Обновляем настройку в конфиге для использования в реальном времени
        app_config.DEFAULT_TESSERACT_LANG = selected_lang
    
    def auto_load_llm_plugin(self, model_type, provider_data, model_data):
        """Автоматическая загрузка LLM плагина синхронно"""
        try:
            provider_name = provider_data.get('provider')
            model_name = model_data.get('model')
            config = provider_data.get('config')
            
            # Получаем настройки провайдера
            llm_settings = settings_manager.get_setting('llm_providers', {})
            provider_settings = llm_settings.get(provider_name, {})
            
            # Получаем API ключ если требуется
            api_key = None
            if config.requires_api_key:
                api_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
                if not api_key:
                    print(f"❌ API ключ для {provider_name} не найден")
                    return False
            
            # Создаем экземпляр универсального плагина
            from .plugins.models.universal_llm_plugin import UniversalLLMPlugin
            
            # Дополнительные параметры
            plugin_kwargs = {
                'generation_config': {
                    'temperature': provider_settings.get('temperature', 0.1),
                    'max_tokens': provider_settings.get('max_tokens', 4096),
                    'top_p': provider_settings.get('top_p', 0.9),
                }
            }
            
            # Для Ollama добавляем base_url
            if provider_name == "ollama":
                plugin_kwargs['base_url'] = provider_settings.get('base_url', 'http://localhost:11434')
            
            plugin = UniversalLLMPlugin(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key,
                **plugin_kwargs
            )
            
            # Инициализируем плагин
            if plugin.load_model():
                self.current_llm_plugin = plugin
                print(f"✅ Автоматически загружен {provider_name} плагин")
                
                # Обновляем статус в UI
                if model_type == "cloud_llm":
                    self.update_cloud_llm_status()
                else:
                    self.update_local_llm_status()
                
                return True
            else:
                print(f"❌ Не удалось инициализировать {provider_name} плагин")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка автоматической загрузки плагина: {e}")
            return False
    
    # NEW: Методы для работы с универсальной системой плагинов
    
    def export_with_plugin(self, data, output_path: str, format_type: str):
        """Экспортирует данные используя плагин экспорта"""
        try:
            success = self.universal_plugin_manager.export_data(data, output_path, format_type)
            if success:
                utils.show_info_message(
                    self, "Успех", f"Данные успешно экспортированы в {output_path}"
                )
            else:
                utils.show_error_message(
                    self, "Ошибка экспорта", f"Не удалось экспортировать данные в формат {format_type}"
                )
            return success
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка экспорта", f"Ошибка при экспорте: {e}"
            )
            return False
    
    def validate_with_plugin(self, data, validator_type: str = "invoice"):
        """Валидирует данные используя плагин валидации"""
        try:
            validation_result = self.universal_plugin_manager.validate_data(data, validator_type)
            
            errors = validation_result.get('errors', [])
            warnings = validation_result.get('warnings', [])
            
            if errors:
                error_msg = "Найдены ошибки валидации:\n" + "\n".join(errors)
                if warnings:
                    error_msg += "\n\nПредупреждения:\n" + "\n".join(warnings)
                utils.show_error_message(self, "Ошибки валидации", error_msg)
                return False
            elif warnings:
                warning_msg = "Предупреждения валидации:\n" + "\n".join(warnings)
                utils.show_warning_message(self, "Предупреждения валидации", warning_msg)
            
            return True
            
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка валидации", f"Ошибка при валидации: {e}"
            )
            return False
    
    def create_data_viewer(self, data, viewer_type: str = "table"):
        """Создает просмотрщик данных используя плагин"""
        try:
            viewer = self.universal_plugin_manager.create_viewer(data, viewer_type, self)
            if viewer:
                # Создаем диалог для отображения просмотрщика
                from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton
                
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Просмотр данных - {viewer_type}")
                dialog.setModal(True)
                dialog.resize(800, 600)
                
                layout = QVBoxLayout()
                layout.addWidget(viewer)
                
                # Кнопка закрытия
                close_button = QPushButton("Закрыть")
                close_button.clicked.connect(dialog.accept)
                layout.addWidget(close_button)
                
                dialog.setLayout(layout)
                dialog.exec()
                
                return True
            else:
                utils.show_error_message(
                    self, "Ошибка просмотра", f"Не удалось создать просмотрщик типа {viewer_type}"
                )
                return False
                
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка просмотра", f"Ошибка при создании просмотрщика: {e}"
            )
            return False
    
    def get_plugin_statistics(self):
        """Возвращает статистику по плагинам"""
        try:
            stats = self.universal_plugin_manager.get_statistics()
            
            # Создаем диалог для отображения статистики
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Статистика плагинов")
            dialog.setModal(True)
            dialog.resize(600, 400)
            
            layout = QVBoxLayout()
            
            # Текстовое поле для статистики
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            
            stats_text = "📊 Статистика универсальной системы плагинов\n\n"
            stats_text += f"Всего доступно: {stats['total']['available']}\n"
            stats_text += f"Всего загружено: {stats['total']['loaded']}\n\n"
            
            for plugin_type, type_stats in stats.items():
                if plugin_type != 'total':
                    stats_text += f"📋 {plugin_type.upper()}:\n"
                    stats_text += f"   Доступно: {type_stats['available']}\n"
                    stats_text += f"   Загружено: {type_stats['loaded']}\n"
                    if type_stats['plugins']:
                        stats_text += f"   Плагины: {', '.join(type_stats['plugins'])}\n"
                    stats_text += "\n"
            
            text_edit.setPlainText(stats_text)
            layout.addWidget(text_edit)
            
            # Кнопка закрытия
            close_button = QPushButton("Закрыть")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка", f"Ошибка получения статистики плагинов: {e}"
            )
    
    def validate_current_data(self):
        """Валидирует текущие данные"""
        try:
            if self.batch_mode:
                if hasattr(self, 'batch_results') and self.batch_results:
                    # Валидируем каждый результат в пакете
                    all_valid = True
                    for i, result in enumerate(self.batch_results):
                        if not self.validate_with_plugin(result, "invoice"):
                            all_valid = False
                            print(f"Ошибка валидации в результате {i+1}")
                    
                    if all_valid:
                        utils.show_info_message(
                            self, "Валидация", "Все результаты пакетной обработки прошли валидацию"
                        )
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для валидации. Сначала обработайте файлы."
                    )
            else:
                if hasattr(self, 'processing_thread') and self.processing_thread and \
                   hasattr(self.processing_thread, 'result') and self.processing_thread.result:
                    if self.validate_with_plugin(self.processing_thread.result, "invoice"):
                        utils.show_info_message(
                            self, "Валидация", "Данные прошли валидацию успешно"
                        )
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для валидации. Сначала обработайте файл."
                    )
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка валидации", f"Ошибка при валидации данных: {e}"
            )
    
    def view_current_data(self):
        """Показывает текущие данные в просмотрщике"""
        try:
            if self.batch_mode:
                if hasattr(self, 'batch_results') and self.batch_results:
                    self.create_data_viewer(self.batch_results, "table")
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для просмотра. Сначала обработайте файлы."
                    )
            else:
                if hasattr(self, 'processing_thread') and self.processing_thread and \
                   hasattr(self.processing_thread, 'result') and self.processing_thread.result:
                    self.create_data_viewer(self.processing_thread.result, "table")
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для просмотра. Сначала обработайте файл."
                    )
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка просмотра", f"Ошибка при просмотре данных: {e}"
            )
    
    def show_plugin_export_dialog(self):
        """Показывает диалог экспорта через плагины"""
        try:
            # Получаем данные для экспорта
            data_to_export = None
            if self.batch_mode:
                if hasattr(self, 'batch_results') and self.batch_results:
                    data_to_export = self.batch_results
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для экспорта. Сначала обработайте файлы."
                    )
                    return
            else:
                if hasattr(self, 'processing_thread') and self.processing_thread and \
                   hasattr(self.processing_thread, 'result') and self.processing_thread.result:
                    data_to_export = self.processing_thread.result
                else:
                    utils.show_warning_message(
                        self, "Предупреждение", "Нет данных для экспорта. Сначала обработайте файл."
                    )
                    return
            
            # Создаем диалог выбора формата экспорта
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QFileDialog
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Экспорт через плагины")
            dialog.setModal(True)
            dialog.resize(400, 150)
            
            layout = QVBoxLayout()
            
            # Выбор формата
            format_layout = QHBoxLayout()
            format_layout.addWidget(QLabel("Формат экспорта:"))
            
            format_combo = QComboBox()
            format_combo.addItem("JSON", "json")
            format_combo.addItem("Excel (XLSX)", "xlsx")
            format_combo.addItem("CSV", "csv")
            format_combo.addItem("PDF", "pdf")
            format_layout.addWidget(format_combo)
            
            layout.addLayout(format_layout)
            
            # Кнопки
            buttons_layout = QHBoxLayout()
            
            export_button = QPushButton("Экспортировать")
            cancel_button = QPushButton("Отмена")
            
            buttons_layout.addWidget(export_button)
            buttons_layout.addWidget(cancel_button)
            
            layout.addLayout(buttons_layout)
            dialog.setLayout(layout)
            
            def on_export():
                format_type = format_combo.currentData()
                
                # Выбираем файл для сохранения
                file_filter = {
                    "json": "JSON файлы (*.json)",
                    "xlsx": "Excel файлы (*.xlsx)",
                    "csv": "CSV файлы (*.csv)",
                    "pdf": "PDF файлы (*.pdf)"
                }.get(format_type, "Все файлы (*)")
                
                output_path, _ = QFileDialog.getSaveFileName(
                    dialog, "Сохранить как", f"export.{format_type}", file_filter
                )
                
                if output_path:
                    if self.export_with_plugin(data_to_export, output_path, format_type):
                        dialog.accept()
                    # Ошибка уже показана в export_with_plugin
            
            export_button.clicked.connect(on_export)
            cancel_button.clicked.connect(dialog.reject)
            
            dialog.exec()
            
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка экспорта", f"Ошибка при создании диалога экспорта: {e}"
            )
    

    def create_menus(self):
        """Создание меню приложения."""
        menu_bar = self.menuBar()

        # Меню Файл
        file_menu = menu_bar.addMenu("Файл")
        
        open_action = QAction("Открыть...", self)
        open_action.triggered.connect(self.select_file)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        # Добавляем действие "Открыть папку" в меню
        open_folder_action = QAction("Открыть папку...", self)
        open_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()

        save_action = QAction("Сохранить результаты...", self)
        save_action.triggered.connect(self.save_results)
        save_action.setShortcut("Ctrl+S")
        save_action.setEnabled(False) 
        self.save_action = save_action
        file_menu.addAction(save_action)
        
        save_excel_action = QAction("Экспорт в Excel...", self)
        save_excel_action.triggered.connect(self.save_excel)
        save_excel_action.setShortcut("Ctrl+E")
        save_excel_action.setEnabled(False)
        self.save_excel_action = save_excel_action
        file_menu.addAction(save_excel_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # Меню Настройки
        settings_menu = menu_bar.addMenu("Настройки")
        
        # Настройки и управление моделями
        models_action = QAction("Настройки и управление моделями...", self)
        models_action.triggered.connect(self.show_model_management_dialog)
        settings_menu.addAction(models_action)
        
        # Управление полями таблицы
        fields_action = QAction("🔧 Управление полями таблицы...", self)
        fields_action.triggered.connect(self.show_field_manager_dialog)
        settings_menu.addAction(fields_action)
        
        settings_menu.addSeparator()
        
        # Добавляем новый пункт меню для LLM плагинов
        llm_plugins_action = QAction("🔌 Управление LLM плагинами...", self)
        llm_plugins_action.triggered.connect(self.show_llm_plugins_dialog)
        settings_menu.addAction(llm_plugins_action)
        
        settings_menu.addSeparator()
        
        # Новая универсальная система плагинов
        universal_plugins_action = QAction("🔧 Универсальная система плагинов...", self)
        universal_plugins_action.triggered.connect(self.get_plugin_statistics)
        settings_menu.addAction(universal_plugins_action)
        
        # Редактор плагинов
        plugin_editor_action = QAction("🔌 Редактор плагинов...", self)
        plugin_editor_action.triggered.connect(self.show_plugin_editor)
        settings_menu.addAction(plugin_editor_action)
        
        settings_menu.addSeparator()
        
        # Интеграция с Paperless-NGX
        paperless_action = QAction("📄 Интеграция с Paperless-NGX...", self)
        paperless_action.triggered.connect(self.show_paperless_integration_dialog)
        settings_menu.addAction(paperless_action)
        
        # Меню Обучение
        training_menu = menu_bar.addMenu("Обучение")
        open_training_action = QAction("Обучение моделей", self)
        open_training_action.triggered.connect(self._open_training_dialog)
        training_menu.addAction(open_training_action)

        # Меню Помощь
        help_menu = menu_bar.addMenu("Помощь")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_action = QAction("Краткая инструкция", self)
        help_action.triggered.connect(self.show_help)
        win7_info_action = QAction("Win7 Совместимость", self)
        win7_info_action.triggered.connect(self.show_win7_info)
        help_menu.addAction(about_action)
        help_menu.addAction(help_action)
        help_menu.addAction(win7_info_action)
    
    def select_file(self):
        """Открытие диалога выбора файла."""
        file_filter = "Изображения и PDF (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.pdf)"
        
        # Получаем последний сохраненный путь из настроек
        last_open_path = settings_manager.get_string('Interface', 'last_open_path', utils.get_documents_dir())
        
        file_path = utils.get_open_file_path(
            self, "Выбрать файл", last_open_path, file_filter
        )
        
        if not file_path:
            return
        
        if not utils.is_supported_format(file_path):
            utils.show_error_message(
                self, "Ошибка", "Неподдерживаемый формат файла. "
                "Поддерживаются форматы: JPG, PNG, BMP, TIFF и PDF."
            )
            return
        
        # Сохраняем директорию выбранного файла в настройках
        file_dir = os.path.dirname(file_path)
        settings_manager.save_interface_setting('last_open_path', file_dir)
            
        self.load_image(file_path)
    
    def load_image(self, file_path):
        """
        Загрузка изображения для отображения.
        
        Args:
            file_path (str): Путь к файлу изображения или PDF
        """
        self.current_image_path = file_path
        self.status_bar.showMessage(f"Загрузка файла: {os.path.basename(file_path)}...")
        
        try:
            # Если это PDF, конвертируем первую страницу в изображение
            if utils.is_pdf_format(file_path):
                try:
                    # Конвертируем первую страницу PDF в изображение
                    pdf_images = pdf2image.convert_from_path(
                        file_path, 
                        first_page=1, 
                        last_page=1,
                        dpi=200,  # Разрешение изображения
                        poppler_path=app_config.POPPLER_PATH  # Используем путь к Poppler из конфига
                    )
                    
                    if pdf_images:
                        # Сохраняем изображение во временный файл
                        temp_img_path = os.path.join(self.temp_dir.name, "temp_pdf_page.jpg")
                        pdf_images[0].save(temp_img_path, "JPEG")
                        
                        # Обновляем путь к текущему изображению
                        self.current_image_path = temp_img_path
                        self.current_folder_path = None # Сбрасываем папку, если был выбран файл
                        
                        # Загружаем изображение из временного файла
                        img = Image.open(temp_img_path)
                    else:
                        raise ValueError("Не удалось конвертировать PDF в изображение")
                except Exception as e:
                    utils.show_error_message(
                        self, "Ошибка конвертации PDF", f"Не удалось преобразовать PDF: {str(e)}"
                    )
                    return
            else:
                # Загружаем обычное изображение
                img = Image.open(file_path)
            
            # Конвертируем PIL Image в QPixmap для отображения
            img_qt = ImageQt.toqpixmap(img)
            
            # Создаем новый QLabel для изображения
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setPixmap(img_qt)
            
            # Обновляем QScrollArea
            self.image_scroll.setWidget(self.image_label)
            
            # Включаем кнопку обработки
            self.process_button.setEnabled(True)
            
            # Обновляем статус
            self.status_bar.showMessage(f"Загружен файл: {os.path.basename(file_path)}")
            
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка загрузки", f"Не удалось загрузить изображение: {str(e)}"
            )
    
    def process_image(self):
        """Обработка изображения или папки выбранной моделью."""
        # NEW: Проверяем, выбран ли файл или папка
        if not self.current_image_path and not self.current_folder_path:
            utils.show_error_message(
                self, "Ошибка", "Сначала необходимо выбрать файл или папку."
            )
            return

        input_path = self.current_folder_path if self.current_folder_path else self.current_image_path
        is_folder = bool(self.current_folder_path)
        
        # NEW: Используем BatchProcessor для обработки папок
        if is_folder and self.batch_processor:
            self._process_folder_with_batch_processor(input_path)
            return
        
        # Определяем тип модели
        model_type = "layoutlm" if self.layoutlm_radio.isChecked() else "donut"
        if self.trocr_radio.isChecked():
            model_type = "trocr"
        elif self.gemini_radio.isChecked():
            model_type = "gemini"
        elif self.cloud_llm_radio.isChecked():
            model_type = "cloud_llm"
        elif self.local_llm_radio.isChecked():
            model_type = "local_llm"
            
        ocr_lang = self.ocr_lang_combo.currentData() if model_type == "layoutlm" else None
        
        # NEW: Обработка LLM плагинов с автоматической загрузкой
        if model_type in ["cloud_llm", "local_llm"]:
            # Проверяем, загружен ли уже плагин
            if not hasattr(self, 'current_llm_plugin') or not self.current_llm_plugin:
                # Пытаемся автоматически загрузить плагин
                try:
                    if model_type == "cloud_llm":
                        provider_data = self.cloud_provider_selector.currentData()
                        model_data = self.cloud_model_selector.currentData()
                    else:  # local_llm
                        provider_data = self.local_provider_selector.currentData()
                        model_data = self.local_model_selector.currentData()
                    
                    if not provider_data or not model_data:
                        utils.show_error_message(
                            self, "Ошибка модели", "Выберите провайдера и модель для LLM плагина."
                        )
                        return
                    
                    # Автоматическая загрузка плагина
                    self.status_bar.showMessage(f"Загружается {model_type} плагин...")
                    success = self.auto_load_llm_plugin(model_type, provider_data, model_data)
                    
                    if not success:
                        utils.show_error_message(
                            self, "Ошибка LLM", 
                            f"Не удалось автоматически загрузить {model_type} плагин. Проверьте настройки API ключей."
                        )
                        return
                        
                except Exception as e:
                    utils.show_error_message(
                        self, "Ошибка загрузки", f"Ошибка автоматической загрузки плагина: {str(e)}"
                    )
                    return
            
            # Используем LLM плагин для обработки
            self.process_with_llm_plugin(input_path, is_folder)
            return
        
        # Получаем ID выбранной под-модели Gemini, если выбрана Gemini
        gemini_sub_model_id = None
        if model_type == 'gemini':
            gemini_sub_model_id = self.gemini_model_selector.currentData()
            # На всякий случай сохраним еще раз перед обработкой
            if gemini_sub_model_id:
                 settings_manager.set_value('Gemini', 'sub_model_id', gemini_sub_model_id)
            else: # Если вдруг currentData пустое, берем из настроек или дефолт
                 gemini_sub_model_id = settings_manager.get_string('Gemini', 'sub_model_id', 'models/gemini-1.5-flash-latest')
                 print(f"Предупреждение: Не удалось получить ID под-модели Gemini из ComboBox, используем: {gemini_sub_model_id}")

        # NEW: Получаем текущий промпт через ModelManager
        processor = self.model_manager.get_model(model_type)
        if not processor:
            utils.show_error_message(self, "Ошибка модели", f"Не удалось инициализировать процессор для модели {model_type}")
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Ошибка инициализации модели")
            return
        
        prompt_text = processor.get_full_prompt()
        if not prompt_text: # Добавим проверку, что промпт получен
            utils.show_error_message(self, "Ошибка промпта", f"Не удалось получить текст промпта для модели {model_type}")
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Ошибка получения промпта")
            return
        
        # Запускаем UI режим обработки
        self.start_processing_ui()
        self.set_processing_status(f"Обработка моделью {model_type.upper()}...")
        
        # Сохраняем выбранную модель в настройках
        settings_manager.set_active_model(model_type)
        
        # Очищаем таблицу перед запуском пакетной обработки
        if is_folder:
            self.results_table.setRowCount(0)
        
        print(f"Запуск обработки для файла: {input_path}, Модель: {model_type}, OCR: {ocr_lang}")
        
        # Создаем поток обработки, передавая model_manager
        self.processing_thread = ProcessingThread(
            model_type, input_path, ocr_lang, is_folder=is_folder, 
            model_manager=self.model_manager, # NEW: Передаем менеджер
            parent=self
        )
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished) # NEW: Переименовали слот
        self.processing_thread.partial_result_signal.connect(self.append_result_to_table) # NEW: Подключаем слот для частичных результатов
        self.processing_thread.error_signal.connect(self.show_processing_error)
        self.processing_thread.start()
    
    def on_process_button_clicked(self):
        """Обработчик клика на кнопку Обработать/Отменить."""
        if self.is_processing:
            # Отменяем обработку
            self.cancel_processing()
        else:
            # Запускаем обработку
            self.process_image()
    
    def start_processing_ui(self):
        """Запуск UI режима обработки."""
        self.is_processing = True
        self.process_button.setText("⛔ Отменить")
        self.process_button.setProperty("mode", "cancel")
        self.process_button.style().unpolish(self.process_button)
        self.process_button.style().polish(self.process_button)
        
        # Показываем прогресс-бар и статус
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_status_label.setVisible(True)
        self.process_status_label.setText("Инициализация...")
        
        # Сообщение в статус-баре
        self.status_bar.showMessage("Обработка...")
    
    def stop_processing_ui(self):
        """Остановка UI режима обработки."""
        self.is_processing = False
        
        # ИСПРАВЛЕНИЕ: Правильно восстанавливаем текст кнопки в зависимости от выбранного режима
        if self.current_folder_path:
            self.process_button.setText("🚀 Обработать папку")
        else:
            self.process_button.setText("🚀 Обработать")
            
        # ИСПРАВЛЕНИЕ: Активируем кнопку только если что-то выбрано
        has_selection = bool(self.current_image_path or self.current_folder_path)
        self.process_button.setEnabled(has_selection)
        
        self.process_button.setProperty("mode", "process")
        self.process_button.style().unpolish(self.process_button)
        self.process_button.style().polish(self.process_button)
        
        # Скрываем прогресс-бар и статус
        self.progress_bar.setVisible(False)
        if hasattr(self, 'process_status_label'):
            self.process_status_label.setVisible(False)
        
        self.processing_thread = None
    
    def cancel_processing(self):
        """Отмена текущей обработки."""
        if self.processing_thread and self.processing_thread.isRunning():
            # Принудительная остановка потока
            self.processing_thread.terminate()
            self.processing_thread.wait(3000)  # Ждем до 3 секунд
            
        self.stop_processing_ui()
        self.status_bar.showMessage("Обработка отменена")
    
    def update_progress(self, value):
        """Обновление индикатора прогресса."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
            
        # NEW: Обновляем прогресс в файловом списке для текущего файла
        if self.current_image_path and not self.current_folder_path:
            self.file_list_widget.update_file_progress(self.current_image_path, value, ProcessingStatus.PROCESSING)
            
    def set_processing_status(self, status: str):
        """Установка статуса обработки."""
        if hasattr(self, 'process_status_label'):
            self.process_status_label.setText(status)
    
    def show_results(self, results):
        """Отображение результатов обработки в таблице (для ОДИНОЧНОГО файла)."""
        try:
            print("🔍 DEBUG: show_results() вызван")
            print(f"🔍 DEBUG: Результаты присутствуют: {bool(results)}")
            
            # Этот метод теперь используется только для отображения результата одного файла
            # Сохраняем результаты для дальнейшего использования
            self.processing_thread.result = results # Сохраняем для совместимости с сохранением одиночного файла
            
            # Сохраняем данные для интеграции с Paperless
            self.current_invoice_data = results
            self.current_file_path = self.current_image_path
            
            # Очищаем таблицу
            self.results_table.setRowCount(0)
            
            # Заполняем таблицу данными
            if results:
                # Добавляем результаты в таблицу
                self.append_result_to_table(results)
            
            # Включаем кнопки сохранения
            self.save_button.setEnabled(True)
            if hasattr(self, 'save_action'): self.save_action.setEnabled(True)
            self.save_excel_button.setEnabled(True)
            if hasattr(self, 'save_excel_action'): self.save_excel_action.setEnabled(True)
            
            # NEW: Включаем кнопку предварительного просмотра
            print("🔍 DEBUG: Активируем кнопку просмотра (preview_button.setEnabled(True))")
            self.preview_button.setEnabled(True)
            print(f"🔍 DEBUG: Статус кнопки просмотра после активации: {self.preview_button.isEnabled()}")
            
            # NEW: Активируем кнопки новой системы плагинов
            if hasattr(self, 'validate_button'):
                self.validate_button.setEnabled(True)
            if hasattr(self, 'view_data_button'):
                self.view_data_button.setEnabled(True)
            if hasattr(self, 'plugin_export_button'):
                self.plugin_export_button.setEnabled(True)
            
            # Останавливаем UI режим обработки
            self.stop_processing_ui()
            self.status_bar.showMessage("Обработка завершена")
        except Exception as e:
            print(f"ОШИБКА в show_results: {e}")
            import traceback
            traceback.print_exc()
            self.show_processing_error(f"Ошибка отображения результатов: {str(e)}")
    
    def show_processing_error(self, error_msg):
        """Обработка ошибки при обработке изображения."""
        self.stop_processing_ui()
        self.status_bar.showMessage("Ошибка обработки")
        
        # ИСПРАВЛЕНИЕ: Обновляем статус файла в списке при ошибке
        if self.current_image_path:
            self.file_list_widget.set_file_error(self.current_image_path, error_msg)
        
        utils.show_error_message(
            self, "Ошибка обработки", f"Произошла ошибка: {error_msg}"
        )
    
    def save_results(self):
        """Сохранение результатов обработки с использованием ExportManager."""
        # Собираем данные из таблицы
        data = []
        headers = [self.results_table.horizontalHeaderItem(col).text() 
                   for col in range(self.results_table.columnCount())]
        
        for row in range(self.results_table.rowCount()):
            row_data = {}
            for col, header in enumerate(headers):
                item = self.results_table.item(row, col)
                row_data[header] = item.text() if item else ""
            data.append(row_data)
            
        if not data:
            utils.show_info_message(self, "Информация", "Нет данных для сохранения")
            return
            
        # Проверяем наличие ExportManager
        if not hasattr(self, 'export_manager') or not self.export_manager:
            # Fallback - используем старый метод
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "", 
                "CSV файлы (*.csv);;JSON файлы (*.json);;Все файлы (*.*)"
            )
            
            if file_path:
                # Определяем формат по расширению  
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    # CSV по умолчанию
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
                        writer.writeheader()
                        writer.writerows(data)
                        
                self.status_bar.showMessage(f"Результаты сохранены: {file_path}")
            return
            
        # Используем ExportManager
        from PyQt6.QtWidgets import QFileDialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "",
            self.export_manager.get_export_filters()
        )
        
        if file_path:
            # Определяем формат по выбранному фильтру или расширению
            format_type = None
            
            # Сначала пробуем определить по фильтру
            if selected_filter:
                for fmt, desc in self.export_manager.SUPPORTED_FORMATS.items():
                    if desc in selected_filter:
                        format_type = fmt
                        break
                        
            # Если не определили по фильтру, используем расширение
            if not format_type:
                ext = Path(file_path).suffix[1:].lower()
                format_type = ext if ext in self.export_manager.SUPPORTED_FORMATS else 'csv'
                
            # Добавляем расширение если его нет
            if not Path(file_path).suffix:
                file_path += f'.{format_type}'
                
            # Экспортируем
            success = self.export_manager.export_data(data, file_path, format_type)
            if success:
                self.status_bar.showMessage(f"Результаты сохранены: {file_path}")
            else:
                utils.show_error_message(self, "Ошибка", "Не удалось сохранить результаты")

    def show_model_management_dialog(self):
        """Показывает диалог управления моделями."""
        dialog = ModelManagementDialog(self)
        dialog.exec()
    
    def show_field_manager_dialog(self):
        """Показывает диалог управления полями таблицы."""
        try:
            dialog = FieldManagerDialog(self)
            dialog.fields_updated.connect(self.on_fields_updated)
            dialog.exec()
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка", f"Не удалось открыть диалог управления полями: {str(e)}"
            )
    
    def on_fields_updated(self):
        """Обработчик обновления полей таблицы."""
        try:
            # Обновляем заголовки таблицы результатов
            self.setup_results_table()
            
            # Автогенерируем промпты для всех моделей
            self.regenerate_all_prompts()
            
            # Можно добавить уведомление об успешном обновлении
            self.status_bar.showMessage("Поля таблицы и промпты обновлены", 3000)
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка", f"Ошибка при обновлении таблицы: {str(e)}"
            )
    
    def regenerate_all_prompts(self):
        """Регенерирует промпты для всех моделей на основе текущих полей таблицы."""
        try:
            print("🔄 Регенерация промптов для всех моделей...")
            
            # Генерируем промпты для всех облачных провайдеров
            cloud_providers = ['openai', 'anthropic', 'google', 'mistral', 'deepseek', 'xai']
            for provider in cloud_providers:
                try:
                    prompt = self.prompt_generator.generate_cloud_llm_prompt(provider)
                    self.prompt_generator.save_prompt_to_file(f"cloud_llm_{provider}", prompt)
                    print(f"✅ Промпт для {provider} обновлен")
                except Exception as e:
                    print(f"❌ Ошибка обновления промпта для {provider}: {e}")
            
            # Генерируем промпты для локальных провайдеров
            local_providers = ['ollama']
            for provider in local_providers:
                try:
                    prompt = self.prompt_generator.generate_local_llm_prompt(provider)
                    self.prompt_generator.save_prompt_to_file(f"local_llm_{provider}", prompt)
                    print(f"✅ Промпт для {provider} обновлен")
                except Exception as e:
                    print(f"❌ Ошибка обновления промпта для {provider}: {e}")
            
            # Генерируем промпт для Gemini
            try:
                prompt = self.prompt_generator.generate_gemini_prompt()
                self.prompt_generator.save_prompt_to_file("gemini", prompt)
                print("✅ Промпт для Gemini обновлен")
            except Exception as e:
                print(f"❌ Ошибка обновления промпта для Gemini: {e}")
            
            print("✅ Регенерация промптов завершена")
            
        except Exception as e:
            print(f"❌ Ошибка регенерации промптов: {e}")

    def show_llm_plugins_dialog(self):
        """Показывает диалог настройки LLM плагинов."""
        try:
            dialog = LLMProvidersDialog(self)
            dialog.providers_updated.connect(self.on_llm_providers_updated)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Обновляем список доступных LLM после изменения настроек
                self.populate_llm_models()
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка открытия диалога LLM провайдеров: {e}")
            print(f"Ошибка открытия диалога LLM провайдеров: {e}")
    
    def show_paperless_integration_dialog(self):
        """Показывает диалог интеграции с Paperless-NGX."""
        try:
            from .ui.paperless_integration_dialog import PaperlessIntegrationDialog
            
            dialog = PaperlessIntegrationDialog(self)
            dialog.exec()
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка открытия диалога Paperless: {e}")
            logging.error(f"Ошибка открытия диалога Paperless: {e}", exc_info=True)
    
    def on_llm_providers_updated(self):
        """Обработчик обновления настроек LLM провайдеров."""
        try:
            print("🔄 Обновление списков LLM провайдеров...")
            
            # Перезагружаем списки провайдеров и моделей
            self.populate_cloud_providers()
            self.populate_local_providers()
            
            # Обновляем статусы
            self.update_cloud_llm_status()
            self.update_local_llm_status()
            
            print("[OK] Списки LLM провайдеров обновлены")
            
        except Exception as e:
            print(f"Ошибка обновления LLM провайдеров: {e}")

    def show_plugin_editor(self):
        """Показывает диалог редактора плагинов."""
        try:
            from .plugins.plugin_editor_dialog import PluginManagerDialog
            from .plugins.advanced_plugin_manager import AdvancedPluginManager
            
            # Инициализируем продвинутый менеджер плагинов
            if not hasattr(self, 'advanced_plugin_manager'):
                self.advanced_plugin_manager = AdvancedPluginManager()
            
            dialog = PluginManagerDialog(parent=self)
            dialog.exec()
        except ImportError as e:
            utils.show_error_message(
                self, 
                "Ошибка",
                f"Не удалось открыть редактор плагинов: {e}"
            )
        except Exception as e:
            utils.show_error_message(
                self, 
                "Ошибка",
                f"Ошибка инициализации редактора плагинов: {e}"
            )

    def show_poppler_settings(self):
        """Открывает диалог настроек Poppler."""
        # dialog = PopplerSettingsDialog(settings_manager, self) 
        # dialog.exec()
        # ПОКА ПРОСТО ВЫЗЫВАЕМ ОБЩИЙ ДИАЛОГ УПРАВЛЕНИЯ МОДЕЛЯМИ
        self.show_model_management_dialog()

    def show_about_dialog(self):
        """Отображение диалога "О программе"."""
        about_text = (f"<h2>{app_config.APP_NAME} v{app_config.APP_VERSION}</h2>"
                   f"<p>© 2025 {app_config.ORGANIZATION_NAME}</p>"
                   "<p>Приложение для извлечения данных из счетов-фактур "
                   "с использованием моделей LayoutLMv3, Donut и Gemini.</p>")
        
        utils.show_info_message(self, "О программе", about_text)
    
    def show_help(self):
        """Отображение справки по использованию приложения."""
        help_text = (
            "<h2>Инструкция по использованию</h2>"
            "<ol>"
            "<li>Нажмите кнопку <b>Выбрать изображение/PDF</b> для загрузки файла.</li>"
            "<li>Выберите модель обработки: <b>LayoutLMv3</b>, <b>Donut</b> или <b>Gemini 2.0</b>.</li>"
            "<li>Нажмите кнопку <b>Обработать</b> для анализа изображения.</li>"
            "<li>Результаты будут отображены в правой части окна.</li>"
            "<li>Используйте кнопку <b>Сохранить результаты</b> для экспорта данных.</li>"
            "</ol>"
            "<p><b>Примечание:</b> Для использования модели Gemini 2.0 требуется указать API ключ Google в настройках.</p>"
        )
        
        utils.show_info_message(self, "Справка", help_text)
    
    def show_win7_info(self):
        """Отображение информации о совместимости с Windows 7."""
        win7_text = (
            "<h2>Информация о совместимости с Windows 7</h2>"
            "<p>Данное приложение разработано с учетом совместимости с Windows 7 (32-bit и 64-bit).</p>"
            "<p><b>Возможные проблемы:</b></p>"
            "<ul>"
            "<li>Tesseract OCR может требовать дополнительной настройки на Windows 7.</li>"
            "<li>При отсутствии актуальных обновлений Windows могут возникать проблемы с SSL/TLS при загрузке моделей.</li>"
            "<li>Производительность на Windows 7 может быть ниже, особенно на старом оборудовании.</li>"
            "</ul>"
            "<p>Если у вас возникают проблемы, проверьте, что Windows 7 обновлена "
            "до последней версии с установленными Service Pack 1 и обновлениями.</p>"
        )
        
        utils.show_info_message(self, "Windows 7", win7_text)
    
    def show_model_prompt(self, model_type):
        """
        Отображает диалог с полным запросом для выбранной модели.
        
        Args:
            model_type (str): Тип модели ('layoutlm', 'donut', 'gemini', 'cloud_llm', 'local_llm')
        """
        full_prompt = ""
        model_display_name = ""
        
        try:
            if model_type in ['layoutlm', 'donut', 'gemini', 'trocr']:
                # Старые модели через model_manager
                processor = self.model_manager.get_model(model_type)
                if processor:
                    full_prompt = processor.get_full_prompt()
                    model_display_name = model_type.upper()
                else:
                    utils.show_error_message(self, "Ошибка", f"Процессор {model_type} не найден")
                    return
                    
            elif model_type == 'cloud_llm':
                # Облачные LLM модели
                provider_data = self.cloud_provider_selector.currentData()
                model_data = self.cloud_model_selector.currentData()
                
                if not provider_data or not model_data:
                    utils.show_error_message(self, "Ошибка", "Выберите провайдера и модель")
                    return
                
                provider_name = provider_data.get('provider')
                model_name = model_data.get('model')
                model_display_name = f"{provider_name.upper()} - {model_name}"
                
                # Получаем промпт из настроек или создаем базовый
                prompt_key = f"cloud_llm_{provider_name}_prompt"
                full_prompt = settings_manager.get_setting(prompt_key, "")
                
                if not full_prompt:
                    # Создаем базовый промпт для облачной модели
                    full_prompt = self._create_default_llm_prompt(provider_name)
                    
            elif model_type == 'local_llm':
                # Локальные LLM модели
                provider_data = self.local_provider_selector.currentData()
                model_data = self.local_model_selector.currentData()
                
                if not provider_data or not model_data:
                    utils.show_error_message(self, "Ошибка", "Выберите провайдера и модель")
                    return
                
                provider_name = provider_data.get('provider')
                model_name = model_data.get('model')
                model_display_name = f"{provider_name.upper()} - {model_name}"
                
                # Получаем промпт из настроек или создаем базовый
                prompt_key = f"local_llm_{provider_name}_prompt"
                full_prompt = settings_manager.get_setting(prompt_key, "")
                
                if not full_prompt:
                    # Создаем базовый промпт для локальной модели
                    full_prompt = self._create_default_llm_prompt(provider_name)
                    
            else:
                utils.show_error_message(self, "Ошибка", f"Неподдерживаемый тип модели: {model_type}")
                return
                
        except Exception as e:
            utils.show_error_message(self, "Ошибка", f"Ошибка получения промпта: {str(e)}")
            return
        
        if not full_prompt:
            utils.show_error_message(self, "Ошибка", "Не удалось получить промпт для модели")
            return
        
        # Создаем диалог с текстом запроса
        prompt_dialog = QDialog(self)
        prompt_dialog.setWindowTitle(f"Промпт для {model_display_name}")
        prompt_dialog.resize(800, 700)
        
        layout = QVBoxLayout(prompt_dialog)
        
        # Информация о модели
        info_label = QLabel(f"<b>Модель:</b> {model_display_name}")
        info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(info_label)
        
        # Добавляем текстовое поле с запросом (с возможностью редактирования)
        text_edit = QTextEdit()
        text_edit.setPlainText(full_prompt)
        text_edit.setReadOnly(False)  # Разрешаем редактирование
        text_edit.setFont(QFont("Consolas", 10))  # Моноширинный шрифт для лучшей читаемости
        layout.addWidget(text_edit)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        # Кнопка сброса к умолчанию
        reset_button = QPushButton("Сбросить к умолчанию")
        reset_button.clicked.connect(lambda: self._reset_prompt_to_default(model_type, text_edit))
        
        # Кнопка сохранения промпта
        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(lambda: self.save_prompt(model_type, text_edit.toPlainText()))
        
        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(prompt_dialog.accept)
        
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # Отображаем диалог
        prompt_dialog.exec()
    
    def _create_default_llm_prompt(self, provider_name: str) -> str:
        """
        Создает базовый промпт для LLM провайдера используя генератор промптов.
        
        Args:
            provider_name: Имя провайдера (openai, anthropic, google, etc.)
            
        Returns:
            str: Базовый промпт для извлечения данных из инвойсов
        """
        try:
            # Используем новый генератор промптов
            return self.prompt_generator.generate_cloud_llm_prompt(provider_name)
        except Exception as e:
            print(f"❌ Ошибка генерации промпта для {provider_name}: {e}")
            # Fallback - базовый промпт
            return f"""Ты эксперт по анализу финансовых документов. Проанализируй предоставленное изображение счета-фактуры или инвойса и извлеки из него структурированные данные.

Извлеки следующие поля из документа:
- sender: Название компании-поставщика или продавца
- invoice_number: Номер счета, инвойса или фактуры
- invoice_date: Дата выставления счета или инвойса
- total: Общая сумма к оплате с учетом НДС
- amount_no_vat: Сумма без НДС
- vat_percent: Ставка НДС в процентах
- currency: Валюта платежа
- category: Категория товаров или услуг
- description: Описание товаров, услуг или содержимого документа
- note: Дополнительные примечания и комментарии

Требования к ответу:
1. Верни результат ТОЛЬКО в формате JSON
2. Используй точные ID полей как ключи
3. Если поле не найдено, используй значение "N/A"
4. Все суммы указывай числами без символов валют
5. Даты в формате DD.MM.YYYY
6. Будь точным и внимательным к деталям

Проанализируй документ и верни JSON с извлеченными данными:"""
    
    def _reset_prompt_to_default(self, model_type: str, text_edit):
        """
        Сбрасывает промпт к значению по умолчанию.
        
        Args:
            model_type: Тип модели
            text_edit: Виджет текстового редактора
        """
        try:
            if model_type in ['layoutlm', 'donut', 'gemini']:
                # Для старых моделей получаем из процессора
                processor = self.model_manager.get_model(model_type)
                if processor:
                    default_prompt = processor.get_full_prompt()
                    text_edit.setPlainText(default_prompt)
                    
            elif model_type == 'cloud_llm':
                provider_data = self.cloud_provider_selector.currentData()
                if provider_data:
                    provider_name = provider_data.get('provider')
                    default_prompt = self._create_default_llm_prompt(provider_name)
                    text_edit.setPlainText(default_prompt)
                    
            elif model_type == 'local_llm':
                provider_data = self.local_provider_selector.currentData()
                if provider_data:
                    provider_name = provider_data.get('provider')
                    default_prompt = self._create_default_llm_prompt(provider_name)
                    text_edit.setPlainText(default_prompt)
                    
        except Exception as e:
            utils.show_error_message(self, "Ошибка", f"Не удалось сбросить промпт: {str(e)}")
        
    def save_prompt(self, model_type, prompt_text):
        """
        Сохраняет промпт для указанной модели.
        
        Args:
            model_type (str): Тип модели ('layoutlm', 'donut', 'gemini', 'cloud_llm', 'local_llm')
            prompt_text (str): Текст промпта
        """
        # Проверяем, что текст не пустой
        if not prompt_text.strip():
            utils.show_warning_message(self, "Предупреждение", "Промпт не может быть пустым")
            return
        
        try:
            if model_type in ['layoutlm', 'donut']:
                # Старые модели - сохраняем в настройки как раньше
                prompt_key = f"{model_type}_prompt"
                settings_manager.set_setting(prompt_key, prompt_text)
                
                # Обновляем промпт в процессоре
                processor = self.model_manager.get_model(model_type)
                if processor:
                    processor.set_prompt(prompt_text)
                    
                model_display_name = model_type.upper()
                
            elif model_type == 'gemini':
                # Gemini - сохраняем в файл
                success = self.prompt_generator.save_prompt_to_file("gemini", prompt_text)
                if success:
                    # Также обновляем промпт в процессоре
                    processor = self.model_manager.get_model(model_type)
                    if processor:
                        processor.set_prompt(prompt_text)
                    model_display_name = "GEMINI"
                else:
                    utils.show_error_message(self, "Ошибка", "Не удалось сохранить промпт Gemini")
                    return
                
            elif model_type == 'cloud_llm':
                # Облачные LLM модели - сохраняем в файлы
                provider_data = self.cloud_provider_selector.currentData()
                if not provider_data:
                    utils.show_error_message(self, "Ошибка", "Провайдер не выбран")
                    return
                
                provider_name = provider_data.get('provider')
                success = self.prompt_generator.save_prompt_to_file(f"cloud_llm_{provider_name}", prompt_text)
                if success:
                    model_display_name = f"Cloud LLM ({provider_name.upper()})"
                else:
                    utils.show_error_message(self, "Ошибка", f"Не удалось сохранить промпт для {provider_name}")
                    return
                
            elif model_type == 'local_llm':
                # Локальные LLM модели - сохраняем в файлы
                provider_data = self.local_provider_selector.currentData()
                if not provider_data:
                    utils.show_error_message(self, "Ошибка", "Провайдер не выбран")
                    return
                
                provider_name = provider_data.get('provider')
                success = self.prompt_generator.save_prompt_to_file(f"local_llm_{provider_name}", prompt_text)
                if success:
                    model_display_name = f"Local LLM ({provider_name.upper()})"
                else:
                    utils.show_error_message(self, "Ошибка", f"Не удалось сохранить промпт для {provider_name}")
                    return
                
            else:
                utils.show_error_message(self, "Ошибка", f"Неподдерживаемый тип модели: {model_type}")
                return
            
            # Сохраняем настройки
            settings_manager.save_settings()
            
            # Выводим сообщение об успешном сохранении
            utils.show_info_message(self, "Сохранение промпта", f"Промпт для {model_display_name} успешно сохранен")
            
        except Exception as e:
            utils.show_error_message(self, "Ошибка", f"Ошибка сохранения промпта: {str(e)}")
    
    # NEW: Метод для обновления видимости селектора под-модели Gemini
    def update_gemini_selector_visibility(self):
        is_gemini_selected = self.gemini_radio.isChecked()
        self.gemini_sub_model_label.setVisible(is_gemini_selected)
        self.gemini_model_selector.setVisible(is_gemini_selected)
    
    # NEW: Обработчик изменения выбранной под-модели Gemini
    def on_gemini_sub_model_changed(self, index):
        selected_model_id = self.gemini_model_selector.itemData(index)
        if selected_model_id:
            settings_manager.set_value('Gemini', 'sub_model_id', selected_model_id)
            print(f"Выбрана под-модель Gemini: {selected_model_id}") # Для отладки 

    # NEW: Метод для заполнения списка моделей Gemini
    def populate_gemini_models(self):
        """Заполняет QComboBox `gemini_model_selector` списком моделей Gemini."""
        print("Заполнение списка моделей Gemini...")
        current_selection = self.gemini_model_selector.currentData() # Сохраняем текущий выбор
        self.gemini_model_selector.clear()

        models_to_load = []
        # Пытаемся загрузить из настроек
        models_json = settings_manager.get_string('[Gemini]', 'available_models_json', None)
        if models_json:
            try:
                saved_models = json.loads(models_json)
                if isinstance(saved_models, list) and saved_models:
                    # Проверяем, что в сохраненном списке есть нужные поля
                    valid_saved_models = [
                        m for m in saved_models 
                        if isinstance(m, dict) and m.get('id') and m.get('display_name')
                    ]
                    if valid_saved_models:
                        models_to_load = valid_saved_models
                        print(f"Загружен список моделей из настроек ({len(models_to_load)} моделей).")
                    else:
                         print("Сохраненный список моделей невалиден, используется дефолтный.")
            except json.JSONDecodeError:
                print("Ошибка декодирования JSON списка моделей из настроек.")

        # Если из настроек не загрузились или они невалидны, используем дефолтный список
        if not models_to_load:
            print("Используется дефолтный список моделей Gemini.")
            # NEW: Обновленный дефолтный список моделей (убраны 1.5, добавлен 2.0 Flash)
            models_to_load = [
                {'id': 'models/gemini-2.0-flash', 'display_name': '2.0 Flash'}, # Добавлена стабильная 2.0
                # Оставляем модели 2.5 Preview как самые новые
                {'id': 'models/gemini-2.5-flash-preview-04-17', 'display_name': '2.5 Flash Preview (04-17)'},
                {'id': 'models/gemini-2.5-pro-preview-05-06', 'display_name': '2.5 Pro Preview (05-06)'},
            ]
        
        # Заполняем комбобокс
        default_index = 0
        current_index_to_set = -1
        for i, model_info in enumerate(models_to_load):
            model_id = model_info.get('id')
            display_name_base = model_info.get('display_name', model_id) # Базовое имя
            
            if not model_id or not display_name_base:
                continue # Пропускаем невалидные записи

            # Добавляем информацию о тарифе/лимите/статусе к известным моделям
            display_text = display_name_base
            # NEW: Обновляем логику пометок для нового списка
            if '2.0-flash' in model_id:
                 display_text += " (Stable, Free Tier*)"
                 default_index = i # Делаем 2.0 Flash основной по умолчанию
            elif '2.5-flash-preview' in model_id:
                display_text += " (Preview, Free Tier*)" 
            elif '2.5-pro-preview' in model_id:
                 display_text += " (Preview, Paid Only*)" # Pro модели обычно платные
            # Для других моделей статус будет неизвестен

            self.gemini_model_selector.addItem(display_text, model_id)
            
            # Проверяем, совпадает ли с предыдущим выбором
            if model_id == current_selection:
                current_index_to_set = i

        # Восстанавливаем предыдущий выбор или ставим по умолчанию
        if current_index_to_set != -1:
            self.gemini_model_selector.setCurrentIndex(current_index_to_set)
        elif self.gemini_model_selector.count() > default_index:
             self.gemini_model_selector.setCurrentIndex(default_index)
        
        # ИСПРАВЛЕНИЕ: Активируем селектор после заполнения
        self.gemini_model_selector.setEnabled(True)
        print(f"✅ Gemini model selector активирован с {self.gemini_model_selector.count()} моделями")
        
        # Обновляем всплывающую подсказку
        self.gemini_model_selector.setToolTip(
            "Выберите модель Gemini. *Free Tier обычно имеет лимиты (e.g., 15 RPM).\n"
            "Paid tier требует привязки биллинга Google Cloud.\n"
            "Preview модели могут быть менее стабильны.\n"
            "Используйте кнопку в Расширенных настройках для обновления списка."
            )
        print("Список моделей Gemini заполнен.") 

    def populate_trocr_models(self):
        """Заполняет выпадающий список моделей TrOCR."""
        self.trocr_model_selector.clear()
        
        # Получаем список доступных моделей TrOCR
        trocr_models = [
            {
                'id': 'microsoft/trocr-base-printed',
                'name': 'Base Printed',
                'description': 'Базовая модель для печатного текста'
            },
            {
                'id': 'microsoft/trocr-base-handwritten',
                'name': 'Base Handwritten',
                'description': 'Базовая модель для рукописного текста'
            },
            {
                'id': 'microsoft/trocr-large-printed',
                'name': 'Large Printed',
                'description': 'Большая модель для печатного текста'
            },
            {
                'id': 'microsoft/trocr-large-handwritten',
                'name': 'Large Handwritten',
                'description': 'Большая модель для рукописного текста'
            }
        ]
        
        # Добавляем базовые модели
        for model in trocr_models:
            display_text = f"{model['name']} ({model['description']})"
            self.trocr_model_selector.addItem(display_text, model['id'])
        
        # Проверяем наличие дообученных моделей
        trained_models_path = app_config.TRAINED_MODELS_PATH
        if os.path.exists(trained_models_path):
            trained_models = []
            
            for d in os.listdir(trained_models_path):
                model_dir = os.path.join(trained_models_path, d)
                if os.path.isdir(model_dir) and d.startswith('trocr_'):
                    # Проверяем наличие финальной модели
                    final_model_path = os.path.join(model_dir, 'final_model')
                    if os.path.exists(final_model_path):
                        # Читаем метаданные для отображения качества
                        metadata_path = os.path.join(final_model_path, 'training_metadata.json')
                        quality_info = ""
                        if os.path.exists(metadata_path):
                            try:
                                import json
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    final_loss = metadata.get('final_loss', 0.0)
                                    if final_loss < 1.0:
                                        quality_info = " 🔥"
                                    elif final_loss < 2.0:
                                        quality_info = " ✅"
                                    elif final_loss < 4.0:
                                        quality_info = " 🟡"
                                    else:
                                        quality_info = " 🟠"
                            except:
                                pass
                        
                        trained_models.append({
                            'name': d,
                            'path': final_model_path,
                            'quality': quality_info,
                            'mtime': os.path.getmtime(model_dir)
                        })
            
            if trained_models:
                # Сортируем по дате (новые сверху)
                trained_models.sort(key=lambda x: x['mtime'], reverse=True)
                
                # Добавляем разделитель
                self.trocr_model_selector.insertSeparator(self.trocr_model_selector.count())
                
                # Добавляем дообученные модели
                for model_info in trained_models:
                    display_text = f"🎓 {model_info['name']}{model_info['quality']} (Дообученная)"
                    self.trocr_model_selector.addItem(display_text, model_info['path'])
        
        # Восстанавливаем последний выбор
        last_model = settings_manager.get_string('Models', 'trocr_model_id', 'microsoft/trocr-base-printed')
        for i in range(self.trocr_model_selector.count()):
            if self.trocr_model_selector.itemData(i) == last_model:
                self.trocr_model_selector.setCurrentIndex(i)
                break
        
        # ИСПРАВЛЕНИЕ: Активируем селектор после заполнения
        self.trocr_model_selector.setEnabled(True)
        print(f"✅ TrOCR model selector активирован с {self.trocr_model_selector.count()} моделями")
        
    def on_trocr_model_changed(self, index):
        """Обработчик изменения модели TrOCR."""
        if index < 0:
            return
            
        model_id = self.trocr_model_selector.currentData()
        if not model_id:
            return
            
        # Определяем тип модели (HuggingFace или кастомная)
        is_custom = not model_id.startswith('microsoft/')
        
        # Сохраняем настройки
        settings_manager.set_value('Models', 'trocr_model_id', model_id)
        settings_manager.set_value('Models', 'trocr_model_source', 'custom' if is_custom else 'huggingface')
        if is_custom:
            settings_manager.set_value('Models', 'custom_trocr_model_name', os.path.basename(model_id))
        
        # Обновляем статус
        if is_custom:
            self.trocr_status_label.setText(f"Дообученная: {os.path.basename(model_id)}")
            self.trocr_status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            model_name = self.trocr_model_selector.currentText().split(' (')[0]
            self.trocr_status_label.setText(f"HuggingFace: {model_name}")
            self.trocr_status_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        
        print(f"Выбрана модель TrOCR: {model_id} ({'custom' if is_custom else 'huggingface'})")

    # NEW: Метод для выбора папки
    def select_folder(self):
        """Открытие диалога выбора папки."""
        last_open_path = settings_manager.get_string('Interface', 'last_open_path', utils.get_documents_dir())
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выбрать папку со счетами", last_open_path
        )

        if not folder_path:
            return

        self.current_folder_path = folder_path
        self.current_image_path = None # Сбрасываем путь к файлу
        self.selected_path_label.setText(f"Выбрана папка: {folder_path}")
        self.image_label.setText("Папка выбрана. Нажмите \"Обработать\" для запуска.")
        self.image_label.setPixmap(QPixmap()) # Очищаем изображение
        self.process_button.setEnabled(True)
        settings_manager.save_interface_setting('last_open_path', folder_path) # Сохраняем путь

    # NEW: Слот для добавления строки результата в таблицу (для пакетной обработки)
    def append_result_to_table(self, result):
        """Добавляет одну строку с результатами в таблицу."""
        if not result:
            print("ОТЛАДКА: append_result_to_table вызван с пустым результатом")
            return

        print(f"ОТЛАДКА: append_result_to_table вызван с результатом: {result}")
        
        # ИСПРАВЛЕНИЕ: Проверяем, что таблица правильно настроена
        if self.results_table.columnCount() == 0:
            print("ПРЕДУПРЕЖДЕНИЕ: Таблица результатов не настроена, выполняем настройку")
            self.setup_results_table()

        row_position = self.results_table.rowCount()
        self.results_table.insertRow(row_position)

        # Создаем маппинг display_name -> column_index на основе заголовков таблицы
        column_mapping = {}
        for col in range(self.results_table.columnCount()):
            header_item = self.results_table.horizontalHeaderItem(col)
            if header_item:
                column_mapping[header_item.text()] = col

        print(f"ОТЛАДКА: Заголовки таблицы: {list(column_mapping.keys())}")

        # Создаем расширенное сопоставление полей для гибкого поиска
        field_aliases = self._create_field_aliases_mapping(column_mapping)
        
        # Заполняем данные по display_name или алиасам
        processed_fields = 0
        result_fields = [k for k in result.keys() if not k.startswith('_')]
        
        for field_name, value in result.items():
            # Пропускаем служебные поля
            if field_name.startswith('_'):
                continue
                
            column_index = None
            
            # Сначала пытаемся точное совпадение
            if field_name in column_mapping:
                column_index = column_mapping[field_name]
                print(f"ОТЛАДКА: Точное совпадение '{field_name}' -> колонка {column_index}")
            else:
                # Затем ищем по алиасам (нечувствительно к регистру)
                field_name_lower = field_name.lower()
                for alias, col_idx in field_aliases.items():
                    if field_name_lower == alias.lower():
                        column_index = col_idx
                        print(f"ОТЛАДКА: Найден алиас '{field_name}' -> '{alias}' -> колонка {column_index}")
                        break
            
            if column_index is not None:
                item = QTableWidgetItem(str(value))
                
                # Выравнивание для числовых колонок
                if any(word in field_name.lower() for word in ["amount", "total", "vat", "сумма", "ндс", "№", "номер", "%"]):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
                self.results_table.setItem(row_position, column_index, item)
                processed_fields += 1
                print(f"ОТЛАДКА: Добавлено поле '{field_name}' = '{value}' в колонку {column_index}")
            else:
                # Логируем неизвестные поля для отладки
                print(f"ПРЕДУПРЕЖДЕНИЕ: Неизвестное поле '{field_name}' со значением '{value}' не добавлено в таблицу")

        try:
            self.results_table.resizeRowsToContents()
            success_rate = processed_fields / len(result_fields) * 100 if result_fields else 0
            print(f"ОТЛАДКА: Добавлена строка в таблицу. Полей обработано: {processed_fields}/{len(result_fields)} ({success_rate:.1f}%)")
        except Exception as e:
            print(f"ОШИБКА при изменении размера строк таблицы: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_field_aliases_mapping(self, column_mapping):
        """Создает расширенное сопоставление полей с алиасами для гибкого поиска"""
        field_aliases = {}
        
        # Определяем алиасы для каждого типа поля - ЗНАЧИТЕЛЬНО РАСШИРЕННЫЙ СПИСОК
        field_patterns = {
            # Номер счета
            "№ счета": ["№ Invoice", "номер счета", "invoice_number", "счет №", "invoice number", "№счета", "invoice №", "invoice_id", "invoice no"],
            "№ Invoice": ["№ счета", "номер счета", "invoice_number", "счет №", "invoice number", "№счета", "invoice №", "invoice_id", "invoice no"],
            
            # НДС
            "% НДС": ["НДС %", "ндс %", "vat_rate", "tax_rate", "ставка ндс", "НДС%", "ндс%", "% ндс", "налоговая ставка", "VAT %", "vat %"],
            "VAT %": ["НДС %", "ндс %", "vat_rate", "tax_rate", "ставка ндс", "НДС%", "ндс%", "% ндс", "налоговая ставка", "% НДС"],
            "НДС %": ["VAT %", "ндс %", "vat_rate", "tax_rate", "ставка ндс", "НДС%", "ндс%", "% ндс", "налоговая ставка"],
            
            # Поставщик
            "Поставщик": ["Sender", "поставщик", "company", "supplier", "vendor", "организация", "название компании"],
            "Sender": ["Поставщик", "поставщик", "company", "supplier", "vendor", "организация", "название компании"],
            
            # Сумма с НДС (Total)
            "Сумма с НДС": ["Total", "total", "итого", "к оплате", "сумма с ндс", "total_amount", "amount", "всего", "общая сумма"],
            "Total": ["Сумма с НДС", "total", "итого", "к оплате", "сумма с ндс", "total_amount", "amount", "всего", "общая сумма"],
            
            # Сумма без НДС
            "Сумма без НДС": ["Amount (0% VAT)", "amount_no_vat", "net_amount", "сумма без ндс", "amount without vat", "сумма без налога"],
            "Amount (0% VAT)": ["Сумма без НДС", "amount_no_vat", "net_amount", "сумма без ндс", "amount without vat", "сумма без налога"],
            
            # Сумма НДС
            "Сумма НДС": ["VAT Amount", "vat_amount", "tax amount", "сумма ндс", "ндс", "налог"],
            "VAT Amount": ["Сумма НДС", "vat_amount", "tax amount", "сумма ндс", "ндс", "налог"],
            
            # Дата счета
            "Дата счета": ["Invoice Date", "invoice_date", "date", "дата", "invoice date"],
            "Invoice Date": ["Дата счета", "invoice_date", "date", "дата", "invoice date"],
            
            # Валюта
            "Валюта": ["Currency", "currency"],
            "Currency": ["Валюта", "currency"],
            
            # Категория
            "Категория": ["Category", "category"],
            "Category": ["Категория", "category"],
            
            # Описание/товары
            "Товары": ["Description", "description", "items", "услуги", "продукция", "наименование"],
            "Description": ["Товары", "description", "items", "услуги", "продукция", "наименование"],
            
            # ИНН
            "ИНН": ["INN", "inn", "tax_id", "supplier_inn", "инн поставщика"],
            "INN": ["ИНН", "inn", "tax_id", "supplier_inn", "инн поставщика"],
            "ИНН Поставщика": ["INN Поставщика", "инн поставщика", "inn", "tax_id", "supplier_inn"],
            "INN Поставщика": ["ИНН Поставщика", "инн поставщика", "inn", "tax_id", "supplier_inn"],
            
            # КПП
            "КПП": ["KPP", "kpp", "supplier_kpp", "кпп поставщика"],
            "KPP": ["КПП", "kpp", "supplier_kpp", "кпп поставщика"],
            "КПП Поставщика": ["KPP Поставщика", "кпп поставщика", "kpp", "supplier_kpp"],
            "KPP Поставщика": ["КПП Поставщика", "кпп поставщика", "kpp", "supplier_kpp"],
            
            # Адрес поставщика
            "Адрес Поставщика": ["адрес поставщика", "address", "supplier_address", "адрес"],
            
            # Покупатель
            "Покупатель": ["buyer", "customer", "заказчик"],
            
            # ИНН покупателя
            "ИНН Покупателя": ["инн покупателя", "buyer_inn", "customer_inn"],
            
            # КПП покупателя  
            "КПП Покупателя": ["кпп покупателя", "buyer_kpp", "customer_kpp"],
            
            # Адрес покупателя
            "Адрес Покупателя": ["адрес покупателя", "buyer_address", "customer_address"],
            
            # Дата оплаты
            "Дата Оплаты": ["дата оплаты", "payment_date", "due_date", "срок оплаты"],
            
            # Банковские реквизиты
            "Банк": ["bank", "банк"],
            "БИК": ["bik", "бик"],
            "Р/С": ["р/с", "расчетный счет", "account"],
            "К/С": ["к/с", "корреспондентский счет", "correspondent_account"],
            
            # Примечание
            "Примечание": ["Note", "note", "Комментарии", "комментарии", "comments", "комментарий", "comment", "замечания"],
            "Note": ["Примечание", "note", "Комментарии", "комментарии", "comments", "комментарий", "comment", "замечания"],
            "Комментарии": ["Note", "note", "Примечание", "комментарии", "comments", "комментарий", "comment", "замечания"],
            
            # Файл (для пакетной обработки)
            "Файл": ["file", "filename", "source_file"]
        }
        
        # Создаем обратное сопоставление: алиас -> column_index
        for column_name, column_index in column_mapping.items():
            # Добавляем само название колонки
            field_aliases[column_name] = column_index
            
            # Добавляем алиасы для этой колонки
            if column_name in field_patterns:
                for alias in field_patterns[column_name]:
                    field_aliases[alias] = column_index
        
        return field_aliases

    # NEW: Слот для обработки завершения всего процесса
    def processing_finished(self, result_or_none):
        """Вызывается, когда поток завершает обработку (файла или папки)."""
        print("🔍 DEBUG: processing_finished() вызван")
        print(f"🔍 DEBUG: result_or_none присутствует: {bool(result_or_none)}")
        print(f"🔍 DEBUG: current_folder_path: {self.current_folder_path}")
        print(f"🔍 DEBUG: current_image_path: {self.current_image_path}")
        
        self.progress_bar.setVisible(False)
        if self.current_folder_path: # Если обрабатывали папку
            self.status_bar.showMessage(f"Пакетная обработка папки {os.path.basename(self.current_folder_path)} завершена.")
            self.results_table.resizeColumnsToContents() # Подгоняем ширину колонок
            self.results_table.resizeRowsToContents()    # Подгоняем высоту строк
            # Сортируем по поставщику (колонка 0)
            self.results_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            # Включаем кнопки сохранения, если есть результаты
            if self.results_table.rowCount() > 0:
                self.save_button.setEnabled(True)
                if hasattr(self, 'save_action'): self.save_action.setEnabled(True)
                self.save_excel_button.setEnabled(True)
                if hasattr(self, 'save_excel_action'): self.save_excel_action.setEnabled(True)
                # NEW: Enable preview button
                self.preview_button.setEnabled(True)
            else:
                self.save_button.setEnabled(False)
                if hasattr(self, 'save_action'): self.save_action.setEnabled(False)
                self.save_excel_button.setEnabled(False)
                if hasattr(self, 'save_excel_action'): self.save_excel_action.setEnabled(False)
                # NEW: Disable preview button  
                self.preview_button.setEnabled(False)
        else: # Если обрабатывали один файл
            print("🔍 DEBUG: Обрабатывали один файл")
            # NEW: Обновляем статус файла в списке
            if self.current_image_path:
                if result_or_none:
                    print("🔍 DEBUG: Результат есть, обновляем статус файла как COMPLETED")
                    # Файл успешно обработан
                    self.file_list_widget.update_file_progress(self.current_image_path, 100, ProcessingStatus.COMPLETED)
                    self.update_file_processing_fields(self.current_image_path, result_or_none)
                else:
                    print("🔍 DEBUG: Результата нет, отмечаем ошибку обработки")
                    # Ошибка обработки файла
                    self.file_list_widget.set_file_error(self.current_image_path, "Ошибка обработки")
            
            # Используем старый метод для отображения одного результата
            if result_or_none:
                print("🔍 DEBUG: Вызываем show_results() для отображения результата")
                self.show_results(result_or_none)
            else:
                print("🔍 DEBUG: Результат None, отключаем кнопки")
                # Если результат None (например, ошибка в потоке, но не исключение)
                self.status_bar.showMessage("Ошибка обработки файла.")
                self.save_button.setEnabled(False)
                if hasattr(self, 'save_action'): self.save_action.setEnabled(False)
                self.save_excel_button.setEnabled(False)
                if hasattr(self, 'save_excel_action'): self.save_excel_action.setEnabled(False)
                # NEW: Disable preview button on error
                print("🔍 DEBUG: Отключаем кнопку просмотра из-за ошибки")
                self.preview_button.setEnabled(False)
    
    def save_excel(self):
        """Сохранение результатов в Excel с использованием ExportManager."""
        # Собираем данные из таблицы
        data = []
        headers = [self.results_table.horizontalHeaderItem(col).text() 
                   for col in range(self.results_table.columnCount())]
        
        for row in range(self.results_table.rowCount()):
            row_data = {}
            for col, header in enumerate(headers):
                item = self.results_table.item(row, col)
                row_data[header] = item.text() if item else ""
            data.append(row_data)
            
        if not data:
            utils.show_info_message(
                self, "Информация", "Нет данных для экспорта в Excel"
            )
            return
            
        # Проверяем наличие ExportManager
        if not hasattr(self, 'export_manager') or not self.export_manager:
            # Используем старый метод
            self._save_excel_legacy()
            return
            
        # Получаем путь для сохранения
        last_export_path = settings_manager.get_string('Interface', 'last_export_path', utils.get_documents_dir())
        
        # Определяем имя файла по умолчанию
        if self.current_folder_path:
            default_folder_name = os.path.basename(self.current_folder_path) or "batch_results"
            default_name = f"{default_folder_name}_результаты.xlsx"
        elif self.current_image_path:
            default_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_результаты.xlsx"
        else:
            default_name = "invoice_results.xlsx"
        
        file_path = utils.get_save_file_path(
            self, "Сохранить в Excel",
            os.path.join(last_export_path, default_name),
            "Excel файлы (*.xlsx)"
        )
        
        if file_path:
            # Убеждаемся, что есть расширение
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
                
            # Экспортируем
            try:
                success = self.export_manager.export_data(data, file_path, 'excel')
                if success:
                    self.status_bar.showMessage(f"Результаты сохранены в Excel: {file_path}")
                    # Сохраняем путь для следующего раза
                    settings_manager.save_interface_setting('last_export_path', os.path.dirname(file_path))
                    utils.show_info_message(
                        self, "Сохранение успешно", 
                        f"Результаты экспортированы в Excel-файл:\n{file_path}"
                    )
                else:
                    utils.show_error_message(
                        self, "Ошибка экспорта", "Не удалось экспортировать в Excel"
                    )
            except Exception as e:
                print(f"Ошибка при экспорте в Excel: {e}")
                # Пробуем старый метод как fallback
                self._save_excel_legacy()
            
    def _save_excel_legacy(self):
        """Сохранение результатов в формате Excel."""
        # NEW: Проверяем, есть ли строки в таблице
        if self.results_table.rowCount() == 0:
            utils.show_info_message(
                self, "Информация", "Нет результатов в таблице для экспорта в Excel. Сначала обработайте файл или папку."
            )
            return
            
        # Определяем имя файла по умолчанию в зависимости от режима (файл или папка)
        if self.current_folder_path:
            default_folder_name = os.path.basename(self.current_folder_path) or "batch_results"
            default_name = f"{default_folder_name}_результаты.xlsx"
        elif self.current_image_path:
             default_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_результаты.xlsx"
        else: # На всякий случай
             default_name = "results.xlsx"
        
        # Получаем последний сохраненный путь экспорта из настроек или используем каталог документов
        last_export_path = settings_manager.get_string('Interface', 'last_export_path', utils.get_documents_dir())
        
        file_path = utils.get_save_file_path(
            self, "Сохранить результаты в Excel", 
            os.path.join(last_export_path, default_name),
            "Excel файл (*.xlsx)"
        )
        
        if not file_path:
            return
        
        # Сохраняем директорию выбранного файла в настройках
        file_dir = os.path.dirname(file_path)
        settings_manager.save_interface_setting('last_export_path', file_dir)
        
        try:
            # Используем pandas для экспорта в Excel
            import pandas as pd
            
            # Собираем данные из ТАБЛИЦЫ для экспорта
            column_headers = []
            for col in range(self.results_table.columnCount()):
                header_item = self.results_table.horizontalHeaderItem(col)
                column_headers.append(header_item.text() if header_item else f"Column_{col+1}") # Запасной вариант, если заголовок пуст
            
            data = []
            for row in range(self.results_table.rowCount()):
                row_data = []
                for col in range(self.results_table.columnCount()):
                    item = self.results_table.item(row, col)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                data.append(row_data)
            
            df = pd.DataFrame(data, columns=column_headers)
            
            # Сохраняем DataFrame в Excel
            df.to_excel(file_path, index=False, sheet_name="Результаты анализа")
            
            self.status_bar.showMessage(f"Результаты сохранены в {file_path}")
            utils.show_info_message(
                self, "Сохранение успешно", f"Результаты экспортированы в Excel-файл {file_path}"
            )
        except ImportError: # NEW: Явно ловим ImportError для pandas
             utils.show_error_message(
                 self, "Отсутствует библиотека", 
                 "Для экспорта в Excel необходимо установить библиотеки pandas и openpyxl. "
                 "Используйте команду: pip install pandas openpyxl"
             )
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка сохранения", f"Не удалось сохранить результаты в Excel: {str(e)}"
            )
            
            # В случае отсутствия pandas, предложим установить (дублируется, но оставим на всякий случай)
            if "No module named 'pandas'" in str(e):
                 utils.show_info_message(
                     self, "Отсутствует pandas", 
                     "Для экспорта в Excel необходимо установить библиотеку pandas. "
                                      "Используйте команду: pip install pandas openpyxl"
             )
    
    def _on_storage_migration_completed(self, success: bool):
        """Обработчик завершения миграции системы хранения"""
        if success:
            logger.info("✅ Миграция системы хранения завершена успешно")
            self.set_processing_status("Система хранения обновлена")
            
            # Показываем уведомление пользователю
            QMessageBox.information(
                self, 
                "Обновление системы",
                "Система хранения настроек была обновлена.\n"
                "Новая система обеспечивает улучшенную производительность\n"
                "и надежность сохранения данных."
            )
        else:
            logger.error("❌ Миграция системы хранения не удалась")
            self.set_processing_status("Ошибка миграции системы хранения")
            
            QMessageBox.warning(
                self,
                "Ошибка обновления",
                "Не удалось обновить систему хранения настроек.\n"
                "Приложение продолжит работу в совместимом режиме."
            )

    def get_all_processed_documents(self):
        """Получает все обработанные документы из базы данных для массовой синхронизации"""
        try:
            # Получаем доступ к хранилищу
            storage = get_storage_integration()
            
            # Получаем все документы
            all_documents = storage.get_all_invoices()
            
            # Преобразуем в формат для синхронизации
            processed_docs = []
            for doc in all_documents:
                # Добавляем путь к файлу если он доступен
                doc_data = doc.copy()
                
                # Проверяем наличие пути к файлу в базе
                if 'file_path' in doc_data and doc_data['file_path']:
                    processed_docs.append(doc_data)
                else:
                    # Если пути нет, пропускаем документ
                    logger.warning(f"Документ {doc_data.get('id', 'unknown')} не содержит путь к файлу")
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Ошибка получения обработанных документов: {e}", exc_info=True)
            return []

    def closeEvent(self, event):
        """Обработка закрытия окна."""
        print("Начинаем корректное закрытие приложения...")
        
        # NEW: Создаем резервную копию настроек перед закрытием
        if self.backup_manager:
            try:
                self.backup_manager.backup_settings()
                print("Резервная копия настроек создана при закрытии приложения")
            except Exception as e:
                print(f"Ошибка создания резервной копии при закрытии: {e}")
        
        # Останавливаем обработку изображений
        if self.processing_thread and self.processing_thread.isRunning():
            print("Останавливаем поток обработки изображений...")
            self.processing_thread.quit()
            self.processing_thread.wait(3000)  # Ждем до 3 секунд
            
        # Останавливаем загрузку LLM моделей
        if self.llm_loading_thread and self.llm_loading_thread.isRunning():
            print("Останавливаем поток загрузки LLM...")
            self.llm_loading_thread.quit()
            self.llm_loading_thread.wait(3000)  # Ждем до 3 секунд
            
        # ВАЖНО: Закрываем диалог обучения
        if hasattr(self, 'training_dialog') and self.training_dialog:
            print("Закрываем диалог обучения...")
            self.training_dialog.close()
            
        # Ищем все экземпляры DataPreparator в системе и останавливаем их
        try:
            # Принудительно завершаем все QThread'ы
            from PyQt6.QtCore import QCoreApplication
            import threading
            
            print("Проверяем активные потоки...")
            active_threads = threading.active_count()
            print(f"Активных потоков: {active_threads}")
            
            # Устанавливаем флаг остановки для всех DataPreparator'ов
            import gc
            for obj in gc.get_objects():
                if hasattr(obj, '__class__') and 'DataPreparator' in obj.__class__.__name__:
                    if hasattr(obj, 'stop_requested'):
                        obj.stop_requested = True
                        print("Установлен флаг остановки для DataPreparator")
                    if hasattr(obj, 'stop'):
                        try:
                            obj.stop()
                            print("Вызван метод stop() для DataPreparator")
                        except (AttributeError, RuntimeError, Exception) as e:
                            # Ошибка при остановке DataPreparator - не критично
                            pass
            
            # NEW: Очищаем кэш перед закрытием
            if self.cache_manager:
                self.cache_manager.clear_expired()
                print("Очищен устаревший кэш")
                
            # NEW: Очищаем ресурсы оптимизированной системы хранения (Phase 3)
            if hasattr(self, 'storage_integration') and self.storage_integration:
                self.storage_integration.cleanup()
                print("Система хранения очищена")
                            
            # Даем время для остановки
            QCoreApplication.processEvents()
            
        except Exception as e:
            print(f"Ошибка при остановке фоновых потоков: {e}")
        
        # Очистка временных файлов при закрытии приложения
        try:
            self.temp_dir.cleanup()
            print("Временные файлы очищены")
        except (OSError, AttributeError, Exception) as e:
            # Ошибка при очистке временных файлов - не критично при закрытии
            pass
            
        print("Закрытие приложения завершено")
        super().closeEvent(event)

    def _open_training_dialog(self):
        # Получаем процессоры из model_manager
        ocr_processor = self.model_manager.get_ocr_processor()
        gemini_processor = self.model_manager.get_gemini_processor()

        # Убедимся, что процессоры успешно получены/инициализированы
        if not ocr_processor:
            self.status_bar.showMessage("Ошибка: OCR процессор не доступен.", 5000)
            return
        if not gemini_processor:
            self.status_bar.showMessage("Ошибка: Gemini процессор не доступен.", 5000)
            return

        # Создаем и показываем диалог обучения
        # Передаем модуль app_config напрямую
        training_dialog = TrainingDialog(
            app_config=app_config, # Все верно, не требует исправления
            ocr_processor=ocr_processor,
            gemini_processor=gemini_processor,
            parent=self
        )
        
        # Подключаем сигнал для обновления списка TrOCR моделей после обучения
        try:
            training_dialog.finished.connect(self._on_training_dialog_finished)
        except AttributeError:
            # Если сигнала нет, игнорируем
            pass
            
        training_dialog.exec()
        
    def _on_training_dialog_finished(self):
        """Обработчик завершения диалога обучения - обновляет списки моделей"""
        try:
            # Обновляем список TrOCR моделей для отображения новых дообученных моделей
            self.populate_trocr_models()
            print("✅ Список TrOCR моделей обновлен после завершения обучения")
        except Exception as e:
            print(f"❌ Ошибка обновления списка TrOCR моделей: {e}")
    
    def refresh_trained_models(self):
        """Публичный метод для обновления списков обученных моделей"""
        try:
            self.populate_trocr_models()
            self.status_bar.showMessage("Списки моделей обновлены", 3000)
            print("✅ Списки моделей принудительно обновлены")
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка обновления моделей: {e}", 5000)
            print(f"❌ Ошибка принудительного обновления моделей: {e}")

    def setup_results_table(self):
        """Настраивает таблицу результатов на основе полей из FieldManager."""
        try:
            # Получаем колонки из FieldManager
            columns = field_manager.get_table_columns()
            
            # Устанавливаем количество колонок
            self.results_table.setColumnCount(len(columns))
            
            # Устанавливаем заголовки колонок
            for i, column in enumerate(columns):
                self.results_table.setHorizontalHeaderItem(
                    i, QTableWidgetItem(column["name"])
                )
                
            # Настраиваем отображение таблицы - улучшенная адаптивная версия
            self.results_table.setAlternatingRowColors(True)
            self.results_table.verticalHeader().setVisible(False)
            self.results_table.setWordWrap(True)
            self.results_table.setMinimumHeight(200)  # Минимальная высота для предотвращения сжатия
            
            # Применяем адаптивную настройку размеров колонок
            self._configure_adaptive_table_columns(columns)
            
            # Сохраняем маппинг полей для последующего использования
            self.field_mapping = {column["id"]: i for i, column in enumerate(columns)}
            
            print(f"Таблица настроена с {len(columns)} колонками: {[c['name'] for c in columns]}")
            
        except Exception as e:
            print(f"Ошибка настройки таблицы результатов: {e}")
            # Fallback на старую логику
            self._setup_results_table_fallback()
    
    def _setup_results_table_fallback(self):
        """Fallback метод настройки таблицы результатов если FieldManager недоступен."""
        try:
            # Получаем поля из настроек (старая логика)
            fields = settings_manager.get_table_fields()
            
            # Получаем только видимые поля и сортируем их по порядку
            visible_fields = [field for field in fields if field.get("visible", True)]
            visible_fields = sorted(visible_fields, key=lambda f: f.get("order", 0))
            
            # Устанавливаем количество колонок
            self.results_table.setColumnCount(len(visible_fields))
            
            # Устанавливаем заголовки колонок
            for i, field in enumerate(visible_fields):
                self.results_table.setHorizontalHeaderItem(i, QTableWidgetItem(field.get("name", "")))
                
            # Настраиваем отображение таблицы - улучшенная адаптивная версия
            self.results_table.setAlternatingRowColors(True)
            self.results_table.verticalHeader().setVisible(False)
            self.results_table.setWordWrap(True)
            self.results_table.setMinimumHeight(200)
            
            # Применяем адаптивную настройку для fallback полей
            self._configure_adaptive_table_columns_fallback(visible_fields)
            
            # Сохраняем маппинг полей для последующего использования
            self.field_mapping = {field.get("id", ""): i for i, field in enumerate(visible_fields)}
            
            print(f"Fallback таблица настроена с {len(visible_fields)} колонками")
            
        except Exception as e:
            print(f"Ошибка fallback настройки таблицы: {e}")
            # Если и fallback не работает, создаем базовую таблицу
            self._setup_basic_results_table()
    
    def _setup_basic_results_table(self):
        """Создает базовую таблицу результатов с минимальными колонками."""
        basic_columns = [
            {"id": "sender", "name": "Sender"},
            {"id": "invoice_number", "name": "№ Invoice"},
            {"id": "invoice_date", "name": "Invoice Date"},
            {"id": "total", "name": "Total"},
            {"id": "note", "name": "Note"}
        ]
        
        self.results_table.setColumnCount(len(basic_columns))
        
        for i, column in enumerate(basic_columns):
            self.results_table.setHorizontalHeaderItem(
                i, QTableWidgetItem(column["name"])
            )
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setWordWrap(True)
        self.results_table.setMinimumHeight(200)
        
        # Применяем адаптивную настройку для базовых колонок
        self._configure_adaptive_table_columns(basic_columns)
        
        self.field_mapping = {column["id"]: i for i, column in enumerate(basic_columns)}
        
        print("Базовая таблица создана с 5 колонками")

    def _configure_adaptive_table_columns(self, columns):
        """Настраивает адаптивные размеры колонок таблицы на основе типа данных."""
        try:
            header = self.results_table.horizontalHeader()
            
            for i, column in enumerate(columns):
                column_name = column.get("name", "").lower()
                column_id = column.get("id", "").lower()
                
                # Узкие колонки для коротких полей (номера, даты, проценты)
                if any(word in column_name or word in column_id for word in 
                       ['№', 'number', 'дата', 'date', '%', 'ндс', 'vat', 'инн', 'inn', 'кпп', 'kpp']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
                    self.results_table.setColumnWidth(i, 120)
                
                # Средние колонки для числовых полей (суммы)
                elif any(word in column_name or word in column_id for word in 
                         ['сумма', 'amount', 'total', 'валюта', 'currency', 'цена', 'price']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                    self.results_table.setColumnWidth(i, 140)
                
                # Широкие колонки для текстовых полей (названия, описания, адреса)
                elif any(word in column_name or word in column_id for word in 
                         ['поставщик', 'supplier', 'sender', 'покупатель', 'buyer', 'customer',
                          'название', 'name', 'описание', 'description', 'товары', 'items',
                          'адрес', 'address', 'примечание', 'note', 'комментарий', 'comment']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
                
                # По умолчанию - интерактивный режим с умеренной шириной
                else:
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                    self.results_table.setColumnWidth(i, 100)
            
            # Автоматическая подгонка высоты строк под содержимое
            self.results_table.resizeRowsToContents()
            
        except Exception as e:
            print(f"Ошибка адаптивной настройки колонок: {e}")
            # Fallback к базовой настройке
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)

    def _configure_adaptive_table_columns_fallback(self, visible_fields):
        """Настраивает адаптивные размеры колонок для fallback полей."""
        try:
            header = self.results_table.horizontalHeader()
            
            for i, field in enumerate(visible_fields):
                field_name = field.get("name", "").lower()
                field_id = field.get("id", "").lower()
                
                # Аналогичная логика как в основном методе
                if any(word in field_name or word in field_id for word in 
                       ['№', 'number', 'дата', 'date', '%', 'ндс', 'vat', 'инн', 'inn']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
                    self.results_table.setColumnWidth(i, 120)
                elif any(word in field_name or word in field_id for word in 
                         ['сумма', 'amount', 'total', 'валюта', 'currency']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                    self.results_table.setColumnWidth(i, 140)
                elif any(word in field_name or word in field_id for word in 
                         ['поставщик', 'supplier', 'sender', 'описание', 'description', 'примечание', 'note']):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
                else:
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                    self.results_table.setColumnWidth(i, 100)
            
            self.results_table.resizeRowsToContents()
            
        except Exception as e:
            print(f"Ошибка fallback настройки колонок: {e}")
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)

    # Метод show_table_fields_dialog удален - заменен на show_field_manager_dialog
    # Метод on_table_fields_changed удален - заменен на on_fields_updated
    
    # NEW: LLM Plugin Integration Methods
    
    def populate_llm_models(self):
        """Заполняет список всех доступных LLM моделей."""
        try:
            # Заполняем локальные модели (только Ollama)
            self.populate_local_models()
            
            # Заполняем облачные модели (все кроме Ollama)
            self.populate_cloud_models()
            
        except Exception as e:
            print(f"❌ Ошибка загрузки LLM моделей: {e}")
    
    def populate_local_models(self):
        """Заполняет список локальных моделей (Ollama)."""
        try:
            # Старый метод - больше не используется, так как llm_model_selector удален
            # Заполнение теперь происходит через populate_local_providers
            print("[INFO] populate_local_models() - метод устарел, используется populate_local_providers()")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки локальных моделей: {e}")
            # llm_model_selector больше не существует
    
    def populate_cloud_models(self):
        """Заполняет список облачных моделей (все кроме Ollama)."""
        try:
            # Старый метод - больше не используется, так как cloud_llm_selector удален
            # Заполнение теперь происходит через populate_cloud_providers
            print("[INFO] populate_cloud_models() - метод устарел, используется populate_cloud_providers()")

            
        except Exception as e:
            print(f"❌ Ошибка загрузки облачных моделей: {e}")
            # cloud_llm_selector больше не существует
    
    def on_llm_model_changed(self):
        """Обработчик изменения выбранного LLM плагина"""
        # Этот метод больше не используется после реструктуризации
        pass
    
    def on_cloud_llm_changed(self):
        """Обработчик изменения выбранной облачной LLM модели"""
        self.update_cloud_llm_status()
    
    # Удален метод update_llm_status - больше не используется
    
    def update_cloud_llm_status(self):
        """Обновляет статус облачной LLM."""
        try:
            provider_data = self.cloud_provider_selector.currentData()
            model_data = self.cloud_model_selector.currentData()
            
            if not provider_data:
                self.cloud_llm_status_label.setText("Статус: Не выбран провайдер")
                return
            
            if not model_data:
                self.cloud_llm_status_label.setText("Статус: Не выбрана модель")
                return
            
            if not provider_data.get('configured', False):
                self.cloud_llm_status_label.setText("Статус: ⚙️ Требуется настройка API")
                return
            
            # Если все готово к загрузке
            provider_name = provider_data.get('provider')
            model_name = model_data.get('model')
            pricing = model_data.get('pricing', '')
            
            status_text = f"Готов: {provider_name}/{model_name} {pricing}"
            self.cloud_llm_status_label.setText(status_text)
            
        except Exception as e:
            print(f"Ошибка обновления статуса облачной LLM: {e}")
            self.cloud_llm_status_label.setText("Статус: Ошибка")

    def update_local_llm_status(self):
        """Обновляет статус локальной LLM."""
        try:
            provider_data = self.local_provider_selector.currentData()
            model_data = self.local_model_selector.currentData()
            
            if not provider_data:
                self.local_llm_status_label.setText("Статус: Не выбран провайдер")
                return
            
            if not model_data:
                self.local_llm_status_label.setText("Статус: Не выбрана модель")
                return
            
            if not provider_data.get('available', False):
                self.local_llm_status_label.setText("Статус: ❌ Провайдер недоступен")
                return
            
            # Если все готово к загрузке
            provider_name = provider_data.get('provider')
            model_name = model_data.get('model')
            size_info = model_data.get('size', '')
            
            status_text = f"Готов: {provider_name}/{model_name} {size_info}"
            self.local_llm_status_label.setText(status_text)
            
        except Exception as e:
            print(f"Ошибка обновления статуса локальной LLM: {e}")
            self.local_llm_status_label.setText("Статус: Ошибка")

    def get_selected_llm_plugin(self):
        """Возвращает настроенный экземпляр выбранного LLM плагина."""
        try:
            # Проверяем, какой тип LLM выбран
            if self.cloud_llm_radio.isChecked():
                provider_data = self.cloud_provider_selector.currentData()
                model_data = self.cloud_model_selector.currentData()
            elif self.local_llm_radio.isChecked():
                provider_data = self.local_provider_selector.currentData()
                model_data = self.local_model_selector.currentData()
            else:
                return None
            
            if not provider_data or not model_data:
                return None
            
            provider_name = provider_data.get('provider')
            model_name = model_data.get('model')
            config = provider_data.get('config')
            
            # Получаем настройки провайдера
            llm_settings = settings_manager.get_setting('llm_providers', {})
            provider_settings = llm_settings.get(provider_name, {})
            
            # Получаем API ключ если требуется
            api_key = None
            if config.requires_api_key:
                api_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
                if not api_key:
                    print(f"❌ API ключ для {provider_name} не найден")
                    return None
            
            # Создаем экземпляр универсального плагина
            from .plugins.models.universal_llm_plugin import UniversalLLMPlugin
            
            # Дополнительные параметры
            plugin_kwargs = {
                'generation_config': {
                    'temperature': provider_settings.get('temperature', 0.1),
                    'max_tokens': provider_settings.get('max_tokens', 4096),
                    'top_p': provider_settings.get('top_p', 0.9),
                }
            }
            
            # Для Ollama добавляем base_url
            if provider_name == "ollama":
                plugin_kwargs['base_url'] = provider_settings.get('base_url', 'http://localhost:11434')
            
            # Создаем плагин
            plugin = UniversalLLMPlugin(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key,
                **plugin_kwargs
            )
            
            return plugin
            
        except Exception as e:
            print(f"❌ Ошибка создания LLM плагина: {e}")
            return None
    
    def load_selected_llm(self):
        """Загружает выбранный LLM плагин - УСТАРЕЛ, используется load_selected_local_llm()"""
        print("[INFO] load_selected_llm() - метод устарел, используется load_selected_local_llm() или load_selected_cloud_llm()")
        return
    
    def load_selected_cloud_llm(self):
        """Загружает выбранный облачный LLM плагин"""
        provider_data = self.cloud_provider_selector.currentData()
        model_data = self.cloud_model_selector.currentData()
        
        if not provider_data or not model_data:
            return
            
        # Проверяем, не идет ли уже загрузка
        if self.llm_loading_thread and self.llm_loading_thread.isRunning():
            utils.show_info_message(self, "Загрузка", "LLM модель уже загружается...")
            return
        
        # Формируем данные для загрузки
        load_data = {
            'provider': provider_data.get('provider'),
            'model': model_data.get('model'), 
            'config': provider_data.get('config')
        }
        
        # Создаем поток для загрузки модели
        self.llm_loading_thread = LLMLoadingThread(self.plugin_manager, load_data)
        self.llm_loading_thread.loading_started.connect(self.on_cloud_llm_loading_started)
        self.llm_loading_thread.loading_finished.connect(self.on_cloud_llm_loading_finished)
        self.llm_loading_thread.loading_error.connect(self.on_cloud_llm_loading_error)
        
        self.llm_loading_thread.start()
    
    def on_llm_loading_started(self, plugin_id: str):
        """Обработчик начала загрузки LLM - УСТАРЕЛ после реструктуризации"""
        # Этот метод больше не используется, т.к. llm_status_label и llm_load_button удалены
        pass
    
    def on_cloud_llm_loading_started(self, plugin_id: str):
        """Обработчик начала загрузки облачной LLM"""
        self.cloud_llm_status_label.setText("Статус: 🔄 Загружается...")
        QApplication.processEvents()

    def on_llm_loading_finished(self, plugin_id: str, plugin_instance):
        """Обработчик завершения загрузки LLM - УСТАРЕЛ после реструктуризации"""
        # Этот метод больше не используется, т.к. update_llm_status удален
        pass
    
    def on_cloud_llm_loading_finished(self, plugin_id: str, plugin_instance):
        """Обработчик завершения загрузки облачной LLM"""
        self.current_llm_plugin = plugin_instance
        self.update_cloud_llm_status()
        
        plugin_info = self.plugin_manager.get_plugin_info(plugin_id)
        plugin_name = plugin_info.get('name', plugin_id) if plugin_info else plugin_id
        
        utils.show_info_message(
            self, 
            "Облачная LLM Загружена", 
            f"Облачный LLM плагин {plugin_name} успешно загружен!"
        )

    def on_llm_loading_error(self, plugin_id: str, error_message: str):
        """Обработчик ошибки загрузки LLM - УСТАРЕЛ после реструктуризации"""
        # Этот метод больше не используется, т.к. llm_status_label и llm_load_button удалены
        pass
    
    def on_cloud_llm_loading_error(self, plugin_id: str, error_message: str):
        """Обработчик ошибки загрузки облачной LLM"""
        self.cloud_llm_status_label.setText("Статус: ❌ Ошибка загрузки")
        
        utils.show_error_message(
            self,
            "Ошибка загрузки облачной LLM",
            f"Не удалось загрузить облачный LLM плагин:\n{error_message}"
        )
    
    def _map_llm_plugin_fields(self, result):
        """
        Универсальный маппинг полей для всех LLM плагинов.
        Преобразует названия полей в стандартные названия колонок таблицы.
        
        Args:
            result (dict): Результат от LLM плагина
            
        Returns:
            dict: Результат с нормализованными названиями полей
        """
        if not result:
            return result
        
        # Маппинг различных вариантов названий полей в стандартные названия колонок
        field_mapping = {
            # Поставщик
            'поставщик': 'Поставщик',
            'sender': 'Поставщик',
            'supplier': 'Поставщик',  
            'vendor': 'Поставщик',
            'company': 'Поставщик',
            'название компании': 'Поставщик',
            'организация': 'Поставщик',
            
            # Номер счета
            '№ счета': '№ счета',
            'номер счета': '№ счета', 
            'invoice_number': '№ счета',
            'invoice number': '№ счета',
            'счет №': '№ счета',
            'invoice_id': '№ счета',
            'invoice no': '№ счета',
            '№счета': '№ счета',
            'invoice №': '№ счета',
            '№ invoice': '№ счета',
            
            # Дата счета
            'дата счета': 'Дата счета',
            'invoice_date': 'Дата счета',
            'invoice date': 'Дата счета',
            'дата': 'Дата счета',
            'date': 'Дата счета',
            
            # Категория
            'категория': 'Category',
            'category': 'Category',
            
            # Товары/Описание
            'товары': 'Description',
            'description': 'Description',
            'услуги': 'Description',
            'items': 'Description',
            'продукция': 'Description',
            'наименование': 'Description',
            
            # Сумма без НДС
            'сумма без ндс': 'Amount (0% VAT)',
            'amount_no_vat': 'Amount (0% VAT)',
            'amount (0% vat)': 'Amount (0% VAT)',
            'net_amount': 'Amount (0% VAT)',
            'amount without vat': 'Amount (0% VAT)',
            'сумма без налога': 'Amount (0% VAT)',
            
            # НДС %
            '% ндс': '% НДС',
            'ндс %': '% НДС',
            'vat %': '% НДС',
            'tax_rate': '% НДС',
            'vat_rate': '% НДС',
            'ставка ндс': '% НДС',
            'налоговая ставка': '% НДС',
            
            # Сумма НДС
            'сумма ндс': 'VAT Amount',
            'ндс': 'VAT Amount',
            'vat amount': 'VAT Amount',
            'vat_amount': 'VAT Amount',
            'tax amount': 'VAT Amount',
            'налог': 'VAT Amount',
            
            # Сумма с НДС (Итого)
            'сумма с ндс': 'Сумма с НДС',
            'total': 'Сумма с НДС',
            'итого': 'Сумма с НДС',
            'total_amount': 'Сумма с НДС',
            'amount': 'Сумма с НДС',
            'к оплате': 'Сумма с НДС',
            'всего': 'Сумма с НДС',
            'общая сумма': 'Сумма с НДС',
            
            # Валюта
            'валюта': 'Currency',
            'currency': 'Currency',
            
            # ИНН
            'инн поставщика': 'INN Поставщика',
            'инн': 'INN Поставщика',
            'inn': 'INN Поставщика',
            'tax_id': 'INN Поставщика',
            'supplier_inn': 'INN Поставщика',
            
            # КПП  
            'кпп поставщика': 'KPP Поставщика',
            'кпп': 'KPP Поставщика',
            'kpp': 'KPP Поставщика',
            'supplier_kpp': 'KPP Поставщика',
            
            # Адрес
            'адрес поставщика': 'Адрес Поставщика',
            'адрес': 'Адрес Поставщика',
            'address': 'Адрес Поставщика',
            'supplier_address': 'Адрес Поставщика',
            
            # Покупатель
            'покупатель': 'Покупатель',
            'buyer': 'Покупатель',
            'customer': 'Покупатель',
            'заказчик': 'Покупатель',
            
            # ИНН покупателя
            'инн покупателя': 'INN Покупателя',
            'buyer_inn': 'INN Покупателя',
            'customer_inn': 'INN Покупателя',
            
            # КПП покупателя
            'кпп покупателя': 'KPP Покупателя', 
            'buyer_kpp': 'KPP Покупателя',
            'customer_kpp': 'KPP Покупателя',
            
            # Адрес покупателя
            'адрес покупателя': 'Адрес Покупателя',
            'buyer_address': 'Адрес Покупателя',
            'customer_address': 'Адрес Покупателя',
            
            # Дата оплаты
            'дата оплаты': 'Дата Оплаты',
            'payment_date': 'Дата Оплаты',
            'due_date': 'Дата Оплаты',
            'срок оплаты': 'Дата Оплаты',
            
            # Банковские реквизиты
            'банк': 'Банк',
            'bank': 'Банк',
            'бик': 'БИК',
            'bik': 'БИК',
            'р/с': 'Р/С',
            'расчетный счет': 'Р/С',
            'account': 'Р/С',
            'к/с': 'К/С',
            'корреспондентский счет': 'К/С',
            'correspondent_account': 'К/С',
            
            # Комментарии
            'комментарии': 'Примечание',
            'комментарий': 'Примечание',
            'примечание': 'Примечание',
            'note': 'Примечание',
            'notes': 'Примечание',
            'comment': 'Примечание',
            'comments': 'Примечание',
            'замечания': 'Примечание',
        }
        
        mapped_result = {}
        
        for field_name, value in result.items():
            # Пропускаем пустые значения
            if value is None or value == "":
                continue
            
            # Пропускаем служебные поля
            if field_name.startswith('_') or field_name in ['source_image', 'processed_at', 'raw_response']:
                mapped_result[field_name] = value
                continue
            
            # Ищем соответствие в маппинге (регистронезависимо)
            field_name_lower = field_name.lower().strip()
            mapped_field = field_mapping.get(field_name_lower, field_name)
            
            # Добавляем в результат
            mapped_result[mapped_field] = value
        
        print(f"ОТЛАДКА: LLM маппинг полей завершен. Исходные поля: {list(result.keys())}")
        print(f"ОТЛАДКА: Результирующие поля: {list(mapped_result.keys())}")
        
        return mapped_result

    def process_with_llm_plugin(self, input_path, is_folder):
        """Обработка файла/папки с использованием LLM плагина"""
        try:
            self.status_bar.showMessage("Обработка LLM плагином...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            if is_folder:
                self.results_table.setRowCount(0)
                # Для папки - обрабатываем все файлы
                self.process_folder_with_llm(input_path)
            else:
                # Для одного файла
                result = self.current_llm_plugin.extract_invoice_data(input_path)
                if result:
                    # Проверяем, не содержит ли результат ошибку
                    if "error" in result:
                        error_msg = result.get("error", "Неизвестная ошибка")
                        self.show_processing_error(f"Ошибка LLM плагина: {error_msg}")
                        return
                    
                    # Применяем маппинг полей для LLM плагинов
                    result = self._map_llm_plugin_fields(result)
                    
                    # Создаем фиктивный объект processing_thread для совместимости
                    class FakeThread:
                        def __init__(self, result):
                            self.result = result
                    
                    self.processing_thread = FakeThread(result)
                    self.show_results(result)
                else:
                    self.show_processing_error("LLM плагин не вернул результат")
            
        except Exception as e:
            self.show_processing_error(f"Ошибка LLM плагина: {str(e)}")
    
    def process_folder_with_llm(self, folder_path):
        """Обработка папки с файлами через LLM плагин"""
        try:
            supported_files = []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and utils.is_supported_format(file_path):
                    supported_files.append(file_path)
            
            if not supported_files:
                utils.show_info_message(self, "Информация", "В папке не найдено поддерживаемых файлов")
                return
            
            total_files = len(supported_files)
            processed = 0
            
            for file_path in supported_files:
                try:
                    # Обновляем прогресс
                    progress = int((processed / total_files) * 100)
                    self.progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
                    # Обрабатываем файл через LLM плагин
                    result = self.current_llm_plugin.process_image(file_path)
                    
                    if result:
                        # Проверяем, не содержит ли результат ошибку
                        if "error" in result:
                            error_msg = result.get("error", "Неизвестная ошибка")
                            error_result = {
                                'Файл': os.path.basename(file_path),
                                'Ошибка': f'LLM ошибка: {error_msg}'
                            }
                            self.append_result_to_table(error_result)
                        else:
                            # Применяем маппинг полей для LLM плагинов
                            result = self._map_llm_plugin_fields(result)
                            
                            # Добавляем имя файла к результату
                            result['Файл'] = os.path.basename(file_path)
                            self.append_result_to_table(result)
                    else:
                        # Добавляем запись о пустом результате
                        error_result = {
                            'Файл': os.path.basename(file_path),
                            'Ошибка': 'Пустой результат от LLM'
                        }
                        self.append_result_to_table(error_result)
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"Ошибка обработки файла {file_path}: {e}")
                    # Добавляем запись об ошибке
                    error_result = {
                        'Файл': os.path.basename(file_path),
                        'Ошибка': str(e)
                    }
                    self.append_result_to_table(error_result)
                    processed += 1
            
            # Завершение обработки
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"Обработано файлов: {processed}")
            
            # Включаем кнопки сохранения
            self.save_button.setEnabled(True)
            if hasattr(self, 'save_action'): 
                self.save_action.setEnabled(True)
            self.save_excel_button.setEnabled(True)
            if hasattr(self, 'save_excel_action'): 
                self.save_excel_action.setEnabled(True)
            # NEW: Enable preview button
            self.preview_button.setEnabled(True)
            
        except Exception as e:
            self.show_processing_error(f"Ошибка обработки папки: {str(e)}")

    # NEW: Preview Dialog Integration Methods
    
    def show_preview_dialog(self):
        """Показывает диалог предварительного просмотра результатов"""
        print("🔍 DEBUG: show_preview_dialog() вызван")
        try:
            print("🔍 DEBUG: Начинаем определение данных для preview")
            print(f"🔍 DEBUG: current_folder_path = {self.current_folder_path}")
            print(f"🔍 DEBUG: results_table.rowCount() = {self.results_table.rowCount()}")
            print(f"🔍 DEBUG: current_image_path = {getattr(self, 'current_image_path', 'None')}")
            
            # Определяем данные для preview
            preview_data = None
            model_type = "unknown"
            file_path = ""
            
            # Получаем данные в зависимости от режима обработки
            if self.current_folder_path:
                # Пакетная обработка - собираем данные из таблицы
                if self.results_table.rowCount() == 0:
                    utils.show_info_message(
                        self, "Информация", 
                        "Нет результатов для предварительного просмотра. Сначала обработайте файлы."
                    )
                    return
                
                # Собираем все результаты из таблицы для batch preview
                batch_results = []
                headers = [self.results_table.horizontalHeaderItem(col).text() 
                          for col in range(self.results_table.columnCount())]
                
                for row in range(self.results_table.rowCount()):
                    row_data = {}
                    for col, header in enumerate(headers):
                        item = self.results_table.item(row, col)
                        row_data[header] = item.text() if item else ""
                    batch_results.append(row_data)
                
                preview_data = {"batch_results": batch_results}
                print(f"🔍 DEBUG: Собрано {len(batch_results)} записей для пакетной обработки")
                file_path = self.current_folder_path
                
            else:
                # Одиночная обработка - проверяем наличие данных в таблице
                if self.results_table.rowCount() == 0:
                    utils.show_info_message(
                        self, "Информация", 
                        "Нет результатов для предварительного просмотра. Сначала обработайте файл."
                    )
                    return
                
                # Собираем данные из таблицы результатов (одна строка, много колонок)
                preview_data = {}
                if self.results_table.rowCount() > 0:
                    row = 0  # Берем первую (и единственную) строку
                    for col in range(self.results_table.columnCount()):
                        header_item = self.results_table.horizontalHeaderItem(col)
                        cell_item = self.results_table.item(row, col)
                        if header_item and cell_item:
                            field_name = header_item.text()
                            field_value = cell_item.text()
                            if field_value:  # Только непустые поля
                                preview_data[field_name] = field_value
                
                print(f"🔍 DEBUG: Собрано {len(preview_data)} полей из таблицы: {list(preview_data.keys())}")
                file_path = self.current_image_path or ""
            
            # Определяем активную модель
            if self.layoutlm_radio.isChecked():
                model_type = "LayoutLMv3"
            elif self.donut_radio.isChecked():
                model_type = "Donut"
            elif self.gemini_radio.isChecked():
                model_type = "Gemini 2.0"
            elif self.cloud_llm_radio.isChecked():
                model_type = f"Cloud LLM ({self.cloud_model_selector.currentText()})"
            elif self.local_llm_radio.isChecked():
                model_type = f"Local LLM ({self.local_model_selector.currentText()})"
            
            print(f"🔍 DEBUG: Создаем PreviewDialog с данными: model_type={model_type}, file_path={file_path}")
            print(f"🔍 DEBUG: preview_data тип: {type(preview_data)}")
            
            # Создаем и показываем диалог preview с дополнительной обработкой ошибок
            try:
                print("🔍 DEBUG: Импортируем PreviewDialog...")
                from .ui.preview_dialog import PreviewDialog
                print("🔍 DEBUG: PreviewDialog импортирован успешно")
                
                print("🔍 DEBUG: Создаем экземпляр PreviewDialog...")
                preview_dialog = PreviewDialog(
                    results=preview_data,
                    model_type=model_type,
                    file_path=file_path,
                    parent=self
                )
                print("🔍 DEBUG: PreviewDialog создан успешно")
            except Exception as create_error:
                print(f"🔍 DEBUG: ОШИБКА создания PreviewDialog: {create_error}")
                import traceback
                traceback.print_exc()
                raise create_error
            
            # Подключаем сигналы
            preview_dialog.results_edited.connect(self.on_preview_results_edited)
            preview_dialog.export_requested.connect(self.on_preview_export_requested)
            print("🔍 DEBUG: Сигналы подключены")
            
            # Показываем диалог
            print("🔍 DEBUG: Показываем диалог")
            result = preview_dialog.exec()
            print(f"🔍 DEBUG: Диалог закрыт с результатом: {result}")
            
            if result == QDialog.DialogCode.Accepted:
                self.status_bar.showMessage("Изменения из предварительного просмотра применены")
            
        except Exception as e:
            print(f"🔍 DEBUG: ОШИБКА в show_preview_dialog(): {e}")
            import traceback
            traceback.print_exc()
            utils.show_error_message(
                self,
                "Ошибка предварительного просмотра",
                f"Не удалось открыть предварительный просмотр:\n{str(e)}"
            )
    
    def on_preview_results_edited(self, edited_results):
        """Обработчик изменения результатов в preview dialog"""
        try:
            if self.current_folder_path:
                # Batch mode - обновляем таблицу
                # Для пакетного режима нужна более сложная логика обновления
                self.status_bar.showMessage("Результаты пакетной обработки обновлены")
            else:
                # Single mode - обновляем таблицу результатов
                # Обновляем отображение в таблице
                self.show_results(edited_results)
                
                # Также обновляем processing_thread.result если он существует
                if hasattr(self, 'processing_thread') and self.processing_thread and \
                   hasattr(self.processing_thread, 'result'):
                    self.processing_thread.result = edited_results
                
                self.status_bar.showMessage("Результаты обновлены из предварительного просмотра")
                
        except Exception as e:
            utils.show_error_message(
                self,
                "Ошибка обновления",
                f"Не удалось применить изменения из предварительного просмотра:\n{str(e)}"
            )
    
    def on_preview_export_requested(self, results, format_type):
        """Обработчик запроса экспорта из preview dialog"""
        try:
            # Определяем формат для экспорта
            if "Excel" in format_type:
                self.save_excel()
            else:
                self.save_results()
                
        except Exception as e:
            utils.show_error_message(
                self,
                "Ошибка экспорта",
                f"Не удалось выполнить экспорт из предварительного просмотра:\n{str(e)}"
            )

    # NEW: Template Designer Integration Methods
    
    def show_template_designer(self):
        """Показывает дизайнер шаблонов экспорта"""
        print("DEBUG: Начинаем запуск дизайнера шаблонов...")
        
        # Проверяем критические зависимости перед запуском
        try:
            print("DEBUG: Проверяем наличие PyQt6...")
            import PyQt6
            from PyQt6.QtWidgets import QDialog, QComboBox, QTextBrowser
            print("DEBUG: ✅ PyQt6 и основные виджеты доступны")
        except ImportError as pyqt_error:
            print(f"DEBUG: ❌ PyQt6 недоступен: {pyqt_error}")
            utils.show_error_message(
                self, "Отсутствуют зависимости", 
                f"PyQt6 не установлен или поврежден.\n\nДля установки выполните:\npip install PyQt6\n\nПодробности: {str(pyqt_error)}"
            )
            return
        
        try:
            print("DEBUG: Проверяем импорт ExportTemplateDesigner...")
            from .ui.export_template_designer import ExportTemplateDesigner
            print("DEBUG: ✅ Импорт ExportTemplateDesigner успешен")
            
            # Подготавливаем данные для дизайнера
            current_results = None
            print("DEBUG: Подготавливаем данные для дизайнера...")
            
            # Получаем данные в зависимости от режима обработки
            if self.current_folder_path:
                print("DEBUG: Режим пакетной обработки")
                # Пакетная обработка - собираем данные из таблицы
                if self.results_table.rowCount() > 0:
                    batch_results = []
                    for row in range(self.results_table.rowCount()):
                        row_data = {}
                        for col in range(self.results_table.columnCount()):
                            header = self.results_table.horizontalHeaderItem(col)
                            item = self.results_table.item(row, col)
                            if header and item:
                                row_data[header.text()] = item.text()
                        batch_results.append(row_data)
                    current_results = {"batch_results": batch_results}
                    print(f"DEBUG: Собрано {len(batch_results)} записей для пакетной обработки")
            else:
                print("DEBUG: Режим одиночной обработки")
                # Одиночная обработка - берем данные из таблицы
                if self.results_table.rowCount() > 0:
                    current_results = {}
                    for row in range(self.results_table.rowCount()):
                        key_item = self.results_table.item(row, 0)
                        value_item = self.results_table.item(row, 1)
                        if key_item and value_item:
                            current_results[key_item.text()] = value_item.text()
                    print(f"DEBUG: Собрано {len(current_results)} полей для одиночной обработки")
            
            print("DEBUG: Создаем экземпляр дизайнера шаблонов...")
            # Создаем и показываем диалог дизайнера шаблонов
            designer = ExportTemplateDesigner(current_results=current_results, parent=self)
            print("DEBUG: ✅ ExportTemplateDesigner создан успешно")
            
            print("DEBUG: Подключаем сигналы...")
            # Подключаем сигналы
            designer.template_applied.connect(self.on_template_applied)
            print("DEBUG: ✅ Сигналы подключены")
            
            print("DEBUG: Показываем диалог...")
            # Показываем диалог
            result = designer.exec()
            print(f"DEBUG: Диалог закрыт с результатом: {result}")
            
            if result == designer.DialogCode.Accepted:
                utils.show_info_message(
                    self, "Дизайнер шаблонов", 
                    "Шаблон успешно создан/настроен"
                )
                print("DEBUG: ✅ Шаблон применен успешно")
                
        except ImportError as e:
            print(f"DEBUG: ❌ Ошибка импорта: {e}")
            print("DEBUG: Пытаемся использовать упрощенный дизайнер...")
            
            try:
                from .ui.simple_template_designer import SimpleTemplateDesigner
                print("DEBUG: ✅ SimpleTemplateDesigner импортирован")
                
                # Подготавливаем те же данные
                current_results = None
                if self.current_folder_path:
                    if self.results_table.rowCount() > 0:
                        batch_results = []
                        for row in range(self.results_table.rowCount()):
                            row_data = {}
                            for col in range(self.results_table.columnCount()):
                                header = self.results_table.horizontalHeaderItem(col)
                                item = self.results_table.item(row, col)
                                if header and item:
                                    row_data[header.text()] = item.text()
                            batch_results.append(row_data)
                        current_results = {"batch_results": batch_results}
                else:
                    if self.results_table.rowCount() > 0:
                        current_results = {}
                        for row in range(self.results_table.rowCount()):
                            key_item = self.results_table.item(row, 0)
                            value_item = self.results_table.item(row, 1)
                            if key_item and value_item:
                                current_results[key_item.text()] = value_item.text()
                
                simple_designer = SimpleTemplateDesigner(current_results=current_results, parent=self)
                simple_designer.template_applied.connect(self.on_template_applied)
                
                result = simple_designer.exec()
                print(f"DEBUG: ✅ Упрощенный дизайнер работал, результат: {result}")
                
                if result == simple_designer.DialogCode.Accepted:
                    utils.show_info_message(
                        self, "Дизайнер шаблонов", 
                        "Шаблон успешно создан/настроен (упрощенная версия)"
                    )
                    
            except Exception as fallback_error:
                print(f"DEBUG: ❌ Ошибка fallback дизайнера: {fallback_error}")
                import traceback
                traceback.print_exc()
                utils.show_error_message(
                    self, "Ошибка импорта", 
                    f"Не удалось запустить ни основной, ни упрощенный дизайнер шаблонов:\n\nОсновная ошибка: {str(e)}\nFallback ошибка: {str(fallback_error)}\n\nВозможно, отсутствуют зависимости PyQt6."
                )
                
        except Exception as e:
            print(f"DEBUG: ❌ Общая ошибка: {e}")
            import traceback
            traceback.print_exc()
            
            # Пытаемся fallback дизайнер
            try:
                print("DEBUG: Пытаемся запустить упрощенный дизайнер после общей ошибки...")
                from .ui.simple_template_designer import SimpleTemplateDesigner
                
                # Подготавливаем данные (упрощенно)
                current_results = {}
                if self.results_table.rowCount() > 0:
                    for row in range(self.results_table.rowCount()):
                        key_item = self.results_table.item(row, 0)
                        value_item = self.results_table.item(row, 1)
                        if key_item and value_item:
                            current_results[key_item.text()] = value_item.text()
                
                simple_designer = SimpleTemplateDesigner(current_results=current_results, parent=self)
                result = simple_designer.exec()
                
                if result == simple_designer.DialogCode.Accepted:
                    utils.show_info_message(
                        self, "Дизайнер шаблонов", 
                        "Шаблон создан с помощью упрощенного дизайнера"
                    )
                
            except Exception as final_error:
                print(f"DEBUG: ❌ Финальная ошибка: {final_error}")
                utils.show_error_message(
                    self, "Критическая ошибка дизайнера шаблонов", 
                    f"Не удалось запустить дизайнер шаблонов:\n\nОсновная ошибка: {str(e)}\nFallback ошибка: {str(final_error)}\n\nПодробности в консоли отладки.\n\nВозможные причины:\n• Отсутствуют зависимости\n• Поврежденные файлы модулей\n• Проблемы с PyQt6"
                )
    
    def on_template_applied(self, template_data):
        """Обработчик применения шаблона"""
        try:
            # Здесь можно добавить логику применения шаблона к текущим результатам
            # Например, сохранить шаблон как предпочтительный для экспорта
            
            template_name = template_data.get("name", "Неизвестный шаблон")
            utils.show_info_message(
                self, "Шаблон применён", 
                f"Шаблон '{template_name}' был применён к текущим результатам"
            )
            
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка применения шаблона", 
                f"Не удалось применить шаблон:\n{str(e)}"
            )

    def check_api_provider_status(self, provider_name: str, config) -> tuple[bool, str]:
        """
        Проверяет реальный статус API провайдера.
        
        Returns:
            tuple[bool, str]: (is_working, status_message)
        """
        try:
            # Специальная обработка для Ollama
            if provider_name == "ollama":
                return self.check_ollama_status()
            
            # Для облачных провайдеров проверяем API ключ
            if not config.requires_api_key:
                return True, "OK"
            
            api_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
            if not api_key:
                return False, "CFG"  # Не настроен
            
            # Проверяем реальное подключение
            try:
                from .plugins.models.universal_llm_plugin import UniversalLLMPlugin
                
                print(f"🔍 Probing connection to {provider_name}...")
                
                # Создаем временный плагин для тестирования
                test_plugin = UniversalLLMPlugin(
                    provider_name=provider_name,
                    model_name=config.default_model,
                    api_key=api_key
                )
                
                # Пытаемся загрузить и протестировать
                success = test_plugin.load_model()
                if success:
                    print(f"✅ Connection to {provider_name} verified successfully")
                    return True, "OK"
                else:
                    print(f"❌ {provider_name}: Connection error")
                    return False, "ERR"
                    
            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ {provider_name}: {str(e)}")
                
                if "timeout" in error_msg or "timed out" in error_msg:
                    return False, "TMO"  # Timeout
                elif "unauthorized" in error_msg or "invalid api key" in error_msg:
                    return False, "KEY"  # Неверный ключ
                elif "credit balance" in error_msg or "insufficient funds" in error_msg:
                    return False, "BAL"  # Недостаточно средств
                elif "rate limit" in error_msg:
                    return False, "LMT"  # Лимит превышен
                else:
                    return False, "ERR"  # Общая ошибка
                    
        except Exception as e:
            print(f"❌ Ошибка проверки статуса {provider_name}: {e}")
            return False, "ERR"
    
    def check_ollama_status(self) -> tuple[bool, str]:
        """Специальная проверка статуса Ollama."""
        try:
            import requests
            
            print(f"🔍 Probing connection to ollama...")
            
            # Проверяем доступность сервера
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                if available_models:
                    print(f"✅ Connection to ollama verified successfully")
                    print(f"📋 Available models: {len(available_models)} found")
                    return True, "OK"
                else:
                    print(f"❌ ollama: No models available")
                    return False, "CFG"  # Нет моделей
            else:
                print(f"❌ ollama: Server returned {response.status_code}")
                return False, "ERR"
                
        except requests.exceptions.ConnectionError:
            print(f"❌ ollama: Connection refused - server not running")
            return False, "ERR"
        except requests.exceptions.Timeout:
            print(f"❌ ollama: Connection timeout")
            return False, "TMO"
        except Exception as e:
            print(f"❌ ollama: {str(e)}")
            return False, "ERR"
    
    def get_status_icon_with_description(self, status_code: str) -> tuple[str, str]:
        """
        Возвращает иконку и описание для статуса провайдера.
        
        Returns:
            tuple[str, str]: (icon, description)
        """
        status_map = {
            "OK": ("[✅]", "Работает корректно"),
            "CFG": ("[⚙️]", "Требуется настройка API ключа"),
            "KEY": ("[🔑]", "Неверный API ключ"),
            "BAL": ("[💰]", "Недостаточно средств"),
            "TMO": ("[⏱️]", "Превышено время ожидания"),
            "LMT": ("[🚫]", "Превышен лимит запросов"),
            "ERR": ("[❌]", "Ошибка подключения")
        }
        return status_map.get(status_code, ("[❓]", "Неизвестная ошибка"))

    def populate_cloud_providers(self):
        """Заполняет список облачных провайдеров, поддерживающих файлы."""
        try:
            self.cloud_provider_selector.clear()
            self.cloud_provider_selector.addItem("Выберите провайдера...", None)
            
            from .plugins.base_llm_plugin import LLM_PROVIDERS
            
            providers_added = 0
            llm_settings = settings_manager.get_setting('llm_providers', {})
            
            # Добавляем только облачных провайдеров, которые поддерживают файлы (все кроме ollama)
            for provider_name, config in LLM_PROVIDERS.items():
                if provider_name != "ollama" and config.supports_files:  # Пропускаем локальные и не поддерживающие файлы
                    # Проверяем реальный статус провайдера
                    is_working, status_code = self.check_api_provider_status(provider_name, config)
                    status_icon, status_description = self.get_status_icon_with_description(status_code)
                    
                    # Формируем название с индикатором
                    files_icon = "📄" if config.supports_files else ""
                    display_name = f"{status_icon} {config.display_name} {files_icon}".strip()
                    
                    self.cloud_provider_selector.addItem(display_name, {
                        'provider': provider_name,
                        'config': config,
                        'configured': is_working,
                        'status_code': status_code,
                        'status_description': status_description
                    })
                    providers_added += 1
                    
                    # Выводим подробную информацию о статусе
                    print(f"🔍 {provider_name}: {status_description}")
            
            print(f"[OK] Загружено {providers_added} облачных провайдеров (только с поддержкой файлов)")
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки облачных провайдеров: {e}")
            self.cloud_provider_selector.clear()
            self.cloud_provider_selector.addItem("Ошибка загрузки", None)

    def populate_local_providers(self):
        """Заполняет список локальных провайдеров, поддерживающих файлы."""
        try:
            self.local_provider_selector.clear()
            self.local_provider_selector.addItem("Выберите провайдера...", None)
            
            from .plugins.base_llm_plugin import LLM_PROVIDERS
            
            providers_added = 0
            
            # Добавляем только локальных провайдеров, которые поддерживают файлы (пока только ollama)
            for provider_name, config in LLM_PROVIDERS.items():
                if provider_name == "ollama" and config.supports_files:  # Только локальные, поддерживающие файлы
                    # Проверяем реальный статус Ollama
                    is_working, status_code = self.check_api_provider_status(provider_name, config)
                    status_icon, status_description = self.get_status_icon_with_description(status_code)
                    
                    # Формируем название с индикатором
                    files_icon = "📄" if config.supports_files else ""
                    display_name = f"{status_icon} {config.display_name} {files_icon}".strip()
                    
                    self.local_provider_selector.addItem(display_name, {
                        'provider': provider_name,
                        'config': config,
                        'available': is_working,
                        'status_code': status_code,
                        'status_description': status_description
                    })
                    providers_added += 1
                    
                    # Выводим подробную информацию о статусе
                    print(f"🔍 {provider_name}: {status_description}")
            
            print(f"[OK] Загружено {providers_added} локальных провайдеров (только с поддержкой файлов)")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки локальных провайдеров: {e}")
            self.local_provider_selector.clear()
            self.local_provider_selector.addItem("Ошибка загрузки", None)

    def check_ollama_availability(self) -> bool:
        """Проверяет доступность Ollama сервера."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.RequestException, requests.ConnectionError, requests.Timeout, ImportError) as e:
            # Ollama недоступен - это нормально, если не установлен
            return False

    def on_cloud_provider_changed(self):
        """Обработчик изменения выбранного облачного провайдера."""
        current_data = self.cloud_provider_selector.currentData()
        
        if not current_data:
            self.cloud_model_selector.clear()
            self.cloud_model_selector.addItem("Сначала выберите провайдера", None)
            self.cloud_model_selector.setEnabled(False)
            self.update_cloud_llm_status()
            return
            
        provider_name = current_data.get('provider')
        config = current_data.get('config')
        is_configured = current_data.get('configured', False)
        
        print(f"🔄 Провайдер изменен на: {provider_name} (настроен: {is_configured})")
        
        # Заполняем модели
        self.populate_cloud_models_for_provider(provider_name, config, is_configured)
        self.update_cloud_llm_status()
        
        # Обновляем видимость компонентов после изменения провайдера
        self.update_model_component_visibility()

    def on_local_provider_changed(self):
        """Обработчик изменения выбранного локального провайдера."""
        current_data = self.local_provider_selector.currentData()
        
        if not current_data:
            self.local_model_selector.clear()
            self.local_model_selector.addItem("Сначала выберите провайдера", None)
            self.local_model_selector.setEnabled(False)
            self.update_local_llm_status()
            return
            
        provider_name = current_data.get('provider')
        config = current_data.get('config')
        is_available = current_data.get('available', False)
        
        print(f"🔄 Локальный провайдер изменен на: {provider_name} (доступен: {is_available})")
        
        # Заполняем модели
        self.populate_local_models_for_provider(provider_name, config, is_available)
        self.update_local_llm_status()
        
        # Обновляем видимость компонентов после изменения провайдера
        self.update_model_component_visibility()

    def populate_cloud_models_for_provider(self, provider_name: str, config, is_configured: bool):
        """Заполняет модели для выбранного облачного провайдера."""
        try:
            self.cloud_model_selector.clear()
            
            if not is_configured:
                self.cloud_model_selector.addItem("⚙️ Настройте API ключ в настройках", None)
                self.cloud_model_selector.setEnabled(False)
                return
            
            # Получаем настройки провайдера
            llm_settings = settings_manager.get_setting('llm_providers', {})
            provider_settings = llm_settings.get(provider_name, {})
            selected_model = provider_settings.get('model', config.default_model)
            
            models_added = 0
            for model in config.models:
                # Добавляем информацию о платности и возможностях
                pricing_info = self.get_model_pricing_info(provider_name, model)
                vision_support = "👁️" if config.supports_vision else ""
                files_support = "📄" if config.supports_files else ""
                
                display_name = f"{model} {pricing_info} {vision_support} {files_support}".strip()
                
                self.cloud_model_selector.addItem(display_name, {
                    'provider': provider_name,
                    'model': model,
                    'config': config,
                    'pricing': pricing_info,
                    'supports_files': config.supports_files
                })
                models_added += 1
                
                # Выбираем сохраненную модель
                if model == selected_model:
                    self.cloud_model_selector.setCurrentIndex(models_added - 1)
            
            self.cloud_model_selector.setEnabled(models_added > 0)
            print(f"[OK] Загружено {models_added} моделей для {config.display_name} (все поддерживают файлы)")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей для {provider_name}: {e}")
            self.cloud_model_selector.clear()
            self.cloud_model_selector.addItem("Ошибка загрузки моделей", None)
            self.cloud_model_selector.setEnabled(False)

    def populate_local_models_for_provider(self, provider_name: str, config, is_available: bool):
        """Заполняет модели для выбранного локального провайдера."""
        try:
            self.local_model_selector.clear()
            
            if not is_available:
                self.local_model_selector.addItem("❌ Ollama недоступен", None)
                self.local_model_selector.setEnabled(False)
                return
            
            if provider_name == "ollama":
                # Получаем доступные модели из Ollama
                available_models = self.get_ollama_models()
                
                if not available_models:
                    self.local_model_selector.addItem("📥 Загрузите модели в Ollama", None)
                    self.local_model_selector.setEnabled(False)
                    return
                
                # Получаем настройки провайдера
                llm_settings = settings_manager.get_setting('llm_providers', {})
                provider_settings = llm_settings.get(provider_name, {})
                selected_model = provider_settings.get('model', config.default_model)
                
                models_added = 0
                for model in available_models:
                    # Проверяем поддержку файлов (через vision модели)
                    model_supports_vision = "vision" in model.lower()
                    model_supports_files = model_supports_vision and config.supports_files
                    
                    # Фильтруем только модели, которые поддерживают файлы
                    if not model_supports_files:
                        continue
                    
                    # Добавляем информацию о модели
                    vision_support = "👁️" if model_supports_vision else ""
                    files_support = "📄" if model_supports_files else ""
                    size_info = self.get_model_size_info(model)
                    
                    display_name = f"{model} {size_info} {vision_support} {files_support}".strip()
                    
                    self.local_model_selector.addItem(display_name, {
                        'provider': provider_name,
                        'model': model,
                        'config': config,
                        'size': size_info,
                        'supports_files': model_supports_files
                    })
                    models_added += 1
                    
                    # Выбираем сохраненную модель
                    if model == selected_model:
                        self.local_model_selector.setCurrentIndex(models_added - 1)
                
                self.local_model_selector.setEnabled(models_added > 0)
                print(f"[OK] Загружено {models_added} локальных моделей для {config.display_name} (только с поддержкой файлов)")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки локальных моделей для {provider_name}: {e}")
            self.local_model_selector.clear()
            self.local_model_selector.addItem("Ошибка загрузки моделей", None)
            self.local_model_selector.setEnabled(False)

    def get_ollama_models(self) -> list:
        """Получает список доступных моделей из Ollama."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except (requests.RequestException, requests.ConnectionError, requests.Timeout, ValueError, KeyError, ImportError) as e:
            # Ошибка при получении моделей Ollama - возвращаем пустой список
            return []

    def get_model_pricing_info(self, provider_name: str, model: str) -> str:
        """Возвращает информацию о платности модели."""
        # Базовая информация о платности моделей
        free_models = {
            'openai': ['gpt-3.5-turbo'],
            'anthropic': [],
            'google': ['models/gemini-1.5-flash-latest', 'models/gemini-1.5-flash-002'],
            'mistral': [],
            'deepseek': ['deepseek-chat', 'deepseek-coder'], 
            'xai': [],
        }
        
        if provider_name in free_models and model in free_models[provider_name]:
            return "🆓"
        else:
            return "💰"

    def get_model_size_info(self, model: str) -> str:
        """Извлекает информацию о размере модели из названия."""
        import re
        size_match = re.search(r'(\d+\.?\d*[bmk])', model.lower())
        if size_match:
            return f"({size_match.group(1).upper()})"
        return ""

    def on_cloud_model_changed(self):
        """Обработчик изменения выбранной облачной модели."""
        self.update_cloud_llm_status()

    def on_local_model_changed(self):
        """Обработчик изменения выбранной локальной модели."""
        self.update_local_llm_status()

    def load_selected_local_llm(self):
        """Загружает выбранный локальный LLM плагин"""
        provider_data = self.local_provider_selector.currentData()
        model_data = self.local_model_selector.currentData()
        
        if not provider_data or not model_data:
            return
            
        # Проверяем, не идет ли уже загрузка
        if self.llm_loading_thread and self.llm_loading_thread.isRunning():
            utils.show_info_message(self, "Загрузка", "LLM модель уже загружается...")
            return
        
        # Формируем данные для загрузки
        load_data = {
            'provider': provider_data.get('provider'),
            'model': model_data.get('model'),
            'config': provider_data.get('config')
        }
        
        # Создаем поток для загрузки модели
        self.llm_loading_thread = LLMLoadingThread(self.plugin_manager, load_data)
        self.llm_loading_thread.loading_started.connect(self.on_local_llm_loading_started)
        self.llm_loading_thread.loading_finished.connect(self.on_local_llm_loading_finished)
        self.llm_loading_thread.loading_error.connect(self.on_local_llm_loading_error)
        
        self.llm_loading_thread.start()
    
    def on_local_llm_loading_started(self, plugin_id: str):
        """Обработчик начала загрузки локальной LLM"""
        self.local_llm_status_label.setText("Статус: 🔄 Загружается...")
        QApplication.processEvents()

    def on_local_llm_loading_finished(self, plugin_id: str, plugin_instance):
        """Обработчик завершения загрузки локальной LLM"""
        self.current_llm_plugin = plugin_instance
        self.update_local_llm_status()
        
        plugin_info = self.plugin_manager.get_plugin_info(plugin_id)
        plugin_name = plugin_info.get('name', plugin_id) if plugin_info else plugin_id
        
        utils.show_info_message(
            self, 
            "Локальная LLM Загружена", 
            f"Локальный LLM плагин {plugin_name} успешно загружен!"
        )

    def on_local_llm_loading_error(self, plugin_id: str, error_message: str):
        """Обработчик ошибки загрузки локальной LLM"""
        self.local_llm_status_label.setText("Статус: ❌ Ошибка загрузки")
        
        utils.show_error_message(
            self,
            "Ошибка загрузки локальной LLM",
            f"Не удалось загрузить локальный LLM плагин:\n{error_message}"
        )

    # NEW: Обработчики для FileListWidget
    
    def on_file_list_selection_changed(self, file_path: str):
        """Обработчик выбора файла из списка."""
        self.current_image_path = file_path
        self.current_folder_path = None
        
        # Загружаем изображение для preview (если нужно)
        if hasattr(self, 'image_widget') and self.image_widget.isVisible():
            self.load_image(file_path)
        
        # ИСПРАВЛЕНИЕ: Включаем кнопку обработки при выборе из списка
        self.process_button.setText("🚀 Обработать")
        self.process_button.setEnabled(True)
        
        # Обновляем статус
        filename = os.path.basename(file_path)
        self.status_bar.showMessage(f"Выбран файл: {filename}")
        print(f"ОТЛАДКА: Выбран файл из списка, кнопка активирована: {file_path}")
        
    def on_process_single_file_requested(self, file_path: str):
        """Обработчик запроса на обработку одного файла."""
        # Устанавливаем текущий файл
        self.current_image_path = file_path
        self.current_folder_path = None
        
        # Обновляем статус в файловом списке
        self.file_list_widget.update_file_progress(file_path, 0, ProcessingStatus.PROCESSING)
        
        # Запускаем обработку
        self.process_image()
        
    def on_process_all_files_requested(self):
        """Обработчик запроса на обработку всех файлов."""
        unprocessed_files = self.file_list_widget.get_unprocessed_files()
        
        if not unprocessed_files:
            utils.show_info_message(
                self, "Информация", 
                "Нет файлов для обработки. Все файлы уже обработаны или произошли ошибки."
            )
            return
            
        # Создаем временную папку с файлами для пакетной обработки
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Копируем файлы в временную папку
            for file_path in unprocessed_files:
                import shutil
                shutil.copy2(file_path, temp_dir)
                
            # Запускаем пакетную обработку
            self.current_folder_path = temp_dir
            self.current_image_path = None
            self._process_folder_with_batch_processor(temp_dir)
            
    def on_filename_clicked(self, file_path: str, processing_data: dict):
        """Обработчик клика по имени файла - открывает окно просмотра."""
        try:
            # Создаем и показываем диалог просмотра файла
            viewer_dialog = FileViewerDialog(file_path, processing_data, self)
            viewer_dialog.exec()
            
        except Exception as e:
            utils.show_error_message(
                self, 
                "Ошибка открытия файла",
                f"Не удалось открыть файл для просмотра:\n{str(e)}"
            )
            
    def update_file_processing_progress(self, file_path: str, progress: int):
        """Обновление прогресса обработки файла."""
        if progress >= 100:
            self.file_list_widget.update_file_progress(file_path, 100, ProcessingStatus.COMPLETED)
        else:
            self.file_list_widget.update_file_progress(file_path, progress, ProcessingStatus.PROCESSING)
            
    def update_file_processing_fields(self, file_path: str, result: dict):
        """Обновление информации о распознанных полях файла."""
        if not result:
            return
            
        # Подсчитываем количество успешно распознанных полей
        recognized_fields = 0
        total_fields = 0
        
        # Получаем все ожидаемые поля из field_manager
        expected_fields = field_manager.get_enabled_fields()
        total_fields = len(expected_fields)
        
        # Проверяем какие поля были распознаны
        for field_config in expected_fields:
            # ИСПРАВЛЕНИЕ: TableField - это объект, используем его атрибуты напрямую
            field_key = field_config.id
            aliases = getattr(field_config, 'gemini_keywords', [])
            
            # Проверяем основное поле и его алиасы
            found = False
            if field_key in result and result[field_key]:
                found = True
            else:
                for alias in aliases:
                    if alias in result and result[alias]:
                        found = True
                        break
                        
            if found:
                recognized_fields += 1
                
        # Обновляем информацию в списке файлов
        self.file_list_widget.update_file_fields(file_path, recognized_fields, total_fields)
        
    def update_files_from_selection(self, path: str, is_folder: bool):
        """Обновление списка файлов на основе выбранного пути."""
        if is_folder:
            # Получаем список всех поддерживаемых файлов в папке
            supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            file_paths = []
            
            for ext in supported_extensions:
                import glob
                pattern = os.path.join(path, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                file_paths.extend(files)
                
            # Сортируем файлы по имени
            file_paths.sort()
            
            # Обновляем список файлов
            self.file_list_widget.set_files(file_paths)
            
        else:
            # Один файл
            self.file_list_widget.set_files([path])

    def check_gemini_api_status(self):
        """Проверяет статус API ключа Gemini и обновляет индикатор."""
        try:
            # ИСПРАВЛЕНИЕ: Проверяем API ключ через новую систему секретов с fallback
            api_key = None
            
            # Проверяем новую систему секретов
            try:
                from config.secrets import SecretsManager
                secrets_manager = SecretsManager()
                api_key = secrets_manager.get_secret("GOOGLE_API_KEY")
                if api_key:
                    print("🔑 Gemini API ключ найден в зашифрованном хранилище")
                else:
                    print("🔍 Gemini API ключ не найден в зашифрованном хранилище")
            except ImportError:
                print("⚠️ Система секретов недоступна, используем fallback")
            except Exception as e:
                print(f"⚠️ Ошибка доступа к системе секретов: {e}")
            
            # Fallback на старую систему
            if not api_key:
                try:
                    api_key = settings_manager.get_gemini_api_key()
                    if api_key:
                        print("🔑 Gemini API ключ найден в настройках (старая система)")
                    else:
                        # Дополнительная проверка через environment
                        import os
                        api_key = os.environ.get('GOOGLE_API_KEY')
                        if api_key:
                            print("🔑 Gemini API ключ найден в переменных окружения")
                except Exception as e:
                    print(f"⚠️ Ошибка проверки старой системы настроек: {e}")
            
            if api_key and len(api_key.strip()) > 10:
                # API ключ настроен
                self.gemini_status_indicator.setText("✅")
                self.gemini_status_indicator.setToolTip("API ключ настроен и доступен")
                self.gemini_radio.setEnabled(True)
                self.gemini_model_selector.setEnabled(True)
                print("✅ Gemini API ключ проверен успешно")
            else:
                # API ключ не настроен
                self.gemini_status_indicator.setText("❌")
                self.gemini_status_indicator.setToolTip("API ключ не настроен. Перейдите в Настройки → LLM провайдеры → Google")
                # Оставляем элементы активными для возможности настройки
                self.gemini_radio.setEnabled(True)
                self.gemini_model_selector.setEnabled(True)
                print("❌ Gemini API ключ не настроен")
                
        except Exception as e:
            self.gemini_status_indicator.setText("❓")
            self.gemini_status_indicator.setToolTip(f"Ошибка проверки: {e}")
            self.gemini_radio.setEnabled(True)  # Оставляем доступным для настройки
            self.gemini_model_selector.setEnabled(True)
            print(f"❓ Ошибка проверки API ключа Gemini: {e}")

    def update_model_component_visibility(self):
        """Обновляет видимость и активность компонентов в зависимости от выбранной модели."""
        # Определяем активную модель
        model_name = 'layoutlm'  # По умолчанию
        if self.layoutlm_radio.isChecked():
            model_name = 'layoutlm'
        elif self.donut_radio.isChecked():
            model_name = 'donut'
        elif self.trocr_radio.isChecked():
            model_name = 'trocr'
        elif self.gemini_radio.isChecked():
            model_name = 'gemini'
        elif self.cloud_llm_radio.isChecked():
            model_name = 'cloud_llm'
        elif self.local_llm_radio.isChecked():
            model_name = 'local_llm'
            
        print(f"🔧 Обновляем видимость компонентов для модели: {model_name}")
            
        # Обновляем доступность компонентов
        is_gemini = (model_name == 'gemini')
        is_trocr = (model_name == 'trocr')
        is_cloud_llm = (model_name == 'cloud_llm')
        is_local_llm = (model_name == 'local_llm')
        is_layoutlm = (model_name == 'layoutlm')
        is_donut = (model_name == 'donut')
        
        # ИСПРАВЛЕНИЕ: Gemini компоненты - всегда видимы и активны
        if hasattr(self, 'gemini_sub_model_label'):
            self.gemini_sub_model_label.setVisible(True)
            self.gemini_sub_model_label.setEnabled(is_gemini)
        if hasattr(self, 'gemini_model_selector'):
            self.gemini_model_selector.setVisible(True)
            self.gemini_model_selector.setEnabled(is_gemini)
        
        # TrOCR компоненты
        if hasattr(self, 'trocr_model_label'):
            self.trocr_model_label.setVisible(True)
            self.trocr_model_label.setEnabled(is_trocr)
        if hasattr(self, 'trocr_model_selector'):
            self.trocr_model_selector.setVisible(True)
            self.trocr_model_selector.setEnabled(is_trocr)
        if hasattr(self, 'trocr_status_label'):
            self.trocr_status_label.setVisible(True)
        
        # УПРОЩЕНИЕ: Все селекторы всегда активны для удобства пользователя
        if hasattr(self, 'cloud_provider_label'):
            self.cloud_provider_label.setEnabled(True)
        if hasattr(self, 'cloud_provider_selector'):
            self.cloud_provider_selector.setEnabled(True)
            # Загружаем провайдеров если их нет
            if self.cloud_provider_selector.count() <= 1:
                print("🔄 Загружаем облачных провайдеров...")
                self.populate_cloud_providers()
            # Автоматически выбираем первого провайдера при переключении на облачную модель
            if is_cloud_llm:
                QTimer.singleShot(100, lambda: self._auto_select_cloud_provider())
        if hasattr(self, 'cloud_model_label'):
            self.cloud_model_label.setEnabled(True)
        if hasattr(self, 'cloud_model_selector'):
            self.cloud_model_selector.setEnabled(True)
            print(f"🔧 Cloud model selector enabled: True (всегда активен)")
        if hasattr(self, 'cloud_llm_status_label'):
            self.cloud_llm_status_label.setVisible(True)
        
        # УПРОЩЕНИЕ: Все селекторы всегда активны для удобства пользователя
        if hasattr(self, 'local_provider_label'):
            self.local_provider_label.setEnabled(True)
        if hasattr(self, 'local_provider_selector'):
            self.local_provider_selector.setEnabled(True)
            # Загружаем провайдеров если их нет
            if self.local_provider_selector.count() <= 1:
                print("🔄 Загружаем локальных провайдеров...")
                self.populate_local_providers()
            # Автоматически выбираем первого провайдера при переключении на локальную модель
            if is_local_llm:
                QTimer.singleShot(100, lambda: self._auto_select_local_provider())
        if hasattr(self, 'local_model_label'):
            self.local_model_label.setEnabled(True)
        if hasattr(self, 'local_model_selector'):
            self.local_model_selector.setEnabled(True)
            print(f"🔧 Local model selector enabled: True (всегда активен)")
        if hasattr(self, 'local_llm_status_label'):
            self.local_llm_status_label.setVisible(True)
        
        # OCR язык только для LayoutLM
        if hasattr(self, 'ocr_lang_combo'):
            self.ocr_lang_combo.setEnabled(is_layoutlm)
        
        # ИСПРАВЛЕНИЕ: Автоматически разворачиваем соответствующую группу только если нужно
        if is_gemini or is_cloud_llm:
            if hasattr(self, 'cloud_models_group') and not self.cloud_models_group.expanded:
                self.cloud_models_group.set_expanded(True)
                print("☁️ Развернута группа облачных моделей")
        elif is_layoutlm or is_donut or is_trocr or is_local_llm:
            if hasattr(self, 'local_models_group') and not self.local_models_group.expanded:
                self.local_models_group.set_expanded(True)
                print("🖥️ Развернута группа локальных моделей")

    def update_group_visibility_based_on_model(self, is_cloud_model: bool, is_local_model: bool):
        """Обновляет видимость групп моделей в зависимости от выбранной модели."""
        if is_cloud_model:
            # Разворачиваем облачные модели, сворачиваем локальные
            self.cloud_models_group.set_expanded(True)
            self.local_models_group.set_expanded(False)
            print("☁️ Развернута группа: Облачные модели")
        elif is_local_model:
            # Разворачиваем локальные модели, сворачиваем облачные
            self.cloud_models_group.set_expanded(False)
            self.local_models_group.set_expanded(True)
            print("🖥️ Развернута группа: Локальные модели")
        else:
            # Если модель неизвестна, разворачиваем локальные по умолчанию
            self.cloud_models_group.set_expanded(False)
            self.local_models_group.set_expanded(True)
            print("⚠️ Развернута группа по умолчанию: Локальные модели")

    def save_model_selection_and_update_ui(self, model_name: str):
        """Сохраняет выбор модели и обновляет UI группы."""
        # Сохраняем выбор в настройках
        settings_manager.set_active_model(model_name)
        print(f"💾 Сохранена выбранная модель: {model_name}")
        
        # Определяем тип модели и обновляем видимость групп
        is_cloud_model = model_name in ['gemini', 'cloud_llm']
        is_local_model = model_name in ['layoutlm', 'donut', 'trocr', 'local_llm']
        
        # Обновляем видимость групп
        self.update_group_visibility_based_on_model(is_cloud_model, is_local_model)

    def _install_translator(self, lang_code: str) -> bool:
        """Загружает и устанавливает переводчик Qt для выбранного языка."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return False
        # Снимаем старый переводчик
        try:
            app.removeTranslator(self._translator)
        except Exception:
            pass
        # Пытаемся загрузить .qm из папки translations
        translations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'translations')
        file_name = f'invoicegemini_{lang_code}.qm'
        qm_path = os.path.join(translations_dir, file_name)
        loaded = False
        try:
            loaded = self._translator.load(qm_path)
        except Exception as e:
            logger.warning(f"Failed to load translation '{qm_path}': {e}")
            loaded = False
        if loaded:
            app.installTranslator(self._translator)
            self._current_language = lang_code
            logger.info(f"Installed translator: {lang_code}")
        else:
            self._current_language = None
            logger.info("No translation loaded; using source locale")
        return loaded

    def change_app_language(self, lang_code: str):
        """Публичный метод смены языка, переустанавливает переводчики и обновляет тексты UI."""
        self._install_translator(lang_code)
        # Перевыставляем тексты в текущем окне
        try:
            self._retranslate_ui()
        except Exception as e:
            logger.warning(f"Retranslate UI failed: {e}")
        # Обновляем активные диалоги, если открыты
        try:
            if self.model_management_dialog is not None:
                if hasattr(self.model_management_dialog, 'setWindowTitle'):
                    # Частичный retranslate: можно расширить при необходимости
                    self.model_management_dialog.setWindowTitle(self.tr("Настройки и управление моделями"))
        except Exception:
            pass

    def _retranslate_ui(self):
        """Обновляет ключевые тексты главного окна. Расширяйте при добавлении новых элементов."""
        try:
            self.setWindowTitle(self.tr("Обработка счетов"))
            # Пример: если есть меню/кнопки, выставить тексты заново
            # Этот метод можно дополнять при необходимости
        except Exception:
            pass

    def changeEvent(self, event):
        try:
            from PyQt6.QtCore import QEvent
            if event.type() == QEvent.Type.LanguageChange:
                self._retranslate_ui()
        except Exception:
            pass
        super().changeEvent(event)


# NEW: LLM Loading Thread
class LLMLoadingThread(QThread):
    """Поток для загрузки LLM плагинов в фоне"""
    loading_started = pyqtSignal(str)
    loading_finished = pyqtSignal(str, object)
    loading_error = pyqtSignal(str, str)
    
    def __init__(self, plugin_manager, plugin_data):
        super().__init__()
        self.plugin_manager = plugin_manager
        self.plugin_data = plugin_data
        self._should_stop = False  # Флаг для корректной остановки
    
    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.plugin_manager = None
            self.plugin_data = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке LLMLoadingThread: {e}")
        
    def run(self):
        try:
            provider_name = self.plugin_data.get('provider')
            model_name = self.plugin_data.get('model')
            config = self.plugin_data.get('config')
            
            plugin_id = f"{provider_name}:{model_name}"
            self.loading_started.emit(plugin_id)
            
            # Получаем настройки провайдера
            llm_settings = settings_manager.get_setting('llm_providers', {})
            provider_settings = llm_settings.get(provider_name, {})
            
            # Получаем API ключ если требуется
            api_key = None
            if config.requires_api_key:
                api_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
                if not api_key:
                    self.loading_error.emit(plugin_id, f"API ключ для {provider_name} не найден")
                    return
            
            # Создаем экземпляр универсального плагина
            from .plugins.models.universal_llm_plugin import UniversalLLMPlugin
            
            # Дополнительные параметры
            plugin_kwargs = {
                'generation_config': {
                    'temperature': provider_settings.get('temperature', 0.1),
                    'max_tokens': provider_settings.get('max_tokens', 4096),
                    'top_p': provider_settings.get('top_p', 0.9),
                }
            }
            
            # Для Ollama добавляем base_url
            if provider_name == "ollama":
                plugin_kwargs['base_url'] = provider_settings.get('base_url', 'http://localhost:11434')
            
            plugin = UniversalLLMPlugin(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key,
                **plugin_kwargs
            )
            
            # Инициализируем плагин
            if plugin.load_model():
                self.loading_finished.emit(plugin_id, plugin)
            else:
                self.loading_error.emit(plugin_id, f"Не удалось инициализировать {config.display_name}")
                
        except Exception as e:
            plugin_id = f"{self.plugin_data.get('provider', 'unknown')}:{self.plugin_data.get('model', 'unknown')}"
            self.loading_error.emit(plugin_id, str(e))
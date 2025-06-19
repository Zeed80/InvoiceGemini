"""
Главное окно приложения для извлечения данных из счетов-фактур.
"""
import os
import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QRadioButton, QLabel, QGroupBox, 
    QScrollArea, QFileDialog, QProgressBar, QComboBox,
    QStatusBar, QSplitter, QMenuBar, QMenu, QApplication,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QTextEdit,
    QSpacerItem, QSizePolicy, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QUrl, QTimer, QThread
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QFont
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


# NEW: Import LLM Plugin Manager
from .plugins.plugin_manager import PluginManager
from .ui.preview_dialog import PreviewDialog
from .ui.export_template_designer import ExportTemplateDesigner
from .ui.field_manager_dialog import FieldManagerDialog
from .field_manager import field_manager
from .ui.llm_providers_dialog import LLMProvidersDialog

# NEW: Import new components for Phase 1 improvements
from .core.cache_manager import CacheManager
from .core.retry_manager import RetryManager
from .core.backup_manager import BackupManager
from .ui.components.file_selector import FileSelector
from .ui.components.progress_indicator import ProgressIndicator
from .ui.components.batch_processor_adapter import BatchProcessor
from .ui.components.export_manager import ExportManager


class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image_path = None
        self.current_folder_path = None # NEW: Добавлено для хранения пути к папке
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
        self.backup_manager = BackupManager()
        
        # NEW: Initialize UI components
        self.file_selector = None  # Will be initialized in init_ui
        self.progress_indicator = None  # Will be initialized in init_ui
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
        
        # Левая часть - для изображения
        self.image_widget = QWidget()
        image_layout = QVBoxLayout(self.image_widget)
        
        # NEW: Use FileSelector component
        self.file_selector = FileSelector()
        self.file_selector.signals.file_selected.connect(self.on_file_selected)
        self.file_selector.signals.folder_selected.connect(self.on_folder_selected)
        
        # For backward compatibility
        self.select_file_button = self.file_selector.select_file_button
        self.select_folder_button = self.file_selector.select_folder_button
        self.selected_path_label = self.file_selector.selection_label
        
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
        
        # Добавляем виджеты в левую часть
        image_layout.addWidget(self.file_selector)
        image_layout.addWidget(image_group, 1)  # Растягивать при изменении размера окна
        
        # Правая часть - для выбора модели и результатов
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        
        # Виджет для выбора модели
        model_group = QGroupBox("Выбор модели")
        model_layout = QVBoxLayout()
        
        # === ОБЛАЧНЫЕ МОДЕЛИ ===
        cloud_section_label = QLabel("☁️ Облачные модели")
        cloud_section_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px; padding: 8px 0px;")
        model_layout.addWidget(cloud_section_label)
        
        # Gemini (перенесено в облачные)
        gemini_layout = QHBoxLayout()
        self.gemini_radio = QRadioButton("Google Gemini")
        self.gemini_radio.toggled.connect(self.on_model_changed)
        self.gemini_prompt_button = QPushButton("Показать промпт")
        self.gemini_prompt_button.clicked.connect(lambda: self.show_model_prompt('gemini'))
        gemini_layout.addWidget(self.gemini_radio)
        gemini_layout.addWidget(self.gemini_prompt_button)
        model_layout.addLayout(gemini_layout)
        
        # Gemini sub-model selector
        gemini_sub_layout = QHBoxLayout()
        gemini_sub_layout.setContentsMargins(20, 0, 0, 0)  # Отступ для подкатегории
        self.gemini_sub_model_label = QLabel("Модель:")
        self.gemini_model_selector = QComboBox()
        self.gemini_model_selector.currentIndexChanged.connect(self.on_gemini_sub_model_changed)
        
        gemini_sub_layout.addWidget(self.gemini_sub_model_label)
        gemini_sub_layout.addWidget(self.gemini_model_selector, 1)
        model_layout.addLayout(gemini_sub_layout)
        
        # Populate Gemini models
        self.populate_gemini_models()
        
        # Cloud LLM Provider Selection
        cloud_llm_layout = QHBoxLayout()
        self.cloud_llm_radio = QRadioButton("Другие облачные LLM")
        self.cloud_llm_radio.toggled.connect(self.on_model_changed)
        
        self.cloud_llm_prompt_button = QPushButton("Показать промпт")
        self.cloud_llm_prompt_button.clicked.connect(lambda: self.show_model_prompt('cloud_llm'))
        cloud_llm_layout.addWidget(self.cloud_llm_radio)
        cloud_llm_layout.addWidget(self.cloud_llm_prompt_button)
        model_layout.addLayout(cloud_llm_layout)
        
        # Cloud provider and model selection (with indent)
        cloud_selection_layout = QVBoxLayout()
        cloud_selection_layout.setContentsMargins(20, 0, 0, 0)  # Отступ для подкатегории
        
        cloud_provider_layout = QHBoxLayout()
        self.cloud_provider_label = QLabel("Провайдер:")
        self.cloud_provider_selector = QComboBox()
        self.cloud_provider_selector.currentIndexChanged.connect(self.on_cloud_provider_changed)
        cloud_provider_layout.addWidget(self.cloud_provider_label)
        cloud_provider_layout.addWidget(self.cloud_provider_selector, 1)
        cloud_selection_layout.addLayout(cloud_provider_layout)
        
        cloud_model_layout = QHBoxLayout()
        self.cloud_model_label = QLabel("Модель:")
        self.cloud_model_selector = QComboBox()
        self.cloud_model_selector.currentIndexChanged.connect(self.on_cloud_model_changed)
        cloud_model_layout.addWidget(self.cloud_model_label)
        cloud_model_layout.addWidget(self.cloud_model_selector, 1)
        cloud_selection_layout.addLayout(cloud_model_layout)
        
        # Cloud status indicator
        self.cloud_llm_status_label = QLabel("Не настроено")
        self.cloud_llm_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        cloud_selection_layout.addWidget(self.cloud_llm_status_label)
        
        model_layout.addLayout(cloud_selection_layout)
        
        # Разделитель между облачными и локальными
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #ccc; margin: 10px 0;")
        model_layout.addWidget(separator)
        
        # === ЛОКАЛЬНЫЕ МОДЕЛИ ===
        local_section_label = QLabel("🖥️ Локальные модели")
        local_section_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 14px; padding: 8px 0px;")
        model_layout.addWidget(local_section_label)
        
        # LayoutLM (перенесено в локальные)
        layoutlm_layout = QHBoxLayout()
        self.layoutlm_radio = QRadioButton("LayoutLMv3")
        self.layoutlm_radio.setChecked(True)  # По умолчанию выбрана
        self.layoutlm_radio.toggled.connect(self.on_model_changed)
        self.layoutlm_prompt_button = QPushButton("Показать промпт")
        self.layoutlm_prompt_button.clicked.connect(lambda: self.show_model_prompt('layoutlm'))
        layoutlm_layout.addWidget(self.layoutlm_radio)
        layoutlm_layout.addWidget(self.layoutlm_prompt_button)
        model_layout.addLayout(layoutlm_layout)
        
        # Donut (перенесено в локальные)
        donut_layout = QHBoxLayout()
        self.donut_radio = QRadioButton("Donut")
        self.donut_radio.toggled.connect(self.on_model_changed)
        self.donut_prompt_button = QPushButton("Показать промпт")
        self.donut_prompt_button.clicked.connect(lambda: self.show_model_prompt('donut'))
        donut_layout.addWidget(self.donut_radio)
        donut_layout.addWidget(self.donut_prompt_button)
        model_layout.addLayout(donut_layout)
        
        # TrOCR (Microsoft Transformer OCR)
        trocr_layout = QHBoxLayout()
        self.trocr_radio = QRadioButton("TrOCR (Microsoft)")
        self.trocr_radio.toggled.connect(self.on_model_changed)
        self.trocr_prompt_button = QPushButton("Показать промпт")
        self.trocr_prompt_button.clicked.connect(lambda: self.show_model_prompt('trocr'))
        trocr_layout.addWidget(self.trocr_radio)
        trocr_layout.addWidget(self.trocr_prompt_button)
        model_layout.addLayout(trocr_layout)
        
        # TrOCR model selection (with indent)
        trocr_selection_layout = QVBoxLayout()
        trocr_selection_layout.setContentsMargins(20, 0, 0, 0)  # Отступ для подкатегории
        
        trocr_model_layout = QHBoxLayout()
        self.trocr_model_label = QLabel("Модель:")
        self.trocr_model_selector = QComboBox()
        self.trocr_model_selector.currentIndexChanged.connect(self.on_trocr_model_changed)
        self.trocr_model_selector.setToolTip("Выберите модель TrOCR для использования")
        trocr_model_layout.addWidget(self.trocr_model_label)
        trocr_model_layout.addWidget(self.trocr_model_selector, 1)
        trocr_selection_layout.addLayout(trocr_model_layout)
        
        # TrOCR status indicator
        self.trocr_status_label = QLabel("Не загружено")
        self.trocr_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        trocr_selection_layout.addWidget(self.trocr_status_label)
        
        model_layout.addLayout(trocr_selection_layout)
        
        # Local LLM Models Section
        local_llm_layout = QHBoxLayout()
        self.local_llm_radio = QRadioButton("Локальные LLM (Ollama)")
        self.local_llm_radio.toggled.connect(self.on_model_changed)
        
        self.local_llm_prompt_button = QPushButton("Показать промпт")
        self.local_llm_prompt_button.clicked.connect(lambda: self.show_model_prompt('local_llm'))
        local_llm_layout.addWidget(self.local_llm_radio)
        local_llm_layout.addWidget(self.local_llm_prompt_button)
        model_layout.addLayout(local_llm_layout)
        
        # Local provider and model selection (with indent)
        local_selection_layout = QVBoxLayout()
        local_selection_layout.setContentsMargins(20, 0, 0, 0)  # Отступ для подкатегории
        
        local_provider_layout = QHBoxLayout()
        self.local_provider_label = QLabel("Провайдер:")
        self.local_provider_selector = QComboBox()
        self.local_provider_selector.currentIndexChanged.connect(self.on_local_provider_changed)
        local_provider_layout.addWidget(self.local_provider_label)
        local_provider_layout.addWidget(self.local_provider_selector, 1)
        local_selection_layout.addLayout(local_provider_layout)
        
        local_model_layout = QHBoxLayout()
        self.local_model_label = QLabel("Модель:")
        self.local_model_selector = QComboBox()
        self.local_model_selector.currentIndexChanged.connect(self.on_local_model_changed)
        local_model_layout.addWidget(self.local_model_label)
        local_model_layout.addWidget(self.local_model_selector, 1)
        local_selection_layout.addLayout(local_model_layout)
        
        # Local status indicator
        self.local_llm_status_label = QLabel("Не настроено")
        self.local_llm_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        local_selection_layout.addWidget(self.local_llm_status_label)
        
        model_layout.addLayout(local_selection_layout)

        model_group.setLayout(model_layout)
        
        # Виджет для выбора языка OCR
        ocr_lang_group = QGroupBox("Язык OCR (для LayoutLMv3)")
        ocr_lang_layout = QVBoxLayout()
        
        self.ocr_lang_combo = QComboBox()
        self.ocr_lang_combo.addItem("English", "eng")
        self.ocr_lang_combo.addItem("Русский", "rus")
        self.ocr_lang_combo.addItem("English + Русский", "eng+rus")
        self.ocr_lang_combo.currentIndexChanged.connect(self.on_ocr_language_changed)
        
        ocr_lang_layout.addWidget(self.ocr_lang_combo)
        ocr_lang_group.setLayout(ocr_lang_layout)
        
        # Кнопка обработки
        self.process_button = QPushButton("Обработать")
        self.process_button.setEnabled(False)  # Отключаем до загрузки изображения
        self.process_button.clicked.connect(self.process_image)
        
        # NEW: Use ProgressIndicator component
        self.progress_indicator = ProgressIndicator()
        # For backward compatibility - используем встроенный progress_bar компонента
        self.progress_bar = self.progress_indicator.progress_bar
        
        # Область результатов с таблицей вместо текста
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        
        # Добавляем кнопку редактирования полей в заголовок таблицы
        table_header_layout = QHBoxLayout()
        table_header_label = QLabel("Извлечённые данные:")
        table_header_layout.addWidget(table_header_label)
        
        # Кнопка управления полями (обновлена)
        self.edit_fields_button = QPushButton("🔧 Управление полями")
        self.edit_fields_button.clicked.connect(self.show_field_manager_dialog)
        self.edit_fields_button.setToolTip("Управление полями таблицы и синхронизация промптов")
        table_header_layout.addWidget(self.edit_fields_button)
        
        results_layout.addLayout(table_header_layout)
        
        # Создаем таблицу для отображения результатов
        self.results_table = QTableWidget()
        
        # Настраиваем таблицу исходя из полей в настройках
        self.setup_results_table()
        
        results_layout.addWidget(self.results_table)
        
        # Добавляем кнопки сохранения результатов
        save_buttons_layout = QHBoxLayout()
        
        # NEW: Template Designer button
        self.template_designer_button = QPushButton("🎨 Дизайнер шаблонов")
        self.template_designer_button.clicked.connect(self.show_template_designer)
        self.template_designer_button.setToolTip("Создать и настроить шаблоны экспорта")
        self.template_designer_button.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }")
        save_buttons_layout.addWidget(self.template_designer_button)
        
        # NEW: Preview button
        self.preview_button = QPushButton("🔍 Предварительный просмотр")
        self.preview_button.setEnabled(False)  # Отключаем до завершения обработки
        self.preview_button.clicked.connect(self.show_preview_dialog)
        self.preview_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        save_buttons_layout.addWidget(self.preview_button)
        
        self.save_button = QPushButton("Сохранить как...")
        self.save_button.setEnabled(False)  # Отключаем до завершения обработки
        self.save_button.clicked.connect(self.save_results)
        save_buttons_layout.addWidget(self.save_button)
        
        self.save_excel_button = QPushButton("Сохранить в Excel")
        self.save_excel_button.setEnabled(False)  # Отключаем до завершения обработки
        self.save_excel_button.clicked.connect(self.save_excel)
        save_buttons_layout.addWidget(self.save_excel_button)
        
        results_layout.addLayout(save_buttons_layout)
        
        results_group.setLayout(results_layout)
        
        # Добавляем виджеты в правую часть
        controls_layout.addWidget(model_group)
        controls_layout.addWidget(ocr_lang_group)
        controls_layout.addWidget(self.process_button)
        controls_layout.addWidget(self.progress_indicator)
        controls_layout.addWidget(results_group)
        
        # Добавляем левую и правую части в сплиттер
        splitter.addWidget(self.image_widget)
        splitter.addWidget(self.controls_widget)
        
        # Устанавливаем начальные размеры сплиттера
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
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
                if hasattr(self, 'progress_indicator') and self.progress_indicator:
                    self.batch_processor.progress_updated.connect(self.progress_indicator.set_progress)
                    self.batch_processor.status_updated.connect(self.progress_indicator.set_stage)
                    
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
                
                if hasattr(self, 'progress_indicator') and self.progress_indicator:
                    self.batch_processor.progress_updated.connect(self.progress_indicator.set_progress)
                    self.batch_processor.status_updated.connect(self.progress_indicator.set_stage)
                    
                print("BatchProcessor инициализирован (отложенная инициализация)")
        except Exception as e:
            print(f"Ошибка отложенной инициализации BatchProcessor: {e}")
    
    def on_file_selected(self, file_path: str):
        """Обработчик выбора файла через FileSelector."""
        self.current_image_path = file_path
        self.current_folder_path = None
        self.load_image(file_path)
        
    def on_folder_selected(self, folder_path: str):
        """Обработчик выбора папки через FileSelector."""
        self.current_folder_path = folder_path
        self.current_image_path = None
        
        # Enable batch processing
        self.process_button.setText("Обработать папку")
        self.process_button.setEnabled(True)
        self.status_bar.showMessage(f"Выбрана папка: {folder_path}")
        

            
    def on_batch_processing_started(self, total_files: int):
        """Обработчик начала пакетной обработки."""
        self.batch_results = []
        self.progress_indicator.set_title(f"Пакетная обработка")
        self.progress_indicator.set_stage(f"Обработка {total_files} файлов...")
        self.progress_indicator.setVisible(True)
        self.progress_indicator.start()
        self.process_button.setEnabled(False)
        
    def on_batch_file_processed(self, file_path: str, result: dict, index: int, total: int):
        """Обработчик обработки одного файла из пакета."""
        self.batch_results.append({
            'file_path': file_path,
            'result': result
        })
        self.progress_indicator.set_stage(f"Обработано {index + 1} из {total} файлов")
        
    def on_batch_processing_finished(self):
        """Обработчик завершения пакетной обработки."""
        self.progress_indicator.setVisible(False)
        self.progress_indicator.stop()
        self.process_button.setEnabled(True)
        
        if self.batch_results:
            # Show batch results
            self.show_batch_results()
            
    def on_batch_error(self, error_message: str):
        """Обработчик ошибки при пакетной обработке."""
        self.progress_indicator.setVisible(False)
        self.progress_indicator.stop()
        self.process_button.setEnabled(True)
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
        # Загрузка активной модели
        active_model = settings_manager.get_active_model()
        if active_model == 'layoutlm':
            self.layoutlm_radio.setChecked(True)
        elif active_model == 'donut':
            self.donut_radio.setChecked(True)
        elif active_model == 'trocr':
            self.trocr_radio.setChecked(True)
        elif active_model == 'gemini':
            self.gemini_radio.setChecked(True)
        elif active_model == 'cloud_llm':
            self.cloud_llm_radio.setChecked(True)
        elif active_model == 'local_llm':
            self.local_llm_radio.setChecked(True)
        else:
            # По умолчанию выбираем LayoutLM
            self.layoutlm_radio.setChecked(True)
        
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
                
            # Сохраняем выбор в настройках
            settings_manager.set_active_model(model_name)
            
            # Обновляем доступность выбора языка OCR (только для LayoutLM)
            self.ocr_lang_combo.setEnabled(model_name == 'layoutlm')
            
            # Обновляем видимость компонентов
            trocr_enabled = (model_name == 'trocr')
            gemini_enabled = (model_name == 'gemini')
            cloud_llm_enabled = (model_name == 'cloud_llm')
            local_llm_enabled = (model_name == 'local_llm')
            
            # TrOCR компоненты
            self.trocr_model_label.setEnabled(trocr_enabled)
            self.trocr_model_selector.setEnabled(trocr_enabled)
            
            # Gemini компоненты
            self.gemini_sub_model_label.setEnabled(gemini_enabled)
            self.gemini_model_selector.setEnabled(gemini_enabled)
            
            # Облачные LLM компоненты
            self.cloud_provider_label.setEnabled(cloud_llm_enabled)
            self.cloud_provider_selector.setEnabled(cloud_llm_enabled)
            self.cloud_model_label.setEnabled(cloud_llm_enabled)
            self.cloud_model_selector.setEnabled(cloud_llm_enabled and self.cloud_provider_selector.count() > 0)
            
            # Локальные LLM компоненты  
            self.local_provider_label.setEnabled(local_llm_enabled)
            self.local_provider_selector.setEnabled(local_llm_enabled)
            self.local_model_label.setEnabled(local_llm_enabled)
            self.local_model_selector.setEnabled(local_llm_enabled and self.local_provider_selector.count() > 0)
            
            # Обновляем статусы LLM, если выбраны
            if cloud_llm_enabled:
                self.update_cloud_llm_status()
            if local_llm_enabled:
                self.update_local_llm_status()
            
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
        
        # Показываем индикатор прогресса
        self.status_bar.showMessage(f"Обработка изображения моделью {model_type.upper()}...")
        self.progress_indicator.setVisible(True)
        self.progress_indicator.set_progress(0)
        self.progress_indicator.set_title(f"Обработка моделью {model_type.upper()}")
        self.progress_indicator.start()
        
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
    
    def update_progress(self, value):
        """Обновление индикатора прогресса."""
        self.progress_indicator.set_progress(value)
    
    def show_results(self, results):
        """Отображение результатов обработки в таблице (для ОДИНОЧНОГО файла)."""
        try:
            # Этот метод теперь используется только для отображения результата одного файла
            # Сохраняем результаты для дальнейшего использования
            self.processing_thread.result = results # Сохраняем для совместимости с сохранением одиночного файла
            
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
            self.preview_button.setEnabled(True)
            
            # NEW: Активируем кнопки новой системы плагинов
            if hasattr(self, 'validate_button'):
                self.validate_button.setEnabled(True)
            if hasattr(self, 'view_data_button'):
                self.view_data_button.setEnabled(True)
            if hasattr(self, 'plugin_export_button'):
                self.plugin_export_button.setEnabled(True)
            
            # Скрываем индикатор прогресса
            if hasattr(self, 'progress_indicator') and self.progress_indicator:
                self.progress_indicator.setVisible(False)
                self.progress_indicator.stop()
            self.status_bar.showMessage("Обработка завершена")
        except Exception as e:
            print(f"ОШИБКА в show_results: {e}")
            import traceback
            traceback.print_exc()
            self.show_processing_error(f"Ошибка отображения результатов: {str(e)}")
    
    def show_processing_error(self, error_msg):
        """Обработка ошибки при обработке изображения."""
        if hasattr(self, 'progress_indicator') and self.progress_indicator:
            self.progress_indicator.setVisible(False)
            self.progress_indicator.stop()
        self.status_bar.showMessage("Ошибка обработки")
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
            # Можно добавить уведомление об успешном обновлении
            self.status_bar.showMessage("Поля таблицы обновлены", 3000)
        except Exception as e:
            utils.show_error_message(
                self, "Ошибка", f"Ошибка при обновлении таблицы: {str(e)}"
            )

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
        Создает базовый промпт для LLM провайдера.
        
        Args:
            provider_name: Имя провайдера (openai, anthropic, google, etc.)
            
        Returns:
            str: Базовый промпт для извлечения данных из инвойсов
        """
        # Получаем поля таблицы для включения в промпт
        table_fields = []
        try:
            from .field_manager import FieldManager
            field_manager = FieldManager()
            enabled_fields = field_manager.get_enabled_fields()
            table_fields = [f"- {field.display_name}: {field.description}" for field in enabled_fields]
        except:
            # Базовые поля если не удалось получить из настроек
            table_fields = [
                "- Номер счета: Номер документа/инвойса",
                "- Дата: Дата выставления счета",
                "- Поставщик: Название компании-поставщика",
                "- Сумма: Общая сумма к оплате",
                "- НДС: Сумма налога на добавленную стоимость",
                "- Валюта: Валюта документа"
            ]
        
        # Базовый промпт с учетом особенностей провайдера
        if provider_name == "anthropic":
            # Claude предпочитает более структурированные инструкции
            prompt = """Ты эксперт по анализу финансовых документов. Проанализируй предоставленное изображение счета-фактуры или инвойса и извлеки из него структурированные данные.

<instructions>
Извлеки следующие поля из документа:

{fields}

Требования к ответу:
1. Верни результат ТОЛЬКО в формате JSON
2. Используй точные названия полей как указано выше
3. Если поле не найдено, используй значение "N/A"
4. Все суммы указывай числами без символов валют
5. Даты в формате DD.MM.YYYY
6. Будь точным и внимательным к деталям
</instructions>

Проанализируй документ и верни JSON с извлеченными данными:"""
            
        elif provider_name == "google":
            # Gemini хорошо работает с четкими инструкциями
            prompt = """Действуй как эксперт по распознаванию счетов-фактур и финансовых документов. 

Твоя задача: проанализировать изображение документа и извлечь из него ключевые данные в формате JSON.

Поля для извлечения:
{fields}

Правила:
• Возвращай ТОЛЬКО валидный JSON без дополнительного текста
• Используй точные названия полей как указано
• Для отсутствующих полей используй "N/A"
• Числовые значения без символов валют
• Даты в формате DD.MM.YYYY
• Будь максимально точным

Проанализируй документ:"""
            
        elif provider_name in ["openai", "deepseek", "xai"]:
            # OpenAI-совместимые модели
            prompt = """You are an expert in invoice and financial document analysis. Analyze the provided document image and extract structured data in JSON format.

Extract the following fields:
{fields}

Requirements:
- Return ONLY valid JSON format
- Use exact field names as specified
- Use "N/A" for missing fields  
- Numeric values without currency symbols
- Dates in DD.MM.YYYY format
- Be precise and thorough

Analyze the document and return JSON:"""
            
        elif provider_name == "mistral":
            # Mistral предпочитает краткие четкие инструкции
            prompt = """Analyse ce document financier et extrais les données en JSON.

Champs à extraire:
{fields}

Format: JSON uniquement, "N/A" si absent, dates DD.MM.YYYY

Analyse:"""
            
        elif provider_name == "ollama":
            # Для локальных моделей более простые инструкции
            prompt = """Extract data from this invoice/document in JSON format.

Fields to extract:
{fields}

Rules:
- JSON format only
- Use "N/A" if field not found
- Dates as DD.MM.YYYY
- Numbers without currency symbols

Extract the data:"""
            
        else:
            # Универсальный промпт для других провайдеров
            prompt = """Analyze this financial document and extract structured data in JSON format.

Extract these fields:
{fields}

Return only valid JSON. Use "N/A" for missing fields. Dates in DD.MM.YYYY format.

Analyze:"""
        
        return prompt.format(fields="\n".join(table_fields))
    
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
            if model_type in ['layoutlm', 'donut', 'gemini']:
                # Старые модели - сохраняем как раньше
                prompt_key = f"{model_type}_prompt"
                settings_manager.set_setting(prompt_key, prompt_text)
                
                # Обновляем промпт в процессоре
                processor = self.model_manager.get_model(model_type)
                if processor:
                    processor.set_prompt(prompt_text)
                    
                model_display_name = model_type.upper()
                
            elif model_type == 'cloud_llm':
                # Облачные LLM модели
                provider_data = self.cloud_provider_selector.currentData()
                if not provider_data:
                    utils.show_error_message(self, "Ошибка", "Провайдер не выбран")
                    return
                
                provider_name = provider_data.get('provider')
                prompt_key = f"cloud_llm_{provider_name}_prompt"
                settings_manager.set_setting(prompt_key, prompt_text)
                
                model_display_name = f"Cloud LLM ({provider_name.upper()})"
                
            elif model_type == 'local_llm':
                # Локальные LLM модели
                provider_data = self.local_provider_selector.currentData()
                if not provider_data:
                    utils.show_error_message(self, "Ошибка", "Провайдер не выбран")
                    return
                
                provider_name = provider_data.get('provider')
                prompt_key = f"local_llm_{provider_name}_prompt"
                settings_manager.set_setting(prompt_key, prompt_text)
                
                model_display_name = f"Local LLM ({provider_name.upper()})"
                
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
        trained_models_path = os.path.join(app_config.TRAINED_MODELS_PATH, 'trocr')
        if os.path.exists(trained_models_path):
            trained_models = [d for d in os.listdir(trained_models_path) 
                            if os.path.isdir(os.path.join(trained_models_path, d))]
            
            if trained_models:
                # Добавляем разделитель
                self.trocr_model_selector.insertSeparator(self.trocr_model_selector.count())
                
                # Добавляем дообученные модели
                for model_name in trained_models:
                    display_text = f"🎓 {model_name} (Дообученная)"
                    model_path = os.path.join(trained_models_path, model_name)
                    self.trocr_model_selector.addItem(display_text, model_path)
        
        # Восстанавливаем последний выбор
        last_model = settings_manager.get_string('Models', 'trocr_model_id', 'microsoft/trocr-base-printed')
        for i in range(self.trocr_model_selector.count()):
            if self.trocr_model_selector.itemData(i) == last_model:
                self.trocr_model_selector.setCurrentIndex(i)
                break
        
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
            return

        row_position = self.results_table.rowCount()
        self.results_table.insertRow(row_position)

        # Создаем маппинг display_name -> column_index на основе заголовков таблицы
        column_mapping = {}
        for col in range(self.results_table.columnCount()):
            header_item = self.results_table.horizontalHeaderItem(col)
            if header_item:
                column_mapping[header_item.text()] = col

        # Создаем расширенное сопоставление полей для гибкого поиска
        field_aliases = self._create_field_aliases_mapping(column_mapping)
        
        # Заполняем данные по display_name или алиасам
        processed_fields = 0
        for field_name, value in result.items():
            # Пропускаем служебные поля
            if field_name.startswith('_'):
                continue
                
            column_index = None
            
            # Сначала пытаемся точное совпадение
            if field_name in column_mapping:
                column_index = column_mapping[field_name]
            else:
                # Затем ищем по алиасам (нечувствительно к регистру)
                field_name_lower = field_name.lower()
                for alias, col_idx in field_aliases.items():
                    if field_name_lower == alias.lower():
                        column_index = col_idx
                        break
            
            if column_index is not None:
                item = QTableWidgetItem(str(value))
                
                # Выравнивание для числовых колонок
                if any(word in field_name for word in ["Amount", "Total", "VAT", "Сумма", "НДС", "№", "номер", "%"]):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
                self.results_table.setItem(row_position, column_index, item)
                processed_fields += 1
            else:
                # Логируем неизвестные поля для отладки
                print(f"ОТЛАДКА: Неизвестное поле '{field_name}' со значением '{value}' не добавлено в таблицу")

        try:
            self.results_table.resizeRowsToContents()
            print(f"ОТЛАДКА: Добавлена строка в таблицу. Полей обработано: {processed_fields}/{len([k for k in result.keys() if not k.startswith('_')])}")
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
            # Используем старый метод для отображения одного результата
            if result_or_none:
                self.show_results(result_or_none)
            else:
                # Если результат None (например, ошибка в потоке, но не исключение)
                self.status_bar.showMessage("Ошибка обработки файла.")
                self.save_button.setEnabled(False)
                if hasattr(self, 'save_action'): self.save_action.setEnabled(False)
                self.save_excel_button.setEnabled(False)
                if hasattr(self, 'save_excel_action'): self.save_excel_action.setEnabled(False)
                # NEW: Disable preview button on error
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
                        except:
                            pass
            
            # NEW: Очищаем кэш перед закрытием
            if self.cache_manager:
                self.cache_manager.clear_expired()
                print("Очищен устаревший кэш")
                            
            # Даем время для остановки
            QCoreApplication.processEvents()
            
        except Exception as e:
            print(f"Ошибка при остановке фоновых потоков: {e}")
        
        # Очистка временных файлов при закрытии приложения
        try:
            self.temp_dir.cleanup()
            print("Временные файлы очищены")
        except:
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
        training_dialog.exec() 

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
                
            # Настраиваем отображение таблицы
            self.results_table.setAlternatingRowColors(True)
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            self.results_table.horizontalHeader().setStretchLastSection(True)
            self.results_table.verticalHeader().setVisible(False)
            
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
                
            # Настраиваем отображение таблицы
            self.results_table.setAlternatingRowColors(True)
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            self.results_table.horizontalHeader().setStretchLastSection(True)
            self.results_table.verticalHeader().setVisible(False)
            
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
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        
        self.field_mapping = {column["id"]: i for i, column in enumerate(basic_columns)}
        
        print("Базовая таблица создана с 5 колонками")

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
        try:
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
                file_path = self.current_folder_path
                
            else:
                # Одиночная обработка
                if not hasattr(self, 'processing_thread') or not self.processing_thread or \
                   not hasattr(self.processing_thread, 'result') or not self.processing_thread.result:
                    utils.show_info_message(
                        self, "Информация", 
                        "Нет результатов для предварительного просмотра. Сначала обработайте файл."
                    )
                    return
                
                preview_data = self.processing_thread.result
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
            
            # Создаем и показываем диалог preview
            preview_dialog = PreviewDialog(
                results=preview_data,
                model_type=model_type,
                file_path=file_path,
                parent=self
            )
            
            # Подключаем сигналы
            preview_dialog.results_edited.connect(self.on_preview_results_edited)
            preview_dialog.export_requested.connect(self.on_preview_export_requested)
            
            # Показываем диалог
            result = preview_dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                self.status_bar.showMessage("Изменения из предварительного просмотра применены")
            
        except Exception as e:
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
                # Single mode - обновляем processing_thread.result и таблицу
                if hasattr(self, 'processing_thread') and self.processing_thread and \
                   hasattr(self.processing_thread, 'result'):
                    self.processing_thread.result = edited_results
                
                # Обновляем отображение в таблице
                self.show_results(edited_results)
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

    def populate_cloud_providers(self):
        """Заполняет список облачных провайдеров."""
        try:
            self.cloud_provider_selector.clear()
            self.cloud_provider_selector.addItem("Выберите провайдера...", None)
            
            from .plugins.base_llm_plugin import LLM_PROVIDERS
            
            providers_added = 0
            llm_settings = settings_manager.get_setting('llm_providers', {})
            
            # Добавляем только облачных провайдеров (все кроме ollama)
            for provider_name, config in LLM_PROVIDERS.items():
                if provider_name != "ollama":  # Пропускаем локальные
                    # Проверяем настроенность провайдера
                    is_configured = False
                    if config.requires_api_key:
                        api_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
                        is_configured = bool(api_key)
                    else:
                        is_configured = True
                    
                    # Формируем название с индикатором
                    status_icon = "[OK]" if is_configured else "[CFG]"
                    display_name = f"{status_icon} {config.display_name}"
                    
                    self.cloud_provider_selector.addItem(display_name, {
                        'provider': provider_name,
                        'config': config,
                        'configured': is_configured
                    })
                    providers_added += 1
            
            print(f"[OK] Загружено {providers_added} облачных провайдеров")
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки облачных провайдеров: {e}")
            self.cloud_provider_selector.clear()
            self.cloud_provider_selector.addItem("Ошибка загрузки", None)

    def populate_local_providers(self):
        """Заполняет список локальных провайдеров."""
        try:
            self.local_provider_selector.clear()
            self.local_provider_selector.addItem("Выберите провайдера...", None)
            
            from .plugins.base_llm_plugin import LLM_PROVIDERS
            
            providers_added = 0
            
            # Добавляем только локальных провайдеров (пока только ollama)
            for provider_name, config in LLM_PROVIDERS.items():
                if provider_name == "ollama":  # Только локальные
                    # Проверяем доступность Ollama
                    is_available = self.check_ollama_availability()
                    
                    # Формируем название с индикатором
                    status_icon = "[OK]" if is_available else "[ERR]"
                    display_name = f"{status_icon} {config.display_name}"
                    
                    self.local_provider_selector.addItem(display_name, {
                        'provider': provider_name,
                        'config': config,
                        'available': is_available
                    })
                    providers_added += 1
            
            print(f"[OK] Загружено {providers_added} локальных провайдеров")
            
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
        except:
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
        
        # Заполняем модели
        self.populate_cloud_models_for_provider(provider_name, config, is_configured)
        self.update_cloud_llm_status()

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
        
        # Заполняем модели
        self.populate_local_models_for_provider(provider_name, config, is_available)
        self.update_local_llm_status()

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
                
                display_name = f"{model} {pricing_info} {vision_support}".strip()
                
                self.cloud_model_selector.addItem(display_name, {
                    'provider': provider_name,
                    'model': model,
                    'config': config,
                    'pricing': pricing_info
                })
                models_added += 1
                
                # Выбираем сохраненную модель
                if model == selected_model:
                    self.cloud_model_selector.setCurrentIndex(models_added - 1)
            
            self.cloud_model_selector.setEnabled(models_added > 0)
            print(f"[OK] Загружено {models_added} моделей для {config.display_name}")
            
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
                    # Добавляем информацию о модели
                    vision_support = "👁️" if "vision" in model.lower() else ""
                    size_info = self.get_model_size_info(model)
                    
                    display_name = f"{model} {size_info} {vision_support}".strip()
                    
                    self.local_model_selector.addItem(display_name, {
                        'provider': provider_name,
                        'model': model,
                        'config': config,
                        'size': size_info
                    })
                    models_added += 1
                    
                    # Выбираем сохраненную модель
                    if model == selected_model:
                        self.local_model_selector.setCurrentIndex(models_added - 1)
                
                self.local_model_selector.setEnabled(models_added > 0)
                print(f"[OK] Загружено {models_added} локальных моделей для {config.display_name}")
            
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
        except:
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
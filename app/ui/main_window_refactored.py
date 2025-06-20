"""
Refactored main window using component architecture.
"""
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QSplitter, QProgressBar, QPushButton,
    QStatusBar, QMenuBar, QMenu, QComboBox
)
from PyQt6.QtCore import Qt, QTimer

from app import config as app_config
from app.ui.components import (
    FileSelectorWidget, ImageViewerWidget,
    ModelSelectorWidget, ResultsViewerWidget,
    ProcessingController
)
from app.settings_dialog import ModelManagementDialog
from app.training_dialog import TrainingDialog
from app.ui.preview_dialog import PreviewDialog
from app.ui.export_template_designer import ExportTemplateDesigner
from app.ui.field_manager_dialog import FieldManagerDialog
from app.ui.llm_providers_dialog import LLMProvidersDialog
from app.ui.plugins_dialog import PluginsDialog


class MainWindowRefactored(QMainWindow):
    """Refactored main window with component architecture."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.file_selector = FileSelectorWidget()
        self.image_viewer = ImageViewerWidget()
        self.model_selector = ModelSelectorWidget()
        self.results_viewer = ResultsViewerWidget()
        self.processing_controller = ProcessingController()
        
        # UI elements
        self.progress_bar = QProgressBar()
        self.process_button = QPushButton(self.tr("Обработать"))
        self.ocr_lang_combo = QComboBox()
        
        # Dialogs
        self.model_management_dialog: Optional[ModelManagementDialog] = None
        self.training_dialog: Optional[TrainingDialog] = None
        self.preview_dialog: Optional[PreviewDialog] = None
        self.template_designer: Optional[ExportTemplateDesigner] = None
        self.field_manager_dialog: Optional[FieldManagerDialog] = None
        self.llm_providers_dialog: Optional[LLMProvidersDialog] = None
        self.plugins_dialog: Optional[PluginsDialog] = None
        
        self._init_ui()
        self._connect_signals()
        self._load_settings()
        
        # Schedule model population after UI init
        QTimer.singleShot(100, self._populate_models)
        
    def _init_ui(self):
        """Initialize UI layout."""
        self.setWindowTitle(f"{app_config.APP_NAME} v{app_config.APP_VERSION}")
        self.setMinimumSize(1024, 768)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - file selector and image viewer
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.file_selector)
        left_layout.addWidget(self.image_viewer, 1)
        
        # Right panel - model selector and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Model selector
        right_layout.addWidget(self.model_selector)
        
        # OCR language selector
        self._init_ocr_selector()
        right_layout.addWidget(self.ocr_lang_combo)
        
        # Process button
        self.process_button.setEnabled(False)
        right_layout.addWidget(self.process_button)
        
        # Progress bar
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # Results viewer
        right_layout.addWidget(self.results_viewer)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.tr("Готов к работе"))
        
        # Menu bar
        self._create_menus()
        
        # Set processing controller UI components
        self.processing_controller.set_ui_components(
            self.progress_bar, 
            self.process_button
        )
        
    def _init_ocr_selector(self):
        """Initialize OCR language selector."""
        self.ocr_lang_combo.addItem("English", "eng")
        self.ocr_lang_combo.addItem("Русский", "rus")
        self.ocr_lang_combo.addItem("English + Русский", "eng+rus")
        
    def _connect_signals(self):
        """Connect component signals."""
        # File selector signals
        self.file_selector.signals.file_selected.connect(self._on_file_selected)
        self.file_selector.signals.folder_selected.connect(self._on_folder_selected)
        self.file_selector.signals.selection_cleared.connect(self._on_selection_cleared)
        
        # Image viewer signals
        self.image_viewer.signals.image_loaded.connect(self._on_image_loaded)
        self.image_viewer.signals.image_error.connect(self._on_image_error)
        
        # Model selector signals
        self.model_selector.signals.model_changed.connect(self._on_model_changed)
        self.model_selector.signals.sub_model_changed.connect(self._on_sub_model_changed)
        self.model_selector.signals.prompt_requested.connect(self._show_model_prompt)
        
        # Results viewer signals
        self.results_viewer.signals.field_manager_requested.connect(
            self._show_field_manager_dialog
        )
        self.results_viewer.signals.template_designer_requested.connect(
            self._show_template_designer
        )
        self.results_viewer.signals.preview_requested.connect(
            self._show_preview_dialog
        )
        self.results_viewer.signals.export_requested.connect(
            self._export_results
        )
        
        # Processing controller signals
        self.processing_controller.signals.processing_finished.connect(
            self._on_processing_finished
        )
        self.processing_controller.signals.processing_error.connect(
            self._on_processing_error
        )
        
        # UI element signals
        self.process_button.clicked.connect(self._on_process_clicked)
        self.ocr_lang_combo.currentIndexChanged.connect(self._on_ocr_language_changed)
        
    def _create_menus(self):
        """Create menu bar and menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu(self.tr("Файл"))
        
        exit_action = file_menu.addAction(self.tr("Выход"))
        exit_action.triggered.connect(self.close)
        
        # Settings menu
        settings_menu = menubar.addMenu(self.tr("Настройки"))
        
        model_action = settings_menu.addAction(self.tr("Управление моделями"))
        model_action.triggered.connect(self._show_model_management_dialog)
        
        llm_action = settings_menu.addAction(self.tr("LLM провайдеры"))
        llm_action.triggered.connect(self._show_llm_providers_dialog)
        
        # Tools menu
        tools_menu = menubar.addMenu(self.tr("Инструменты"))
        
        training_action = tools_menu.addAction(self.tr("Обучение моделей"))
        training_action.triggered.connect(self._show_training_dialog)
        
        plugins_action = tools_menu.addAction(self.tr("Плагины"))
        plugins_action.triggered.connect(self._show_plugins_dialog)
        
        # Help menu
        help_menu = menubar.addMenu(self.tr("Помощь"))
        
        about_action = help_menu.addAction(self.tr("О программе"))
        about_action.triggered.connect(self._show_about_dialog)
        
    def _load_settings(self):
        """Load saved settings."""
        # OCR language
        from app.settings_manager import settings_manager
        ocr_lang = settings_manager.get_string('OCR', 'language', 'rus+eng')
        
        for i in range(self.ocr_lang_combo.count()):
            if self.ocr_lang_combo.itemData(i) == ocr_lang:
                self.ocr_lang_combo.setCurrentIndex(i)
                break
                
    def _populate_models(self):
        """Populate model selectors with available models."""
        # Populate Gemini models
        gemini_models = [
            ("Gemini 2.0 Flash Experimental", "models/gemini-2.0-flash-exp"),
            ("Gemini 1.5 Flash", "models/gemini-1.5-flash"),
            ("Gemini 1.5 Pro", "models/gemini-1.5-pro"),
        ]
        self.model_selector.populate_selector('gemini', 'model_selector', gemini_models)
        
        # Populate cloud providers
        self._populate_cloud_providers()
        
        # Populate local providers
        self._populate_local_providers()
        
    def _populate_cloud_providers(self):
        """Populate cloud LLM providers."""
        providers = [
            ("OpenAI", "openai"),
            ("Anthropic", "anthropic"),
            ("Cohere", "cohere"),
        ]
        self.model_selector.populate_selector('cloud_llm', 'provider_selector', providers)
        
    def _populate_local_providers(self):
        """Populate local LLM providers."""
        providers = [
            ("Ollama", "ollama"),
        ]
        self.model_selector.populate_selector('local_llm', 'provider_selector', providers)
        
    # Signal handlers
    def _on_file_selected(self, file_path: str):
        """Handle file selection."""
        self.logger.info(f"File selected: {file_path}")
        self.image_viewer.load_image(file_path)
        self.process_button.setEnabled(True)
        self.status_bar.showMessage(self.tr("Файл выбран: {0}").format(file_path))
        
    def _on_folder_selected(self, folder_path: str):
        """Handle folder selection."""
        self.logger.info(f"Folder selected: {folder_path}")
        self.process_button.setEnabled(True)
        self.status_bar.showMessage(self.tr("Папка выбрана: {0}").format(folder_path))
        
    def _on_selection_cleared(self):
        """Handle selection clearing."""
        self.image_viewer.clear()
        self.results_viewer.clear_results()
        self.process_button.setEnabled(False)
        self.status_bar.showMessage(self.tr("Готов к работе"))
        
    def _on_image_loaded(self, image_path: str):
        """Handle successful image loading."""
        self.logger.info(f"Image loaded: {image_path}")
        
    def _on_image_error(self, error_msg: str):
        """Handle image loading error."""
        self.logger.error(f"Image loading error: {error_msg}")
        self.status_bar.showMessage(self.tr("Ошибка загрузки: {0}").format(error_msg))
        
    def _on_model_changed(self, model_name: str):
        """Handle model selection change."""
        self.processing_controller.set_current_model(model_name)
        
        # Update OCR combo visibility
        self.ocr_lang_combo.setEnabled(model_name == 'layoutlm')
        
        self.status_bar.showMessage(
            self.tr("Выбрана модель: {0}").format(model_name)
        )
        
    def _on_sub_model_changed(self, model_name: str, sub_model: str):
        """Handle sub-model selection change."""
        self.logger.info(f"Sub-model changed: {model_name} -> {sub_model}")
        
    def _on_ocr_language_changed(self, index: int):
        """Handle OCR language change."""
        if index >= 0:
            lang = self.ocr_lang_combo.currentData()
            from app.settings_manager import settings_manager
            settings_manager.set_value('OCR', 'language', lang)
            app_config.DEFAULT_TESSERACT_LANG = lang
            
    def _on_process_clicked(self):
        """Handle process button click."""
        if self.file_selector.is_file_selected:
            self.processing_controller.process_file(
                self.file_selector.current_path
            )
        elif self.file_selector.is_folder_selected:
            self.processing_controller.process_folder(
                self.file_selector.current_path
            )
            
    def _on_processing_finished(self, results: dict):
        """Handle processing completion."""
        self.results_viewer.show_results(results)
        self.status_bar.showMessage(self.tr("Обработка завершена"))
        
    def _on_processing_error(self, error_msg: str):
        """Handle processing error."""
        self.status_bar.showMessage(
            self.tr("Ошибка обработки: {0}").format(error_msg)
        )
        
    # Dialog methods
    def _show_model_management_dialog(self):
        """Show model management dialog."""
        if not self.model_management_dialog:
            self.model_management_dialog = ModelManagementDialog(self)
        self.model_management_dialog.show()
        
    def _show_training_dialog(self):
        """Show training dialog."""
        if not self.training_dialog:
            # Получаем необходимые компоненты
            processors = self.processing_controller.model_manager.get_processors()
            
            # Создаем TrainingDialog с правильными параметрами
            self.training_dialog = TrainingDialog(
                app_config=app_config,
                ocr_processor=processors.get('ocr_processor'),
                gemini_processor=processors.get('gemini_processor'),
                parent=self
            )
        self.training_dialog.show()
        
    def _show_field_manager_dialog(self):
        """Show field manager dialog."""
        if not self.field_manager_dialog:
            self.field_manager_dialog = FieldManagerDialog(self)
            self.field_manager_dialog.fields_updated.connect(
                self.results_viewer.update_fields
            )
        self.field_manager_dialog.show()
        
    def _show_template_designer(self):
        """Show template designer."""
        if not self.template_designer:
            self.template_designer = ExportTemplateDesigner(self)
        self.template_designer.show()
        
    def _show_preview_dialog(self):
        """Show preview dialog."""
        if self.results_viewer.has_results:
            if not self.preview_dialog:
                self.preview_dialog = PreviewDialog(self)
            
            self.preview_dialog.set_results(
                self.results_viewer.current_results,
                self.results_viewer.get_column_headers()
            )
            self.preview_dialog.show()
            
    def _show_llm_providers_dialog(self):
        """Show LLM providers dialog."""
        if not self.llm_providers_dialog:
            self.llm_providers_dialog = LLMProvidersDialog(self)
        self.llm_providers_dialog.show()
        
    def _show_plugins_dialog(self):
        """Show plugins dialog."""
        if not self.plugins_dialog:
            self.plugins_dialog = PluginsDialog(
                self.processing_controller.universal_plugin_manager,
                self
            )
        self.plugins_dialog.show()
        
    def _show_model_prompt(self, model_name: str):
        """Show model prompt dialog."""
        # Implementation for showing model prompts
        pass
        
    def _show_about_dialog(self):
        """Show about dialog."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            self.tr("О программе"),
            f"{app_config.APP_NAME} v{app_config.APP_VERSION}\n\n"
            f"{app_config.APP_DESCRIPTION}"
        )
        
    def _export_results(self, format_type: str):
        """Export results in specified format."""
        # Implementation for exporting results
        pass
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.processing_controller.cleanup()
        event.accept() 
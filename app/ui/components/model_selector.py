"""
Model selector widget component.
"""
from typing import Dict, Optional, Tuple
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QRadioButton,
    QGroupBox, QComboBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt

from app.settings_manager import settings_manager


class ModelSelectorSignals(QObject):
    """Signals for model selector widget."""
    model_changed = pyqtSignal(str)  # model name
    sub_model_changed = pyqtSignal(str, str)  # model name, sub-model
    prompt_requested = pyqtSignal(str)  # model name
    
    
class ModelGroup:
    """Model group data."""
    def __init__(self, name: str, display_name: str, is_cloud: bool = False):
        self.name = name
        self.display_name = display_name
        self.is_cloud = is_cloud
        self.radio_button: Optional[QRadioButton] = None
        self.prompt_button: Optional[QPushButton] = None
        self.sub_widgets: Dict[str, QWidget] = {}


class ModelSelectorWidget(QWidget):
    """Widget for model selection."""
    
    CLOUD_MODELS = ['gemini', 'cloud_llm']
    LOCAL_MODELS = ['layoutlm', 'donut', 'local_llm']
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.signals = ModelSelectorSignals()
        
        self._model_groups: Dict[str, ModelGroup] = {}
        self._current_model: Optional[str] = None
        
        self._init_models()
        self._init_ui()
        self._load_settings()
        
    def _init_models(self):
        """Initialize model configurations."""
        # Cloud models
        self._model_groups['gemini'] = ModelGroup('gemini', 'Google Gemini', is_cloud=True)
        self._model_groups['cloud_llm'] = ModelGroup('cloud_llm', 'Другие облачные LLM', is_cloud=True)
        
        # Local models
        self._model_groups['layoutlm'] = ModelGroup('layoutlm', 'LayoutLMv3', is_cloud=False)
        self._model_groups['donut'] = ModelGroup('donut', 'Donut', is_cloud=False)
        self._model_groups['local_llm'] = ModelGroup('local_llm', 'Локальные LLM (Ollama)', is_cloud=False)
        
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create group box
        group_box = QGroupBox(self.tr("Выбор модели"))
        model_layout = QVBoxLayout()
        
        # Cloud models section
        self._add_section_header(model_layout, "☁️ Облачные модели", "#2196F3")
        
        # Add cloud models
        for model_name in self.CLOUD_MODELS:
            self._add_model_group(model_layout, model_name)
            
        # Separator
        separator = self._create_separator()
        model_layout.addWidget(separator)
        
        # Local models section
        self._add_section_header(model_layout, "🖥️ Локальные модели", "#4CAF50")
        
        # Add local models
        for model_name in self.LOCAL_MODELS:
            self._add_model_group(model_layout, model_name)
            
        group_box.setLayout(model_layout)
        layout.addWidget(group_box)
        
    def _add_section_header(self, layout: QVBoxLayout, text: str, color: str):
        """Add section header label."""
        label = QLabel(text)
        label.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 14px; padding: 8px 0px;")
        layout.addWidget(label)
        
    def _create_separator(self) -> QFrame:
        """Create horizontal separator."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #ccc; margin: 10px 0;")
        return separator
        
    def _add_model_group(self, layout: QVBoxLayout, model_name: str):
        """Add model group with radio button and prompt button."""
        model_group = self._model_groups[model_name]
        
        # Main layout for model
        model_layout = QHBoxLayout()
        
        # Radio button
        radio = QRadioButton(model_group.display_name)
        radio.toggled.connect(lambda checked: self._on_model_toggled(model_name, checked))
        model_group.radio_button = radio
        model_layout.addWidget(radio)
        
        # Prompt button
        prompt_btn = QPushButton(self.tr("Показать промпт"))
        prompt_btn.clicked.connect(lambda: self.signals.prompt_requested.emit(model_name))
        model_group.prompt_button = prompt_btn
        model_layout.addWidget(prompt_btn)
        
        layout.addLayout(model_layout)
        
        # Add sub-widgets based on model type
        if model_name == 'gemini':
            self._add_gemini_sub_widgets(layout, model_group)
        elif model_name == 'cloud_llm':
            self._add_cloud_llm_sub_widgets(layout, model_group)
        elif model_name == 'local_llm':
            self._add_local_llm_sub_widgets(layout, model_group)
            
    def _add_gemini_sub_widgets(self, layout: QVBoxLayout, model_group: ModelGroup):
        """Add Gemini-specific sub-widgets."""
        sub_layout = QHBoxLayout()
        sub_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        label = QLabel(self.tr("Модель:"))
        model_group.sub_widgets['model_label'] = label
        sub_layout.addWidget(label)
        
        selector = QComboBox()
        selector.currentIndexChanged.connect(
            lambda: self._on_sub_model_changed('gemini', selector)
        )
        model_group.sub_widgets['model_selector'] = selector
        sub_layout.addWidget(selector, 1)
        
        layout.addLayout(sub_layout)
        
    def _add_cloud_llm_sub_widgets(self, layout: QVBoxLayout, model_group: ModelGroup):
        """Add cloud LLM sub-widgets."""
        selection_layout = QVBoxLayout()
        selection_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel(self.tr("Провайдер:"))
        model_group.sub_widgets['provider_label'] = provider_label
        provider_layout.addWidget(provider_label)
        
        provider_selector = QComboBox()
        provider_selector.currentIndexChanged.connect(
            lambda: self._on_provider_changed('cloud_llm', provider_selector)
        )
        model_group.sub_widgets['provider_selector'] = provider_selector
        provider_layout.addWidget(provider_selector, 1)
        selection_layout.addLayout(provider_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel(self.tr("Модель:"))
        model_group.sub_widgets['model_label'] = model_label
        model_layout.addWidget(model_label)
        
        model_selector = QComboBox()
        model_selector.currentIndexChanged.connect(
            lambda: self._on_sub_model_changed('cloud_llm', model_selector)
        )
        model_group.sub_widgets['model_selector'] = model_selector
        model_layout.addWidget(model_selector, 1)
        selection_layout.addLayout(model_layout)
        
        # Status label
        status_label = QLabel(self.tr("Не настроено"))
        status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        model_group.sub_widgets['status_label'] = status_label
        selection_layout.addWidget(status_label)
        
        layout.addLayout(selection_layout)
        
    def _add_local_llm_sub_widgets(self, layout: QVBoxLayout, model_group: ModelGroup):
        """Add local LLM sub-widgets."""
        selection_layout = QVBoxLayout()
        selection_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel(self.tr("Провайдер:"))
        model_group.sub_widgets['provider_label'] = provider_label
        provider_layout.addWidget(provider_label)
        
        provider_selector = QComboBox()
        provider_selector.currentIndexChanged.connect(
            lambda: self._on_provider_changed('local_llm', provider_selector)
        )
        model_group.sub_widgets['provider_selector'] = provider_selector
        provider_layout.addWidget(provider_selector, 1)
        selection_layout.addLayout(provider_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel(self.tr("Модель:"))
        model_group.sub_widgets['model_label'] = model_label
        model_layout.addWidget(model_label)
        
        model_selector = QComboBox()
        model_selector.currentIndexChanged.connect(
            lambda: self._on_sub_model_changed('local_llm', model_selector)
        )
        model_group.sub_widgets['model_selector'] = model_selector
        model_layout.addWidget(model_selector, 1)
        selection_layout.addLayout(model_layout)
        
        # Status label
        status_label = QLabel(self.tr("Не настроено"))
        status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        model_group.sub_widgets['status_label'] = status_label
        selection_layout.addWidget(status_label)
        
        layout.addLayout(selection_layout)
        
    def _on_model_toggled(self, model_name: str, checked: bool):
        """Handle model radio button toggle."""
        if checked:
            self._current_model = model_name
            self._update_sub_widgets_visibility()
            self.signals.model_changed.emit(model_name)
            settings_manager.set_active_model(model_name)
            self.logger.info(f"Model selected: {model_name}")
            
    def _on_sub_model_changed(self, model_name: str, selector: QComboBox):
        """Handle sub-model selection change."""
        if selector.currentIndex() >= 0:
            sub_model = selector.currentData() or selector.currentText()
            self.signals.sub_model_changed.emit(model_name, sub_model)
            
    def _on_provider_changed(self, model_name: str, selector: QComboBox):
        """Handle provider selection change."""
        # This will be handled by the parent widget
        pass
        
    def _update_sub_widgets_visibility(self):
        """Update visibility of sub-widgets based on selected model."""
        for model_name, model_group in self._model_groups.items():
            enabled = (model_name == self._current_model)
            
            for widget in model_group.sub_widgets.values():
                widget.setEnabled(enabled)
                
    def _load_settings(self):
        """Load saved settings."""
        # Load active model
        active_model = settings_manager.get_active_model()
        if active_model in self._model_groups:
            self._model_groups[active_model].radio_button.setChecked(True)
        else:
            # Default to LayoutLM
            self._model_groups['layoutlm'].radio_button.setChecked(True)
            
    @property
    def current_model(self) -> Optional[str]:
        """Get currently selected model."""
        return self._current_model
        
    def get_sub_widget(self, model_name: str, widget_name: str) -> Optional[QWidget]:
        """Get sub-widget for a model."""
        if model_name in self._model_groups:
            return self._model_groups[model_name].sub_widgets.get(widget_name)
        return None
        
    def populate_selector(self, model_name: str, selector_name: str, items: list):
        """Populate a combo box selector."""
        selector = self.get_sub_widget(model_name, selector_name)
        if isinstance(selector, QComboBox):
            selector.clear()
            for item in items:
                if isinstance(item, tuple):
                    selector.addItem(item[0], item[1])
                else:
                    selector.addItem(str(item))
                    
    def update_status(self, model_name: str, status: str, color: str = "#666"):
        """Update status label for a model."""
        status_label = self.get_sub_widget(model_name, 'status_label')
        if isinstance(status_label, QLabel):
            status_label.setText(status)
            status_label.setStyleSheet(f"color: {color}; font-size: 11px; padding: 2px 0;") 
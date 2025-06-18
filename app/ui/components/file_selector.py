"""
File selector widget component.
"""
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog,
    QGroupBox
)
from PyQt6.QtCore import pyqtSignal, QObject
import logging


class FileSelectorSignals(QObject):
    """Signals for file selector widget."""
    file_selected = pyqtSignal(str)  # file path
    folder_selected = pyqtSignal(str)  # folder path
    selection_cleared = pyqtSignal()


class FileSelectorWidget(QWidget):
    """Widget for file and folder selection."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.signals = FileSelectorSignals()
        
        self._current_file_path: Optional[Path] = None
        self._current_folder_path: Optional[Path] = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create group box
        group_box = QGroupBox(self.tr("Выбор файла"))
        group_layout = QVBoxLayout()
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # File selection button
        self.select_file_button = QPushButton(self.tr("Выбрать файл"))
        self.select_file_button.clicked.connect(self._on_select_file)
        buttons_layout.addWidget(self.select_file_button)
        
        # Folder selection button
        self.select_folder_button = QPushButton(self.tr("Выбрать папку"))
        self.select_folder_button.clicked.connect(self._on_select_folder)
        buttons_layout.addWidget(self.select_folder_button)
        
        # Clear button
        self.clear_button = QPushButton(self.tr("Очистить"))
        self.clear_button.clicked.connect(self._on_clear_selection)
        self.clear_button.setEnabled(False)
        buttons_layout.addWidget(self.clear_button)
        
        group_layout.addLayout(buttons_layout)
        
        # Selection label
        self.selection_label = QLabel(self.tr("Файл или папка не выбраны"))
        self.selection_label.setWordWrap(True)
        group_layout.addWidget(self.selection_label)
        
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        
    def _on_select_file(self):
        """Handle file selection."""
        file_filter = self.tr("Файлы изображений и PDF (*.png *.jpg *.jpeg *.pdf);;Все файлы (*.*)")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Выберите файл"),
            str(self._current_file_path.parent) if self._current_file_path else "",
            file_filter
        )
        
        if file_path:
            self._set_file_path(file_path)
            
    def _on_select_folder(self):
        """Handle folder selection."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Выберите папку"),
            str(self._current_folder_path) if self._current_folder_path else ""
        )
        
        if folder_path:
            self._set_folder_path(folder_path)
            
    def _on_clear_selection(self):
        """Clear current selection."""
        self._current_file_path = None
        self._current_folder_path = None
        self.selection_label.setText(self.tr("Файл или папка не выбраны"))
        self.clear_button.setEnabled(False)
        self.signals.selection_cleared.emit()
        
    def _set_file_path(self, path: str):
        """Set current file path."""
        self._current_file_path = Path(path)
        self._current_folder_path = None
        self.selection_label.setText(self.tr("Файл: {0}").format(path))
        self.clear_button.setEnabled(True)
        self.signals.file_selected.emit(path)
        self.logger.info(f"File selected: {path}")
        
    def _set_folder_path(self, path: str):
        """Set current folder path."""
        self._current_folder_path = Path(path)
        self._current_file_path = None
        self.selection_label.setText(self.tr("Папка: {0}").format(path))
        self.clear_button.setEnabled(True)
        self.signals.folder_selected.emit(path)
        self.logger.info(f"Folder selected: {path}")
        
    @property
    def current_path(self) -> Optional[Path]:
        """Get current selected path (file or folder)."""
        return self._current_file_path or self._current_folder_path
        
    @property
    def is_file_selected(self) -> bool:
        """Check if a file is selected."""
        return self._current_file_path is not None
        
    @property
    def is_folder_selected(self) -> bool:
        """Check if a folder is selected."""
        return self._current_folder_path is not None
        
    def set_enabled(self, enabled: bool):
        """Enable or disable the widget."""
        self.select_file_button.setEnabled(enabled)
        self.select_folder_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled and (self._current_file_path or self._current_folder_path))


# Alias for backward compatibility
FileSelector = FileSelectorWidget 
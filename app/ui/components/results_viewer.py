"""
Results viewer widget component.
"""
from typing import Dict, List, Any, Optional
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QGroupBox, QHeaderView,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt

from app.field_manager import field_manager
from app.settings_manager import settings_manager


class ResultsViewerSignals(QObject):
    """Signals for results viewer widget."""
    field_manager_requested = pyqtSignal()
    template_designer_requested = pyqtSignal()
    preview_requested = pyqtSignal()
    export_requested = pyqtSignal(str)  # format
    results_updated = pyqtSignal(dict)


class ResultsViewerWidget(QWidget):
    """Widget for displaying extraction results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.signals = ResultsViewerSignals()
        
        self._current_results: Dict[str, Any] = {}
        self._has_results = False
        
        self._init_ui()
        self._setup_table()
        
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create group box
        group_box = QGroupBox(self.tr("Результаты"))
        results_layout = QVBoxLayout()
        
        # Header with field manager button
        header_layout = QHBoxLayout()
        header_label = QLabel(self.tr("Извлечённые данные:"))
        header_layout.addWidget(header_label)
        
        # Field manager button
        self.field_manager_button = QPushButton("🔧 " + self.tr("Управление полями"))
        self.field_manager_button.clicked.connect(
            lambda: self.signals.field_manager_requested.emit()
        )
        self.field_manager_button.setToolTip(
            self.tr("Управление полями таблицы и синхронизация промптов")
        )
        header_layout.addWidget(self.field_manager_button)
        
        results_layout.addLayout(header_layout)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        # Template designer button
        self.template_button = QPushButton("🎨 " + self.tr("Дизайнер шаблонов"))
        self.template_button.clicked.connect(
            lambda: self.signals.template_designer_requested.emit()
        )
        self.template_button.setToolTip(self.tr("Создать и настроить шаблоны экспорта"))
        self.template_button.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }"
        )
        buttons_layout.addWidget(self.template_button)
        
        # Preview button
        self.preview_button = QPushButton("🔍 " + self.tr("Предварительный просмотр"))
        self.preview_button.clicked.connect(
            lambda: self.signals.preview_requested.emit()
        )
        self.preview_button.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; }"
        )
        self.preview_button.setEnabled(False)
        buttons_layout.addWidget(self.preview_button)
        
        # Save button
        self.save_button = QPushButton(self.tr("Сохранить как..."))
        self.save_button.clicked.connect(self._on_save_as)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        # Excel button
        self.excel_button = QPushButton(self.tr("Сохранить в Excel"))
        self.excel_button.clicked.connect(self._on_save_excel)
        self.excel_button.setEnabled(False)
        buttons_layout.addWidget(self.excel_button)
        
        results_layout.addLayout(buttons_layout)
        
        group_box.setLayout(results_layout)
        layout.addWidget(group_box)
        
    def _setup_table(self):
        """Setup results table based on field configuration."""
        try:
            # Get configured fields
            fields = field_manager.get_all_fields()
            
            if not fields:
                self._setup_fallback_table()
                return
                
            # Extract column info
            column_names = []
            column_keys = []
            
            for field in fields:
                if field.get('enabled', True):
                    column_names.append(field.get('display_name', field.get('name', '')))
                    column_keys.append(field.get('key', field.get('name', '')))
                    
            if not column_names:
                self._setup_fallback_table()
                return
                
            # Setup table
            self.results_table.setColumnCount(len(column_names))
            self.results_table.setHorizontalHeaderLabels(column_names)
            
            # Store column mapping
            self._column_mapping = dict(zip(column_keys, range(len(column_keys))))
            
            # Configure table
            header = self.results_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            self.logger.info(f"Table configured with {len(column_names)} columns")
            
        except Exception as e:
            self.logger.error(f"Failed to setup table: {e}")
            self._setup_fallback_table()
            
    def _setup_fallback_table(self):
        """Setup basic fallback table structure."""
        basic_columns = [
            self.tr("Поле"),
            self.tr("Значение")
        ]
        
        self.results_table.setColumnCount(len(basic_columns))
        self.results_table.setHorizontalHeaderLabels(basic_columns)
        
        # Simple mapping
        self._column_mapping = {
            'field': 0,
            'value': 1
        }
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.logger.warning("Using fallback table structure")
        
    def show_results(self, results: Dict[str, Any]):
        """Display extraction results in the table."""
        try:
            self._current_results = results
            self._has_results = True
            
            # Clear existing data
            self.results_table.setRowCount(0)
            
            # Handle different result formats
            if isinstance(results, dict):
                if 'predictions' in results:
                    # LayoutLM/Donut format
                    self._show_predictions_format(results['predictions'])
                elif any(key in results for key in ['invoice_number', 'invoice_date', 'total']):
                    # Direct field format
                    self._show_direct_format(results)
                else:
                    # Generic format
                    self._show_generic_format(results)
            else:
                self.logger.warning(f"Unexpected results format: {type(results)}")
                
            # Enable export buttons
            self._enable_export_buttons(True)
            
            # Emit signal
            self.signals.results_updated.emit(results)
            
        except Exception as e:
            self.logger.error(f"Failed to show results: {e}")
            self._show_error_message(
                self.tr("Ошибка отображения результатов"),
                str(e)
            )
            
    def _show_predictions_format(self, predictions: List[Dict]):
        """Show results in predictions format (LayoutLM/Donut)."""
        # Group predictions by field
        field_values = {}
        
        for pred in predictions:
            label = pred.get('label', 'UNKNOWN')
            value = pred.get('text', '')
            score = pred.get('score', 0.0)
            
            if label not in field_values:
                field_values[label] = []
            field_values[label].append((value, score))
            
        # Add rows to table
        row = 0
        for field, values in field_values.items():
            # Use highest scoring value
            if values:
                best_value = max(values, key=lambda x: x[1])
                self._add_table_row(row, field, best_value[0])
                row += 1
                
    def _show_direct_format(self, results: Dict):
        """Show results in direct field format."""
        # If we have field mapping, use it
        if hasattr(self, '_column_mapping') and len(self._column_mapping) > 2:
            # Multi-column format
            self.results_table.setRowCount(1)
            
            for field_key, col_index in self._column_mapping.items():
                value = results.get(field_key, '')
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(0, col_index, item)
        else:
            # Two-column format
            row = 0
            for field, value in results.items():
                self._add_table_row(row, field, value)
                row += 1
                
    def _show_generic_format(self, results: Dict):
        """Show results in generic format."""
        row = 0
        for field, value in results.items():
            if not field.startswith('_'):  # Skip private fields
                self._add_table_row(row, field, value)
                row += 1
                
    def _add_table_row(self, row: int, field: str, value: Any):
        """Add a row to the results table."""
        if self.results_table.columnCount() == 2:
            # Two-column format
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(field)))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))
        else:
            # Multi-column format - handled by caller
            pass
            
    def clear_results(self):
        """Clear current results."""
        self._current_results = {}
        self._has_results = False
        self.results_table.setRowCount(0)
        self._enable_export_buttons(False)
        
    def update_fields(self):
        """Update table structure based on field changes."""
        self._setup_table()
        
        # Re-display current results if any
        if self._has_results and self._current_results:
            self.show_results(self._current_results)
            
    def _enable_export_buttons(self, enabled: bool):
        """Enable or disable export buttons."""
        self.preview_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.excel_button.setEnabled(enabled)
        
    def _on_save_as(self):
        """Handle save as button click."""
        if self._has_results:
            self.signals.export_requested.emit('json')
            
    def _on_save_excel(self):
        """Handle save to Excel button click."""
        if self._has_results:
            self.signals.export_requested.emit('excel')
            
    def _show_error_message(self, title: str, message: str):
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
        
    @property
    def current_results(self) -> Dict[str, Any]:
        """Get current results."""
        return self._current_results
        
    @property
    def has_results(self) -> bool:
        """Check if there are results to display."""
        return self._has_results
        
    def get_table_data(self) -> List[List[str]]:
        """Get table data as list of rows."""
        data = []
        
        for row in range(self.results_table.rowCount()):
            row_data = []
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                row_data.append(item.text() if item else '')
            data.append(row_data)
            
        return data
        
    def get_column_headers(self) -> List[str]:
        """Get column headers."""
        headers = []
        for col in range(self.results_table.columnCount()):
            header = self.results_table.horizontalHeaderItem(col)
            headers.append(header.text() if header else f"Column {col}")
        return headers 
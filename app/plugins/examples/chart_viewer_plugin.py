"""
Chart Viewer Plugin для InvoiceGemini
Плагин для отображения данных в виде графиков и диаграмм
"""
from typing import Dict, Any, List
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QComboBox, QLabel, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..base_plugin import ViewerPlugin, PluginMetadata, PluginType, PluginCapability, PluginStatus


class ChartViewerPlugin(ViewerPlugin):
    """Плагин для отображения данных в виде графиков"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chart_types = ['bar', 'pie', 'line', 'scatter', 'histogram']
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Chart Viewer",
            version="1.0.0",
            description="Просмотр данных счетов в виде графиков и диаграмм",
            author="InvoiceGemini Team",
            plugin_type=PluginType.VIEWER,
            capabilities=[PluginCapability.VISION, PluginCapability.TEXT],
            dependencies=['matplotlib', 'pandas', 'numpy'],
            config_schema={
                "required": [],
                "types": {
                    "default_chart_type": str,
                    "show_legend": bool,
                    "grid_enabled": bool
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        if not MATPLOTLIB_AVAILABLE:
            self.set_error("Matplotlib не установлен")
            return False
        
        self.status = PluginStatus.LOADED
        return True
    
    def cleanup(self):
        """Очистка ресурсов"""
        plt.close('all')
    
    def create_viewer(self, data: Any, parent=None) -> Any:
        """
        Создает виджет для отображения графиков
        
        Args:
            data: Данные для отображения
            parent: Родительский виджет
            
        Returns:
            QWidget с графиками
        """
        try:
            if not MATPLOTLIB_AVAILABLE:
                from PyQt6.QtWidgets import QLabel
                error_label = QLabel("Matplotlib не установлен")
                return error_label
            
            # Создаем главный виджет
            main_widget = QWidget(parent)
            layout = QVBoxLayout(main_widget)
            
            # Панель управления
            control_panel = self._create_control_panel(main_widget)
            layout.addWidget(control_panel)
            
            # Сплиттер для графика и таблицы
            splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Виджет для matplotlib
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            splitter.addWidget(self.canvas)
            
            # Таблица с исходными данными
            self.data_table = QTableWidget()
            splitter.addWidget(self.data_table)
            
            # Устанавливаем пропорции
            splitter.setSizes([700, 300])
            
            layout.addWidget(splitter)
            
            # Сохраняем данные и обновляем отображение
            self.current_data = data
            self._populate_data_table(data)
            self._update_chart()
            
            return main_widget
            
        except Exception as e:
            self.set_error(f"Ошибка создания просмотрщика: {e}")
            from PyQt6.QtWidgets import QLabel
            error_label = QLabel(f"Ошибка: {e}")
            return error_label
    
    def _create_control_panel(self, parent) -> QWidget:
        """Создает панель управления графиками"""
        panel = QGroupBox("Настройки графика", parent)
        layout = QHBoxLayout(panel)
        
        # Тип графика
        layout.addWidget(QLabel("Тип:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['Столбчатая диаграмма', 'Круговая диаграмма', 
                                       'Линейный график', 'Точечная диаграмма', 'Гистограмма'])
        self.chart_type_combo.currentTextChanged.connect(self._on_chart_type_changed)
        layout.addWidget(self.chart_type_combo)
        
        # Поле для анализа
        layout.addWidget(QLabel("Поле:"))
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self._update_chart)
        layout.addWidget(self.field_combo)
        
        # Топ N записей
        layout.addWidget(QLabel("Показать топ:"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setMinimum(5)
        self.top_n_spin.setMaximum(100)
        self.top_n_spin.setValue(10)
        self.top_n_spin.valueChanged.connect(self._update_chart)
        layout.addWidget(self.top_n_spin)
        
        # Показать легенду
        self.show_legend_cb = QCheckBox("Легенда")
        self.show_legend_cb.setChecked(True)
        self.show_legend_cb.toggled.connect(self._update_chart)
        layout.addWidget(self.show_legend_cb)
        
        # Показать сетку
        self.show_grid_cb = QCheckBox("Сетка")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(self._update_chart)
        layout.addWidget(self.show_grid_cb)
        
        # Кнопка обновления
        refresh_btn = QPushButton("Обновить")
        refresh_btn.clicked.connect(self._update_chart)
        layout.addWidget(refresh_btn)
        
        return panel
    
    def _populate_data_table(self, data):
        """Заполняет таблицу данными"""
        try:
            if isinstance(data, list) and data:
                # Преобразуем в DataFrame для удобства
                df = pd.DataFrame(data)
                
                # Настраиваем таблицу
                self.data_table.setRowCount(len(df))
                self.data_table.setColumnCount(len(df.columns))
                self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
                
                # Заполняем данными
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        value = str(df.iloc[row, col])
                        item = QTableWidgetItem(value)
                        self.data_table.setItem(row, col, item)
                
                # Заполняем комбобокс полей
                self.field_combo.clear()
                # Добавляем числовые поля
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_columns:
                    self.field_combo.addItem(col)
                
                # Если нет числовых полей, добавляем все
                if not numeric_columns:
                    for col in df.columns:
                        self.field_combo.addItem(col)
                        
        except Exception as e:
            print(f"Ошибка заполнения таблицы: {e}")
    
    def _on_chart_type_changed(self, chart_name):
        """Обработчик изменения типа графика"""
        self._update_chart()
    
    def _update_chart(self):
        """Обновляет график"""
        try:
            if not hasattr(self, 'current_data') or not self.current_data:
                return
            
            # Очищаем предыдущий график
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Преобразуем данные
            df = pd.DataFrame(self.current_data)
            
            # Получаем выбранное поле
            selected_field = self.field_combo.currentText()
            if not selected_field or selected_field not in df.columns:
                ax.text(0.5, 0.5, 'Выберите поле для отображения', 
                       ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
                return
            
            # Получаем тип графика
            chart_type_map = {
                'Столбчатая диаграмма': 'bar',
                'Круговая диаграмма': 'pie', 
                'Линейный график': 'line',
                'Точечная диаграмма': 'scatter',
                'Гистограмма': 'histogram'
            }
            chart_type = chart_type_map.get(self.chart_type_combo.currentText(), 'bar')
            
            # Строим график
            self._create_chart(ax, df, selected_field, chart_type)
            
            # Настройки отображения
            if self.show_grid_cb.isChecked():
                ax.grid(True, alpha=0.3)
            
            ax.set_title(f'{selected_field} - {self.chart_type_combo.currentText()}')
            
            # Обновляем canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Ошибка обновления графика: {e}")
    
    def _create_chart(self, ax, df, field, chart_type):
        """Создает конкретный тип графика"""
        try:
            top_n = self.top_n_spin.value()
            
            if chart_type == 'bar':
                # Столбчатая диаграмма - группируем по значениям
                value_counts = df[field].value_counts().head(top_n)
                ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45)
                ax.set_ylabel('Количество')
                
            elif chart_type == 'pie':
                # Круговая диаграмма
                value_counts = df[field].value_counts().head(top_n)
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                
            elif chart_type == 'line':
                # Линейный график
                if pd.api.types.is_numeric_dtype(df[field]):
                    ax.plot(df[field].head(top_n).values)
                    ax.set_ylabel(field)
                    ax.set_xlabel('Индекс записи')
                else:
                    value_counts = df[field].value_counts().head(top_n)
                    ax.plot(range(len(value_counts)), value_counts.values, marker='o')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45)
                    
            elif chart_type == 'scatter':
                # Точечная диаграмма
                if pd.api.types.is_numeric_dtype(df[field]):
                    y_values = df[field].head(top_n).values
                    x_values = range(len(y_values))
                    ax.scatter(x_values, y_values)
                    ax.set_xlabel('Индекс записи')
                    ax.set_ylabel(field)
                    
            elif chart_type == 'histogram':
                # Гистограмма
                if pd.api.types.is_numeric_dtype(df[field]):
                    ax.hist(df[field].dropna().values, bins=min(20, top_n))
                    ax.set_xlabel(field)
                    ax.set_ylabel('Частота')
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка построения графика: {e}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def update_view(self, viewer: Any, data: Any):
        """Обновляет отображение данных"""
        try:
            self.current_data = data
            self._populate_data_table(data)
            self._update_chart()
        except Exception as e:
            self.set_error(f"Ошибка обновления вида: {e}")
    
    def get_supported_data_types(self) -> List[str]:
        """Возвращает поддерживаемые типы данных"""
        return ['list', 'dict', 'dataframe']


class StatisticsViewerPlugin(ViewerPlugin):
    """Плагин для отображения статистики по счетам"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Statistics Viewer",
            version="1.0.0",
            description="Отображение статистики и аналитики по счетам",
            author="InvoiceGemini Team",
            plugin_type=PluginType.VIEWER,
            capabilities=[PluginCapability.TEXT, PluginCapability.BATCH],
            config_schema={
                "required": [],
                "types": {
                    "show_charts": bool,
                    "precision": int
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        self.status = PluginStatus.LOADED
        return True
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def create_viewer(self, data: Any, parent=None) -> Any:
        """Создает виджет статистики"""
        try:
            from PyQt6.QtWidgets import QScrollArea, QTextEdit
            
            # Создаем скроллируемую область
            scroll_area = QScrollArea(parent)
            
            # Текстовый виджет для статистики
            text_widget = QTextEdit()
            text_widget.setReadOnly(True)
            
            # Генерируем статистику
            stats_text = self._generate_statistics(data)
            text_widget.setHtml(stats_text)
            
            scroll_area.setWidget(text_widget)
            return scroll_area
            
        except Exception as e:
            self.set_error(f"Ошибка создания статистики: {e}")
            from PyQt6.QtWidgets import QLabel
            return QLabel(f"Ошибка: {e}")
    
    def _generate_statistics(self, data) -> str:
        """Генерирует HTML со статистикой"""
        try:
            if not data or not isinstance(data, list):
                return "<h3>Нет данных для анализа</h3>"
            
            df = pd.DataFrame(data)
            
            html = "<h2>📊 Статистика по счетам</h2>"
            
            # Общая информация
            html += f"<h3>📋 Общая информация</h3>"
            html += f"<ul>"
            html += f"<li><b>Всего записей:</b> {len(df)}</li>"
            html += f"<li><b>Количество полей:</b> {len(df.columns)}</li>"
            html += f"<li><b>Пустых значений:</b> {df.isnull().sum().sum()}</li>"
            html += f"</ul>"
            
            # Анализ по полям
            html += "<h3>📈 Анализ полей</h3>"
            
            for column in df.columns:
                html += f"<h4>🔹 {column}</h4>"
                html += "<ul>"
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Числовые поля
                    non_null_values = df[column].dropna()
                    if len(non_null_values) > 0:
                        html += f"<li>Мин: {non_null_values.min():.2f}</li>"
                        html += f"<li>Макс: {non_null_values.max():.2f}</li>"
                        html += f"<li>Среднее: {non_null_values.mean():.2f}</li>"
                        html += f"<li>Медиана: {non_null_values.median():.2f}</li>"
                    else:
                        html += "<li>Нет числовых данных</li>"
                else:
                    # Текстовые поля
                    unique_values = df[column].nunique()
                    most_common = df[column].value_counts().head(3)
                    
                    html += f"<li>Уникальных значений: {unique_values}</li>"
                    html += f"<li>Наиболее частые:</li>"
                    html += "<ul>"
                    for value, count in most_common.items():
                        html += f"<li>{value}: {count} раз</li>"
                    html += "</ul>"
                
                html += f"<li>Пустых значений: {df[column].isnull().sum()}</li>"
                html += "</ul>"
            
            return html
            
        except Exception as e:
            return f"<h3>Ошибка генерации статистики: {e}</h3>"
    
    def update_view(self, viewer: Any, data: Any):
        """Обновляет статистику"""
        try:
            if hasattr(viewer, 'widget') and hasattr(viewer.widget(), 'setHtml'):
                stats_text = self._generate_statistics(data)
                viewer.widget().setHtml(stats_text)
        except Exception as e:
            self.set_error(f"Ошибка обновления статистики: {e}")
    
    def get_supported_data_types(self) -> List[str]:
        """Возвращает поддерживаемые типы данных"""
        return ['list', 'dict', 'dataframe'] 
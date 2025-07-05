"""
Оптимизированные UI компоненты для повышения производительности
"""

import sys
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QProgressBar, QComboBox, QTextEdit, QListWidget,
    QListWidgetItem, QStyledItemDelegate, QApplication
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QRect, QSize, QPoint,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QAbstractItemModel, QModelIndex, QAbstractTableModel
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QFont, QColor, QPainter, QPen, QBrush,
    QFontMetrics, QTextOption, QTextDocument
)

logger = logging.getLogger(__name__)


@dataclass
class RenderingConfig:
    """Конфигурация рендеринга для оптимизации"""
    enable_animations: bool = True
    lazy_loading: bool = True
    virtual_scrolling: bool = True
    batch_updates: bool = True
    cache_rendered_items: bool = True
    max_visible_items: int = 50
    animation_duration: int = 200
    scroll_step: int = 3


class LazyLoadingMixin:
    """Миксин для ленивой загрузки данных"""
    
    def __init__(self):
        self._data_loader: Optional[Callable] = None
        self._loading_timer = QTimer()
        self._loading_timer.setSingleShot(True)
        self._loading_timer.timeout.connect(self._load_data)
        self._is_loading = False
        
    def set_data_loader(self, loader: Callable):
        """Устанавливает функцию загрузки данных"""
        self._data_loader = loader
        
    def trigger_load(self, delay_ms: int = 100):
        """Запускает загрузку с задержкой"""
        if self._is_loading:
            return
            
        self._loading_timer.start(delay_ms)
        
    def _load_data(self):
        """Загружает данные в фоновом потоке"""
        if self._data_loader and not self._is_loading:
            self._is_loading = True
            try:
                self._data_loader()
            finally:
                self._is_loading = False


class VirtualScrollArea(QScrollArea):
    """Виртуальная прокрутка для больших списков"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._items: List[Any] = []
        self._visible_items: List[QWidget] = []
        self._item_height = 50
        self._viewport_start = 0
        self._viewport_end = 0
        self._max_visible = 20
        
        # Оптимизация прокрутки
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
    def set_items(self, items: List[Any]):
        """Устанавливает элементы для отображения"""
        self._items = items
        self._update_viewport()
        
    def _on_scroll(self, value: int):
        """Обработка прокрутки"""
        self._update_viewport()
        
    def _update_viewport(self):
        """Обновляет видимые элементы"""
        if not self._items:
            return
            
        # Определяем видимый диапазон
        scroll_value = self.verticalScrollBar().value()
        start_index = max(0, scroll_value // self._item_height)
        end_index = min(len(self._items), start_index + self._max_visible)
        
        # Обновляем только если изменился диапазон
        if start_index != self._viewport_start or end_index != self._viewport_end:
            self._viewport_start = start_index
            self._viewport_end = end_index
            self._render_visible_items()
            
    def _render_visible_items(self):
        """Рендерит только видимые элементы"""
        # Очищаем старые виджеты
        for widget in self._visible_items:
            widget.deleteLater()
        self._visible_items.clear()
        
        # Создаем новые виджеты только для видимых элементов
        for i in range(self._viewport_start, self._viewport_end):
            if i < len(self._items):
                widget = self._create_item_widget(self._items[i])
                self._visible_items.append(widget)
                
    def _create_item_widget(self, item: Any) -> QWidget:
        """Создает виджет для элемента (переопределяется в подклассах)"""
        widget = QLabel(str(item))
        widget.setFixedHeight(self._item_height)
        return widget


class OptimizedTableWidget(QTableWidget):
    """Оптимизированная таблица с виртуальной прокруткой"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Оптимизации производительности
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setShowGrid(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # Виртуальная прокрутка
        self.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        
        # Оптимизация заголовков
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        
        # Кэширование данных
        self._row_cache: Dict[int, Dict[int, Any]] = {}
        self._cache_timeout = 30000  # 30 секунд
        self._cache_timer = QTimer()
        self._cache_timer.timeout.connect(self._clear_cache)
        self._cache_timer.start(self._cache_timeout)
        
    def setData(self, data: List[Dict], headers: List[str] = None):
        """Устанавливает данные с оптимизацией"""
        if not data:
            return
            
        # Отключаем сортировку для быстрой загрузки
        self.setSortingEnabled(False)
        
        # Устанавливаем заголовки
        if headers:
            self.setColumnCount(len(headers))
            self.setHorizontalHeaderLabels(headers)
        else:
            self.setColumnCount(len(data[0]) if data else 0)
            
        # Устанавливаем количество строк
        self.setRowCount(len(data))
        
        # Заполняем данные батчами для отзывчивости
        self._fill_data_batched(data)
        
        # Включаем сортировку обратно
        self.setSortingEnabled(True)
        
    def _fill_data_batched(self, data: List[Dict], batch_size: int = 100):
        """Заполняет данные батчами"""
        QApplication.processEvents()  # Обновляем UI
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            for row_idx, row_data in enumerate(batch):
                actual_row = i + row_idx
                
                if isinstance(row_data, dict):
                    for col_idx, value in enumerate(row_data.values()):
                        item = QTableWidgetItem(str(value))
                        self.setItem(actual_row, col_idx, item)
                elif isinstance(row_data, (list, tuple)):
                    for col_idx, value in enumerate(row_data):
                        item = QTableWidgetItem(str(value))
                        self.setItem(actual_row, col_idx, item)
                        
            # Обновляем UI между батчами
            QApplication.processEvents()
            
    def _clear_cache(self):
        """Очищает кэш данных"""
        self._row_cache.clear()
        
    def optimizeColumns(self):
        """Оптимизирует ширину колонок"""
        for column in range(self.columnCount()):
            self.resizeColumnToContents(column)
            
            # Ограничиваем максимальную ширину
            max_width = 300
            if self.columnWidth(column) > max_width:
                self.setColumnWidth(column, max_width)


class AnimatedButton(QPushButton):
    """Кнопка с анимацией наведения"""
    
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        
        self._animation = QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self._original_style = self.styleSheet()
        self._hover_style = """
            QPushButton {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: translateY(-2px);
            }
        """
        
        self.setStyleSheet(self._hover_style)
        
    def enterEvent(self, event):
        """Анимация при наведении"""
        super().enterEvent(event)
        self._animate_scale(1.05)
        
    def leaveEvent(self, event):
        """Анимация при уходе курсора"""
        super().leaveEvent(event)
        self._animate_scale(1.0)
        
    def _animate_scale(self, scale: float):
        """Анимация масштабирования"""
        rect = self.geometry()
        center = rect.center()
        
        new_width = int(rect.width() * scale)
        new_height = int(rect.height() * scale)
        
        new_rect = QRect(
            center.x() - new_width // 2,
            center.y() - new_height // 2,
            new_width,
            new_height
        )
        
        self._animation.setStartValue(rect)
        self._animation.setEndValue(new_rect)
        self._animation.start()


class SmartProgressBar(QProgressBar):
    """Умная полоса прогресса с предсказанием времени"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._start_time = 0
        self._last_update_time = 0
        self._last_value = 0
        self._speed_samples: List[float] = []
        self._max_samples = 10
        
        # Стиль
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
    def setValue(self, value: int):
        """Устанавливает значение с расчетом времени"""
        super().setValue(value)
        
        current_time = time.time()
        
        if self._start_time == 0:
            self._start_time = current_time
            self._last_update_time = current_time
            self._last_value = value
            return
            
        # Вычисляем скорость
        time_delta = current_time - self._last_update_time
        value_delta = value - self._last_value
        
        if time_delta > 0 and value_delta > 0:
            speed = value_delta / time_delta
            self._speed_samples.append(speed)
            
            # Ограничиваем количество образцов
            if len(self._speed_samples) > self._max_samples:
                self._speed_samples.pop(0)
            
            # Обновляем текст с ETA
            self._update_eta_text(value)
            
        self._last_update_time = current_time
        self._last_value = value
        
    def _update_eta_text(self, current_value: int):
        """Обновляет текст с предсказанием времени"""
        if not self._speed_samples:
            return
            
        # Средняя скорость
        avg_speed = sum(self._speed_samples) / len(self._speed_samples)
        
        # Оставшееся время
        remaining_work = self.maximum() - current_value
        if avg_speed > 0:
            eta_seconds = remaining_work / avg_speed
            eta_text = self._format_time(eta_seconds)
            
            progress_text = f"{current_value}/{self.maximum()} ({eta_text})"
            self.setFormat(progress_text)
            
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в читаемый вид"""
        if seconds < 60:
            return f"{seconds:.0f}с"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}м {seconds}с"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}ч {minutes}м"
            
    def reset(self):
        """Сбрасывает состояние"""
        super().reset()
        self._start_time = 0
        self._last_update_time = 0
        self._last_value = 0
        self._speed_samples.clear()


class OptimizedFileListWidget(QListWidget, LazyLoadingMixin):
    """Оптимизированный список файлов с ленивой загрузкой"""
    
    def __init__(self, parent=None):
        QListWidget.__init__(self, parent)
        LazyLoadingMixin.__init__(self)
        
        # Оптимизации
        self.setUniformItemSizes(True)
        self.setAlternatingRowColors(True)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        
        # Кэширование иконок
        self._icon_cache: Dict[str, QIcon] = {}
        self._icon_loader = QTimer()
        self._icon_loader.timeout.connect(self._load_icons_batch)
        self._pending_icons: List[str] = []
        
    def addFileItem(self, file_path: str, size: int = 0, modified: str = ""):
        """Добавляет файл в список"""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        
        # Ленивая загрузка иконки
        self._queue_icon_loading(file_path, item)
        
        # Устанавливаем текст
        file_name = file_path.split("/")[-1]
        if size > 0:
            size_text = self._format_size(size)
            item.setText(f"{file_name} ({size_text})")
        else:
            item.setText(file_name)
            
        self.addItem(item)
        
    def _queue_icon_loading(self, file_path: str, item: QListWidgetItem):
        """Ставит иконку в очередь на загрузку"""
        if file_path not in self._pending_icons:
            self._pending_icons.append(file_path)
            
        # Запускаем загрузку через 100ms
        if not self._icon_loader.isActive():
            self._icon_loader.start(100)
            
    def _load_icons_batch(self):
        """Загружает иконки батчами"""
        batch_size = 5
        loaded_count = 0
        
        for file_path in self._pending_icons[:batch_size]:
            if self._load_file_icon(file_path):
                loaded_count += 1
                
        # Удаляем обработанные
        self._pending_icons = self._pending_icons[batch_size:]
        
        # Продолжаем если есть еще иконки
        if self._pending_icons:
            self._icon_loader.start(50)
        else:
            self._icon_loader.stop()
            
    def _load_file_icon(self, file_path: str) -> bool:
        """Загружает иконку для файла"""
        if file_path in self._icon_cache:
            return True
            
        try:
            # Определяем тип файла
            ext = file_path.split(".")[-1].lower()
            
            # Загружаем иконку (можно заменить на QFileIconProvider)
            icon = self._get_icon_for_extension(ext)
            self._icon_cache[file_path] = icon
            
            # Находим соответствующий элемент и устанавливаем иконку
            for i in range(self.count()):
                item = self.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == file_path:
                    item.setIcon(icon)
                    break
                    
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки иконки для {file_path}: {e}")
            return False
            
    def _get_icon_for_extension(self, ext: str) -> QIcon:
        """Возвращает иконку для расширения файла"""
        # Здесь можно использовать QFileIconProvider или кастомные иконки
        icon_map = {
            'pdf': '📄',
            'jpg': '🖼️',
            'jpeg': '🖼️',
            'png': '🖼️',
            'txt': '📝',
            'doc': '📝',
            'docx': '📝',
        }
        
        emoji = icon_map.get(ext, '📄')
        
        # Создаем иконку из эмодзи (упрощенный вариант)
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.drawText(QRect(0, 0, 16, 16), Qt.AlignmentFlag.AlignCenter, emoji)
        painter.end()
        
        return QIcon(pixmap)
        
    def _format_size(self, size: int) -> str:
        """Форматирует размер файла"""
        if size < 1024:
            return f"{size}B"
        elif size < 1024**2:
            return f"{size/1024:.1f}KB"
        elif size < 1024**3:
            return f"{size/1024**2:.1f}MB"
        else:
            return f"{size/1024**3:.1f}GB"


class ResponsiveLayout(QVBoxLayout):
    """Адаптивный макет, который подстраивается под размер окна"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._breakpoints = {
            'small': 600,
            'medium': 1024,
            'large': 1440
        }
        
        self._current_breakpoint = 'large'
        self._layout_configs = {}
        
    def addBreakpointConfig(self, breakpoint: str, config: Dict):
        """Добавляет конфигурацию для точки останова"""
        self._layout_configs[breakpoint] = config
        
    def resizeEvent(self, event):
        """Обработка изменения размера"""
        width = event.size().width()
        new_breakpoint = self._get_breakpoint(width)
        
        if new_breakpoint != self._current_breakpoint:
            self._current_breakpoint = new_breakpoint
            self._apply_breakpoint_config()
            
    def _get_breakpoint(self, width: int) -> str:
        """Определяет текущую точку останова"""
        if width <= self._breakpoints['small']:
            return 'small'
        elif width <= self._breakpoints['medium']:
            return 'medium'
        else:
            return 'large'
            
    def _apply_breakpoint_config(self):
        """Применяет конфигурацию для текущей точки останова"""
        if self._current_breakpoint in self._layout_configs:
            config = self._layout_configs[self._current_breakpoint]
            
            # Применяем spacing
            if 'spacing' in config:
                self.setSpacing(config['spacing'])
                
            # Применяем margins
            if 'margins' in config:
                margins = config['margins']
                self.setContentsMargins(
                    margins.get('left', 0),
                    margins.get('top', 0),
                    margins.get('right', 0),
                    margins.get('bottom', 0)
                )


# Глобальная конфигурация рендеринга
rendering_config = RenderingConfig()


def set_rendering_config(config: RenderingConfig):
    """Устанавливает глобальную конфигурацию рендеринга"""
    global rendering_config
    rendering_config = config
    
    # Применяем настройки к QApplication
    app = QApplication.instance()
    if app:
        if not config.enable_animations:
            app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
            app.setEffectEnabled(Qt.UIEffect.UI_AnimateTooltip, False)
            app.setEffectEnabled(Qt.UIEffect.UI_AnimateMenu, False)


def optimize_widget_performance(widget: QWidget):
    """Применяет оптимизации производительности к виджету"""
    # Включаем аппаратное ускорение если доступно
    widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    widget.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
    
    # Оптимизируем обновления
    widget.setUpdatesEnabled(True)
    widget.setAttribute(Qt.WidgetAttribute.WA_UpdatesDisabled, False)
    
    # Оптимизируем размер
    widget.setMinimumSize(QSize(0, 0))
    widget.setSizePolicy(
        widget.sizePolicy().horizontalPolicy(),
        widget.sizePolicy().verticalPolicy()
    ) 
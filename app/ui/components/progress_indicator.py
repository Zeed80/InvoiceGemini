"""
Улучшенные компоненты для отображения прогресса операций.
"""

import logging
from typing import Optional, Callable
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QFrame, QGraphicsOpacityEffect
)
from PyQt6.QtGui import QFont, QColor, QPalette

logger = logging.getLogger(__name__)


class EnhancedProgressBar(QProgressBar):
    """Улучшенный прогресс-бар с анимацией и дополнительными возможностями"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Настройка стиля
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Анимация для плавного изменения значения
        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(300)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Настройка внешнего вида
        self._setup_style()
        
    def _setup_style(self):
        """Настройка стиля прогресс-бара"""
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #f5f5f5;
                min-height: 25px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 3px;
            }
        """)
        
    def set_value_animated(self, value: int):
        """Установка значения с анимацией"""
        self._animation.setStartValue(self.value())
        self._animation.setEndValue(value)
        self._animation.start()
        
    def set_indeterminate(self, indeterminate: bool = True):
        """Переключение в режим неопределенного прогресса"""
        if indeterminate:
            self.setRange(0, 0)
        else:
            self.setRange(0, 100)
            
    def set_color(self, color: str):
        """Изменение цвета прогресс-бара"""
        self.setStyleSheet(self.styleSheet().replace("#4CAF50", color).replace("#45a049", color))


class ProcessingIndicator(QWidget):
    """Комплексный индикатор процесса обработки"""
    
    cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Таймер для обновления времени
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._start_time = None
        self._elapsed_seconds = 0
        
    def _setup_ui(self):
        """Настройка интерфейса"""
        layout = QVBoxLayout(self)
        
        # Заголовок операции
        self.title_label = QLabel("Обработка...")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        
        # Описание текущего этапа
        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)
        
        # Прогресс-бар
        self.progress_bar = EnhancedProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Панель с дополнительной информацией
        info_layout = QHBoxLayout()
        
        # Прошедшее время
        self.elapsed_label = QLabel("Время: 0:00")
        info_layout.addWidget(self.elapsed_label)
        
        info_layout.addStretch()
        
        # Оставшееся время (оценка)
        self.eta_label = QLabel("")
        info_layout.addWidget(self.eta_label)
        
        layout.addLayout(info_layout)
        
        # Кнопка отмены
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.clicked.connect(self.cancelled.emit)
        layout.addWidget(self.cancel_button)
        
        # Настройка отступов
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
    def set_title(self, title: str):
        """Установка заголовка операции"""
        self.title_label.setText(title)
        
    def set_stage(self, stage: str):
        """Установка описания текущего этапа"""
        self.stage_label.setText(stage)
        
    def set_progress(self, value: int, maximum: int = 100):
        """Установка прогресса"""
        if maximum != self.progress_bar.maximum():
            self.progress_bar.setMaximum(maximum)
            
        self.progress_bar.set_value_animated(value)
        
        # Обновляем оценку оставшегося времени
        self._update_eta(value, maximum)
        
    def set_indeterminate(self, indeterminate: bool = True):
        """Переключение в режим неопределенного прогресса"""
        self.progress_bar.set_indeterminate(indeterminate)
        self.eta_label.setText("")
        
    def start(self):
        """Начало отсчета времени"""
        self._start_time = QTimer.currentTime()
        self._elapsed_seconds = 0
        self._elapsed_timer.start(1000)  # Обновление каждую секунду
        
    def stop(self):
        """Остановка отсчета времени"""
        self._elapsed_timer.stop()
        
    def _update_elapsed_time(self):
        """Обновление прошедшего времени"""
        self._elapsed_seconds += 1
        minutes = self._elapsed_seconds // 60
        seconds = self._elapsed_seconds % 60
        self.elapsed_label.setText(f"Время: {minutes}:{seconds:02d}")
        
    def _update_eta(self, current: int, total: int):
        """Обновление оценки оставшегося времени"""
        if current > 0 and total > 0 and self._elapsed_seconds > 0:
            # Простая линейная оценка
            progress_ratio = current / total
            if progress_ratio > 0:
                total_estimated = self._elapsed_seconds / progress_ratio
                remaining = int(total_estimated - self._elapsed_seconds)
                
                if remaining > 0:
                    minutes = remaining // 60
                    seconds = remaining % 60
                    self.eta_label.setText(f"Осталось: ~{minutes}:{seconds:02d}")
                else:
                    self.eta_label.setText("Почти готово...")
            else:
                self.eta_label.setText("")
                
    def set_cancel_enabled(self, enabled: bool):
        """Включение/отключение кнопки отмены"""
        self.cancel_button.setEnabled(enabled)


class MultiStageProgressIndicator(QWidget):
    """Индикатор прогресса для многоэтапных операций"""
    
    def __init__(self, stages: list[str], parent=None):
        super().__init__(parent)
        self.stages = stages
        self.current_stage_index = -1
        self._setup_ui()
        
    def _setup_ui(self):
        """Настройка интерфейса"""
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Прогресс выполнения")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Создаем индикаторы для каждого этапа
        self.stage_widgets = []
        for i, stage_name in enumerate(self.stages):
            stage_widget = self._create_stage_widget(stage_name)
            self.stage_widgets.append(stage_widget)
            layout.addWidget(stage_widget)
            
        layout.addStretch()
        
    def _create_stage_widget(self, stage_name: str) -> QFrame:
        """Создание виджета для отдельного этапа"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Индикатор состояния (круг)
        status_indicator = QLabel("○")
        status_indicator.setFixedSize(20, 20)
        status_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_indicator)
        
        # Название этапа
        name_label = QLabel(stage_name)
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        # Статус выполнения
        status_label = QLabel("Ожидание")
        status_label.setStyleSheet("color: #666;")
        layout.addWidget(status_label)
        
        # Сохраняем ссылки на элементы
        frame.status_indicator = status_indicator
        frame.name_label = name_label
        frame.status_label = status_label
        
        return frame
        
    def start_stage(self, index: int):
        """Начало выполнения этапа"""
        if 0 <= index < len(self.stage_widgets):
            self.current_stage_index = index
            widget = self.stage_widgets[index]
            
            # Обновляем визуальное состояние
            widget.status_indicator.setText("◉")
            widget.status_indicator.setStyleSheet("color: #2196F3;")  # Синий
            widget.status_label.setText("Выполняется...")
            widget.status_label.setStyleSheet("color: #2196F3;")
            
            # Анимация появления
            self._animate_widget(widget)
            
    def complete_stage(self, index: int, success: bool = True):
        """Завершение этапа"""
        if 0 <= index < len(self.stage_widgets):
            widget = self.stage_widgets[index]
            
            if success:
                widget.status_indicator.setText("✓")
                widget.status_indicator.setStyleSheet("color: #4CAF50;")  # Зеленый
                widget.status_label.setText("Завершено")
                widget.status_label.setStyleSheet("color: #4CAF50;")
            else:
                widget.status_indicator.setText("✗")
                widget.status_indicator.setStyleSheet("color: #f44336;")  # Красный
                widget.status_label.setText("Ошибка")
                widget.status_label.setStyleSheet("color: #f44336;")
                
    def _animate_widget(self, widget: QFrame):
        """Анимация виджета"""
        # Создаем эффект прозрачности
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        # Анимация появления
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(300)
        animation.setStartValue(0.7)
        animation.setEndValue(1.0)
        animation.start()
        
    def reset(self):
        """Сброс всех этапов"""
        self.current_stage_index = -1
        for widget in self.stage_widgets:
            widget.status_indicator.setText("○")
            widget.status_indicator.setStyleSheet("")
            widget.status_label.setText("Ожидание")
            widget.status_label.setStyleSheet("color: #666;")


class CircularProgressIndicator(QWidget):
    """Круговой индикатор прогресса"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.maximum = 100
        self.setFixedSize(100, 100)
        
    def setValue(self, value: int):
        """Установка значения прогресса"""
        self.value = min(value, self.maximum)
        self.update()
        
    def setMaximum(self, maximum: int):
        """Установка максимального значения"""
        self.maximum = maximum
        self.update()
        
    def paintEvent(self, event):
        """Отрисовка индикатора"""
        from PyQt6.QtGui import QPainter, QPen, QBrush
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Параметры
        rect = self.rect().adjusted(10, 10, -10, -10)
        start_angle = 90 * 16  # Начинаем сверху
        
        # Фоновый круг
        painter.setPen(QPen(QColor("#e0e0e0"), 8, Qt.PenStyle.SolidLine))
        painter.drawArc(rect, 0, 360 * 16)
        
        # Прогресс
        if self.maximum > 0:
            progress_angle = int(-(self.value / self.maximum) * 360 * 16)
            painter.setPen(QPen(QColor("#4CAF50"), 8, Qt.PenStyle.SolidLine))
            painter.drawArc(rect, start_angle, progress_angle)
            
        # Текст в центре
        painter.setPen(QPen(QColor("#333"), 1))
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        
        percentage = int((self.value / self.maximum * 100) if self.maximum > 0 else 0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{percentage}%") 
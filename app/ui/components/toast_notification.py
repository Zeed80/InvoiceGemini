"""
Система неинвазивных toast уведомлений
Заменяет модальные QMessageBox на удобные всплывающие сообщения
"""
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QPushButton, 
    QVBoxLayout, QGraphicsOpacityEffect, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QCursor


class ToastNotification(QWidget):
    """Неинвазивное всплывающее уведомление"""
    
    clicked = pyqtSignal()  # Сигнал при клике
    closed = pyqtSignal()   # Сигнал при закрытии
    
    def __init__(
        self, 
        message: str, 
        level: str = "info",
        duration: int = 3000,
        closable: bool = True,
        action_text: str = None,
        parent: QWidget = None
    ):
        """
        Args:
            message: Текст уведомления
            level: Уровень - info, success, warning, error
            duration: Длительность показа в мс (0 = бесконечно)
            closable: Можно ли закрыть крестиком
            action_text: Текст кнопки действия
            parent: Родительский виджет
        """
        super().__init__(parent)
        
        self.duration = duration
        self.level = level
        
        # Настройка окна
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # Создаем UI
        self._setup_ui(message, level, closable, action_text)
        
        # Эффект прозрачности для анимации
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)
        
        # Таймер автозакрытия
        if duration > 0:
            QTimer.singleShot(duration, self.hide_animated)
    
    def _setup_ui(self, message, level, closable, action_text):
        """Создание интерфейса"""
        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Контейнер с фоном
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(16, 12, 16, 12)
        container_layout.setSpacing(12)
        
        # Стиль по уровню
        style_config = self._get_style_config(level)
        
        container.setStyleSheet(f"""
            QWidget {{
                background-color: {style_config['bg_color']};
                border-left: 4px solid {style_config['accent_color']};
                border-radius: 8px;
            }}
            QWidget:hover {{
                background-color: {style_config['hover_color']};
            }}
        """)
        
        # Иконка
        icon_label = QLabel(style_config['icon'])
        icon_label.setStyleSheet(f"""
            font-size: 20px;
            color: {style_config['accent_color']};
        """)
        container_layout.addWidget(icon_label)
        
        # Текст сообщения
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"""
            color: {style_config['text_color']};
            font-size: 12px;
            font-weight: 500;
        """)
        container_layout.addWidget(message_label, 1)
        
        # Кнопка действия (опционально)
        if action_text:
            action_btn = QPushButton(action_text)
            action_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {style_config['accent_color']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {style_config['accent_dark']};
                }}
            """)
            action_btn.clicked.connect(self._on_action_clicked)
            action_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            container_layout.addWidget(action_btn)
        
        # Кнопка закрытия
        if closable:
            close_btn = QPushButton("✕")
            close_btn.setFixedSize(20, 20)
            close_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    color: {style_config['text_color']};
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    color: {style_config['accent_color']};
                }}
            """)
            close_btn.clicked.connect(self.hide_animated)
            close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            container_layout.addWidget(close_btn)
        
        main_layout.addWidget(container)
        
        # Добавляем тень
        self.setStyleSheet("""
            ToastNotification {
                background: transparent;
            }
        """)
        
        # Фиксированная ширина
        self.setFixedWidth(400)
        self.adjustSize()
    
    def _get_style_config(self, level):
        """Получить конфигурацию стиля по уровню"""
        styles = {
            'info': {
                'bg_color': '#ebf5fb',
                'hover_color': '#d6eaf8',
                'accent_color': '#3498db',
                'accent_dark': '#2980b9',
                'text_color': '#21618c',
                'icon': 'ℹ️'
            },
            'success': {
                'bg_color': '#eafaf1',
                'hover_color': '#d5f4e6',
                'accent_color': '#27ae60',
                'accent_dark': '#1e8449',
                'text_color': '#186a3b',
                'icon': '✅'
            },
            'warning': {
                'bg_color': '#fef5e7',
                'hover_color': '#fdebd0',
                'accent_color': '#f39c12',
                'accent_dark': '#d68910',
                'text_color': '#9c640c',
                'icon': '⚠️'
            },
            'error': {
                'bg_color': '#fadbd8',
                'hover_color': '#f5b7b1',
                'accent_color': '#e74c3c',
                'accent_dark': '#c0392b',
                'text_color': '#943126',
                'icon': '❌'
            }
        }
        return styles.get(level, styles['info'])
    
    def show_animated(self):
        """Показать с анимацией"""
        # Позиционируем в правом нижнем углу
        self._position_toast()
        
        # Показываем
        self.show()
        
        # Анимация появления
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_animation.start()
        
        # Анимация слайда снизу
        start_pos = self.pos()
        self.move(start_pos.x(), start_pos.y() + 50)
        
        self.slide_animation = QPropertyAnimation(self, b"pos")
        self.slide_animation.setDuration(300)
        self.slide_animation.setStartValue(self.pos())
        self.slide_animation.setEndValue(start_pos)
        self.slide_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.slide_animation.start()
    
    def hide_animated(self):
        """Скрыть с анимацией"""
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(200)
        self.fade_animation.setStartValue(1)
        self.fade_animation.setEndValue(0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_animation.finished.connect(self._on_hide_finished)
        self.fade_animation.start()
    
    def _on_hide_finished(self):
        """Обработка завершения скрытия"""
        self.close()
        self.closed.emit()
    
    def _on_action_clicked(self):
        """Обработка клика по кнопке действия"""
        self.clicked.emit()
        self.hide_animated()
    
    def _position_toast(self):
        """Позиционировать toast в правом нижнем углу"""
        screen = QApplication.primaryScreen().geometry()
        
        # Получаем менеджер для учета других toast'ов
        manager = ToastManager.instance()
        offset = manager.get_next_position_offset()
        
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 60 - offset
        
        self.move(x, y)


class ToastManager:
    """Менеджер toast уведомлений"""
    
    _instance = None
    
    @classmethod
    def instance(cls):
        """Получить singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.active_toasts: List[ToastNotification] = []
        self.toast_spacing = 10
    
    def show_toast(
        self,
        message: str,
        level: str = "info",
        duration: int = 3000,
        closable: bool = True,
        action_text: str = None,
        parent: QWidget = None
    ) -> ToastNotification:
        """
        Показать toast уведомление
        
        Args:
            message: Текст сообщения
            level: info, success, warning, error
            duration: Длительность в мс
            closable: Можно ли закрыть
            action_text: Текст кнопки действия
            parent: Родительский виджет
        
        Returns:
            ToastNotification объект
        """
        toast = ToastNotification(
            message=message,
            level=level,
            duration=duration,
            closable=closable,
            action_text=action_text,
            parent=parent
        )
        
        # Подключаем обработку закрытия
        toast.closed.connect(lambda: self._on_toast_closed(toast))
        
        # Добавляем в список активных
        self.active_toasts.append(toast)
        
        # Показываем
        toast.show_animated()
        
        return toast
    
    def get_next_position_offset(self) -> int:
        """Получить смещение для следующего toast'а"""
        total_offset = 0
        for toast in self.active_toasts:
            if toast.isVisible():
                total_offset += toast.height() + self.toast_spacing
        return total_offset
    
    def _on_toast_closed(self, toast: ToastNotification):
        """Обработка закрытия toast'а"""
        if toast in self.active_toasts:
            self.active_toasts.remove(toast)
            toast.deleteLater()
        
        # Переместить оставшиеся toast'ы
        self._reposition_toasts()
    
    def _reposition_toasts(self):
        """Переместить все активные toast'ы"""
        offset = 0
        screen = QApplication.primaryScreen().geometry()
        
        for toast in self.active_toasts:
            if toast.isVisible():
                x = screen.width() - toast.width() - 20
                y = screen.height() - toast.height() - 60 - offset
                
                # Анимация перемещения
                animation = QPropertyAnimation(toast, b"pos")
                animation.setDuration(200)
                animation.setStartValue(toast.pos())
                animation.setEndValue(QPoint(x, y))
                animation.setEasingCurve(QEasingCurve.Type.OutCubic)
                animation.start()
                
                offset += toast.height() + self.toast_spacing
    
    def clear_all(self):
        """Закрыть все toast'ы"""
        for toast in self.active_toasts[:]:  # Копия списка
            toast.hide_animated()


# Глобальные функции для удобства
def show_toast(
    message: str,
    level: str = "info",
    duration: int = 3000,
    closable: bool = True,
    action_text: str = None,
    parent: QWidget = None
) -> ToastNotification:
    """
    Показать toast уведомление (удобная функция)
    
    Пример:
        show_toast("Файл сохранен", "success")
        show_toast("Произошла ошибка", "error", duration=5000)
        show_toast("Обновление доступно", "info", action_text="Скачать")
    """
    return ToastManager.instance().show_toast(
        message, level, duration, closable, action_text, parent
    )


def show_info(message: str, duration: int = 3000):
    """Показать информационное уведомление"""
    return show_toast(message, "info", duration)


def show_success(message: str, duration: int = 3000):
    """Показать уведомление об успехе"""
    return show_toast(message, "success", duration)


def show_warning(message: str, duration: int = 4000):
    """Показать предупреждение"""
    return show_toast(message, "warning", duration)


def show_error(message: str, duration: int = 5000):
    """Показать ошибку"""
    return show_toast(message, "error", duration)


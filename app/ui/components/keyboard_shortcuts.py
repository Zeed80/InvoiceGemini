"""
Keyboard Shortcuts System - централизованная система горячих клавиш
Обеспечивает единообразие shortcuts во всем приложении
"""
from typing import Dict, Callable
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt

import logging
logger = logging.getLogger(__name__)


class ShortcutManager:
    """
    Менеджер горячих клавиш приложения
    
    Использование:
        manager = ShortcutManager(main_window)
        manager.register_shortcut('Ctrl+O', main_window.open_file, "Открыть файл")
        manager.setup_all_shortcuts()
    """
    
    # Стандартные shortcuts приложения
    STANDARD_SHORTCUTS = {
        # Файлы
        'Ctrl+O': {
            'action': 'open_file',
            'description': 'Открыть файл'
        },
        'Ctrl+Shift+O': {
            'action': 'open_folder',
            'description': 'Открыть папку (batch)'
        },
        'Ctrl+S': {
            'action': 'save_results',
            'description': 'Сохранить результаты'
        },
        'Ctrl+E': {
            'action': 'export_results',
            'description': 'Экспорт результатов'
        },
        
        # Обработка
        'Ctrl+P': {
            'action': 'process_current',
            'description': 'Обработать текущий файл',
            'alt': 'F5'
        },
        'F5': {
            'action': 'process_current',
            'description': 'Обработать текущий файл'
        },
        'Ctrl+B': {
            'action': 'batch_process',
            'description': 'Пакетная обработка'
        },
        'Escape': {
            'action': 'cancel_processing',
            'description': 'Отменить обработку'
        },
        
        # Модели
        'Ctrl+1': {
            'action': 'select_auto',
            'description': 'Выбрать Авто'
        },
        'Ctrl+2': {
            'action': 'select_gemini',
            'description': 'Выбрать Gemini'
        },
        'Ctrl+3': {
            'action': 'select_layoutlm',
            'description': 'Выбрать LayoutLM'
        },
        'Ctrl+4': {
            'action': 'select_donut',
            'description': 'Выбрать Donut'
        },
        'Ctrl+5': {
            'action': 'select_trocr',
            'description': 'Выбрать TrOCR'
        },
        
        # Просмотр
        'Ctrl+R': {
            'action': 'refresh_view',
            'description': 'Обновить просмотр'
        },
        'Ctrl+[': {
            'action': 'previous_file',
            'description': 'Предыдущий файл'
        },
        'Ctrl+]': {
            'action': 'next_file',
            'description': 'Следующий файл'
        },
        
        # Интерфейс
        'Ctrl+,': {
            'action': 'open_settings',
            'description': 'Открыть настройки'
        },
        'F1': {
            'action': 'show_help',
            'description': 'Показать справку'
        },
        'Ctrl+T': {
            'action': 'open_training',
            'description': 'Открыть обучение моделей'
        },
        'Ctrl+L': {
            'action': 'view_logs',
            'description': 'Просмотр логов'
        },
        
        # Приложение
        'Ctrl+Q': {
            'action': 'quit_app',
            'description': 'Выйти из приложения'
        },
        'F11': {
            'action': 'toggle_fullscreen',
            'description': 'Полноэкранный режим'
        },
    }
    
    def __init__(self, parent: QWidget):
        """
        Args:
            parent: Родительский виджет (обычно MainWindow)
        """
        self.parent = parent
        self.shortcuts = {}  # key: (shortcut_obj, handler, description)
        self.custom_shortcuts = {}  # Пользовательские shortcuts
        
    def register_shortcut(
        self, 
        key_sequence: str, 
        handler: Callable, 
        description: str = "",
        context: Qt.ShortcutContext = Qt.ShortcutContext.WindowShortcut
    ):
        """
        Зарегистрировать shortcut
        
        Args:
            key_sequence: Комбинация клавиш (например, 'Ctrl+O')
            handler: Функция-обработчик
            description: Описание действия
            context: Контекст срабатывания
        """
        try:
            shortcut = QShortcut(QKeySequence(key_sequence), self.parent)
            shortcut.setContext(context)
            shortcut.activated.connect(handler)
            
            self.shortcuts[key_sequence] = (shortcut, handler, description)
            logger.debug(f"Registered shortcut: {key_sequence} -> {description}")
            
        except Exception as e:
            logger.error(f"Failed to register shortcut {key_sequence}: {e}")
    
    def setup_all_shortcuts(self):
        """Настроить все стандартные shortcuts"""
        for key_seq, config in self.STANDARD_SHORTCUTS.items():
            action_name = config['action']
            description = config['description']
            
            # Проверяем наличие метода в parent
            if hasattr(self.parent, action_name):
                handler = getattr(self.parent, action_name)
                self.register_shortcut(key_seq, handler, description)
            else:
                logger.warning(
                    f"Action '{action_name}' not found in parent widget, "
                    f"skipping shortcut '{key_seq}'"
                )
    
    def unregister_shortcut(self, key_sequence: str):
        """
        Удалить shortcut
        
        Args:
            key_sequence: Комбинация клавиш
        """
        if key_sequence in self.shortcuts:
            shortcut, _, _ = self.shortcuts[key_sequence]
            shortcut.setEnabled(False)
            shortcut.deleteLater()
            del self.shortcuts[key_sequence]
            logger.debug(f"Unregistered shortcut: {key_sequence}")
    
    def disable_shortcut(self, key_sequence: str):
        """Временно отключить shortcut"""
        if key_sequence in self.shortcuts:
            shortcut, _, _ = self.shortcuts[key_sequence]
            shortcut.setEnabled(False)
    
    def enable_shortcut(self, key_sequence: str):
        """Включить shortcut"""
        if key_sequence in self.shortcuts:
            shortcut, _, _ = self.shortcuts[key_sequence]
            shortcut.setEnabled(True)
    
    def get_all_shortcuts(self) -> Dict[str, str]:
        """
        Получить все зарегистрированные shortcuts
        
        Returns:
            Словарь {key_sequence: description}
        """
        return {key: desc for key, (_, _, desc) in self.shortcuts.items()}
    
    def show_shortcuts_dialog(self):
        """Показать диалог со всеми shortcuts"""
        dialog = ShortcutsDialog(self.parent, self.get_all_shortcuts())
        dialog.exec()


class ShortcutsDialog(QDialog):
    """Диалог с отображением всех горячих клавиш"""
    
    def __init__(self, parent=None, shortcuts: Dict[str, str] = None):
        super().__init__(parent)
        self.shortcuts = shortcuts or {}
        
        self.setWindowTitle("⌨️ Горячие клавиши")
        self.setMinimumSize(600, 500)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Создание интерфейса"""
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Доступные горячие клавиши")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
        """)
        layout.addWidget(title)
        
        # Таблица shortcuts
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Комбинация", "Действие"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Стилизация таблицы
        table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #d5dbdb;
                border-radius: 6px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 10px;
                border: none;
                font-weight: 600;
            }
        """)
        
        # Заполняем таблицу
        # Группируем shortcuts по категориям
        categories = {
            'Файлы': ['Ctrl+O', 'Ctrl+Shift+O', 'Ctrl+S', 'Ctrl+E'],
            'Обработка': ['Ctrl+P', 'F5', 'Ctrl+B', 'Escape'],
            'Модели': ['Ctrl+1', 'Ctrl+2', 'Ctrl+3', 'Ctrl+4', 'Ctrl+5'],
            'Просмотр': ['Ctrl+R', 'Ctrl+[', 'Ctrl+]'],
            'Интерфейс': ['Ctrl+,', 'F1', 'Ctrl+T', 'Ctrl+L'],
            'Приложение': ['Ctrl+Q', 'F11']
        }
        
        row = 0
        for category, keys in categories.items():
            # Добавляем заголовок категории
            table.insertRow(row)
            
            category_item = QTableWidgetItem(f"📁 {category}")
            category_item.setFont(self._get_bold_font())
            category_item.setBackground(Qt.GlobalColor.lightGray)
            table.setItem(row, 0, category_item)
            
            spacer_item = QTableWidgetItem("")
            spacer_item.setBackground(Qt.GlobalColor.lightGray)
            table.setItem(row, 1, spacer_item)
            
            table.setSpan(row, 0, 1, 2)
            row += 1
            
            # Добавляем shortcuts категории
            for key in keys:
                if key in self.shortcuts:
                    table.insertRow(row)
                    
                    key_item = QTableWidgetItem(key)
                    key_item.setFont(self._get_mono_font())
                    table.setItem(row, 0, key_item)
                    
                    desc_item = QTableWidgetItem(self.shortcuts[key])
                    table.setItem(row, 1, desc_item)
                    
                    row += 1
        
        # Настраиваем ширину колонок
        table.setColumnWidth(0, 150)
        
        layout.addWidget(table)
        
        # Кнопки
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Закрыть")
        close_button.setMinimumWidth(100)
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def _get_bold_font(self):
        """Получить жирный шрифт"""
        font = self.font()
        font.setBold(True)
        return font
    
    def _get_mono_font(self):
        """Получить моноширинный шрифт"""
        from PyQt6.QtGui import QFont
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
        return font


# Удобные функции для быстрой настройки
def setup_standard_shortcuts(main_window):
    """
    Быстрая настройка стандартных shortcuts для главного окна
    
    Args:
        main_window: Экземпляр MainWindow
    
    Returns:
        ShortcutManager instance
    """
    manager = ShortcutManager(main_window)
    manager.setup_all_shortcuts()
    
    # Сохраняем ссылку в главном окне
    main_window.shortcut_manager = manager
    
    logger.info(f"Setup {len(manager.shortcuts)} keyboard shortcuts")
    
    return manager


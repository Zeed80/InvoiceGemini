"""
Диалог для редактирования настраиваемых полей таблицы результатов.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QLineEdit, QCheckBox,
    QMessageBox, QWidget, QAbstractItemView, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QAction

from . import config
from .settings_manager import settings_manager


class FieldItem(QWidget):
    """Виджет элемента списка для поля таблицы."""
    
    def __init__(self, field, parent=None):
        super().__init__(parent)
        self.field = field.copy()  # Создаем копию поля
        
        # Основная компоновка
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Чекбокс для видимости
        self.visible_check = QCheckBox()
        self.visible_check.setChecked(field.get("visible", True))
        self.visible_check.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.visible_check)
        
        # Лейбл с ID поля
        self.id_label = QLabel(field.get("id", ""))
        self.id_label.setStyleSheet("font-weight: bold;")
        self.id_label.setFixedWidth(120)
        layout.addWidget(self.id_label)
        
        # Поле для имени поля
        self.name_edit = QLineEdit(field.get("name", ""))
        self.name_edit.textChanged.connect(self._on_name_changed)
        layout.addWidget(self.name_edit)
        
        # Кнопка удаления
        self.delete_button = QPushButton("×")
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self.delete_button)
        
    def _on_visibility_changed(self, checked):
        """Обработчик изменения видимости поля."""
        self.field["visible"] = checked
        
    def _on_name_changed(self, text):
        """Обработчик изменения имени поля."""
        self.field["name"] = text
        
    def _on_delete_clicked(self):
        """Обработчик нажатия кнопки удаления."""
        # Находим родительский диалог
        dialog = self.window()
        if dialog and isinstance(dialog, TableFieldsDialog):
            # Находим индекс этого виджета в списке
            list_widget = dialog.fields_list
            for i in range(list_widget.count()):
                if list_widget.itemWidget(list_widget.item(i)) == self:
                    dialog._delete_field(i)
                    break
                    
    def get_field(self):
        """Возвращает текущее состояние поля."""
        return self.field.copy()  # Возвращаем копию, чтобы избежать случайных изменений


class TableFieldsDialog(QDialog):
    """Диалог для редактирования полей таблицы результатов."""
    
    fieldsChanged = pyqtSignal(list)  # Сигнал об изменении полей
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка полей таблицы")
        self.setMinimumSize(500, 400)
        
        # Загружаем текущие поля из настроек
        self.fields = self._load_fields()
        
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        layout = QVBoxLayout(self)
        
        # Заголовок
        header_label = QLabel("Настройка отображаемых полей и их порядка в таблице результатов:")
        layout.addWidget(header_label)
        
        # Подсказка по перемещению
        hint_label = QLabel("Для изменения порядка полей используйте перетаскивание или кнопки ↑↓")
        hint_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(hint_label)
        
        # Список полей
        self.fields_list = QListWidget()
        self.fields_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.fields_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fields_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.fields_list.customContextMenuRequested.connect(self._show_context_menu)
        self.fields_list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.fields_list)
        
        # Кнопки перемещения
        move_buttons_layout = QHBoxLayout()
        
        self.move_up_button = QPushButton("↑")
        self.move_up_button.setFixedWidth(40)
        self.move_up_button.clicked.connect(self._move_field_up)
        move_buttons_layout.addWidget(self.move_up_button)
        
        self.move_down_button = QPushButton("↓")
        self.move_down_button.setFixedWidth(40)
        self.move_down_button.clicked.connect(self._move_field_down)
        move_buttons_layout.addWidget(self.move_down_button)
        
        move_buttons_layout.addStretch()
        layout.addLayout(move_buttons_layout)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Добавить")
        self.add_button.clicked.connect(self._add_field)
        button_layout.addWidget(self.add_button)
        
        self.reset_button = QPushButton("Сбросить")
        self.reset_button.clicked.connect(self._reset_fields)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Кнопки OK и Отмена
        buttons_layout = QHBoxLayout()
        
        buttons_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        
        layout.addLayout(buttons_layout)
        
        # Заполняем список полей
        self._populate_fields_list()
        
        # Подключаем обработчик выделения для управления кнопками перемещения
        self.fields_list.itemSelectionChanged.connect(self._update_move_buttons_state)
        self._update_move_buttons_state()
        
    def _load_fields(self):
        """Загружает поля из настроек."""
        return settings_manager.get_table_fields()
        
    def _populate_fields_list(self):
        """Заполняет список полей."""
        self.fields_list.clear()
        
        # Сортируем поля по порядку
        sorted_fields = sorted(self.fields, key=lambda f: f.get("order", 0))
        
        for field in sorted_fields:
            item = QListWidgetItem()
            field_widget = FieldItem(field)
            item.setSizeHint(field_widget.sizeHint())
            
            self.fields_list.addItem(item)
            self.fields_list.setItemWidget(item, field_widget)
            
    def _add_field(self):
        """Добавляет новое поле."""
        # Создаем диалог для ввода ID и имени поля
        dialog = QDialog(self)
        dialog.setWindowTitle("Добавление нового поля")
        dialog.setMinimumWidth(300)
        
        dialog_layout = QVBoxLayout(dialog)
        
        form_layout = QVBoxLayout()
        
        id_layout = QHBoxLayout()
        id_label = QLabel("ID поля:")
        id_edit = QLineEdit()
        id_layout.addWidget(id_label)
        id_layout.addWidget(id_edit)
        form_layout.addLayout(id_layout)
        
        name_layout = QHBoxLayout()
        name_label = QLabel("Название поля:")
        name_edit = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)
        form_layout.addLayout(name_layout)
        
        dialog_layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Отмена")
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        dialog_layout.addLayout(button_layout)
        
        # Подключаем обработчики
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Показываем диалог
        if dialog.exec() == QDialog.DialogCode.Accepted:
            field_id = id_edit.text().strip()
            field_name = name_edit.text().strip()
            
            if not field_id or not field_name:
                QMessageBox.warning(self, "Ошибка", "ID и название поля не могут быть пустыми")
                return
                
            # Проверяем уникальность ID
            if any(f["id"] == field_id for f in self.fields):
                QMessageBox.warning(self, "Ошибка", f"Поле с ID '{field_id}' уже существует")
                return
                
            # Добавляем новое поле
            field = {
                "id": field_id,
                "name": field_name,
                "visible": True,
                "order": len(self.fields)
            }
            
            self.fields.append(field)
            
            # Обновляем список
            item = QListWidgetItem()
            field_widget = FieldItem(field)
            item.setSizeHint(field_widget.sizeHint())
            
            self.fields_list.addItem(item)
            self.fields_list.setItemWidget(item, field_widget)
            
    def _move_field_up(self):
        """Перемещает выбранное поле вверх по списку."""
        current_row = self.fields_list.currentRow()
        if current_row > 0:
            # Сохраняем все поля в текущем состоянии
            fields_backup = []
            for i in range(self.fields_list.count()):
                item = self.fields_list.item(i)
                if item:
                    widget = self.fields_list.itemWidget(item)
                    if widget:
                        fields_backup.append(widget.get_field())

            # Меняем местами поля в списке
            fields_backup[current_row], fields_backup[current_row - 1] = \
                fields_backup[current_row - 1], fields_backup[current_row]

            # Пересоздаем список с новым порядком
            self.fields_list.clear()
            for field in fields_backup:
                item = QListWidgetItem()
                widget = FieldItem(field)
                item.setSizeHint(widget.sizeHint())
                self.fields_list.addItem(item)
                self.fields_list.setItemWidget(item, widget)

            # Выделяем перемещенный элемент
            self.fields_list.setCurrentRow(current_row - 1)

            # Обновляем порядок полей
            self._update_fields_order()

    def _move_field_down(self):
        """Перемещает выбранное поле вниз по списку."""
        current_row = self.fields_list.currentRow()
        if current_row < self.fields_list.count() - 1:
            # Сохраняем все поля в текущем состоянии
            fields_backup = []
            for i in range(self.fields_list.count()):
                item = self.fields_list.item(i)
                if item:
                    widget = self.fields_list.itemWidget(item)
                    if widget:
                        fields_backup.append(widget.get_field())

            # Меняем местами поля в списке
            fields_backup[current_row], fields_backup[current_row + 1] = \
                fields_backup[current_row + 1], fields_backup[current_row]

            # Пересоздаем список с новым порядком
            self.fields_list.clear()
            for field in fields_backup:
                item = QListWidgetItem()
                widget = FieldItem(field)
                item.setSizeHint(widget.sizeHint())
                self.fields_list.addItem(item)
                self.fields_list.setItemWidget(item, widget)

            # Выделяем перемещенный элемент
            self.fields_list.setCurrentRow(current_row + 1)

            # Обновляем порядок полей
            self._update_fields_order()

    def _update_move_buttons_state(self):
        """Обновляет состояние кнопок перемещения в зависимости от выделения."""
        current_row = self.fields_list.currentRow()
        self.move_up_button.setEnabled(current_row > 0)
        self.move_down_button.setEnabled(current_row >= 0 and current_row < self.fields_list.count() - 1)
        
    def _show_context_menu(self, position):
        """Показывает контекстное меню для поля."""
        menu = QMenu(self)
        
        move_up_action = QAction("Переместить вверх", self)
        move_up_action.triggered.connect(self._move_field_up)
        move_up_action.setEnabled(self.move_up_button.isEnabled())
        menu.addAction(move_up_action)
        
        move_down_action = QAction("Переместить вниз", self)
        move_down_action.triggered.connect(self._move_field_down)
        move_down_action.setEnabled(self.move_down_button.isEnabled())
        menu.addAction(move_down_action)
        
        menu.addSeparator()
        
        delete_action = QAction("Удалить", self)
        delete_action.triggered.connect(lambda: self._delete_field(self.fields_list.currentRow()))
        menu.addAction(delete_action)
        
        menu.exec(self.fields_list.viewport().mapToGlobal(position))
        
    def _delete_field(self, row):
        """Удаляет поле."""
        if row < 0 or row >= len(self.fields):
            return
            
        # Спрашиваем подтверждение
        result = QMessageBox.question(
            self, 
            "Подтверждение удаления", 
            "Вы действительно хотите удалить выбранное поле?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result != QMessageBox.StandardButton.Yes:
            return
            
        # Удаляем поле из списка полей
        self.fields.pop(row)
        # Удаляем элемент из QListWidget
        self.fields_list.takeItem(row)
        
        # Обновляем порядок полей
        self._update_fields_order()
        
    def _on_rows_moved(self, parent, start, end, destination, row):
        """Обработчик перемещения строк в списке."""
        # Сохраняем все поля в текущем состоянии
        fields_backup = []
        for i in range(self.fields_list.count()):
            item = self.fields_list.item(i)
            if item:
                widget = self.fields_list.itemWidget(item)
                if widget:
                    fields_backup.append(widget.get_field())

        # Определяем индексы для перемещения
        source_row = start
        dest_row = row if row <= start else row - 1

        # Перемещаем элемент в списке
        field = fields_backup.pop(source_row)
        fields_backup.insert(dest_row, field)

        # Пересоздаем список с новым порядком
        self.fields_list.clear()
        for field in fields_backup:
            item = QListWidgetItem()
            widget = FieldItem(field)
            item.setSizeHint(widget.sizeHint())
            self.fields_list.addItem(item)
            self.fields_list.setItemWidget(item, widget)

        # Выделяем перемещенный элемент
        self.fields_list.setCurrentRow(dest_row)

        # Обновляем порядок полей
        self._update_fields_order()
        
    def _update_fields_order(self):
        """Обновляет порядок полей и сохраняет изменения."""
        # Собираем поля из виджетов
        self.fields = []
        for i in range(self.fields_list.count()):
            item = self.fields_list.item(i)
            field_widget = self.fields_list.itemWidget(item)
            if field_widget:
                field = field_widget.get_field()  # Используем метод get_field()
                field["order"] = i  # Обновляем порядок
                self.fields.append(field)
            
        # Сохраняем изменения
        if settings_manager.save_table_fields(self.fields):
            # Отправляем сигнал об изменении полей
            self.fieldsChanged.emit(self.fields)
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить изменения порядка полей")
        
    def _reset_fields(self):
        """Сбрасывает поля к значениям по умолчанию."""
        # Спрашиваем подтверждение
        result = QMessageBox.question(
            self, 
            "Сброс полей", 
            "Вы действительно хотите сбросить все поля к значениям по умолчанию?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result != QMessageBox.StandardButton.Yes:
            return
            
        # Загружаем поля по умолчанию
        self.fields = config.DEFAULT_TABLE_FIELDS.copy()
        
        # Обновляем список полей
        self._populate_fields_list()
        
        # Сохраняем изменения
        self._update_fields_order()
        
    def accept(self):
        """Переопределение метода принятия диалога."""
        # Обновляем порядок полей перед сохранением
        self._update_fields_order()
        super().accept() 
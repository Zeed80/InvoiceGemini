#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диалог управления полями таблицы и синхронизации промптов моделей.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QPushButton, QGroupBox,
                             QLabel, QLineEdit, QTextEdit, QComboBox, QCheckBox,
                             QSpinBox, QSplitter, QTabWidget, QWidget, QMessageBox,
                             QFormLayout, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..field_manager import field_manager, TableField

class FieldManagerDialog(QDialog):
    """Диалог для управления полями таблицы и синхронизации промптов."""
    
    fields_updated = pyqtSignal()  # Сигнал при обновлении полей
    
    def __init__(self, parent=None):
        """Инициализация диалога."""
        super().__init__(parent)
        self.setWindowTitle("Управление полями таблицы")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # Создаем интерфейс
        self._setup_ui()
        self._load_fields()
        
        # Подключаем сигналы
        self._connect_signals()
    
    def _setup_ui(self):
        """Создает интерфейс диалога."""
        layout = QVBoxLayout(self)
        
        # Заголовок и описание
        header_label = QLabel("Управление полями таблицы результатов")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        description_label = QLabel(
            "Настройте поля таблицы результатов и синхронизируйте промпты всех моделей (Gemini, LayoutLM, LLM плагины)."
        )
        layout.addWidget(description_label)
        
        # Основной сплиттер
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Левая панель - список полей
        left_widget = self._create_fields_list()
        splitter.addWidget(left_widget)
        
        # Правая панель - редактор поля
        right_widget = self._create_field_editor()
        splitter.addWidget(right_widget)
        
        # Пропорции сплиттера
        splitter.setSizes([400, 600])
        
        # Нижняя панель - кнопки управления
        buttons_layout = self._create_bottom_buttons()
        layout.addLayout(buttons_layout)
        
        # Кнопки диалога
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_fields_list(self):
        """Создает список полей."""
        widget = QGroupBox("Поля таблицы")
        layout = QVBoxLayout(widget)
        
        # Таблица полей
        self.fields_table = QTableWidget()
        self.fields_table.setColumnCount(5)
        self.fields_table.setHorizontalHeaderLabels([
            "Включено", "Название", "Приоритет", "Тип", "Обязательное"
        ])
        
        # Настройка таблицы
        header = self.fields_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Включено
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)           # Название
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Приоритет
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Тип
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Обязательное
        
        layout.addWidget(self.fields_table)
        
        # Кнопки управления полями
        fields_buttons = QHBoxLayout()
        
        self.add_field_btn = QPushButton("➕ Добавить поле")
        self.remove_field_btn = QPushButton("➖ Удалить поле")
        self.move_up_btn = QPushButton("⬆️ Вверх")
        self.move_down_btn = QPushButton("⬇️ Вниз")
        
        fields_buttons.addWidget(self.add_field_btn)
        fields_buttons.addWidget(self.remove_field_btn)
        fields_buttons.addStretch()
        fields_buttons.addWidget(self.move_up_btn)
        fields_buttons.addWidget(self.move_down_btn)
        
        layout.addLayout(fields_buttons)
        
        return widget
    
    def _create_field_editor(self):
        """Создает редактор поля."""
        widget = QGroupBox("Редактор поля")
        layout = QVBoxLayout(widget)
        
        # Табы для разных настроек
        self.tabs = QTabWidget()
        
        # Основные настройки
        basic_tab = self._create_basic_tab()
        self.tabs.addTab(basic_tab, "Основные")
        
        # Настройки для моделей
        models_tab = self._create_models_tab()
        self.tabs.addTab(models_tab, "Модели")
        
        # Предварительный просмотр промптов
        preview_tab = self._create_preview_tab()
        self.tabs.addTab(preview_tab, "Просмотр промптов")
        
        layout.addWidget(self.tabs)
        
        return widget
    
    def _create_basic_tab(self):
        """Создает вкладку основных настроек."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Поля ввода
        self.field_id_edit = QLineEdit()
        self.field_id_edit.setPlaceholderText("уникальный_id_поля")
        layout.addRow("ID поля:", self.field_id_edit)
        
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setPlaceholderText("Отображаемое название")
        layout.addRow("Название в таблице:", self.display_name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Описание поля для генерации промптов")
        layout.addRow("Описание:", self.description_edit)
        
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["text", "number", "date", "currency"])
        layout.addRow("Тип данных:", self.data_type_combo)
        
        self.required_check = QCheckBox()
        layout.addRow("Обязательное поле:", self.required_check)
        
        self.priority_spin = QSpinBox()
        self.priority_spin.setMinimum(1)
        self.priority_spin.setMaximum(5)
        self.priority_spin.setValue(3)
        layout.addRow("Приоритет (1-высший):", self.priority_spin)
        
        self.position_spin = QSpinBox()
        self.position_spin.setMinimum(1)
        self.position_spin.setMaximum(100)
        self.position_spin.setValue(1)
        layout.addRow("Позиция в таблице:", self.position_spin)
        
        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(True)
        layout.addRow("Включено:", self.enabled_check)
        
        return widget
    
    def _create_models_tab(self):
        """Создает вкладку настроек для моделей."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Gemini ключевые слова
        gemini_group = QGroupBox("Ключевые слова для Gemini API")
        gemini_layout = QVBoxLayout(gemini_group)
        
        self.gemini_keywords_edit = QTextEdit()
        self.gemini_keywords_edit.setMaximumHeight(100)
        self.gemini_keywords_edit.setPlaceholderText("Введите ключевые слова через запятую")
        gemini_layout.addWidget(self.gemini_keywords_edit)
        
        layout.addWidget(gemini_group)
        
        # LayoutLM лейблы
        layoutlm_group = QGroupBox("Лейблы для LayoutLM")
        layoutlm_layout = QVBoxLayout(layoutlm_group)
        
        self.layoutlm_labels_edit = QTextEdit()
        self.layoutlm_labels_edit.setMaximumHeight(100)
        self.layoutlm_labels_edit.setPlaceholderText("Введите лейблы через запятую")
        layoutlm_layout.addWidget(self.layoutlm_labels_edit)
        
        layout.addWidget(layoutlm_group)
        
        # OCR паттерны
        ocr_group = QGroupBox("Паттерны для OCR (regex)")
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.ocr_patterns_edit = QTextEdit()
        self.ocr_patterns_edit.setMaximumHeight(100)
        self.ocr_patterns_edit.setPlaceholderText("Введите regex паттерны через перенос строки")
        ocr_layout.addWidget(self.ocr_patterns_edit)
        
        layout.addWidget(ocr_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_preview_tab(self):
        """Создает вкладку предварительного просмотра промптов."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Кнопка обновления
        self.update_preview_btn = QPushButton("🔄 Обновить предварительный просмотр")
        layout.addWidget(self.update_preview_btn)
        
        # Промпт для Gemini
        gemini_group = QGroupBox("Промпт для Gemini API")
        gemini_layout = QVBoxLayout(gemini_group)
        
        self.gemini_preview = QTextEdit()
        self.gemini_preview.setReadOnly(True)
        self.gemini_preview.setMaximumHeight(200)
        gemini_layout.addWidget(self.gemini_preview)
        
        layout.addWidget(gemini_group)
        
        # Промпт для LLM плагинов
        llm_group = QGroupBox("Промпт для LLM плагинов")
        llm_layout = QVBoxLayout(llm_group)
        
        self.llm_preview = QTextEdit()
        self.llm_preview.setReadOnly(True)
        self.llm_preview.setMaximumHeight(200)
        llm_layout.addWidget(self.llm_preview)
        
        layout.addWidget(llm_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_bottom_buttons(self):
        """Создает нижние кнопки управления."""
        layout = QHBoxLayout()
        
        # Синхронизация промптов
        self.sync_prompts_btn = QPushButton("🔄 Синхронизировать промпты всех моделей")
        self.sync_prompts_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.sync_prompts_btn)
        
        layout.addStretch()
        
        # Импорт/Экспорт
        self.export_config_btn = QPushButton("💾 Экспорт конфигурации")
        self.import_config_btn = QPushButton("📁 Импорт конфигурации")
        
        layout.addWidget(self.export_config_btn)
        layout.addWidget(self.import_config_btn)
        
        # Сброс к умолчаниям
        self.reset_defaults_btn = QPushButton("🔄 Сброс к умолчаниям")
        layout.addWidget(self.reset_defaults_btn)
        
        return layout
    
    def _connect_signals(self):
        """Подключает сигналы."""
        # Таблица полей
        self.fields_table.itemSelectionChanged.connect(self._on_field_selected)
        self.fields_table.cellChanged.connect(self._on_table_cell_changed)
        
        # Кнопки управления полями
        self.add_field_btn.clicked.connect(self._add_field)
        self.remove_field_btn.clicked.connect(self._remove_field)
        self.move_up_btn.clicked.connect(self._move_field_up)
        self.move_down_btn.clicked.connect(self._move_field_down)
        
        # Редактор поля
        self.field_id_edit.textChanged.connect(self._save_current_field)
        self.display_name_edit.textChanged.connect(self._save_current_field)
        self.description_edit.textChanged.connect(self._save_current_field)
        self.data_type_combo.currentTextChanged.connect(self._save_current_field)
        self.required_check.toggled.connect(self._save_current_field)
        self.priority_spin.valueChanged.connect(self._save_current_field)
        self.position_spin.valueChanged.connect(self._save_current_field)
        self.enabled_check.toggled.connect(self._save_current_field)
        
        # Настройки моделей
        self.gemini_keywords_edit.textChanged.connect(self._save_current_field)
        self.layoutlm_labels_edit.textChanged.connect(self._save_current_field)
        self.ocr_patterns_edit.textChanged.connect(self._save_current_field)
        
        # Управляющие кнопки
        self.sync_prompts_btn.clicked.connect(self._sync_prompts)
        self.update_preview_btn.clicked.connect(self._update_preview)
        self.export_config_btn.clicked.connect(self._export_config)
        self.import_config_btn.clicked.connect(self._import_config)
        self.reset_defaults_btn.clicked.connect(self._reset_defaults)
        
    def _load_fields(self):
        """Загружает поля в таблицу."""
        fields = field_manager.get_all_fields()
        self.fields_table.setRowCount(len(fields))
        
        for row, (field_id, field) in enumerate(sorted(fields.items(), key=lambda x: x[1].position)):
            # Включено
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            enabled_item.setCheckState(Qt.CheckState.Checked if field.enabled else Qt.CheckState.Unchecked)
            enabled_item.setData(Qt.ItemDataRole.UserRole, field_id)
            self.fields_table.setItem(row, 0, enabled_item)
            
            # Название
            name_item = QTableWidgetItem(field.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, field_id)
            self.fields_table.setItem(row, 1, name_item)
            
            # Приоритет
            priority_item = QTableWidgetItem(str(field.priority))
            priority_item.setData(Qt.ItemDataRole.UserRole, field_id)
            self.fields_table.setItem(row, 2, priority_item)
            
            # Тип
            type_item = QTableWidgetItem(field.data_type)
            type_item.setData(Qt.ItemDataRole.UserRole, field_id)
            self.fields_table.setItem(row, 3, type_item)
            
            # Обязательное
            required_item = QTableWidgetItem("Да" if field.required else "Нет")
            required_item.setData(Qt.ItemDataRole.UserRole, field_id)
            self.fields_table.setItem(row, 4, required_item)
    
    def _on_field_selected(self):
        """Обработчик выбора поля в таблице."""
        current_row = self.fields_table.currentRow()
        if current_row >= 0:
            # Получаем field_id из первой колонки
            field_id_item = self.fields_table.item(current_row, 0)
            if field_id_item:
                field_id = field_id_item.data(Qt.ItemDataRole.UserRole)
                self._load_field_to_editor(field_id)
    
    def _load_field_to_editor(self, field_id):
        """Загружает поле в редактор."""
        field = field_manager.get_field(field_id)
        if not field:
            return
        
        # Блокируем сигналы чтобы избежать циклических обновлений
        self._block_editor_signals(True)
        
        try:
            # Основные настройки
            self.field_id_edit.setText(field.id)
            self.display_name_edit.setText(field.display_name)
            self.description_edit.setPlainText(field.description)
            self.data_type_combo.setCurrentText(field.data_type)
            self.required_check.setChecked(field.required)
            self.priority_spin.setValue(field.priority)
            self.position_spin.setValue(field.position)
            self.enabled_check.setChecked(field.enabled)
            
            # Настройки моделей
            self.gemini_keywords_edit.setPlainText(", ".join(field.gemini_keywords))
            self.layoutlm_labels_edit.setPlainText(", ".join(field.layoutlm_labels))
            self.ocr_patterns_edit.setPlainText("\n".join(field.ocr_patterns))
            
        finally:
            self._block_editor_signals(False)
        
        # Обновляем предварительный просмотр
        self._update_preview()
    
    def _block_editor_signals(self, block):
        """Блокирует/разблокирует сигналы редактора."""
        widgets = [
            self.field_id_edit, self.display_name_edit, self.description_edit,
            self.data_type_combo, self.required_check, self.priority_spin,
            self.position_spin, self.enabled_check, self.gemini_keywords_edit, 
            self.layoutlm_labels_edit, self.ocr_patterns_edit
        ]
        
        for widget in widgets:
            widget.blockSignals(block)
    
    def _save_current_field(self):
        """Сохраняет текущее поле из редактора."""
        field_id = self.field_id_edit.text().strip()
        if not field_id:
            return
        
        # Собираем данные из редактора
        try:
            gemini_keywords = [kw.strip() for kw in self.gemini_keywords_edit.toPlainText().split(',') if kw.strip()]
            layoutlm_labels = [lb.strip() for lb in self.layoutlm_labels_edit.toPlainText().split(',') if lb.strip()]
            ocr_patterns = [pt.strip() for pt in self.ocr_patterns_edit.toPlainText().split('\n') if pt.strip()]
            
            # Обновляем поле
            field_manager.update_field(
                field_id,
                display_name=self.display_name_edit.text().strip(),
                description=self.description_edit.toPlainText().strip(),
                data_type=self.data_type_combo.currentText(),
                required=self.required_check.isChecked(),
                priority=self.priority_spin.value(),
                position=self.position_spin.value(),
                enabled=self.enabled_check.isChecked(),
                gemini_keywords=gemini_keywords,
                layoutlm_labels=layoutlm_labels,
                ocr_patterns=ocr_patterns
            )
            
            # Обновляем таблицу
            self._load_fields()
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка сохранения поля: {str(e)}")
    
    def _on_table_cell_changed(self, row, column):
        """Обработчик изменения ячейки таблицы."""
        if column == 0:  # Колонка "Включено"
            item = self.fields_table.item(row, 0)
            if item:
                field_id = item.data(Qt.ItemDataRole.UserRole)
                enabled = item.checkState() == Qt.CheckState.Checked
                field_manager.update_field(field_id, enabled=enabled)
    
    def _add_field(self):
        """Добавляет новое поле."""
        from PyQt6.QtWidgets import QInputDialog
        
        field_id, ok = QInputDialog.getText(
            self, "Новое поле", "Введите ID нового поля:"
        )
        
        if ok and field_id.strip():
            field_id = field_id.strip()
            
            # Проверяем, что поле не существует
            if field_manager.get_field(field_id):
                QMessageBox.warning(self, "Ошибка", f"Поле с ID '{field_id}' уже существует")
                return
            
            # Создаем новое поле
            all_fields = field_manager.get_all_fields()
            max_position = max(field.position for field in all_fields.values()) if all_fields else 0
            
            new_field = TableField(
                id=field_id,
                display_name=field_id.replace('_', ' ').title(),
                description=f"Описание поля {field_id}",
                data_type="text",
                required=False,
                priority=3,
                position=max_position + 1,  # Добавляем в конец
                gemini_keywords=[field_id],
                layoutlm_labels=[field_id.upper()],
                ocr_patterns=[],
                enabled=True
            )
            
            field_manager.add_field(new_field)
            self._load_fields()
    
    def _remove_field(self):
        """Удаляет выбранное поле."""
        current_row = self.fields_table.currentRow()
        if current_row >= 0:
            field_id_item = self.fields_table.item(current_row, 0)
            if field_id_item:
                field_id = field_id_item.data(Qt.ItemDataRole.UserRole)
                
                reply = QMessageBox.question(
                    self, "Удаление поля",
                    f"Вы уверены, что хотите удалить поле '{field_id}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    field_manager.remove_field(field_id)
                    self._load_fields()
    
    def _move_field_up(self):
        """Перемещает поле вверх (уменьшает position)."""
        current_row = self.fields_table.currentRow()
        if current_row > 0:
            field_id_item = self.fields_table.item(current_row, 0)
            if field_id_item:
                field_id = field_id_item.data(Qt.ItemDataRole.UserRole)
                if field_manager.move_field_up(field_id):
                    self._load_fields()
                    # Сохраняем выделение на том же поле (оно переместилось вверх)
                    if current_row > 0:
                        self.fields_table.selectRow(current_row - 1)
    
    def _move_field_down(self):
        """Перемещает поле вниз (увеличивает position)."""
        current_row = self.fields_table.currentRow()
        if current_row >= 0 and current_row < self.fields_table.rowCount() - 1:
            field_id_item = self.fields_table.item(current_row, 0)
            if field_id_item:
                field_id = field_id_item.data(Qt.ItemDataRole.UserRole)
                if field_manager.move_field_down(field_id):
                    self._load_fields()
                    # Сохраняем выделение на том же поле (оно переместилось вниз)
                    if current_row < self.fields_table.rowCount() - 1:
                        self.fields_table.selectRow(current_row + 1)
    
    def _sync_prompts(self):
        """Синхронизирует промпты всех моделей."""
        try:
            field_manager.sync_prompts_for_all_models()
            QMessageBox.information(
                self, "Синхронизация завершена", 
                "Промпты всех моделей успешно синхронизированы с полями таблицы!"
            )
            self.fields_updated.emit()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка синхронизации", f"Ошибка: {str(e)}")
    
    def _update_preview(self):
        """Обновляет предварительный просмотр промптов."""
        try:
            # Gemini промпт
            gemini_prompt = field_manager.get_gemini_prompt()
            self.gemini_preview.setPlainText(gemini_prompt)
            
            # LLM промпт  
            llm_prompt = field_manager._generate_llm_plugin_prompt()
            self.llm_preview.setPlainText(llm_prompt)
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка обновления предварительного просмотра: {str(e)}")
    
    def _export_config(self):
        """Экспортирует конфигурацию полей."""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Экспорт конфигурации полей", 
            "table_fields_config.json", 
            "JSON файлы (*.json)"
        )
        
        if filename:
            try:
                fields_data = {}
                for field_id, field in field_manager.get_all_fields().items():
                    fields_data[field_id] = {
                        'display_name': field.display_name,
                        'description': field.description,
                        'data_type': field.data_type,
                        'required': field.required,
                        'priority': field.priority,
                        'position': field.position,
                        'gemini_keywords': field.gemini_keywords,
                        'layoutlm_labels': field.layoutlm_labels,
                        'ocr_patterns': field.ocr_patterns,
                        'enabled': field.enabled
                    }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(fields_data, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "Экспорт завершен", f"Конфигурация сохранена в {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка: {str(e)}")
    
    def _import_config(self):
        """Импортирует конфигурацию полей."""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Импорт конфигурации полей", 
            "", "JSON файлы (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    fields_data = json.load(f)
                
                # Импортируем поля
                for field_id, field_data in fields_data.items():
                    new_field = TableField(
                        id=field_id,
                        **field_data
                    )
                    field_manager.add_field(new_field)
                
                self._load_fields()
                QMessageBox.information(self, "Импорт завершен", "Конфигурация успешно импортирована!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка импорта", f"Ошибка: {str(e)}")
    
    def _reset_defaults(self):
        """Сбрасывает поля к умолчаниям."""
        reply = QMessageBox.question(
            self, "Сброс к умолчаниям",
            "Вы уверены, что хотите сбросить все поля к умолчаниям? Все изменения будут потеряны.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Очищаем текущие поля
                for field_id in list(field_manager.get_all_fields().keys()):
                    field_manager.remove_field(field_id)
                
                # Перезагружаем поля по умолчанию
                field_manager._load_default_fields()
                field_manager.save_fields_config()
                
                self._load_fields()
                QMessageBox.information(self, "Сброс завершен", "Поля сброшены к умолчаниям!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сброса", f"Ошибка: {str(e)}")
    
    def accept(self):
        """Сохраняет изменения и закрывает диалог."""
        # Сохраняем текущее поле
        self._save_current_field()
        
        # Сохраняем конфигурацию
        field_manager.save_fields_config()
        
        # Уведомляем об изменениях
        self.fields_updated.emit()
        
        super().accept() 
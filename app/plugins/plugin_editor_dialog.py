"""
Графический редактор плагинов для InvoiceGemini
Интерфейс создания, редактирования и управления плагинами
"""
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton, QListWidget,
    QListWidgetItem, QCheckBox, QSpinBox, QFileDialog, QMessageBox,
    QProgressBar, QSplitter, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QScrollArea, QWidget, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QSyntaxHighlighter, QTextCharFormat, QColor

from .base_plugin import PluginType, PluginCapability, PluginPriority, PluginMetadata
from .advanced_plugin_manager import AdvancedPluginManager


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Подсветка синтаксиса Python"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Форматы
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor(128, 0, 255))
        self.keyword_format.setFontWeight(700)
        
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor(0, 128, 0))
        
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor(128, 128, 128))
        self.comment_format.setFontItalic(True)
        
        # Ключевые слова Python
        self.keywords = [
            'def', 'class', 'import', 'from', 'return', 'if', 'elif', 'else',
            'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'pass',
            'break', 'continue', 'True', 'False', 'None', 'self'
        ]
    
    def highlightBlock(self, text):
        import re
        
        # Ключевые слова
        for keyword in self.keywords:
            pattern = f'\\b{keyword}\\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.keyword_format)
        
        # Строки (улучшенная обработка)
        string_patterns = [
            r'"[^"\\]*(?:\\.[^"\\]*)*"',  # Двойные кавычки
            r"'[^'\\]*(?:\\.[^'\\]*)*'"   # Одинарные кавычки
        ]
        for pattern in string_patterns:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.string_format)
        
        # Комментарии
        comment_match = re.search(r'#.*$', text)
        if comment_match:
            self.setFormat(comment_match.start(), comment_match.end() - comment_match.start(), self.comment_format)


class PluginTemplateGenerator:
    """Генератор шаблонов плагинов"""
    
    @staticmethod
    def get_template(plugin_type: PluginType) -> str:
        """Возвращает шаблон кода для плагина"""
        
        templates = {
            PluginType.LLM: """
from app.plugins.base_plugin import LLMPlugin, PluginMetadata, PluginType, PluginCapability

class CustomLLMPlugin(LLMPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Custom LLM Plugin",
            version="1.0.0",
            description="Описание вашего LLM плагина",
            author="Ваше имя",
            plugin_type=PluginType.LLM,
            capabilities=[PluginCapability.TEXT, PluginCapability.VISION],
            dependencies=[]
        )
    
    def initialize(self) -> bool:
        try:
            # Инициализация модели
            return True
        except Exception as e:
            self.set_error(str(e))
            return False
    
    def cleanup(self):
        # Очистка ресурсов
        pass
    
    def load_model(self) -> bool:
        # Загрузка модели
        return True
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        # Генерация ответа
        return "Ответ модели"
""",
            
            PluginType.EXPORTER: """
from app.plugins.base_plugin import ExporterPlugin, PluginMetadata, PluginType, PluginCapability
from typing import Any, List

class CustomExporterPlugin(ExporterPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Custom Exporter Plugin",
            version="1.0.0",
            description="Описание вашего экспортера",
            author="Ваше имя",
            plugin_type=PluginType.EXPORTER,
            capabilities=[PluginCapability.TEXT],
            supported_formats=["custom"]
        )
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self):
        pass
    
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        try:
            # Логика экспорта
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            return True
        except Exception as e:
            self.set_error(str(e))
            return False
    
    def get_supported_formats(self) -> List[str]:
        return ["custom"]
""",
            
            PluginType.INTEGRATION: """
from app.plugins.base_plugin import IntegrationPlugin, PluginMetadata, PluginType, PluginCapability
from typing import Any, Dict

class CustomIntegrationPlugin(IntegrationPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Custom Integration Plugin",
            version="1.0.0",
            description="Интеграция с внешней системой",
            author="Ваше имя",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[PluginCapability.API],
            config_schema={
                "required": ["api_url", "api_key"],
                "types": {"api_url": str, "api_key": str}
            }
        )
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self):
        if hasattr(self, '_connected'):
            self.disconnect()
    
    def connect(self, **kwargs) -> bool:
        # Подключение к внешней системе
        api_url = self.config.get('api_url')
        api_key = self.config.get('api_key')
        
        # Логика подключения
        self._connected = True
        return True
    
    def disconnect(self):
        self._connected = False
    
    def test_connection(self) -> bool:
        return getattr(self, '_connected', False)
    
    def sync_data(self, data: Any, direction: str = "export") -> Dict[str, Any]:
        # Синхронизация данных
        return {"status": "success", "synced_items": 0}
"""
        }
        
        return templates.get(plugin_type, templates[PluginType.EXPORTER])


class PluginCreatorWidget(QWidget):
    """Виджет создания нового плагина"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Основная информация
        info_group = QGroupBox("Основная информация")
        info_layout = QFormLayout(info_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Введите название плагина")
        info_layout.addRow("Название:", self.name_edit)
        
        self.version_edit = QLineEdit("1.0.0")
        info_layout.addRow("Версия:", self.version_edit)
        
        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("Ваше имя")
        info_layout.addRow("Автор:", self.author_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Описание плагина")
        info_layout.addRow("Описание:", self.description_edit)
        
        layout.addWidget(info_group)
        
        # Тип плагина
        type_group = QGroupBox("Тип плагина")
        type_layout = QVBoxLayout(type_group)
        
        self.type_combo = QComboBox()
        for plugin_type in PluginType:
            self.type_combo.addItem(plugin_type.value.upper(), plugin_type)
        self.type_combo.currentTextChanged.connect(self.update_template)
        
        type_layout.addWidget(self.type_combo)
        layout.addWidget(type_group)
        
        # Возможности
        capabilities_group = QGroupBox("Возможности")
        capabilities_layout = QGridLayout(capabilities_group)
        
        self.capability_checkboxes = {}
        row, col = 0, 0
        for capability in PluginCapability:
            checkbox = QCheckBox(capability.value.upper())
            checkbox.setObjectName(capability.value)
            self.capability_checkboxes[capability] = checkbox
            capabilities_layout.addWidget(checkbox, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        layout.addWidget(capabilities_group)
        
        # Зависимости
        deps_group = QGroupBox("Зависимости")
        deps_layout = QVBoxLayout(deps_group)
        
        self.dependencies_edit = QTextEdit()
        self.dependencies_edit.setMaximumHeight(60)
        self.dependencies_edit.setPlaceholderText("Список зависимостей (по одной на строку)")
        deps_layout.addWidget(self.dependencies_edit)
        
        layout.addWidget(deps_group)
        
        # Редактор кода
        code_group = QGroupBox("Код плагина")
        code_layout = QVBoxLayout(code_group)
        
        self.code_edit = QTextEdit()
        self.code_edit.setFont(QFont("Consolas", 10))
        self.highlighter = PythonSyntaxHighlighter(self.code_edit.document())
        
        code_layout.addWidget(self.code_edit)
        layout.addWidget(code_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Генерировать шаблон")
        self.generate_btn.clicked.connect(self.generate_template)
        buttons_layout.addWidget(self.generate_btn)
        
        self.validate_btn = QPushButton("Проверить код")
        self.validate_btn.clicked.connect(self.validate_code)
        buttons_layout.addWidget(self.validate_btn)
        
        self.save_btn = QPushButton("Сохранить плагин")
        self.save_btn.clicked.connect(self.save_plugin)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        
        # Инициализация
        self.update_template()
    
    def update_template(self):
        """Обновляет шаблон кода при изменении типа плагина"""
        plugin_type = self.type_combo.currentData()
        if plugin_type:
            template = PluginTemplateGenerator.get_template(plugin_type)
            self.code_edit.setText(template)
    
    def generate_template(self):
        """Генерирует персонализированный шаблон"""
        plugin_type = self.type_combo.currentData()
        template = PluginTemplateGenerator.get_template(plugin_type)
        
        # Персонализация
        name = self.name_edit.text() or "Custom Plugin"
        author = self.author_edit.text() or "Unknown"
        description = self.description_edit.toPlainText() or "Plugin description"
        version = self.version_edit.text() or "1.0.0"
        
        # Получаем выбранные возможности
        capabilities = []
        for capability, checkbox in self.capability_checkboxes.items():
            if checkbox.isChecked():
                capabilities.append(f"PluginCapability.{capability.name}")
        
        capabilities_str = "[" + ", ".join(capabilities) + "]" if capabilities else "[]"
        
        # Заменяем в шаблоне
        template = template.replace("Custom LLM Plugin", name)
        template = template.replace("Custom Exporter Plugin", name)  
        template = template.replace("Custom Integration Plugin", name)
        template = template.replace("Ваше имя", author)
        template = template.replace("Описание вашего LLM плагина", description)
        template = template.replace("Описание вашего экспортера", description)
        template = template.replace("Интеграция с внешней системой", description)
        template = template.replace("1.0.0", version)
        template = template.replace("[PluginCapability.TEXT, PluginCapability.VISION]", capabilities_str)
        template = template.replace("[PluginCapability.TEXT]", capabilities_str)
        template = template.replace("[PluginCapability.API]", capabilities_str)
        
        self.code_edit.setText(template)
    
    def validate_code(self):
        """Проверяет синтаксис кода"""
        code = self.code_edit.toPlainText()
        try:
            compile(code, '<string>', 'exec')
            QMessageBox.information(self, "Проверка", "✅ Код синтаксически корректен!")
        except SyntaxError as e:
            QMessageBox.warning(self, "Ошибка синтаксиса", 
                              f"❌ Ошибка в строке {e.lineno}:\n{e.msg}")
    
    def save_plugin(self):
        """Сохраняет плагин"""
        name = self.name_edit.text()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите название плагина!")
            return
        
        code = self.code_edit.toPlainText()
        if not code.strip():
            QMessageBox.warning(self, "Ошибка", "Код плагина не может быть пустым!")
            return
        
        # Выбираем директорию
        plugins_dir = Path("plugins/user")
        plugins_dir.mkdir(parents=True, exist_ok=True)
        
        plugin_id = name.lower().replace(" ", "_")
        plugin_dir = plugins_dir / plugin_id
        plugin_dir.mkdir(exist_ok=True)
        
        # Сохраняем код плагина
        plugin_file = plugin_dir / f"{plugin_id}_plugin.py"
        with open(plugin_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Создаем метаданные
        metadata = {
            "name": name,
            "version": self.version_edit.text(),
            "description": self.description_edit.toPlainText(),
            "author": self.author_edit.text(),
            "type": self.type_combo.currentData().value,
            "capabilities": [cap.value for cap, cb in self.capability_checkboxes.items() if cb.isChecked()],
            "dependencies": [dep.strip() for dep in self.dependencies_edit.toPlainText().split('\n') if dep.strip()]
        }
        
        metadata_file = plugin_dir / "plugin.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Создаем requirements.txt если есть зависимости
        if metadata["dependencies"]:
            requirements_file = plugin_dir / "requirements.txt"
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(metadata["dependencies"]))
        
        QMessageBox.information(self, "Успех", 
                              f"✅ Плагин '{name}' сохранен в {plugin_dir}")


class PluginManagerDialog(QDialog):
    """Диалог управления плагинами"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plugin_manager = AdvancedPluginManager()
        self.setup_ui()
        self.load_plugins()
    
    def setup_ui(self):
        self.setWindowTitle("Менеджер плагинов InvoiceGemini")
        self.setFixedSize(1200, 800)
        
        layout = QVBoxLayout(self)
        
        # Вкладки
        self.tabs = QTabWidget()
        
        # Вкладка создания плагинов
        self.creator_tab = PluginCreatorWidget()
        self.tabs.addTab(self.creator_tab, "🛠️ Создать плагин")
        
        # Вкладка управления
        self.management_tab = self.create_management_tab()
        self.tabs.addTab(self.management_tab, "📦 Управление")
        
        # Вкладка магазина
        self.store_tab = self.create_store_tab()
        self.tabs.addTab(self.store_tab, "🛒 Магазин")
        
        layout.addWidget(self.tabs)
        
        # Строка состояния
        self.status_label = QLabel("Готов")
        layout.addWidget(self.status_label)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Callbacks
        self.plugin_manager.set_status_callback(self.update_status)
        self.plugin_manager.set_progress_callback(self.update_progress)
    
    def create_management_tab(self) -> QWidget:
        """Создает вкладку управления плагинами"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Список установленных плагинов
        plugins_group = QGroupBox("Установленные плагины")
        plugins_layout = QVBoxLayout(plugins_group)
        
        self.plugins_tree = QTreeWidget()
        self.plugins_tree.setHeaderLabels(["Название", "Версия", "Тип", "Статус"])
        plugins_layout.addWidget(self.plugins_tree)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.clicked.connect(self.load_plugins)
        buttons_layout.addWidget(self.refresh_btn)
        
        self.update_btn = QPushButton("⬆️ Обновить плагин")
        self.update_btn.clicked.connect(self.update_selected_plugin)
        buttons_layout.addWidget(self.update_btn)
        
        self.remove_btn = QPushButton("🗑️ Удалить")
        self.remove_btn.clicked.connect(self.remove_selected_plugin)
        buttons_layout.addWidget(self.remove_btn)
        
        plugins_layout.addLayout(buttons_layout)
        layout.addWidget(plugins_group)
        
        return widget
    
    def create_store_tab(self) -> QWidget:
        """Создает вкладку магазина плагинов"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Поиск
        search_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Поиск плагинов...")
        self.search_edit.returnPressed.connect(self.search_plugins)
        search_layout.addWidget(self.search_edit)
        
        self.search_btn = QPushButton("🔍 Поиск")
        self.search_btn.clicked.connect(self.search_plugins)
        search_layout.addWidget(self.search_btn)
        
        layout.addLayout(search_layout)
        
        # Результаты поиска
        self.search_results = QListWidget()
        layout.addWidget(self.search_results)
        
        # Кнопки
        store_buttons = QHBoxLayout()
        
        self.update_catalogs_btn = QPushButton("📥 Обновить каталоги")
        self.update_catalogs_btn.clicked.connect(self.update_catalogs)
        store_buttons.addWidget(self.update_catalogs_btn)
        
        self.install_btn = QPushButton("💾 Установить")
        self.install_btn.clicked.connect(self.install_selected_plugin)
        store_buttons.addWidget(self.install_btn)
        
        layout.addLayout(store_buttons)
        
        return widget
    
    def load_plugins(self):
        """Загружает список установленных плагинов"""
        self.plugins_tree.clear()
        
        plugins = self.plugin_manager.get_installed_plugins()
        for plugin in plugins:
            item = QTreeWidgetItem([
                plugin.get("name", "Unknown"),
                plugin.get("version", "0.0.0"),
                plugin.get("type", "unknown"),
                "Установлен"
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, plugin)
            self.plugins_tree.addItem(item)
    
    def update_selected_plugin(self):
        """Обновляет выбранный плагин"""
        current = self.plugins_tree.currentItem()
        if not current:
            QMessageBox.warning(self, "Предупреждение", "Выберите плагин для обновления")
            return
        
        plugin_data = current.data(0, Qt.ItemDataRole.UserRole)
        plugin_id = plugin_data.get("name", "").lower().replace(" ", "_")
        
        if self.plugin_manager.update_plugin(plugin_id):
            self.load_plugins()
    
    def remove_selected_plugin(self):
        """Удаляет выбранный плагин"""
        current = self.plugins_tree.currentItem()
        if not current:
            QMessageBox.warning(self, "Предупреждение", "Выберите плагин для удаления")
            return
        
        plugin_data = current.data(0, Qt.ItemDataRole.UserRole)
        plugin_name = plugin_data.get("name", "Unknown")
        plugin_id = plugin_name.lower().replace(" ", "_")
        
        reply = QMessageBox.question(self, "Подтверждение", 
                                   f"Удалить плагин '{plugin_name}'?")
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.plugin_manager.uninstall_plugin(plugin_id):
                self.load_plugins()
    
    def search_plugins(self):
        """Поиск плагинов в репозиториях"""
        query = self.search_edit.text().strip()
        if not query:
            return
        
        self.search_results.clear()
        results = self.plugin_manager.search_plugins(query)
        
        for plugin in results:
            item_text = f"{plugin['name']} v{plugin.get('version', '1.0.0')} - {plugin.get('description', '')}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, plugin)
            self.search_results.addItem(item)
    
    def update_catalogs(self):
        """Обновляет каталоги репозиториев"""
        self.plugin_manager.update_repositories()
        QMessageBox.information(self, "Информация", "Каталоги обновлены")
    
    def install_selected_plugin(self):
        """Устанавливает выбранный плагин"""
        current = self.search_results.currentItem()
        if not current:
            QMessageBox.warning(self, "Предупреждение", "Выберите плагин для установки")
            return
        
        plugin_data = current.data(Qt.ItemDataRole.UserRole)
        plugin_id = plugin_data.get("id", "")
        
        if self.plugin_manager.install_plugin(plugin_id):
            self.load_plugins()
            QMessageBox.information(self, "Успех", f"Плагин '{plugin_data['name']}' установлен")
    
    def update_status(self, message: str):
        """Обновляет строку состояния"""
        self.status_label.setText(message)
    
    def update_progress(self, value: int, message: str = ""):
        """Обновляет прогресс бар"""
        if value == 0:
            self.progress_bar.setVisible(True)
        elif value == 100:
            self.progress_bar.setVisible(False)
        
        self.progress_bar.setValue(value)
        if message:
            self.update_status(message) 
"""
Диалог интеграции с Paperless-NGX и Paperless-AI
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QMessageBox,
    QTabWidget, QWidget, QComboBox, QListWidget, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from app.core.scheduler import get_scheduler, ScheduleInterval


class PaperlessSyncWorker(QThread):
    """Воркер для синхронизации с Paperless в фоне"""
    progress_updated = pyqtSignal(int, str)
    sync_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, plugin, data: Dict[str, Any], direction: str):
        super().__init__()
        self.plugin = plugin
        self.data = data
        self.direction = direction
    
    def run(self):
        try:
            self.progress_updated.emit(10, "Подключение к Paperless...")
            
            if not self.plugin.test_connection():
                if not self.plugin.connect():
                    self.error_occurred.emit("Не удалось подключиться к Paperless")
                    return
            
            self.progress_updated.emit(30, f"Синхронизация ({self.direction})...")
            
            result = self.plugin.sync_data(self.data, self.direction)
            
            self.progress_updated.emit(100, "Синхронизация завершена")
            self.sync_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка синхронизации: {e}")


class PaperlessIntegrationDialog(QDialog):
    """Диалог настройки и управления интеграцией с Paperless-NGX"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Интеграция с Paperless-NGX"))
        self.setMinimumSize(700, 600)
        
        self.paperless_plugin = None
        self.paperless_ai_plugin = None
        self.sync_worker = None
        
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)
        
        # Вкладки
        tabs = QTabWidget()
        
        # Вкладка настроек подключения
        tabs.addTab(self._create_connection_tab(), self.tr("Подключение"))
        
        # Вкладка синхронизации
        tabs.addTab(self._create_sync_tab(), self.tr("Синхронизация"))
        
        # Вкладка AI настроек
        tabs.addTab(self._create_ai_tab(), self.tr("AI Тегирование"))
        
        # Вкладка статуса
        tabs.addTab(self._create_status_tab(), self.tr("Статус"))
        
        layout.addWidget(tabs)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.test_btn = QPushButton(self.tr("Проверить подключение"))
        self.test_btn.clicked.connect(self._test_connection)
        buttons_layout.addWidget(self.test_btn)
        
        buttons_layout.addStretch()
        
        self.save_btn = QPushButton(self.tr("Сохранить"))
        self.save_btn.clicked.connect(self._save_settings)
        buttons_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton(self.tr("Закрыть"))
        self.close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def _create_connection_tab(self) -> QWidget:
        """Создает вкладку настроек подключения"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа основных настроек
        main_group = QGroupBox(self.tr("Основные настройки"))
        main_layout = QFormLayout()
        
        self.server_url_edit = QLineEdit()
        self.server_url_edit.setPlaceholderText("http://192.168.1.125:8000")
        main_layout.addRow(self.tr("URL сервера:"), self.server_url_edit)
        
        self.api_token_edit = QLineEdit()
        self.api_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_token_edit.setPlaceholderText(self.tr("Введите API токен"))
        
        token_layout = QHBoxLayout()
        token_layout.addWidget(self.api_token_edit)
        
        self.show_token_btn = QPushButton(self.tr("👁"))
        self.show_token_btn.setMaximumWidth(40)
        self.show_token_btn.setCheckable(True)
        self.show_token_btn.toggled.connect(self._toggle_token_visibility)
        token_layout.addWidget(self.show_token_btn)
        
        main_layout.addRow(self.tr("API токен:"), token_layout)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setSuffix(self.tr(" сек"))
        main_layout.addRow(self.tr("Таймаут:"), self.timeout_spin)
        
        self.ssl_verify_check = QCheckBox(self.tr("Проверять SSL сертификат"))
        self.ssl_verify_check.setChecked(True)
        main_layout.addRow("", self.ssl_verify_check)
        
        main_group.setLayout(main_layout)
        layout.addWidget(main_group)
        
        # Группа автоматизации
        auto_group = QGroupBox(self.tr("Автоматизация"))
        auto_layout = QFormLayout()
        
        self.auto_sync_check = QCheckBox(self.tr("Автоматическая синхронизация после обработки"))
        auto_layout.addRow("", self.auto_sync_check)
        
        self.sync_interval_spin = QSpinBox()
        self.sync_interval_spin.setRange(60, 3600)
        self.sync_interval_spin.setValue(300)
        self.sync_interval_spin.setSuffix(self.tr(" сек"))
        auto_layout.addRow(self.tr("Интервал синхронизации:"), self.sync_interval_spin)
        
        self.create_correspondents_check = QCheckBox(self.tr("Создавать корреспондентов автоматически"))
        self.create_correspondents_check.setChecked(True)
        auto_layout.addRow("", self.create_correspondents_check)
        
        self.create_doc_types_check = QCheckBox(self.tr("Создавать типы документов автоматически"))
        self.create_doc_types_check.setChecked(True)
        auto_layout.addRow("", self.create_doc_types_check)
        
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_sync_tab(self) -> QWidget:
        """Создает вкладку синхронизации"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа направления синхронизации
        direction_group = QGroupBox(self.tr("Направление синхронизации"))
        direction_layout = QVBoxLayout()
        
        self.export_radio = QCheckBox(self.tr("Экспорт в Paperless-NGX"))
        self.export_radio.setChecked(True)
        direction_layout.addWidget(self.export_radio)
        
        self.import_radio = QCheckBox(self.tr("Импорт из Paperless-NGX"))
        direction_layout.addWidget(self.import_radio)
        
        self.bidirectional_radio = QCheckBox(self.tr("Двусторонняя синхронизация"))
        direction_layout.addWidget(self.bidirectional_radio)
        
        direction_group.setLayout(direction_layout)
        layout.addWidget(direction_group)
        
        # Прогресс синхронизации
        progress_group = QGroupBox(self.tr("Прогресс"))
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel(self.tr("Готово к синхронизации"))
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Кнопки действий
        actions_layout = QHBoxLayout()
        
        self.sync_current_btn = QPushButton(self.tr("Синхронизировать текущий документ"))
        self.sync_current_btn.clicked.connect(self._sync_current_document)
        actions_layout.addWidget(self.sync_current_btn)
        
        self.sync_all_btn = QPushButton(self.tr("Синхронизировать все"))
        self.sync_all_btn.clicked.connect(self._sync_all_documents)
        actions_layout.addWidget(self.sync_all_btn)
        
        layout.addLayout(actions_layout)
        
        # Лог синхронизации
        log_group = QGroupBox(self.tr("Лог синхронизации"))
        log_layout = QVBoxLayout()
        
        self.sync_log = QTextEdit()
        self.sync_log.setReadOnly(True)
        self.sync_log.setMaximumHeight(200)
        log_layout.addWidget(self.sync_log)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_ai_tab(self) -> QWidget:
        """Создает вкладку AI настроек"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа Paperless-AI
        ai_group = QGroupBox(self.tr("Paperless-AI"))
        ai_layout = QFormLayout()
        
        self.ai_enabled_check = QCheckBox(self.tr("Включить AI тегирование"))
        self.ai_enabled_check.setChecked(True)
        ai_layout.addRow("", self.ai_enabled_check)
        
        self.auto_tag_check = QCheckBox(self.tr("Автоматическое применение тегов"))
        self.auto_tag_check.setChecked(True)
        ai_layout.addRow("", self.auto_tag_check)
        
        self.auto_categorize_check = QCheckBox(self.tr("Автоматическая категоризация"))
        self.auto_categorize_check.setChecked(True)
        ai_layout.addRow("", self.auto_categorize_check)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setDecimals(2)
        ai_layout.addRow(self.tr("Порог уверенности:"), self.confidence_spin)
        
        self.sync_tags_check = QCheckBox(self.tr("Синхронизировать теги обратно в InvoiceGemini"))
        self.sync_tags_check.setChecked(True)
        ai_layout.addRow("", self.sync_tags_check)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Тестирование AI
        test_group = QGroupBox(self.tr("Тестирование"))
        test_layout = QVBoxLayout()
        
        test_btn_layout = QHBoxLayout()
        self.test_ai_btn = QPushButton(self.tr("Протестировать AI тегирование"))
        self.test_ai_btn.clicked.connect(self._test_ai_tagging)
        test_btn_layout.addWidget(self.test_ai_btn)
        test_layout.addLayout(test_btn_layout)
        
        self.ai_test_result = QTextEdit()
        self.ai_test_result.setReadOnly(True)
        self.ai_test_result.setMaximumHeight(150)
        test_layout.addWidget(self.ai_test_result)
        
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)
        
        # Расширенные функции AI
        advanced_group = QGroupBox(self.tr("Расширенные функции AI"))
        advanced_layout = QVBoxLayout()
        
        advanced_info = QLabel(
            self.tr("Кастомные правила тегирования, статистика и обучение AI моделей")
        )
        advanced_info.setWordWrap(True)
        advanced_layout.addWidget(advanced_info)
        
        self.advanced_ai_btn = QPushButton(self.tr("🎓 Управление AI (правила, статистика, обучение)"))
        self.advanced_ai_btn.clicked.connect(self._show_advanced_ai_dialog)
        advanced_layout.addWidget(self.advanced_ai_btn)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_status_tab(self) -> QWidget:
        """Создает вкладку статуса"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Статус подключения
        status_group = QGroupBox(self.tr("Статус подключения"))
        status_layout = QFormLayout()
        
        self.connection_status_label = QLabel(self.tr("Не подключено"))
        status_layout.addRow(self.tr("Соединение:"), self.connection_status_label)
        
        self.last_sync_label = QLabel(self.tr("Никогда"))
        status_layout.addRow(self.tr("Последняя синхронизация:"), self.last_sync_label)
        
        self.cached_items_label = QLabel(self.tr("0 / 0 / 0"))
        status_layout.addRow(self.tr("Кэш (корр./типы/теги):"), self.cached_items_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Статистика
        stats_group = QGroupBox(self.tr("Статистика"))
        stats_layout = QFormLayout()
        
        self.exported_count_label = QLabel("0")
        stats_layout.addRow(self.tr("Экспортировано:"), self.exported_count_label)
        
        self.imported_count_label = QLabel("0")
        stats_layout.addRow(self.tr("Импортировано:"), self.imported_count_label)
        
        self.errors_count_label = QLabel("0")
        stats_layout.addRow(self.tr("Ошибок:"), self.errors_count_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Кнопка обновления статуса
        refresh_btn = QPushButton(self.tr("Обновить статус"))
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        return widget
    
    def _toggle_token_visibility(self, checked: bool):
        """Переключает видимость токена"""
        if checked:
            self.api_token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.api_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
    
    def _test_connection(self):
        """Тестирует подключение к Paperless"""
        try:
            # Создаем временный плагин с текущими настройками
            config = self._get_current_config()
            
            from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin
            
            plugin = PaperlessNGXPlugin(config)
            
            if not plugin.initialize():
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                   self.tr(f"Ошибка инициализации: {plugin.last_error}"))
                return
            
            if plugin.test_connection():
                QMessageBox.information(self, self.tr("Успех"), 
                                       self.tr("Подключение к Paperless-NGX успешно!"))
                self._log_sync(self.tr("✅ Подключение успешно"))
                
                # Обновляем статус
                if plugin.connect():
                    status = plugin.get_connection_status()
                    self.connection_status_label.setText(self.tr("✅ Подключено"))
                    self.cached_items_label.setText(
                        f"{status.get('cached_correspondents', 0)} / "
                        f"{status.get('cached_document_types', 0)} / "
                        f"{status.get('cached_tags', 0)}"
                    )
            else:
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                   self.tr("Не удалось подключиться к Paperless-NGX.\nПроверьте настройки."))
                self._log_sync(self.tr("❌ Ошибка подключения"))
            
            plugin.cleanup()
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка тестирования: {e}"))
            self._log_sync(self.tr(f"❌ Ошибка: {e}"))
    
    def _save_settings(self):
        """Сохраняет настройки"""
        try:
            config = self._get_current_config()
            
            # Сохраняем в SecretsManager
            from app.security.secrets_manager import get_secrets_manager
            secrets_manager = get_secrets_manager()
            
            # Сохраняем API токен безопасно
            secrets_manager.set_secret("paperless_ngx_api_token", config["api_token"])
            secrets_manager.set_secret("paperless_ngx_server_url", config["server_url"])
            
            # Сохраняем остальные настройки
            from app.settings_manager import settings_manager
            settings_manager.set_setting("paperless_timeout", config["timeout"])
            settings_manager.set_setting("paperless_ssl_verify", config["ssl_verify"])
            settings_manager.set_setting("paperless_auto_sync", config["auto_sync"])
            settings_manager.set_setting("paperless_sync_interval", config["sync_interval"])
            settings_manager.set_setting("paperless_create_correspondents", config["create_correspondents"])
            settings_manager.set_setting("paperless_create_document_types", config["create_document_types"])
            settings_manager.set_setting("paperless_ai_enabled", config["paperless_ai_enabled"])
            settings_manager.set_setting("paperless_auto_tag", config["auto_tag"])
            settings_manager.set_setting("paperless_auto_categorize", config.get("auto_categorize", True))
            settings_manager.set_setting("paperless_confidence_threshold", config.get("confidence_threshold", 0.7))
            settings_manager.set_setting("paperless_sync_tags_to_invoicegemini", config.get("sync_tags_to_invoicegemini", True))
            
            # Настраиваем планировщик
            self._configure_scheduler(config)
            
            QMessageBox.information(self, self.tr("Успех"), 
                                   self.tr("Настройки сохранены!"))
            self._log_sync(self.tr("💾 Настройки сохранены"))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка сохранения настроек: {e}"))
            logging.error(f"Ошибка сохранения настроек Paperless: {e}")
    
    def _load_settings(self):
        """Загружает настройки"""
        try:
            from app.security.secrets_manager import get_secrets_manager
            from app.settings_manager import settings_manager
            
            secrets_manager = get_secrets_manager()
            
            # Загружаем из SecretsManager
            api_token = secrets_manager.get_secret("paperless_ngx_api_token")
            server_url = secrets_manager.get_secret("paperless_ngx_server_url")
            
            if server_url:
                self.server_url_edit.setText(server_url)
            else:
                self.server_url_edit.setText("http://192.168.1.125:8000")
            
            if api_token:
                self.api_token_edit.setText(api_token)
            
            # Загружаем остальные настройки
            self.timeout_spin.setValue(int(settings_manager.get_setting("paperless_timeout", 30)))
            self.ssl_verify_check.setChecked(bool(settings_manager.get_setting("paperless_ssl_verify", True)))
            self.auto_sync_check.setChecked(bool(settings_manager.get_setting("paperless_auto_sync", False)))
            self.sync_interval_spin.setValue(int(settings_manager.get_setting("paperless_sync_interval", 300)))
            self.create_correspondents_check.setChecked(settings_manager.get_setting("paperless_create_correspondents", True))
            self.create_doc_types_check.setChecked(settings_manager.get_setting("paperless_create_document_types", True))
            self.ai_enabled_check.setChecked(settings_manager.get_setting("paperless_ai_enabled", True))
            self.auto_tag_check.setChecked(settings_manager.get_setting("paperless_auto_tag", True))
            self.auto_categorize_check.setChecked(settings_manager.get_setting("paperless_auto_categorize", True))
            self.confidence_spin.setValue(settings_manager.get_setting("paperless_confidence_threshold", 0.7))
            self.sync_tags_check.setChecked(settings_manager.get_setting("paperless_sync_tags_to_invoicegemini", True))
            
        except Exception as e:
            logging.error(f"Ошибка загрузки настроек Paperless: {e}")
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Получает текущую конфигурацию из UI"""
        return {
            "server_url": self.server_url_edit.text().strip(),
            "api_token": self.api_token_edit.text().strip(),
            "timeout": self.timeout_spin.value(),
            "ssl_verify": self.ssl_verify_check.isChecked(),
            "auto_sync": self.auto_sync_check.isChecked(),
            "sync_interval": self.sync_interval_spin.value(),
            "create_correspondents": self.create_correspondents_check.isChecked(),
            "create_document_types": self.create_doc_types_check.isChecked(),
            "paperless_ai_enabled": self.ai_enabled_check.isChecked(),
            "auto_tag": self.auto_tag_check.isChecked(),
            "auto_categorize": self.auto_categorize_check.isChecked(),
            "confidence_threshold": self.confidence_spin.value(),
            "sync_tags_to_invoicegemini": self.sync_tags_check.isChecked()
        }
    
    def _sync_current_document(self):
        """Синхронизирует текущий документ"""
        try:
            # Получаем текущие данные из parent (main_window)
            if not hasattr(self.parent(), 'current_invoice_data'):
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                   self.tr("Нет обработанного документа для синхронизации"))
                return
            
            current_data = self.parent().current_invoice_data
            
            if not current_data:
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                   self.tr("Данные документа пусты"))
                return
            
            # Проверяем наличие файла
            if not hasattr(self.parent(), 'current_file_path') or not self.parent().current_file_path:
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                   self.tr("Путь к файлу не найден"))
                return
            
            # Добавляем путь к файлу
            sync_data = current_data.copy()
            sync_data['file_path'] = self.parent().current_file_path
            
            # Определяем направление синхронизации
            direction = "export"
            if self.export_radio.isChecked() and self.import_radio.isChecked():
                direction = "both"
            elif self.import_radio.isChecked():
                direction = "import"
            
            # Запускаем синхронизацию в фоне
            self._start_sync(sync_data, direction, single_document=True)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка синхронизации: {e}"))
            logging.error(f"Ошибка синхронизации текущего документа: {e}", exc_info=True)
    
    def _sync_all_documents(self):
        """Синхронизирует все документы"""
        try:
            # Получаем список всех обработанных документов
            if not hasattr(self.parent(), 'get_all_processed_documents'):
                QMessageBox.warning(self, self.tr("Предупреждение"), 
                                   self.tr("Функция массовой синхронизации недоступна в текущей версии"))
                return
            
            all_docs = self.parent().get_all_processed_documents()
            
            if not all_docs:
                QMessageBox.information(self, self.tr("Информация"), 
                                       self.tr("Нет обработанных документов для синхронизации"))
                return
            
            # Подтверждение
            reply = QMessageBox.question(
                self, 
                self.tr("Подтверждение"), 
                self.tr(f"Синхронизировать {len(all_docs)} документов?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Определяем направление
                direction = "export"
                if self.export_radio.isChecked() and self.import_radio.isChecked():
                    direction = "both"
                elif self.import_radio.isChecked():
                    direction = "import"
                
                # Запускаем массовую синхронизацию
                self._start_batch_sync(all_docs, direction)
                
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка массовой синхронизации: {e}"))
            logging.error(f"Ошибка массовой синхронизации: {e}", exc_info=True)
    
    def _test_ai_tagging(self):
        """Тестирует AI тегирование"""
        try:
            self.ai_test_result.clear()
            self.ai_test_result.append(self.tr("🤖 Тестирование AI тегирования...\n"))
            
            # Проверяем наличие AI плагина
            if not self.paperless_ai_plugin:
                self.ai_test_result.append(self.tr("❌ AI плагин не загружен"))
                return
            
            # Получаем тестовый документ
            if not hasattr(self.parent(), 'current_invoice_data') or not self.parent().current_invoice_data:
                self.ai_test_result.append(self.tr("❌ Нет документа для тестирования"))
                self.ai_test_result.append(self.tr("\nОбработайте документ перед тестированием AI"))
                return
            
            test_data = self.parent().current_invoice_data
            
            # Запускаем анализ
            self.ai_test_result.append(self.tr("📊 Анализ документа...\n"))
            
            # Получаем теги от AI
            tags = self.paperless_ai_plugin.analyze_document_for_tags(test_data)
            
            if tags:
                self.ai_test_result.append(self.tr(f"✅ Найдено {len(tags)} тегов:\n"))
                for tag_name, confidence in tags:
                    self.ai_test_result.append(self.tr(f"  • {tag_name} (уверенность: {confidence:.1%})"))
            else:
                self.ai_test_result.append(self.tr("⚠️ Теги не найдены"))
            
            # Получаем категорию
            if test_data.get('category'):
                self.ai_test_result.append(self.tr(f"\n📂 Категория: {test_data.get('category')}"))
            
            self.ai_test_result.append(self.tr("\n\n✅ Тестирование завершено"))
            
        except Exception as e:
            self.ai_test_result.append(self.tr(f"\n❌ Ошибка: {e}"))
            logging.error(f"Ошибка тестирования AI: {e}", exc_info=True)
    
    def _start_sync(self, data, direction, single_document=False):
        """Запускает синхронизацию в фоновом потоке"""
        try:
            from PyQt6.QtCore import QThread, pyqtSignal
            
            class SyncWorker(QThread):
                finished = pyqtSignal(bool, str)
                progress = pyqtSignal(str)
                
                def __init__(self, plugin, data, direction):
                    super().__init__()
                    self.plugin = plugin
                    self.data = data
                    self.direction = direction
                
                def run(self):
                    try:
                        self.progress.emit(self.tr("🔄 Начало синхронизации..."))
                        
                        # Выполняем синхронизацию через плагин
                        result = self.plugin.sync_data(self.data, direction=self.direction)
                        
                        if result.get('success'):
                            msg = self.tr("✅ Синхронизация завершена успешно")
                            self.finished.emit(True, msg)
                        else:
                            msg = self.tr(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                            self.finished.emit(False, msg)
                    except Exception as e:
                        self.finished.emit(False, str(e))
            
            # Создаем и запускаем worker
            self.sync_worker = SyncWorker(self.paperless_plugin, data, direction)
            self.sync_worker.progress.connect(lambda msg: self._log_sync(msg))
            self.sync_worker.finished.connect(self._on_sync_finished)
            self.sync_worker.start()
            
            # Показываем прогресс
            self._log_sync(self.tr("🔄 Синхронизация начата..."))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка запуска синхронизации: {e}"))
            logging.error(f"Ошибка запуска синхронизации: {e}", exc_info=True)
    
    def _start_batch_sync(self, documents, direction):
        """Запускает массовую синхронизацию"""
        try:
            from PyQt6.QtCore import QThread, pyqtSignal
            
            class BatchSyncWorker(QThread):
                finished = pyqtSignal(int, int)
                progress = pyqtSignal(str, int, int)
                
                def __init__(self, plugin, documents, direction):
                    super().__init__()
                    self.plugin = plugin
                    self.documents = documents
                    self.direction = direction
                
                def run(self):
                    success_count = 0
                    total = len(self.documents)
                    
                    for idx, doc in enumerate(self.documents, 1):
                        try:
                            self.progress.emit(
                                self.tr(f"🔄 Синхронизация {idx}/{total}: {doc.get('file_name', 'документ')}"),
                                idx, total
                            )
                            
                            result = self.plugin.sync_data(doc, direction=self.direction)
                            
                            if result.get('success'):
                                success_count += 1
                        except Exception as e:
                            logging.error(f"Ошибка синхронизации документа {idx}: {e}")
                    
                    self.finished.emit(success_count, total)
            
            # Создаем и запускаем worker
            self.batch_worker = BatchSyncWorker(self.paperless_plugin, documents, direction)
            self.batch_worker.progress.connect(
                lambda msg, current, total: self._log_sync(f"{msg} [{current}/{total}]")
            )
            self.batch_worker.finished.connect(self._on_batch_sync_finished)
            self.batch_worker.start()
            
            self._log_sync(self.tr(f"🔄 Начата массовая синхронизация {len(documents)} документов..."))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка массовой синхронизации: {e}"))
            logging.error(f"Ошибка массовой синхронизации: {e}", exc_info=True)
    
    def _on_sync_finished(self, success, message):
        """Обработчик завершения синхронизации"""
        self._log_sync(message)
        if success:
            QMessageBox.information(self, self.tr("Успех"), message)
        else:
            QMessageBox.warning(self, self.tr("Ошибка"), message)
        self._refresh_status()
    
    def _on_batch_sync_finished(self, success_count, total):
        """Обработчик завершения массовой синхронизации"""
        msg = self.tr(f"✅ Синхронизация завершена: {success_count}/{total} успешно")
        self._log_sync(msg)
        QMessageBox.information(self, self.tr("Результат"), msg)
        self._refresh_status()
    
    def _show_advanced_ai_dialog(self):
        """Показывает диалог расширенных функций AI"""
        try:
            from .paperless_ai_manager_dialog import PaperlessAIManagerDialog
            
            dialog = PaperlessAIManagerDialog(self)
            
            # Если есть AI плагин, передаем его
            if self.paperless_ai_plugin:
                dialog.set_ai_plugin(self.paperless_ai_plugin)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка открытия диалога AI: {e}"))
            logging.error(f"Ошибка открытия диалога AI: {e}", exc_info=True)
    
    def _refresh_status(self):
        """Обновляет статус подключения"""
        try:
            config = self._get_current_config()
            
            from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin
            
            plugin = PaperlessNGXPlugin(config)
            
            if plugin.initialize() and plugin.connect():
                status = plugin.get_connection_status()
                
                self.connection_status_label.setText(self.tr("✅ Подключено"))
                
                if status.get("last_sync"):
                    from datetime import datetime
                    last_sync = datetime.fromisoformat(status["last_sync"])
                    self.last_sync_label.setText(last_sync.strftime("%d.%m.%Y %H:%M:%S"))
                
                self.cached_items_label.setText(
                    f"{status.get('cached_correspondents', 0)} / "
                    f"{status.get('cached_document_types', 0)} / "
                    f"{status.get('cached_tags', 0)}"
                )
                
                plugin.cleanup()
            else:
                self.connection_status_label.setText(self.tr("❌ Не подключено"))
                
        except Exception as e:
            logging.error(f"Ошибка обновления статуса: {e}")
            self.connection_status_label.setText(self.tr("❌ Ошибка"))
    
    def _log_sync(self, message: str):
        """Добавляет сообщение в лог синхронизации"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.sync_log.append(f"[{timestamp}] {message}")
    
    def _configure_scheduler(self, config: Dict[str, Any]):
        """Настраивает планировщик автоматической синхронизации"""
        try:
            scheduler = get_scheduler()
            task_id = "paperless_auto_sync"
            
            if config.get("auto_sync", False):
                # Удаляем старую задачу если есть
                scheduler.remove_task(task_id)
                
                # Конвертируем интервал в минуты
                interval_seconds = config.get("sync_interval", 300)
                interval_minutes = max(1, interval_seconds // 60)
                
                # Добавляем новую задачу
                def auto_sync_task():
                    """Задача автоматической синхронизации"""
                    try:
                        if hasattr(self.parent(), 'get_all_processed_documents'):
                            docs = self.parent().get_all_processed_documents()
                            
                            if docs and self.paperless_plugin:
                                logging.info(f"Автосинхронизация: {len(docs)} документов")
                                
                                for doc in docs:
                                    try:
                                        self.paperless_plugin.sync_data(doc, direction="export")
                                    except Exception as e:
                                        logging.error(f"Ошибка синхронизации документа: {e}")
                                
                                logging.info("Автосинхронизация завершена")
                    except Exception as e:
                        logging.error(f"Ошибка автосинхронизации: {e}", exc_info=True)
                
                # Планируем задачу
                success = scheduler.add_task(
                    task_id=task_id,
                    name=self.tr("Автосинхронизация Paperless"),
                    func=auto_sync_task,
                    interval=ScheduleInterval.MINUTES,
                    interval_value=interval_minutes,
                    enabled=True
                )
                
                if success:
                    # Запускаем планировщик если еще не запущен
                    if not scheduler.running:
                        scheduler.start()
                    
                    self._log_sync(
                        self.tr(f"⏰ Автосинхронизация настроена: каждые {interval_minutes} мин")
                    )
                    logging.info(f"Планировщик настроен: {interval_minutes} минут")
                else:
                    logging.error("Не удалось настроить автосинхронизацию")
            else:
                # Отключаем автосинхронизацию
                if scheduler.remove_task(task_id):
                    self._log_sync(self.tr("⏰ Автосинхронизация отключена"))
                    logging.info("Автосинхронизация отключена")
                    
        except Exception as e:
            logging.error(f"Ошибка настройки планировщика: {e}", exc_info=True)
    
    def set_paperless_plugin(self, plugin):
        """Устанавливает плагин Paperless для использования"""
        self.paperless_plugin = plugin
    
    def set_paperless_ai_plugin(self, plugin):
        """Устанавливает плагин Paperless-AI для использования"""
        self.paperless_ai_plugin = plugin


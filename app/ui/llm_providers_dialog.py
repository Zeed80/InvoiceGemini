"""
Диалог для настройки LLM провайдеров и их API ключей.
Поддерживает OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, Ollama.
"""
import os
import json
from typing import Dict, Optional, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QComboBox, QGroupBox, QTabWidget, QWidget,
    QFormLayout, QTextEdit, QMessageBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QScrollArea, QSizePolicy, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QIcon

from ..plugins.base_llm_plugin import BaseLLMPlugin, LLM_PROVIDERS
from ..settings_manager import settings_manager


class LLMProviderTestThread(QThread):
    """Поток для тестирования подключения к LLM провайдеру"""
    test_completed = pyqtSignal(str, bool, str)  # provider_name, success, message
    
    def __init__(self, provider_name: str, model_name: str, api_key: str):
        super().__init__()
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key
    
    def run(self):
        try:
            from ..plugins.models.universal_llm_plugin import UniversalLLMPlugin
            
            # Создаем плагин и тестируем подключение
            plugin = UniversalLLMPlugin(
                provider_name=self.provider_name,
                model_name=self.model_name,
                api_key=self.api_key
            )
            
            success = plugin.load_model()
            if success:
                # Тестируем простой запрос
                try:
                    response = plugin.generate_response("Test connection. Respond with 'OK'.")
                    if response and len(response.strip()) > 0:
                        self.test_completed.emit(self.provider_name, True, "Подключение успешно!")
                    else:
                        self.test_completed.emit(self.provider_name, False, "Пустой ответ от API")
                except Exception as e:
                    self.test_completed.emit(self.provider_name, False, f"Ошибка запроса: {str(e)}")
            else:
                self.test_completed.emit(self.provider_name, False, "Ошибка инициализации клиента")
                
        except Exception as e:
            self.test_completed.emit(self.provider_name, False, f"Ошибка: {str(e)}")


class ModelRefreshThread(QThread):
    """Поток для обновления списка моделей провайдера"""
    refresh_completed = pyqtSignal(str, list, str)  # provider_name, models, error
    
    def __init__(self, provider_name: str, api_key: str):
        super().__init__()
        self.provider_name = provider_name
        self.api_key = api_key
    
    def run(self):
        """Выполняет обновление списка моделей"""
        try:
            # Получаем актуальный список моделей
            models = BaseLLMPlugin.refresh_provider_models(self.provider_name, self.api_key)
            
            if models:
                self.refresh_completed.emit(self.provider_name, models, "")
            else:
                self.refresh_completed.emit(self.provider_name, [], "Не удалось получить список моделей")
                
        except Exception as e:
            self.refresh_completed.emit(self.provider_name, [], str(e))


class LLMProvidersDialog(QDialog):
    """Диалог для настройки LLM провайдеров"""
    
    providers_updated = pyqtSignal()  # Сигнал об обновлении настроек провайдеров
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("🔌 Настройка LLM провайдеров"))
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        # Данные провайдеров
        self.provider_widgets = {}
        self.test_threads = {}
        
        self.init_ui()
        self.load_settings()
    
    def tr(self, text):
        """Простая реализация tr для совместимости"""
        return text
    
    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)
        
        # Заголовок
        title_label = QLabel(self.tr("🔌 Конфигурация LLM провайдеров"))
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Описание
        desc_label = QLabel(self.tr(
            "Настройте API ключи и параметры для различных LLM провайдеров.\n"
            "После настройки вы сможете использовать их для извлечения данных из счетов."
        ))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin: 10px 0;")
        layout.addWidget(desc_label)
        
        # Создаем табы для каждого провайдера
        self.tab_widget = QTabWidget()
        
        for provider_name, config in LLM_PROVIDERS.items():
            tab = self._create_provider_tab(provider_name, config)
            icon_text = self._get_provider_icon(provider_name)
            self.tab_widget.addTab(tab, f"{icon_text} {config.display_name}")
        
        layout.addWidget(self.tab_widget)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.test_all_button = QPushButton(self.tr("🧪 Тестировать все"))
        self.test_all_button.clicked.connect(self.test_all_providers)
        buttons_layout.addWidget(self.test_all_button)
        
        buttons_layout.addStretch()
        
        self.reset_button = QPushButton(self.tr("↻ Сброс"))
        self.reset_button.clicked.connect(self.reset_to_defaults)
        buttons_layout.addWidget(self.reset_button)
        
        self.save_button = QPushButton(self.tr("💾 Сохранить"))
        self.save_button.clicked.connect(self.save_settings)
        self.save_button.setDefault(True)
        buttons_layout.addWidget(self.save_button)
        
        self.close_button = QPushButton(self.tr("❌ Закрыть"))
        self.close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_button)
        
        layout.addLayout(buttons_layout)
    
    def _get_provider_icon(self, provider_name: str) -> str:
        """Возвращает иконку для провайдера"""
        icons = {
            "openai": "🤖",
            "anthropic": "🧠", 
            "google": "🔍",
            "mistral": "🌪️",
            "deepseek": "🔬",
            "xai": "❌",
            "ollama": "🦙"
        }
        return icons.get(provider_name, "🔌")
    
    def _create_provider_tab(self, provider_name: str, config) -> QWidget:
        """Создает вкладку для конкретного провайдера"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Инициализируем словарь для виджетов провайдера в самом начале
        if provider_name not in self.provider_widgets:
            self.provider_widgets[provider_name] = {}
        
        # Scroll area для длинного содержимого
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Основная информация о провайдере
        info_group = QGroupBox(self.tr("ℹ️ Информация о провайдере"))
        info_layout = QFormLayout()
        
        info_layout.addRow(self.tr("Название:"), QLabel(config.display_name))
        info_layout.addRow(self.tr("Требует API ключ:"), 
                          QLabel(self.tr("Да") if config.requires_api_key else self.tr("Нет")))
        
        # Показываем поддержку vision
        vision_support = self.tr("Да") if config.supports_vision else self.tr("Нет (только текст + OCR)")
        info_layout.addRow(self.tr("Поддержка изображений:"), QLabel(vision_support))
        
        info_group.setLayout(info_layout)
        scroll_layout.addWidget(info_group)
        
        # Настройки API
        api_group = QGroupBox(self.tr("🔑 Настройки API"))
        api_layout = QFormLayout()
        
        # API ключ (если требуется)
        if config.requires_api_key:
            api_key_edit = QLineEdit()
            api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            api_key_edit.setPlaceholderText(self.tr("Введите API ключ..."))
            
            # Кнопка показать/скрыть
            show_key_btn = QPushButton(self.tr("👁️"))
            show_key_btn.setMaximumWidth(40)
            show_key_btn.clicked.connect(lambda: self._toggle_password_visibility(api_key_edit))
            
            key_layout = QHBoxLayout()
            key_layout.addWidget(api_key_edit, 1)
            key_layout.addWidget(show_key_btn)
            
            api_layout.addRow(self.tr("API ключ:"), key_layout)
            
            # Сохраняем ссылку на виджет
            self.provider_widgets[provider_name]['api_key_edit'] = api_key_edit
        
        # Выбор модели
        model_layout = QHBoxLayout()
        model_combo = QComboBox()
        model_combo.addItems(config.models)
        if config.default_model in config.models:
            model_combo.setCurrentText(config.default_model)
        model_layout.addWidget(model_combo)
        
        # Кнопка обновления списка моделей
        refresh_models_btn = QPushButton("🔄")
        refresh_models_btn.setToolTip(self.tr("Обновить список доступных моделей"))
        refresh_models_btn.setMaximumWidth(30)
        refresh_models_btn.clicked.connect(lambda: self.refresh_models(provider_name))
        model_layout.addWidget(refresh_models_btn)
        
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        api_layout.addRow(self.tr("Модель:"), model_widget)
        
        # Дополнительные настройки для Ollama
        if provider_name == "ollama":
            base_url_edit = QLineEdit()
            base_url_edit.setText("http://localhost:11434")
            base_url_edit.setPlaceholderText("http://localhost:11434")
            api_layout.addRow(self.tr("URL сервера:"), base_url_edit)
            self.provider_widgets[provider_name]['base_url_edit'] = base_url_edit
        
        api_group.setLayout(api_layout)
        scroll_layout.addWidget(api_group)
        
        # Сохраняем ссылки на виджеты
        self.provider_widgets[provider_name]['model_combo'] = model_combo
        self.provider_widgets[provider_name]['refresh_models_btn'] = refresh_models_btn
        
        # Параметры генерации
        gen_group = QGroupBox(self.tr("⚙️ Параметры генерации"))
        gen_layout = QFormLayout()
        
        # Temperature
        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.1)
        temp_spin.setValue(0.1)
        temp_spin.setDecimals(1)
        gen_layout.addRow(self.tr("Temperature:"), temp_spin)
        
        # Max tokens
        tokens_spin = QSpinBox()
        tokens_spin.setRange(100, 8192)
        tokens_spin.setValue(4096)
        gen_layout.addRow(self.tr("Max tokens:"), tokens_spin)
        
        # Top P
        top_p_spin = QDoubleSpinBox()
        top_p_spin.setRange(0.1, 1.0)
        top_p_spin.setSingleStep(0.1)
        top_p_spin.setValue(0.9)
        top_p_spin.setDecimals(1)
        gen_layout.addRow(self.tr("Top P:"), top_p_spin)
        
        gen_group.setLayout(gen_layout)
        scroll_layout.addWidget(gen_group)
        
        # Сохраняем параметры генерации
        self.provider_widgets[provider_name]['temperature'] = temp_spin
        self.provider_widgets[provider_name]['max_tokens'] = tokens_spin
        self.provider_widgets[provider_name]['top_p'] = top_p_spin
        
        # Кнопки тестирования и статус
        test_group = QGroupBox(self.tr("🧪 Тестирование"))
        test_layout = QVBoxLayout()
        
        test_buttons_layout = QHBoxLayout()
        
        test_btn = QPushButton(self.tr("🧪 Тестировать подключение"))
        test_btn.clicked.connect(lambda: self.test_provider(provider_name))
        test_buttons_layout.addWidget(test_btn)
        
        # Прогресс-бар для тестирования
        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        test_buttons_layout.addWidget(progress_bar)
        
        test_layout.addLayout(test_buttons_layout)
        
        # Статус тестирования
        status_label = QLabel(self.tr("Статус: Не тестировался"))
        status_label.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
        test_layout.addWidget(status_label)
        
        test_group.setLayout(test_layout)
        scroll_layout.addWidget(test_group)
        
        # Сохраняем виджеты тестирования
        self.provider_widgets[provider_name]['test_button'] = test_btn
        self.provider_widgets[provider_name]['progress_bar'] = progress_bar
        self.provider_widgets[provider_name]['status_label'] = status_label
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return tab
    
    def _toggle_password_visibility(self, line_edit: QLineEdit):
        """Переключает видимость пароля"""
        if line_edit.echoMode() == QLineEdit.EchoMode.Password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)
    
    def test_provider(self, provider_name: str):
        """Тестирует подключение к конкретному провайдеру"""
        widgets = self.provider_widgets.get(provider_name, {})
        
        # Получаем настройки
        config = LLM_PROVIDERS[provider_name]
        
        if config.requires_api_key:
            api_key = widgets.get('api_key_edit', QLineEdit()).text().strip()
            if not api_key:
                QMessageBox.warning(self, self.tr("Ошибка"), 
                                  self.tr("Введите API ключ для тестирования"))
                return
        else:
            api_key = None
        
        model_name = widgets.get('model_combo', QComboBox()).currentText()
        
        # Обновляем UI
        test_btn = widgets.get('test_button')
        progress_bar = widgets.get('progress_bar')
        status_label = widgets.get('status_label')
        
        if test_btn:
            test_btn.setEnabled(False)
        if progress_bar:
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)  # Indeterminate progress
        if status_label:
            status_label.setText(self.tr("Статус: Тестирование..."))
            status_label.setStyleSheet("padding: 5px; border: 1px solid #2196F3; border-radius: 3px; color: #2196F3;")
        
        # Запускаем тест в отдельном потоке
        test_thread = LLMProviderTestThread(provider_name, model_name, api_key)
        test_thread.test_completed.connect(self._on_test_completed)
        test_thread.start()
        
        self.test_threads[provider_name] = test_thread
    
    def _on_test_completed(self, provider_name: str, success: bool, message: str):
        """Обработчик завершения тестирования"""
        widgets = self.provider_widgets.get(provider_name, {})
        
        test_btn = widgets.get('test_button')
        progress_bar = widgets.get('progress_bar')
        status_label = widgets.get('status_label')
        
        # Обновляем UI
        if test_btn:
            test_btn.setEnabled(True)
        if progress_bar:
            progress_bar.setVisible(False)
        
        if status_label:
            if success:
                status_label.setText(f"Статус: ✅ {message}")
                status_label.setStyleSheet("padding: 5px; border: 1px solid #4CAF50; border-radius: 3px; color: #4CAF50;")
            else:
                status_label.setText(f"Статус: ❌ {message}")
                status_label.setStyleSheet("padding: 5px; border: 1px solid #F44336; border-radius: 3px; color: #F44336;")
        
        # Очищаем поток
        if provider_name in self.test_threads:
            del self.test_threads[provider_name]
    
    def test_all_providers(self):
        """Тестирует все настроенные провайдеры"""
        for provider_name in LLM_PROVIDERS.keys():
            widgets = self.provider_widgets.get(provider_name, {})
            config = LLM_PROVIDERS[provider_name]
            
            # Проверяем, есть ли API ключ (если требуется)
            if config.requires_api_key:
                api_key = widgets.get('api_key_edit', QLineEdit()).text().strip()
                if api_key:
                    QTimer.singleShot(100 * list(LLM_PROVIDERS.keys()).index(provider_name), 
                                    lambda p=provider_name: self.test_provider(p))
            else:
                # Для Ollama тестируем без API ключа
                QTimer.singleShot(100 * list(LLM_PROVIDERS.keys()).index(provider_name),
                                lambda p=provider_name: self.test_provider(p))
    
    def load_settings(self):
        """Загружает сохраненные настройки"""
        try:
            llm_settings = settings_manager.get_setting('llm_providers', {})
            
            for provider_name, widgets in self.provider_widgets.items():
                provider_settings = llm_settings.get(provider_name, {})
                config = LLM_PROVIDERS[provider_name]
                
                # API ключ
                if config.requires_api_key and 'api_key_edit' in widgets:
                    # Пытаемся загрузить зашифрованный ключ
                    encrypted_key = settings_manager.get_encrypted_setting(f'{provider_name}_api_key')
                    if encrypted_key:
                        widgets['api_key_edit'].setText(encrypted_key)
                
                # Модель
                if 'model_combo' in widgets:
                    saved_model = provider_settings.get('model', config.default_model)
                    if saved_model in config.models:
                        widgets['model_combo'].setCurrentText(saved_model)
                
                # Параметры генерации
                if 'temperature' in widgets:
                    widgets['temperature'].setValue(provider_settings.get('temperature', 0.1))
                if 'max_tokens' in widgets:
                    widgets['max_tokens'].setValue(provider_settings.get('max_tokens', 4096))
                if 'top_p' in widgets:
                    widgets['top_p'].setValue(provider_settings.get('top_p', 0.9))
                
                # Дополнительные настройки для Ollama
                if provider_name == "ollama" and 'base_url_edit' in widgets:
                    widgets['base_url_edit'].setText(provider_settings.get('base_url', 'http://localhost:11434'))
                    
        except Exception as e:
            print(f"Ошибка загрузки настроек LLM: {e}")
    
    def save_settings(self):
        """Сохраняет настройки провайдеров"""
        try:
            llm_settings = {}
            
            for provider_name, widgets in self.provider_widgets.items():
                config = LLM_PROVIDERS[provider_name]
                provider_settings = {}
                
                # API ключ
                if config.requires_api_key and 'api_key_edit' in widgets:
                    api_key = widgets['api_key_edit'].text().strip()
                    if api_key:
                        # Сохраняем зашифрованный ключ
                        settings_manager.save_encrypted_setting(f'{provider_name}_api_key', api_key)
                
                # Модель
                if 'model_combo' in widgets:
                    provider_settings['model'] = widgets['model_combo'].currentText()
                
                # Параметры генерации
                if 'temperature' in widgets:
                    provider_settings['temperature'] = widgets['temperature'].value()
                if 'max_tokens' in widgets:
                    provider_settings['max_tokens'] = widgets['max_tokens'].value()
                if 'top_p' in widgets:
                    provider_settings['top_p'] = widgets['top_p'].value()
                
                # Дополнительные настройки для Ollama
                if provider_name == "ollama" and 'base_url_edit' in widgets:
                    provider_settings['base_url'] = widgets['base_url_edit'].text()
                
                llm_settings[provider_name] = provider_settings
            
            # Сохраняем в settings_manager
            settings_manager.save_setting('llm_providers', llm_settings)
            settings_manager.save_settings()
            
            QMessageBox.information(self, self.tr("Успех"), 
                                  self.tr("Настройки LLM провайдеров сохранены!"))
            
            self.providers_updated.emit()
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                               self.tr(f"Ошибка сохранения настроек: {e}"))
    
    def reset_to_defaults(self):
        """Сбрасывает настройки к значениям по умолчанию"""
        reply = QMessageBox.question(self, self.tr("Подтверждение"),
                                   self.tr("Сбросить все настройки к значениям по умолчанию?"),
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            for provider_name, widgets in self.provider_widgets.items():
                config = LLM_PROVIDERS[provider_name]
                
                # Очищаем API ключ
                if 'api_key_edit' in widgets:
                    widgets['api_key_edit'].clear()
                
                # Сбрасываем модель
                if 'model_combo' in widgets:
                    widgets['model_combo'].setCurrentText(config.default_model)
                
                # Сбрасываем параметры генерации
                if 'temperature' in widgets:
                    widgets['temperature'].setValue(0.1)
                if 'max_tokens' in widgets:
                    widgets['max_tokens'].setValue(4096)
                if 'top_p' in widgets:
                    widgets['top_p'].setValue(0.9)
                
                # Сбрасываем статус
                if 'status_label' in widgets:
                    widgets['status_label'].setText(self.tr("Статус: Не тестировался"))
                    widgets['status_label'].setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
    
    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        """Получает настройки для конкретного провайдера"""
        widgets = self.provider_widgets.get(provider_name, {})
        config = LLM_PROVIDERS[provider_name]
        
        settings = {
            'model': widgets.get('model_combo', QComboBox()).currentText(),
            'temperature': widgets.get('temperature', QDoubleSpinBox()).value(),
            'max_tokens': widgets.get('max_tokens', QSpinBox()).value(),
            'top_p': widgets.get('top_p', QDoubleSpinBox()).value(),
        }
        
        if config.requires_api_key and 'api_key_edit' in widgets:
            settings['api_key'] = widgets['api_key_edit'].text().strip()
        
        if provider_name == "ollama" and 'base_url_edit' in widgets:
            settings['base_url'] = widgets['base_url_edit'].text()
        
        return settings
    
    def refresh_models(self, provider_name: str):
        """Обновляет список доступных моделей для провайдера"""
        widgets = self.provider_widgets.get(provider_name, {})
        config = LLM_PROVIDERS[provider_name]
        
        # Получаем API ключ если требуется
        api_key = None
        if config.requires_api_key:
            api_key_edit = widgets.get('api_key_edit')
            if api_key_edit:
                api_key = api_key_edit.text().strip()
                if not api_key:
                    QMessageBox.warning(self, self.tr("Ошибка"), 
                                      self.tr("Введите API ключ для обновления списка моделей"))
                    return
        
        # Получаем виджеты
        model_combo = widgets.get('model_combo')
        refresh_btn = widgets.get('refresh_models_btn')
        
        if not model_combo:
            return
        
        # Отключаем кнопку и показываем процесс
        if refresh_btn:
            refresh_btn.setEnabled(False)
            refresh_btn.setText("⏳")
        
        # Сохраняем текущую выбранную модель
        current_model = model_combo.currentText()
        
        # Запускаем обновление в отдельном потоке
        refresh_thread = ModelRefreshThread(provider_name, api_key)
        refresh_thread.refresh_completed.connect(
            lambda pn, models, error: self._on_models_refreshed(pn, models, error, current_model)
        )
        refresh_thread.start()
        
        # Сохраняем поток
        self.refresh_threads = getattr(self, 'refresh_threads', {})
        self.refresh_threads[provider_name] = refresh_thread
    
    def _on_models_refreshed(self, provider_name: str, models: list, error: str, previous_model: str):
        """Обработчик завершения обновления моделей"""
        widgets = self.provider_widgets.get(provider_name, {})
        model_combo = widgets.get('model_combo')
        refresh_btn = widgets.get('refresh_models_btn')
        
        # Восстанавливаем кнопку
        if refresh_btn:
            refresh_btn.setEnabled(True)
            refresh_btn.setText("🔄")
        
        if error:
            QMessageBox.warning(self, self.tr("Ошибка"), 
                              self.tr(f"Не удалось обновить список моделей: {error}"))
            return
        
        if not model_combo or not models:
            return
        
        # Обновляем список моделей
        model_combo.clear()
        model_combo.addItems(models)
        
        # Восстанавливаем выбранную модель или выбираем первую
        if previous_model in models:
            model_combo.setCurrentText(previous_model)
        elif models:
            model_combo.setCurrentIndex(0)
        
        # Показываем сообщение об успехе
        QMessageBox.information(self, self.tr("Успех"), 
                              self.tr(f"Список моделей обновлен! Найдено {len(models)} моделей."))
        
        # Очищаем поток
        if hasattr(self, 'refresh_threads') and provider_name in self.refresh_threads:
            del self.refresh_threads[provider_name]
    
    def closeEvent(self, event):
        """Обработчик закрытия диалога"""
        # Останавливаем все активные тесты
        for thread in self.test_threads.values():
            if thread.isRunning():
                thread.terminate()
                thread.wait(1000)
        
        # Останавливаем потоки обновления моделей
        if hasattr(self, 'refresh_threads'):
            for thread in self.refresh_threads.values():
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(1000)
        
        event.accept() 
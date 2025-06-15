"""
Модуль содержит диалоги настроек для приложения.
"""
import os
import sys # NEW: Добавлено для sys.executable
import json
import base64
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QCheckBox, QTabWidget, QWidget, QProgressBar,
    QComboBox, QMessageBox, QSpinBox, QTextEdit, QDoubleSpinBox, QSizePolicy, QScrollArea, 
    QPlainTextEdit, QApplication, QFormLayout # NEW: Добавляем QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject # NEW: QObject для Worker

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from . import config as app_config # Используем относительный импорт
from . import utils
from .settings_manager import settings_manager
from .processing_engine import ModelManager # Должен быть импортирован для доступа к parent_window.model_manager
# from .threads import ModelDownloadThread # ModelDownloadThread будет переопределен или адаптирован ниже
from huggingface_hub import hf_hub_download, HfApi, list_models # NEW: Для проверки HF токена
from huggingface_hub import scan_cache_dir

# Импортируем систему безопасности
try:
    from config.secrets import SecretsManager
    SECRETS_MANAGER_AVAILABLE = True
except ImportError:
    SECRETS_MANAGER_AVAILABLE = False
    print("Предупреждение: SecretsManager недоступен. Функции безопасности ограничены.")


# NEW: Worker для выполнения задач в фоновом потоке (например, скачивание)
class Worker(QObject):
    finished = pyqtSignal(bool, str) # success, message_or_model_path
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, task_callable, *args, **kwargs):
        super().__init__()
        self.task_callable = task_callable
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Передаем progress_callback, если он ожидается функцией
            if "progress_callback" in self.task_callable.__code__.co_varnames:
                self.kwargs["progress_callback"] = lambda p: self.progress.emit(p)
            
            result = self.task_callable(*self.args, **self.kwargs)
            
            if isinstance(result, tuple) and len(result) == 2:
                success, message = result
                self.finished.emit(success, message)
            elif isinstance(result, bool): # Если функция возвращает только success
                self.finished.emit(result, "Операция завершена.")
            else: # Если функция возвращает что-то другое (например, путь)
                self.finished.emit(True, str(result) if result is not None else "")

        except Exception as e:
            import traceback
            error_msg = f"Ошибка в фоновой задаче: {e}\n{traceback.format_exc()}"
            print(error_msg) # Для отладки в консоли
            self.error.emit(error_msg)


class ModelManagementDialog(QDialog):
    """Диалог управления моделями и основными настройками."""
    
    geminiModelsUpdated = pyqtSignal() # Сигнал об обновлении списка моделей Gemini

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent # Сохраняем ссылку на родительское окно (MainWindow)
        self.setWindowTitle("Настройки и управление моделями")
        self.setMinimumSize(700, 600) # Увеличенный размер для удобства
        
        self.current_worker = None
        self.current_thread = None
        
        # Инициализируем менеджер секретов
        if SECRETS_MANAGER_AVAILABLE:
            self.secrets_manager = SecretsManager()
        else:
            self.secrets_manager = None

        self.init_ui()
        self._setup_text_change_handlers() # Подключаем обработчики изменения текста
        
        # Загружаем настройки и обновляем UI
        self.load_settings()
        
        # Принудительно синхронизируем настройки с app_config
        self._sync_settings_with_config()
        
        # Обновляем статусы моделей
        self.check_models_availability()

    def _sync_settings_with_config(self):
        """Синхронизирует текущие настройки с глобальными переменными в app_config."""
        # Gemini settings
        if hasattr(self, 'gemini_temperature_spinner'):
            app_config.DEFAULT_GEMINI_TEMPERATURE = self.gemini_temperature_spinner.value()
        if hasattr(self, 'gemini_max_tokens_spinner'):
            app_config.DEFAULT_GEMINI_MAX_TOKENS = self.gemini_max_tokens_spinner.value()
        if hasattr(self, 'gemini_pdf_dpi_spinner'):
            app_config.GEMINI_PDF_DPI = self.gemini_pdf_dpi_spinner.value()
        
        # Model paths
        if hasattr(self, 'layoutlm_model_id_edit'):
            app_config.LAYOUTLM_MODEL_ID = self.layoutlm_model_id_edit.text()
        if hasattr(self, 'donut_model_id_edit'):
            app_config.DONUT_MODEL_ID = self.donut_model_id_edit.text()
        
        # Training settings
        if hasattr(self, 'layoutlm_base_model_edit'):
            app_config.LAYOUTLM_MODEL_ID_FOR_TRAINING = self.layoutlm_base_model_edit.text()
        if hasattr(self, 'epochs_spinbox'):
            app_config.DEFAULT_TRAIN_EPOCHS = self.epochs_spinbox.value()
        if hasattr(self, 'batch_size_spinbox'):
            app_config.DEFAULT_TRAIN_BATCH_SIZE = self.batch_size_spinbox.value()
        if hasattr(self, 'learning_rate_dspinbox'):
            app_config.DEFAULT_LEARNING_RATE = self.learning_rate_dspinbox.value()
        
        # Paths
        if hasattr(self, 'tesseract_path_edit') and hasattr(self.tesseract_path_edit, 'line_edit'):
            app_config.TESSERACT_PATH = self.tesseract_path_edit.line_edit.text()
        if hasattr(self, 'poppler_path_edit') and hasattr(self.poppler_path_edit, 'line_edit'):
            app_config.POPPLER_PATH = self.poppler_path_edit.line_edit.text()
        if hasattr(self, 'training_datasets_path_edit') and hasattr(self.training_datasets_path_edit, 'line_edit'):
            app_config.TRAINING_DATASETS_PATH = self.training_datasets_path_edit.line_edit.text()
        if hasattr(self, 'trained_models_path_edit') and hasattr(self.trained_models_path_edit, 'line_edit'):
            app_config.TRAINED_MODELS_PATH = self.trained_models_path_edit.line_edit.text()
        
        # Network settings
        if hasattr(self, 'offline_mode_checkbox'):
            app_config.OFFLINE_MODE = self.offline_mode_checkbox.isChecked()
        if hasattr(self, 'http_timeout_spinbox'):
            app_config.HTTP_TIMEOUT = self.http_timeout_spinbox.value()

    def _log(self, message):
        """Простой логгер для отладки."""
        print(f"ModelManagementDialog: {message}")

    def _setup_text_change_handlers(self):
        """Настраивает обработчики изменения текста в полях ввода."""
        # LayoutLM
        if hasattr(self, 'layoutlm_model_id_edit'):
            self.layoutlm_model_id_edit.textChanged.connect(lambda: self.check_models_availability())
        if hasattr(self, 'custom_layoutlm_name_edit'):
            self.custom_layoutlm_name_edit.textChanged.connect(lambda: self.check_models_availability())
            
        # Donut
        if hasattr(self, 'donut_model_id_edit'):
            self.donut_model_id_edit.textChanged.connect(lambda: self.check_models_availability())
            
        # Gemini
        # Удалено: обработчик для gemini_api_key_edit (поле удалено с вкладки "Управление моделями")

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()

        # Вкладка 0: API Ключи и секреты (первая и главная вкладка)
        self._create_api_keys_tab()

        # Вкладка 1: Облачные модели (новая объединённая вкладка)
        self._create_cloud_models_tab()

        # Вкладка 2: Локальные модели (новая объединённая вкладка)  
        self._create_local_models_tab()

        # Вкладка 3: Общие настройки (пути, языки, параметры)
        self._create_general_settings_tab()

        main_layout.addWidget(self.tab_widget)

        # Кнопки диалога
        buttons_layout = QHBoxLayout()
        
        self.test_all_button = QPushButton("🧪 Тестировать все подключения")
        self.test_all_button.clicked.connect(self._test_all_connections)
        buttons_layout.addWidget(self.test_all_button)
        
        buttons_layout.addStretch()
        
        self.save_button = QPushButton("💾 Сохранить настройки")
        self.save_button.clicked.connect(self.save_all_settings)
        self.save_button.setDefault(True)
        buttons_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("❌ Отмена")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(buttons_layout)

    def _create_path_setting(self, layout, label_text, settings_key_part=None):
        """Вспомогательный метод для создания поля выбора пути."""
        path_layout = QHBoxLayout()
        path_label = QLabel(label_text)
        path_edit = QLineEdit()
        browse_button = QPushButton("Обзор...")
        # Если settings_key_part не None, значит это путь к директории, иначе - файл
        if settings_key_part and "tesseract" not in settings_key_part and "poppler" not in settings_key_part : 
            browse_button.clicked.connect(lambda checked, le=path_edit, sk=settings_key_part: self._select_directory_for_setting(le, sk))
        elif settings_key_part and ("tesseract" in settings_key_part or "poppler" in settings_key_part):
             # Для tesseract нужен файл, для poppler - папка bin
            if "tesseract" in settings_key_part:
                browse_button.clicked.connect(lambda checked, le=path_edit, sk=settings_key_part: self._select_executable_for_setting(le, sk))
            else: # poppler
                browse_button.clicked.connect(lambda checked, le=path_edit, sk=settings_key_part: self._select_directory_for_setting(le, sk))
        else: # Файл, но без settings_key_part (если понадобится)
            browse_button.clicked.connect(lambda checked, le=path_edit: self._select_file_for_setting(le))
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(path_edit, 1)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)
        return path_edit

    def _create_prompt_setting(self, layout, label_text):
        """Вспомогательный метод для создания поля редактирования промпта."""
        prompt_edit = QTextEdit()
        prompt_edit.setAcceptRichText(False)
        prompt_edit.setFixedHeight(100) # Ограничиваем высоту
        self._add_widget_with_label(layout, label_text, prompt_edit)
        return prompt_edit

    def _add_widget_with_label(self, layout, label_text, widget, stretch_factor=0):
        """Добавляет виджет с меткой в указанный layout."""
        if isinstance(layout, QFormLayout):
            layout.addRow(label_text, widget)
        else:
            label = QLabel(label_text)
            layout.addWidget(label)
            if stretch_factor > 0:
                layout.addWidget(widget, stretch_factor)
            else:
                layout.addWidget(widget)
                
    def _select_directory_for_setting(self, line_edit_widget, settings_key_part):
        current_path = line_edit_widget.text() or os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(self, f"Выберите папку для {settings_key_part}", current_path)
        if dir_path:
            line_edit_widget.setText(dir_path)

    def _select_executable_for_setting(self, line_edit_widget, settings_key_part):
        current_path = line_edit_widget.text() or os.path.expanduser("~")
        # Фильтр для tesseract.exe
        file_filter = "Исполняемые файлы (*.exe)" if sys.platform == "win32" else "Все файлы (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, f"Выберите {settings_key_part}", current_path, file_filter)
        if file_path:
            line_edit_widget.setText(file_path)

    def _select_file_for_setting(self, line_edit_widget):
        current_path = line_edit_widget.text() or os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", current_path)
        if file_path:
            line_edit_widget.setText(file_path)

    def _log_callback_for_task(self, message):
        # Этот метод можно будет использовать для вывода логов в какой-нибудь QTextEdit в диалоге
        # Например, если добавить секцию логов на одну из вкладок.
        self._log(f"TASK_LOG: {message}")
        # self.log_display_widget.appendPlainText(message) # Если есть такой виджет

    def _execute_task_in_thread(self, task_callable, on_finished_slot, on_error_slot, on_progress_slot=None, *args, **kwargs):
        if self.current_thread and self.current_thread.isRunning():
            QMessageBox.warning(self, "Операция выполняется", 
                                "Пожалуйста, дождитесь завершения текущей операции.")
            return

        self.current_worker = Worker(task_callable, *args, **kwargs)
        self.current_thread = QThread(self)
        self.current_worker.moveToThread(self.current_thread)

        self.current_worker.finished.connect(on_finished_slot)
        self.current_worker.error.connect(on_error_slot)
        if on_progress_slot:
            self.current_worker.progress.connect(on_progress_slot)
        
        # Очистка после завершения потока
        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.current_thread.finished.connect(self.current_thread.deleteLater)
        self.current_thread.finished.connect(self._reset_current_task_state)
        self.current_worker.error.connect(self._reset_current_task_state) # Также сбрасываем при ошибке

        self.current_thread.start()

    def _reset_current_task_state(self):
        self.current_thread = None
        self.current_worker = None
        # Обновить UI кнопок и прогресс-баров, если нужно
        # Например, сделать кнопки снова активными
        if hasattr(self, 'download_layoutlm_button'): self.download_layoutlm_button.setEnabled(True)
        if hasattr(self, 'download_donut_button'): self.download_donut_button.setEnabled(True)
        # Удалено: test_gemini_key_button и test_hf_token_button (кнопки удалены с соответствующих вкладок)
        if hasattr(self, 'update_gemini_list_button'): self.update_gemini_list_button.setEnabled(True)

        if hasattr(self, 'layoutlm_progress'): self.layoutlm_progress.setVisible(False)
        if hasattr(self, 'donut_progress'): self.donut_progress.setVisible(False)
        # и т.д. для других прогресс-баров, если они есть
        
    # Обработчики загрузки моделей
    def _on_layoutlm_load_success(self, success, model_path):
        """Обработка успешной загрузки модели LayoutLM."""
        self._reset_current_task_state()
        
        if success:
            # Сохраняем статус модели и обновляем UI
            settings_manager.set_value("ModelsStatus", "layoutlm_loaded", True)
            
            # Определяем, какая модель была загружена (кастомная или HF)
            selected_layoutlm_type = self.layoutlm_model_type_combo.currentData()
            is_custom = (selected_layoutlm_type == "custom")
            
            # Сохраняем информацию о типе модели
            settings_manager.set_value("Models", "layoutlm_is_custom", str(is_custom))
            
            # Если кастомная, сохраняем путь
            if is_custom:
                custom_name = os.path.basename(model_path)
                settings_manager.set_value("Models", "layoutlm_custom_path", custom_name)
            else:
                # Для HF сохраняем ID из поля ввода (не перезаписывая само поле)
                current_model_id = self.layoutlm_model_id_edit.text().strip()
                settings_manager.set_value("Models", "layoutlm_id", current_model_id)
                
            # Обновляем UI (статус и кнопку, но не поле ввода ID)
            self._update_layoutlm_status_label_and_button(model_path if is_custom else current_model_id, is_custom)
            
            QMessageBox.information(
                self, 
                "Модель загружена", 
                f"Модель LayoutLM успешно загружена: {os.path.basename(model_path)}"
            )
        else:
            # Обработка неуспешной загрузки (хотя странно, что _on_layoutlm_load_success вызывается с success=False)
            QMessageBox.warning(
                self, 
                "Ошибка загрузки", 
                f"Не удалось загрузить модель LayoutLM: {model_path}"
            )
            
    def _on_layoutlm_load_error(self, error_message):
        """Обработка ошибки загрузки модели LayoutLM."""
        self._reset_current_task_state()
        
        # Определяем текущий ID модели для обновления UI
        selected_layoutlm_type = self.layoutlm_model_type_combo.currentData()
        is_custom = (selected_layoutlm_type == "custom")
        model_id_or_path = ""
        
        if is_custom:
            model_id_or_path = self.custom_layoutlm_name_edit.text().strip()
            # Полный путь для кастомной модели
            if model_id_or_path:
                model_id_or_path = os.path.join(
                    settings_manager.get_string("Paths", "trained_models_path", app_config.TRAINED_MODELS_PATH),
                    model_id_or_path
                )
        else:
            model_id_or_path = self.layoutlm_model_id_edit.text().strip()
            
        # Обновляем UI с учетом ошибки
        self.update_model_status_label("layoutlm", f"Ошибка загрузки")
        self._update_layoutlm_status_label_and_button(model_id_or_path, is_custom)
        
        QMessageBox.critical(
            self, 
            "Ошибка загрузки", 
            f"Не удалось загрузить модель LayoutLM: {error_message}"
        )

    # --- LayoutLM Specific Methods --- 
    def _on_layoutlm_model_type_changed(self, index):
        self._update_layoutlm_section_visibility()
        self.check_models_availability() # Обновляем статус кнопки/метки и инфо-лейбла

    def _update_layoutlm_section_visibility(self):
        selected_type = self.layoutlm_model_type_combo.currentData()
        is_hf = selected_type == "huggingface"
        is_custom = selected_type == "custom"

        self.hf_layoutlm_group.setVisible(is_hf)
        self.custom_layoutlm_group.setVisible(is_custom)
        # Обновление кнопки и токена также перенесено в _update_layoutlm_status_label_and_button
        # и вызывается из check_models_availability

    def _populate_custom_layoutlm_models_combo(self):
        self.custom_layoutlm_model_selector.clear()
        trained_models_path = settings_manager.get_string("Paths", "trained_models_path", app_config.TRAINED_MODELS_PATH)
        
        if not os.path.isdir(trained_models_path):
            self.custom_layoutlm_model_selector.addItem("Папка с моделями не найдена")
            self.custom_layoutlm_model_selector.setEnabled(False)
            return

        found_models = []
        try:
            for item_name in os.listdir(trained_models_path):
                item_path = os.path.join(trained_models_path, item_name)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    found_models.append(item_name)
        except FileNotFoundError:
            self.custom_layoutlm_model_selector.addItem("Ошибка доступа к папке моделей")
            self.custom_layoutlm_model_selector.setEnabled(False)
            return
        
        if found_models:
            self.custom_layoutlm_model_selector.addItems(found_models)
            self.custom_layoutlm_model_selector.setEnabled(True)
            saved_custom_name = settings_manager.get_custom_layoutlm_model_name()
            if saved_custom_name in found_models:
                self.custom_layoutlm_model_selector.setCurrentText(saved_custom_name)
                # self.custom_layoutlm_name_edit.setText(saved_custom_name) # Это сделает _on_custom_layoutlm_selected_from_combo
        else:
            self.custom_layoutlm_model_selector.addItem("Локальные модели не найдены")
            self.custom_layoutlm_model_selector.setEnabled(False)

    def _on_custom_layoutlm_selected_from_combo(self, index_or_text_arg):
        """
        Обработчик выбора локальной модели LayoutLM из выпадающего списка.
        Обновляет поле ввода имени кастомной модели.
        
        Args:
            index_or_text_arg: Индекс выбранного элемента или текст элемента (зависит от сигнала)
        """
        # QComboBox может передавать индекс или текст в зависимости от версии/ситуации
        selected_model_name = ""
        if isinstance(index_or_text_arg, int):
            if index_or_text_arg >= 0: # Убедимся, что индекс валидный
                 selected_model_name = self.custom_layoutlm_model_selector.itemText(index_or_text_arg)
        elif isinstance(index_or_text_arg, str):
            selected_model_name = index_or_text_arg
        
        # Проверяем, что выбранное имя не является плейсхолдером
        if selected_model_name and "не найдены" not in selected_model_name and "Ошибка доступа" not in selected_model_name:
            # Обновляем поле ввода имени модели
            self.custom_layoutlm_name_edit.setText(selected_model_name)
            
            # Обновляем статус и доступность кнопки загрузки
            self.check_models_availability()

    def _update_donut_status_and_button(self):
        model_id = self.donut_model_id_edit.text().strip()
        is_actually_loaded = False
        status_text = "Неизвестно"
        button_text = "Действие с моделью"
        button_enabled = False

        if not model_id:
            status_text = "ID модели не указан"
        elif not (self.parent_window and hasattr(self.parent_window, 'model_manager')):
            status_text = "Менеджер моделей недоступен"
        else:
            button_enabled = True
            manager = self.parent_window.model_manager
            # Construct a cache key that is consistent with how ModelManager might store it
            donut_cache_key = f"donut_{model_id.replace(os.sep, '_')}" 
            
            if donut_cache_key in manager.models:
                cached_processor = manager.models[donut_cache_key]
                # Check if the cached processor is for the current model_id and is loaded
                if hasattr(cached_processor, 'model_id') and cached_processor.model_id == model_id and \
                   hasattr(cached_processor, 'is_loaded') and cached_processor.is_loaded:
                    is_actually_loaded = True
                    status_text = f"Загружена"
                elif hasattr(cached_processor, 'model_id') and cached_processor.model_id == model_id:
                    # It was for this ID, but not loaded (e.g., previous error)
                    status_text = "Ошибка предыдущей загрузки" 
                else:
                    # A processor exists for this key, but it's for a different model_id
                    status_text = "Готова к загрузке" 
            else:
                # Not in active session (manager.models), check if files are in HF cache
                # This is a simplified check; DonutProcessorImpl.load_model has more robust HF cache handling.
                actual_donut_cache_dir = os.path.join(app_config.MODELS_PATH, 'donut', model_id.replace("/", "_"))
                if os.path.isdir(actual_donut_cache_dir) and os.path.exists(os.path.join(actual_donut_cache_dir, "config.json")):
                    status_text = "В кэше HF (не в сессии)"
                else:
                    status_text = "Не загружена (нет в кэше HF)"
        
        self.donut_status_label.setText(status_text)
        if is_actually_loaded:
            self.donut_status_label.setStyleSheet("color: green;")
            button_text = f"Обновить ({model_id.split('/')[-1]})"
        else:
            if "Ошибка" in status_text or "Не загружена" in status_text or "не указан" in status_text:
                self.donut_status_label.setStyleSheet("color: red;")
            else: # "Готова к загрузке", "В кэше HF"
                self.donut_status_label.setStyleSheet("color: orange;")
            button_text = f"Скачать ({model_id.split('/')[-1]})"
        
        self.download_donut_button.setText(button_text)
        self.download_donut_button.setEnabled(button_enabled)

    def check_models_availability(self):
        """Перенаправление к основному методу check_models_availability выше."""
        # Оставляем для совместимости, но вызываем основной метод
        # Этот метод появился из-за дублирования кода
        self.check_models_availability = lambda: None # Избегаем рекурсии
        self.check_gemini_availability()
        # Восстанавливаем метод для следующих вызовов
        self.check_models_availability = self.__class__.check_models_availability.__get__(self, self.__class__)
        
    def check_gemini_availability(self):
        """Проверяет доступность API Gemini и обновляет статус."""
        if not hasattr(self, 'gemini_key_status_label'):
            return
            
        if not GENAI_AVAILABLE:
            self.gemini_key_status_label.setText("Статус: Библиотека не установлена")
            self.gemini_key_status_label.setStyleSheet("color: orange;")
            return
        
        # Проверяем наличие API ключа
        api_key = settings_manager.get_gemini_api_key()
        
        if not api_key:
            self.gemini_key_status_label.setText("Статус ключа: Не указан")
            self.gemini_key_status_label.setStyleSheet("color: red;")
        else:
            self.gemini_key_status_label.setText("Статус ключа: Сохранён")
            self.gemini_key_status_label.setStyleSheet("color: green;")

    def load_settings(self):
        """Загружает настройки и применяет их к элементам интерфейса."""
        # LayoutLM модель
        if hasattr(self, 'layoutlm_model_id_edit'):
            model_id = settings_manager.get_string('Models', 'layoutlm_id', app_config.LAYOUTLM_MODEL_ID)
            self.layoutlm_model_id_edit.setText(model_id)
            
        # Тип модели LayoutLM
        if hasattr(self, 'layoutlm_model_type_combo'):
            model_type = settings_manager.get_string('Models', 'layoutlm_type', 'huggingface')
            index = 0 if model_type == 'huggingface' else 1
            self.layoutlm_model_type_combo.setCurrentIndex(index)
            self._update_layoutlm_section_visibility()
            
        # Кастомная LayoutLM модель
        if hasattr(self, 'custom_layoutlm_name_edit'):
            custom_name = settings_manager.get_string('Models', 'layoutlm_custom_name', '')
            self.custom_layoutlm_name_edit.setText(custom_name)
            self._populate_custom_layoutlm_models_combo()

        # Donut модель
        if hasattr(self, 'donut_model_id_edit'):
            model_id = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
            self.donut_model_id_edit.setText(model_id)

        # Gemini настройки
        if hasattr(self, 'gemini_model_selector'):
            self.populate_gemini_models_from_settings()
            
        if hasattr(self, 'gemini_temperature_spinner'):
            temp = settings_manager.get_float('Gemini', 'temperature', app_config.DEFAULT_GEMINI_TEMPERATURE)
            self.gemini_temperature_spinner.setValue(temp)
            
        if hasattr(self, 'gemini_max_tokens_spinner'):
            tokens = settings_manager.get_int('Gemini', 'max_tokens', app_config.DEFAULT_GEMINI_MAX_TOKENS)
            self.gemini_max_tokens_spinner.setValue(tokens)
            
        if hasattr(self, 'gemini_pdf_dpi_spinner'):
            dpi = settings_manager.get_int('Gemini', 'pdf_dpi', app_config.GEMINI_PDF_DPI)
            self.gemini_pdf_dpi_spinner.setValue(dpi)

        # Пути к инструментам
        if hasattr(self, 'tesseract_path_edit'):
            path = settings_manager.get_string('Paths', 'tesseract_path', app_config.TESSERACT_PATH or '')
            self.tesseract_path_edit.line_edit.setText(path)
            
        if hasattr(self, 'poppler_path_edit'):
            path = settings_manager.get_string('Paths', 'poppler_path', app_config.POPPLER_PATH or '')
            self.poppler_path_edit.line_edit.setText(path)
            
        if hasattr(self, 'training_datasets_path_edit'):
            path = settings_manager.get_string('Paths', 'training_datasets_path', app_config.TRAINING_DATASETS_PATH or '')
            self.training_datasets_path_edit.line_edit.setText(path)
            
        if hasattr(self, 'trained_models_path_edit'):
            path = settings_manager.get_string('Paths', 'trained_models_path', app_config.TRAINED_MODELS_PATH or '')
            self.trained_models_path_edit.line_edit.setText(path)

        # Промпты
        if hasattr(self, 'layoutlm_prompt_edit'):
            prompt = settings_manager.get_model_prompt('layoutlm')
            self.layoutlm_prompt_edit.setPlainText(prompt)
            
        if hasattr(self, 'donut_prompt_edit'):
            prompt = settings_manager.get_model_prompt('donut')
            self.donut_prompt_edit.setPlainText(prompt)
            
        if hasattr(self, 'gemini_prompt_edit'):
            prompt = settings_manager.get_model_prompt('gemini')
            self.gemini_prompt_edit.setPlainText(prompt)

        # Параметры обучения
        if hasattr(self, 'layoutlm_base_model_edit'):
            base_model = settings_manager.get_string('Training', 'layoutlm_base_model', app_config.LAYOUTLM_MODEL_ID_FOR_TRAINING)
            self.layoutlm_base_model_edit.setText(base_model)
            
        if hasattr(self, 'epochs_spinbox'):
            epochs = settings_manager.get_int('Training', 'epochs', app_config.DEFAULT_TRAIN_EPOCHS)
            self.epochs_spinbox.setValue(epochs)
            
        if hasattr(self, 'batch_size_spinbox'):
            batch_size = settings_manager.get_int('Training', 'batch_size', app_config.DEFAULT_TRAIN_BATCH_SIZE)
            self.batch_size_spinbox.setValue(batch_size)
            
        if hasattr(self, 'learning_rate_dspinbox'):
            lr = settings_manager.get_float('Training', 'learning_rate', app_config.DEFAULT_LEARNING_RATE)
            self.learning_rate_dspinbox.setValue(lr)

        # Общие параметры
        if hasattr(self, 'batch_delay_spinner'):
            delay = settings_manager.get_int('Misc', 'batch_processing_delay', app_config.DEFAULT_BATCH_PROCESSING_DELAY)
            self.batch_delay_spinner.setValue(delay)
            
        if hasattr(self, 'vat_rate_spinner'):
            vat_rate = settings_manager.get_default_vat_rate()
            self.vat_rate_spinner.setValue(vat_rate)

        # Название компании-получателя
        if hasattr(self, 'company_receiver_name_edit'):
            company_name = settings_manager.get_company_receiver_name()
            self.company_receiver_name_edit.setText(company_name)

        # Сетевые настройки
        if hasattr(self, 'offline_mode_checkbox'):
            offline_mode = settings_manager.get_bool('Network', 'offline_mode', app_config.OFFLINE_MODE)
            self.offline_mode_checkbox.setChecked(offline_mode)
            
        if hasattr(self, 'http_timeout_spinbox'):
            timeout = settings_manager.get_int('Network', 'http_timeout', app_config.HTTP_TIMEOUT)
            self.http_timeout_spinbox.setValue(timeout)

        # Проверяем доступность моделей
        self.check_models_availability()
        
        # Загружаем значения из секретов
        self._load_secrets_values()
        
        print("Настройки загружены для новой структуры интерфейса")

    def populate_gemini_models_from_settings(self):
        """Заполняет выпадающий список доступных моделей Gemini из сохраненных настроек."""
        if not hasattr(self, 'gemini_model_selector'):
            return
            
        self.gemini_model_selector.clear()
        
        # Попытка получить сохраненный список моделей из настроек
        saved_models_json = settings_manager.get_string('Gemini', 'available_models_json', "[]")
        try:
            saved_models = json.loads(saved_models_json)
            if saved_models and isinstance(saved_models, list):
                self.gemini_model_selector.addItems(saved_models)
                # Выбираем текущую модель, если она есть в списке
                current_model = settings_manager.get_string('Gemini', 'sub_model_id', app_config.GEMINI_MODEL_ID)
                index = self.gemini_model_selector.findText(current_model)
                if index >= 0:
                    self.gemini_model_selector.setCurrentIndex(index)
                return
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Ошибка при загрузке списка моделей Gemini из настроек: {e}")
            
        # Если не удалось загрузить из настроек, используем дефолтный список
        default_models = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-pro"
        ]
        self.gemini_model_selector.addItems(default_models)
        # Выбираем текущую модель, если она есть в списке
        current_model = settings_manager.get_string('Gemini', 'sub_model_id', app_config.GEMINI_MODEL_ID)
        index = self.gemini_model_selector.findText(current_model)
        if index >= 0:
            self.gemini_model_selector.setCurrentIndex(index)
        else:
            # Если текущей модели нет в списке, добавляем её
            self.gemini_model_selector.addItem(current_model)
            self.gemini_model_selector.setCurrentText(current_model)

    def save_settings(self):
        """Сохраняет настройки из интерфейса."""
        # Сохраняем настройки Gemini API (только модель, ключ настраивается на вкладке "🔐 API Ключи")
        if hasattr(self, 'gemini_model_selector'):
            self._on_gemini_sub_model_changed(self.gemini_model_selector.currentIndex())
            
        # Сохраняем параметры генерации Gemini
        if hasattr(self, 'gemini_temperature_spinner') and hasattr(self, 'gemini_max_tokens_spinner'):
            self.save_gemini_generation_parameters_action()
            
        # Сохраняем ставку НДС
        if hasattr(self, 'vat_rate_spinner'):
            settings_manager.set_default_vat_rate(self.vat_rate_spinner.value())
            
        # Сохраняем настройки моделей
        # LayoutLM
        layoutlm_is_custom = settings_manager.get_bool('Models', 'layoutlm_is_custom', False)
        if hasattr(self, 'layoutlm_model_type_combo'):
            self.layoutlm_model_type_combo.setCurrentIndex(1 if layoutlm_is_custom else 0)
            
        if hasattr(self, 'layoutlm_model_id_edit'):
            current_model_id = settings_manager.get_string('Models', 'layoutlm_id', app_config.LAYOUTLM_MODEL_ID)
            self.layoutlm_model_id_edit.setText(current_model_id)
            
        if hasattr(self, 'custom_layoutlm_name_edit'):
            self.custom_layoutlm_name_edit.setText(settings_manager.get_string('Models', 'layoutlm_custom_path', ""))
            
        if hasattr(self, 'donut_model_id_edit'):
            current_model_id = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
            self.donut_model_id_edit.setText(current_model_id)

        # Загрузка настроек HF токена
        if hasattr(self, 'hf_token_edit'):
            self.hf_token_edit.setText(settings_manager.get_huggingface_token())
            
        # Загрузка настроек тесераккта и попплера
        if hasattr(self, 'tesseract_path_edit') and hasattr(self.tesseract_path_edit, 'line_edit'):
            self.tesseract_path_edit.line_edit.setText(settings_manager.get_string('Paths', 'tesseract_path', app_config.TESSERACT_PATH or ''))
            
        if hasattr(self, 'poppler_path_edit') and hasattr(self.poppler_path_edit, 'line_edit'):
            self.poppler_path_edit.line_edit.setText(settings_manager.get_string('Paths', 'poppler_path', app_config.POPPLER_PATH or ''))
            
        # Сохраняем настройки всех промптов
        self.save_model_prompt('layoutlm', called_from_save_settings=True)
        self.save_model_prompt('donut', called_from_save_settings=True)
        self.save_model_prompt('gemini', called_from_save_settings=True)

        # NEW: Сохранение настроек обучения
        self.save_training_parameters_action(called_from_save_settings=True)

        # NEW: Сохранение настроек обучения, если элементы существуют
        if hasattr(self, 'training_datasets_path_edit') and hasattr(self.training_datasets_path_edit, 'line_edit'):
            settings_manager.set_value('Paths', 'training_datasets_path', self.training_datasets_path_edit.line_edit.text())
            app_config.TRAINING_DATASETS_PATH = self.training_datasets_path_edit.line_edit.text()
            
        if hasattr(self, 'trained_models_path_edit') and hasattr(self.trained_models_path_edit, 'line_edit'):
            settings_manager.set_value('Paths', 'trained_models_path', self.trained_models_path_edit.line_edit.text())
            app_config.TRAINED_MODELS_PATH = self.trained_models_path_edit.line_edit.text()
            
        if hasattr(self, 'gemini_annotation_prompt_edit'):
            # Кодируем текст промпта в Base64 для безопасного сохранения в конфиг-файле
            try:
                prompt_text = self.gemini_annotation_prompt_edit.toPlainText()
                prompt_encoded = base64.b64encode(prompt_text.encode('utf-8')).decode('ascii')
                settings_manager.set_value('Training', 'gemini_annotation_prompt_encoded', prompt_encoded)
                app_config.GEMINI_ANNOTATION_PROMPT_DEFAULT = prompt_text
            except Exception as e:
                print(f"Ошибка при сохранении промпта аннотации: {e}")

        settings_manager.save_settings()
        QMessageBox.information(self, "Настройки сохранены", "Настройки успешно сохранены.")
        print("DEBUG: save_settings FINISHED in ModelManagementDialog")
        
        # Обновляем конфигурацию в реальном времени, если это необходимо
        if hasattr(self, 'tesseract_path_edit') and hasattr(self.tesseract_path_edit, 'line_edit'):
            app_config.TESSERACT_PATH = self.tesseract_path_edit.line_edit.text()
            
        if hasattr(self, 'poppler_path_edit') and hasattr(self.poppler_path_edit, 'line_edit'):
            app_config.POPPLER_PATH = self.poppler_path_edit.line_edit.text()
            
        app_config.OFFLINE_MODE = self.offline_mode_checkbox.isChecked()
        app_config.HTTP_TIMEOUT = self.http_timeout_spinbox.value()
        
        if hasattr(self, 'batch_delay_spinner'):
            app_config.DEFAULT_BATCH_PROCESSING_DELAY = self.batch_delay_spinner.value()
            
        app_config.HF_TOKEN = self.hf_token_edit.text()
        
        if hasattr(self, 'layoutlm_model_id_edit'):
            app_config.LAYOUTLM_MODEL_ID = self.layoutlm_model_id_edit.text()
            
        if hasattr(self, 'donut_model_id_edit'):
            app_config.DONUT_MODEL_ID = self.donut_model_id_edit.text()
        # Обновление GEMINI_MODEL_ID происходит в MainWindow
        
        # Обновляем информацию в config для новых полей
        if hasattr(self, 'training_datasets_path_edit') and hasattr(self.training_datasets_path_edit, 'line_edit'):
            app_config.TRAINING_DATASETS_PATH = self.training_datasets_path_edit.line_edit.text()
            
        if hasattr(self, 'trained_models_path_edit') and hasattr(self.trained_models_path_edit, 'line_edit'):
            app_config.TRAINED_MODELS_PATH = self.trained_models_path_edit.line_edit.text()
        
        # Информация из gemini_annotation_prompt_edit уже сохранена выше

        # Обновляем информацию о моделях в основном окне, если оно доступно
        if self.parent_window:
            self.parent_window.populate_gemini_models() # Обновляем список моделей Gemini, если он изменился
            # Возможно, потребуется обновить и другие элементы UI в MainWindow, если они зависят от этих настроек

    def select_directory(self, line_edit_widget):
        """Открывает диалог выбора директории и устанавливает путь в QLineEdit."""
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку", line_edit_widget.text() or os.path.expanduser("~"))
        if directory:
            line_edit_widget.setText(directory)

    def update_batch_delay(self, value):
        """Обработчик изменения значения задержки между обработкой файлов в пакетном режиме."""
        settings_manager.set_value('Misc', 'batch_processing_delay', value)
        app_config.DEFAULT_BATCH_PROCESSING_DELAY = value
        print(f"Задержка пакетной обработки обновлена: {value} сек.")
        
    def update_default_vat_rate(self, value):
        """Обработчик изменения значения ставки НДС по умолчанию."""
        settings_manager.set_default_vat_rate(value)
        print(f"Ставка НДС по умолчанию обновлена: {value}%")
    
    def update_company_receiver_name(self, name):
        """Обработчик изменения названия компании-получателя."""
        settings_manager.set_company_receiver_name(name)
        print(f"Название компании-получателя обновлено: {name}")
        
    def save_hf_token_action(self):
        """Действие: Сохраняет токен Hugging Face."""
        new_token = self.hf_token_edit.text().strip()
        if new_token:
            # Сохраняем токен в настройках
            settings_manager.set_value('HuggingFace', 'token', new_token)
            app_config.HF_TOKEN = new_token
            
            # Сохраняем настройки
            settings_manager.save_settings()
            
            print(f"Токен Hugging Face успешно сохранен в настройках")
        else:
            # Очищаем токен, если поле пустое
            settings_manager.set_value('HuggingFace', 'token', '')
            app_config.HF_TOKEN = ''
            settings_manager.save_settings()
            print(f"Токен Hugging Face очищен")
        
        # Автоматически обновляем статус поля
        self.hf_token_status_label.setText(f"Статус токена: {'Задан' if new_token else 'Не задан'}")
        self.hf_token_status_label.setStyleSheet(f"color: {'green' if new_token else 'red'};")

    # NEW: Метод для обновления отображения информации о модели
    def update_model_info_display(self, model_type):
        """Обновляет текстовую информацию о модели на соответствующей вкладке."""
        if model_type == 'layoutlm':
            tab_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "layoutlm_tab"))
            if tab_index >= 0:
                 layoutlm_tab = self.tab_widget.widget(tab_index)
                 info_label = layoutlm_tab.findChild(QLabel, "layoutlm_info_label") # Нужен objectName
                 if info_label:
                     # Пересобираем info_text с актуальным ID
                     current_id = settings_manager.get_string('Models', 'layoutlm_id', app_config.LAYOUTLM_MODEL_ID)
                     info_text = f"<b>LayoutLMv3</b><br><br>"
                     if "layoutlm" in app_config.MODELS_INFO:
                         model_info = app_config.MODELS_INFO["layoutlm"]
                         info_text += f"<b>Название:</b> {model_info.get('name', 'LayoutLMv3')}<br>"
                         info_text += f"<b>ID:</b> {current_id}<br>" # Используем актуальный ID
                         info_text += f"<b>Версия:</b> {model_info.get('version', 'base')}<br>"
                         info_text += f"<b>Задача:</b> {model_info.get('task', 'document-understanding')}<br>"
                         info_text += f"<b>Размер:</b> ~{model_info.get('size_mb', 500)} МБ<br>"
                         info_text += f"<b>Требует OCR:</b> {'Да' if model_info.get('requires_ocr', True) else 'Нет'}<br>"
                         info_text += f"<b>Поддерживаемые языки:</b> {', '.join(model_info.get('languages', ['eng']))}<br>"
                     info_label.setText(info_text)
        elif model_type == 'donut':
            if hasattr(self, 'donut_info_label'): # Проверяем, что label существует
                 current_id = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
                 info_text = f"<b>Donut</b><br><br>"
                 if "donut" in app_config.MODELS_INFO: 
                     model_info_template = app_config.MODELS_INFO["donut"]
                     info_text += f"<b>Название:</b> {model_info_template.get('name', 'Donut')}<br>"
                     info_text += f"<b>ID:</b> {current_id}<br>"
                     info_text += f"<b>Версия:</b> {model_info_template.get('version', 'base')}<br>"
                     info_text += f"<b>Задача:</b> {model_info_template.get('task', 'document-understanding')}<br>"
                     info_text += f"<b>Размер:</b> ~{model_info_template.get('size_mb', 700)} МБ<br>"
                     info_text += f"<b>Требует OCR:</b> {'Да' if model_info_template.get('requires_ocr', False) else 'Нет'}<br>"
                     info_text += f"<b>Поддерживаемые языки:</b> {', '.join(model_info_template.get('languages', ['eng']))}<br>"
                 else: 
                     info_text += f"<b>ID:</b> {current_id}<br>"

                 self.donut_info_label.setText(info_text)
                 
                 # Также обновляем поле ввода ID, если оно есть и видимо
                 if hasattr(self, 'donut_model_id_edit') and self.donut_model_id_edit.isVisible():
                     self.donut_model_id_edit.setText(current_id)

            # Обновляем статус загрузки модели Donut
            is_loaded = settings_manager.get_bool('ModelsStatus', 'donut_loaded', False)
            current_model_id_for_status = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
            if hasattr(self, 'donut_status_label'): # Убедимся, что donut_status_label существует
                if is_loaded:
                    self.donut_status_label.setText(f"Загружена ({current_model_id_for_status})")
                else:
                    # Здесь можно добавить более сложную логику проверки наличия файлов, если нужно
                    self.donut_status_label.setText(f"Не загружена ({current_model_id_for_status})")

        elif model_type == 'gemini':
            # Аналогично для Gemini, если потребуется
            pass

    def save_model_id_action(self, model_type):
        """Сохраняет ID модели (LayoutLM или Donut) и очищает кэш при необходимости."""
        if model_type == 'layoutlm':
            edit_field = self.layoutlm_model_id_edit
            current_id_key = 'layoutlm_id'
            default_id = app_config.LAYOUTLM_MODEL_ID
            model_name = "LayoutLMv3"
            # status_label_attr = 'layoutlm_status_label' # Атрибут для прямого доступа к QLabel статуса
        elif model_type == 'donut':
            edit_field = self.donut_model_id_edit
            current_id_key = 'donut_id'
            default_id = app_config.DONUT_MODEL_ID
            model_name = "Donut"
            # status_label_attr = 'donut_status_label'
        else:
            return

        new_id = edit_field.text().strip()
        if not new_id:
            QMessageBox.warning(self, f"Ошибка ID {model_name}", f"ID модели {model_name} не может быть пустым.")
            old_id_for_revert = settings_manager.get_string('Models', current_id_key, default_id)
            edit_field.setText(old_id_for_revert) # Возвращаем старый ID в поле ввода
            return

        old_id = settings_manager.get_string('Models', current_id_key, default_id)

        if new_id != old_id:
            settings_manager.set_value('Models', current_id_key, new_id)
            # settings_manager.save_settings() # Сохранение произойдет при закрытии диалога или общем save_settings

            QMessageBox.information(self, f"ID модели {model_name}",
                                    f"ID модели {model_name} изменен на '{new_id}'.\n"
                                    "Статус загрузки модели сброшен. При необходимости загрузите модель с новым ID.")
            
            # Нужен учет смены модели чтобы потом не грузилась устаревшая модель
            settings_manager.set_value('ModelsStatus', f'{model_type}_loaded', False)
            
            # Обновляем UI
            self.update_model_info_display(model_type)
            # Обновление кнопок/статуса загрузки моделей
            if model_type == 'layoutlm':
                self._update_layoutlm_status_label_and_button(new_id, False)
            elif model_type == 'donut':
                self._update_donut_status_and_button()
            
            # Здесь мы НЕ очищаем кэш автоматически, чтобы не удалять случайно нужные модели
            # Вместо этого, пользователь может использовать кнопку "Очистить кэш"
        else:
            QMessageBox.information(self, f"ID модели {model_name}", f"ID модели {model_name} не изменился.")

    def update_model_status_label(self, model_type, status):
        """Обновляет статус загрузки модели в соответствующем QLabel."""
        if model_type == 'layoutlm':
            if hasattr(self, 'layoutlm_status_label'): # Добавим проверку на существование
                self.layoutlm_status_label.setText(status)
        elif model_type == 'donut':
            if hasattr(self, 'donut_status_label'): # Добавим проверку на существование
                self.donut_status_label.setText(status)
        # elif model_type == 'gemini': # Пока Gemini не имеет такого явного статуса в этом диалоге
        #     pass
            
    def test_huggingface_connection(self):
        """Тестирует соединение с Hugging Face."""
        try:
            # Попробуй указать свой токен, если он есть и нужен для этой модели
            # token = "твой_hf_токен" 
            token = None # или так, если токен не обязателен
            downloaded_path = hf_hub_download(
                repo_id="microsoft/layoutlmv3-base",
                filename="config.json",
                token=token
            )
            print(f"Successfully downloaded: {downloaded_path}")
        except Exception as e:
            print(f"Error downloading: {e}")
            import traceback
            traceback.print_exc()

    def _update_layoutlm_status_label_and_button(self, model_id_or_path, is_custom):
        """Обновляет статус и кнопку для LayoutLM в зависимости от того, загружена ли модель."""
        is_actually_loaded = False
        status_text = "Неизвестно"
        can_check_status = self.parent_window and hasattr(self.parent_window, 'model_manager')

        if not model_id_or_path:
            status_text = "Модель не выбрана / Путь не указан"
            # No early return, let button and color update
        elif can_check_status:
            try:
                manager = self.parent_window.model_manager
                # Construct a cache key consistent with ModelManager
                cache_key = f"layoutlm_{model_id_or_path.replace(os.sep, '_')}"
                
                if cache_key in manager.models:
                    cached_processor = manager.models[cache_key]
                    if cached_processor.model_id_loaded == model_id_or_path and \
                       cached_processor.is_custom_loaded == is_custom and \
                       cached_processor.is_loaded:
                        is_actually_loaded = True
                        status_text = f"Загружена: {os.path.basename(model_id_or_path) if is_custom else model_id_or_path}"
                    elif cached_processor.model_id_loaded == model_id_or_path and \
                         cached_processor.is_custom_loaded == is_custom and \
                         not cached_processor.is_loaded:
                         status_text = "Ошибка предыдущей загрузки (см. логи)"
                    else:
                        # A processor exists for this key, but it's for a different model_id/type
                        status_text = "Готова к загрузке" if not is_custom else "Готова к проверке"
                else:
                    # Not in active session (manager.models), check local files
                    if is_custom:
                        if os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "config.json")):
                            status_text = "Локально доступна (не в сессии)"
                        else:
                            status_text = "Локальный путь/модель не найдена"
                    else: # Hugging Face model, check HF cache
                        # Более сложная проверка кэша HF с учетом разных путей
                        model_cache_found = False
                        
                        # Проверяем кэш в стандартном месте
                        model_cache_dir_for_hf = os.path.join(app_config.MODELS_PATH, 'layoutlm', model_id_or_path.replace("/", "_"))
                        if os.path.isdir(model_cache_dir_for_hf) and os.path.exists(os.path.join(model_cache_dir_for_hf, "config.json")):
                            model_cache_found = True
                        
                        # Проверяем кэш в стандартном кэше huggingface
                        try:
                            # Сканируем кэш HF
                            cache_info = scan_cache_dir()
                            for repo in cache_info.repos:
                                if repo.repo_id.lower() == model_id_or_path.lower():
                                    model_cache_found = True
                                    break
                        except Exception as e:
                            print(f"Ошибка при сканировании кэша Hugging Face: {e}")
                        
                        # Устанавливаем статус в зависимости от результата
                        if model_cache_found:
                            status_text = "В кэше HF (не в сессии)"
                        else:
                            status_text = "Не загружена (нет в кэше HF)"
            except Exception as e:
                self._log(f"Ошибка при проверке статуса LayoutLM в ModelManagementDialog: {e}") # Use self._log
                status_text = "Ошибка проверки статуса"
        else:
            status_text = "Менеджер моделей недоступен для проверки"
            if not model_id_or_path: # Ensure correct status if path is empty and manager is unavailable
                 status_text = "Модель не выбрана / Путь не указан"

        # Обновляем только статус, но не меняем ID модели в поле ввода
        self.layoutlm_status_label.setText(status_text)
        if is_actually_loaded:
            self.layoutlm_status_label.setStyleSheet("color: green;")
        elif "Ошибка" in status_text or "не найдена" in status_text or "Не загружена" in status_text or "не указан" in status_text:
            self.layoutlm_status_label.setStyleSheet("color: red;")
        else: # "Готова к загрузке", "В кэше HF", "Локально доступна", "Менеджер недоступен" (if path valid)
            self.layoutlm_status_label.setStyleSheet("color: orange;")

        self._update_layoutlm_button_text(is_custom, is_actually_loaded, model_id_or_path)

    def _update_layoutlm_button_text(self, is_custom, is_loaded, model_id_or_path):
        """Обновляет текст и состояние кнопки LayoutLM."""
        if not model_id_or_path:
            self.download_layoutlm_button.setText("Действие с моделью")
            self.download_layoutlm_button.setEnabled(False)
            return

        if is_custom:
            if is_loaded:
                self.download_layoutlm_button.setText("Перезагрузить локальную")
                self.download_layoutlm_button.setEnabled(True)
            else:
                self.download_layoutlm_button.setText("Загрузить/Проверить локальную")
                self.download_layoutlm_button.setEnabled(os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "config.json")))
        else: # HuggingFace модель
            if is_loaded:
                self.download_layoutlm_button.setText("Обновить с Hugging Face")
            else:
                self.download_layoutlm_button.setText("Скачать с Hugging Face")
            self.download_layoutlm_button.setEnabled(True)
            
    def perform_layoutlm_action(self):
        """Загружает или очищает модель LayoutLM в зависимости от текущего состояния."""
        # Проверяем доступность менеджера моделей
        if not (self.parent_window and hasattr(self.parent_window, 'model_manager')):
            QMessageBox.warning(self, "Ошибка", "Менеджер моделей недоступен")
            return

        # Используем логику из check_models_availability для определения типа модели
        selected_layoutlm_type = self.layoutlm_model_type_combo.currentData()
        is_custom = (selected_layoutlm_type == "custom")
        model_id_or_path = ""

        if is_custom:
            # Получаем модель из поля с именем или комбо
            model_id_or_path = self.custom_layoutlm_name_edit.text().strip()
            # Если имя в поле пустое, но что-то выбрано в комбо, берем из комбо
            if not model_id_or_path and self.custom_layoutlm_model_selector.currentIndex() >= 0:
                candidate_name = self.custom_layoutlm_model_selector.currentText()
                # Убедимся, что это не плейсхолдер типа "Локальные модели не найдены"
                if candidate_name and "не найдены" not in candidate_name and "Ошибка доступа" not in candidate_name:
                    model_id_or_path = candidate_name

            # Полный путь для кастомной модели
            if model_id_or_path:  # Убедимся, что имя не пустое
                model_id_or_path = os.path.join(
                    settings_manager.get_string("Paths", "trained_models_path", app_config.TRAINED_MODELS_PATH),
                    model_id_or_path
                )
        else:  # Hugging Face
            model_id_or_path = self.layoutlm_model_id_edit.text().strip()

        if not model_id_or_path:
            QMessageBox.warning(self, "Ошибка", "ID модели или путь не может быть пустым.")
            return
        
        # Проверяем, что токен HF настроен (для некастомных моделей)
        if not is_custom and not settings_manager.get_string('HuggingFace', 'token', ''):
            # Если токен не настроен, уточняем у пользователя желание продолжить
            result = QMessageBox.question(
                self, 
                "Отсутствует токен Hugging Face", 
                "Для скачивания модели рекомендуется использовать токен Hugging Face. Продолжить без токена?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result == QMessageBox.StandardButton.No:
                return

        # Проверяем, загружена ли модель
        is_loaded = settings_manager.get_bool('ModelsStatus', 'layoutlm_loaded', False)
        
        # Определяем действие: загрузка или очистка
        if not is_loaded:
            # Загрузка модели
            if not is_custom:
                # Обычная модель из Hugging Face Hub
                self._execute_task_in_thread(
                    lambda: self.parent_window.model_manager.load_layoutlm_model(model_id_or_path),
                    self._on_layoutlm_load_success,
                    self._on_layoutlm_load_error
                )
            else:
                # Кастомная модель из локального пути
                self._execute_task_in_thread(
                    lambda: self.parent_window.model_manager.load_layoutlm_model(model_id_or_path, is_custom=True),
                    self._on_layoutlm_load_success,
                    self._on_layoutlm_load_error
                )
        else:
            # Очистка модели
            self.parent_window.model_manager.clear_layoutlm_model()
            settings_manager.set_value('ModelsStatus', 'layoutlm_loaded', False)
            settings_manager.set_value("Models", "layoutlm_is_custom", "False")
            settings_manager.save_settings()
            self._update_layoutlm_status_label_and_button(model_id_or_path, is_custom)
            QMessageBox.information(self, "Модель выгружена", "Модель LayoutLM выгружена из памяти.")
    
    def perform_donut_action(self):
        """Выполняет действие с моделью Donut (загрузка/обновление) в зависимости от текущего состояния."""
        model_id = self.donut_model_id_edit.text().strip()
        if not model_id:
            QMessageBox.warning(self, "Ошибка", "ID модели Donut не может быть пустым")
            return
            
        # Проверяем сохранён ли токен HF
        hf_token = settings_manager.get_string("HuggingFace", "token", "")
        if not hf_token:
            result = QMessageBox.question(
                self, 
                "Отсутствует токен Hugging Face", 
                "Для скачивания модели Donut рекомендуется использовать токен Hugging Face. Продолжить без токена?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result == QMessageBox.StandardButton.No:
                return
                
        # Сохраняем ID модели в настройках
        settings_manager.set_value("Models", "donut_id", model_id)
        
        # Выполняем проверку доступности модели
        # В реальной реализации здесь будет код скачивания/проверки модели
        # через ModelManager или DonutProcessor
        QMessageBox.information(
            self, 
            "Модель Donut", 
            f"ID модели Donut сохранён: {model_id}\n\nМодель будет загружена при первом использовании."
        )
        
        # Обновляем статус
        self._update_donut_status_and_button()
    
    def test_gemini_api_key_action(self):
        """Заглушка для старого метода тестирования API ключа Gemini (удален с вкладки)."""
        QMessageBox.information(
            self, 
            "Настройка API ключа", 
            "Настройка API ключа Google Gemini теперь доступна на вкладке \"🔐 API Ключи\"."
        )
    
    def save_gemini_api_key_action(self, called_from_save_settings=False):
        """Заглушка для старого метода сохранения API ключа Gemini (удален с вкладки)."""
        if not called_from_save_settings:
            QMessageBox.information(
                self, 
                "Настройка API ключа", 
                "Настройка API ключа Google Gemini теперь доступна на вкладке \"🔐 API Ключи\"."
            )
        return True
    
    def populate_gemini_models(self, models=None):
        """Заполняет выпадающий список доступных моделей Gemini."""
        if not hasattr(self, 'gemini_model_selector'):
            return
            
        self.gemini_model_selector.clear()
        
        if not GENAI_AVAILABLE:
            self.gemini_model_selector.addItem("Библиотека Google Gemini не установлена")
            self.gemini_model_selector.setEnabled(False)
            return
            
        try:
            if models is None:
                # Если ключ не настроен или нет интернета, используем дефолтный список
                if not app_config.GOOGLE_API_KEY or app_config.OFFLINE_MODE:
                    print("Используется дефолтный список моделей Gemini.")
                    models = [
                        "models/gemini-1.5-flash",
                        "models/gemini-1.5-pro",
                        "models/gemini-1.0-pro",
                        "models/gemini-2.0-flash",
                        "models/gemini-2.0-pro"
                    ]
                else:
                    try:
                        genai.configure(api_key=app_config.GOOGLE_API_KEY)
                        models_list = genai.list_models()
                        models = [m.name for m in models_list if "gemini" in m.name.lower()]
                    except Exception as e:
                        print(f"Ошибка при получении списка моделей Gemini: {e}")
                        models = ["models/gemini-2.0-flash", "models/gemini-2.0-pro"]
            else:
                # Если модели уже переданы, но это объекты из API, извлекаем имена
                if hasattr(models[0], 'name'):
                    models = [m.name for m in models if "gemini" in m.name.lower()]
                    
            # Добавляем модели в селектор
            if isinstance(models, list) and models:
                self.gemini_model_selector.addItems(models)
                
                # Сохраняем список моделей в настройки
                # Проверяем, содержит ли models объекты или строки
                if models and hasattr(models[0], 'name'):
                    models_json = json.dumps([model.name for model in models])
                else:
                    models_json = json.dumps(models)
                settings_manager.set_value('Gemini', 'available_models_json', models_json)
                
                # Выбираем текущую модель или первую в списке
                current_model = settings_manager.get_string('Gemini', 'sub_model_id', app_config.GEMINI_MODEL_ID)
                index = self.gemini_model_selector.findText(current_model)
                if index >= 0:
                    self.gemini_model_selector.setCurrentIndex(index)
                else:
                    # Если текущая модель не найдена в списке, добавляем её отдельно
                    self.gemini_model_selector.addItem(current_model)
                    self.gemini_model_selector.setCurrentText(current_model)
                
                self.gemini_model_selector.setEnabled(True)
                return
        except Exception as e:
            print(f"Ошибка при заполнении списка моделей Gemini: {e}")
            
        # Если что-то пошло не так, добавляем дефолтную модель
        self.gemini_model_selector.addItem(app_config.GEMINI_MODEL_ID)
        self.gemini_model_selector.setCurrentText(app_config.GEMINI_MODEL_ID)

    def update_gemini_model_list_action(self):
        """Действие: Обновляет список моделей Gemini."""
        self.populate_gemini_models()
        QMessageBox.information(self, "Список моделей Gemini", "Список моделей Gemini обновлен.")

    def test_huggingface_token_action(self):
        """Заглушка для старого метода тестирования HF токена (удален с вкладки)."""
        QMessageBox.information(
            self, 
            "Настройка токена", 
            "Настройка токена Hugging Face теперь доступна на вкладке \"🔐 API Ключи\"."
        )

    def clear_model_cache_action(self, model_type):
        """Действие: Очищает кэш модели."""
        if self.parent_window and hasattr(self.parent_window, 'model_manager'):
            manager = self.parent_window.model_manager
            if model_type == 'layoutlm':
                manager.clear_model_cache()
            elif model_type == 'donut':
                manager.clear_model_cache()
            else:
                return
            QMessageBox.information(self, f"Кэш {model_type}", f"Кэш модели {model_type} успешно очищен.")
        else:
            QMessageBox.warning(self, f"Ошибка очистки кэша {model_type}", "Менеджер моделей недоступен для очистки кэша.")

    def clear_all_cache_action(self):
        """Действие: Очищает все кэши."""
        if self.parent_window and hasattr(self.parent_window, 'model_manager'):
            manager = self.parent_window.model_manager
            manager.clear_all_caches()
            QMessageBox.information(self, "Все кэши", "Все кэши успешно очищены.")
        else:
            QMessageBox.warning(self, "Ошибка очистки всех кэшей", "Менеджер моделей недоступен для очистки всех кэшей.")

    def save_all_settings(self):
        """Сохраняет все настройки из всех вкладок"""
        # Сохранение настроек локальных моделей
        if hasattr(self, 'layoutlm_model_id_edit'):
            settings_manager.set_value('Models', 'layoutlm_id', self.layoutlm_model_id_edit.text())
            app_config.LAYOUTLM_MODEL_ID = self.layoutlm_model_id_edit.text()
            
        if hasattr(self, 'layoutlm_model_type_combo'):
            model_type = 'huggingface' if self.layoutlm_model_type_combo.currentIndex() == 0 else 'custom'
            settings_manager.set_value('Models', 'layoutlm_type', model_type)
            
        if hasattr(self, 'custom_layoutlm_name_edit'):
            settings_manager.set_value('Models', 'layoutlm_custom_name', self.custom_layoutlm_name_edit.text())
            
        if hasattr(self, 'donut_model_id_edit'):
            settings_manager.set_value('Models', 'donut_id', self.donut_model_id_edit.text())
            app_config.DONUT_MODEL_ID = self.donut_model_id_edit.text()

        # Сохранение настроек Gemini
        if hasattr(self, 'gemini_model_selector'):
            current_model = self.gemini_model_selector.currentData()
            if current_model:
                settings_manager.set_value('Gemini', 'selected_model', current_model)
                
        if hasattr(self, 'gemini_temperature_spinner'):
            settings_manager.set_value('Gemini', 'temperature', self.gemini_temperature_spinner.value())
            app_config.DEFAULT_GEMINI_TEMPERATURE = self.gemini_temperature_spinner.value()
            
        if hasattr(self, 'gemini_max_tokens_spinner'):
            settings_manager.set_value('Gemini', 'max_tokens', self.gemini_max_tokens_spinner.value())
            app_config.DEFAULT_GEMINI_MAX_TOKENS = self.gemini_max_tokens_spinner.value()
            
        if hasattr(self, 'gemini_pdf_dpi_spinner'):
            settings_manager.set_value('Gemini', 'pdf_dpi', self.gemini_pdf_dpi_spinner.value())
            app_config.GEMINI_PDF_DPI = self.gemini_pdf_dpi_spinner.value()

        # Сохранение путей
        if hasattr(self, 'tesseract_path_edit'):
            path = self.tesseract_path_edit.line_edit.text()
            settings_manager.set_value('Paths', 'tesseract_path', path)
            app_config.TESSERACT_PATH = path
            
        if hasattr(self, 'poppler_path_edit'):
            path = self.poppler_path_edit.line_edit.text()
            settings_manager.set_value('Paths', 'poppler_path', path)
            app_config.POPPLER_PATH = path
            
        if hasattr(self, 'training_datasets_path_edit'):
            path = self.training_datasets_path_edit.line_edit.text()
            settings_manager.set_value('Paths', 'training_datasets_path', path)
            app_config.TRAINING_DATASETS_PATH = path
            
        if hasattr(self, 'trained_models_path_edit'):
            path = self.trained_models_path_edit.line_edit.text()
            settings_manager.set_value('Paths', 'trained_models_path', path)
            app_config.TRAINED_MODELS_PATH = path

        # Сохранение промптов
        if hasattr(self, 'layoutlm_prompt_edit'):
            settings_manager.set_value('Prompts', 'layoutlm', self.layoutlm_prompt_edit.toPlainText())
            app_config.LAYOUTLM_PROMPT_DEFAULT = self.layoutlm_prompt_edit.toPlainText()
            
        if hasattr(self, 'donut_prompt_edit'):
            settings_manager.set_value('Prompts', 'donut', self.donut_prompt_edit.toPlainText())
            app_config.DONUT_PROMPT_DEFAULT = self.donut_prompt_edit.toPlainText()
            
        if hasattr(self, 'gemini_prompt_edit'):
            settings_manager.set_value('Prompts', 'gemini', self.gemini_prompt_edit.toPlainText())
            app_config.GEMINI_PROMPT_DEFAULT = self.gemini_prompt_edit.toPlainText()

        # Сохранение параметров обучения
        if hasattr(self, 'layoutlm_base_model_edit'):
            settings_manager.set_value('Training', 'layoutlm_base_model', self.layoutlm_base_model_edit.text())
            app_config.LAYOUTLM_MODEL_ID_FOR_TRAINING = self.layoutlm_base_model_edit.text()
            
        if hasattr(self, 'epochs_spinbox'):
            settings_manager.set_value('Training', 'epochs', self.epochs_spinbox.value())
            app_config.DEFAULT_TRAIN_EPOCHS = self.epochs_spinbox.value()
            
        if hasattr(self, 'batch_size_spinbox'):
            settings_manager.set_value('Training', 'batch_size', self.batch_size_spinbox.value())
            app_config.DEFAULT_TRAIN_BATCH_SIZE = self.batch_size_spinbox.value()
            
        if hasattr(self, 'learning_rate_dspinbox'):
            settings_manager.set_value('Training', 'learning_rate', self.learning_rate_dspinbox.value())
            app_config.DEFAULT_LEARNING_RATE = self.learning_rate_dspinbox.value()

        # Сохранение общих параметров
        if hasattr(self, 'batch_delay_spinner'):
            settings_manager.set_value('Misc', 'batch_processing_delay', self.batch_delay_spinner.value())
            app_config.DEFAULT_BATCH_PROCESSING_DELAY = self.batch_delay_spinner.value()
            
        if hasattr(self, 'vat_rate_spinner'):
            settings_manager.set_default_vat_rate(self.vat_rate_spinner.value())

        # Сохранение сетевых настроек
        if hasattr(self, 'offline_mode_checkbox'):
            settings_manager.set_value('Network', 'offline_mode', self.offline_mode_checkbox.isChecked())
            app_config.OFFLINE_MODE = self.offline_mode_checkbox.isChecked()
            
        if hasattr(self, 'http_timeout_spinbox'):
            settings_manager.set_value('Network', 'http_timeout', self.http_timeout_spinbox.value())
            app_config.HTTP_TIMEOUT = self.http_timeout_spinbox.value()

        # Сохранение API ключей и секретов (обрабатывается отдельно)
        if hasattr(self, '_save_gemini_key_from_secrets_tab'):
            self._save_gemini_key_from_secrets_tab()
        if hasattr(self, '_save_hf_token_from_secrets_tab'):
            self._save_hf_token_from_secrets_tab()
        if hasattr(self, '_save_paths_from_secrets_tab'):
            self._save_paths_from_secrets_tab()

        # Применяем настройки
        settings_manager.save_settings()
        
        # Синхронизируем с глобальной конфигурацией
        self._sync_settings_with_config()
        
        # Обновляем интерфейс главного окна, если доступно
        if self.parent_window:
            if hasattr(self.parent_window, 'populate_gemini_models'):
                self.parent_window.populate_gemini_models()
            if hasattr(self.parent_window, 'populate_cloud_providers'):
                self.parent_window.populate_cloud_providers()
            if hasattr(self.parent_window, 'populate_local_providers'):
                self.parent_window.populate_local_providers()
        
        QMessageBox.information(self, "Настройки сохранены", 
                               "Все настройки успешно сохранены и применены.")
        
        print("Все настройки сохранены для новой структуры интерфейса")

    def _on_gemini_sub_model_changed(self, index):
        """Действие: Обновляет статус модели Gemini при изменении выбора подмодели."""
        self.check_models_availability()

    def save_gemini_generation_parameters_action(self):
        """Действие: Сохраняет параметры генерации для модели Gemini."""
        # Проверяем существование элементов управления
        if hasattr(self, 'gemini_temperature_spinner'):
            temperature = self.gemini_temperature_spinner.value()
            settings_manager.set_value('Gemini', 'temperature', str(temperature))
            app_config.DEFAULT_GEMINI_TEMPERATURE = temperature
            
        if hasattr(self, 'gemini_max_tokens_spinner'):
            max_tokens = self.gemini_max_tokens_spinner.value()
            settings_manager.set_value('Gemini', 'max_tokens', str(max_tokens))
            app_config.DEFAULT_GEMINI_MAX_TOKENS = max_tokens
            
        if hasattr(self, 'gemini_pdf_dpi_spinner'):
            pdf_dpi = self.gemini_pdf_dpi_spinner.value()
            settings_manager.set_value('Gemini', 'pdf_dpi', str(pdf_dpi))
            app_config.GEMINI_PDF_DPI = pdf_dpi
            
        if hasattr(self, 'gemini_model_selector') and self.gemini_model_selector.currentText():
            model_id = self.gemini_model_selector.currentText()
            settings_manager.set_value('Gemini', 'sub_model_id', model_id)
            app_config.GEMINI_MODEL_ID = model_id
            
        # Сохраняем настройки если вызываем напрямую (не из save_settings)
        print(f"Сохранены параметры генерации Gemini: температура={app_config.DEFAULT_GEMINI_TEMPERATURE}, max_tokens={app_config.DEFAULT_GEMINI_MAX_TOKENS}, PDF DPI={app_config.GEMINI_PDF_DPI}")

    def save_model_prompt(self, model_type, called_from_save_settings=False):
        """Действие: Сохраняет промпт для модели."""
        prompt_text = ""
        if model_type == 'layoutlm' and hasattr(self, 'layoutlm_prompt_edit'):
            prompt_text = self.layoutlm_prompt_edit.toPlainText()
        elif model_type == 'donut' and hasattr(self, 'donut_prompt_edit'):
            prompt_text = self.donut_prompt_edit.toPlainText()
        elif model_type == 'gemini' and hasattr(self, 'gemini_prompt_edit'):
            prompt_text = self.gemini_prompt_edit.toPlainText()
        else:
            return False
            
        if prompt_text:
            # Проверяем и создаем папку для промптов, если она не существует
            prompts_path = os.path.join(app_config.APP_DATA_PATH, "prompts")
            if not os.path.exists(prompts_path):
                try:
                    os.makedirs(prompts_path, exist_ok=True)
                    print(f"Создана папка для промптов: {prompts_path}")
                except Exception as e:
                    print(f"Ошибка при создании папки для промптов: {e}")
                    
            # Сохраняем промпт в файл
            prompt_file_path = os.path.join(prompts_path, f"{model_type}_prompt.txt")
            try:
                with open(prompt_file_path, 'w', encoding='utf-8') as f:
                    f.write(prompt_text)
                print(f"Промпт для {model_type} сохранен в файл: {prompt_file_path}")
            except Exception as e:
                print(f"Ошибка при сохранении промпта в файл: {e}")
            
            # Сохраняем промпт в настройки
            settings_manager.set_value('Prompts', model_type, prompt_text)
            
            # Обновляем соответствующую переменную в конфиге
            if model_type == 'layoutlm':
                app_config.LAYOUTLM_PROMPT_DEFAULT = prompt_text
            elif model_type == 'donut':
                app_config.DONUT_PROMPT_DEFAULT = prompt_text
            elif model_type == 'gemini':
                app_config.GEMINI_PROMPT_DEFAULT = prompt_text
            
            if not called_from_save_settings:
                settings_manager.save_settings()
                QMessageBox.information(self, f"Промпт {model_type}", f"Промпт для модели {model_type} успешно сохранён.")
            return True
        return False

    def save_training_parameters_action(self, called_from_save_settings=False):
        """Сохраняет параметры обучения моделей."""
        # Проверяем существование необходимых полей
        if hasattr(self, 'layoutlm_base_model_edit'):
            base_model = self.layoutlm_base_model_edit.text().strip()
            settings_manager.set_value('Training', 'layoutlm_base_model_for_training', base_model)
            app_config.LAYOUTLM_MODEL_ID_FOR_TRAINING = base_model

        if hasattr(self, 'epochs_spinbox'):
            epochs = self.epochs_spinbox.value()
            settings_manager.set_value('Training', 'default_train_epochs', epochs)
            app_config.DEFAULT_TRAIN_EPOCHS = epochs

        if hasattr(self, 'batch_size_spinbox'):
            batch_size = self.batch_size_spinbox.value()
            settings_manager.set_value('Training', 'default_train_batch_size', batch_size)
            app_config.DEFAULT_TRAIN_BATCH_SIZE = batch_size

        if hasattr(self, 'learning_rate_dspinbox'):
            learning_rate = self.learning_rate_dspinbox.value()
            settings_manager.set_value('Training', 'default_learning_rate', str(learning_rate))
            app_config.DEFAULT_LEARNING_RATE = learning_rate

        # Сохраняем пути для обучения, если они есть
        if hasattr(self, 'training_datasets_path_edit') and hasattr(self.training_datasets_path_edit, 'line_edit'):
            training_path = self.training_datasets_path_edit.line_edit.text().strip()
            settings_manager.set_value('Training', 'training_datasets_path', training_path)
            app_config.TRAINING_DATASETS_PATH = training_path

        if hasattr(self, 'trained_models_path_edit') and hasattr(self.trained_models_path_edit, 'line_edit'):
            models_path = self.trained_models_path_edit.line_edit.text().strip()
            settings_manager.set_value('Training', 'trained_models_path', models_path)
            app_config.TRAINED_MODELS_PATH = models_path

        # Сохраняем промпт аннотации, если он есть
        if hasattr(self, 'gemini_annotation_prompt_edit'):
            try:
                prompt_text = self.gemini_annotation_prompt_edit.toPlainText()
                prompt_encoded = base64.b64encode(prompt_text.encode('utf-8')).decode('ascii')
                settings_manager.set_value('Training', 'gemini_annotation_prompt_encoded', prompt_encoded)
                app_config.GEMINI_ANNOTATION_PROMPT_DEFAULT = prompt_text
            except Exception as e:
                print(f"Ошибка при сохранении промпта аннотации: {e}")

        if not called_from_save_settings:
            settings_manager.save_settings()
            QMessageBox.information(self, "Параметры обучения", "Параметры обучения моделей успешно сохранены.")

    def update_default_vat_rate(self, value):
        """Обновляет настройку ставки НДС по умолчанию при изменении в спиннере."""
        settings_manager.set_default_vat_rate(value)

    def save_settings(self):
        """Сохранение всех настроек при закрытии диалога."""
        # Удалено: сохранение HF токена (теперь настраивается на вкладке "🔐 API Ключи")

        # Сохраняем состояние оффлайн режима
        settings_manager.set_value('Network', 'offline_mode', self.offline_mode_checkbox.isChecked())
        app_config.OFFLINE_MODE = self.offline_mode_checkbox.isChecked()

        # Сохраняем таймаут HTTP
        settings_manager.set_value('Network', 'http_timeout', self.http_timeout_spinbox.value())
        app_config.HTTP_TIMEOUT = self.http_timeout_spinbox.value()
        
        # NEW: Сохраняем задержку пакетной обработки, если элемент существует
        if hasattr(self, 'batch_delay_spinner'):
            settings_manager.set_value('Misc', 'batch_processing_delay', self.batch_delay_spinner.value())
            app_config.DEFAULT_BATCH_PROCESSING_DELAY = self.batch_delay_spinner.value()
        
        # NEW: Сохраняем ставку НДС по умолчанию, если элемент существует
        if hasattr(self, 'vat_rate_spinner'):
            settings_manager.set_default_vat_rate(self.vat_rate_spinner.value())
            
        # NEW: Сохраняем название компании-получателя, если элемент существует
        if hasattr(self, 'company_receiver_name_edit'):
            settings_manager.set_company_receiver_name(self.company_receiver_name_edit.text())

        # Сохраняем пути (если эти элементы есть)
        if hasattr(self, 'tesseract_path_edit') and hasattr(self.tesseract_path_edit, 'line_edit'):
            settings_manager.set_value('Paths', 'tesseract_path', self.tesseract_path_edit.line_edit.text())
            app_config.TESSERACT_PATH = self.tesseract_path_edit.line_edit.text()
            
        if hasattr(self, 'poppler_path_edit') and hasattr(self.poppler_path_edit, 'line_edit'):
            settings_manager.set_value('Paths', 'poppler_path', self.poppler_path_edit.line_edit.text())
            app_config.POPPLER_PATH = self.poppler_path_edit.line_edit.text()
            
        if hasattr(self, 'training_datasets_path_edit') and hasattr(self.training_datasets_path_edit, 'line_edit'):
            settings_manager.set_value('Training', 'training_datasets_path', self.training_datasets_path_edit.line_edit.text())
            app_config.TRAINING_DATASETS_PATH = self.training_datasets_path_edit.line_edit.text()
            
        if hasattr(self, 'trained_models_path_edit') and hasattr(self.trained_models_path_edit, 'line_edit'):
            settings_manager.set_value('Training', 'trained_models_path', self.trained_models_path_edit.line_edit.text())
            app_config.TRAINED_MODELS_PATH = self.trained_models_path_edit.line_edit.text()
            
        if hasattr(self, 'layoutlm_prompt_edit'):
            settings_manager.set_value('Prompts', 'layoutlm', self.layoutlm_prompt_edit.toPlainText())
            app_config.LAYOUTLM_PROMPT_DEFAULT = self.layoutlm_prompt_edit.toPlainText()
            
        if hasattr(self, 'donut_prompt_edit'):
            settings_manager.set_value('Prompts', 'donut', self.donut_prompt_edit.toPlainText())
            app_config.DONUT_PROMPT_DEFAULT = self.donut_prompt_edit.toPlainText()
            
        if hasattr(self, 'gemini_prompt_edit'):
            settings_manager.set_value('Prompts', 'gemini', self.gemini_prompt_edit.toPlainText())
            app_config.GEMINI_PROMPT_DEFAULT = self.gemini_prompt_edit.toPlainText()
            
        if hasattr(self, 'gemini_temperature_spinner'):
            settings_manager.set_value('Gemini', 'temperature', str(self.gemini_temperature_spinner.value()))
            app_config.DEFAULT_GEMINI_TEMPERATURE = self.gemini_temperature_spinner.value()
            
        if hasattr(self, 'gemini_max_tokens_spinner'):
            settings_manager.set_value('Gemini', 'max_tokens', str(self.gemini_max_tokens_spinner.value()))
            app_config.DEFAULT_GEMINI_MAX_TOKENS = self.gemini_max_tokens_spinner.value()
            
        if hasattr(self, 'gemini_pdf_dpi_spinner'):
            settings_manager.set_value('Gemini', 'pdf_dpi', str(self.gemini_pdf_dpi_spinner.value()))
            app_config.GEMINI_PDF_DPI = self.gemini_pdf_dpi_spinner.value()
            
        if hasattr(self, 'layoutlm_base_model_edit'):
            settings_manager.set_value('Training', 'layoutlm_base_model_for_training', self.layoutlm_base_model_edit.text().strip())
            app_config.LAYOUTLM_MODEL_ID_FOR_TRAINING = self.layoutlm_base_model_edit.text().strip()
            
        if hasattr(self, 'epochs_spinbox'):
            settings_manager.set_value('Training', 'default_train_epochs', self.epochs_spinbox.value())
            app_config.DEFAULT_TRAIN_EPOCHS = self.epochs_spinbox.value()
            
        if hasattr(self, 'batch_size_spinbox'):
            settings_manager.set_value('Training', 'default_train_batch_size', self.batch_size_spinbox.value())
            app_config.DEFAULT_TRAIN_BATCH_SIZE = self.batch_size_spinbox.value()
            
        if hasattr(self, 'learning_rate_dspinbox'):
            settings_manager.set_value('Training', 'default_learning_rate', str(self.learning_rate_dspinbox.value()))
            app_config.DEFAULT_LEARNING_RATE = self.learning_rate_dspinbox.value()
            
        if hasattr(self, 'gemini_annotation_prompt_edit'):
            try:
                prompt_text = self.gemini_annotation_prompt_edit.toPlainText()
                prompt_encoded = base64.b64encode(prompt_text.encode('utf-8')).decode('ascii')
                settings_manager.set_value('Training', 'gemini_annotation_prompt_encoded', prompt_encoded)
                app_config.GEMINI_ANNOTATION_PROMPT_DEFAULT = prompt_text
            except Exception as e:
                print(f"Ошибка при сохранении промпта аннотации: {e}")
        
        # Сохраняем настройки и уведомляем пользователя
        settings_manager.save_settings()
        QMessageBox.information(self, "Параметры", "Параметры успешно сохранены.")

    def _create_api_keys_tab(self):
        """Создает вкладку для управления API ключами и секретами."""
        api_keys_tab = QWidget()
        layout = QVBoxLayout(api_keys_tab)
        
        # Заголовок
        header_label = QLabel("🔐 Безопасное управление API ключами")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin: 10px 0;")
        layout.addWidget(header_label)
        
        # Описание
        description = QLabel(
            "Введите ваши API ключи здесь. Они будут автоматически сохранены в зашифрованном виде и добавлены в .env файл.\n"
            "Никогда не коммитьте .env файл в Git - он автоматически исключен из версионного контроля."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        layout.addWidget(description)
        
        # Область прокрутки для секретов
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        
        secrets_widget = QWidget()
        secrets_layout = QVBoxLayout(secrets_widget)
        
        # --- Google Gemini API ---
        gemini_group = QGroupBox("🤖 Google Gemini API")
        gemini_layout = QFormLayout(gemini_group)
        
        # API ключ Gemini
        self.secrets_gemini_api_key_edit = QLineEdit()
        self.secrets_gemini_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.secrets_gemini_api_key_edit.setPlaceholderText("Вставьте ваш Google AI API ключ здесь...")
        
        # Кнопки для Gemini
        gemini_buttons_layout = QHBoxLayout()
        self.secrets_test_gemini_button = QPushButton("🔍 Проверить ключ")
        self.secrets_save_gemini_button = QPushButton("💾 Сохранить")
        self.secrets_show_gemini_button = QPushButton("👁️ Показать/Скрыть")
        
        self.secrets_test_gemini_button.clicked.connect(self._test_gemini_key_from_secrets_tab)
        self.secrets_save_gemini_button.clicked.connect(self._save_gemini_key_from_secrets_tab)
        self.secrets_show_gemini_button.clicked.connect(lambda: self._toggle_password_visibility(self.secrets_gemini_api_key_edit))
        
        gemini_buttons_layout.addWidget(self.secrets_test_gemini_button)
        gemini_buttons_layout.addWidget(self.secrets_save_gemini_button)
        gemini_buttons_layout.addWidget(self.secrets_show_gemini_button)
        gemini_buttons_layout.addStretch()
        
        # Статус Gemini
        self.secrets_gemini_status_label = QLabel("Статус: Не проверен")
        self.secrets_gemini_status_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
        
        # Справка по Gemini
        gemini_help = QLabel(
            '<a href="https://makersuite.google.com/app/apikey">Получить Google AI API ключ</a> | '
            'Требуется для обработки изображений с помощью Gemini'
        )
        gemini_help.setOpenExternalLinks(True)
        gemini_help.setStyleSheet("color: #3498db; margin-top: 5px;")
        
        gemini_layout.addRow("API Ключ:", self.secrets_gemini_api_key_edit)
        gemini_layout.addRow("Действия:", gemini_buttons_layout)
        gemini_layout.addRow("Статус:", self.secrets_gemini_status_label)
        gemini_layout.addRow("Справка:", gemini_help)
        
        secrets_layout.addWidget(gemini_group)
        
        # --- Hugging Face Token ---
        hf_group = QGroupBox("🤗 Hugging Face Token")
        hf_layout = QFormLayout(hf_group)
        
        # HF токен
        self.secrets_hf_token_edit = QLineEdit()
        self.secrets_hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.secrets_hf_token_edit.setPlaceholderText("Вставьте ваш Hugging Face токен здесь...")
        
        # Кнопки для HF
        hf_buttons_layout = QHBoxLayout()
        self.secrets_test_hf_button = QPushButton("🔍 Проверить токен")
        self.secrets_save_hf_button = QPushButton("💾 Сохранить")
        self.secrets_show_hf_button = QPushButton("👁️ Показать/Скрыть")
        
        self.secrets_test_hf_button.clicked.connect(self._test_hf_token_from_secrets_tab)
        self.secrets_save_hf_button.clicked.connect(self._save_hf_token_from_secrets_tab)
        self.secrets_show_hf_button.clicked.connect(lambda: self._toggle_password_visibility(self.secrets_hf_token_edit))
        
        hf_buttons_layout.addWidget(self.secrets_test_hf_button)
        hf_buttons_layout.addWidget(self.secrets_save_hf_button)
        hf_buttons_layout.addWidget(self.secrets_show_hf_button)
        hf_buttons_layout.addStretch()
        
        # Статус HF
        self.secrets_hf_status_label = QLabel("Статус: Не проверен")
        self.secrets_hf_status_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
        
        # Справка по HF
        hf_help = QLabel(
            '<a href="https://huggingface.co/settings/tokens">Получить Hugging Face токен</a> | '
            'Требуется для загрузки моделей LayoutLM и Donut'
        )
        hf_help.setOpenExternalLinks(True)
        hf_help.setStyleSheet("color: #3498db; margin-top: 5px;")
        
        hf_layout.addRow("Токен:", self.secrets_hf_token_edit)
        hf_layout.addRow("Действия:", hf_buttons_layout)
        hf_layout.addRow("Статус:", self.secrets_hf_status_label)
        hf_layout.addRow("Справка:", hf_help)
        
        secrets_layout.addWidget(hf_group)
        
        # --- Пути к внешним инструментам ---
        paths_group = QGroupBox("🛠️ Пути к внешним инструментам")
        paths_layout = QFormLayout(paths_group)
        
        # Tesseract OCR
        self.secrets_tesseract_path_edit = QLineEdit()
        self.secrets_tesseract_path_edit.setPlaceholderText("Оставьте пустым для автопоиска...")
        tesseract_browse_button = QPushButton("📁 Обзор")
        tesseract_browse_button.clicked.connect(lambda: self._browse_executable(self.secrets_tesseract_path_edit, "Tesseract OCR"))
        
        tesseract_layout = QHBoxLayout()
        tesseract_layout.addWidget(self.secrets_tesseract_path_edit, 1)
        tesseract_layout.addWidget(tesseract_browse_button)
        
        # Poppler 
        self.secrets_poppler_path_edit = QLineEdit()
        self.secrets_poppler_path_edit.setPlaceholderText("Путь к папке bin Poppler...")
        poppler_browse_button = QPushButton("📁 Обзор")
        poppler_browse_button.clicked.connect(lambda: self._browse_directory(self.secrets_poppler_path_edit, "Poppler bin"))
        
        poppler_layout = QHBoxLayout()
        poppler_layout.addWidget(self.secrets_poppler_path_edit, 1)
        poppler_layout.addWidget(poppler_browse_button)
        
        paths_layout.addRow("Tesseract OCR:", tesseract_layout)
        paths_layout.addRow("Poppler PDF:", poppler_layout)
        
        secrets_layout.addWidget(paths_group)
        
        # --- Сводка и действия ---
        summary_group = QGroupBox("📊 Сводка безопасности")
        summary_layout = QVBoxLayout(summary_group)
        
        self.secrets_summary_label = QLabel("Загрузка статуса секретов...")
        self.secrets_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.secrets_summary_label)
        
        # Кнопки управления
        actions_layout = QHBoxLayout()
        
        self.refresh_secrets_button = QPushButton("🔄 Обновить статус")
        self.refresh_secrets_button.clicked.connect(self._refresh_secrets_status)
        
        self.create_env_button = QPushButton("📝 Создать .env файл")
        self.create_env_button.clicked.connect(self._create_env_file)
        
        self.open_env_button = QPushButton("📂 Открыть .env файл")
        self.open_env_button.clicked.connect(self._open_env_file)
        
        actions_layout.addWidget(self.refresh_secrets_button)
        actions_layout.addWidget(self.create_env_button)
        actions_layout.addWidget(self.open_env_button)
        actions_layout.addStretch()
        
        summary_layout.addLayout(actions_layout)
        secrets_layout.addWidget(summary_group)
        
        # Завершение области прокрутки
        secrets_layout.addStretch()
        scroll_area.setWidget(secrets_widget)
        layout.addWidget(scroll_area)
        
        # Добавляем вкладку
        self.tab_widget.addTab(api_keys_tab, "🔐 API Ключи")
        
        # Загружаем текущие значения
        self._load_secrets_values()

    def _load_secrets_values(self):
        """Загружает текущие значения секретов в поля ввода."""
        try:
            # Загружаем Google Gemini API ключ
            if self.secrets_manager:
                gemini_key = self.secrets_manager.get_secret("GOOGLE_API_KEY")
            else:
                gemini_key = settings_manager.get_gemini_api_key()
            
            if gemini_key:
                self.secrets_gemini_api_key_edit.setText(gemini_key)
                self.secrets_gemini_status_label.setText("Статус: Загружен из настроек")
                self.secrets_gemini_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            # Загружаем Hugging Face токен
            if self.secrets_manager:
                hf_token = self.secrets_manager.get_secret("HF_TOKEN")
            else:
                hf_token = settings_manager.get_huggingface_token()
            
            if hf_token:
                self.secrets_hf_token_edit.setText(hf_token)
                self.secrets_hf_status_label.setText("Статус: Загружен из настроек")
                self.secrets_hf_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            # Загружаем пути к инструментам
            tesseract_path = settings_manager.get_string('Paths', 'tesseract_path', app_config.TESSERACT_PATH or '')
            self.secrets_tesseract_path_edit.setText(tesseract_path)
            
            poppler_path = settings_manager.get_string('Paths', 'poppler_path', app_config.POPPLER_PATH or '')
            self.secrets_poppler_path_edit.setText(poppler_path)
            
            # Обновляем сводку
            self._refresh_secrets_status()
            
        except Exception as e:
            print(f"Ошибка при загрузке значений секретов: {e}")
            self.secrets_summary_label.setText(f"Ошибка загрузки: {e}")

    def _toggle_password_visibility(self, line_edit):
        """Переключает видимость пароля в поле ввода."""
        if line_edit.echoMode() == QLineEdit.EchoMode.Password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def _test_gemini_key_from_secrets_tab(self):
        """Тестирует Google Gemini API ключ с вкладки секретов."""
        api_key = self.secrets_gemini_api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Ошибка", "API ключ Google не может быть пустым")
            self.secrets_gemini_status_label.setText("Статус: Не указан")
            self.secrets_gemini_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
            
        try:
            self.secrets_gemini_status_label.setText("Статус: Проверка...")
            self.secrets_gemini_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.secrets_test_gemini_button.setEnabled(False)
            QApplication.processEvents()
            
            # Проверяем наличие библиотеки
            if not GENAI_AVAILABLE:
                QMessageBox.warning(
                    self, 
                    "Ошибка", 
                    "Библиотека google-generativeai не установлена.\nУстановите её командой: pip install google-generativeai"
                )
                self.secrets_gemini_status_label.setText("Статус: Ошибка библиотеки")
                self.secrets_gemini_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                return
                
            # Тестируем ключ
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            models = genai.list_models()
            gemini_models = [m for m in models if "gemini" in m.name.lower()]
            
            if gemini_models:
                self.secrets_gemini_status_label.setText("Статус: ✅ Ключ работает")
                self.secrets_gemini_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Проверка API ключа", 
                    f"✅ API ключ Google Gemini работает!\nДоступно {len(gemini_models)} моделей Gemini."
                )
            else:
                self.secrets_gemini_status_label.setText("Статус: ⚠️ Нет доступа к Gemini")
                self.secrets_gemini_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                QMessageBox.warning(
                    self, 
                    "Проверка API ключа", 
                    "⚠️ API ключ работает, но нет доступа к моделям Gemini.\nВозможно, у вашего аккаунта нет доступа к API Gemini."
                )
        except Exception as e:
            self.secrets_gemini_status_label.setText("Статус: ❌ Ошибка проверки")
            self.secrets_gemini_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            QMessageBox.critical(
                self, 
                "Ошибка проверки", 
                f"❌ Ошибка при проверке API ключа Google:\n{str(e)}"
            )
        finally:
            self.secrets_test_gemini_button.setEnabled(True)
            self._refresh_secrets_status()

    def _save_gemini_key_from_secrets_tab(self):
        """Сохраняет Google Gemini API ключ с вкладки секретов."""
        api_key = self.secrets_gemini_api_key_edit.text().strip()
        
        try:
            if self.secrets_manager and api_key:
                # Сохраняем через безопасную систему
                self.secrets_manager.set_secret("GOOGLE_API_KEY", api_key)
                
                # Также сохраняем в старую систему для совместимости
                settings_manager.set_value('Gemini', 'api_key', api_key)
                app_config.GOOGLE_API_KEY = api_key
                
                # Обновляем поле на основной вкладке, если оно существует
                if hasattr(self, 'gemini_api_key_edit'):
                    self.gemini_api_key_edit.setText(api_key)
                
                self.secrets_gemini_status_label.setText("Статус: 💾 Сохранен")
                self.secrets_gemini_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Сохранение ключа", 
                    "✅ Google Gemini API ключ успешно сохранен в зашифрованном виде!"
                )
            elif api_key:
                # Fallback на старую систему
                settings_manager.set_value('Gemini', 'api_key', api_key)
                app_config.GOOGLE_API_KEY = api_key
                settings_manager.save_settings()
                
                self.secrets_gemini_status_label.setText("Статус: 💾 Сохранен (незашифрованно)")
                self.secrets_gemini_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Сохранение ключа", 
                    "⚠️ API ключ сохранен, но система шифрования недоступна.\nРекомендуется обновить систему безопасности."
                )
            else:
                # Очистка ключа
                if self.secrets_manager:
                    self.secrets_manager.delete_secret("GOOGLE_API_KEY")
                settings_manager.set_value('Gemini', 'api_key', '')
                app_config.GOOGLE_API_KEY = None
                
                self.secrets_gemini_status_label.setText("Статус: Очищен")
                self.secrets_gemini_status_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
                
                QMessageBox.information(self, "Очистка ключа", "🗑️ Google Gemini API ключ удален.")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка сохранения", 
                f"❌ Ошибка при сохранении API ключа Google:\n{str(e)}"
            )
        finally:
            self._refresh_secrets_status()

    def _test_hf_token_from_secrets_tab(self):
        """Тестирует Hugging Face токен с вкладки секретов."""
        token = self.secrets_hf_token_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "Ошибка", "Hugging Face токен не может быть пустым")
            self.secrets_hf_status_label.setText("Статус: Не указан")
            self.secrets_hf_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
            
        try:
            self.secrets_hf_status_label.setText("Статус: Проверка...")
            self.secrets_hf_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.secrets_test_hf_button.setEnabled(False)
            QApplication.processEvents()
            
            # Тестируем токен загрузкой небольшого файла
            repo_id = "microsoft/layoutlmv3-base"
            filename = "config.json"
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token
            )
            
            self.secrets_hf_status_label.setText("Статус: ✅ Токен работает")
            self.secrets_hf_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            QMessageBox.information(
                self, 
                "Проверка токена", 
                f"✅ Hugging Face токен работает!\nТестовый файл загружен: {downloaded_path}"
            )
            
        except Exception as e:
            self.secrets_hf_status_label.setText("Статус: ❌ Ошибка проверки")
            self.secrets_hf_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            QMessageBox.critical(
                self, 
                "Ошибка проверки", 
                f"❌ Ошибка при проверке Hugging Face токена:\n{str(e)}\n\nПроверьте правильность токена и подключение к интернету."
            )
        finally:
            self.secrets_test_hf_button.setEnabled(True)
            self._refresh_secrets_status()

    def _save_hf_token_from_secrets_tab(self):
        """Сохраняет Hugging Face токен с вкладки секретов."""
        token = self.secrets_hf_token_edit.text().strip()
        
        try:
            if self.secrets_manager and token:
                # Сохраняем через безопасную систему
                self.secrets_manager.set_secret("HF_TOKEN", token)
                
                # Также сохраняем в старую систему для совместимости
                settings_manager.set_value('HuggingFace', 'token', token)
                app_config.HF_TOKEN = token
                
                # Обновляем поле на основной вкладке, если оно существует
                if hasattr(self, 'hf_token_edit'):
                    self.hf_token_edit.setText(token)
                
                self.secrets_hf_status_label.setText("Статус: 💾 Сохранен")
                self.secrets_hf_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Сохранение токена", 
                    "✅ Hugging Face токен успешно сохранен в зашифрованном виде!"
                )
            elif token:
                # Fallback на старую систему
                settings_manager.set_value('HuggingFace', 'token', token)
                app_config.HF_TOKEN = token
                settings_manager.save_settings()
                
                self.secrets_hf_status_label.setText("Статус: 💾 Сохранен (незашифрованно)")
                self.secrets_hf_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Сохранение токена", 
                    "⚠️ Токен сохранен, но система шифрования недоступна.\nРекомендуется обновить систему безопасности."
                )
            else:
                # Очистка токена
                if self.secrets_manager:
                    self.secrets_manager.delete_secret("HF_TOKEN")
                settings_manager.set_value('HuggingFace', 'token', '')
                app_config.HF_TOKEN = ''
                
                self.secrets_hf_status_label.setText("Статус: Очищен")
                self.secrets_hf_status_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
                
                QMessageBox.information(self, "Очистка токена", "🗑️ Hugging Face токен удален.")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка сохранения", 
                f"❌ Ошибка при сохранении Hugging Face токена:\n{str(e)}"
            )
        finally:
            self._refresh_secrets_status()

    def _browse_executable(self, line_edit, tool_name):
        """Открывает диалог выбора исполняемого файла."""
        current_path = line_edit.text() or os.path.expanduser("~")
        file_filter = "Исполняемые файлы (*.exe)" if sys.platform == "win32" else "Все файлы (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Выберите {tool_name}", 
            current_path, 
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)
            self._save_paths_from_secrets_tab()

    def _browse_directory(self, line_edit, tool_name):
        """Открывает диалог выбора директории."""
        current_path = line_edit.text() or os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            f"Выберите папку {tool_name}", 
            current_path
        )
        if dir_path:
            line_edit.setText(dir_path)
            self._save_paths_from_secrets_tab()

    def _save_paths_from_secrets_tab(self):
        """Сохраняет пути к инструментам с вкладки секретов."""
        try:
            # Сохраняем пути
            tesseract_path = self.secrets_tesseract_path_edit.text().strip()
            poppler_path = self.secrets_poppler_path_edit.text().strip()
            
            settings_manager.set_value('Paths', 'tesseract_path', tesseract_path)
            settings_manager.set_value('Paths', 'poppler_path', poppler_path)
            
            app_config.TESSERACT_PATH = tesseract_path
            app_config.POPPLER_PATH = poppler_path
            
            # Обновляем поля на других вкладках, если они существуют
            if hasattr(self, 'tesseract_path_edit') and hasattr(self.tesseract_path_edit, 'line_edit'):
                self.tesseract_path_edit.line_edit.setText(tesseract_path)
            if hasattr(self, 'poppler_path_edit') and hasattr(self.poppler_path_edit, 'line_edit'):
                self.poppler_path_edit.line_edit.setText(poppler_path)
            
            settings_manager.save_settings()
            self._refresh_secrets_status()
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка сохранения", 
                f"❌ Ошибка при сохранении путей:\n{str(e)}"
            )

    def _refresh_secrets_status(self):
        """Обновляет сводку статуса секретов."""
        try:
            status_parts = []
            
            # Проверяем Google Gemini API
            gemini_key = ""
            if self.secrets_manager:
                gemini_key = self.secrets_manager.get_secret("GOOGLE_API_KEY")
            else:
                gemini_key = settings_manager.get_gemini_api_key()
            
            if gemini_key:
                status_parts.append("✅ Google Gemini API: Настроен")
            else:
                status_parts.append("❌ Google Gemini API: Не настроен")
            
            # Проверяем Hugging Face токен
            hf_token = ""
            if self.secrets_manager:
                hf_token = self.secrets_manager.get_secret("HF_TOKEN")
            else:
                hf_token = settings_manager.get_huggingface_token()
            
            if hf_token:
                status_parts.append("✅ Hugging Face: Настроен")
            else:
                status_parts.append("❌ Hugging Face: Не настроен")
            
            # Проверяем пути к инструментам
            tesseract_path = self.secrets_tesseract_path_edit.text().strip()
            if tesseract_path and os.path.exists(tesseract_path):
                status_parts.append("✅ Tesseract OCR: Найден")
            elif tesseract_path:
                status_parts.append("⚠️ Tesseract OCR: Путь указан, но файл не найден")
            else:
                status_parts.append("🔍 Tesseract OCR: Автопоиск")
            
            poppler_path = self.secrets_poppler_path_edit.text().strip()
            if poppler_path and os.path.exists(poppler_path):
                status_parts.append("✅ Poppler PDF: Найден")
            elif poppler_path:
                status_parts.append("⚠️ Poppler PDF: Путь указан, но папка не найдена")
            else:
                status_parts.append("🔍 Poppler PDF: Используется встроенный")
            
            # Проверяем .env файл
            env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
            if os.path.exists(env_file_path):
                status_parts.append("✅ .env файл: Существует")
            else:
                status_parts.append("❌ .env файл: Не создан")
            
            # Проверяем систему шифрования
            if self.secrets_manager:
                status_parts.append("🔒 Шифрование: Активно")
            else:
                status_parts.append("⚠️ Шифрование: Недоступно")
            
            summary_text = "\n".join(status_parts)
            self.secrets_summary_label.setText(summary_text)
            
        except Exception as e:
            self.secrets_summary_label.setText(f"❌ Ошибка обновления статуса: {e}")

    def _create_env_file(self):
        """Создает .env файл с текущими секретами."""
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            env_file_path = os.path.join(project_root, ".env")
            
            # Собираем секреты
            env_content = []
            env_content.append("# =============================================================================")
            env_content.append("# INVOICEGEMINI ENVIRONMENT VARIABLES")
            env_content.append("# =============================================================================")
            env_content.append("# Автоматически создано через интерфейс настроек")
            env_content.append(f"# Дата создания: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            env_content.append("#")
            env_content.append("# ⚠️  ВАЖНО: НЕ КОММИТЬТЕ ЭТОТ ФАЙЛ В GIT!")
            env_content.append("# Файл автоматически исключен из версионного контроля")
            env_content.append("#")
            env_content.append("# =============================================================================")
            env_content.append("")
            
            # API ключи
            env_content.append("# -----------------------------------------------------------------------------")
            env_content.append("# API КЛЮЧИ И ТОКЕНЫ")
            env_content.append("# -----------------------------------------------------------------------------")
            
            gemini_key = self.secrets_gemini_api_key_edit.text().strip()
            if gemini_key:
                env_content.append(f"GOOGLE_API_KEY={gemini_key}")
            else:
                env_content.append("# GOOGLE_API_KEY=your_google_api_key_here")
            
            hf_token = self.secrets_hf_token_edit.text().strip()
            if hf_token:
                env_content.append(f"HF_TOKEN={hf_token}")
            else:
                env_content.append("# HF_TOKEN=your_hugging_face_token_here")
            
            env_content.append("")
            
            # Пути к инструментам
            env_content.append("# -----------------------------------------------------------------------------")
            env_content.append("# ПУТИ К ВНЕШНИМ ИНСТРУМЕНТАМ")
            env_content.append("# -----------------------------------------------------------------------------")
            
            tesseract_path = self.secrets_tesseract_path_edit.text().strip()
            if tesseract_path:
                env_content.append(f"TESSERACT_PATH={tesseract_path}")
            else:
                env_content.append("# TESSERACT_PATH=")
            
            poppler_path = self.secrets_poppler_path_edit.text().strip()
            if poppler_path:
                env_content.append(f"POPPLER_PATH={poppler_path}")
            else:
                env_content.append("# POPPLER_PATH=")
            
            env_content.append("")
            
            # Дополнительные настройки
            env_content.append("# -----------------------------------------------------------------------------")
            env_content.append("# ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ")
            env_content.append("# -----------------------------------------------------------------------------")
            env_content.append("# OFFLINE_MODE=false")
            env_content.append("# MAX_MODEL_MEMORY=4000")
            env_content.append("# DEFAULT_TESSERACT_LANG=rus+eng")
            env_content.append("")
            
            # Записываем файл
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(env_content))
            
            # Устанавливаем права доступа (только для владельца)
            if hasattr(os, 'chmod'):
                os.chmod(env_file_path, 0o600)
            
            QMessageBox.information(
                self, 
                "Создание .env файла", 
                f"✅ .env файл успешно создан!\n\nПуть: {env_file_path}\n\nФайл содержит ваши API ключи и настройки."
            )
            
            self._refresh_secrets_status()
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка создания .env файла", 
                f"❌ Ошибка при создании .env файла:\n{str(e)}"
            )

    def _open_env_file(self):
        """Открывает .env файл в системном редакторе."""
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            env_file_path = os.path.join(project_root, ".env")
            
            if not os.path.exists(env_file_path):
                result = QMessageBox.question(
                    self, 
                    "Файл не найден", 
                    ".env файл не существует. Создать его сейчас?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if result == QMessageBox.StandardButton.Yes:
                    self._create_env_file()
                    return
                else:
                    return
            
            # Открываем файл в системном редакторе
            if sys.platform == "win32":
                os.startfile(env_file_path)
            elif sys.platform == "darwin":  # macOS
                os.system(f"open '{env_file_path}'")
            else:  # Linux
                os.system(f"xdg-open '{env_file_path}'")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка открытия файла", 
                f"❌ Ошибка при открытии .env файла:\n{str(e)}"
            )

    def _create_cloud_models_tab(self):
        """Создает вкладку для настройки облачных моделей"""
        cloud_tab = QWidget()
        cloud_layout = QVBoxLayout(cloud_tab)
        
        # Заголовок
        header_label = QLabel("☁️ Облачные модели")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3; margin-bottom: 10px;")
        cloud_layout.addWidget(header_label)
        
        # Scroll area для длинного содержимого
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Gemini настройки
        gemini_group = QGroupBox("🔍 Google Gemini")
        gemini_layout = QVBoxLayout()
        
        # Ссылка на API ключи
        api_info_label = QLabel("💡 Настройка API ключа доступна на вкладке \"🔐 API Ключи\"")
        api_info_label.setStyleSheet("color: #3498db; font-style: italic; margin: 5px 0;")
        gemini_layout.addWidget(api_info_label)
        
        # Выбор модели Gemini
        gemini_model_layout = QHBoxLayout()
        gemini_model_label = QLabel("Модель:")
        self.gemini_model_selector = QComboBox()
        self.gemini_model_selector.currentIndexChanged.connect(self._on_gemini_sub_model_changed)
        gemini_model_layout.addWidget(gemini_model_label)
        gemini_model_layout.addWidget(self.gemini_model_selector, 1)
        gemini_layout.addLayout(gemini_model_layout)
        
        # Кнопка обновления списка моделей
        self.update_gemini_list_button = QPushButton("🔄 Обновить список моделей")
        self.update_gemini_list_button.clicked.connect(self.update_gemini_model_list_action)
        gemini_layout.addWidget(self.update_gemini_list_button)
        
        # Параметры генерации Gemini
        gemini_params_group = QGroupBox("Параметры генерации")
        gemini_params_layout = QFormLayout()
        
        self.gemini_temperature_spinner = QDoubleSpinBox()
        self.gemini_temperature_spinner.setRange(0.0, 1.0)
        self.gemini_temperature_spinner.setSingleStep(0.05)
        self.gemini_temperature_spinner.setDecimals(2)
        gemini_params_layout.addRow("Температура (0.0 - 1.0):", self.gemini_temperature_spinner)
        
        self.gemini_max_tokens_spinner = QSpinBox()
        self.gemini_max_tokens_spinner.setRange(1, 32768)
        self.gemini_max_tokens_spinner.setSingleStep(512)
        gemini_params_layout.addRow("Максимальные токены:", self.gemini_max_tokens_spinner)
        
        self.gemini_pdf_dpi_spinner = QSpinBox()
        self.gemini_pdf_dpi_spinner.setRange(72, 600)
        self.gemini_pdf_dpi_spinner.setSingleStep(50)
        gemini_params_layout.addRow("DPI для PDF:", self.gemini_pdf_dpi_spinner)
        
        gemini_params_group.setLayout(gemini_params_layout)
        gemini_layout.addWidget(gemini_params_group)
        gemini_group.setLayout(gemini_layout)
        scroll_layout.addWidget(gemini_group)
        
        # Другие облачные LLM провайдеры
        llm_group = QGroupBox("🤖 Другие LLM провайдеры")
        llm_layout = QVBoxLayout()
        
        llm_info_label = QLabel("💡 Настройка API ключей и моделей доступна через \"Настройки → Управление LLM плагинами\"")
        llm_info_label.setStyleSheet("color: #3498db; font-style: italic; margin: 5px 0;")
        llm_layout.addWidget(llm_info_label)
        
        # Кнопка открытия LLM настроек
        self.open_llm_settings_button = QPushButton("🔌 Открыть настройки LLM провайдеров")
        self.open_llm_settings_button.clicked.connect(self._open_llm_providers_dialog)
        llm_layout.addWidget(self.open_llm_settings_button)
        
        # Список доступных провайдеров
        providers_label = QLabel("Поддерживаемые провайдеры:")
        providers_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        llm_layout.addWidget(providers_label)
        
        providers_list = QLabel("• OpenAI (GPT-4, GPT-3.5)\n• Anthropic (Claude)\n• Mistral AI\n• DeepSeek\n• xAI (Grok)")
        providers_list.setStyleSheet("margin-left: 20px; color: #666;")
        llm_layout.addWidget(providers_list)
        
        llm_group.setLayout(llm_layout)
        scroll_layout.addWidget(llm_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        cloud_layout.addWidget(scroll)
        
        self.tab_widget.addTab(cloud_tab, "☁️ Облачные модели")

    def _create_local_models_tab(self):
        """Создает вкладку для настройки локальных моделей"""
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        
        # Заголовок
        header_label = QLabel("🖥️ Локальные модели")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50; margin-bottom: 10px;")
        local_layout.addWidget(header_label)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # LayoutLM настройки
        layoutlm_group = QGroupBox("📄 LayoutLMv3 (Анализ документов)")
        layoutlm_layout = QVBoxLayout()
        
        # Тип модели LayoutLM
        layoutlm_type_layout = QHBoxLayout()
        layoutlm_type_label = QLabel("Тип модели:")
        self.layoutlm_model_type_combo = QComboBox()
        self.layoutlm_model_type_combo.addItem("Hugging Face модель", "huggingface")
        self.layoutlm_model_type_combo.addItem("Локальная дообученная", "custom")
        self.layoutlm_model_type_combo.currentIndexChanged.connect(self._on_layoutlm_model_type_changed)
        layoutlm_type_layout.addWidget(layoutlm_type_label)
        layoutlm_type_layout.addWidget(self.layoutlm_model_type_combo, 1)
        layoutlm_layout.addLayout(layoutlm_type_layout)
        
        # HF модель настройки
        self.hf_layoutlm_group = QGroupBox("Hugging Face модель")
        hf_layoutlm_layout = QFormLayout()
        self.layoutlm_model_id_edit = QLineEdit()
        self.layoutlm_model_id_edit.setPlaceholderText("microsoft/layoutlmv3-base")
        hf_layoutlm_layout.addRow("ID модели:", self.layoutlm_model_id_edit)
        self.hf_layoutlm_group.setLayout(hf_layoutlm_layout)
        layoutlm_layout.addWidget(self.hf_layoutlm_group)
        
        # Локальная модель настройки
        self.custom_layoutlm_group = QGroupBox("Локальная дообученная модель")
        custom_layoutlm_layout = QVBoxLayout()
        
        self.custom_layoutlm_name_edit = QLineEdit()
        self.custom_layoutlm_name_edit.setPlaceholderText("my_layoutlm_model")
        custom_layoutlm_layout.addWidget(QLabel("Имя папки модели:"))
        custom_layoutlm_layout.addWidget(self.custom_layoutlm_name_edit)
        
        custom_layoutlm_layout.addWidget(QLabel("Или выберите из найденных:"))
        self.custom_layoutlm_model_selector = QComboBox()
        self.custom_layoutlm_model_selector.currentIndexChanged.connect(self._on_custom_layoutlm_selected_from_combo)
        custom_layoutlm_layout.addWidget(self.custom_layoutlm_model_selector)
        
        self.custom_layoutlm_group.setLayout(custom_layoutlm_layout)
        layoutlm_layout.addWidget(self.custom_layoutlm_group)
        
        # Статус LayoutLM
        layoutlm_status_layout = QHBoxLayout()
        self.layoutlm_status_label = QLabel("Статус: Проверка...")
        self.download_layoutlm_button = QPushButton("Управление моделью")
        self.download_layoutlm_button.clicked.connect(self.perform_layoutlm_action)
        layoutlm_status_layout.addWidget(self.layoutlm_status_label, 1)
        layoutlm_status_layout.addWidget(self.download_layoutlm_button)
        layoutlm_layout.addLayout(layoutlm_status_layout)
        
        self.layoutlm_progress = QProgressBar()
        self.layoutlm_progress.setVisible(False)
        layoutlm_layout.addWidget(self.layoutlm_progress)
        
        layoutlm_group.setLayout(layoutlm_layout)
        scroll_layout.addWidget(layoutlm_group)
        
        # Donut настройки
        donut_group = QGroupBox("🍩 Donut (Извлечение из документов)")
        donut_layout = QVBoxLayout()
        
        donut_model_layout = QHBoxLayout()
        donut_model_label = QLabel("ID модели Hugging Face:")
        self.donut_model_id_edit = QLineEdit()
        self.donut_model_id_edit.setPlaceholderText("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.donut_model_id_edit.textChanged.connect(lambda: self.check_models_availability())
        donut_model_layout.addWidget(donut_model_label)
        donut_model_layout.addWidget(self.donut_model_id_edit, 1)
        donut_layout.addLayout(donut_model_layout)
        
        # Статус Donut
        donut_status_layout = QHBoxLayout()
        self.donut_status_label = QLabel("Статус: Проверка...")
        self.download_donut_button = QPushButton("Управление моделью")
        self.download_donut_button.clicked.connect(self.perform_donut_action)
        donut_status_layout.addWidget(self.donut_status_label, 1)
        donut_status_layout.addWidget(self.download_donut_button)
        donut_layout.addLayout(donut_status_layout)
        
        self.donut_progress = QProgressBar()
        self.donut_progress.setVisible(False)
        donut_layout.addWidget(self.donut_progress)
        
        donut_group.setLayout(donut_layout)
        scroll_layout.addWidget(donut_group)
        
        # Ollama настройки
        ollama_group = QGroupBox("🦙 Ollama (Локальные LLM)")
        ollama_layout = QVBoxLayout()
        
        ollama_info_label = QLabel("Ollama позволяет запускать LLM модели локально на вашем компьютере")
        ollama_info_label.setStyleSheet("color: #666; font-style: italic; margin: 5px 0;")
        ollama_layout.addWidget(ollama_info_label)
        
        # Проверка статуса Ollama
        ollama_status_layout = QHBoxLayout()
        self.ollama_status_label = QLabel("Статус: Проверка...")
        self.check_ollama_button = QPushButton("🔄 Проверить Ollama")
        self.check_ollama_button.clicked.connect(self._check_ollama_status)
        ollama_status_layout.addWidget(self.ollama_status_label, 1)
        ollama_status_layout.addWidget(self.check_ollama_button)
        ollama_layout.addLayout(ollama_status_layout)
        
        # Ссылка на установку
        install_info = QLabel('<a href="https://ollama.ai">Скачать и установить Ollama</a>')
        install_info.setOpenExternalLinks(True)
        install_info.setStyleSheet("color: #2196F3; margin: 10px 0;")
        ollama_layout.addWidget(install_info)
        
        ollama_group.setLayout(ollama_layout)
        scroll_layout.addWidget(ollama_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        local_layout.addWidget(scroll)
        
        self.tab_widget.addTab(local_tab, "🖥️ Локальные модели")

    def _create_general_settings_tab(self):
        """Создает вкладку общих настроек"""
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Заголовок
        header_label = QLabel("⚙️ Общие настройки")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #666; margin-bottom: 10px;")
        general_layout.addWidget(header_label)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Основные настройки организации (в самом верху)
        company_group = QGroupBox("🏢 Настройки организации")
        company_layout = QFormLayout()
        
        # Название компании-получателя (главная настройка)
        self.company_receiver_name_edit = QLineEdit()
        self.company_receiver_name_edit.setPlaceholderText(app_config.DEFAULT_COMPANY_RECEIVER_NAME)
        self.company_receiver_name_edit.textChanged.connect(self.update_company_receiver_name)
        self.company_receiver_name_edit.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        company_layout.addRow("📋 Название компании-получателя:", self.company_receiver_name_edit)
        
        # Ставка НДС по умолчанию  
        self.vat_rate_spinner = QDoubleSpinBox()
        self.vat_rate_spinner.setRange(0, 100)
        self.vat_rate_spinner.setSingleStep(0.1)
        self.vat_rate_spinner.setDecimals(1)
        self.vat_rate_spinner.setSuffix(" %")
        self.vat_rate_spinner.valueChanged.connect(self.update_default_vat_rate)
        company_layout.addRow("💰 Ставка НДС по умолчанию:", self.vat_rate_spinner)
        
        company_group.setLayout(company_layout)
        scroll_layout.addWidget(company_group)
        
        # Пути к внешним инструментам
        paths_group = QGroupBox("📁 Пути к внешним инструментам")
        paths_layout = QFormLayout()
        
        self.tesseract_path_edit = self._create_path_input()
        paths_layout.addRow("Tesseract OCR:", self.tesseract_path_edit)
        
        self.poppler_path_edit = self._create_path_input()
        paths_layout.addRow("Poppler (папка bin):", self.poppler_path_edit)
        
        paths_group.setLayout(paths_layout)
        scroll_layout.addWidget(paths_group)
        
        # Пути к данным
        data_paths_group = QGroupBox("💾 Пути к данным")
        data_paths_layout = QFormLayout()
        
        self.training_datasets_path_edit = self._create_path_input()
        data_paths_layout.addRow("Датасеты обучения:", self.training_datasets_path_edit)
        
        self.trained_models_path_edit = self._create_path_input()
        data_paths_layout.addRow("Дообученные модели:", self.trained_models_path_edit)
        
        data_paths_group.setLayout(data_paths_layout)
        scroll_layout.addWidget(data_paths_group)
        
        # Промпты
        prompts_group = QGroupBox("💬 Промпты по умолчанию")
        prompts_layout = QVBoxLayout()
        
        # LayoutLM промпт
        prompts_layout.addWidget(QLabel("Промпт для LayoutLM:"))
        self.layoutlm_prompt_edit = QTextEdit()
        self.layoutlm_prompt_edit.setMaximumHeight(80)
        prompts_layout.addWidget(self.layoutlm_prompt_edit)
        
        # Donut промпт
        prompts_layout.addWidget(QLabel("Промпт для Donut:"))
        self.donut_prompt_edit = QTextEdit()
        self.donut_prompt_edit.setMaximumHeight(80)
        prompts_layout.addWidget(self.donut_prompt_edit)
        
        # Gemini промпт
        prompts_layout.addWidget(QLabel("Промпт для Gemini:"))
        self.gemini_prompt_edit = QTextEdit()
        self.gemini_prompt_edit.setMaximumHeight(80)
        prompts_layout.addWidget(self.gemini_prompt_edit)
        
        prompts_group.setLayout(prompts_layout)
        scroll_layout.addWidget(prompts_group)
        
        # Общие параметры
        general_params_group = QGroupBox("🔧 Общие параметры")
        general_params_layout = QFormLayout()
        
        self.batch_delay_spinner = QSpinBox()
        self.batch_delay_spinner.setRange(0, 60)
        self.batch_delay_spinner.setSuffix(" сек")
        self.batch_delay_spinner.valueChanged.connect(self.update_batch_delay)
        general_params_layout.addRow("⏱️ Задержка между файлами:", self.batch_delay_spinner)
        
        general_params_group.setLayout(general_params_layout)
        scroll_layout.addWidget(general_params_group)
        
        # Сетевые настройки
        network_group = QGroupBox("🌐 Сетевые настройки")
        network_layout = QFormLayout()
        
        self.offline_mode_checkbox = QCheckBox("Автономный режим (только локальный кэш)")
        network_layout.addRow("", self.offline_mode_checkbox)
        
        self.http_timeout_spinbox = QSpinBox()
        self.http_timeout_spinbox.setRange(5, 300)
        self.http_timeout_spinbox.setSuffix(" сек")
        network_layout.addRow("Тайм-аут HTTP:", self.http_timeout_spinbox)
        
        network_group.setLayout(network_layout)
        scroll_layout.addWidget(network_group)
        
        # Управление кэшем
        cache_group = QGroupBox("🗄️ Управление кэшем")
        cache_layout = QVBoxLayout()
        
        self.cache_info_label = QLabel("Информация о кэше будет отображена здесь")
        self.cache_info_label.setWordWrap(True)
        self.cache_info_label.setStyleSheet("color: #666; margin: 10px 0;")
        cache_layout.addWidget(self.cache_info_label)
        
        cache_buttons_layout = QHBoxLayout()
        
        clear_layoutlm_button = QPushButton("🗑️ Очистить кэш LayoutLM")
        clear_layoutlm_button.clicked.connect(lambda: self.clear_model_cache_action('layoutlm'))
        cache_buttons_layout.addWidget(clear_layoutlm_button)
        
        clear_donut_button = QPushButton("🗑️ Очистить кэш Donut")
        clear_donut_button.clicked.connect(lambda: self.clear_model_cache_action('donut'))
        cache_buttons_layout.addWidget(clear_donut_button)
        
        cache_layout.addLayout(cache_buttons_layout)
        
        clear_all_button = QPushButton("🗑️ Очистить ВЕСЬ кэш моделей")
        clear_all_button.setStyleSheet("background-color: #ffdddd; color: #d32f2f; font-weight: bold;")
        clear_all_button.clicked.connect(self.clear_all_cache_action)
        cache_layout.addWidget(clear_all_button)
        
        cache_group.setLayout(cache_layout)
        scroll_layout.addWidget(cache_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        general_layout.addWidget(scroll)
        
        self.tab_widget.addTab(general_tab, "⚙️ Общие настройки")

    def _create_path_input(self):
        """Создает виджет для ввода пути с кнопкой обзора"""
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        browse_button = QPushButton("📁")
        browse_button.setMaximumWidth(40)
        browse_button.clicked.connect(lambda: self._browse_path(line_edit))
        
        widget = QWidget()
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_button)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        # Сохраняем ссылку на line_edit для доступа к значению
        widget.line_edit = line_edit
        
        return widget

    def _browse_path(self, line_edit):
        """Открывает диалог выбора пути"""
        current_path = line_edit.text() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Выберите папку", current_path)
        if path:
            line_edit.setText(path)

    def _open_llm_providers_dialog(self):
        """Открывает диалог настройки LLM провайдеров"""
        try:
            from .ui.llm_providers_dialog import LLMProvidersDialog
            dialog = LLMProvidersDialog(self)
            dialog.exec()
        except ImportError:
            QMessageBox.warning(self, "Ошибка", "Диалог LLM провайдеров не найден")

    def _check_ollama_status(self):
        """Проверяет статус Ollama"""
        # Проверка будет реализована позже
        self.ollama_status_label.setText("Статус: Проверка не реализована")

    def _test_all_connections(self):
        """Тестирует все подключения"""
        # Реализация тестирования будет добавлена позже
        QMessageBox.information(self, "Тестирование", "Функция тестирования будет реализована позже")

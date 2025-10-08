"""
Диалог управления плагинами для InvoiceGemini
Максимально дружелюбный интерфейс для пользователей
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading

from PyQt6.QtWidgets import (    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QGroupBox,    QLabel, QPushButton, QComboBox, QProgressBar, QTextEdit, QTableWidget,    QTableWidgetItem, QFileDialog, QMessageBox, QSplitter, QFrame,    QScrollArea, QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox,    QLineEdit, QFormLayout, QHeaderView, QApplication, QSlider, QInputDialog)

# ФАЗА 2: Импорт оптимизированных UI компонентов
from .performance_optimized_widgets import OptimizedTableWidget
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QIcon, QFont, QPalette, QMovie

from ..plugins.unified_plugin_manager import get_unified_plugin_manager
from ..plugins.llm_trainer import LLMTrainer, TrainingMetrics

class PluginsDialog(QDialog):
    """
    Главный диалог управления плагинами
    """
    
    def __init__(self, parent=None, model_manager=None):
        super().__init__(parent)
        self.parent_window = parent
        self.model_manager = model_manager
        
        # Инициализируем менеджер плагинов
        self.plugin_manager = get_unified_plugin_manager()
        
        # Текущий активный тренер
        self.current_trainer = None
        self.training_thread = None
        
        self.setup_ui()
        self.load_plugin_info()
        
        # Таймер для обновления прогресса
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
    
    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("🔌 " + self.tr("Управление LLM плагинами - InvoiceGemini"))
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # Основной layout
        main_layout = QVBoxLayout(self)
        
        # Заголовок с информацией
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Табы
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Вкладки
        self.create_plugins_tab()        # Управление плагинами
        self.create_training_tab()       # Обучение моделей
        self.create_dataset_tab()        # Подготовка датасета
        self.create_monitoring_tab()     # Мониторинг
        
        main_layout.addWidget(self.tab_widget)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        self.refresh_button = QPushButton("🔄 " + self.tr("Обновить"))
        self.refresh_button.clicked.connect(self.refresh_all)
        
        self.close_button = QPushButton(self.tr("Закрыть"))
        self.close_button.clicked.connect(self.accept)
        
        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.close_button)
        main_layout.addLayout(buttons_layout)
    
    def create_header(self) -> QWidget:
        """Создает информационный заголовок"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        
        # Главный заголовок
        title_label = QLabel("🚀 " + self.tr("Система плагинов локальных LLM"))
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Подзаголовок
        subtitle_label = QLabel(self.tr("Обучайте и используйте модели Llama, Mistral, CodeLlama локально"))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666; font-size: 12px;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        return header_widget
    
    def create_plugins_tab(self):
        """Вкладка управления плагинами"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Статистика плагинов
        stats_group = QGroupBox("📊 " + self.tr("Статистика плагинов"))
        stats_layout = QGridLayout(stats_group)
        
        self.total_plugins_label = QLabel("0")
        self.loaded_plugins_label = QLabel("0")
        self.available_plugins_label = QLabel("0")
        
        stats_layout.addWidget(QLabel(self.tr("Всего плагинов:")), 0, 0)
        stats_layout.addWidget(self.total_plugins_label, 0, 1)
        stats_layout.addWidget(QLabel(self.tr("Загружено:")), 0, 2)
        stats_layout.addWidget(self.loaded_plugins_label, 0, 3)
        stats_layout.addWidget(QLabel(self.tr("Доступно:")), 0, 4)
        stats_layout.addWidget(self.available_plugins_label, 0, 5)
        
        layout.addWidget(stats_group)
        
        # Список плагинов
        plugins_group = QGroupBox("🔌 " + self.tr("Доступные плагины"))
        plugins_layout = QVBoxLayout(plugins_group)
        
        # ФАЗА 2: Оптимизированная таблица плагинов
        self.plugins_table = OptimizedTableWidget()
        self.plugins_table.setColumnCount(6)
        self.plugins_table.setHorizontalHeaderLabels([
            self.tr("Название"), self.tr("Тип"), self.tr("Статус"), self.tr("Память"), self.tr("Действия"), self.tr("Информация")
        ])
        
        # Настройка таблицы
        header = self.plugins_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        
        plugins_layout.addWidget(self.plugins_table)
        layout.addWidget(plugins_group)
        
        # Установка пользовательских плагинов
        install_group = QGroupBox("📦 " + self.tr("Установка плагинов"))
        install_layout = QHBoxLayout(install_group)
        
        install_button = QPushButton("📁 " + self.tr("Установить из файла"))
        install_button.clicked.connect(self.install_plugin_from_file)
        
        create_template_button = QPushButton("📝 " + self.tr("Создать шаблон"))
        create_template_button.clicked.connect(self.create_plugin_template)
        
        install_layout.addWidget(install_button)
        install_layout.addWidget(create_template_button)
        install_layout.addStretch()
        
        layout.addWidget(install_group)
        
        self.tab_widget.addTab(tab, "🔌 " + self.tr("Плагины"))
    
    def create_training_tab(self):
        """Вкладка обучения моделей"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Выбор плагина для обучения
        plugin_group = QGroupBox("🎯 " + self.tr("Выбор модели для обучения"))
        plugin_layout = QFormLayout(plugin_group)
        
        self.training_plugin_combo = QComboBox()
        self.training_plugin_combo.currentTextChanged.connect(self.on_training_plugin_changed)
        plugin_layout.addRow(self.tr("Плагин:"), self.training_plugin_combo)
        
        self.plugin_info_label = QLabel(self.tr("Выберите плагин для получения информации"))
        self.plugin_info_label.setWordWrap(True)
        plugin_layout.addRow(self.tr("Информация:"), self.plugin_info_label)
        
        layout.addWidget(plugin_group)
        
        # Настройки обучения
        settings_group = QGroupBox("⚙️ " + self.tr("Параметры обучения"))
        settings_layout = QFormLayout(settings_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(3)
        settings_layout.addRow(self.tr("Эпохи:"), self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        settings_layout.addRow(self.tr("Размер батча:"), self.batch_size_spin)
        
        self.learning_rate_edit = QLineEdit("2e-4")
        settings_layout.addRow(self.tr("Learning rate:"), self.learning_rate_edit)
        
        self.lora_rank_spin = QSpinBox()
        self.lora_rank_spin.setRange(4, 128)
        self.lora_rank_spin.setValue(16)
        settings_layout.addRow(self.tr("LoRA Rank:"), self.lora_rank_spin)
        
        layout.addWidget(settings_group)
        
        # Выбор датасета
        dataset_group = QGroupBox("📊 " + self.tr("Датасет"))
        dataset_layout = QVBoxLayout(dataset_group)
        
        dataset_select_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText(self.tr("Выберите датасет для обучения..."))
        
        select_dataset_button = QPushButton("📁 " + self.tr("Выбрать"))
        select_dataset_button.clicked.connect(self.select_training_dataset)
        
        dataset_select_layout.addWidget(self.dataset_path_edit)
        dataset_select_layout.addWidget(select_dataset_button)
        dataset_layout.addLayout(dataset_select_layout)
        
        # Информация о датасете
        self.dataset_info_label = QLabel(self.tr("Информация о датасете появится после выбора"))
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_info_label)
        
        layout.addWidget(dataset_group)
        
        # Управление обучением
        control_group = QGroupBox("🚀 " + self.tr("Управление обучением"))
        control_layout = QVBoxLayout(control_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.start_training_button = QPushButton("🚀 " + self.tr("Начать обучение"))
        self.start_training_button.clicked.connect(self.start_training)
        
        self.stop_training_button = QPushButton("🛑 " + self.tr("Остановить"))
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setEnabled(False)
        
        buttons_layout.addWidget(self.start_training_button)
        buttons_layout.addWidget(self.stop_training_button)
        buttons_layout.addStretch()
        control_layout.addLayout(buttons_layout)
        
        # Прогресс
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        control_layout.addWidget(self.training_progress)
        
        self.training_status_label = QLabel(self.tr("Готов к обучению"))
        control_layout.addWidget(self.training_status_label)
        
        layout.addWidget(control_group)
        
        self.tab_widget.addTab(tab, "🎓 " + self.tr("Обучение"))
    
    def create_dataset_tab(self):
        """Вкладка подготовки датасета"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Информация
        info_label = QLabel(
            self.tr("💡 Эта вкладка поможет подготовить качественный датасет для обучения LLM моделей.\n") +
            self.tr("Используется Gemini API для создания правильных аннотаций на основе ваших документов."))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #e3f2fd; padding: 10px; border-radius: 5px; color: #1976d2;")
        layout.addWidget(info_label)
        
        # Выбор изображений
        images_group = QGroupBox("📁 Исходные документы")
        images_layout = QVBoxLayout(images_group)
        
        select_images_layout = QHBoxLayout()
        self.images_folder_edit = QLineEdit()
        self.images_folder_edit.setPlaceholderText(self.tr("Выберите папку с изображениями счетов..."))
        
        select_folder_button = QPushButton("📁 " + self.tr("Выбрать папку"))
        select_folder_button.clicked.connect(self.select_images_folder)
        
        select_images_layout.addWidget(self.images_folder_edit)
        select_images_layout.addWidget(select_folder_button)
        images_layout.addLayout(select_images_layout)
        
        # Информация о найденных файлах
        self.images_info_label = QLabel(self.tr("Информация о файлах появится после выбора папки"))
        images_layout.addWidget(self.images_info_label)
        
        layout.addWidget(images_group)
        
        # Настройки генерации
        generation_group = QGroupBox("⚙️ Настройки генерации")
        generation_layout = QFormLayout(generation_group)
        
        self.output_dataset_edit = QLineEdit()
        self.output_dataset_edit.setPlaceholderText(self.tr("Имя датасета..."))
        generation_layout.addRow(self.tr("Название датасета:"), self.output_dataset_edit)
        
        self.use_gemini_checkbox = QCheckBox("Использовать Gemini для аннотаций")
        self.use_gemini_checkbox.setChecked(True)
        generation_layout.addRow("", self.use_gemini_checkbox)
        
        layout.addWidget(generation_group)
        
        # Процесс генерации
        process_group = QGroupBox("🔄 Процесс генерации")
        process_layout = QVBoxLayout(process_group)
        
        # Кнопка запуска
        start_generation_layout = QHBoxLayout()
        self.start_generation_button = QPushButton("🚀 " + self.tr("Начать генерацию датасета"))
        self.start_generation_button.clicked.connect(self.start_dataset_generation)
        
        self.stop_generation_button = QPushButton("🛑 " + self.tr("Остановить"))
        self.stop_generation_button.clicked.connect(self.stop_dataset_generation)
        self.stop_generation_button.setEnabled(False)
        
        start_generation_layout.addWidget(self.start_generation_button)
        start_generation_layout.addWidget(self.stop_generation_button)
        start_generation_layout.addStretch()
        process_layout.addLayout(start_generation_layout)
        
        # Прогресс
        self.generation_progress = QProgressBar()
        self.generation_progress.setVisible(False)
        process_layout.addWidget(self.generation_progress)
        
        # Лог процесса
        self.generation_log = QTextEdit()
        self.generation_log.setMaximumHeight(200)
        self.generation_log.setReadOnly(True)
        process_layout.addWidget(self.generation_log)
        
        layout.addWidget(process_group)
        
        self.tab_widget.addTab(tab, "📊 " + self.tr("Датасет"))
    
    def create_monitoring_tab(self):
        """Вкладка мониторинга"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Текущее обучение
        current_group = QGroupBox("📈 Текущее обучение")
        current_layout = QGridLayout(current_group)
        
        # Метрики в реальном времени
        self.current_epoch_label = QLabel("0")
        self.current_step_label = QLabel("0")
        self.current_loss_label = QLabel("0.000")
        self.current_lr_label = QLabel("0.0000")
        
        current_layout.addWidget(QLabel(self.tr("Эпоха:")), 0, 0)
        current_layout.addWidget(self.current_epoch_label, 0, 1)
        current_layout.addWidget(QLabel(self.tr("Шаг:")), 0, 2)
        current_layout.addWidget(self.current_step_label, 0, 3)
        
        current_layout.addWidget(QLabel(self.tr("Loss:")), 1, 0)
        current_layout.addWidget(self.current_loss_label, 1, 1)
        current_layout.addWidget(QLabel(self.tr("Learning Rate:")), 1, 2)
        current_layout.addWidget(self.current_lr_label, 1, 3)
        
        layout.addWidget(current_group)
        
        # История обучения
        history_group = QGroupBox("📋 История обучения")
        history_layout = QVBoxLayout(history_group)
        
        # ФАЗА 2: Оптимизированная таблица истории
        self.history_table = OptimizedTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            self.tr("Эпоха"), self.tr("Шаг"), self.tr("Loss"), self.tr("Eval Loss"), self.tr("Время")
        ])
        
        history_layout.addWidget(self.history_table)
        layout.addWidget(history_group)
        
        # Системная информация
        system_group = QGroupBox("💻 Система")
        system_layout = QGridLayout(system_group)
        
        self.gpu_memory_label = QLabel("Загрузка...")
        self.gpu_utilization_label = QLabel("Загрузка...")
        self.cpu_usage_label = QLabel("Загрузка...")
        
        system_layout.addWidget(QLabel(self.tr("GPU память:")), 0, 0)
        system_layout.addWidget(self.gpu_memory_label, 0, 1)
        system_layout.addWidget(QLabel(self.tr("GPU загрузка:")), 0, 2)
        system_layout.addWidget(self.gpu_utilization_label, 0, 3)
        
        system_layout.addWidget(QLabel(self.tr("CPU:")), 1, 0)
        system_layout.addWidget(self.cpu_usage_label, 1, 1)
        
        layout.addWidget(system_group)
        
        self.tab_widget.addTab(tab, "📊 " + self.tr("Мониторинг"))
    
    def load_plugin_info(self):
        """Загружает информацию о плагинах"""
        # Обновляем статистику
        stats = self.plugin_manager.get_plugin_statistics()
        self.total_plugins_label.setText(str(stats["total_plugins"]))
        self.loaded_plugins_label.setText(str(stats["loaded_instances"]))
        self.available_plugins_label.setText(str(len(stats["available_plugins"])))
        
        # Заполняем таблицу плагинов
        self.populate_plugins_table()
        
        # Заполняем комбобокс для обучения
        self.populate_training_combo()
    
    def populate_plugins_table(self):
        """Заполняет таблицу плагинов"""
        available_plugins = self.plugin_manager.get_available_plugins()
        self.plugins_table.setRowCount(len(available_plugins))
        
        for i, plugin_id in enumerate(available_plugins):
            plugin_info = self.plugin_manager.get_plugin_info(plugin_id)
            
            # Название
            self.plugins_table.setItem(i, 0, QTableWidgetItem(plugin_info["name"]))
            
            # Тип
            self.plugins_table.setItem(i, 1, QTableWidgetItem(plugin_id.title()))
            
            # Статус
            status = "✅ Загружен" if plugin_info["is_loaded"] else "⚪ Не загружен"
            self.plugins_table.setItem(i, 2, QTableWidgetItem(status))
            
            # Оценка памяти
            instance = self.plugin_manager.get_plugin_instance(plugin_id)
            if instance:
                memory_req = instance.get_model_info().get("memory_requirements", "Неизвестно")
            else:
                # Создаем временный экземпляр для получения информации
                temp_instance = self.plugin_manager.create_plugin_instance(plugin_id)
                if temp_instance:
                    memory_req = temp_instance.get_model_info().get("memory_requirements", "Неизвестно")
                    self.plugin_manager.remove_plugin_instance(plugin_id)
                else:
                    memory_req = "Ошибка"
            
            self.plugins_table.setItem(i, 3, QTableWidgetItem(memory_req))
            
            # Кнопки действий
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(5, 2, 5, 2)
            
            if plugin_info["is_loaded"]:
                unload_btn = QPushButton("Выгрузить")
                unload_btn.clicked.connect(lambda checked, pid=plugin_id: self.unload_plugin(pid))
                actions_layout.addWidget(unload_btn)
            else:
                load_btn = QPushButton("Загрузить")
                load_btn.clicked.connect(lambda checked, pid=plugin_id: self.load_plugin(pid))
                actions_layout.addWidget(load_btn)
            
            self.plugins_table.setCellWidget(i, 4, actions_widget)
            
            # Кнопка информации
            info_btn = QPushButton("ℹ️")
            info_btn.clicked.connect(lambda checked, pid=plugin_id: self.show_plugin_info(pid))
            self.plugins_table.setCellWidget(i, 5, info_btn)
    
    def populate_training_combo(self):
        """Заполняет комбобокс для выбора плагина обучения"""
        self.training_plugin_combo.clear()
        available_plugins = self.plugin_manager.get_available_plugins()
        
        for plugin_id in available_plugins:
            self.training_plugin_combo.addItem(f"{plugin_id.title()} Plugin", plugin_id)
    
    def on_training_plugin_changed(self):
        """Обработчик смены плагина для обучения"""
        plugin_id = self.training_plugin_combo.currentData()
        if plugin_id:
            # Создаем временный экземпляр для получения информации
            instance = self.plugin_manager.get_plugin_instance(plugin_id)
            if not instance:
                instance = self.plugin_manager.create_plugin_instance(plugin_id)
            
            if instance:
                info = instance.get_model_info()
                config = instance.get_training_config()
                
                info_text = f"""
                <b>Модель:</b> {info.get('name', 'Unknown')}<br>
                <b>Семейство:</b> {info.get('model_family', 'Unknown')}<br>
                <b>Устройство:</b> {info.get('device', 'Unknown')}<br>
                <b>Требования памяти:</b> {info.get('memory_requirements', 'Неизвестно')}<br>
                <b>LoRA поддержка:</b> {'Да' if config.get('supports_lora') else 'Нет'}<br>
                <b>Рекомендуемый batch size:</b> {config.get('training_args', {}).get('batch_size', 4)}
                """
                
                self.plugin_info_label.setText(info_text)
                
                # Устанавливаем рекомендуемые параметры
                training_args = config.get('training_args', {})
                self.batch_size_spin.setValue(training_args.get('batch_size', 4))
                self.epochs_spin.setValue(training_args.get('num_epochs', 3))
                self.lora_rank_spin.setValue(config.get('default_lora_rank', 16))
    
    def load_plugin(self, plugin_id: str):
        """Загружает плагин"""
        try:
            instance = self.plugin_manager.create_plugin_instance(plugin_id)
            if instance:
                QMessageBox.information(self, "Успех", f"Плагин {plugin_id} загружен успешно!")
                self.populate_plugins_table()
            else:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить плагин {plugin_id}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки плагина: {e}")
    
    def unload_plugin(self, plugin_id: str):
        """Выгружает плагин"""
        try:
            if self.plugin_manager.remove_plugin_instance(plugin_id):
                QMessageBox.information(self, "Успех", f"Плагин {plugin_id} выгружен")
                self.populate_plugins_table()
            else:
                QMessageBox.warning(self, "Ошибка", f"Не удалось выгрузить плагин {plugin_id}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка выгрузки плагина: {e}")
    
    def show_plugin_info(self, plugin_id: str):
        """Показывает детальную информацию о плагине"""
        plugin_info = self.plugin_manager.get_plugin_info(plugin_id)
        
        # Получаем дополнительную информацию
        instance = self.plugin_manager.get_plugin_instance(plugin_id)
        if not instance:
            instance = self.plugin_manager.create_plugin_instance(plugin_id)
            temp_created = True
        else:
            temp_created = False
        
        if instance:
            model_info = instance.get_model_info()
            training_config = instance.get_training_config()
            
            info_text = f"""
            <h3>📋 Информация о плагине: {plugin_info['name']}</h3>
            
            <h4>🔧 Основные параметры:</h4>
            <ul>
            <li><b>ID:</b> {plugin_id}</li>
            <li><b>Модуль:</b> {plugin_info['module']}</li>
            <li><b>Семейство модели:</b> {model_info.get('model_family', 'Unknown')}</li>
            <li><b>Путь к модели:</b> {model_info.get('model_path', 'Unknown')}</li>
            <li><b>Устройство:</b> {model_info.get('device', 'Unknown')}</li>
            <li><b>Загружена:</b> {'Да' if model_info.get('is_loaded') else 'Нет'}</li>
            </ul>
            
            <h4>💾 Требования к ресурсам:</h4>
            <ul>
            <li><b>Память:</b> {model_info.get('memory_requirements', 'Неизвестно')}</li>
            <li><b>PyTorch доступен:</b> {'Да' if model_info.get('torch_available') else 'Нет'}</li>
            </ul>
            
            <h4>🎓 Параметры обучения:</h4>
            <ul>
            <li><b>LoRA поддержка:</b> {'Да' if training_config.get('supports_lora') else 'Нет'}</li>
            <li><b>QLoRA поддержка:</b> {'Да' if training_config.get('supports_qlora') else 'Нет'}</li>
            <li><b>Рекомендуемый LoRA rank:</b> {training_config.get('default_lora_rank', 16)}</li>
            <li><b>Максимальная длина последовательности:</b> {training_config.get('max_sequence_length', 2048)}</li>
            </ul>
            
            <h4>⚙️ Рекомендуемые параметры обучения:</h4>
            <ul>
            <li><b>Batch size:</b> {training_config.get('training_args', {}).get('batch_size', 4)}</li>
            <li><b>Learning rate:</b> {training_config.get('training_args', {}).get('learning_rate', '2e-4')}</li>
            <li><b>Эпохи:</b> {training_config.get('training_args', {}).get('num_epochs', 3)}</li>
            </ul>
            """
            
            if temp_created:
                self.plugin_manager.remove_plugin_instance(plugin_id)
        else:
            info_text = f"<h3>❌ Не удалось получить информацию о плагине {plugin_id}</h3>"
        
        # Показываем диалог с информацией
        info_dialog = QDialog(self)
        info_dialog.setWindowTitle(f"Информация о плагине: {plugin_info['name']}")
        info_dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(info_dialog)
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(info_label)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)
        
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(info_dialog.accept)
        layout.addWidget(close_button)
        
        info_dialog.exec()
    
    def install_plugin_from_file(self):
        """Устанавливает плагин из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Выберите файл плагина"),
            "",
            self.tr("Python файлы (*.py);;Все файлы (*)")
        )
        
        if file_path:
            try:
                if self.plugin_manager.install_plugin_from_file(file_path):
                    QMessageBox.information(self, "Успех", self.tr("Плагин установлен успешно!"))
                    self.refresh_all()
                else:
                    QMessageBox.warning(self, "Ошибка", self.tr("Не удалось установить плагин"))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка установки плагина: {e}")
    
    def create_plugin_template(self):
        """Создает шаблон плагина"""
        plugin_name, ok = QInputDialog.getText(
            self, 
            self.tr("Создание шаблона плагина"),
            self.tr("Введите название плагина:")
        )
        
        if ok and plugin_name:
            try:
                template_path = self.plugin_manager.create_plugin_template(plugin_name)
                QMessageBox.information(
                    self, 
                    self.tr("Успех"), 
                    self.tr("Шаблон плагина создан:\n") + template_path + "\n\n" +
                    self.tr("Откройте файл в редакторе и реализуйте методы загрузки модели."))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка создания шаблона: {e}")
    
    def select_training_dataset(self):
        """Выбор датасета для обучения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Выберите датасет для обучения"),
            "",
            self.tr("JSON файлы (*.json);;Все файлы (*)")
        )
        
        if file_path:
            self.dataset_path_edit.setText(file_path)
            self.analyze_dataset(file_path)
    
    def analyze_dataset(self, dataset_path: str):
        """Анализирует выбранный датасет"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                count = len(data)
                # Анализируем структуру
                if data and isinstance(data[0], dict):
                    keys = list(data[0].keys())
                    info_text = f"""
                    <b>Датасет проанализирован:</b><br>
                    📊 Количество примеров: {count}<br>
                    🔑 Структура: {', '.join(keys)}<br>
                    📈 Рекомендуемые эпохи: {min(10, max(3, 20 // (count // 100 + 1)))}<br>
                    ⏱️ Примерное время обучения: {self.estimate_training_time(count)} минут
                    """
                else:
                    info_text = f"❌ Неподдерживаемая структура датасета"
            else:
                info_text = f"❌ Датасет должен быть списком примеров"
            
            self.dataset_info_label.setText(info_text)
            
        except Exception as e:
            self.dataset_info_label.setText(f"❌ Ошибка анализа датасета: {e}")
    
    def estimate_training_time(self, dataset_size: int) -> int:
        """Оценивает время обучения в минутах"""
        plugin_id = self.training_plugin_combo.currentData()
        if plugin_id and "70b" in plugin_id:
            return dataset_size * 2  # 2 минуты на пример для больших моделей
        elif plugin_id and "13b" in plugin_id:
            return dataset_size // 2  # 30 секунд на пример
        else:
            return dataset_size // 5  # 12 секунд на пример для 7B моделей
    
    def start_training(self):
        """Запускает обучение модели"""
        plugin_id = self.training_plugin_combo.currentData()
        dataset_path = self.dataset_path_edit.text()
        
        if not plugin_id:
            QMessageBox.warning(self, "Ошибка", self.tr("Выберите плагин для обучения"))
            return
        
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", self.tr("Выберите корректный датасет"))
            return
        
        try:
            # Получаем экземпляр плагина
            plugin_instance = self.plugin_manager.get_plugin_instance(plugin_id)
            if not plugin_instance:
                plugin_instance = self.plugin_manager.create_plugin_instance(plugin_id)
            
            if not plugin_instance:
                QMessageBox.critical(self, "Ошибка", self.tr("Не удалось создать экземпляр плагина"))
                return
            
            # Создаем тренер
            self.current_trainer = LLMTrainer(plugin_instance, self.training_progress_callback)
            
            # Подготавливаем конфигурацию обучения
            training_config = plugin_instance.get_training_config()
            training_config["training_args"].update({
                "num_epochs": self.epochs_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "learning_rate": float(self.learning_rate_edit.text()),
            })
            training_config["default_lora_rank"] = self.lora_rank_spin.value()
            
            # Создаем директорию для сохранения
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("data", "trained_models", f"{plugin_id}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Запускаем обучение в отдельном потоке
            self.training_thread = TrainingThread(
                self.current_trainer, 
                dataset_path, 
                output_dir, 
                training_config
            )
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            
            # Обновляем UI
            self.start_training_button.setEnabled(False)
            self.stop_training_button.setEnabled(True)
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            self.training_status_label.setText("🚀 Инициализация обучения...")
            
            # Запускаем поток и таймер
            self.training_thread.start()
            self.progress_timer.start(1000)  # Обновление каждую секунду
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка запуска обучения: {e}")
    
    def training_progress_callback(self, progress: int, message: str):
        """Callback для обновления прогресса обучения"""
        if progress >= 0:
            self.training_progress.setValue(progress)
        
        if message:
            if progress == -1:  # Лог сообщение
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {message}")
            else:
                self.training_status_label.setText(message)
    
    def update_training_progress(self):
        """Обновляет прогресс обучения"""
        if self.current_trainer:
            metrics = self.current_trainer.get_training_metrics()
            
            # Обновляем метрики на вкладке мониторинга
            self.current_epoch_label.setText(str(metrics.epoch))
            self.current_step_label.setText(str(metrics.step))
            self.current_loss_label.setText(f"{metrics.loss:.4f}")
            self.current_lr_label.setText(f"{metrics.learning_rate:.2e}")
            
            # Обновляем таблицу истории
            self.update_history_table()
    
    def update_history_table(self):
        """Обновляет таблицу истории обучения"""
        if not self.current_trainer:
            return
        
        history = self.current_trainer.get_training_history()
        self.history_table.setRowCount(len(history))
        
        for i, entry in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(entry.get("epoch", 0))))
            self.history_table.setItem(i, 1, QTableWidgetItem(str(entry.get("step", 0))))
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{entry.get('loss', 0):.4f}"))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{entry.get('eval_loss', 0):.4f}"))
            
            timestamp = entry.get("timestamp", "")
            if timestamp:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                self.history_table.setItem(i, 4, QTableWidgetItem(time_str))
        
        # Прокручиваем к последней записи
        if history:
            self.history_table.scrollToBottom()
    
    def stop_training(self):
        """Останавливает обучение"""
        if self.current_trainer:
            self.current_trainer.stop_training()
            self.training_status_label.setText("🛑 Остановка обучения...")
            self.stop_training_button.setEnabled(False)
    
    def on_training_finished(self, success: bool, output_path: str):
        """Обработчик завершения обучения"""
        self.progress_timer.stop()
        
        # Восстанавливаем UI
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.training_progress.setVisible(False)
        
        if success:
            self.training_status_label.setText(f"✅ Обучение завершено: {output_path}")
            QMessageBox.information(
                self, 
                self.tr("Успех"), 
                self.tr("Модель обучена успешно!\n\nСохранена в: ") + output_path)
        else:
            self.training_status_label.setText("❌ Обучение прервано с ошибкой")
        
        self.current_trainer = None
        self.training_thread = None
    
    def on_training_error(self, error_message: str):
        """Обработчик ошибки обучения"""
        self.progress_timer.stop()
        
        # Восстанавливаем UI
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.training_progress.setVisible(False)
        
        self.training_status_label.setText(f"❌ Ошибка: {error_message}")
        QMessageBox.critical(self, self.tr("Ошибка обучения"), error_message)
        
        self.current_trainer = None
        self.training_thread = None
    
    def select_images_folder(self):
        """Выбор папки с изображениями для генерации датасета"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Выберите папку с изображениями счетов")
        )
        
        if folder_path:
            self.images_folder_edit.setText(folder_path)
            self.analyze_images_folder(folder_path)
    
    def analyze_images_folder(self, folder_path: str):
        """Анализирует папку с изображениями"""
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf'}
            files = []
            
            for file_name in os.listdir(folder_path):
                if any(file_name.lower().endswith(ext) for ext in image_extensions):
                    files.append(file_name)
            
            info_text = f"""
            <b>Анализ папки завершен:</b><br>
            📁 Путь: {folder_path}<br>
            📊 Найдено файлов: {len(files)}<br>
            🕒 Примерное время генерации: {len(files) * 2} минут<br>
            💾 Размер датасета: ~{len(files) * 1.5:.1f} МБ
            """
            
            self.images_info_label.setText(info_text)
            
            # Устанавливаем название датасета по умолчанию
            if not self.output_dataset_edit.text():
                folder_name = os.path.basename(folder_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dataset_edit.setText(f"dataset_{folder_name}_{timestamp}")
                
        except Exception as e:
            self.images_info_label.setText(f"❌ Ошибка анализа папки: {e}")
    
    def start_dataset_generation(self):
        """Запускает генерацию датасета"""
        folder_path = self.images_folder_edit.text()
        dataset_name = self.output_dataset_edit.text()
        
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "Ошибка", self.tr("Выберите корректную папку с изображениями"))
            return
        
        if not dataset_name:
            QMessageBox.warning(self, "Ошибка", self.tr("Введите название датасета"))
            return
        
        # Проверяем наличие Gemini процессора
        if not self.use_gemini_checkbox.isChecked():
            QMessageBox.warning(self, "Ошибка", self.tr("В данной версии поддерживается только генерация с Gemini"))
            return
        
        try:
            # Получаем Gemini процессор из главного окна
            if self.parent_window and hasattr(self.parent_window, 'model_manager'):
                gemini_processor = self.parent_window.model_manager.get_gemini_processor()
                if not gemini_processor:
                    QMessageBox.warning(
                        self, 
                        self.tr("Ошибка"), 
                        self.tr("Gemini процессор недоступен. Проверьте настройки API ключа."))
                    return
            else:
                QMessageBox.critical(self, "Ошибка", self.tr("Не удалось получить доступ к Gemini процессору"))
                return
            
            # Создаем тренер для генерации датасета
            temp_plugin = self.plugin_manager.create_plugin_instance("llama")  # Используем любой доступный
            if not temp_plugin:
                QMessageBox.critical(self, "Ошибка", self.tr("Не удалось создать временный плагин"))
                return
            
            trainer = LLMTrainer(temp_plugin, self.generation_progress_callback)
            
            # Собираем пути к изображениям
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf'}
            image_paths = []
            
            for file_name in os.listdir(folder_path):
                if any(file_name.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(folder_path, file_name))
            
            # Создаем выходную директорию
            output_path = os.path.join("data", "training_datasets", dataset_name)
            os.makedirs(output_path, exist_ok=True)
            
            # Запускаем генерацию в отдельном потоке
            self.generation_thread = DatasetGenerationThread(
                trainer, image_paths, gemini_processor, output_path
            )
            self.generation_thread.finished.connect(self.on_generation_finished)
            self.generation_thread.error.connect(self.on_generation_error)
            self.generation_thread.log_message.connect(self.on_generation_log)
            
            # Обновляем UI
            self.start_generation_button.setEnabled(False)
            self.stop_generation_button.setEnabled(True)
            self.generation_progress.setVisible(True)
            self.generation_progress.setValue(0)
            self.generation_log.clear()
            self.generation_log.append("🚀 Начинаем генерацию датасета...")
            
            self.generation_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка запуска генерации: {e}")
    
    def generation_progress_callback(self, progress: int, message: str):
        """Callback для прогресса генерации датасета"""
        if progress >= 0:
            self.generation_progress.setValue(progress)
        
        if message:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.generation_log.append(f"[{timestamp}] {message}")
            
            # Прокручиваем к концу
            scrollbar = self.generation_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def on_generation_log(self, message: str):
        """Обработчик логов генерации"""
        self.generation_log.append(message)
        scrollbar = self.generation_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_generation_finished(self, dataset_path: str):
        """Обработчик завершения генерации датасета"""
        # Восстанавливаем UI
        self.start_generation_button.setEnabled(True)
        self.stop_generation_button.setEnabled(False)
        self.generation_progress.setVisible(False)
        
        self.generation_log.append(f"✅ Датасет создан: {dataset_path}")
        
        QMessageBox.information(
            self,
            self.tr("Успех"),
            self.tr("Датасет создан успешно!\n\n") + dataset_path + "\n\n" +
            self.tr("Теперь вы можете использовать его для обучения."))
    
    def on_generation_error(self, error_message: str):
        """Обработчик ошибки генерации"""
        # Восстанавливаем UI
        self.start_generation_button.setEnabled(True)
        self.stop_generation_button.setEnabled(False)
        self.generation_progress.setVisible(False)
        
        self.generation_log.append(f"❌ Ошибка: {error_message}")
        QMessageBox.critical(self, self.tr("Ошибка генерации"), error_message)
    
    def stop_dataset_generation(self):
        """Останавливает генерацию датасета"""
        # TODO: Реализовать остановку генерации
        self.generation_log.append("🛑 Запрос на остановку генерации...")
    
    def refresh_all(self):
        """Обновляет всю информацию"""
        self.plugin_manager.reload_plugins()
        self.load_plugin_info()
        QMessageBox.information(self, self.tr("Обновлено"), self.tr("Информация о плагинах обновлена"))


class TrainingThread(QThread):
    """Поток для обучения модели"""
    finished = pyqtSignal(bool, str)
    error = pyqtSignal(str)
    
    def __init__(self, trainer, dataset_path, output_dir, training_config):
        super().__init__()
        self.trainer = trainer
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.training_config = training_config
        self._should_stop = False  # Флаг для корректной остановки
    
    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.trainer = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке TrainingThread: {e}")
    
    def run(self):
        try:
            success = self.trainer.train_model(
                self.dataset_path,
                self.output_dir,
                self.training_config
            )
            
            if success:
                self.finished.emit(True, self.output_dir)
            else:
                self.error.emit("Обучение завершилось неуспешно")
                
        except Exception as e:
            self.error.emit(str(e))


class DatasetGenerationThread(QThread):
    """Поток для генерации датасета"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, trainer, image_paths, gemini_processor, output_path):
        super().__init__()
        self.trainer = trainer
        self.image_paths = image_paths
        self.gemini_processor = gemini_processor
        self.output_path = output_path
        self._should_stop = False  # Флаг для корректной остановки
    
    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.trainer = None
            self.gemini_processor = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке DatasetGenerationThread: {e}")
    
    def run(self):
        try:
            dataset_path = self.trainer.prepare_dataset_with_gemini(
                self.image_paths,
                self.gemini_processor,
                self.output_path
            )
            self.finished.emit(dataset_path)
            
        except Exception as e:
            self.error.emit(str(e)) 
"""
Мастер первого запуска InvoiceGemini
Помогает новым пользователям настроить приложение и начать работу
"""
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QCheckBox, QLineEdit, QTextEdit,
    QGroupBox, QButtonGroup, QProgressBar, QWidget, QComboBox,
    QFrame, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QMovie

from ...settings_manager import settings_manager
from ... import config as app_config


class OnboardingWizard(QWizard):
    """Мастер первого запуска с пошаговой настройкой"""
    
    setup_completed = pyqtSignal(dict)  # Emits configuration chosen by user
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Добро пожаловать в InvoiceGemini!")
        self.setMinimumSize(900, 700)
        self.resize(1000, 750)
        
        # Стиль
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
        
        # Страницы
        self.addPage(WelcomePage(self))
        self.addPage(WorkspaceSelectionPage(self))
        self.addPage(ModelSetupPage(self))
        self.addPage(APIKeysPage(self))
        self.addPage(CompletionPage(self))
        
        # Применяем стиль
        self.setStyleSheet(self._get_stylesheet())
        
        # Подключаем сигналы
        self.finished.connect(self._on_finished)
    
    def _on_finished(self, result):
        """Обработка завершения мастера"""
        if result == QWizard.DialogCode.Accepted:
            # Собираем конфигурацию
            config = {
                'workspace_mode': self.field('workspace_mode'),
                'model_preference': self.field('model_preference'),
                'gpu_enabled': self.field('gpu_enabled'),
                'gemini_key_set': self.field('gemini_key_set'),
                'first_run_completed': True
            }
            
            # Сохраняем настройки
            settings_manager.set_value('General', 'first_run_completed', True)
            settings_manager.set_value('General', 'workspace_mode', config['workspace_mode'])
            
            # Отправляем сигнал
            self.setup_completed.emit(config)
    
    def _get_stylesheet(self):
        """Стилизация мастера"""
        return """
            QWizard {
                background-color: #f8f9fa;
            }
            
            QWizardPage {
                background-color: white;
            }
            
            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px 0;
            }
            
            QLabel#subtitle {
                font-size: 14px;
                color: #7f8c8d;
                padding-bottom: 20px;
            }
            
            QLabel#description {
                font-size: 12px;
                color: #5a6c7d;
                line-height: 1.6;
            }
            
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2c3e50;
            }
            
            QRadioButton, QCheckBox {
                font-size: 12px;
                padding: 8px;
            }
            
            QRadioButton::indicator, QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QRadioButton:hover, QCheckBox:hover {
                background-color: #f0f3f5;
                border-radius: 4px;
            }
            
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
            }
            
            QPushButton:hover {
                background-color: #2980b9;
            }
            
            QPushButton:pressed {
                background-color: #1c5985;
            }
        """


class WelcomePage(QWizardPage):
    """Страница приветствия"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")  # Убираем стандартный заголовок
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Добавляем отступ сверху
        layout.addSpacing(30)
        
        # Заголовок с emoji
        title = QLabel("🎉 Добро пожаловать в InvoiceGemini!")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Подзаголовок
        subtitle = QLabel("Автоматизация извлечения данных из счетов-фактур с помощью AI")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #e0e0e0;")
        layout.addWidget(line)
        
        layout.addSpacing(10)
        
        # Основное описание
        description = QLabel(
            "Этот мастер поможет вам:\n\n"
            "  ✓  Выбрать оптимальный режим работы\n"
            "  ✓  Настроить AI-модели для обработки\n"
            "  ✓  Подключить облачные сервисы (опционально)\n"
            "  ✓  Обработать первый тестовый документ\n\n"
            "Настройка займет всего 2-3 минуты."
        )
        description.setObjectName("description")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(description)
        
        layout.addSpacing(20)
        
        # Возможности приложения
        features_group = QGroupBox("🌟 Ключевые возможности")
        features_layout = QVBoxLayout()
        
        features = [
            "🤖 Множественные AI-модели: LayoutLMv3, Donut, Google Gemini",
            "⚡ Быстрая обработка: от 3 секунд на документ",
            "📊 Пакетная обработка: сотни документов одновременно",
            "🎯 Высокая точность: 90%+ распознавания",
            "🔗 Интеграции: Paperless-NGX, 1C ERP, облачные хранилища",
            "🎓 Обучение моделей: настройка под ваши документы"
        ]
        
        for feature in features:
            label = QLabel(feature)
            label.setStyleSheet("padding: 5px; font-size: 12px;")
            features_layout.addWidget(label)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Растягиваем
        layout.addStretch()
        
        # Чекбокс "Больше не показывать"
        skip_checkbox = QCheckBox("Не показывать этот мастер при следующем запуске")
        skip_checkbox.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-top: 20px;")
        layout.addWidget(skip_checkbox)


class WorkspaceSelectionPage(QWizardPage):
    """Страница выбора режима работы"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Выбор режима работы")
        self.setSubTitle("Выберите профиль, наиболее соответствующий вашим задачам")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Группа радиокнопок
        self.mode_group = QButtonGroup(self)
        
        # Режим 1: Бухгалтерия
        accounting_widget = self._create_mode_card(
            "🏢 Бухгалтерия",
            "Фокус на точности и валидации",
            [
                "• Автоматическая валидация всех полей",
                "• Экспорт в Excel для 1C",
                "• Проверка соответствия сумм и НДС",
                "• Использование LayoutLMv3 для точности"
            ],
            "accountant"
        )
        layout.addWidget(accounting_widget)
        
        # Режим 2: Массовая обработка
        batch_widget = self._create_mode_card(
            "📦 Массовая обработка",
            "Фокус на скорости и производительности",
            [
                "• Параллельная обработка нескольких документов",
                "• Использование быстрого Gemini API",
                "• Автоматический экспорт в JSON",
                "• Минимальные задержки"
            ],
            "batch"
        )
        layout.addWidget(batch_widget)
        
        # Режим 3: Универсальный
        universal_widget = self._create_mode_card(
            "🔬 Универсальный",
            "Все возможности и настройки",
            [
                "• Доступ ко всем моделям и функциям",
                "• Сравнение результатов разных моделей",
                "• Обучение собственных моделей",
                "• Максимальная гибкость"
            ],
            "universal"
        )
        layout.addWidget(universal_widget)
        
        layout.addStretch()
        
        # Регистрируем поле для получения значения
        # PyQt6 требует pyqtSignal, а не строку
        self.registerField('workspace_mode', self, 'selected_mode', self.modeChanged)
        self.selected_mode = 'accountant'
        
    modeChanged = pyqtSignal()
    
    def _create_mode_card(self, title, subtitle, features, mode_id):
        """Создать карточку режима"""
        card = QGroupBox()
        card_layout = QVBoxLayout()
        
        # Радиокнопка с заголовком
        radio = QRadioButton(title)
        radio.setStyleSheet("font-size: 14px; font-weight: 600;")
        if mode_id == 'accountant':
            radio.setChecked(True)
        
        radio.toggled.connect(lambda checked: self._on_mode_selected(mode_id) if checked else None)
        self.mode_group.addButton(radio)
        card_layout.addWidget(radio)
        
        # Подзаголовок
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-left: 25px;")
        card_layout.addWidget(subtitle_label)
        
        # Возможности
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("font-size: 11px; margin-left: 25px; padding: 2px;")
            card_layout.addWidget(feature_label)
        
        card.setLayout(card_layout)
        return card
    
    def _on_mode_selected(self, mode_id):
        """Обработка выбора режима"""
        self.selected_mode = mode_id
        self.modeChanged.emit()


class ModelSetupPage(QWizardPage):
    """Страница настройки моделей"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Настройка AI-моделей")
        self.setSubTitle("Выберите, какие модели вы хотите использовать")
        
        layout = QVBoxLayout(self)
        
        # Определение GPU
        import torch
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_info = QLabel(
                f"✅ Обнаружен GPU: {torch.cuda.get_device_name(0)}\n"
                f"Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            gpu_info.setStyleSheet(
                "background-color: #d4edda; color: #155724; padding: 12px; "
                "border-radius: 6px; border: 1px solid #c3e6cb;"
            )
        else:
            gpu_info = QLabel(
                "ℹ️ GPU не обнаружен. Рекомендуется использовать облачные модели (Gemini) "
                "для быстрой обработки."
            )
            gpu_info.setStyleSheet(
                "background-color: #d1ecf1; color: #0c5460; padding: 12px; "
                "border-radius: 6px; border: 1px solid #bee5eb;"
            )
        
        gpu_info.setWordWrap(True)
        layout.addWidget(gpu_info)
        
        layout.addSpacing(15)
        
        # Рекомендация
        recommendation_group = QGroupBox("💡 Рекомендуемая конфигурация")
        rec_layout = QVBoxLayout()
        
        if gpu_available:
            rec_text = (
                "У вас есть GPU, поэтому мы рекомендуем:\n\n"
                "• LayoutLMv3 - для максимальной точности (локально)\n"
                "• Google Gemini - для быстрой обработки (облако)\n\n"
                "Это даст вам лучшее соотношение скорости и точности."
            )
        else:
            rec_text = (
                "Без GPU рекомендуем использовать:\n\n"
                "• Google Gemini - быстрая облачная обработка\n"
                "• Donut - для офлайн режима (медленнее)\n\n"
                "Локальные модели будут работать на CPU (значительно медленнее)."
            )
        
        rec_label = QLabel(rec_text)
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet("font-size: 12px;")
        rec_layout.addWidget(rec_label)
        
        recommendation_group.setLayout(rec_layout)
        layout.addWidget(recommendation_group)
        
        layout.addSpacing(10)
        
        # Опции моделей
        models_group = QGroupBox("Выберите модели")
        models_layout = QVBoxLayout()
        
        self.layoutlm_check = QCheckBox("🎯 LayoutLMv3 - Максимальная точность (требует ~2GB)")
        self.layoutlm_check.setChecked(gpu_available)
        models_layout.addWidget(self.layoutlm_check)
        
        self.donut_check = QCheckBox("🍩 Donut - Универсальная модель (требует ~1.5GB)")
        self.donut_check.setChecked(False)
        models_layout.addWidget(self.donut_check)
        
        self.gemini_check = QCheckBox("💎 Google Gemini - Облачная модель (требует API ключ)")
        self.gemini_check.setChecked(True)
        models_layout.addWidget(self.gemini_check)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        layout.addSpacing(10)
        
        # Опция автозагрузки
        self.autodownload_check = QCheckBox("Автоматически загрузить выбранные модели сейчас")
        self.autodownload_check.setChecked(False)
        self.autodownload_check.setToolTip(
            "Если отключено, модели будут загружены при первом использовании"
        )
        layout.addWidget(self.autodownload_check)
        
        layout.addStretch()
        
        # Регистрируем поля
        self.registerField('gpu_enabled', self.layoutlm_check)
        # Добавляем свойство для model_preference
        self.preferred_model = 'gemini' if not gpu_available else 'layoutlm'
        
    
    def validatePage(self):
        """Валидация перед переходом"""
        if not (self.layoutlm_check.isChecked() or 
                self.donut_check.isChecked() or 
                self.gemini_check.isChecked()):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Выбор модели",
                "Пожалуйста, выберите хотя бы одну модель для работы."
            )
            return False
        
        return True


class APIKeysPage(QWizardPage):
    """Страница настройки API ключей"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("API ключи (опционально)")
        self.setSubTitle("Настройте API ключи для облачных сервисов")
        
        layout = QVBoxLayout(self)
        
        # Информация
        info = QLabel(
            "API ключи требуются только для облачных моделей. "
            "Вы можете пропустить этот шаг и настроить их позже в настройках."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #7f8c8d; padding: 10px;")
        layout.addWidget(info)
        
        # Google Gemini
        gemini_group = QGroupBox("💎 Google Gemini API")
        gemini_layout = QVBoxLayout()
        
        gemini_desc = QLabel(
            "Для использования Google Gemini необходим API ключ.\n"
            "Получить ключ можно на: https://makersuite.google.com/app/apikey"
        )
        gemini_desc.setWordWrap(True)
        gemini_desc.setStyleSheet("font-size: 11px; margin-bottom: 10px;")
        gemini_layout.addWidget(gemini_desc)
        
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setPlaceholderText("Введите Google API ключ...")
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        gemini_layout.addWidget(self.gemini_key_input)
        
        show_key_btn = QPushButton("👁️ Показать")
        show_key_btn.setMaximumWidth(100)
        show_key_btn.clicked.connect(self._toggle_key_visibility)
        gemini_layout.addWidget(show_key_btn)
        
        test_gemini_btn = QPushButton("🧪 Проверить подключение")
        test_gemini_btn.clicked.connect(self._test_gemini_connection)
        gemini_layout.addWidget(test_gemini_btn)
        
        gemini_group.setLayout(gemini_layout)
        layout.addWidget(gemini_group)
        
        layout.addSpacing(10)
        
        # Другие API (скрыты по умолчанию)
        self.other_apis_check = QCheckBox("Настроить другие API (OpenAI, Anthropic, etc.)")
        self.other_apis_check.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.other_apis_check)
        
        layout.addStretch()
        
        # Кнопка "Пропустить"
        skip_label = QLabel("Вы можете настроить API ключи позже в Настройки → LLM Провайдеры")
        skip_label.setStyleSheet("font-size: 10px; color: #95a5a6; font-style: italic;")
        skip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(skip_label)
        
        # Регистрируем поля
        self.key_is_set = False
        # self.registerField('gemini_key_set', self)  # Упростим - не используем это поле
    
    def _toggle_key_visibility(self):
        """Переключить видимость ключа"""
        if self.gemini_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.sender().setText("🔒 Скрыть")
        else:
            self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.sender().setText("👁️ Показать")
    
    def _test_gemini_connection(self):
        """Проверить подключение к Gemini"""
        api_key = self.gemini_key_input.text().strip()
        
        if not api_key:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "API ключ", "Пожалуйста, введите API ключ.")
            return
        
        # TODO: Реальная проверка подключения
        from PyQt6.QtWidgets import QMessageBox
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content("Test")
            
            if response:
                # Сохраняем ключ
                from config.secrets import SecretsManager
                secrets = SecretsManager()
                secrets.set_secret('google_api_key', api_key)
                
                self.key_is_set = True
                
                QMessageBox.information(
                    self,
                    "Успех",
                    "✅ Подключение к Google Gemini успешно!\nAPI ключ сохранен."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не удалось получить ответ от API. Проверьте ключ."
                )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Ошибка подключения",
                f"Ошибка при проверке API ключа:\n{str(e)}"
            )


class CompletionPage(QWizardPage):
    """Страница завершения"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("🎊 Готово к работе!")
        self.setSubTitle("Настройка завершена успешно")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Сообщение о завершении
        completion_msg = QLabel(
            "Отличная работа! InvoiceGemini настроен и готов к использованию.\n\n"
            "Сейчас вы сможете:\n"
            "  ✓  Обработать свой первый документ\n"
            "  ✓  Изучить интерфейс с помощью интерактивного тура\n"
            "  ✓  Прочитать краткое руководство\n\n"
            "Нажмите 'Готово' чтобы начать!"
        )
        completion_msg.setWordWrap(True)
        completion_msg.setStyleSheet("font-size: 12px; line-height: 1.6;")
        layout.addWidget(completion_msg)
        
        # Опции после завершения
        options_group = QGroupBox("Что дальше?")
        options_layout = QVBoxLayout()
        
        self.show_tour_check = QCheckBox("🎓 Показать интерактивный тур по интерфейсу (рекомендуется)")
        self.show_tour_check.setChecked(True)
        options_layout.addWidget(self.show_tour_check)
        
        self.load_sample_check = QCheckBox("📄 Загрузить тестовый документ для пробной обработки")
        self.load_sample_check.setChecked(True)
        options_layout.addWidget(self.load_sample_check)
        
        self.open_docs_check = QCheckBox("📚 Открыть документацию в браузере")
        self.open_docs_check.setChecked(False)
        options_layout.addWidget(self.open_docs_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        layout.addSpacing(20)
        
        # Полезные ссылки
        links_group = QGroupBox("📌 Полезные ресурсы")
        links_layout = QVBoxLayout()
        
        links = [
            ("📖 Документация", "docs/README.md"),
            ("💬 GitHub Discussions", "https://github.com/yourusername/InvoiceGemini/discussions"),
            ("🐛 Сообщить об ошибке", "https://github.com/yourusername/InvoiceGemini/issues"),
            ("⭐ Поддержать проект", "https://github.com/yourusername/InvoiceGemini"),
        ]
        
        for title, link in links:
            link_label = QLabel(f'<a href="{link}" style="text-decoration: none;">{title}</a>')
            link_label.setOpenExternalLinks(True)
            link_label.setStyleSheet("padding: 5px;")
            links_layout.addWidget(link_label)
        
        links_group.setLayout(links_layout)
        layout.addWidget(links_group)
        
        layout.addStretch()
        
        # Благодарность
        thanks = QLabel(
            "Спасибо за выбор InvoiceGemini! 🙏\n"
            "Если у вас есть вопросы, мы всегда рады помочь."
        )
        thanks.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thanks.setStyleSheet("font-size: 11px; color: #7f8c8d; font-style: italic; margin-top: 20px;")
        layout.addWidget(thanks)


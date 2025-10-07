#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvoiceExtractorGUI - Приложение для извлечения данных из счетов-фактур.

Точка входа в приложение с улучшенным логированием и обработкой ошибок.
"""
import os
import sys
import logging
from pathlib import Path
import builtins

# Патч для безопасного вывода emoji на Windows
_original_print = builtins.print

def safe_print(*args, **kwargs):
    """Безопасный print с заменой emoji для Windows"""
    emoji_map = {
        '🚀': '[START]', '✅': '[OK]', '⚠': '[WARN]', '❌': '[ERROR]',
        '📊': '[STATS]', '🔍': '[SEARCH]', '⚡': '[FAST]', '🔧': '[CONFIG]',
        '🎯': '[TARGET]', '📂': '[FOLDER]', '💾': '[SAVE]', '🤖': '[AI]',
        '📄': '[DOC]', '📁': '[DIR]', '🔄': '[SYNC]', '📝': '[NOTE]',
        '🔐': '[SECURE]', '📈': '[CHART]', '🎨': '[DESIGN]', '🌐': '[WEB]',
        '🔌': '[PLUGIN]',
    }
    
    safe_args = []
    for arg in args:
        text = str(arg)
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        safe_args.append(text)
    
    try:
        _original_print(*safe_args, **kwargs)
    except UnicodeEncodeError:
        ascii_args = [arg.encode('ascii', 'replace').decode('ascii') for arg in safe_args]
        _original_print(*ascii_args, **kwargs)

# Заменяем встроенную функцию print
builtins.print = safe_print

# Setup HF_HUB_OFFLINE first (важно для корректной работы)
os.environ["HF_HUB_OFFLINE"] = "0"

# Setup UTF-8 encoding for Windows
if sys.platform == "win32":
    try:
        # Устанавливаем UTF-8 кодировку для stdout/stderr
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        
        # Пытаемся изменить кодировку консоли Windows на UTF-8
        import subprocess
        try:
            subprocess.run(['chcp', '65001'], capture_output=True, check=False)
        except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
            # Тихо игнорируем ошибки изменения кодировки консоли - не критично для работы
            pass
    except Exception as e:
        print(f"Не удалось настроить UTF-8 кодировку: {e}")

import huggingface_hub

# Устанавливаем переменную окружения для оффлайн-режима до импорта библиотек
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Добавляем директорию проекта в sys.path если необходимо
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtCore import Qt, QLocale, QTranslator

from app.main_window import MainWindow
from app import config
from app import utils
from app.settings_manager import settings_manager


class SafeFormatter(logging.Formatter):
    """Форматтер с безопасной обработкой emoji для Windows"""
    
    emoji_map = {
        '🚀': '[START]', '✅': '[OK]', '⚠': '[WARN]', '❌': '[ERROR]',
        '📊': '[STATS]', '🔍': '[SEARCH]', '⚡': '[FAST]', '🔧': '[CONFIG]',
        '🎯': '[TARGET]', '📂': '[FOLDER]', '💾': '[SAVE]', '🤖': '[AI]',
        '📄': '[DOC]', '📁': '[DIR]', '🔄': '[SYNC]', '📝': '[NOTE]',
        '🔐': '[SECURE]', '📈': '[CHART]', '🎨': '[DESIGN]', '🌐': '[WEB]',
        '🔌': '[PLUGIN]', '🧹': '[CLEAN]',
    }
    
    def format(self, record):
        # Форматируем запись
        message = super().format(record)
        # Заменяем emoji
        for emoji, replacement in self.emoji_map.items():
            message = message.replace(emoji, replacement)
        return message

def setup_logging():
    """Setup enhanced application logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters with safe emoji handling
    detailed_formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = SafeFormatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_dir / 'app.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger


def setup_portable_mode():
    """Настройка путей для портативного режима с улучшенным логированием."""
    logger = logging.getLogger(__name__)
    logger.info("Начинаем настройку портативного режима...")
    
    # Путь к папке с приложением
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Создаем папки для данных, если они не существуют
    data_dir = os.path.join(app_dir, "data")
    models_dir = os.path.join(data_dir, "models")
    temp_dir = os.path.join(data_dir, "temp")
    secrets_dir = os.path.join(data_dir, "secrets")
    
    for directory in [data_dir, models_dir, temp_dir, secrets_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Создана/проверена директория: {directory}")
    
    logger.info(f"Директория данных: {data_dir}")
    logger.info(f"Директория моделей: {models_dir}")
    logger.info(f"Директория временных файлов: {temp_dir}")
    
    # Проверка наличия настроек и загрузка настроек из файла
    if hasattr(settings_manager, 'load_settings'):
        logger.info("Загружаем сохранённые настройки...")
        settings_manager.load_settings()
    
    # Обновляем пути в config из настроек
    logger.info("Обновляем пути к инструментам...")
    try:
        config.update_paths_from_settings()
        logger.info(f"Poppler Path: {config.POPPLER_PATH}")
    except Exception as e:
        logger.warning(f"Ошибка обновления путей: {e}")
    
    # Настройка HF токена
    if config.HF_TOKEN:
        logger.info("Токен Hugging Face настроен")
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        # Просто установим переменную окружения, без проверки токена
    else:
        logger.info("Токен Hugging Face не найден")
    
    # Настройка Google API ключа с улучшенной обработкой ошибок
    google_api_configured = False
    
    # Пытаемся использовать зашифрованное хранение, если доступно
    try:
        from app.security.secrets_manager import SecretsManager
        
        secrets_manager = SecretsManager()
        google_api_key = secrets_manager.get_secret('google_api_key')
        
        if google_api_key:
            logger.info("Google API ключ получен из зашифрованного хранилища")
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            logger.info("Google Gemini API успешно инициализирован из зашифрованного хранилища")
            google_api_configured = True
    except ImportError:
        logger.debug("Модули безопасности недоступны, используем fallback")
    except Exception as e:
        logger.warning(f"Ошибка при работе с зашифрованным хранилищем: {e}")
    
    # Fallback к обычному API ключу
    if not google_api_configured and config.GOOGLE_API_KEY:
        logger.info("Google API ключ настроен из конфигурации")
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            logger.info("Google Gemini API успешно инициализирован из конфигурации")
        except ImportError:
            logger.warning("Библиотека google-genai не установлена, Gemini будет недоступен")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Gemini API: {e}")
    elif not google_api_configured:
        logger.info("Google API ключ не найден")
    
    # Настройка оффлайн-режима
    if config.OFFLINE_MODE:
        logger.info("Оффлайн-режим включен")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        logger.info("Оффлайн-режим выключен")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


def setup_application(app: QApplication):
    """Setup Qt application properties with enhanced logging."""
    logger = logging.getLogger(__name__)
    
    logger.info("Конфигурируем QApplication...")
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    app.setOrganizationName(config.ORGANIZATION_NAME)
    logger.info("QApplication сконфигурирован")
    
    # Настройка локализации (для будущего использования)
    logger.info("Настраиваем локализацию...")
    translator = QTranslator()
    locale = QLocale.system().name()
    if translator.load(f"translations/{locale}.qm"):
        app.installTranslator(translator)
        logger.info(f"Загружен перевод для {locale}")
    else:
        logger.info("Файл перевода не найден, используем по умолчанию")
    logger.info("Локализация настроена")
    
    # Устанавливаем стиль приложения
    logger.info("Устанавливаем стиль Fusion...")
    app.setStyle(QStyleFactory.create("Fusion"))
    logger.info("Стиль установлен")
    
    # Устанавливаем локаль по умолчанию
    logger.info("Устанавливаем русскую локаль...")
    locale = QLocale(QLocale.Language.Russian, QLocale.Country.Russia)
    QLocale.setDefault(locale)
    logger.info("Локаль установлена")


def create_main_window():
    """Create and configure main window with error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Создаем MainWindow...")
        main_window = MainWindow()
        logger.info("MainWindow создан успешно")
        
        # Применяем улучшения к таблице результатов
        if hasattr(main_window, 'results_table'):
            improve_table_display(main_window.results_table)
            logger.info("Применены улучшения отображения таблицы")
        
        # 🚀 АКТИВАЦИЯ ОПТИМИЗАЦИЙ
        logger.info("🚀 Активируем оптимизированные компоненты...")
        try:
            from app.core.optimization_integration import apply_optimizations_to_main_window
            
            success = apply_optimizations_to_main_window(main_window)
            if success:
                logger.info("✅ Все оптимизации успешно применены!")
                logger.info("   🔍 Preview Dialog v2.0 активирован")
                logger.info("   ⚡ Optimized File Processor активирован") 
                logger.info("   🔧 Unified Plugin Manager активирован")
                logger.info("   🎯 Performance Monitoring активирован")
            else:
                logger.warning("⚠️ Некоторые оптимизации не удалось применить")
                
        except ImportError as e:
            logger.warning(f"⚠️ Модули оптимизации недоступны: {e}")
            logger.info("Приложение будет работать в стандартном режиме")
        except Exception as e:
            logger.error(f"❌ Ошибка применения оптимизаций: {e}")
            logger.info("Приложение будет работать в стандартном режиме")
        
        return main_window
        
    except Exception as e:
        logger.error(f"Ошибка создания MainWindow: {e}", exc_info=True)
        raise


def improve_table_display(table):
    """Improve table column display and sizing."""
    try:
        from PyQt6.QtWidgets import QHeaderView
        
        header = table.horizontalHeader()
        
        # Улучшенная настройка размеров колонок
        column_count = table.columnCount()
        if column_count > 0:
            # Настройка специфичных колонок
            for i in range(column_count):
                header_item = table.horizontalHeaderItem(i)
                if header_item:
                    column_name = header_item.text().lower()
                    
                    # Узкие колонки для коротких полей
                    if any(word in column_name for word in ['№', 'number', 'дата', 'date', '%']):
                        header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
                        table.setColumnWidth(i, 120)
                    
                    # Средние колонки для числовых полей
                    elif any(word in column_name for word in ['сумма', 'amount', 'total', 'ндс', 'vat']):
                        header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                        table.setColumnWidth(i, 140)
                    
                    # Широкие колонки для текстовых полей
                    elif any(word in column_name for word in ['поставщик', 'supplier', 'sender', 'название', 'name', 'описание', 'description']):
                        header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
                    
                    # По умолчанию - интерактивный режим
                    else:
                        header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                        table.setColumnWidth(i, 100)
            
            # Настройка общих свойств таблицы
            table.setAlternatingRowColors(True)
            table.verticalHeader().setVisible(False)
            table.setWordWrap(True)
            
            # Автоматическая подгонка высоты строк под содержимое
            table.resizeRowsToContents()
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Ошибка улучшения отображения таблицы: {e}")


def main():
    """
    Точка входа в приложение с улучшенным логированием и обработкой ошибок.
    Создает экземпляр QApplication и запускает главное окно.
    """
    # Setup logging first
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Запускаем InvoiceExtractorGUI...")
    logger.info(f"HF_HUB_OFFLINE принудительно установлен в 0")
    
    try:
        # Настройка portable-режима
        logger.info("Начинаем настройку portable-режима...")
        setup_portable_mode()
        logger.info("Portable режим настроен")
        
        # Создаем экземпляр приложения
        logger.info("Создаем QApplication...")
        app = QApplication(sys.argv)
        logger.info("QApplication создан")
        
        # Настройка приложения
        setup_application(app)
        
        # Создаем и показываем главное окно
        main_window = create_main_window()
        
        logger.info("Показываем главное окно...")
        main_window.show()
        logger.info("Главное окно показано")
        
        # UX IMPROVEMENT: Проверка первого запуска и показ онбординга
        from app.settings_manager import settings_manager
        first_run = settings_manager.get_bool('General', 'first_run_completed', False)
        
        if not first_run:
            logger.info("Первый запуск приложения - показываем мастер онбординга")
            try:
                from app.ui.components.onboarding_wizard import OnboardingWizard
                
                wizard = OnboardingWizard(main_window)
                wizard.setup_completed.connect(main_window.on_onboarding_completed)
                
                # Показываем мастер с небольшой задержкой для корректного отображения
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(500, wizard.show)
                
                logger.info("Мастер онбординга инициализирован")
            except Exception as e:
                logger.error(f"Ошибка при запуске мастера онбординга: {e}", exc_info=True)
        
        logger.info("Запускаем цикл событий...")
        logger.info("=" * 60)
        
        # Запускаем цикл обработки событий
        exit_code = app.exec()
        logger.info("Приложение завершено")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Сбой запуска приложения: {e}", exc_info=True)
        # Также выводим в консоль для видимости
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
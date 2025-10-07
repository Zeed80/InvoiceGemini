#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback
import logging
import os
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

# Добавляем текущую директорию в путь
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

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

def setup_debug_logging():
    """Настраивает расширенное логирование для отладки."""
    
    # Создаем директорию для логов
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Создаем безопасный форматтер
    safe_formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Создаем обработчики с безопасным форматтером
    file_handler = logging.FileHandler(logs_dir / "debug_session.log", mode='w', encoding='utf-8')
    file_handler.setFormatter(safe_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(safe_formatter)
    
    # Настраиваем основной логгер
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    # Включаем отладку для важных модулей
    logging.getLogger('PyQt6').setLevel(logging.WARNING)  # Уменьшаем шум от PyQt
    logging.getLogger('app').setLevel(logging.DEBUG)
    logging.getLogger('app.main_window').setLevel(logging.DEBUG)
    logging.getLogger('app.processing_engine').setLevel(logging.DEBUG)
    logging.getLogger('app.gemini_processor').setLevel(logging.DEBUG)
    
    print("Отладочное логирование настроено")
    print(f"Логи сохраняются в: {logs_dir.absolute()}")

def check_dependencies():
    """Проверяет наличие всех необходимых зависимостей."""
    
    print("Проверка зависимостей...")
    
    missing_deps = []
    
    try:
        import PyQt6
        print("OK PyQt6 доступен")
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import PIL
        print("OK Pillow доступен")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import transformers
        print("OK transformers доступен")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import torch
        print("OK torch доступен")
        if torch.cuda.is_available():
            print(f"CUDA доступен: {torch.cuda.get_device_name()}")
        else:
            print("Используется CPU")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import pytesseract
        print("OK pytesseract доступен")
    except ImportError:
        missing_deps.append("pytesseract")
    
    if missing_deps:
        print(f"Отсутствуют зависимости: {', '.join(missing_deps)}")
        print("Установите их командой: pip install " + " ".join(missing_deps))
        return False
    
    print("Все зависимости в порядке")
    return True

def check_tesseract():
    """Проверяет доступность Tesseract OCR."""
    
    print("Проверка Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"OK Tesseract доступен: v{version}")
        return True
    except Exception as e:
        print(f"Проблема с Tesseract: {e}")
        print("Убедитесь, что Tesseract установлен и путь к нему указан в настройках")
        return False

def check_cloud_models_availability():
    """Проверяет доступность облачных моделей."""
    print("\nПроверка доступности облачных моделей:")
    
    try:
        # ИСПРАВЛЕНИЕ: Используем правильный импорт
        import app.config as app_config
        
        # Проверяем Google Gemini API ключ
        google_api_key = getattr(app_config, 'GOOGLE_API_KEY', None)
        if google_api_key and len(google_api_key.strip()) > 10:
            print("OK Google Gemini API ключ настроен")
        else:
            print("Google Gemini API ключ не настроен")
            print("   Настройте в: Настройки - LLM провайдеры - Google")
        
        # Проверяем доступность библиотеки Google Generative AI
        try:
            import google.generativeai as genai
            print("OK Google GenerativeAI библиотека установлена")
        except ImportError:
            print("Google GenerativeAI библиотека не установлена")
            print("   Установите: pip install google-generativeai")
        
        # Проверяем другие облачные провайдеры
        openai_key = getattr(app_config, 'OPENAI_API_KEY', None)
        if openai_key and len(openai_key.strip()) > 10:
            print("OK OpenAI API ключ настроен")
        else:
            print("OpenAI API ключ не настроен (опционально)")
            
        anthropic_key = getattr(app_config, 'ANTHROPIC_API_KEY', None)
        if anthropic_key and len(anthropic_key.strip()) > 10:
            print("OK Anthropic API ключ настроен")
        else:
            print("Anthropic API ключ не настроен (опционально)")
            
    except Exception as e:
        print(f"Ошибка проверки облачных моделей: {e}")

def check_processing_components():
    """Проверяет компоненты обработки."""
    print("\nПроверка компонентов обработки:")
    
    try:
        # Проверяем основные модули
        modules_to_check = [
            ('app.main_window', 'Главное окно'),
            ('app.processing_engine', 'Движок обработки'),
            ('app.gemini_processor', 'Gemini процессор'),
            ('app.threads', 'Потоки обработки'),
            ('app.plugins.universal_plugin_manager', 'Менеджер плагинов')
        ]
        
        for module_name, description in modules_to_check:
            try:
                __import__(module_name)
                print(f"OK {description}: импортирован")
            except ImportError as e:
                print(f"Ошибка: {description} не импортирован - {e}")
        
    except Exception as e:
        print(f"Ошибка проверки компонентов: {e}")

def main():
    """Основная функция запуска с отладкой."""
    print("=== Запуск InvoiceGemini в отладочном режиме ===")
    print("=" * 50)
    
    # Настраиваем логирование
    setup_debug_logging()
    
    # Проверяем компоненты
    check_cloud_models_availability()
    check_processing_components()
    
    print("\nНачинаем запуск приложения...")
    print("=" * 50)
    
    try:
        # Импортируем и запускаем приложение
        from app.main_window import MainWindow
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt, QTranslator
        
        # Создаем QApplication
        app = QApplication(sys.argv)
        # ИСПРАВЛЕНИЕ: В PyQt6 эти атрибуты устарели/удалены
        # app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)  # Устарело в PyQt6
        # app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)  # Устарело в PyQt6
        
        # Устанавливаем переводчик приложения до создания главного окна
        try:
            from app.settings_manager import settings_manager
            lang_code = settings_manager.get_string('Interface', 'language', 'ru')
            translator = QTranslator()
            translations_dir = os.path.join(os.path.dirname(__file__), 'translations')
            qm_path = os.path.join(translations_dir, f'invoicegemini_{lang_code}.qm')
            if translator.load(qm_path):
                app.installTranslator(translator)
                # Храним ссылку, чтобы переводчик не был удален сборщиком мусора
                setattr(app, '_invoice_translator', translator)
        except Exception as e:
            print(f"⚠️ Локализация при старте не установлена: {e}")
        
        # Создаем главное окно
        window = MainWindow()
        window.show()
        
        print("Приложение запущено успешно!")
        print("Следите за логами в реальном времени")
        print("При проблемах с облачными моделями проверьте настройки API ключей")
        
        # Запускаем основной цикл
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Критическая ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
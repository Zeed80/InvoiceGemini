#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvoiceExtractorGUI - Приложение для извлечения данных из счетов-фактур.

Точка входа в приложение.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
print(f"DEBUG main.py: Принудительно установлен HF_HUB_OFFLINE=0 в начале main.py.")

import sys
import huggingface_hub

# Настройка кодировки для Windows
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
        except:
            pass
    except Exception as e:
        print(f"Не удалось настроить UTF-8 кодировку: {e}")

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

# Настройка путей для portable-режима
def setup_portable_mode():
    """Настройка путей для портативного режима."""
    print("Настройка портативного режима...")
    # Путь к папке с приложением
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Создаем папки для данных, если они не существуют
    data_dir = os.path.join(app_dir, "data")
    models_dir = os.path.join(data_dir, "models")
    temp_dir = os.path.join(data_dir, "temp")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Директория данных: {data_dir}")
    print(f"Директория моделей: {models_dir}")
    print(f"Директория временных файлов: {temp_dir}")
    
    # Проверка наличия настроек и загрузка настроек из файла
    if hasattr(settings_manager, 'load_settings'):
        print("Загрузка сохранённых настроек...")
        settings_manager.load_settings()
    
    # Обновляем пути в config из настроек
    print("Обновление путей к инструментам...")
    try:
        config.update_paths_from_settings()
        print(f"Poppler Path: {config.POPPLER_PATH}")
    except Exception as e:
        print(f"Предупреждение: Ошибка обновления путей: {e}")
    
    # Настройка HF токена
    if config.HF_TOKEN:
        print("Токен Hugging Face настроен")
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        # Просто установим переменную окружения, без проверки токена
        # huggingface_hub.login(token=config.HF_TOKEN, add_to_git_credential=False)
    else:
        print("Токен Hugging Face не найден")
    
    # Настройка Google API ключа
    if config.GOOGLE_API_KEY:
        print("Google API ключ настроен")
        # Инициализируем Google Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            print("Google Gemini API успешно инициализирован")
        except ImportError:
            print("Библиотека google-genai не установлена, Gemini будет недоступен")
        except Exception as e:
            print(f"Ошибка при инициализации Gemini API: {e}")
    else:
        print("Google API ключ не найден")
    
    # Настройка оффлайн-режима
    if config.OFFLINE_MODE:
        print("Оффлайн-режим включен")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print("Оффлайн-режим выключен")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

def main():
    """
    Точка входа в приложение.
    Создает экземпляр QApplication и запускает главное окно.
    """
    print("DEBUG: Начинаем инициализацию приложения...")
    
    # Настройка portable-режима
    setup_portable_mode()
    print("DEBUG: Portable режим настроен")
    
    # Создаем экземпляр приложения
    print("DEBUG: Создаем QApplication...")
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    app.setOrganizationName(config.ORGANIZATION_NAME)
    print("DEBUG: QApplication создан")
    
    # Настройка локализации (для будущего использования)
    print("DEBUG: Настраиваем локализацию...")
    translator = QTranslator()
    locale = QLocale.system().name()
    if translator.load(f"translations/{locale}.qm"):
        app.installTranslator(translator)
    print("DEBUG: Локализация настроена")
    
    # Устанавливаем стиль приложения
    print("DEBUG: Устанавливаем стиль Fusion...")
    app.setStyle(QStyleFactory.create("Fusion"))
    print("DEBUG: Стиль установлен")
    
    # Устанавливаем локаль по умолчанию
    print("DEBUG: Устанавливаем русскую локаль...")
    locale = QLocale(QLocale.Language.Russian, QLocale.Country.Russia)
    QLocale.setDefault(locale)
    print("DEBUG: Локаль установлена")
    
    # Создаем и показываем главное окно
    print("DEBUG: Создаем MainWindow...")
    main_window = MainWindow()
    print("DEBUG: MainWindow создан")
    
    print("DEBUG: Показываем главное окно...")
    main_window.show()
    print("DEBUG: Главное окно показано")
    
    print("DEBUG: Запускаем цикл событий...")
    # Запускаем цикл обработки событий
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
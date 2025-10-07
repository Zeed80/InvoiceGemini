#!/usr/bin/env python3
"""
Скрипт для тестирования основного функционала InvoiceGemini
"""
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Тестирует основные импорты"""
    print("Тестирование импортов...")
    
    try:
        import PyQt6
        print("✅ PyQt6 установлен")
    except ImportError:
        print("❌ PyQt6 не установлен")
    
    try:
        import torch
        print(f"✅ PyTorch установлен (версия {torch.__version__})")
    except ImportError:
        print("❌ PyTorch не установлен")
    
    try:
        import transformers
        print(f"✅ Transformers установлен (версия {transformers.__version__})")
    except ImportError:
        print("❌ Transformers не установлен")
    
    try:
        import google.generativeai
        print("✅ Google Generative AI установлен")
    except ImportError:
        print("❌ Google Generative AI не установлен")
    
    try:
        import pytesseract
        print("✅ pytesseract установлен")
    except ImportError:
        print("❌ pytesseract не установлен")
    
    try:
        import cryptography
        print("✅ cryptography установлен")
    except ImportError:
        print("❌ cryptography не установлен")

def test_app_modules():
    """Тестирует модули приложения"""
    print("\nТестирование модулей приложения...")
    
    modules_to_test = [
        ("app.config", "Конфигурация"),
        ("app.utils", "Утилиты"),
        ("app.settings_manager", "Менеджер настроек"),
        ("app.processing_engine", "Движок обработки"),
        ("app.security.crypto_manager", "Крипто менеджер"),
        ("app.security.secrets_manager", "Менеджер секретов"),
        ("app.core.memory_manager", "Менеджер памяти"),
        ("app.core.resource_manager", "Менеджер ресурсов"),
        ("app.plugins.base_plugin", "Базовый плагин"),
        ("app.plugins.base_llm_plugin", "Базовый LLM плагин"),
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
        except ImportError as e:
            print(f"❌ {description} ({module_name}): {e}")

def test_security_features():
    """Тестирует функции безопасности"""
    print("\nТестирование функций безопасности...")
    
    try:
        from app.security.crypto_manager import CryptoManager
        crypto = CryptoManager()
        
        # Тест шифрования
        test_data = "Тестовый API ключ"
        encrypted = crypto.encrypt(test_data)
        decrypted = crypto.decrypt(encrypted)
        
        if decrypted == test_data:
            print("✅ Шифрование/дешифрование работает")
        else:
            print("❌ Ошибка шифрования/дешифрования")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования безопасности: {e}")

def test_memory_management():
    """Тестирует управление памятью"""
    print("\nТестирование управления памятью...")
    
    try:
        from app.core.memory_manager import get_memory_manager
        memory_mgr = get_memory_manager()
        
        # Проверка доступной памяти
        memory_info = memory_mgr.get_memory_info()
        print(f"  Доступно RAM: {memory_info.available_mb/1024:.2f} GB")
        print(f"  Всего RAM: {memory_info.total_mb/1024:.2f} GB")
        print(f"  Использовано: {memory_info.percent:.1f}%")
        
        if memory_info.gpu_available_mb is not None:
            print(f"  GPU доступен")
            print(f"  GPU память свободно: {memory_info.gpu_available_mb/1024:.2f} GB")
            print(f"  GPU память всего: {memory_info.gpu_total_mb/1024:.2f} GB")
        else:
            print("  GPU не обнаружен")
            
        # Проверка возможности загрузки модели
        can_load, message = memory_mgr.can_load_model("layoutlm")
        print(f"  Можно загрузить LayoutLM: {'✅' if can_load else '❌'} {message}")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования памяти: {e}")

def test_plugin_system():
    """Тестирует систему плагинов"""
    print("\nТестирование системы плагинов...")
    
    try:
        from app.plugins.base_plugin import BasePlugin, PluginMetadata, PluginType
        print("✅ Базовые классы плагинов загружены")
        
        # Проверка LLM провайдеров
        from app.plugins.base_llm_plugin import LLM_PROVIDERS
        print(f"  Доступно LLM провайдеров: {len(LLM_PROVIDERS)}")
        for provider_name, config in LLM_PROVIDERS.items():
            print(f"    - {config.display_name}: {len(config.models)} моделей")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования плагинов: {e}")

def test_ocr():
    """Тестирует OCR"""
    print("\nТестирование OCR...")
    
    try:
        from app.processing_engine import OCRProcessor
        
        if OCRProcessor.validate_tesseract():
            print("✅ Tesseract OCR установлен и доступен")
            
            # Проверка доступных языков
            languages = OCRProcessor.get_available_languages()
            print(f"  Доступно языков: {len(languages)}")
            if 'rus' in languages:
                print("  ✅ Русский язык доступен")
            if 'eng' in languages:
                print("  ✅ Английский язык доступен")
        else:
            print("❌ Tesseract OCR не найден")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования OCR: {e}")

def main():
    """Основная функция"""
    print("="*60)
    print("Тестирование функционала InvoiceGemini")
    print("="*60)
    
    test_imports()
    test_app_modules()
    test_security_features()
    test_memory_management()
    test_plugin_system()
    test_ocr()
    
    print("\n" + "="*60)
    print("Тестирование завершено")
    print("="*60)

if __name__ == "__main__":
    main() 
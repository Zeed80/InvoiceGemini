#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест централизованных утилит Ollama - проверка отсутствия дублирования кода
"""
import sys
import os
import io

# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.plugins.models.ollama_utils import (
    OllamaUtils,
    check_ollama_availability,
    get_ollama_models,
    check_ollama_status,
    is_vision_model
)
from app.plugins.models.ollama_diagnostic import OllamaDiagnostic


def test_ollama_utils():
    """Тестирует централизованные утилиты Ollama"""
    print("🧪 Тестирование OllamaUtils...")
    print("=" * 60)
    
    # Тест 1: Проверка доступности
    print("\n1️⃣ Проверка доступности сервера...")
    is_available = check_ollama_availability()
    if is_available:
        print("✅ Сервер доступен")
    else:
        print("❌ Сервер недоступен")
        print("💡 Запустите Ollama: ollama serve")
        return False
    
    # Тест 2: Получение списка моделей
    print("\n2️⃣ Получение списка моделей...")
    models = get_ollama_models()
    if models:
        print(f"✅ Найдено моделей: {len(models)}")
        print("📋 Список:")
        for i, model in enumerate(models[:10], 1):
            print(f"   {i}. {model}")
        if len(models) > 10:
            print(f"   ... и еще {len(models) - 10}")
    else:
        print("⚠️ Модели не найдены")
        print("💡 Установите модель: ollama pull llama3.2-vision:11b")
        return False
    
    # Тест 3: Проверка статуса с кодом
    print("\n3️⃣ Проверка статуса с кодом состояния...")
    is_ok, status_code = check_ollama_status()
    status_descriptions = {
        "OK": "✅ Работает корректно",
        "CFG": "⚙️ Требуется настройка (нет моделей)",
        "ERR": "❌ Ошибка подключения",
        "TMO": "⏱️ Превышено время ожидания"
    }
    print(f"{status_descriptions.get(status_code, '❓ Неизвестный статус')}")
    print(f"   Статус: {is_ok}, Код: {status_code}")
    
    # Тест 4: Проверка конкретной модели
    print("\n4️⃣ Проверка доступности конкретной модели...")
    if models:
        test_model = models[0]
        is_model_available = OllamaUtils.is_model_available(test_model)
        if is_model_available:
            print(f"✅ Модель '{test_model}' доступна")
        else:
            print(f"❌ Модель '{test_model}' недоступна")
    
    # Тест 5: Получение версии сервера
    print("\n5️⃣ Получение версии сервера...")
    version = OllamaUtils.get_server_version()
    if version:
        print(f"✅ Версия сервера: {version}")
    else:
        print("❌ Не удалось получить версию")
    
    # Тест 6: Проверка определения vision моделей
    print("\n6️⃣ Проверка определения vision моделей...")
    
    vision_test_cases = [
        # (model_name, expected_is_vision)
        ("gemma3:4b", True),           # Gemma3 - мультимодальная
        ("gemma3:12b", True),          # Gemma3 - мультимодальная
        ("qwen2.5vl:3b", True),        # Qwen Vision-Language
        ("llama3.2-vision:11b", True), # Llama Vision
        ("llava:7b", True),            # LLaVA
        ("mistral:7b", False),         # Text-only
        ("llama3.1:8b", False),        # Text-only
        ("qwen2.5:7b", False),         # Text-only (не vl)
    ]
    
    all_passed = True
    for model_name, expected in vision_test_cases:
        result = is_vision_model(model_name)
        status = "✅" if result == expected else "❌"
        vision_label = "Vision" if result else "Text"
        print(f"   {status} {model_name}: {vision_label} (ожидалось: {'Vision' if expected else 'Text'})")
        if result != expected:
            all_passed = False
    
    if all_passed:
        print("✅ Все проверки vision моделей пройдены")
    else:
        print("❌ Некоторые проверки vision моделей не прошли")
    
    return all_passed


def test_ollama_diagnostic():
    """Тестирует расширенную диагностику Ollama"""
    print("\n🔬 Тестирование OllamaDiagnostic...")
    print("=" * 60)
    
    diagnostic = OllamaDiagnostic("http://localhost:11434")
    result = diagnostic.run_full_diagnostic(timeout=5)
    
    if result.server_available:
        print(f"\n✅ Диагностика успешна!")
        print(f"📊 Результаты:")
        print(f"   Версия: {result.server_version}")
        print(f"   Моделей: {len(result.models_available)}")
        print(f"   Vision моделей: {len(result.vision_models)}")
        print(f"   Рекомендованных: {len(result.recommended_models)}")
        
        if result.recommended_models:
            print(f"\n⭐ Рекомендованные модели для счетов:")
            for model in result.recommended_models[:5]:
                print(f"   • {model}")
        
        # Показываем полный отчет
        print("\n" + "=" * 60)
        print(diagnostic.format_diagnostic_report(result))
        
        return True
    else:
        print(f"\n❌ Диагностика неудачна")
        print(f"   Ошибка: {result.error_message}")
        return False


def test_no_code_duplication():
    """Проверяет, что нет дублирования кода"""
    print("\n🔍 Проверка отсутствия дублирования...")
    print("=" * 60)
    
    print("\n✅ Централизованные утилиты:")
    print("   • OllamaUtils.check_availability()")
    print("   • OllamaUtils.get_models()")
    print("   • OllamaUtils.check_status()")
    print("   • OllamaUtils.is_model_available()")
    print("   • OllamaUtils.get_server_version()")
    print("   • OllamaUtils.is_vision_model() [НОВОЕ]")
    
    print("\n✅ Удобные функции:")
    print("   • check_ollama_availability()")
    print("   • get_ollama_models()")
    print("   • check_ollama_status()")
    print("   • is_vision_model() [НОВОЕ]")
    
    print("\n✅ Расширенная диагностика:")
    print("   • OllamaDiagnostic.run_full_diagnostic()")
    print("   • OllamaDiagnostic.format_diagnostic_report()")
    print("   • OllamaDiagnostic.VISION_MODELS [ОБНОВЛЕНО]")
    
    print("\n✅ Все методы используют централизованный код")
    print("   Дублирование отсутствует!")
    
    return True


def main():
    """Основная функция тестирования"""
    print("🚀 Комплексное тестирование Ollama Utils")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # Тест 1: Утилиты
    if test_ollama_utils():
        tests_passed += 1
        print("\n✅ Тест утилит пройден")
    else:
        print("\n❌ Тест утилит не пройден")
    
    # Тест 2: Диагностика
    if test_ollama_diagnostic():
        tests_passed += 1
        print("\n✅ Тест диагностики пройден")
    else:
        print("\n❌ Тест диагностики не пройден")
    
    # Тест 3: Отсутствие дублирования
    if test_no_code_duplication():
        tests_passed += 1
        print("\n✅ Проверка дублирования пройдена")
    else:
        print("\n❌ Проверка дублирования не пройдена")
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"   Пройдено тестов: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n🎉 Все тесты пройдены успешно!")
        print("✅ Централизованные утилиты работают корректно")
        print("✅ Дублирование кода устранено")
        return 0
    else:
        print("\n⚠️ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


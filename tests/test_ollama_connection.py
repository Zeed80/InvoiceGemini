#!/usr/bin/env python3
"""
Тестовый скрипт для проверки подключения к Ollama
"""

import requests
import sys
import time

def test_ollama_connection():
    """Тестирует подключение к Ollama серверу."""
    
    print("🔍 Тестирование подключения к Ollama...")
    print("=" * 50)
    
    base_url = "http://localhost:11434"
    
    # Тест 1: Проверка доступности сервера
    print("1️⃣ Проверка доступности сервера...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер Ollama доступен")
            
            # Получаем список моделей
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if available_models:
                print(f"📋 Найдено моделей: {len(available_models)}")
                print("   Доступные модели:")
                for model in available_models[:10]:  # Показываем первые 10
                    print(f"   - {model}")
                if len(available_models) > 10:
                    print(f"   ... и еще {len(available_models) - 10} моделей")
            else:
                print("❌ Модели не найдены")
                print("💡 Установите модель: ollama pull llama3.2-vision")
                return False
                
        else:
            print(f"❌ Сервер недоступен (код: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к серверу Ollama")
        print("💡 Убедитесь, что Ollama запущен: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Таймаут подключения")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    
    # Тест 2: Проверка генерации ответа
    print("\n2️⃣ Тестирование генерации ответа...")
    
    if available_models:
        test_model = available_models[0]  # Берем первую доступную модель
        print(f"🧪 Тестируем модель: {test_model}")
        
        test_data = {
            "model": test_model,
            "prompt": "Hello! Please respond with 'OK' if you can hear me.",
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.1
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                if response_text:
                    print(f"✅ Модель отвечает: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
                    return True
                else:
                    print("❌ Модель не дает ответа")
                    return False
            else:
                print(f"❌ Ошибка генерации (код: {response.status_code})")
                try:
                    error_data = response.json()
                    print(f"   Детали: {error_data.get('error', 'Неизвестная ошибка')}")
                except:
                    pass
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Таймаут при генерации ответа")
            return False
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return False
    
    return True

def test_universal_llm_plugin():
    """Тестирует UniversalLLMPlugin с Ollama."""
    
    print("\n3️⃣ Тестирование UniversalLLMPlugin...")
    
    try:
        # Импортируем плагин
        sys.path.append('.')
        from app.plugins.models.universal_llm_plugin import UniversalLLMPlugin
        
        # Сначала получаем список доступных моделей
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if available_models:
                # Предпочитаем более быстрые модели для тестирования
                preferred_models = ["gemma3:1b", "gemma3:4b", "qwen3:4b"]
                test_model = None
                
                for preferred in preferred_models:
                    if preferred in available_models:
                        test_model = preferred
                        break
                
                if not test_model:
                    test_model = available_models[0]  # Берем первую доступную
                    
                print(f"🎯 Используем модель: {test_model}")
                
                # Создаем плагин для Ollama
                plugin = UniversalLLMPlugin(
                    provider_name="ollama",
                    model_name=test_model
                )
            else:
                print("❌ Нет доступных моделей")
                return False
        else:
            print("❌ Не удалось получить список моделей")
            return False
        
        # Пытаемся загрузить
        print("🔧 Загружаем плагин...")
        success = plugin.load_model()
        
        if success:
            print("✅ UniversalLLMPlugin успешно загружен")
            
            # Тестируем генерацию
            print("🧪 Тестируем генерацию через плагин...")
            try:
                response = plugin.generate_response("Test message", timeout=30)
                if response and len(response.strip()) > 0:
                    print(f"✅ Плагин работает: '{response[:100]}{'...' if len(response) > 100 else ''}'")
                    return True
                else:
                    print("❌ Плагин не дает ответа")
                    return False
            except Exception as e:
                print(f"❌ Ошибка генерации через плагин: {e}")
                return False
        else:
            print("❌ Не удалось загрузить UniversalLLMPlugin")
            return False
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка плагина: {e}")
        return False

def main():
    """Основная функция тестирования."""
    
    print("🚀 Комплексное тестирование Ollama")
    print("=" * 50)
    
    # Тест базового подключения
    basic_test = test_ollama_connection()
    
    if basic_test:
        # Тест плагина
        plugin_test = test_universal_llm_plugin()
        
        print("\n" + "=" * 50)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print(f"   Базовое подключение: {'✅ OK' if basic_test else '❌ FAIL'}")
        print(f"   UniversalLLMPlugin: {'✅ OK' if plugin_test else '❌ FAIL'}")
        
        if basic_test and plugin_test:
            print("\n🎉 Все тесты пройдены! Ollama готов к работе.")
            return 0
        else:
            print("\n⚠️ Некоторые тесты не прошли.")
            return 1
    else:
        print("\n❌ Базовое подключение не работает.")
        print("\n🔧 Рекомендации по устранению:")
        print("   1. Запустите Ollama: ollama serve")
        print("   2. Установите модель: ollama pull llama3.2-vision")
        print("   3. Проверьте, что порт 11434 не заблокирован")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
#!/usr/bin/env python3
"""
Скрипт для инициализации промптов для всех моделей
"""

import sys
import os
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.settings_manager import SettingsManager
from app.prompt_generator import PromptGenerator

def main():
    """Главная функция для инициализации промптов"""
    print("🚀 Инициализация промптов для всех моделей...")
    
    try:
        # Инициализируем менеджеры
        settings_manager = SettingsManager()
        prompt_generator = PromptGenerator(settings_manager)
        
        # Создаем директорию для промптов
        prompts_dir = Path("data/prompts")
        prompts_dir.mkdir(exist_ok=True)
        print(f"📁 Директория промптов: {prompts_dir}")
        
        # Генерируем промпты для всех облачных провайдеров
        cloud_providers = ['openai', 'anthropic', 'google', 'mistral', 'deepseek', 'xai']
        for provider in cloud_providers:
            try:
                print(f"🔄 Генерирую промпт для {provider}...")
                prompt = prompt_generator.generate_cloud_llm_prompt(provider)
                success = prompt_generator.save_prompt_to_file(f"cloud_llm_{provider}", prompt)
                if success:
                    print(f"✅ Промпт для {provider} создан")
                else:
                    print(f"❌ Ошибка создания промпта для {provider}")
            except Exception as e:
                print(f"❌ Ошибка генерации промпта для {provider}: {e}")
        
        # Генерируем промпты для локальных провайдеров
        local_providers = ['ollama']
        for provider in local_providers:
            try:
                print(f"🔄 Генерирую промпт для {provider}...")
                prompt = prompt_generator.generate_local_llm_prompt(provider)
                success = prompt_generator.save_prompt_to_file(f"local_llm_{provider}", prompt)
                if success:
                    print(f"✅ Промпт для {provider} создан")
                else:
                    print(f"❌ Ошибка создания промпта для {provider}")
            except Exception as e:
                print(f"❌ Ошибка генерации промпта для {provider}: {e}")
        
        # Генерируем промпт для Gemini
        try:
            print("🔄 Генерирую промпт для Gemini...")
            prompt = prompt_generator.generate_gemini_prompt()
            success = prompt_generator.save_prompt_to_file("gemini", prompt)
            if success:
                print("✅ Промпт для Gemini создан")
            else:
                print("❌ Ошибка создания промпта для Gemini")
        except Exception as e:
            print(f"❌ Ошибка генерации промпта для Gemini: {e}")
        
        print("✅ Инициализация промптов завершена!")
        print(f"📂 Промпты сохранены в: {prompts_dir.absolute()}")
        
        # Показываем список созданных файлов
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.txt"))
            if prompt_files:
                print(f"\n📋 Созданные файлы промптов ({len(prompt_files)}):")
                for file in sorted(prompt_files):
                    print(f"  - {file.name}")
            else:
                print("⚠️ Файлы промптов не найдены")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
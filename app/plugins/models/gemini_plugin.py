"""
Плагин для Google Gemini API.
Интегрирует существующий GeminiProcessor с новой системой плагинов.
"""
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
from PIL import Image

from ..base_llm_plugin import BaseLLMPlugin

class GeminiPlugin(BaseLLMPlugin):
    """
    Специализированный плагин для Google Gemini API.
    Использует проверенный GeminiProcessor под капотом.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, **kwargs):
        """
        Инициализация плагина Google Gemini.
        
        Args:
            model_name: Название модели Gemini
            api_key: API ключ Google
            **kwargs: Дополнительные параметры
        """
        super().__init__("google", model_name, api_key, **kwargs)
        
        # Импортируем существующий GeminiProcessor
        try:
            from ...gemini_processor import GeminiProcessor
            self.gemini_processor = None
            self.GeminiProcessor = GeminiProcessor
        except ImportError:
            print("❌ Не удалось импортировать GeminiProcessor")
            # Для совместимости создаем заглушку
            self.GeminiProcessor = None
            self.gemini_processor = None
        
        print(f"🤖 Создан плагин Google Gemini для модели {self.model_name}")
    
    def load_model(self) -> bool:
        """
        Загружает модель Google Gemini.
        
        Returns:
            bool: True если загрузка успешна
        """
        if not self.GeminiProcessor:
            print("❌ GeminiProcessor недоступен")
            return False
        
        try:
            # Создаем экземпляр GeminiProcessor
            self.gemini_processor = self.GeminiProcessor()
            
            # Настраиваем модель, если указана
            if self.model_name:
                self.gemini_processor.model_id = self.model_name
            
            # Устанавливаем API ключ, если предоставлен
            if self.api_key:
                self.gemini_processor.api_key = self.api_key
            
            # Загружаем модель
            success = self.gemini_processor.load_model()
            if success:
                self.is_loaded = True
                self.client = self.gemini_processor
                print(f"✅ Google Gemini модель {self.model_name} загружена успешно")
                return True
            else:
                print(f"❌ Не удалось загрузить модель {self.model_name}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки Google Gemini: {e}")
            return False
    
    def generate_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """
        Генерирует ответ от Google Gemini.
        
        Args:
            prompt: Промпт для модели
            image_path: Путь к изображению (необязательно)
            image_context: Контекст изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded or not self.gemini_processor:
            raise RuntimeError("Модель Google Gemini не загружена")
        
        try:
            if image_path:
                # Используем существующий метод process для работы с изображениями
                result = self.gemini_processor.process(image_path)
                if result:
                    return json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    return "Не удалось обработать изображение"
            else:
                # Для текстовых запросов используем базовую функциональность
                try:
                    import google.generativeai as genai
                    response = self.gemini_processor.model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    print(f"❌ Ошибка генерации ответа: {e}")
                    return f"Ошибка: {str(e)}"
                    
        except Exception as e:
            print(f"❌ Ошибка в generate_response: {e}")
            return f"Ошибка: {str(e)}"
    
    def process_image(self, image_path: str, ocr_lang=None, custom_prompt=None) -> Optional[Dict]:
        """
        Обрабатывает изображение и извлекает данные инвойса.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR (не используется в Gemini)
            custom_prompt: Пользовательский промпт
            
        Returns:
            Optional[Dict]: Извлеченные данные или None
        """
        if not self.is_loaded or not self.gemini_processor:
            print("❌ Модель Google Gemini не загружена")
            return None
        
        try:
            # Используем метод process_image из существующего GeminiProcessor
            result = self.gemini_processor.process_image(image_path, ocr_lang, custom_prompt)
            return result
            
        except Exception as e:
            print(f"❌ Ошибка обработки изображения: {e}")
            return None
    
    def extract_invoice_data(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Извлекает данные из инвойса.
        
        Args:
            image_path: Путь к изображению инвойса
            prompt: Пользовательский промпт
            
        Returns:
            Dict[str, Any]: Извлеченные данные
        """
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        result = self.process_image(image_path, custom_prompt=prompt)
        return result if result else {}
    
    def get_model_info(self) -> dict:
        """
        Возвращает информацию о модели.
        
        Returns:
            dict: Информация о модели
        """
        info = super().get_model_info()
        info.update({
            "provider": "Google Gemini",
            "supports_vision": True,
            "supports_pdf": True,
            "max_tokens": 30720,
            "context_window": 1048576,
            "input_cost_per_1k": 0.075,  # USD per 1K tokens
            "output_cost_per_1k": 0.30   # USD per 1K tokens
        })
        
        if self.gemini_processor:
            info["processor_loaded"] = True
            info["temp_files_count"] = len(getattr(self.gemini_processor, 'temp_files', []))
        
        return info
    
    def cleanup(self):
        """Очищает ресурсы плагина."""
        if self.gemini_processor:
            # Очищаем временные файлы
            if hasattr(self.gemini_processor, 'cleanup_temp_files'):
                self.gemini_processor.cleanup_temp_files()
        
        self.is_loaded = False
        self.client = None
        self.gemini_processor = None
        print("🧹 Ресурсы плагина Google Gemini очищены")
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        try:
            self.cleanup()
        except (AttributeError, RuntimeError, Exception) as e:
            # Ошибка при очистке ресурсов в деструкторе - безопасно игнорируем
            pass 
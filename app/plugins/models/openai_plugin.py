"""
Плагин для OpenAI API (ChatGPT, GPT-4 Vision).
Поддерживает все модели OpenAI с возможностью анализа изображений.
"""
import os
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
from PIL import Image

from ..base_llm_plugin import BaseLLMPlugin

class OpenAIPlugin(BaseLLMPlugin):
    """
    Специализированный плагин для OpenAI API.
    Поддерживает ChatGPT и GPT-4 Vision для анализа документов.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, **kwargs):
        """
        Инициализация плагина OpenAI.
        
        Args:
            model_name: Название модели OpenAI
            api_key: API ключ OpenAI
            **kwargs: Дополнительные параметры
        """
        super().__init__("openai", model_name, api_key, **kwargs)
        
        # Специфичные настройки для OpenAI
        self.max_image_size = kwargs.get('max_image_size', 20 * 1024 * 1024)  # 20MB
        self.image_quality = kwargs.get('image_quality', 'high')  # high, low, auto
        
        print(f"🤖 Создан плагин OpenAI для модели {self.model_name}")
    
    def load_model(self) -> bool:
        """
        Загружает клиент OpenAI.
        
        Returns:
            bool: True если загрузка успешна
        """
        try:
            import openai
            
            if not self.api_key:
                print("❌ API ключ OpenAI не предоставлен")
                return False
            
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Проверяем доступность API
            try:
                # Простой тестовый запрос
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                
                self.is_loaded = True
                print(f"✅ OpenAI клиент инициализирован с моделью {self.model_name}")
                return True
                
            except Exception as e:
                print(f"❌ Ошибка тестирования API: {e}")
                return False
                
        except ImportError:
            print("❌ Библиотека openai не установлена. Установите: pip install openai")
            return False
        except Exception as e:
            print(f"❌ Ошибка инициализации OpenAI: {e}")
            return False
    
    def _encode_image_base64(self, image_path: str) -> str:
        """
        Кодирует изображение в base64.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            str: Base64 строка изображения
        """
        try:
            # Оптимизируем изображение, если оно слишком большое
            image = Image.open(image_path)
            
            # Конвертируем в RGB если необходимо
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Сжимаем изображение, если оно слишком большое
            max_size = (2048, 2048)  # Максимальный размер для GPT-4 Vision
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"📏 Изображение сжато до {image.size}")
            
            # Сохраняем в временный файл для кодирования
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image.save(temp_file.name, 'JPEG', quality=85)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"❌ Ошибка кодирования изображения: {e}")
            raise
    
    def generate_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """
        Генерирует ответ от OpenAI.
        
        Args:
            prompt: Промпт для модели
            image_path: Путь к изображению (необязательно)
            image_context: Контекст изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded:
            raise RuntimeError("Модель OpenAI не загружена")
        
        try:
            messages = []
            
            if image_path and os.path.exists(image_path):
                # Проверяем, поддерживает ли модель изображения
                vision_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"]
                if not any(vm in self.model_name for vm in vision_models):
                    return f"Модель {self.model_name} не поддерживает анализ изображений"
                
                # Кодируем изображение
                base64_image = self._encode_image_base64(image_path)
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": self.image_quality
                            }
                        }
                    ]
                })
            else:
                # Обычный текстовый запрос
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            # Отправляем запрос
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.generation_config.get("max_tokens", 4096),
                temperature=self.generation_config.get("temperature", 0.1),
                top_p=self.generation_config.get("top_p", 0.9)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Ошибка генерации ответа OpenAI: {e}")
            return f"Ошибка: {str(e)}"
    
    def process_image(self, image_path: str, ocr_lang=None, custom_prompt=None) -> Optional[Dict]:
        """
        Обрабатывает изображение и извлекает данные инвойса.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR (используется в промпте)
            custom_prompt: Пользовательский промпт
            
        Returns:
            Optional[Dict]: Извлеченные данные или None
        """
        if not self.is_loaded:
            print("❌ Модель OpenAI не загружена")
            return None
        
        try:
            # Используем пользовательский промпт или создаем базовый
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.create_invoice_prompt(custom_prompt, True)
            
            # Добавляем информацию о языке, если указан
            if ocr_lang:
                lang_info = f"\nЯзык документа: {ocr_lang}"
                prompt += lang_info
            
            # Генерируем ответ
            response = self.generate_response(prompt, image_path)
            
            # Парсим JSON ответ
            try:
                # Извлекаем JSON из ответа
                json_match = None
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    json_match = response[json_start:json_end].strip()
                elif '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_match = response[json_start:json_end]
                
                if json_match:
                    result = json.loads(json_match)
                    return self._normalize_invoice_data(result)
                else:
                    print("❌ JSON не найден в ответе")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка парсинга JSON: {e}")
                print(f"Ответ модели: {response}")
                return None
                
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
        
        # Информация о конкретных моделях OpenAI
        model_specs = {
            "gpt-4o": {
                "max_tokens": 4096,
                "context_window": 128000,
                "supports_vision": True,
                "input_cost_per_1k": 0.005,
                "output_cost_per_1k": 0.015
            },
            "gpt-4o-mini": {
                "max_tokens": 16384,
                "context_window": 128000,
                "supports_vision": True,
                "input_cost_per_1k": 0.00015,
                "output_cost_per_1k": 0.0006
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "context_window": 128000,
                "supports_vision": True,
                "input_cost_per_1k": 0.01,
                "output_cost_per_1k": 0.03
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "context_window": 16385,
                "supports_vision": False,
                "input_cost_per_1k": 0.0015,
                "output_cost_per_1k": 0.002
            }
        }
        
        # Находим спецификацию для текущей модели
        spec = None
        for model_key, model_spec in model_specs.items():
            if model_key in self.model_name:
                spec = model_spec
                break
        
        if spec:
            info.update(spec)
        
        info.update({
            "provider": "OpenAI",
            "supports_pdf": False,  # Требует предварительной конвертации
            "image_quality": self.image_quality,
            "max_image_size": self.max_image_size
        })
        
        return info
    
    def cleanup(self):
        """Очищает ресурсы плагина."""
        self.is_loaded = False
        self.client = None
        print("🧹 Ресурсы плагина OpenAI очищены")
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        try:
            self.cleanup()
        except:
            pass 
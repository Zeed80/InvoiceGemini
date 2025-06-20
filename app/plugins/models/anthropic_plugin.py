"""
Плагин для Anthropic API (Claude).
Поддерживает Claude 3.5 Sonnet, Haiku и другие модели с анализом изображений.
"""
import os
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
from PIL import Image

from ..base_llm_plugin import BaseLLMPlugin

class AnthropicPlugin(BaseLLMPlugin):
    """
    Специализированный плагин для Anthropic Claude API.
    Поддерживает Claude 3.5 и другие модели для анализа документов.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, **kwargs):
        """
        Инициализация плагина Anthropic.
        
        Args:
            model_name: Название модели Claude
            api_key: API ключ Anthropic
            **kwargs: Дополнительные параметры
        """
        super().__init__("anthropic", model_name, api_key, **kwargs)
        
        # Специфичные настройки для Anthropic
        self.max_image_size = kwargs.get('max_image_size', 5 * 1024 * 1024)  # 5MB
        self.max_images = kwargs.get('max_images', 20)  # До 20 изображений за запрос
        
        print(f"🤖 Создан плагин Anthropic Claude для модели {self.model_name}")
    
    def load_model(self) -> bool:
        """
        Загружает клиент Anthropic.
        
        Returns:
            bool: True если загрузка успешна
        """
        try:
            import anthropic
            
            if not self.api_key:
                print("❌ API ключ Anthropic не предоставлен")
                return False
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
            # Проверяем доступность API простым запросом
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                
                self.is_loaded = True
                print(f"✅ Anthropic клиент инициализирован с моделью {self.model_name}")
                return True
                
            except Exception as e:
                print(f"❌ Ошибка тестирования API: {e}")
                return False
                
        except ImportError:
            print("❌ Библиотека anthropic не установлена. Установите: pip install anthropic")
            return False
        except Exception as e:
            print(f"❌ Ошибка инициализации Anthropic: {e}")
            return False
    
    def _encode_image_base64(self, image_path: str) -> tuple:
        """
        Кодирует изображение в base64 для Anthropic API.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            tuple: (base64_string, media_type)
        """
        try:
            # Определяем тип изображения
            image = Image.open(image_path)
            
            # Конвертируем в RGB если необходимо
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Проверяем размер и сжимаем при необходимости
            max_size = (1568, 1568)  # Максимальный размер для Claude
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"📏 Изображение сжато до {image.size}")
            
            # Сохраняем в память и кодируем
            import io
            buffered = io.BytesIO()
            
            # Определяем формат для сохранения
            if image_path.lower().endswith('.png'):
                image.save(buffered, format="PNG")
                media_type = "image/png"
            else:
                image.save(buffered, format="JPEG", quality=85)
                media_type = "image/jpeg"
            
            img_bytes = buffered.getvalue()
            
            # Проверяем размер файла
            if len(img_bytes) > self.max_image_size:
                # Сжимаем еще больше
                buffered = io.BytesIO()
                if media_type == "image/png":
                    image.save(buffered, format="JPEG", quality=70)
                    media_type = "image/jpeg"
                else:
                    image.save(buffered, format="JPEG", quality=60)
                img_bytes = buffered.getvalue()
                print(f"📦 Изображение дополнительно сжато до {len(img_bytes)} байт")
            
            encoded = base64.b64encode(img_bytes).decode('utf-8')
            return encoded, media_type
            
        except Exception as e:
            print(f"❌ Ошибка кодирования изображения: {e}")
            raise
    
    def generate_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """
        Генерирует ответ от Claude.
        
        Args:
            prompt: Промпт для модели
            image_path: Путь к изображению (необязательно)
            image_context: Контекст изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded:
            raise RuntimeError("Модель Anthropic не загружена")
        
        try:
            # Формируем сообщение
            content = []
            
            if image_path and os.path.exists(image_path):
                # Проверяем, поддерживает ли модель изображения
                vision_models = ["claude-3", "claude-3.5"]
                if not any(vm in self.model_name for vm in vision_models):
                    return f"Модель {self.model_name} не поддерживает анализ изображений"
                
                # Кодируем изображение
                base64_image, media_type = self._encode_image_base64(image_path)
                
                # Добавляем изображение в контент
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                })
            
            # Добавляем текстовый промпт
            content.append({
                "type": "text",
                "text": prompt
            })
            
            # Отправляем запрос
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.generation_config.get("max_tokens", 4096),
                temperature=self.generation_config.get("temperature", 0.1),
                top_p=self.generation_config.get("top_p", 0.9),
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # Извлекаем текст ответа
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return "Пустой ответ от модели"
            
        except Exception as e:
            print(f"❌ Ошибка генерации ответа Anthropic: {e}")
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
            print("❌ Модель Anthropic не загружена")
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
            
            # Специальная инструкция для Claude
            claude_instruction = """
            
            Пожалуйста, проанализируй изображение документа и извлеки данные в точном JSON формате.
            Будь максимально точным и внимательным к деталям.
            """
            prompt += claude_instruction
            
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
                    print("❌ JSON не найден в ответе Claude")
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
        
        # Информация о конкретных моделях Claude
        model_specs = {
            "claude-3-5-sonnet": {
                "max_tokens": 8192,
                "context_window": 200000,
                "supports_vision": True,
                "input_cost_per_1k": 0.003,
                "output_cost_per_1k": 0.015
            },
            "claude-3-5-haiku": {
                "max_tokens": 8192,
                "context_window": 200000,
                "supports_vision": True,
                "input_cost_per_1k": 0.0008,
                "output_cost_per_1k": 0.004
            },
            "claude-3-opus": {
                "max_tokens": 4096,
                "context_window": 200000,
                "supports_vision": True,
                "input_cost_per_1k": 0.015,
                "output_cost_per_1k": 0.075
            },
            "claude-3-sonnet": {
                "max_tokens": 4096,
                "context_window": 200000,
                "supports_vision": True,
                "input_cost_per_1k": 0.003,
                "output_cost_per_1k": 0.015
            },
            "claude-3-haiku": {
                "max_tokens": 4096,
                "context_window": 200000,
                "supports_vision": True,
                "input_cost_per_1k": 0.00025,
                "output_cost_per_1k": 0.00125
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
            "provider": "Anthropic Claude",
            "supports_pdf": False,  # Требует предварительной конвертации
            "max_image_size": self.max_image_size,
            "max_images": self.max_images,
            "supports_system_prompt": True
        })
        
        return info
    
    def cleanup(self):
        """Очищает ресурсы плагина."""
        self.is_loaded = False
        self.client = None
        print("🧹 Ресурсы плагина Anthropic очищены")
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        try:
            self.cleanup()
        except (AttributeError, RuntimeError, Exception) as e:
            # Ошибка при очистке ресурсов в деструкторе - безопасно игнорируем
            pass 
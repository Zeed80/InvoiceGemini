"""
Универсальный LLM плагин для работы с различными провайдерами API.
Поддерживает OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, Ollama.
"""
import os
import json
import base64
import tempfile
import re
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
from PIL import Image

from ..base_llm_plugin import BaseLLMPlugin, LLM_PROVIDERS
from .adaptive_prompt_manager import (
    create_adaptive_invoice_prompt,
    get_model_generation_params,
    AdaptivePromptManager
)

class UniversalLLMPlugin(BaseLLMPlugin):
    """
    Универсальный плагин для работы с различными провайдерами LLM.
    Автоматически адаптируется к выбранному провайдеру и модели.
    """
    
    def __init__(self, provider_name: str, model_name: str = None, api_key: str = None, **kwargs):
        """
        Инициализация универсального LLM плагина.
        
        Args:
            provider_name: Название провайдера (openai, anthropic, google, etc.)
            model_name: Название модели (необязательно)
            api_key: API ключ для провайдера
            **kwargs: Дополнительные параметры
        """
        super().__init__(provider_name, model_name, api_key, **kwargs)
        
        # Специфичные настройки для каждого провайдера
        self.base_url = kwargs.get('base_url', None)
        self.temp_files = []
        
        # Настройки для Ollama
        if provider_name == "ollama":
            self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        
        print(f"✅ Создан универсальный LLM плагин для {self.provider_config.display_name}")
    
    def load_model(self) -> bool:
        """
        Инициализирует клиент для выбранного провайдера и проверяет подключение.
        
        Returns:
            bool: True если инициализация и проверка подключения успешны
        """
        try:
            success = False
            if self.provider_name == "openai":
                success = self._load_openai_client()
            elif self.provider_name == "anthropic":
                success = self._load_anthropic_client()
            elif self.provider_name == "google":
                success = self._load_google_client()
            elif self.provider_name == "mistral":
                success = self._load_mistral_client()
            elif self.provider_name == "deepseek":
                success = self._load_deepseek_client()
            elif self.provider_name == "xai":
                success = self._load_xai_client()
            elif self.provider_name == "ollama":
                success = self._load_ollama_client()
            else:
                print(f"❌ Неподдерживаемый провайдер: {self.provider_name}")
                return False
            
            # Если клиент создан, проверяем реальное подключение
            if success:
                print(f"🔍 Проверяем подключение к {self.provider_name}...")
                test_success = self._test_connection()
                if test_success:
                    print(f"✅ Подключение к {self.provider_name} проверено успешно")
                    self.is_loaded = True
                    return True
                else:
                    print(f"❌ Проверка подключения к {self.provider_name} неудачна")
                    self.is_loaded = False
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"❌ Ошибка инициализации {self.provider_name}: {e}")
            self.is_loaded = False
            return False
    
    def _load_openai_client(self) -> bool:
        """Инициализация OpenAI клиента."""
        try:
            import openai
            if not self.api_key:
                print("❌ API ключ OpenAI не предоставлен")
                return False
            
            self.client = openai.OpenAI(api_key=self.api_key)
            print(f"🔧 OpenAI клиент создан для модели {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека openai не установлена. Установите: pip install openai")
            return False
    
    def _load_anthropic_client(self) -> bool:
        """Инициализация Anthropic клиента."""
        try:
            import anthropic
            if not self.api_key:
                print("❌ API ключ Anthropic не предоставлен")
                return False
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print(f"🔧 Anthropic клиент создан для модели {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека anthropic не установлена. Установите: pip install anthropic")
            return False
    
    def _load_google_client(self) -> bool:
        """Инициализация Google Gemini клиента."""
        try:
            import google.generativeai as genai
            if not self.api_key:
                print("❌ API ключ Google не предоставлен")
                return False
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            print(f"🔧 Google Gemini клиент создан для модели {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека google-generativeai не установлена. Установите: pip install google-generativeai")
            return False
    
    def _load_mistral_client(self) -> bool:
        """Инициализация Mistral клиента."""
        try:
            from mistralai.client import MistralClient
            if not self.api_key:
                print("❌ API ключ Mistral не предоставлен")
                return False
            
            self.client = MistralClient(api_key=self.api_key)
            self.is_loaded = True
            print(f"✅ Mistral клиент инициализирован с моделью {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека mistralai не установлена. Установите: pip install mistralai")
            return False
    
    def _load_deepseek_client(self) -> bool:
        """Инициализация DeepSeek клиента (через OpenAI API)."""
        try:
            import openai
            if not self.api_key:
                print("❌ API ключ DeepSeek не предоставлен")
                return False
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            self.is_loaded = True
            print(f"✅ DeepSeek клиент инициализирован с моделью {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека openai не установлена. Установите: pip install openai")
            return False
    
    def _load_xai_client(self) -> bool:
        """Инициализация xAI клиента (через OpenAI API)."""
        try:
            import openai
            if not self.api_key:
                print("❌ API ключ xAI не предоставлен")
                return False
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
            self.is_loaded = True
            print(f"✅ xAI клиент инициализирован с моделью {self.model_name}")
            return True
        except ImportError:
            print("❌ Библиотека openai не установлена. Установите: pip install openai")
            return False
    
    def _load_ollama_client(self) -> bool:
        """Инициализация Ollama клиента."""
        try:
            import requests
            # Проверяем доступность Ollama сервера
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                # Проверяем, что модель доступна
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                if self.model_name in available_models:
                    self.client = True  # Для Ollama используем requests напрямую
                    print(f"🔧 Ollama клиент создан для модели {self.model_name}")
                    return True
                else:
                    print(f"❌ Модель {self.model_name} не найдена в Ollama")
                    print(f"📋 Доступные модели: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")
                    return False
            else:
                print(f"❌ Ollama сервер недоступен на {self.base_url} (код: {response.status_code})")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Не удается подключиться к Ollama серверу на {self.base_url}")
            print("💡 Убедитесь, что Ollama запущен: ollama serve")
            return False
        except requests.exceptions.Timeout:
            print(f"❌ Таймаут подключения к Ollama на {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Ошибка подключения к Ollama: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """
        Проверяет реальное подключение к API провайдера.
        
        Returns:
            bool: True если подключение работает
        """
        try:
            # Для Ollama используем более простой тест
            if self.provider_name == "ollama":
                return self._test_ollama_connection()
            
            # Для других провайдеров выполняем минимальный тестовый запрос
            test_response = self.generate_response("Test", timeout=10)
            
            # Проверяем, что получили валидный ответ
            if test_response and len(test_response.strip()) > 0:
                return True
            else:
                print(f"❌ {self.provider_name}: Получен пустой ответ")
                return False
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Обрабатываем специфичные ошибки
            if "timeout" in error_msg or "timed out" in error_msg:
                print(f"❌ {self.provider_name}: Превышено время ожидания")
            elif "unauthorized" in error_msg or "invalid api key" in error_msg:
                print(f"❌ {self.provider_name}: Неверный API ключ")
            elif "credit balance" in error_msg or "insufficient funds" in error_msg:
                print(f"❌ {self.provider_name}: Недостаточно средств на балансе")
            elif "rate limit" in error_msg:
                print(f"❌ {self.provider_name}: Превышен лимит запросов")
            elif "model not found" in error_msg:
                print(f"❌ {self.provider_name}: Модель {self.model_name} не найдена")
            elif "connection" in error_msg:
                print(f"❌ {self.provider_name}: Ошибка подключения")
            else:
                print(f"❌ {self.provider_name}: Ошибка подключения - {e}")
            
            return False
    
    def _test_ollama_connection(self) -> bool:
        """Специальный тест для Ollama - проверяем только доступность модели."""
        try:
            import requests
            
            # Проверяем, что сервер отвечает
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"❌ Ollama сервер недоступен (код: {response.status_code})")
                return False
            
            # Проверяем, что модель загружена
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if self.model_name not in available_models:
                print(f"❌ Модель {self.model_name} не найдена в Ollama")
                return False
            
            # Проверяем, что модель может генерировать ответы
            test_data = {
                "model": self.model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 10}
            }
            
            test_response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_data,
                timeout=15
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                if result.get("response"):
                    return True
                else:
                    print(f"❌ Ollama модель {self.model_name} не отвечает")
                    return False
            else:
                print(f"❌ Ошибка тестирования Ollama модели: {test_response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Не удается подключиться к Ollama серверу")
            return False
        except requests.exceptions.Timeout:
            print(f"❌ Таймаут при тестировании Ollama")
            return False
        except Exception as e:
            print(f"❌ Ошибка тестирования Ollama: {e}")
            return False
    
    def generate_response(self, prompt: str, image_path: str = None, image_context: str = "", timeout: int = 30) -> str:
        """
        Генерирует ответ от выбранного провайдера.
        
        Args:
            prompt: Промпт для модели
            image_path: Путь к изображению (если поддерживается)
            image_context: Контекст изображения (для провайдеров без vision)
            timeout: Таймаут запроса в секундах
            
        Returns:
            str: Ответ модели
        """
        if not self.client:
            raise ValueError(f"Клиент {self.provider_name} не инициализирован")
        
        try:
            if self.provider_name == "openai":
                return self._generate_openai_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "anthropic":
                return self._generate_anthropic_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "google":
                return self._generate_google_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "mistral":
                return self._generate_mistral_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "deepseek":
                return self._generate_deepseek_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "xai":
                return self._generate_xai_response(prompt, image_path, image_context, timeout)
            elif self.provider_name == "ollama":
                return self._generate_ollama_response(prompt, image_path, image_context, timeout)
            else:
                raise ValueError(f"Неподдерживаемый провайдер: {self.provider_name}")
                
        except Exception as e:
            # Поднимаем исключение для правильной обработки в _test_connection
            raise e
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Кодирует изображение в base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Ошибка кодирования изображения: {e}")
            return ""
    
    def _generate_openai_response(self, prompt: str, image_path: str = None, image_context: str = "", timeout: int = 30) -> str:
        """Генерация ответа через OpenAI API."""
        messages = []
        
        if image_path and self.provider_config.supports_vision:
            # Поддержка vision
            base64_image = self._encode_image_base64(image_path)
            if base64_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": f"{prompt}\n\nКонтекст изображения: {image_context}"})
        else:
            # Текстовый режим
            content = prompt
            if image_context:
                content += f"\n\nКонтекст изображения: {image_context}"
            messages.append({"role": "user", "content": content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.generation_config.get("max_tokens", 4096),
            temperature=self.generation_config.get("temperature", 0.1),
            timeout=timeout
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic_response(self, prompt: str, image_path: str = None, image_context: str = "", timeout: int = 30) -> str:
        """Генерация ответа через Anthropic API."""
        content = []
        
        if image_path and self.provider_config.supports_vision:
            # Поддержка vision
            try:
                import base64
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode()
                
                # Определяем тип изображения
                image_type = "image/jpeg"
                if image_path.lower().endswith('.png'):
                    image_type = "image/png"
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type,
                        "data": image_data
                    }
                })
            except Exception as e:
                print(f"❌ Ошибка обработки изображения для Claude: {e}")
        
        # Добавляем текст
        text_content = prompt
        if image_context and not (image_path and self.provider_config.supports_vision):
            text_content += f"\n\nКонтекст изображения: {image_context}"
        
        content.append({"type": "text", "text": text_content})
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.generation_config.get("max_tokens", 4096),
            temperature=self.generation_config.get("temperature", 0.1),
            messages=[{"role": "user", "content": content}],
            timeout=timeout
        )
        
        return response.content[0].text
    
    def _generate_google_response(self, prompt: str, image_path: str = None, image_context: str = "", timeout: int = 30) -> str:
        """Генерация ответа через Google Gemini API."""
        content = []
        
        if image_path and self.provider_config.supports_vision:
            # Поддержка vision
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                content.append(image)
            except Exception as e:
                print(f"❌ Ошибка обработки изображения для Gemini: {e}")
        
        # Добавляем текст
        text_content = prompt
        if image_context and not (image_path and self.provider_config.supports_vision):
            text_content += f"\n\nКонтекст изображения: {image_context}"
        
        content.append(text_content)
        
        generation_config = {
            "temperature": self.generation_config.get("temperature", 0.1),
            "max_output_tokens": self.generation_config.get("max_tokens", 4096),
        }
        
        response = self.client.generate_content(
            content,
            generation_config=generation_config
        )
        
        return response.text
    
    def _generate_mistral_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """Генерация ответа через Mistral API."""
        from mistralai.models.chat_completion import ChatMessage, ImageURLChunk, TextChunk
        
        content = []
        
        if image_path and self.provider_config.supports_vision and "pixtral" in self.model_name.lower():
            # Поддержка vision для Pixtral
            base64_image = self._encode_image_base64(image_path)
            if base64_image:
                content.extend([
                    TextChunk(text=prompt),
                    ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_image}")
                ])
            else:
                content.append(TextChunk(text=f"{prompt}\n\nКонтекст изображения: {image_context}"))
        else:
            # Текстовый режим
            text_content = prompt
            if image_context:
                text_content += f"\n\nКонтекст изображения: {image_context}"
            content.append(TextChunk(text=text_content))
        
        messages = [ChatMessage(role="user", content=content)]
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            max_tokens=self.generation_config.get("max_tokens", 4096),
            temperature=self.generation_config.get("temperature", 0.1)
        )
        
        return response.choices[0].message.content
    
    def _generate_deepseek_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """Генерация ответа через DeepSeek API (OpenAI-совместимый)."""
        # DeepSeek пока не поддерживает vision
        content = prompt
        if image_context:
            content += f"\n\nКонтекст изображения: {image_context}"
        
        messages = [{"role": "user", "content": content}]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.generation_config.get("max_tokens", 4096),
            temperature=self.generation_config.get("temperature", 0.1)
        )
        
        return response.choices[0].message.content
    
    def _generate_xai_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """Генерация ответа через xAI API (OpenAI-совместимый)."""
        messages = []
        
        if image_path and self.provider_config.supports_vision and "vision" in self.model_name.lower():
            # Поддержка vision для Grok Vision
            base64_image = self._encode_image_base64(image_path)
            if base64_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": f"{prompt}\n\nКонтекст изображения: {image_context}"})
        else:
            # Текстовый режим
            content = prompt
            if image_context:
                content += f"\n\nКонтекст изображения: {image_context}"
            messages.append({"role": "user", "content": content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.generation_config.get("max_tokens", 4096),
            temperature=self.generation_config.get("temperature", 0.1)
        )
        
        return response.choices[0].message.content
    
    def _generate_ollama_response(self, prompt: str, image_path: str = None, image_context: str = "", timeout: int = 30) -> str:
        """Генерация ответа через Ollama API с адаптивными параметрами."""
        import requests
        from .ollama_utils import is_vision_model
        
        # Получаем оптимальные параметры для модели
        generation_params = get_model_generation_params(self.model_name)
        
        # ОТЛАДКА: Выводим промпт
        print(f"\n[DEBUG] Ollama prompt (first 800 chars):\n{prompt[:800]}\n")
        
        # Подготавливаем данные для запроса
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": generation_params
        }
        
        # Добавляем изображение если модель поддерживает vision
        # ИСПРАВЛЕНИЕ: используем is_vision_model вместо проверки "vision" в названии
        if image_path and is_vision_model(self.model_name):
            try:
                base64_image = self._encode_image_base64(image_path)
                if base64_image:
                    data["images"] = [base64_image]
                    print(f"✅ Изображение добавлено в запрос к Ollama (модель: {self.model_name})")
                else:
                    print(f"⚠️ Не удалось закодировать изображение для {self.model_name}")
            except Exception as e:
                print(f"❌ Ошибка добавления изображения в Ollama: {e}")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Для gemma3 ВСЕГДА добавляем OCR текст
        # даже если модель поддерживает vision, т.к. она плохо работает с изображениями
        if image_context:
            # Добавляем OCR текст для лучшего извлечения данных
            data["prompt"] += f"\n\nDocument OCR text:\n{image_context[:3000]}\n"
            print(f"ℹ️ OCR текст добавлен в промпт для улучшения извлечения (модель: {self.model_name})")
        
        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Увеличиваем таймаут для gemma3 с vision+OCR
            effective_timeout = 120 if "gemma" in self.model_name.lower() else max(timeout, 60)
            print(f"⏱️ Таймаут для {self.model_name}: {effective_timeout} секунд")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=effective_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # ОТЛАДКА: Выводим ответ модели
                print(f"\n[DEBUG] Ollama raw response (first 1500 chars):\n{response_text[:1500]}\n")
                
                if response_text:
                    return response_text
                else:
                    raise ValueError("Пустой ответ от Ollama")
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except:
                    pass
                raise ValueError(f"Ollama API ошибка: {error_msg}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Не удается подключиться к Ollama серверу")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Таймаут запроса к Ollama ({timeout}s)")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ошибка запроса к Ollama: {e}")
    
    def process_image(self, image_path: str, ocr_lang=None, custom_prompt=None) -> Optional[Dict]:
        """
        Основной метод обработки изображения.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык для OCR (если нужен fallback)
            custom_prompt: Пользовательский промпт
            
        Returns:
            Optional[Dict]: Извлеченные данные или None
        """
        if not self.is_loaded:
            print("❌ Модель не загружена")
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            return None
        
        try:
            # Получаем OCR контекст если провайдер не поддерживает vision
            image_context = ""
            model_has_vision = False
            
            # Определяем поддержку vision для модели
            if self.provider_name == "ollama":
                from .ollama_utils import is_vision_model
                model_has_vision = is_vision_model(self.model_name)
            else:
                model_has_vision = self.provider_config.supports_vision and not (
                    (self.provider_name == "deepseek") or
                    (self.provider_name == "mistral" and "pixtral" not in self.model_name.lower())
                )
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Для gemma3 ВСЕГДА извлекаем OCR
            # т.к. она плохо работает с vision, несмотря на поддержку
            should_use_ocr = (
                not model_has_vision or 
                (self.provider_name == "ollama" and "gemma" in self.model_name.lower())
            )
            
            if should_use_ocr:
                print(f"📝 Извлекаем OCR текст для улучшения точности (модель: {self.model_name})")
                image_context = self.extract_text_from_image(image_path, ocr_lang or "rus+eng")
                if image_context:
                    print(f"✅ OCR извлечен, длина: {len(image_context)} символов")
            
            # Создаем промпт с адаптивной системой
            prompt = custom_prompt or self.create_invoice_prompt(
                use_adaptive=True,
                ocr_text=image_context if image_context else None,
                image_available=model_has_vision
            )
            
            # Генерируем ответ
            response = self.generate_response(prompt, image_path, image_context)
            
            # Парсим результат
            result = self.parse_llm_response(response)
            
            if result:
                # Добавляем метаинформацию
                result["_meta"] = {
                    "provider": self.provider_config.display_name,
                    "model": self.model_name,
                    "processed_at": str(datetime.now()),
                    "supports_vision": self.provider_config.supports_vision,
                    "used_ocr": bool(image_context)
                }
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка обработки изображения: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Очистка временных файлов."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"❌ Ошибка удаления временного файла {temp_file}: {e}")
        self.temp_files.clear()
    
    def get_saved_prompt(self) -> str:
        """
        Получает сохраненный промпт для текущего провайдера.
        
        Returns:
            str: Сохраненный промпт или базовый промпт по умолчанию
        """
        try:
            from ...settings_manager import settings_manager
            
            # Определяем тип модели (облачная или локальная)
            model_type = "cloud_llm" if self.provider_name in ["openai", "anthropic", "google", "mistral", "deepseek", "xai"] else "local_llm"
            
            # Формируем ключ для настроек
            prompt_key = f"{model_type}_{self.provider_name}_prompt"
            
            # Получаем сохраненный промпт
            saved_prompt = settings_manager.get_setting(prompt_key, "")
            
            if saved_prompt:
                print(f"✅ Использую сохраненный промпт для {self.provider_name}")
                return saved_prompt
            else:
                print(f"ℹ️ Сохраненный промпт не найден, использую базовый для {self.provider_name}")
                return self.create_invoice_prompt()
                
        except Exception as e:
            print(f"⚠️ Ошибка получения сохраненного промпта: {e}")
            return self.create_invoice_prompt()
    
    def get_full_prompt(self) -> str:
        """
        Возвращает полный промпт для модели.
        Совместимый метод для интеграции с UI.
        
        Returns:
            str: Полный промпт модели
        """
        return self.get_saved_prompt()
    
    def extract_invoice_data(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Извлекает данные из счета-фактуры.
        Совместимый метод для интеграции с основным приложением.
        
        Args:
            image_path: Путь к изображению счета
            prompt: Пользовательский промпт (опционально)
            
        Returns:
            Dict[str, Any]: Извлеченные данные счета
        """
        try:
            # Используем сохраненный промпт если не передан пользовательский
            if not prompt:
                prompt = self.get_saved_prompt()
            
            # Используем основной метод process_image
            result = self.process_image(image_path, custom_prompt=prompt)
            
            if result is None:
                print(f"❌ Не удалось извлечь данные из {image_path}")
                return {}
            
            print(f"✅ Успешно извлечены данные из {image_path}")
            return result
            
        except Exception as e:
            print(f"❌ Ошибка извлечения данных счета: {e}")
            return {}
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        self.cleanup_temp_files() 
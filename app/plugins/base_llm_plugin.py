"""
Базовый класс для всех LLM плагинов
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import os
import json
import re
import logging
from PIL import Image

# Настройка логирования
logger = logging.getLogger(__name__)

# Пытаемся импортировать torch, но делаем это опционально
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch не установлен. LLM плагины будут работать в ограниченном режиме.")

# Пытаемся импортировать SecretsManager для защиты API ключей
try:
    from ..security.secrets_manager import get_secrets_manager
    SECRETS_MANAGER_AVAILABLE = True
except ImportError:
    SECRETS_MANAGER_AVAILABLE = False
    logger.warning("SecretsManager недоступен. API ключи будут храниться в незашифрованном виде.")

from ..base_processor import BaseProcessor

class LLMProviderConfig:
    """Конфигурация для провайдера LLM"""
    def __init__(self, name: str, display_name: str, models: List[str], 
                 requires_api_key: bool = True, api_key_name: str = None,
                 default_model: str = None, supports_vision: bool = True):
        self.name = name
        self.display_name = display_name
        self.models = models
        self.requires_api_key = requires_api_key
        self.api_key_name = api_key_name or f"{name.upper()}_API_KEY"
        self.default_model = default_model or (models[0] if models else None)
        self.supports_vision = supports_vision

# Конфигурации поддерживаемых провайдеров
LLM_PROVIDERS = {
    "openai": LLMProviderConfig(
        name="openai",
        display_name="OpenAI (ChatGPT)",
        models=[
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        default_model="gpt-4o",
        supports_vision=True
    ),
    "anthropic": LLMProviderConfig(
        name="anthropic", 
        display_name="Anthropic (Claude)",
        models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        default_model="claude-3-5-sonnet-20241022",
        supports_vision=True
    ),
    "google": LLMProviderConfig(
        name="google",
        display_name="Google (Gemini)",
        models=[
            "models/gemini-2.0-flash-exp",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro-002",
            "models/gemini-1.5-flash-002"
        ],
        default_model="models/gemini-2.0-flash-exp",
        supports_vision=True
    ),
    "mistral": LLMProviderConfig(
        name="mistral",
        display_name="Mistral AI",
        models=[
            "mistral-large-latest",
            "mistral-medium-latest", 
            "mistral-small-latest",
            "pixtral-12b-2409"
        ],
        default_model="mistral-large-latest",
        supports_vision=True
    ),
    "deepseek": LLMProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        models=[
            "deepseek-chat",
            "deepseek-coder"
        ],
        default_model="deepseek-chat", 
        supports_vision=False
    ),
    "xai": LLMProviderConfig(
        name="xai",
        display_name="xAI (Grok)",
        models=[
            "grok-beta",
            "grok-vision-beta"
        ],
        default_model="grok-vision-beta",
        supports_vision=True
    ),
    "ollama": LLMProviderConfig(
        name="ollama",
        display_name="Ollama (Локально)",
        models=[
            "llama3.2-vision:11b",
            "llama3.2:3b",
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral:7b",
            "qwen2.5:7b"
        ],
        default_model="llama3.2-vision:11b",
        requires_api_key=False,
        supports_vision=True
    )
}

class BaseLLMPlugin(BaseProcessor):
    """
    Базовый класс для всех LLM плагинов.
    Расширяет BaseProcessor специфичной функциональностью для LLM.
    """
    
    def __init__(self, provider_name: str, model_name: str = None, api_key: str = None, **kwargs):
        """
        Инициализация LLM плагина.
        
        Args:
            provider_name: Название провайдера (openai, anthropic, google, etc.)
            model_name: Название модели (необязательно, используется default)
            api_key: API ключ для провайдера
            **kwargs: Дополнительные параметры для конкретного плагина
        """
        self.provider_name = provider_name
        self.provider_config = LLM_PROVIDERS.get(provider_name)
        
        if not self.provider_config:
            raise ValueError(f"Неподдерживаемый провайдер LLM: {provider_name}")
        
        self.model_name = model_name or self.provider_config.default_model
        self.api_key = self._get_secure_api_key(api_key)
        self.client = None
        self.is_loaded = False
        
        # Параметры генерации по умолчанию
        self.generation_config = {
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        
        # Обновляем конфигурацию пользовательскими параметрами
        self.generation_config.update(kwargs.get("generation_config", {}))
        
        logger.info(f"Инициализирован LLM плагин: {self.provider_config.display_name} - {self.model_name}")
    
    def _get_secure_api_key(self, api_key: str = None) -> Optional[str]:
        """
        Получает API ключ безопасным способом
        
        Args:
            api_key: Переданный API ключ
            
        Returns:
            Optional[str]: API ключ или None
        """
        if not self.provider_config.requires_api_key:
            return None
        
        # Если ключ передан напрямую, используем его
        if api_key:
            return api_key
        
        # Пытаемся получить из SecretsManager
        if SECRETS_MANAGER_AVAILABLE:
            try:
                secrets_manager = get_secrets_manager()
                secret_key = f"{self.provider_name}_api_key"
                stored_key = secrets_manager.get_secret(secret_key)
                if stored_key:
                    logger.debug(f"API ключ для {self.provider_name} получен из SecretsManager")
                    return stored_key
            except Exception as e:
                logger.warning(f"Ошибка получения ключа из SecretsManager: {e}")
        
        # Пытаемся получить из переменных окружения
        env_key = self.provider_config.api_key_name
        api_key = os.environ.get(env_key)
        if api_key:
            logger.debug(f"API ключ для {self.provider_name} получен из переменной окружения {env_key}")
            # Сохраняем в SecretsManager для будущего использования
            if SECRETS_MANAGER_AVAILABLE:
                try:
                    secrets_manager = get_secrets_manager()
                    secrets_manager.store_secret(f"{self.provider_name}_api_key", api_key)
                except Exception as e:
                    logger.warning(f"Не удалось сохранить ключ в SecretsManager: {e}")
        
        return api_key
    
    def validate_api_key(self) -> bool:
        """
        Проверяет наличие и валидность API ключа
        
        Returns:
            bool: True если ключ есть и валиден
        """
        if not self.provider_config.requires_api_key:
            return True
        
        if not self.api_key:
            logger.error(f"API ключ для {self.provider_name} не найден")
            return False
        
        # Базовая проверка формата ключа
        if len(self.api_key) < 10:
            logger.error(f"API ключ для {self.provider_name} слишком короткий")
            return False
        
        return True
    
    # Обязательные методы от BaseProcessor
    @abstractmethod
    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """Основной метод обработки изображения."""
        pass
    
    def supports_training(self) -> bool:
        """LLM плагины поддерживают обучение."""
        return True
    
    def get_trainer_class(self):
        """Возвращает класс LLMTrainer."""
        try:
            from .llm_trainer import LLMTrainer
            return LLMTrainer
        except ImportError:
            logger.warning("LLMTrainer еще не реализован")
            return None
    
    def get_model_type(self) -> str:
        return f"llm_{self.provider_name}"
    
    def get_full_prompt(self):
        """Возвращает базовый промпт для извлечения данных из инвойсов."""
        return self.create_invoice_prompt()
    
    # Специфичные для LLM методы
    @abstractmethod
    def load_model(self) -> bool:
        """
        Загружает модель и инициализирует клиент.
        
        Returns:
            bool: True если загрузка успешна
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, image_path: str = None, image_context: str = "") -> str:
        """
        Генерирует ответ на основе промпта и изображения/контекста.
        
        Args:
            prompt: Промпт для модели
            image_path: Путь к изображению (если поддерживается)
            image_context: Текстовое описание/OCR текст изображения
            
        Returns:
            str: Ответ модели
        """
        pass
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию для обучения модели.
        Для внешних API обычно не применимо.
        
        Returns:
            dict: Конфигурация обучения
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "supports_fine_tuning": False,
            "generation_config": self.generation_config
        }
    
    # Вспомогательные методы
    def preprocess_image_for_llm(self, image_path: str) -> Image.Image:
        """
        Предобработка изображения для LLM.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            PIL.Image: Обработанное изображение
        """
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            # Возвращаем пустое изображение
            return Image.new('RGB', (100, 100), color='white')
    
    def extract_text_from_image(self, image_path: str, ocr_lang: str = "rus+eng") -> str:
        """
        Извлекает текст из изображения с помощью OCR.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Языки для OCR
            
        Returns:
            str: Извлеченный текст
        """
        try:
            import pytesseract
            image = self.preprocess_image_for_llm(image_path)
            text = pytesseract.image_to_string(image, lang=ocr_lang)
            return text.strip()
        except ImportError:
            logger.warning("pytesseract не установлен. OCR недоступен.")
            return "OCR недоступен - установите pytesseract"
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return "Не удалось извлечь текст из изображения"
    
    def create_invoice_prompt(self, custom_prompt: Optional[str] = None, include_context_fields: bool = True) -> str:
        """
        Создает промпт для извлечения данных из инвойса.
        
        Args:
            custom_prompt: Пользовательский промпт
            include_context_fields: Включать ли контекстные поля из настроек
            
        Returns:
            str: Промпт для модели
        """
        if custom_prompt:
            return custom_prompt
        
        # Базовый промпт
        base_prompt = """Действуй как эксперт по распознаванию счетов-фактур и документов. Проанализируй предоставленное изображение документа и извлеки из него все ключевые данные в формате JSON.

Формат должен включать следующие поля (включай только если они присутствуют в документе):"""

        # Получаем поля из настроек таблицы если доступно
        fields_json = {}
        if include_context_fields:
            try:
                from ..settings_manager import settings_manager
                table_fields = settings_manager.get_table_fields()
                if table_fields:
                    fields_json = {field['name']: field.get('description', '') for field in table_fields}
            except (ImportError, AttributeError, KeyError, TypeError) as e:
                # Ошибка при получении настроек полей таблицы - используем стандартные
                pass
        
        # Если нет настроенных полей, используем стандартные
        if not fields_json:
            fields_json = {
                "Поставщик": "название организации-поставщика",
                "ИНН поставщика": "ИНН в формате 10 или 12 цифр",
                "КПП поставщика": "КПП в формате 9 цифр",
                "Адрес поставщика": "полный юридический адрес",
                "Покупатель": "название организации-покупателя",
                "ИНН покупателя": "ИНН в формате 10 или 12 цифр",
                "КПП покупателя": "КПП в формате 9 цифр",
                "Адрес покупателя": "полный юридический адрес",
                "№ Счета": "номер счета точно как в документе",
                "Дата счета": "дата в формате DD.MM.YYYY",
                "Дата оплаты": "срок оплаты в формате DD.MM.YYYY, если указан",
                "Категория": "определи основную категорию товаров/услуг",
                "Товары": "список всех товаров/услуг с количеством и ценами",
                "Сумма без НДС": "сумма до НДС числом",
                "НДС %": "ставка НДС числом",
                "Сумма НДС": "сумма НДС числом",
                "Сумма с НДС": "итоговая сумма числом",
                "Валюта": "RUB/USD/EUR и т.д.",
                "Банк": "название банка, если указано",
                "БИК": "БИК банка в формате 9 цифр",
                "Р/с": "расчетный счет в формате 20 цифр",
                "К/с": "корреспондентский счет в формате 20 цифр",
                "Комментарии": "любая дополнительная информация"
            }
        
        # Формируем JSON структуру для промпта
        json_structure = "{\n"
        for field_name, description in fields_json.items():
            json_structure += f'  "{field_name}": "{description}",\n'
        json_structure = json_structure.rstrip(',\n') + "\n}"
        
        instructions = """

Важные требования:
1. Представь результат ТОЛЬКО в виде JSON, без лишнего текста до и после.
2. Сохраняй точное форматирование и орфографию из оригинала.
3. Вычисли категорию товаров/услуг на основе их описания.
4. Убедись, что числа форматированы корректно, без лишних пробелов.
5. Для полей с числами (суммы, ИНН, КПП, счета) удали все пробелы и используй точку как разделитель для дробных чисел.
6. Даты всегда приводи к формату DD.MM.YYYY.
7. Если какое-то поле отсутствует в документе, не включай его в результат.
8. Будь максимально точным и внимательным к деталям."""

        return base_prompt + "\n\n" + json_structure + instructions
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Парсит ответ LLM и извлекает JSON данные.
        
        Args:
            response: Ответ от LLM
            
        Returns:
            dict: Извлеченные данные
        """
        try:
            # Проверяем, не содержит ли ответ сообщение об ошибке API
            if self._is_error_response(response):
                error_msg = self._extract_error_message(response)
                logger.error(f"Ответ содержит ошибку API: {error_msg}")
                return {"error": error_msg, "note_gemini": f"Ошибка API {self.provider_name}: {error_msg}"}
            
            # Очищаем ответ от лишнего текста
            cleaned_response = self._clean_json_string(response)
            
            # Пытаемся извлечь JSON
            json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return self._normalize_invoice_data(data)
            else:
                logger.warning("JSON не найден в ответе LLM")
                logger.debug(f"Ответ LLM: {response[:300]}...")
                return {"error": "JSON не найден в ответе", "raw_response": response[:500]}
                
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.debug(f"Ответ LLM: {response[:500]}...")
            return {"error": f"Ошибка парсинга JSON: {e}", "raw_response": response[:500]}
        except Exception as e:
            logger.error(f"Ошибка обработки ответа LLM: {e}")
            return {"error": f"Ошибка обработки: {e}", "raw_response": response[:500] if response else "Пустой ответ"}
    
    def _is_error_response(self, response: str) -> bool:
        """Проверяет, содержит ли ответ ошибку API."""
        if not response:
            return True
        
        error_indicators = [
            "error code:",
            "error:",
            "insufficient_quota",
            "rate_limit_exceeded", 
            "invalid_api_key",
            "user location is not supported",
            "authentication failed",
            "permission denied",
            "service unavailable",
            "internal server error"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in error_indicators)
    
    def _extract_error_message(self, response: str) -> str:
        """Извлекает сообщение об ошибке из ответа."""
        try:
            # Пытаемся найти JSON с ошибкой
            import json
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                error_data = json.loads(json_match.group())
                if 'error' in error_data:
                    if isinstance(error_data['error'], dict):
                        return error_data['error'].get('message', str(error_data['error']))
                    else:
                        return str(error_data['error'])
            
            # Если JSON не найден, возвращаем первые 200 символов
            return response[:200] + "..." if len(response) > 200 else response
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Ошибка при извлечении сообщения об ошибке - возвращаем исходный ответ
            return response[:200] + "..." if len(response) > 200 else response
    
    def _clean_json_string(self, json_str: str) -> str:
        """Очищает строку JSON от лишнего текста."""
        # Удаляем markdown форматирование
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        return json_str.strip()
    
    def _normalize_invoice_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализует данные инвойса."""
        normalized = {}
        
        for key, value in data.items():
            if value is None or value == "":
                continue
                
            # Нормализуем числовые поля
            if any(word in key.lower() for word in ['сумма', 'ндс', 'инн', 'кпп', 'бик']):
                if isinstance(value, str):
                    # Удаляем пробелы и запятые, заменяем на точки
                    cleaned_value = re.sub(r'[^\d,.]', '', value)
                    cleaned_value = cleaned_value.replace(',', '.')
                    try:
                        if '.' in cleaned_value:
                            normalized[key] = float(cleaned_value)
                        else:
                            normalized[key] = int(cleaned_value) if cleaned_value else 0
                    except ValueError:
                        normalized[key] = value
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
                
        return normalized
    
    def _safe_float(self, value) -> float:
        """Безопасное преобразование в float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Удаляем все кроме цифр, точек и запятых
                cleaned = re.sub(r'[^\d,.]', '', value)
                cleaned = cleaned.replace(',', '.')
                return float(cleaned) if cleaned else 0.0
            except ValueError:
                return 0.0
        return 0.0
    
    def get_model_info(self) -> dict:
        """Возвращает информацию о модели."""
        return {
            "provider": self.provider_config.display_name,
            "model": self.model_name,
            "supports_vision": self.provider_config.supports_vision,
            "requires_api_key": self.provider_config.requires_api_key,
            "is_loaded": self.is_loaded,
            "has_api_key": bool(self.api_key) if self.provider_config.requires_api_key else True
        }
    
    @staticmethod
    def get_available_providers() -> Dict[str, LLMProviderConfig]:
        """Возвращает список доступных провайдеров LLM."""
        return LLM_PROVIDERS
    
    @staticmethod
    def get_provider_models(provider_name: str) -> List[str]:
        """Возвращает список моделей для провайдера."""
        provider = LLM_PROVIDERS.get(provider_name)
        return provider.models if provider else []
    
    @staticmethod
    def update_provider_models(provider_name: str, models: List[str]) -> bool:
        """
        Обновляет список моделей для провайдера.
        Используется для динамического обновления после получения из API.
        
        Args:
            provider_name: Имя провайдера
            models: Новый список моделей
            
        Returns:
            bool: True если обновление успешно
        """
        if provider_name in LLM_PROVIDERS:
            LLM_PROVIDERS[provider_name].models = models
            logger.info(f"Обновлен список моделей для {provider_name}: {len(models)} моделей")
            return True
        return False
    
    @staticmethod
    def refresh_provider_models(provider_name: str, api_key: str = None) -> List[str]:
        """
        Обновляет список моделей из API провайдера.
        
        Args:
            provider_name: Имя провайдера
            api_key: API ключ для доступа
            
        Returns:
            List[str]: Обновленный список моделей
        """
        if provider_name == "openai":
            return BaseLLMPlugin._refresh_openai_models(api_key)
        elif provider_name == "google":
            return BaseLLMPlugin._refresh_google_models(api_key)
        elif provider_name == "anthropic":
            return BaseLLMPlugin._refresh_anthropic_models(api_key)
        else:
            logger.warning(f"Обновление моделей для {provider_name} не поддерживается")
            return LLM_PROVIDERS.get(provider_name, LLMProviderConfig("", "", [])).models
    
    @staticmethod
    def _refresh_openai_models(api_key: str) -> List[str]:
        """Получает список моделей OpenAI через API."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            
            # Фильтруем только релевантные модели
            relevant_models = []
            for model in models.data:
                if any(prefix in model.id for prefix in ['gpt-', 'dall-e', 'whisper']):
                    if not BaseLLMPlugin._is_openai_model_deprecated(model.id):
                        relevant_models.append(model.id)
            
            # Сортируем модели
            relevant_models.sort(reverse=True)
            
            # Обновляем глобальный список
            if relevant_models:
                BaseLLMPlugin.update_provider_models("openai", relevant_models)
            
            return relevant_models
            
        except Exception as e:
            logger.error(f"Ошибка получения моделей OpenAI: {e}")
            return LLM_PROVIDERS["openai"].models
    
    @staticmethod
    def _refresh_google_models(api_key: str) -> List[str]:
        """Получает список моделей Google через API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name)
            
            # Сортируем модели
            models.sort(reverse=True)
            
            # Обновляем глобальный список
            if models:
                BaseLLMPlugin.update_provider_models("google", models)
            
            return models
            
        except Exception as e:
            logger.error(f"Ошибка получения моделей Google: {e}")
            return LLM_PROVIDERS["google"].models
    
    @staticmethod
    def _refresh_anthropic_models(api_key: str) -> List[str]:
        """Получает список моделей Anthropic."""
        # Anthropic не предоставляет API для получения списка моделей
        # Возвращаем предопределенный список
        return LLM_PROVIDERS["anthropic"].models
    
    @staticmethod
    def _is_openai_model_deprecated(model_id: str) -> bool:
        """Проверяет, является ли модель OpenAI устаревшей."""
        deprecated_patterns = [
            'davinci', 'curie', 'babbage', 'ada',
            'text-', 'code-', 'edit-', 'if-',
            '-001', '-002', '-003'
        ]
        return any(pattern in model_id for pattern in deprecated_patterns) 
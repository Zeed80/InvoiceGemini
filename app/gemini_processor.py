"""
Модуль для взаимодействия с Google Gemini API.
"""
import os
import re
import json
import uuid
import time
from PIL import Image
from typing import Optional, Dict
from datetime import datetime
import tempfile
import shutil
import logging

from .base_processor import BaseProcessor
from . import config as app_config
from .settings_manager import settings_manager
from .invoice_formatter import InvoiceFormatter

# Настройка логирования
logger = logging.getLogger(__name__)

# Глобальная функция для логирования
def log_message(message):
    """Простая функция логирования для GeminiProcessor"""
    print(message)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class GeminiProcessor(BaseProcessor):
    """
    Процессор для модели Google Gemini.
    Взаимодействует с Gemini API для извлечения данных из документов.
    """
    def __init__(self, model_id: Optional[str] = None):
        """
        Инициализация процессора Gemini.
        
        Args:
            model_id: ID модели Gemini (если не указан, используется из настроек)
        """
        super().__init__()
        self.model_id = model_id or settings_manager.get_string('Gemini', 'sub_model_id', 'models/gemini-2.0-flash')
        self.model = None
        self.is_loaded = False
        self._full_prompt_sent = None
        self.temp_files = []  # Список для хранения путей к временным файлам
        self.temp_dirs = []  # Список для хранения путей к временным директориям
        
        # Инициализируем менеджер ресурсов если доступен
        try:
            from .core.resource_manager import get_resource_manager
            self.resource_manager = get_resource_manager()
        except ImportError:
            self.resource_manager = None
            logger.warning("Менеджер ресурсов недоступен, используется старая система")
        
        # Создаем подпапку для временных файлов Gemini, если её нет
        self.temp_folder = os.path.join(app_config.TEMP_PATH, "gemini_temp")
        os.makedirs(self.temp_folder, exist_ok=True)
        self._custom_prompt_override = None  # Для временного переопределения промпта
        
        if GENAI_AVAILABLE:
            # Получаем API ключ через безопасную систему с fallback
            try:
                from config.secrets import get_google_api_key
                self.api_key = get_google_api_key()
            except ImportError:
                # Fallback на старую систему для обратной совместимости
                self.api_key = settings_manager.get_gemini_api_key()
                
            app_config.GOOGLE_API_KEY = self.api_key  # Обновляем значение в config для совместимости
            
            if self.api_key:
                # Вызываем метод load_model(), чтобы инициализировать модель
                self.load_model()
            else:
                print("API ключ для Gemini не найден. GeminiProcessor не будет работать.")
                self.is_loaded = False
        else:
            print("Библиотека google-generativeai не установлена. GeminiProcessor не будет работать.")
            self.is_loaded = False

    def load_model(self, model_id=None):
        """
        Для Gemini 'загрузка' означает проверку доступности API и ключа, и инициализацию конкретной модели.
        """
        if model_id:
            self.model_id = model_id
        
        if not GENAI_AVAILABLE:
            print("Ошибка: Библиотека google-generativeai не установлена.")
            self.is_loaded = False
            return False

        # Получаем API ключ через безопасную систему с fallback
        try:
            from config.secrets import get_google_api_key
            current_api_key = get_google_api_key()
        except ImportError:
            # Fallback на старую систему для обратной совместимости
            current_api_key = settings_manager.get_gemini_api_key()
            
        app_config.GOOGLE_API_KEY = current_api_key  # Обновляем для совместимости
        
        if not current_api_key:
            print("Ошибка: API ключ Google не установлен в настройках.")
            self.is_loaded = False
            self.model = None
            return False
        
        try:
            genai.configure(api_key=current_api_key)
            self.model = genai.GenerativeModel(self.model_id)
            print(f"GeminiProcessor successfully configured to use model: {self.model_id}")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Ошибка при конфигурации Gemini API или инициализации модели {self.model_id}: {e}")
            self.is_loaded = False
            self.model = None
            return False

    def cleanup_temp_files(self):
        """
        Очищает все временные файлы, созданные во время обработки.
        """
        if self.resource_manager:
            # Новая система автоматически очищает ресурсы
            logger.info("Очистка временных файлов выполняется автоматически менеджером ресурсов")
            return
            
        # Fallback на старую систему
        print("Очистка временных файлов Gemini...")
        
        # Удаляем временные файлы
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"  Удален временный файл: {temp_file}")
            except Exception as e:
                print(f"  Ошибка при удалении файла {temp_file}: {e}")
        
        # Удаляем временные директории
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"  Удалена временная директория: {temp_dir}")
            except Exception as e:
                print(f"  Ошибка при удалении директории {temp_dir}: {e}")
        
        # Очищаем списки
        self.temp_files.clear()
        self.temp_dirs.clear()
        
        print("Очистка временных файлов завершена")

    def process(self, image_path: str) -> Optional[Dict]:
        """
        Обрабатывает изображение с помощью Gemini API и извлекает поля из счёта.
        
        Args:
            image_path: Путь к файлу изображения
            
        Returns:
            Optional[Dict]: Извлеченные данные или None в случае ошибки
        """
        try:
            log_message(f"Обработка изображения: {image_path}")
            
            # Проверяем существование файла
            if not os.path.exists(image_path):
                log_message(f"ОШИБКА: Файл не существует: {image_path}")
                return None
                
            # Загружаем изображение с учетом возможных ошибок
            try:
                # Проверяем, является ли файл PDF-документом
                if image_path.lower().endswith('.pdf'):
                    # Для PDF файлов используем конвертацию через pdf2image
                    try:
                        from pdf2image import convert_from_path
                        import tempfile
                        
                        # Проверяем настройки poppler
                        poppler_path = app_config.POPPLER_PATH if hasattr(app_config, 'POPPLER_PATH') else None
                        
                        log_message(f"Конвертация PDF в изображение для Gemini: {image_path}")
                        
                        # Конвертируем PDF в изображения (только первую страницу)
                        images = convert_from_path(
                            image_path, 
                            dpi=300, 
                            first_page=1, 
                            last_page=1,
                            poppler_path=poppler_path
                        )
                        
                        if not images:
                            log_message(f"ОШИБКА: Не удалось конвертировать PDF в изображения: {image_path}")
                            return None
                        
                        # Используем первое изображение
                        image = images[0]
                        log_message(f"PDF успешно сконвертирован в изображение")
                        
                        # Сохраняем временное изображение для очистки позже
                        temp_dir = tempfile.mkdtemp()
                        temp_image_path = os.path.join(temp_dir, "temp_gemini_pdf.jpg")
                        image.save(temp_image_path, "JPEG")
                        self.temp_files.append(temp_image_path)
                        self.temp_dirs.append(temp_dir)
                        
                    except ImportError:
                        log_message("ОШИБКА: pdf2image не установлен. Установите: pip install pdf2image")
                        return None
                    except Exception as e:
                        log_message(f"ОШИБКА: Не удалось обработать PDF файл: {e}")
                        import traceback
                        log_message(traceback.format_exc())
                        return None
                else:
                    # Для обычных изображений
                    image = Image.open(image_path)
                
                # Конвертируем в RGB если изображение не в RGB формате
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
            except Exception as e:
                log_message(f"Ошибка при загрузке изображения: {str(e)}")
                return None
                
            # Получаем промпт для модели
            prompt = self._get_prompt()
            
            # Подготавливаем запрос к API
            safety_settings = self._get_safety_settings()
            generation_config = self._get_generation_config()
            
            # Попытка вызова API с повторами при необходимости
            response = None
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Создаем и выполняем запрос с явным указанием JSON формата ответа
                    log_message("Отправка запроса к Gemini API с response_mime_type='application/json'")
                    response = self.model.generate_content(
                        [prompt, image],
                        safety_settings=safety_settings,
                        generation_config=generation_config
                    )
                    break
                except Exception as e:
                    log_message(f"Ошибка при вызове API (попытка {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        log_message(f"Повторная попытка через {retry_delay} сек...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Увеличиваем задержку для следующей попытки
                    else:
                        log_message("Все попытки вызова API завершились неудачно")
                        return None
            
            if not response:
                log_message("Не удалось получить ответ от API")
                return None
                
            # Проверка на наличие ошибок в ответе
            if hasattr(response, 'error') and response.error:
                log_message(f"API вернул ошибку: {response.error}")
                return None
                
            # Извлекаем и парсим ответ
            try:
                response_text = response.text
                log_message("Получен ответ от API")
                
                # ОТЛАДКА: Показываем исходный ответ от API
                log_message(f"ОТЛАДКА: Исходный ответ от Gemini API (первые 1000 символов):")
                log_message(f"ОТЛАДКА: {response_text[:1000]}...")
                
                # Сначала пробуем напрямую разобрать JSON (с новым response_mime_type должно работать)
                try:
                    processed_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # Если прямой разбор не удался, используем метод извлечения JSON
                    log_message("Прямой разбор JSON не удался, используем _extract_json_from_response")
                    processed_data = self._extract_json_from_response(response_text)
                
                # Проверка структуры и преобразование формата данных
                processed_data = self._standardize_response_format(processed_data)
                
                # Логирование информации о полях
                fields_count = len(processed_data.get('fields', []))
                log_message(f"Извлечено полей: {fields_count}")
                
                # ДОПОЛНИТЕЛЬНАЯ ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
                log_message(f"ОТЛАДКА: Структура ответа от API:")
                log_message(f"ОТЛАДКА: Тип processed_data: {type(processed_data)}")
                log_message(f"ОТЛАДКА: Ключи в processed_data: {list(processed_data.keys()) if isinstance(processed_data, dict) else 'N/A'}")
                if isinstance(processed_data, dict) and 'fields' in processed_data:
                    log_message(f"ОТЛАДКА: Количество полей в fields: {len(processed_data['fields'])}")
                    for i, field in enumerate(processed_data['fields'][:5]):  # Показываем первые 5 полей
                        log_message(f"ОТЛАДКА: Поле {i+1}: {field}")
                log_message(f"ОТЛАДКА: Полный ответ (первые 500 символов): {str(processed_data)[:500]}...")
                # КОНЕЦ ОТЛАДОЧНОЙ ИНФОРМАЦИИ
                
                # Добавляем дополнительные метаданные
                processed_data['source_image'] = image_path
                processed_data['processed_at'] = datetime.now().isoformat()
                
                return processed_data
                
            except Exception as e:
                log_message(f"Ошибка при обработке ответа API: {str(e)}")
                import traceback
                log_message(traceback.format_exc())
                return None
                
        except Exception as e:
            log_message(f"Ошибка в методе process: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
            return None

    def _clean_json_string(self, s):
        """
        Очищает строку от маркеров кода и извлекает JSON.
        
        Args:
            s (str): Исходная строка с JSON данными
            
        Returns:
            str: Очищенная строка, содержащая только JSON
        """
        if not s:
            raise ValueError("Пустая строка")
            
        # Убираем маркеры кода
        s = s.strip()
        s = re.sub(r'^```\s*(?:json\s*\n)?', '', s)  # Начальные маркеры
        s = re.sub(r'\s*```$', '', s)  # Конечные маркеры
        
        # Убираем возможные дополнительные маркеры
        s = s.strip('`').strip()
        
        # Если строка начинается с 'json', убираем это
        if s.lower().startswith('json'):
            s = s[4:].strip()
            
        # Проверяем базовую структуру JSON
        if not (s.startswith('{') and s.endswith('}')):
            raise ValueError("Строка не содержит валидный JSON объект")
            
        return s

    def get_full_prompt(self, custom_prompt_text=None, image_name=None):
        """
        Формирует полный промпт для модели Gemini.
        
        Args:
            custom_prompt_text (str, optional): Пользовательский текст промпта
            image_name (str, optional): Имя обрабатываемого изображения/файла
            
        Returns:
            str: Полный текст промпта для модели
        """
        # Базовый промпт
        base_prompt = custom_prompt_text if custom_prompt_text else settings_manager.get_string(
            'Prompts', 
            'gemini_prompt', 
            app_config.GEMINI_ANNOTATION_PROMPT_DEFAULT
        )
        
        # Добавляем информацию о формате
        format_info = """
        Формат ответа должен быть строго в JSON:
        {
            "invoice_number": "номер счета",
            "date": "дата счета",
            "total_amount": "общая сумма",
            "currency": "валюта",
            "supplier": {
                "name": "название поставщика",
                "tax_id": "ИНН/налоговый номер",
                "address": "адрес"
            },
            "customer": {
                "name": "название клиента",
                "tax_id": "ИНН/налоговый номер",
                "address": "адрес"
            },
            "items": [
                {
                    "description": "описание товара/услуги",
                    "quantity": "количество",
                    "unit_price": "цена за единицу",
                    "total": "общая стоимость позиции"
                }
            ],
            "tax_amount": "сумма налога",
            "tax_rate": "ставка налога",
            "payment_info": {
                "bank_name": "название банка",
                "account_number": "номер счета",
                "bic": "БИК/SWIFT"
            }
        }
        """
        
        # Добавляем контекст о файле
        file_context = f"\nОбрабатываемый файл: {image_name}\n" if image_name else ""
        
        # Формируем полный промпт
        full_prompt = f"{base_prompt}\n{format_info}{file_context}\nОтвет должен быть только в формате JSON, без дополнительного текста."
        
        return full_prompt

    def get_available_models(self):
        """
        Возвращает список доступных моделей Gemini.
        
        Returns:
            list: Список доступных моделей с их характеристиками
        """
        if not GENAI_AVAILABLE:
            return [{"error": "Библиотека google-generativeai не установлена"}]
            
        if not self.api_key:
            return [{"error": "API ключ не настроен"}]
            
        try:
            # Конфигурируем API
            genai.configure(api_key=self.api_key)
            
            # Получаем список моделей
            models = genai.list_models()
            
            # Фильтруем и форматируем информацию о моделях
            available_models = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "input_types": [str(t) for t in model.supported_generation_methods],
                    "version": getattr(model, "version", "unknown"),
                    "supports_text": "generateContent" in model.supported_generation_methods,
                    "supports_vision": hasattr(model, "image_supported") and model.image_supported,
                }
                available_models.append(model_info)
            
            return available_models
            
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {str(e)}")
            return [{"error": f"Ошибка получения списка моделей: {str(e)}"}]

    def _convert_to_training_format(self, json_data):
        """
        Преобразует структурированный JSON в формат для сопоставления с OCR.
        
        Args:
            json_data (dict): Структурированный JSON с данными счета
            
        Returns:
            dict: Данные в формате для сопоставления с OCR
        """
        # Добавляем отладочный вывод для анализа проблемы
        log_message(f"_convert_to_training_format: Получен json_data типа {type(json_data)}")
        log_message(f"_convert_to_training_format: Содержимое json_data: {json_data}")
        
        # Проверяем, что json_data это словарь
        if not isinstance(json_data, dict):
            log_message(f"ОШИБКА: _convert_to_training_format получил не словарь, а {type(json_data)}: {json_data}")
            # Пытаемся преобразовать строку в JSON, если это возможно
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError:
                    # Если не удалось разобрать JSON, возвращаем пустой словарь
                    print(f"ОШИБКА: Не удалось преобразовать строку в JSON: {json_data}")
                    return {"note_gemini": "ОШИБКА: Недопустимый формат данных"}
            else:
                return {"note_gemini": "ОШИБКА: Недопустимый формат данных"}
        
        # Инициализируем базовые поля для обучения
        training_data = {
            'invoice_number': [],
            'date': [],
            'total_amount': [],
            'supplier_name': [],
            'supplier_tax_id': [],
            'customer_name': [],
            'customer_tax_id': []
        }
        
        # Проверяем наличие поля fields в формате Gemini API
        if 'fields' in json_data and isinstance(json_data['fields'], list):
            # Обрабатываем поля в формате Gemini API
            for field in json_data['fields']:
                if isinstance(field, dict) and 'field_name' in field and 'field_value' in field:
                    field_name = field['field_name'].lower()
                    field_value = field['field_value']
                    
                    # Маппинг полей Gemini API на названия полей для обучения
                    if field_name in ['company', 'название компании', 'поставщик', 'vendor', 'supplier', 'company_name', 'supplier_name']:
                        training_data['supplier_name'].append(field_value)
                    elif field_name in ['invoice_number', 'номер счета', 'номер инвойса', 'invoice_id', 'счет №']:
                        training_data['invoice_number'].append(field_value)
                    elif field_name in ['date', 'invoice_date', 'дата', 'дата счета', 'дата инвойса']:
                        training_data['date'].append(field_value)
                    elif field_name in ['total_amount', 'total', 'amount', 'сумма', 'итого', 'общая сумма', 'к оплате']:
                        training_data['total_amount'].append(field_value)
                    elif field_name in ['inn', 'инн', 'tax_id', 'tax id', 'налоговый номер']:
                        training_data['supplier_tax_id'].append(field_value)
                    elif field_name in ['kpp', 'кпп']:
                        if 'kpp' not in training_data:
                            training_data['kpp'] = []
                        training_data['kpp'].append(field_value)
                    elif field_name in ['customer', 'customer_name', 'client', 'покупатель', 'заказчик']:
                        training_data['customer_name'].append(field_value)
                    elif field_name in ['customer_inn', 'customer_tax_id', 'инн покупателя', 'инн заказчика']:
                        training_data['customer_tax_id'].append(field_value)
                    else:
                        # Добавляем поле под его оригинальным названием
                        if field_name not in training_data:
                            training_data[field_name] = []
                        training_data[field_name].append(field_value)
        else:
            # Старый формат - прямые поля в корне объекта
            mapping = {
                'invoice_number': ['invoice_number', 'номер счета', 'номер инвойса'],
                'date': ['date', 'invoice_date', 'дата счета', 'дата'],
                'total_amount': ['total_amount', 'total', 'amount', 'сумма', 'итого'],
                'supplier_name': ['supplier_name', 'company', 'vendor', 'поставщик'],
                'supplier_tax_id': ['supplier_tax_id', 'tax_id', 'inn', 'инн'],
                'customer_name': ['customer_name', 'client', 'покупатель', 'заказчик'],
                'customer_tax_id': ['customer_tax_id', 'customer_inn', 'инн заказчика']
            }
            
            # Ищем соответствия между ключами json_data и полями для обучения
            for train_field, possible_keys in mapping.items():
                for key in possible_keys:
                    if key in json_data and json_data[key]:
                        training_data[train_field].append(json_data[key])
                        break
            
            # Добавляем другие поля из json_data, которые не попали в маппинг
            for key, value in json_data.items():
                if key not in ['source_image', 'processed_at', 'fields'] and value and not any(key in possible_keys for possible_keys in mapping.values()):
                    if key not in training_data:
                        training_data[key] = []
                    training_data[key].append(value)
        
        # Добавляем метаданные
        if 'source_image' in json_data:
            training_data['source_image'] = json_data['source_image']
        if 'processed_at' in json_data:
            training_data['processed_at'] = json_data['processed_at']
            
        # Удаляем пустые списки из результата
        return {k: v for k, v in training_data.items() if (isinstance(v, list) and v) or not isinstance(v, list)}

    def process_file(self, input_data, custom_prompt=None):
        """
        Обрабатывает входные данные (текст или изображение) с помощью Gemini API.
        
        Args:
            input_data: Строка текста или путь к изображению/объект PIL.Image
            custom_prompt: Пользовательский промпт для модели
            
        Returns:
            dict: Обработанные данные или словарь с ошибкой
        """
        if not self.is_loaded or not self.model:
            if not self.load_model(self.model_id):
                print("ОШИБКА: GeminiProcessor.process_file: Не удалось загрузить/сконфигурировать модель Gemini.")
                return {"note_gemini": "ОШИБКА: Gemini процессор не готов. Проверьте API ключ и настройки."}

        try:
            # Определяем тип входных данных
            is_image = isinstance(input_data, (str, Image.Image))
            is_text = isinstance(input_data, str) and not (input_data.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')))
            
            if not is_image and not is_text:
                print(f"ОШИБКА: Неподдерживаемый тип входных данных: {type(input_data)}")
                return {"note_gemini": "ОШИБКА: Неподдерживаемый формат входных данных"}
            
            # Получаем базовый промпт
            base_prompt = custom_prompt if custom_prompt else settings_manager.get_string(
                'Prompts', 
                'gemini_extract_prompt', 
                app_config.GEMINI_ANNOTATION_PROMPT_DEFAULT
            )
            
            # Для текстовых данных
            if is_text:
                if not input_data.strip():
                    return {"note_gemini": "ОШИБКА: Пустой текст"}
                
                # Подготавливаем полный промпт
                full_prompt = self.get_full_prompt(
                    custom_prompt_text=base_prompt,
                    image_name="текстовые данные"
                )
                
                try:
                    # Отправляем запрос к модели с явным указанием JSON формата ответа
                    generation_config = self._get_generation_config()
                    safety_settings = self._get_safety_settings()
                    
                    response = self.model.generate_content(
                        [full_prompt, input_data],
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Проверяем ответ
                    if not response:
                        print("ОШИБКА: Нет ответа от модели Gemini")
                        return {"note_gemini": "ОШИБКА: Нет ответа от модели"}
                    
                    # Извлекаем текст из ответа
                    response_text = ""
                    if hasattr(response, 'text'):
                        response_text = response.text
                    elif hasattr(response, 'parts') and response.parts:
                        response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                    else:
                        response_text = str(response)
                    
                    if not response_text:
                        print("ОШИБКА: Пустой текст ответа от модели Gemini")
                        return {"note_gemini": "ОШИБКА: Пустой текст ответа от модели"}
                    
                    # Пытаемся разобрать JSON - теперь API должен возвращать чистый JSON
                    try:
                        # Сначала пробуем напрямую разобрать текст как JSON
                        result = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        print(f"ОШИБКА разбора JSON: {str(e)}")
                        print(f"Сырой ответ: {response_text[:200]}...")
                        
                        # Если прямой разбор не удался, пробуем очистить текст
                        try:
                            json_str = self._clean_json_string(response_text)
                            result = json.loads(json_str)
                        except (ValueError, json.JSONDecodeError) as e2:
                            print(f"ОШИБКА очистки и разбора JSON: {str(e2)}")
                            return {"note_gemini": f"ОШИБКА: Не удалось разобрать JSON ответ от модели. {str(e)}"}
                    
                    # Проверяем, что результат это словарь
                    if not isinstance(result, dict):
                        print(f"ОШИБКА: Неверный формат ответа от модели: {type(result)}")
                        # Если результат не словарь, попробуем вернуть его как строку в словаре
                        if isinstance(result, str):
                            return {"raw_text": result, "note_gemini": "Получена строка вместо структурированных данных"}
                        return {"note_gemini": f"ОШИБКА: Неверный формат ответа. Получен {type(result)} вместо словаря."}
                    
                    # Преобразуем в формат для сопоставления с OCR
                    return self._convert_to_training_format(result)
                    
                except Exception as e:
                    print(f"ОШИБКА при обработке текста через Gemini: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    return {"note_gemini": f"ОШИБКА обработки: {str(e)}"}
            
            # Для изображений
            else:
                result = self.process(input_data)
                # Убедимся, что результат это словарь
                if not isinstance(result, dict):
                    print(f"ОШИБКА: process вернул не словарь, а {type(result)}: {result}")
                    return {"note_gemini": f"ОШИБКА: Неверный формат результата от process"}
                    
                # Проверяем наличие ошибки
                if "note_gemini" in result and "ОШИБКА" in result["note_gemini"]:
                    return result
                    
                # Преобразуем в формат для обучения
                return self._convert_to_training_format(result)
                
        except Exception as e:
            print(f"ОШИБКА в GeminiProcessor.process_file: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {"note_gemini": f"ОШИБКА: {str(e)}"}
            
        finally:
            # Очищаем временные файлы
            self.cleanup_temp_files()

    def _get_prompt(self):
        """
        Получает промпт для Gemini API из FieldManager или настроек приложения.
        
        Returns:
            str: Промпт для модели
        """
        # Проверяем, есть ли переопределенный промпт
        if hasattr(self, '_custom_prompt_override') and self._custom_prompt_override:
            log_message(f"Используется пользовательский промпт ({len(self._custom_prompt_override)} символов)")
            return self._custom_prompt_override
        
        # Пытаемся использовать FieldManager для генерации промпта
        try:
            from .field_manager import field_manager
            
            # Получаем базовый промпт из настроек (если есть)
            custom_prompt = settings_manager.get_string('Prompts', 'gemini_extract_prompt', "")
            
            # Генерируем промпт на основе активных полей
            generated_prompt = field_manager.get_gemini_prompt(custom_prompt)
            
            log_message(f"Промпт сгенерирован FieldManager ({len(generated_prompt)} символов)")
            return generated_prompt
            
        except ImportError:
            log_message("FieldManager недоступен, используется fallback промпт")
            
        # Fallback на статический промпт если FieldManager недоступен
        default_prompt = """
        Проанализируй изображение счета или фактуры и извлеки следующие поля в JSON формате.

        Извлеки эти данные (если присутствуют в документе):
        - Поставщик: название компании-поставщика
        - № Счета: номер счета/инвойса
        - Дата счета: дата выставления счета
        - Категория: тип товаров/услуг
        - Товары: описание товаров/услуг с количеством и ценами
        - Сумма без НДС: сумма без налога
        - НДС %: ставка НДС в процентах
        - Сумма с НДС: итоговая сумма с налогом
        - Валюта: валюта (RUB, USD, EUR и т.д.)
        - ИНН: ИНН поставщика
        - КПП: КПП поставщика
        - Комментарии: дополнительные примечания
        
        ВАЖНО: Возвращай ответ ТОЛЬКО в формате JSON. Используй точные названия полей как показано выше.
        Если поле не найдено в документе, используй "N/A".
        Не добавляй никаких дополнительных объяснений или текста вне JSON.
        """
        
        # Получаем промпт из настроек приложения, если доступно
        final_prompt = settings_manager.get_string('Prompts', 'gemini_extract_prompt', default_prompt)
        
        # Логирование финального промпта
        log_message(f"Fallback промпт для Gemini ({len(final_prompt)} символов): {final_prompt[:200]}...")
        
        return final_prompt
    
    def _get_safety_settings(self):
        """
        Получает настройки безопасности для Gemini API.
        
        Returns:
            list: Настройки безопасности
        """
        # Стандартные настройки безопасности с низким порогом блокировки
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    
    def _get_generation_config(self):
        """
        Получает конфигурацию генерации для Gemini API.
        
        Returns:
            dict: Конфигурация генерации
        """
        # Стандартная конфигурация для структурированных ответов
        temperature = settings_manager.get_float('Models', 'gemini_temperature', 0.2)
        max_tokens = settings_manager.get_int('Models', 'gemini_max_tokens', 1024)
        
        return {
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": int(max_tokens),
            "response_mime_type": "application/json"  # Запрашиваем JSON напрямую
        }
    
    def _extract_json_from_response(self, response_text):
        """
        Извлекает JSON данные из текстового ответа Gemini API.
        
        Args:
            response_text: Текстовый ответ от API
            
        Returns:
            dict: Извлеченные JSON данные
        """
        try:
            # Пытаемся напрямую разобрать полный текст как JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Если не удалось, ищем JSON в тексте
                pass
                
            # Пытаемся найти JSON между фигурными скобками
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
                
            # Пытаемся найти JSON между тройными обратными кавычками
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
                        
            # Пытаемся найти JSON между обычными обратными кавычками
            json_pattern = r'```\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
                        
            # Если все методы не сработали, очищаем текст и пробуем еще раз
            cleaned_text = self._clean_json_string(response_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Если все методы извлечения JSON не сработали, создаем базовую структуру
                log_message("ПРЕДУПРЕЖДЕНИЕ: Не удалось извлечь JSON из ответа API. Создаю базовую структуру.")
                
                # Получаем хотя бы текстовый ответ
                return {
                    "fields": [
                        {
                            "field_name": "raw_text",
                            "field_value": response_text
                        }
                    ],
                    "error": "JSON extraction failed"
                }
                
        except Exception as e:
            log_message(f"Ошибка при извлечении JSON из ответа: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
            
            # Возвращаем базовую структуру с информацией об ошибке
            return {
                "fields": [],
                "error": f"JSON extraction error: {str(e)}"
            }
    
    def _standardize_response_format(self, data):
        """
        Стандартизирует формат ответа от Gemini API для последующей обработки.
        
        Args:
            data: Данные, извлеченные из ответа API
            
        Returns:
            dict: Стандартизированные данные
        """
        try:
            # Проверяем, что data - словарь
            if not isinstance(data, dict):
                log_message(f"ОШИБКА: Входные данные не являются словарем: {type(data)}")
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        return {"fields": [], "error": "Invalid data format"}
                else:
                    return {"fields": [], "error": "Invalid data format"}
            
            # НОВАЯ ЛОГИКА: Если data уже содержит прямой JSON от Gemini с response_mime_type="application/json"
            # и НЕ содержит структуру fields, то это правильный формат - преобразуем его в структуру fields
            if "fields" not in data and len(data) > 0:
                log_message(f"ОТЛАДКА: Обнаружен прямой JSON ответ от Gemini с {len(data)} полями")
                
                # Преобразуем прямой JSON в структуру fields
                fields = []
                for key, value in data.items():
                    # Пропускаем служебные поля
                    if key in ['source_image', 'processed_at', 'error']:
                        continue
                        
                    fields.append({
                        "field_name": key,
                        "field_value": str(value) if value is not None else ""
                    })
                
                log_message(f"ОТЛАДКА: Преобразовано {len(fields)} полей в структуру fields")
                
                # Создаем новую структуру данных
                standardized_data = {"fields": fields}
                
                # Сохраняем служебные поля
                for key in ['source_image', 'processed_at', 'error']:
                    if key in data:
                        standardized_data[key] = data[key]
                
                return standardized_data
            
            # Если уже есть структура fields, проверяем её корректность
            if "fields" in data:
                # Проверяем, что fields - список
                if not isinstance(data["fields"], list):
                    log_message(f"ОШИБКА: 'fields' не является списком: {type(data['fields'])}")
                    # Если fields не список, пытаемся преобразовать
                    if isinstance(data["fields"], dict):
                        fields = []
                        for key, value in data["fields"].items():
                            fields.append({
                                "field_name": key,
                                "field_value": value
                            })
                        data["fields"] = fields
                    else:
                        data["fields"] = []
                
                # Проверяем каждое поле на корректный формат
                standardized_fields = []
                for field in data["fields"]:
                    if not isinstance(field, dict):
                        continue
                        
                    # Проверяем наличие необходимых ключей
                    field_name = None
                    field_value = None
                    
                    if "field_name" in field and "field_value" in field:
                        field_name = field["field_name"]
                        field_value = field["field_value"]
                    elif "name" in field and "value" in field:
                        field_name = field["name"]
                        field_value = field["value"]
                    elif "type" in field and "text" in field:
                        field_name = field["type"]
                        field_value = field["text"]
                    elif "key" in field and "value" in field:
                        field_name = field["key"]
                        field_value = field["value"]
                    # Если есть только один ключ в поле, пытаемся использовать его
                    elif len(field) == 1:
                        key = list(field.keys())[0]
                        field_name = key
                        field_value = field[key]
                        
                    # Если не удалось получить имя и значение поля, пропускаем его
                    if field_name is None or field_value is None:
                        continue
                    
                    # Преобразуем в строки, если требуется
                    field_name = str(field_name).strip()
                    field_value = str(field_value).strip()
                    
                    # Добавляем поле в стандартизированный список
                    standardized_fields.append({
                        "field_name": field_name,
                        "field_value": field_value
                    })
                
                # Обновляем список полей
                data["fields"] = standardized_fields
                
                return data
            
            # Если нет структуры fields и нет других полей, создаем пустую структуру
            log_message("ПРЕДУПРЕЖДЕНИЕ: Не найдены поля для обработки")
            return {"fields": [], "error": "No extractable fields found"}
            
        except Exception as e:
            log_message(f"Ошибка при стандартизации формата ответа: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
            
            # Возвращаем базовую структуру с информацией об ошибке
            return {
                "fields": [],
                "error": f"Format standardization error: {str(e)}"
            }

    def process_image(self, image_path: str, ocr_lang=None, custom_prompt=None) -> Optional[Dict]:
        """
        Обрабатывает изображение с помощью Gemini API и извлекает поля из счёта.
        Это обертка над методом process() для совместимости с BaseProcessor.
        
        Args:
            image_path: Путь к файлу изображения
            ocr_lang: Язык OCR (не используется для Gemini, но нужен для совместимости)
            custom_prompt: Пользовательский промпт (будет использован если передан)
            
        Returns:
            Optional[Dict]: Извлеченные данные или None в случае ошибки
        """
        # Если передан custom_prompt, сохраняем его для использования в _get_prompt()
        if custom_prompt:
            self._custom_prompt_override = custom_prompt
        else:
            self._custom_prompt_override = None
            
        try:
            return self.process(image_path)
        finally:
            # Очищаем временный промпт после обработки
            self._custom_prompt_override = None

    def _get_api_key(self):
        """Получение API ключа из настроек безопасным способом."""
        # Используем новую систему безопасности через SettingsManager
        api_key = settings_manager.get_encrypted_setting('google_api_key')
        
        if not api_key:
            # Fallback на переменную окружения
            api_key = os.environ.get('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ Google API ключ не найден. Установите его через настройки или переменную окружения GOOGLE_API_KEY")
        
        return api_key
    
    def _get_model_id_from_settings(self):
        """Получает ID модели из настроек."""
        return settings_manager.get_string('Gemini', 'sub_model_id', 'models/gemini-2.0-flash')

    def _convert_pdf_to_images(self, pdf_path):
        """Конвертирует PDF в изображения."""
        try:
            print(f"Конвертация PDF в изображения: {pdf_path}")
            
            # Используем менеджер ресурсов для временной директории
            if self.resource_manager:
                with self.resource_manager.temp_directory(prefix="pdf_images_") as temp_dir:
                    images = self._convert_pdf_internal(pdf_path, temp_dir)
                    return images
            else:
                # Fallback на старую систему
                temp_dir = tempfile.mkdtemp(prefix="pdf_images_", dir=app_config.TEMP_PATH)
                self.temp_dirs.append(temp_dir)
                return self._convert_pdf_internal(pdf_path, temp_dir)
                
        except Exception as e:
            print(f"Ошибка при конвертации PDF: {e}")
            return []
    
    def _convert_pdf_internal(self, pdf_path, temp_dir):
        """Внутренняя логика конвертации PDF."""
        # Проверяем настройки poppler
        poppler_path = settings_manager.get_poppler_path() if hasattr(settings_manager, 'get_poppler_path') else app_config.POPPLER_PATH
        
        # Конвертируем PDF в изображения с правильным DPI
        dpi = app_config.GEMINI_PDF_DPI if hasattr(app_config, 'GEMINI_PDF_DPI') else 200
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        
        # Сохраняем изображения
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(image_path, "JPEG", quality=95)
            if not self.resource_manager:
                self.temp_files.append(image_path)
            image_paths.append(image_path)
            print(f"  Страница {i+1} сохранена: {image_path}")
        
        return image_paths

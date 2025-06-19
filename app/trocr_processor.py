"""
TrOCR Processor для извлечения данных из счетов с использованием Microsoft TrOCR
"""

import os
import json
import logging
import torch
from PIL import Image
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

try:
    from transformers import TrOCRProcessor as HfTrOCRProcessor, VisionEncoderDecoderModel
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Импортируем базовый процессор для интеграции
try:
    from .base_processor import BaseProcessor
except ImportError:
    # Fallback если base_processor недоступен
    class BaseProcessor:
        def __init__(self):
            self.is_loaded = False
            self.device = None

class TrOCRProcessor(BaseProcessor):
    """
    Процессор для извлечения данных из изображений документов с использованием Microsoft TrOCR
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str = "auto"):
        """
        Инициализация TrOCR процессора
        
        Args:
            model_name: Имя модели TrOCR
            device: Устройство для обработки ('cuda', 'cpu', 'auto')
        """
        super().__init__()  # Инициализируем BaseProcessor
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers не установлен. Выполните: pip install transformers torch")
            
        # Настройка устройства
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        self.model_id_loaded = None  # Для отслеживания загруженной модели
        self.processor = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.is_loaded = False  # Флаг загрузки модели
        
        # Кэш директория
        self.cache_dir = "data/models"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Интеграция с field_manager для структурированного извлечения
        try:
            from .field_manager import field_manager
            self.field_manager = field_manager
        except ImportError:
            self.field_manager = None
        
        self.logger.info(f"TrOCRProcessor инициализирован с моделью: {model_name}")
        self.logger.info(f"Устройство: {self.device}")
    
    def load_model(self) -> bool:
        """
        Загружает модель TrOCR
        
        Returns:
            bool: True если модель загружена успешно
        """
        try:
            self.logger.info(f"Загрузка TrOCR модели: {self.model_name}")
            
            # Загружаем процессор
            self.processor = HfTrOCRProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Загружаем модель
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Перемещаем на устройство
            self.model.to(self.device)
            self.model.eval()
            
            self.model_id_loaded = self.model_name
            self.is_loaded = True
            self.logger.info("TrOCR модель успешно загружена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки TrOCR модели: {e}")
            return False
    
    def extract_text_from_image(self, image: Union[str, Image.Image]) -> str:
        """
        Извлекает текст из изображения с помощью TrOCR
        
        Args:
            image: Путь к изображению или PIL Image
            
        Returns:
            str: Извлеченный текст
        """
        if not self.model or not self.processor:
            if not self.load_model():
                raise RuntimeError("Не удалось загрузить TrOCR модель")
        
        try:
            # Загружаем изображение если это путь
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("image должен быть путем к файлу или PIL Image")
            
            # Обрабатываем изображение
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Генерируем текст
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Декодируем текст
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения текста TrOCR: {e}")
            return ""
    
    def process_invoice_image(self, image_path: str, fields: List[str] = None) -> Dict[str, Any]:
        """
        Обрабатывает изображение счета и извлекает структурированные данные
        
        Args:
            image_path: Путь к изображению счета
            fields: Список полей для извлечения (если None, извлекается весь текст)
            
        Returns:
            Dict[str, Any]: Извлеченные данные
        """
        try:
            # Проверяем файл
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            self.logger.info(f"Обработка изображения: {image_path}")
            
            # Извлекаем текст
            extracted_text = self.extract_text_from_image(image_path)
            
            if not extracted_text:
                return {
                    'success': False,
                    'error': 'Не удалось извлечь текст из изображения',
                    'raw_text': '',
                    'extracted_fields': {}
                }
            
            # Если поля не указаны, возвращаем весь текст
            if not fields:
                return {
                    'success': True,
                    'raw_text': extracted_text,
                    'extracted_fields': {'full_text': extracted_text},
                    'processing_time': datetime.now().isoformat()
                }
            
            # Извлекаем конкретные поля
            extracted_fields = self._extract_fields_from_text(extracted_text, fields)
            
            return {
                'success': True,
                'raw_text': extracted_text,
                'extracted_fields': extracted_fields,
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения счета: {e}")
            return {
                'success': False,
                'error': str(e),
                'raw_text': '',
                'extracted_fields': {}
            }
    
    def _extract_fields_from_text(self, text: str, fields: List[str]) -> Dict[str, str]:
        """
        Извлекает конкретные поля из текста с помощью структурированного подхода
        
        Args:
            text: Исходный текст
            fields: Список полей для извлечения
            
        Returns:
            Dict[str, str]: Извлеченные поля
        """
        extracted = {}
        
        # Если модель дообучена для структурированного извлечения, используем её
        if self._is_fine_tuned_model():
            # Для дообученной модели используем специальный промпт
            return self._extract_with_fine_tuned_model(text, fields)
        
        # Для базовой модели используем интеллектуальный парсинг
        # Разбиваем текст на строки для анализа
        lines = text.split('\n')
        
        # Словарь для хранения потенциальных значений с их вероятностями
        field_candidates = {field: [] for field in fields}
        
        # Анализируем каждую строку
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Проверяем каждое поле
            for field in fields:
                confidence = 0.0
                value = ""
                
                if field == 'invoice_number':
                    # Ищем номер счета
                    if any(marker in line.lower() for marker in ['invoice', 'счет', '№', 'номер', 'no.']):
                        # Извлекаем числовую или буквенно-числовую часть
                        import re
                        matches = re.findall(r'[A-Za-z0-9\-/]+', line)
                        for match in matches:
                            if len(match) > 3 and any(c.isdigit() for c in match):
                                value = match
                                confidence = 0.9
                                break
                                
                elif field == 'date':
                    # Ищем дату
                    import re
                    date_patterns = [
                        r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}',
                        r'\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2}'
                    ]
                    for pattern in date_patterns:
                        match = re.search(pattern, line)
                        if match:
                            value = match.group()
                            confidence = 0.95
                            break
                            
                elif field in ['total_amount', 'vat_amount']:
                    # Ищем суммы
                    if any(marker in line.lower() for marker in ['total', 'итого', 'всего', 'сумма', 'amount', 'ндс', 'vat']):
                        import re
                        # Ищем числа с десятичными знаками
                        matches = re.findall(r'[\d\s]+[,.]?\d*', line)
                        for match in matches:
                            cleaned = match.replace(' ', '')
                            if cleaned:
                                value = cleaned
                                confidence = 0.85
                                if 'total' in line.lower() or 'итого' in line.lower():
                                    confidence = 0.95
                                break
                                
                elif field == 'vendor_name':
                    # Ищем название поставщика
                    if any(marker in line.lower() for marker in ['from', 'от', 'поставщик', 'vendor', 'company', 'ооо', 'зао', 'ип']):
                        # Убираем маркеры и берем оставшуюся часть
                        value = line
                        for marker in ['from:', 'от:', 'поставщик:', 'vendor:', 'company:']:
                            if marker in line.lower():
                                value = line.split(marker, 1)[-1].strip()
                                break
                        if value and len(value) > 2:
                            confidence = 0.8
                            
                elif field == 'description':
                    # Описание обычно идет после заголовков
                    if any(marker in line.lower() for marker in ['товар', 'услуга', 'наименование', 'description', 'item']):
                        # Следующие строки могут содержать описание
                        value = line
                        confidence = 0.7
                
                # Добавляем кандидата если есть значение
                if value and confidence > 0:
                    field_candidates[field].append({
                        'value': value,
                        'confidence': confidence,
                        'line': line
                    })
        
        # Выбираем лучшего кандидата для каждого поля
        for field, candidates in field_candidates.items():
            if candidates:
                # Сортируем по уверенности и берем лучший
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                extracted[field] = best_candidate['value']
            else:
                extracted[field] = ""
                
        # Постобработка для улучшения качества
        extracted = self._postprocess_extracted_fields(extracted)
        
        return extracted
    
    def _is_fine_tuned_model(self) -> bool:
        """Проверяет, является ли модель дообученной"""
        # Проверяем по пути модели
        if self.model_id_loaded and 'trained_models' in self.model_id_loaded:
            return True
        return False
    
    def _extract_with_fine_tuned_model(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Использует дообученную модель для структурированного извлечения"""
        # Формируем специальный промпт для дообученной модели
        prompt = f"Extract the following fields from the invoice: {', '.join(fields)}\n\nText: {text}\n\nExtracted fields:"
        
        # Здесь должна быть логика использования дообученной модели
        # Пока используем базовый подход
        return {field: "" for field in fields}
    
    def _postprocess_extracted_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """Постобработка извлеченных полей для улучшения качества"""
        # Очищаем и форматируем значения
        for field, value in fields.items():
            if field == 'date' and value:
                # Нормализуем формат даты
                import re
                value = re.sub(r'[./\-]', '.', value)
                
            elif field in ['total_amount', 'vat_amount'] and value:
                # Нормализуем числовые значения
                value = value.replace(',', '.')
                value = re.sub(r'[^\d.]', '', value)
                
            elif field == 'vendor_name' and value:
                # Очищаем название компании
                value = value.strip('"\'')
                
            fields[field] = value
            
        return fields
    
    def process_multiple_images(self, image_paths: List[str], fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Обрабатывает несколько изображений счетов
        
        Args:
            image_paths: Список путей к изображениям
            fields: Список полей для извлечения
            
        Returns:
            List[Dict[str, Any]]: Результаты обработки каждого изображения
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Обработка изображения {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_invoice_image(image_path, fields)
            result['image_path'] = image_path
            result['image_index'] = i
            
            results.append(result)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Возвращает список доступных TrOCR моделей
        
        Returns:
            List[str]: Список названий моделей
        """
        return [
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-base-stage1", 
            "microsoft/trocr-large-printed",
            "microsoft/trocr-large-handwritten"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущей модели
        
        Returns:
            Dict[str, Any]: Информация о модели
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'processor_loaded': self.processor is not None,
            'available_models': self.get_available_models()
        }
    
    def unload_model(self):
        """Выгружает модель из памяти"""
        if self.model:
            del self.model
            self.model = None
            
        if self.processor:
            del self.processor
            self.processor = None
            
        # Очищаем GPU память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("TrOCR модель выгружена из памяти")
    
    def process_image(self, image_path: str, ocr_lang: str = None, custom_prompt: str = None) -> Dict[str, Any]:
        """
        Основной метод обработки изображения для интеграции с системой
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR (не используется в TrOCR, для совместимости)
            custom_prompt: Кастомный промпт (для будущих версий)
            
        Returns:
            Dict с результатами обработки
        """
        try:
            # Загружаем модель если не загружена
            if not self.is_loaded:
                if not self.load_model():
                    return {"error": "Не удалось загрузить модель TrOCR"}
                    
            # Получаем поля для извлечения из field_manager
            if self.field_manager:
                fields_to_extract = [field.id for field in self.field_manager.get_enabled_fields()]
            else:
                # Базовый набор полей
                fields_to_extract = [
                    'invoice_number', 'date', 'total_amount', 
                    'vendor_name', 'vat_amount', 'description'
                ]
            
            # Обрабатываем изображение
            result = self.process_invoice_image(image_path, fields_to_extract)
            
            # Форматируем результат для совместимости с системой
            if result['success']:
                return result['extracted_fields']
            else:
                return {"error": result.get('error', 'Неизвестная ошибка')}
                
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения: {e}")
            return {"error": str(e)}
    
    def get_full_prompt(self, custom_prompt: str = None) -> str:
        """
        Возвращает полный промпт для модели (для совместимости)
        
        Args:
            custom_prompt: Кастомный промпт
            
        Returns:
            str: Промпт
        """
        if custom_prompt:
            return custom_prompt
            
        if self.field_manager:
            # Генерируем промпт на основе включенных полей
            fields = self.field_manager.get_enabled_fields()
            prompt = "Извлеките следующие поля из счета:\n"
            for field in fields:
                prompt += f"- {field.display_name}: {field.description}\n"
            return prompt
        else:
            return "Извлеките данные счета: номер, дата, сумма, поставщик"
    
    def unload_model(self):
        """Выгружает модель из памяти"""
        if self.model:
            del self.model
            self.model = None
            
        if self.processor:
            del self.processor
            self.processor = None
            
        # Очищаем GPU память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_loaded = False
        self.model_id_loaded = None
        self.logger.info("TrOCR модель выгружена из памяти")
    
    def __del__(self):
        """Деструктор для очистки ресурсов"""
        try:
            self.unload_model()
        except:
            pass


# Тестирование
if __name__ == "__main__":
    # Пример использования
    processor = TrOCRProcessor()
    
    # Загружаем модель
    if processor.load_model():
        print("✅ TrOCR модель загружена успешно")
        
        # Получаем информацию о модели
        info = processor.get_model_info()
        print(f"📊 Информация о модели: {info}")
        
        # Пример обработки (если есть изображение)
        # result = processor.process_invoice_image("test_invoice.jpg", ['invoice_number', 'total_amount', 'date'])
        # print(f"Результат: {result}")
        
    else:
        print("❌ Не удалось загрузить TrOCR модель") 
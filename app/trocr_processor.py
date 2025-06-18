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
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TrOCRProcessor:
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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers не установлен. Выполните: pip install transformers torch")
            
        # Настройка устройства
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Кэш директория
        self.cache_dir = "data/models"
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
            self.processor = TrOCRProcessor.from_pretrained(
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
        Извлекает конкретные поля из текста с помощью регулярных выражений
        
        Args:
            text: Исходный текст
            fields: Список полей для извлечения
            
        Returns:
            Dict[str, str]: Извлеченные поля
        """
        extracted = {}
        
        # Паттерны для извлечения полей
        patterns = {
            'invoice_number': [
                r'(?:invoice|счет|№)\s*[:№#]?\s*([A-Za-z0-9\-/_]+)',
                r'№\s*([A-Za-z0-9\-/_]+)',
                r'Invoice\s*#?\s*([A-Za-z0-9\-/_]+)'
            ],
            'date': [
                r'(?:date|дата)\s*[:.]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
                r'(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
                r'(\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2})'
            ],
            'total_amount': [
                r'(?:total|итого|всего)\s*[:.]?\s*([0-9,.\s]+)',
                r'(?:sum|сумма)\s*[:.]?\s*([0-9,.\s]+)',
                r'(\d+[,.]?\d*)\s*(?:руб|рублей|$|€|₽)'
            ],
            'vendor_name': [
                r'(?:from|от|поставщик)\s*[:.]?\s*([^\n\r]+)',
                r'(?:vendor|company)\s*[:.]?\s*([^\n\r]+)'
            ],
            'description': [
                r'(?:description|описание|наименование)\s*[:.]?\s*([^\n\r]+)',
                r'(?:item|товар)\s*[:.]?\s*([^\n\r]+)'
            ]
        }
        
        # Извлекаем каждое поле
        for field in fields:
            field_lower = field.lower()
            extracted[field] = ""
            
            # Ищем подходящие паттерны
            for pattern_key, pattern_list in patterns.items():
                if field_lower in pattern_key or pattern_key in field_lower:
                    for pattern in pattern_list:
                        import re
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            extracted[field] = match.group(1).strip()
                            break
                    if extracted[field]:
                        break
        
        return extracted
    
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
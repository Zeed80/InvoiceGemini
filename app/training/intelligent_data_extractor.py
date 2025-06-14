"""
Интеллектуальный экстрактор данных для InvoiceGemini
Использует Gemini для извлечения ВСЕХ полезных данных из документов
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from pathlib import Path

@dataclass
class ExtractedField:
    """Класс для представления извлеченного поля"""
    name: str                    # Название поля (например, "номер_счета")
    value: str                   # Значение поля
    confidence: float            # Уверенность в извлечении (0.0-1.0)
    field_type: str             # Тип поля (text, number, date, amount, etc.)
    category: str               # Категория (invoice_info, company_info, items, etc.)
    coordinates: Optional[Dict] = None  # Координаты на документе (если доступны)
    
class IntelligentDataExtractor:
    """
    Интеллектуальный экстрактор данных с использованием Gemini
    """
    
    def __init__(self, gemini_processor, logger=None):
        self.gemini_processor = gemini_processor
        self.logger = logger or logging.getLogger(__name__)
        
        # Промпт для извлечения всех полезных данных
        self.extraction_prompt = """
Проанализируй этот документ (счет, накладная, договор и т.д.) и извлеки ВСЕ полезные структурированные данные.

ВАЖНО: Не ограничивайся только основными полями - извлекай ВСЕ данные, которые могут быть полезны для автоматизации документооборота.

Верни результат в формате JSON со следующей структурой:

{
  "document_type": "тип документа (счет, накладная, договор, и т.д.)",
  "extracted_fields": [
    {
      "name": "название_поля_на_русском",
      "value": "значение",
      "confidence": 0.95,
      "field_type": "text|number|date|amount|email|phone|address|tax_id|etc",
      "category": "invoice_info|company_info|client_info|items|payment_info|legal_info|etc"
    }
  ]
}

КАТЕГОРИИ ПОЛЕЙ:
- invoice_info: номер счета, дата, срок оплаты, валюта
- company_info: название компании, ИНН, КПП, адрес, телефон, email поставщика
- client_info: данные клиента/покупателя
- items: товары, услуги, количество, цены
- payment_info: банковские реквизиты, способы оплаты
- amounts: суммы, НДС, итого
- legal_info: юридическая информация, подписи, печати
- logistics: адреса доставки, сроки, условия
- other: прочие полезные данные

ТИПЫ ПОЛЕЙ:
- text: обычный текст
- number: числовые значения
- date: даты
- amount: денежные суммы
- email: email адреса
- phone: телефоны
- address: адреса
- tax_id: налоговые номера (ИНН, КПП)
- bank_account: банковские счета
- percentage: проценты

Извлекай максимально подробно - лучше больше данных, чем меньше!
"""

    def extract_all_data(self, image_path: str) -> Dict[str, Any]:
        """
        Извлекает все полезные данные из документа
        
        Args:
            image_path: Путь к изображению документа
            
        Returns:
            Словарь с извлеченными данными
        """
        try:
            self.logger.info(f"🧠 Начинаем интеллектуальное извлечение данных из: {image_path}")
            
            # Отправляем запрос к Gemini
            response = self.gemini_processor.process_image_with_prompt(
                image_path, 
                self.extraction_prompt
            )
            
            if not response:
                self.logger.error("❌ Gemini не вернул ответ")
                return self._create_empty_result()
            
            # Парсим JSON ответ
            extracted_data = self._parse_gemini_response(response)
            
            if not extracted_data:
                self.logger.error("❌ Не удалось распарсить ответ Gemini")
                return self._create_empty_result()
            
            # Обрабатываем и валидируем данные
            processed_data = self._process_extracted_data(extracted_data)
            
            self.logger.info(f"✅ Извлечено {len(processed_data.get('fields', []))} полей")
            self._log_extraction_summary(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка при извлечении данных: {e}")
            return self._create_empty_result()
    
    def _parse_gemini_response(self, response: str) -> Optional[Dict]:
        """Парсит ответ от Gemini"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                self.logger.error("❌ JSON не найден в ответе Gemini")
                return None
            
            json_str = json_match.group(0)
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Ошибка парсинга JSON: {e}")
            # Пытаемся исправить частые ошибки JSON
            return self._try_fix_json(response)
        except Exception as e:
            self.logger.error(f"❌ Неожиданная ошибка парсинга: {e}")
            return None
    
    def _try_fix_json(self, response: str) -> Optional[Dict]:
        """Пытается исправить поврежденный JSON"""
        try:
            # Удаляем комментарии и лишние символы
            cleaned = re.sub(r'//.*?\n', '', response)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            
            # Ищем JSON снова
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            self.logger.error(f"❌ Не удалось исправить JSON: {e}")
        
        return None
    
    def _process_extracted_data(self, raw_data: Dict) -> Dict[str, Any]:
        """Обрабатывает и валидирует извлеченные данные"""
        try:
            processed_fields = []
            
            for field_data in raw_data.get('extracted_fields', []):
                field = self._create_extracted_field(field_data)
                if field:
                    processed_fields.append(field)
            
            return {
                'document_type': raw_data.get('document_type', 'unknown'),
                'fields': processed_fields,
                'total_fields': len(processed_fields),
                'extraction_timestamp': datetime.now().isoformat(),
                'categories': self._get_categories_summary(processed_fields)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки данных: {e}")
            return self._create_empty_result()
    
    def _create_extracted_field(self, field_data: Dict) -> Optional[ExtractedField]:
        """Создает объект ExtractedField из данных"""
        try:
            # Валидация обязательных полей
            if not all(key in field_data for key in ['name', 'value']):
                return None
            
            # Нормализация значений
            name = str(field_data['name']).strip().lower()
            value = str(field_data['value']).strip()
            
            if not name or not value:
                return None
            
            # Создаем поле
            field = ExtractedField(
                name=name,
                value=value,
                confidence=float(field_data.get('confidence', 0.8)),
                field_type=field_data.get('field_type', 'text'),
                category=field_data.get('category', 'other')
            )
            
            return field
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания поля: {e}")
            return None
    
    def _get_categories_summary(self, fields: List[ExtractedField]) -> Dict[str, int]:
        """Создает сводку по категориям полей"""
        categories = {}
        for field in fields:
            category = field.category
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _log_extraction_summary(self, data: Dict):
        """Логирует сводку по извлеченным данным"""
        try:
            self.logger.info(f"📊 Сводка извлечения:")
            self.logger.info(f"   📄 Тип документа: {data.get('document_type', 'неизвестно')}")
            self.logger.info(f"   📝 Всего полей: {data.get('total_fields', 0)}")
            
            categories = data.get('categories', {})
            for category, count in categories.items():
                self.logger.info(f"   📂 {category}: {count} полей")
                
            # Показываем примеры полей
            fields = data.get('fields', [])[:5]  # Первые 5 полей
            if fields:
                self.logger.info("   🔍 Примеры полей:")
                for field in fields:
                    self.logger.info(f"      • {field.name}: {field.value[:50]}...")
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка логирования сводки: {e}")
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Создает пустой результат при ошибке"""
        return {
            'document_type': 'unknown',
            'fields': [],
            'total_fields': 0,
            'extraction_timestamp': datetime.now().isoformat(),
            'categories': {}
        }
    
    def convert_to_training_format(self, extracted_data: Dict, ocr_data: List[Dict]) -> Dict[str, Any]:
        """
        Конвертирует извлеченные данные в формат для обучения
        
        Args:
            extracted_data: Данные от интеллектуального экстрактора
            ocr_data: OCR данные с координатами слов
            
        Returns:
            Данные в формате для обучения модели
        """
        try:
            self.logger.info("🔄 Конвертация в формат обучения...")
            
            # Создаем маппинг значений к словам OCR
            field_mappings = self._map_fields_to_ocr(extracted_data['fields'], ocr_data)
            
            # Создаем метки для обучения
            training_labels = self._create_training_labels(field_mappings, ocr_data)
            
            result = {
                'words': [word['text'] for word in ocr_data],
                'bboxes': [word['bbox'] for word in ocr_data],
                'labels': training_labels,
                'field_mappings': field_mappings,
                'extracted_fields_count': len(extracted_data['fields']),
                'document_type': extracted_data.get('document_type', 'unknown')
            }
            
            self.logger.info(f"✅ Создано {len(training_labels)} меток для обучения")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации: {e}")
            return {
                'words': [word['text'] for word in ocr_data],
                'bboxes': [word['bbox'] for word in ocr_data],
                'labels': ['O'] * len(ocr_data),
                'field_mappings': {},
                'extracted_fields_count': 0,
                'document_type': 'unknown'
            }
    
    def _map_fields_to_ocr(self, fields: List[ExtractedField], ocr_data: List[Dict]) -> Dict[str, List[int]]:
        """Сопоставляет извлеченные поля со словами OCR"""
        mappings = {}
        
        for field in fields:
            # Ищем соответствия в OCR данных
            matches = self._find_field_matches(field.value, ocr_data)
            if matches:
                label_name = self._create_label_name(field)
                mappings[label_name] = matches
                
        return mappings
    
    def _find_field_matches(self, field_value: str, ocr_data: List[Dict]) -> List[int]:
        """Находит индексы слов OCR, соответствующих значению поля"""
        matches = []
        field_words = field_value.lower().split()
        
        # Точное совпадение
        for i, word_data in enumerate(ocr_data):
            word = word_data['text'].lower()
            if word in field_words:
                matches.append(i)
        
        # Частичное совпадение для длинных значений
        if not matches and len(field_value) > 10:
            for i, word_data in enumerate(ocr_data):
                word = word_data['text'].lower()
                if len(word) > 3 and word in field_value.lower():
                    matches.append(i)
        
        return matches
    
    def _create_label_name(self, field: ExtractedField) -> str:
        """Создает название метки для поля"""
        # Нормализуем название поля
        normalized_name = field.name.upper().replace(' ', '_').replace('-', '_')
        
        # Добавляем префикс категории если нужно
        if field.category != 'other':
            category_prefix = field.category.upper().replace('_INFO', '')
            if not normalized_name.startswith(category_prefix):
                normalized_name = f"{category_prefix}_{normalized_name}"
        
        return normalized_name
    
    def _create_training_labels(self, field_mappings: Dict[str, List[int]], ocr_data: List[Dict]) -> List[str]:
        """Создает метки для обучения в формате BIO"""
        labels = ['O'] * len(ocr_data)
        
        for label_name, indices in field_mappings.items():
            if not indices:
                continue
                
            # Сортируем индексы
            indices = sorted(indices)
            
            # Первый токен получает метку B- (Beginning)
            labels[indices[0]] = f"B-{label_name}"
            
            # Остальные токены получают метку I- (Inside)
            for idx in indices[1:]:
                labels[idx] = f"I-{label_name}"
        
        return labels
    
    def save_extraction_results(self, results: Dict, output_path: str):
        """Сохраняет результаты извлечения в файл"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Конвертируем ExtractedField объекты в словари
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"💾 Результаты сохранены в: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения: {e}")
    
    def _make_serializable(self, data: Any) -> Any:
        """Делает данные сериализуемыми для JSON"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, ExtractedField):
            return asdict(data)
        else:
            return data 
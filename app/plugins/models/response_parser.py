"""
Улучшенная система парсинга ответов от LLM моделей.
Обрабатывает различные форматы ответов с множественными fallback механизмами.
"""
import json
import re
from typing import Dict, Optional, Any, List


class ResponseParser:
    """Парсер ответов от LLM с адаптивной обработкой"""
    
    @staticmethod
    def extract_json_from_response(response: str, strict: bool = False) -> Optional[Dict]:
        """
        Извлекает JSON из ответа модели различными способами.
        
        Args:
            response: Ответ от модели
            strict: Строгий режим (только чистый JSON)
            
        Returns:
            Optional[Dict]: Извлеченный JSON или None
        """
        if not response or not response.strip():
            return None
        
        # Метод 1: Чистый JSON
        json_obj = ResponseParser._try_parse_clean_json(response)
        if json_obj:
            return json_obj
        
        # Метод 2: JSON в markdown блоке
        json_obj = ResponseParser._try_parse_markdown_json(response)
        if json_obj:
            return json_obj
        
        # Метод 3: JSON в коде (backticks)
        json_obj = ResponseParser._try_parse_code_block_json(response)
        if json_obj:
            return json_obj
        
        if strict:
            return None
        
        # Метод 4: Поиск JSON-подобной структуры
        json_obj = ResponseParser._try_extract_json_pattern(response)
        if json_obj:
            return json_obj
        
        # Метод 5: Исправление частично валидного JSON
        json_obj = ResponseParser._try_fix_broken_json(response)
        if json_obj:
            return json_obj
        
        return None
    
    @staticmethod
    def _try_parse_clean_json(response: str) -> Optional[Dict]:
        """Пробует распарсить чистый JSON"""
        try:
            return json.loads(response.strip())
        except:
            return None
    
    @staticmethod
    def _try_parse_markdown_json(response: str) -> Optional[Dict]:
        """Извлекает JSON из markdown блока ```json ... ```"""
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except:
                continue
        
        return None
    
    @staticmethod
    def _try_parse_code_block_json(response: str) -> Optional[Dict]:
        """Извлекает JSON из блока ``` ... ```"""
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            # Убираем возможный маркер языка
            json_text = re.sub(r'^(json|javascript)\s*\n', '', match.strip(), flags=re.IGNORECASE)
            try:
                return json.loads(json_text)
            except:
                continue
        
        return None
    
    @staticmethod
    def _try_extract_json_pattern(response: str) -> Optional[Dict]:
        """Ищет JSON-подобный паттерн { ... }"""
        # Ищем самый внешний объект { ... }
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, response, re.DOTALL)
        
        # Пробуем от самого длинного к самому короткому
        matches.sort(key=len, reverse=True)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        return None
    
    @staticmethod
    def _try_fix_broken_json(response: str) -> Optional[Dict]:
        """Пытается исправить частично валидный JSON"""
        try:
            # Удаляем комментарии
            cleaned = re.sub(r'//.*?\n', '\n', response)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            
            # Пробуем найти JSON объект
            start = cleaned.find('{')
            if start == -1:
                return None
            
            # Пробуем различные концы
            for end_offset in range(len(cleaned) - 1, start, -1):
                if cleaned[end_offset] == '}':
                    try:
                        potential_json = cleaned[start:end_offset + 1]
                        return json.loads(potential_json)
                    except:
                        continue
            
            return None
        except:
            return None
    
    @staticmethod
    def _is_template_value(value: str, field_name: str = "") -> bool:
        """
        Проверяет, является ли значение шаблонным/плейсхолдером.
        
        Args:
            value: Значение для проверки
            field_name: Название поля (опционально, для контекстной проверки)
            
        Returns:
            bool: True если значение - шаблон
        """
        if not value or not isinstance(value, str):
            return False
        
        value_lower = value.lower().strip()
        
        # Расширенные паттерны шаблонных значений
        template_patterns = [
            r"^1{2,}$",  # 11, 111, 1111
            r"^2{2,}$",  # 22, 222, 2222
            r"^12345+\d*$",  # 12345, 123456, 1234567
            r"^0{2,}$",  # 00, 000, 0000
            r"^n/?a$",
            r"^none$",
            r"^null$",
            r"^undefined$",
            r"^value$",
            r"^значение$",
            r"^example.*$",
            r"^пример.*$",
            r"^test.*$",
            r"^тест.*$",
            r"^sample.*$",
            r"^образец.*$",
            r"^placeholder.*$",
            r"^<actual\s*value.*>$",
            r"^<.*value.*>$",
            r"^\[.*value.*\]$",
        ]
        
        # Проверка по паттернам
        if any(re.fullmatch(pattern, value_lower) for pattern in template_patterns):
            print(f"[TEMPLATE] Обнаружено шаблонное значение по паттерну: '{value}'")
            return True
        
        # Проверка ТОЧНОГО совпадения с названиями полей (с учетом регистра!)
        # Это отловит случаи когда поле вернуло свое же название как значение
        exact_field_names = [
            "Поставщик", "№ счета", "Дата счета", "Description", 
            "Сумма с НДС", "Amount (0% VAT)", "% НДС", "Currency", 
            "Category", "Примечание",
            "sender", "invoice_number", "invoice_date", "description",
            "total", "amount_no_vat", "vat_percent", "currency",
            "category", "note"
        ]
        
        # ТОЧНОЕ совпадение (с учетом регистра) - это шаблон
        if value.strip() in exact_field_names:
            print(f"[TEMPLATE] Значение точно совпадает с ID/названием поля: '{value}'")
            return True
        
        # Также проверяем lowercase варианты общих названий полей
        common_lowercase_fields = [
            "поставщик", "номер счета", "дата счета", "сумма", "total", 
            "amount", "ндс", "vat", "валюта", "категория", "описание",
            "примечание", "note", "comment"
        ]
        
        if value_lower in common_lowercase_fields and len(value_lower) < 20:
            print(f"[TEMPLATE] Значение - общее название поля: '{value}'")
            return True
        
        return False
    
    @staticmethod
    def validate_and_normalize_invoice_data(
        data: Dict,
        required_fields: List[str],
        default_value: str = "N/A"
    ) -> Dict:
        """
        Валидирует и нормализует данные счета.
        
        Args:
            data: Извлеченные данные
            required_fields: Список обязательных полей
            default_value: Значение по умолчанию для отсутствующих полей
            
        Returns:
            Dict: Нормализованные данные
        """
        normalized = {}
        
        for field in required_fields:
            value = data.get(field, default_value)
            
            # Нормализация значения
            if value is None or value == "":
                value = default_value
            elif isinstance(value, (int, float)):
                value = str(value)
            elif not isinstance(value, str):
                value = str(value)
            
            # Очистка значения
            value = value.strip()
            
            # Замена пустых значений
            if value == "" or value.lower() in ["none", "null", "undefined"]:
                value = default_value
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем на шаблонные значения
            if ResponseParser._is_template_value(value, field):
                print(f"[VALIDATION] Поле '{field}' содержит шаблонное значение '{value}', заменяем на '{default_value}'")
                value = default_value
            
            normalized[field] = value
        
        return normalized
    
    @staticmethod
    def smart_parse_llm_response(
        response: str,
        required_fields: List[str],
        model_name: str = "unknown"
    ) -> Dict:
        """
        Умный парсинг ответа LLM с адаптивной обработкой.
        
        Args:
            response: Ответ от модели
            required_fields: Список обязательных полей
            model_name: Название модели (для адаптивной обработки)
            
        Returns:
            Dict: Извлеченные данные
        """
        # Попытка извлечь JSON
        json_data = ResponseParser.extract_json_from_response(response)
        
        if json_data:
            # Валидация и нормализация
            normalized = ResponseParser.validate_and_normalize_invoice_data(
                json_data,
                required_fields
            )

            # Если все поля выглядят как шаблонные значения, возвращаем сообщение об ошибке
            if ResponseParser._looks_like_dummy_payload(normalized):
                return {"error": "LLM вернуло шаблонные значения", "raw_response": response[:500]}

            return normalized
        
        # Если JSON не найден, пытаемся извлечь данные из текста
        print(f"[PARSER] JSON не найден, пробуем текстовое извлечение для {model_name}")
        return ResponseParser._extract_from_text(response, required_fields)
    
    @staticmethod
    def _extract_from_text(response: str, required_fields: List[str]) -> Dict:
        """Извлекает данные из текстового ответа"""
        result = {}
        
        for field in required_fields:
            # Ищем паттерны типа "field_name: value" или "field_name = value"
            patterns = [
                rf'{field}\s*[:=]\s*["\']*([^"\'\n]+)["\']*',
                rf'"{field}"\s*[:=]\s*["\']*([^"\'\n]+)["\']*',
                rf"'{field}'\s*[:=]\s*['\"]*([^'\"\n]+)['\"]",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    result[field] = value
                    break
            
            if field not in result:
                result[field] = "N/A"
        
        if ResponseParser._looks_like_dummy_payload(result):
            return {"error": "LLM вернуло шаблонные значения", "raw_response": response[:500]}
        
        return result

    @staticmethod
    def _looks_like_dummy_payload(data: Dict[str, Any]) -> bool:
        """Простейшая эвристика: все поля пустые или однотипные шаблоны."""
        if not data:
            return False

        values = [str(v).strip().lower() for v in data.values() if not isinstance(v, dict)]
        if not values:
            return False

        # Проверяем, что все значения одинаковы или относятся к очевидным заглушкам
        unique_values = {v for v in values if v}
        dummy_markers = {"n/a", "12345", "dummy", "sample", "placeholder"}

        if unique_values and unique_values <= dummy_markers:
            return True

        if len(unique_values) == 1 and any(marker in next(iter(unique_values)) for marker in dummy_markers):
            return True

        return False


# Удобные функции
def parse_llm_invoice_response(
    response: str,
    required_fields: List[str],
    model_name: str = "unknown"
) -> Dict:
    """Парсит ответ LLM модели с извлечением данных счета"""
    return ResponseParser.smart_parse_llm_response(response, required_fields, model_name)


def extract_json_safely(response: str, strict: bool = False) -> Optional[Dict]:
    """Безопасно извлекает JSON из ответа"""
    return ResponseParser.extract_json_from_response(response, strict)


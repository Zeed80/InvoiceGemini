"""
Автогенератор промптов для различных моделей ИИ.
Генерирует промпты на основе полей таблицы и настроек получателя.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from .settings_manager import SettingsManager

class PromptGenerator:
    """Класс для автоматической генерации промптов для различных моделей ИИ."""
    
    def __init__(self, settings_manager: SettingsManager):
        """
        Инициализация генератора промптов.
        
        Args:
            settings_manager: Менеджер настроек
        """
        self.settings_manager = settings_manager
        self.prompts_dir = Path("data/prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Загружаем поля таблицы
        self._load_table_fields()
        
    def _load_table_fields(self) -> None:
        """Загружает поля таблицы из JSON файла."""
        try:
            with open("data/table_fields.json", "r", encoding="utf-8") as f:
                self.table_fields = json.load(f)
        except FileNotFoundError:
            # Создаем базовые поля если файл не найден
            self.table_fields = self._get_default_table_fields()
            
    def _get_default_table_fields(self) -> Dict[str, Any]:
        """Возвращает базовые поля таблицы по умолчанию."""
        return {
            "sender": {
                "display_name": "Поставщик",
                "description": "Название компании-поставщика или продавца",
                "enabled": True
            },
            "invoice_number": {
                "display_name": "№ счета", 
                "description": "Номер счета, инвойса или фактуры",
                "enabled": True
            },
            "invoice_date": {
                "display_name": "Дата счета",
                "description": "Дата выставления счета или инвойса", 
                "enabled": True
            },
            "total": {
                "display_name": "Сумма с НДС",
                "description": "Общая сумма к оплате с учетом НДС",
                "enabled": True
            },
            "amount_no_vat": {
                "display_name": "Amount (0% VAT)",
                "description": "Сумма без НДС",
                "enabled": True
            },
            "vat_percent": {
                "display_name": "% НДС",
                "description": "Ставка НДС в процентах",
                "enabled": True
            },
            "currency": {
                "display_name": "Currency",
                "description": "Валюта платежа",
                "enabled": True
            },
            "category": {
                "display_name": "Category", 
                "description": "Категория товаров или услуг",
                "enabled": True
            },
            "description": {
                "display_name": "Description",
                "description": "Описание товаров, услуг или содержимого документа",
                "enabled": True
            },
            "note": {
                "display_name": "Примечание",
                "description": "Дополнительные примечания и комментарии",
                "enabled": True
            }
        }
        
    def get_receiver_company(self) -> str:
        """Получает название компании-получателя из настроек."""
        return self.settings_manager.get_setting("receiver_name", "АО \"ПТС\"")
        
    def get_enabled_fields(self) -> List[Dict[str, str]]:
        """Возвращает список включенных полей для извлечения."""
        enabled_fields = []
        
        for field_id, field_data in self.table_fields.items():
            if field_data.get("enabled", True):
                enabled_fields.append({
                    "id": field_id,
                    "name": field_data.get("display_name", field_id),
                    "description": field_data.get("description", "")
                })
                
        return enabled_fields
        
    def generate_json_schema(self) -> str:
        """Генерирует JSON схему для ответа модели."""
        enabled_fields = self.get_enabled_fields()
        schema_fields = []
        
        for field in enabled_fields:
            schema_fields.append(f'  "{field["id"]}" : "{field["description"]}"')
            
        return "{\n" + ",\n".join(schema_fields) + "\n}"
        
    def generate_fields_list(self) -> str:
        """Генерирует список полей для промпта."""
        enabled_fields = self.get_enabled_fields()
        fields_list = []
        
        for field in enabled_fields:
            fields_list.append(f"- {field['name']}: {field['description']}")
            
        return "\n".join(fields_list)
        
    def generate_cloud_llm_prompt(self, provider_name: str) -> str:
        """
        Генерирует промпт для облачных LLM моделей.
        
        Args:
            provider_name: Название провайдера (openai, anthropic, google, etc.)
            
        Returns:
            str: Сгенерированный промпт
        """
        receiver_company = self.get_receiver_company()
        fields_list = self.generate_fields_list()
        json_schema = self.generate_json_schema()
        
        if provider_name in ["anthropic", "mistral"]:
            # Для Anthropic и Mistral - краткий и четкий промпт
            prompt = f"""Ты эксперт по анализу финансовых документов. Компания "{receiver_company}" является получателем данного счета-фактуры.

Проанализируй предоставленное изображение счета-фактуры или инвойса и извлеки из него структурированные данные.

Извлеки следующие поля из документа:
{fields_list}

Требования к ответу:
1. Верни результат ТОЛЬКО в формате JSON
2. Используй точные ID полей как ключи: {', '.join([f['id'] for f in self.get_enabled_fields()])}
3. Если поле не найдено, используй значение "N/A"
4. Все суммы указывай числами без символов валют
5. Даты в формате DD.MM.YYYY
6. Будь точным и внимательным к деталям

Проанализируй документ и верни JSON с извлеченными данными:"""

        elif provider_name == "google":
            # Для Google Gemini - структурированный промпт
            prompt = f"""Действуй как эксперт по распознаванию счетов-фактур и финансовых документов.

Компания "{receiver_company}" является получателем данного счета-фактуры.

Твоя задача: проанализировать изображение документа и извлечь из него ключевые данные в формате JSON.

Поля для извлечения:
{fields_list}

Правила:
• Возвращай ТОЛЬКО валидный JSON без дополнительного текста
• Используй точные ID полей как ключи JSON объекта
• Для отсутствующих полей используй "N/A"
• Числовые значения без символов валют
• Даты в формате DD.MM.YYYY
• Будь максимально точным

Ожидаемый формат ответа:
{json_schema}

Проанализируй документ:"""

        else:
            # Для OpenAI, DeepSeek, xAI - подробный промпт
            prompt = f"""You are an expert in invoice and financial document analysis. The company "{receiver_company}" is the recipient of this invoice.

Analyze the provided document image and extract structured data in JSON format.

Extract the following fields:
{fields_list}

Requirements:
- Return ONLY valid JSON format
- Use exact field IDs as JSON keys: {', '.join([f['id'] for f in self.get_enabled_fields()])}
- Use "N/A" for missing fields
- Numeric values without currency symbols
- Dates in DD.MM.YYYY format
- Be precise and thorough

Expected JSON format:
{json_schema}

Analyze the document and return JSON:"""

        return prompt
        
    def generate_local_llm_prompt(self, provider_name: str = "ollama") -> str:
        """
        Генерирует промпт для локальных LLM моделей.
        
        Args:
            provider_name: Название провайдера (ollama)
            
        Returns:
            str: Сгенерированный промпт
        """
        receiver_company = self.get_receiver_company()
        fields_list = self.generate_fields_list()
        json_schema = self.generate_json_schema()
        
        prompt = f"""Ты эксперт по анализу финансовых документов. Компания "{receiver_company}" является получателем данного счета-фактуры.

Проанализируй предоставленное изображение счета-фактуры или инвойса и извлеки из него структурированные данные.

Извлеки следующие поля из документа:
{fields_list}

Требования к ответу:
1. Верни результат ТОЛЬКО в формате JSON
2. Используй точные ID полей как ключи: {', '.join([f['id'] for f in self.get_enabled_fields()])}
3. Если поле не найдено, используй значение "N/A"
4. Все суммы указывай числами без символов валют
5. Даты в формате DD.MM.YYYY
6. Будь точным и внимательным к деталям

Ожидаемый формат:
{json_schema}

Проанализируй документ и верни JSON с извлеченными данными:"""

        return prompt
        
    def generate_gemini_prompt(self) -> str:
        """Генерирует промпт для Gemini модели."""
        receiver_company = self.get_receiver_company()
        fields_list = self.generate_fields_list()
        json_schema = self.generate_json_schema()
        
        prompt = f"""Ты эксперт по анализу финансовых документов. Компания "{receiver_company}" является получателем данного счета-фактуры.

Проанализируй предоставленное изображение счета-фактуры или инвойса и извлеки из него структурированные данные.

Извлеки следующие поля из документа:
{fields_list}

Требования к ответу:
1. Верни результат ТОЛЬКО в формате JSON
2. Используй точные ID полей как ключи: {', '.join([f['id'] for f in self.get_enabled_fields()])}
3. Если поле не найдено, используй значение "N/A"
4. Все суммы указывай числами без символов валют
5. Даты в формате DD.MM.YYYY
6. Будь точным и внимательным к деталям

Ожидаемый формат:
{json_schema}

Проанализируй документ и верни JSON с извлеченными данными:"""

        return prompt
        
    def save_prompt_to_file(self, prompt_type: str, content: str) -> bool:
        """
        Сохраняет промпт в файл.
        
        Args:
            prompt_type: Тип промпта (cloud_llm_openai, gemini, etc.)
            content: Содержимое промпта
            
        Returns:
            bool: True если сохранение успешно
        """
        try:
            file_path = self.prompts_dir / f"{prompt_type}_prompt.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения промпта {prompt_type}: {e}")
            return False
            
    def load_prompt_from_file(self, prompt_type: str) -> Optional[str]:
        """
        Загружает промпт из файла.
        
        Args:
            prompt_type: Тип промпта
            
        Returns:
            str: Содержимое промпта или None если файл не найден
        """
        try:
            file_path = self.prompts_dir / f"{prompt_type}_prompt.txt"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            return None
        except Exception as e:
            print(f"❌ Ошибка загрузки промпта {prompt_type}: {e}")
            return None
            
    def regenerate_all_prompts(self) -> Dict[str, bool]:
        """
        Перегенерирует все промпты на основе текущих полей таблицы.
        
        Returns:
            Dict[str, bool]: Результаты сохранения для каждого типа промпта
        """
        results = {}
        
        # Облачные LLM провайдеры
        cloud_providers = ["openai", "anthropic", "google", "mistral", "deepseek", "xai"]
        for provider in cloud_providers:
            prompt = self.generate_cloud_llm_prompt(provider)
            results[f"cloud_llm_{provider}"] = self.save_prompt_to_file(f"cloud_llm_{provider}", prompt)
            
        # Локальные LLM
        local_prompt = self.generate_local_llm_prompt("ollama")
        results["local_llm_ollama"] = self.save_prompt_to_file("local_llm_ollama", local_prompt)
        
        # Gemini
        gemini_prompt = self.generate_gemini_prompt()
        results["gemini"] = self.save_prompt_to_file("gemini", gemini_prompt)
        
        return results
        
    def get_prompt_for_model(self, model_type: str, provider_name: str = None) -> str:
        """
        Получает промпт для указанной модели.
        
        Args:
            model_type: Тип модели (cloud_llm, local_llm, gemini)
            provider_name: Название провайдера (для cloud_llm и local_llm)
            
        Returns:
            str: Промпт для модели
        """
        if model_type == "cloud_llm" and provider_name:
            # Сначала пытаемся загрузить из файла
            prompt = self.load_prompt_from_file(f"cloud_llm_{provider_name}")
            if prompt:
                return prompt
            # Если файла нет, генерируем и сохраняем
            prompt = self.generate_cloud_llm_prompt(provider_name)
            self.save_prompt_to_file(f"cloud_llm_{provider_name}", prompt)
            return prompt
            
        elif model_type == "local_llm":
            prompt = self.load_prompt_from_file("local_llm_ollama")
            if prompt:
                return prompt
            prompt = self.generate_local_llm_prompt("ollama")
            self.save_prompt_to_file("local_llm_ollama", prompt)
            return prompt
            
        elif model_type == "gemini":
            prompt = self.load_prompt_from_file("gemini")
            if prompt:
                return prompt
            prompt = self.generate_gemini_prompt()
            self.save_prompt_to_file("gemini", prompt)
            return prompt
            
        else:
            return "Промпт не найден для указанного типа модели" 
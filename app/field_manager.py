#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер полей для синхронизации таблицы результатов с промптами всех моделей.
Обеспечивает единообразие извлекаемых данных независимо от используемой модели.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from .settings_manager import settings_manager
from . import config

@dataclass
class TableField:
    """Описание поля таблицы результатов."""
    id: str                    # Внутренний ID поля
    display_name: str          # Отображаемое название в таблице
    description: str           # Описание поля для промптов
    data_type: str            # Тип данных (text, number, date, currency)
    required: bool            # Обязательное ли поле
    priority: int             # Приоритет (1-высший, 5-низший)
    position: int             # Позиция в таблице (для сортировки)
    gemini_keywords: List[str] # Ключевые слова для Gemini API
    layoutlm_labels: List[str] # Лейблы для LayoutLM
    ocr_patterns: List[str]    # Паттерны для OCR/regex поиска
    enabled: bool             # Включено ли поле

class FieldManager:
    """
    Централизованный менеджер полей для синхронизации таблицы с промптами всех моделей.
    """
    
    def __init__(self):
        """Инициализация менеджера полей."""
        self.fields_config_path = os.path.join(config.APP_DATA_PATH, 'table_fields.json')
        self._fields: Dict[str, TableField] = {}
        
        # НОВАЯ ЛОГИКА: Если есть сохраненные поля - загружаем только их
        if os.path.exists(self.fields_config_path):
            print("📂 Найден файл пользовательских настроек - загружаем только их")
            self._load_custom_fields()
        else:
            print("📂 Файл пользовательских настроек не найден - создаем поля по умолчанию")
            self._load_default_fields()
            # Сразу сохраняем дефолты как пользовательские настройки
            self.save_fields_config()
    
    def _load_default_fields(self):
        """Загружает конфигурацию полей по умолчанию."""
        default_fields = [
            TableField(
                id="sender",
                display_name="Sender",
                description="Название компании-поставщика или продавца",
                data_type="text",
                required=True,
                priority=1,
                position=1,
                gemini_keywords=["Поставщик", "поставщик", "company", "supplier", "vendor", "организация"],
                layoutlm_labels=["SELLER", "VENDOR", "COMPANY"],
                ocr_patterns=[r"ООО.*", r"ИП.*", r"АО.*", r"ПАО.*"],
                enabled=True
            ),
            TableField(
                id="invoice_number",
                display_name="№ Invoice",
                description="Номер счета, инвойса или фактуры",
                data_type="text",
                required=True,
                priority=1,
                position=2,
                gemini_keywords=["№ Счета", "№ счета", "номер счета", "invoice_number", "счет №"],
                layoutlm_labels=["INVOICE_NUMBER", "DOC_NUMBER"],
                ocr_patterns=[r"№\s*\d+", r"счет\s*№?\s*\d+", r"инвойс\s*№?\s*\d+"],
                enabled=True
            ),
            TableField(
                id="invoice_date",
                display_name="Invoice Date",
                description="Дата выставления счета или инвойса",
                data_type="date",
                required=True,
                priority=1,
                position=3,
                gemini_keywords=["Дата счета", "дата счета", "invoice_date", "date", "дата"],
                layoutlm_labels=["DATE", "INVOICE_DATE"],
                ocr_patterns=[r"\d{1,2}\.\d{1,2}\.\d{4}", r"\d{1,2}\s+\w+\s+\d{4}"],
                enabled=True
            ),
            TableField(
                id="total",
                display_name="Total",
                description="Общая сумма к оплате с учетом НДС",
                data_type="currency",
                required=True,
                priority=1,
                position=4,
                gemini_keywords=["Сумма с НДС", "сумма с ндс", "total", "итого", "к оплате"],
                layoutlm_labels=["TOTAL", "AMOUNT", "TOTAL_AMOUNT"],
                ocr_patterns=[r"\d+[,\.\s]\d+\s*руб", r"итого.*\d+"],
                enabled=True
            ),
            TableField(
                id="amount_no_vat",
                display_name="Amount (0% VAT)",
                description="Сумма без НДС",
                data_type="currency",
                required=False,
                priority=2,
                position=5,
                gemini_keywords=["Сумма без НДС", "сумма без ндс", "amount_no_vat", "net_amount"],
                layoutlm_labels=["NET_AMOUNT", "AMOUNT_NO_VAT"],
                ocr_patterns=[r"без\s+НДС.*\d+", r"net.*amount.*\d+"],
                enabled=True
            ),
            TableField(
                id="vat_percent",
                display_name="VAT %",
                description="Ставка НДС в процентах",
                data_type="number",
                required=False,
                priority=2,
                position=6,
                gemini_keywords=["НДС %", "ндс %", "vat_rate", "tax_rate", "ставка ндс"],
                layoutlm_labels=["VAT_RATE", "TAX_RATE"],
                ocr_patterns=[r"НДС\s*\d+%", r"\d+%\s*НДС"],
                enabled=True
            ),
            TableField(
                id="currency",
                display_name="Currency",
                description="Валюта платежа",
                data_type="text",
                required=False,
                priority=3,
                position=7,
                gemini_keywords=["Валюта", "валюта", "currency"],
                layoutlm_labels=["CURRENCY"],
                ocr_patterns=[r"RUB|руб|USD|EUR|₽"],
                enabled=True
            ),
            TableField(
                id="category",
                display_name="Category",
                description="Категория товаров или услуг",
                data_type="text",
                required=False,
                priority=3,
                position=8,
                gemini_keywords=["Категория", "категория", "category"],
                layoutlm_labels=["CATEGORY", "ITEM_TYPE"],
                ocr_patterns=[r"категория.*", r"тип.*товар"],
                enabled=True
            ),
            TableField(
                id="description",
                display_name="Description",
                description="Описание товаров, услуг или содержимого документа",
                data_type="text",
                required=False,
                priority=3,
                position=9,
                gemini_keywords=["Товары", "товары", "description", "items", "услуги"],
                layoutlm_labels=["DESCRIPTION", "ITEMS"],
                ocr_patterns=[r"наименование.*", r"товар.*", r"услуг.*"],
                enabled=True
            ),
            TableField(
                id="inn",
                display_name="INN",
                description="ИНН поставщика (налоговый номер)",
                data_type="text",
                required=False,
                priority=4,
                position=10,
                gemini_keywords=["ИНН", "инн", "inn", "tax_id", "налоговый номер"],
                layoutlm_labels=["INN", "TAX_ID"],
                ocr_patterns=[r"ИНН\s*\d{10,12}", r"\d{10,12}"],
                enabled=True
            ),
            TableField(
                id="kpp",
                display_name="KPP",
                description="КПП поставщика (код причины постановки)",
                data_type="text",
                required=False,
                priority=4,
                position=11,
                gemini_keywords=["КПП", "кпп", "kpp"],
                layoutlm_labels=["KPP"],
                ocr_patterns=[r"КПП\s*\d{9}", r"\d{9}"],
                enabled=True
            ),
            TableField(
                id="note",
                display_name="Note",
                description="Дополнительные примечания и комментарии",
                data_type="text",
                required=False,
                priority=5,
                position=12,
                gemini_keywords=["Комментарии", "комментарии", "note", "comment", "примечание"],
                layoutlm_labels=["NOTE", "COMMENT"],
                ocr_patterns=[r"примечание.*", r"комментарий.*"],
                enabled=True
            )
        ]
        
        for field in default_fields:
            self._fields[field.id] = field
    
    def _load_custom_fields(self):
        """Загружает пользовательские настройки полей из файла."""
        try:
            print(f"📄 Загрузка пользовательских полей из: {self.fields_config_path}")
            
            with open(self.fields_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            print(f"📄 Найдено {len(data)} полей в файле конфигурации")
            
            # Очищаем текущие поля
            self._fields.clear()
                
            for field_id, field_data in data.items():
                print(f"  📥 Загрузка поля '{field_id}': {field_data.get('display_name', 'НЕТ НАЗВАНИЯ')}")
                
                # Убеждаемся, что все обязательные поля присутствуют
                if 'position' not in field_data:
                    field_data['position'] = field_data.get('priority', 1)
                
                # Создаем объект поля из сохраненных данных
                self._fields[field_id] = TableField(**field_data)
            
            print(f"✅ Успешно загружены пользовательские поля")
                    
        except Exception as e:
            print(f"❌ Ошибка загрузки пользовательских полей: {e}")
            import traceback
            traceback.print_exc()
            # Если ошибка загрузки - создаем поля по умолчанию
            print("🔄 Создание полей по умолчанию из-за ошибки")
            self._load_default_fields()
        
        # Отладочная информация
        print(f"📊 Итого загружено полей: {len(self._fields)}")
        print(f"📊 Включенных полей: {len([f for f in self._fields.values() if f.enabled])}")
        
        # Проверяем несколько ключевых полей
        test_fields = ['invoice_number', 'total', 'sender']
        for test_field in test_fields:
            if test_field in self._fields:
                field = self._fields[test_field]
                print(f"📊 Поле '{test_field}': '{field.display_name}' (позиция: {field.position}, включено: {field.enabled})")
            else:
                print(f"⚠️ Поле '{test_field}' НЕ НАЙДЕНО")
    
    def save_fields_config(self):
        """Сохраняет текущую конфигурацию полей в файл."""
        try:
            print(f"💾 СОХРАНЕНИЕ КОНФИГУРАЦИИ ПОЛЕЙ...")
            
            data = {}
            for field_id, field in self._fields.items():
                data[field_id] = asdict(field)
                print(f"   💾 {field_id}: '{field.display_name}' (включено: {field.enabled})")
            
            os.makedirs(os.path.dirname(self.fields_config_path), exist_ok=True)
            with open(self.fields_config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ КОНФИГУРАЦИЯ СОХРАНЕНА: {self.fields_config_path}")
            print(f"   Всего полей: {len(data)}")
                
        except Exception as e:
            print(f"❌ ОШИБКА СОХРАНЕНИЯ КОНФИГУРАЦИИ ПОЛЕЙ: {e}")
            import traceback
            traceback.print_exc()
    
    def get_enabled_fields(self) -> List[TableField]:
        """Возвращает список включенных полей, отсортированных по позиции."""
        enabled = [f for f in self._fields.values() if f.enabled]
        return sorted(enabled, key=lambda x: x.position)
    
    def get_table_columns(self) -> List[Dict]:
        """Возвращает конфигурацию колонок для таблицы результатов."""
        columns = []
        for field in self.get_enabled_fields():
            columns.append({
                "id": field.id,
                "name": field.display_name,
                "type": "text"  # PyQt6 DataTable использует text для всех типов
            })
        return columns
    
    def get_gemini_prompt(self, base_prompt: str = "") -> str:
        """Генерирует промпт для Gemini API на основе активных полей."""
        enabled_fields = self.get_enabled_fields()
        
        prompt_parts = [
            "Проанализируй изображение документа и извлеки следующие поля в JSON формате:",
            ""
        ]
        
        # Добавляем описание каждого поля
        for field in enabled_fields:
            keywords_str = ", ".join(field.gemini_keywords[:3])  # Первые 3 ключевых слова
            prompt_parts.append(f"- {field.gemini_keywords[0]}: {field.description} (варианты: {keywords_str})")
        
        prompt_parts.extend([
            "",
            "ВАЖНО: Возвращай ответ ТОЛЬКО в формате JSON. Используй точные названия полей как показано выше.",
            "Если поле не найдено в документе, используй \"N/A\".",
            "Не добавляй никаких дополнительных объяснений или текста вне JSON."
        ])
        
        if base_prompt:
            return f"{base_prompt}\n\n{chr(10).join(prompt_parts)}"
        else:
            return chr(10).join(prompt_parts)
    
    def get_layoutlm_labels(self) -> List[str]:
        """Возвращает список лейблов для обучения LayoutLM."""
        labels = []
        for field in self.get_enabled_fields():
            labels.extend(field.layoutlm_labels)
        return list(set(labels))  # Убираем дубликаты
    
    def get_ocr_patterns(self) -> List[Tuple[str, str]]:
        """Возвращает список паттернов для OCR поиска (паттерн, field_id)."""
        patterns = []
        for field in self.get_enabled_fields():
            for pattern in field.ocr_patterns:
                patterns.append((pattern, field.id))
        return patterns
    
    def get_field_mapping_for_model(self, model_type: str) -> Dict[str, str]:
        """Возвращает маппинг полей для конкретной модели."""
        mapping = {}
        
        for field in self.get_enabled_fields():
            if model_type == 'gemini':
                for keyword in field.gemini_keywords:
                    mapping[keyword.lower()] = field.display_name
            elif model_type == 'layoutlm':
                for label in field.layoutlm_labels:
                    mapping[label.lower()] = field.display_name
            # Добавить другие модели при необходимости
        
        return mapping
    
    def get_field_id_mapping_for_model(self, model_type: str) -> Dict[str, str]:
        """Возвращает маппинг полей API -> field_id для использования в _map_gemini_fields."""
        mapping = {}
        
        for field in self.get_enabled_fields():
            if model_type == 'gemini':
                for keyword in field.gemini_keywords:
                    mapping[keyword.lower()] = field.id
            elif model_type == 'layoutlm':
                for label in field.layoutlm_labels:
                    mapping[label.lower()] = field.id
            # Добавить другие модели при необходимости
        
        return mapping
    
    def update_field(self, field_id: str, **kwargs):
        """Обновляет параметры поля."""
        if field_id in self._fields:
            for key, value in kwargs.items():
                if hasattr(self._fields[field_id], key):
                    setattr(self._fields[field_id], key, value)
            self.save_fields_config()
    
    def add_field(self, field: TableField):
        """Добавляет новое поле."""
        self._fields[field.id] = field
        self.save_fields_config()
    
    def remove_field(self, field_id: str):
        """Удаляет поле."""
        if field_id in self._fields:
            del self._fields[field_id]
            self.save_fields_config()
    
    def get_field(self, field_id: str) -> Optional[TableField]:
        """Возвращает поле по ID."""
        return self._fields.get(field_id)
    
    def get_all_fields(self) -> Dict[str, TableField]:
        """Возвращает все поля."""
        return self._fields.copy()
    
    def move_field_up(self, field_id: str) -> bool:
        """Перемещает поле вверх в списке (уменьшает position)."""
        if field_id not in self._fields:
            return False
        
        current_field = self._fields[field_id]
        target_position = current_field.position - 1
        
        if target_position < 1:
            return False
        
        # Находим поле с target_position и меняем позиции местами
        for other_field in self._fields.values():
            if other_field.position == target_position:
                other_field.position = current_field.position
                break
        
        current_field.position = target_position
        self.save_fields_config()
        return True
    
    def move_field_down(self, field_id: str) -> bool:
        """Перемещает поле вниз в списке (увеличивает position)."""
        if field_id not in self._fields:
            return False
        
        current_field = self._fields[field_id]
        max_position = max(field.position for field in self._fields.values())
        target_position = current_field.position + 1
        
        if target_position > max_position:
            return False
        
        # Находим поле с target_position и меняем позиции местами
        for other_field in self._fields.values():
            if other_field.position == target_position:
                other_field.position = current_field.position
                break
        
        current_field.position = target_position
        self.save_fields_config()
        return True
    
    def sync_prompts_for_all_models(self):
        """Синхронизирует промпты для всех моделей с текущими полями."""
        try:
            # Обновляем промпт для Gemini
            gemini_prompt = self.get_gemini_prompt()
            settings_manager.set_value('Prompts', 'gemini_extract_prompt', gemini_prompt)
            
            # Обновляем промпты для LLM плагинов
            for plugin_name in ['llama', 'mistral', 'codellama']:
                plugin_prompt = self._generate_llm_plugin_prompt()
                settings_manager.set_value('Prompts', f'{plugin_name}_prompt', plugin_prompt)
            
            # Сохраняем настройки
            settings_manager.save_settings()
            
            print("✅ Промпты всех моделей синхронизированы с полями таблицы")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка синхронизации промптов: {e}")
            return False
    
    def _generate_llm_plugin_prompt(self) -> str:
        """Генерирует промпт для LLM плагинов."""
        enabled_fields = self.get_enabled_fields()
        
        prompt_parts = [
            "Analyze the document text and extract the following fields:",
            ""
        ]
        
        for field in enabled_fields:
            prompt_parts.append(f"- {field.display_name}: {field.description}")
        
        prompt_parts.extend([
            "",
            "Return the results as a JSON object with the exact field names shown above.",
            "If a field is not found, use 'N/A' as the value."
        ])
        
        return chr(10).join(prompt_parts)

# Глобальный экземпляр менеджера полей
field_manager = FieldManager() 
"""
Адаптивная система управления промптами для разных моделей Ollama.
Автоматически подбирает оптимальный формат промпта в зависимости от модели.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Профиль модели с оптимальными настройками"""
    name_pattern: str  # Паттерн для определения модели
    complexity_level: str  # simple, medium, advanced
    supports_json: bool  # Поддержка структурированного JSON
    supports_russian: bool  # Хорошая поддержка русского языка
    max_context: int  # Максимальный контекст
    temperature: float  # Оптимальная температура
    requires_xml: bool  # Требуется XML формат
    requires_markdown: bool  # Требуется Markdown формат
    instruction_style: str  # direct, detailed, conversational


class AdaptivePromptManager:
    """Менеджер адаптивных промптов для разных моделей"""
    
    # Профили известных моделей
    MODEL_PROFILES = {
        # Gemma модели - требуют простых инструкций
        "gemma3": ModelProfile(
            name_pattern="gemma3",
            complexity_level="medium",  # Изменено на medium для более детального промпта
            supports_json=True,
            supports_russian=True,
            max_context=8192,
            temperature=0.05,  # Снижена температура для точности
            requires_xml=False,
            requires_markdown=True,
            instruction_style="direct"
        ),
        
        # Llama Vision модели - продвинутые
        "llama3.2-vision": ModelProfile(
            name_pattern="llama3.2-vision",
            complexity_level="advanced",
            supports_json=True,
            supports_russian=True,
            max_context=128000,
            temperature=0.1,
            requires_xml=False,
            requires_markdown=False,
            instruction_style="detailed"
        ),
        
        # Llama текстовые модели
        "llama3": ModelProfile(
            name_pattern="llama3",
            complexity_level="medium",
            supports_json=True,
            supports_russian=True,
            max_context=8192,
            temperature=0.2,
            requires_xml=False,
            requires_markdown=False,
            instruction_style="detailed"
        ),
        
        # Qwen Vision-Language модели
        "qwen2.5vl": ModelProfile(
            name_pattern="qwen2.5vl",
            complexity_level="advanced",
            supports_json=True,
            supports_russian=True,
            max_context=32768,
            temperature=0.1,
            requires_xml=False,
            requires_markdown=True,
            instruction_style="detailed"
        ),
        
        # Qwen текстовые модели
        "qwen": ModelProfile(
            name_pattern="qwen",
            complexity_level="medium",
            supports_json=True,
            supports_russian=True,
            max_context=32768,
            temperature=0.2,
            requires_xml=False,
            requires_markdown=False,
            instruction_style="conversational"
        ),
        
        # Mistral модели
        "mistral": ModelProfile(
            name_pattern="mistral",
            complexity_level="advanced",
            supports_json=True,
            supports_russian=True,
            max_context=32768,
            temperature=0.1,
            requires_xml=False,
            requires_markdown=False,
            instruction_style="detailed"
        ),
        
        # LLaVA модели
        "llava": ModelProfile(
            name_pattern="llava",
            complexity_level="medium",
            supports_json=True,
            supports_russian=False,
            max_context=4096,
            temperature=0.2,
            requires_xml=False,
            requires_markdown=True,
            instruction_style="direct"
        ),
    }
    
    @staticmethod
    def get_model_profile(model_name: str) -> ModelProfile:
        """
        Определяет профиль модели по её названию.
        
        Args:
            model_name: Название модели
            
        Returns:
            ModelProfile: Профиль модели
        """
        model_lower = model_name.lower()
        
        # Ищем подходящий профиль
        for pattern, profile in AdaptivePromptManager.MODEL_PROFILES.items():
            if pattern in model_lower:
                return profile
        
        # Дефолтный профиль для неизвестных моделей
        return ModelProfile(
            name_pattern="default",
            complexity_level="medium",
            supports_json=True,
            supports_russian=True,
            max_context=4096,
            temperature=0.2,
            requires_xml=False,
            requires_markdown=False,
            instruction_style="detailed"
        )
    
    @staticmethod
    def create_adaptive_prompt(
        model_name: str,
        fields: Dict[str, str],
        image_available: bool = False,
        ocr_text: Optional[str] = None
    ) -> str:
        """
        Создает адаптивный промпт для конкретной модели.
        
        Args:
            model_name: Название модели
            fields: Словарь полей для извлечения {field_id: field_label}
            image_available: Доступно ли изображение
            ocr_text: Текст из OCR (если есть)
            
        Returns:
            str: Оптимизированный промпт для модели
        """
        profile = AdaptivePromptManager.get_model_profile(model_name)
        
        if profile.complexity_level == "simple":
            return AdaptivePromptManager._create_simple_prompt(fields, profile, ocr_text)
        elif profile.complexity_level == "medium":
            return AdaptivePromptManager._create_medium_prompt(fields, profile, image_available, ocr_text)
        else:  # advanced
            return AdaptivePromptManager._create_advanced_prompt(fields, profile, image_available, ocr_text)
    
    @staticmethod
    def _create_simple_prompt(
        fields: Dict[str, str],
        profile: ModelProfile,
        ocr_text: Optional[str]
    ) -> str:
        """Создает простой промпт для моделей типа Gemma3"""
        
        # КРИТИЧЕСКИ ВАЖНО: Явно указываем, что нужны РЕАЛЬНЫЕ данные из изображения
        prompt = "You are analyzing an invoice image. Extract the ACTUAL data you see in the image.\n\n"
        prompt += "IMPORTANT RULES:\n"
        prompt += "1. Extract ONLY the real text and numbers visible in the invoice image\n"
        prompt += "2. Do NOT invent or generate example values\n"
        prompt += "3. If a field is not visible in the image, use \"N/A\"\n"
        prompt += "4. Extract exact values as they appear in the document\n\n"
        
        if ocr_text:
            prompt += f"Document text (for reference):\n{ocr_text[:1000]}\n\n"
        
        prompt += "Required fields to extract:\n"
        for field_id, field_label in fields.items():
            prompt += f"- {field_id}: {field_label}\n"
        
        prompt += "\nOutput format - JSON with exact values from the image:\n"
        prompt += "{\n"
        for i, field_id in enumerate(fields.keys()):
            comma = "," if i < len(fields) - 1 else ""
            prompt += f'  "{field_id}": "<actual value from image or N/A>"{comma}\n'
        prompt += "}\n\n"
        
        prompt += "CRITICAL: Return ONLY the JSON object with REAL data extracted from the invoice image.\n"
        
        if profile.requires_markdown:
            prompt += "Wrap JSON in ```json``` code block.\n"
        
        return prompt
    
    @staticmethod
    def _create_medium_prompt(
        fields: Dict[str, str],
        profile: ModelProfile,
        image_available: bool,
        ocr_text: Optional[str]
    ) -> str:
        """Создает промпт средней сложности"""
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: OCR текст ПЕРВЫМ (как в чате Ollama)
        prompt = ""
        
        if ocr_text:
            # OCR в самом начале как основной контекст
            prompt += f"Invoice document text:\n---\n{ocr_text[:2500]}\n---\n\n"
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Добавляем ОПИСАНИЯ полей на русском для точного извлечения
        field_descriptions = {
            "sender": "company/seller name (Название компании-поставщика)",
            "invoice_number": "SHORT invoice/document number (1-6 digits or alphanumeric like '1740', '№345', 'INV-123'). NEVER use 20-digit bank account numbers! (Короткий номер счета/документа, обычно 1-6 символов)",
            "invoice_date": "invoice issue date (Дата выставления счета)",
            "description": "brief description of goods/services (Краткое описание товаров или услуг)",
            "total": "total amount with VAT (Общая сумма к оплате с НДС)",
            "amount_no_vat": "amount without VAT (Сумма без НДС)",
            "vat_percent": "VAT rate percentage only, like 20 or 0 (Только процент НДС, например 20)",
            "currency": "currency code or symbol (Валюта, например RUB или руб.)",
            "category": "GENERAL category like 'tools', 'office supplies', 'services', 'electronics', NOT specific product name (Общая КАТЕГОРИЯ товаров, НЕ название конкретного товара)",
            "note": "additional notes or payment terms (Дополнительные примечания)"
        }
        
        # Список полей С ОПИСАНИЯМИ для точного понимания
        prompt += "Extract these fields (with descriptions):\n"
        for field_id in fields.keys():
            desc = field_descriptions.get(field_id, "")
            if desc:
                prompt += f"- {field_id}: {desc}\n"
            else:
                prompt += f"- {field_id}\n"
        
        prompt += "\nIMPORTANT EXAMPLES:\n"
        prompt += "- invoice_number: If you see 'Счет № 1740' → use '1740', NOT '30101810645250000092'\n"
        prompt += "- category: If selling 'пластина' → use 'tools' or 'materials', NOT 'пластина'\n\n"
        prompt += "Return as valid JSON with these exact field names.\n"
        prompt += "Use \"N/A\" for missing fields.\n"
        prompt += "Dates in DD.MM.YYYY format.\n"
        prompt += "Numbers without currency symbols.\n"
        
        if profile.requires_markdown:
            prompt += "Wrap result in ```json``` code block.\n"
        
        return prompt
    
    @staticmethod
    def _create_advanced_prompt(
        fields: Dict[str, str],
        profile: ModelProfile,
        image_available: bool,
        ocr_text: Optional[str]
    ) -> str:
        """Создает детальный промпт для продвинутых моделей"""
        
        prompt = "You are an expert in invoice and financial document analysis.\n\n"
        
        prompt += "Task: Extract structured data from the provided invoice.\n\n"
        
        if image_available:
            prompt += "Input: Invoice image with visual and textual information.\n\n"
        elif ocr_text:
            prompt += f"Input: Document text extracted via OCR:\n---\n{ocr_text}\n---\n\n"
        
        prompt += "Fields to extract:\n"
        for field_id, field_label in fields.items():
            prompt += f"- {field_id}: {field_label}\n"
        
        prompt += "\nExtraction Guidelines:\n"
        prompt += "1. Extract REAL values visible in the invoice image\n"
        prompt += "2. DO NOT use field labels as values\n"
        prompt += "3. For missing fields, use 'N/A'\n"
        prompt += "4. Dates must be in DD.MM.YYYY format\n"
        prompt += "5. Numeric values without currency symbols\n"
        prompt += "6. Be precise and extract exact text/numbers\n\n"
        
        prompt += "JSON example format (replace with real data):\n"
        prompt += "{\n"
        example_values = {
            "sender": "ABC Company",
            "invoice_number": "123-456",
            "invoice_date": "01.01.2024",
            "description": "services rendered",
            "total": "5000",
            "amount_no_vat": "4200",
            "vat_percent": "20",
            "currency": "RUB",
            "category": "services",
            "note": "payment due"
        }
        
        for i, field_id in enumerate(fields.keys()):
            comma = "," if i < len(fields) - 1 else ""
            example_val = example_values.get(field_id, "real_value_here")
            prompt += f'  "{field_id}": "{example_val}"{comma}\n'
        prompt += "}\n\n"
        
        prompt += "NOW extract ACTUAL data from the invoice image and return JSON:"
        
        return prompt
    
    @staticmethod
    def get_generation_params(model_name: str) -> Dict[str, any]:
        """
        Получает оптимальные параметры генерации для модели.
        
        Args:
            model_name: Название модели
            
        Returns:
            Dict: Параметры генерации
        """
        profile = AdaptivePromptManager.get_model_profile(model_name)
        
        params = {
            "temperature": profile.temperature,
            "num_predict": 2048,  # Увеличено для полного JSON
            "top_p": 0.85,
            "top_k": 30,
            "repeat_penalty": 1.15,  # Против повторяющихся шаблонов
            "stop": ["```\n\n", "</json>"]
        }
        
        # Для простых и средних моделей - МИНИМАЛЬНЫЕ параметры (как в чате Ollama)
        if profile.complexity_level == "simple":
            params.update({
                "num_predict": 2048,
                "temperature": 0.1,  # Более естественная температура
                "repeat_penalty": 1.1,  # Умеренная защита от повторов
                "top_p": 0.9,
                "top_k": 40
            })
        elif profile.complexity_level == "medium":
            params.update({
                "num_predict": 2048,
                "temperature": 0.1,  # Такая же как в чате
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40
            })
        
        return params


# Удобные функции
def create_adaptive_invoice_prompt(
    model_name: str,
    fields: Dict[str, str],
    image_available: bool = False,
    ocr_text: Optional[str] = None
) -> str:
    """Создает адаптивный промпт для извлечения данных из счета"""
    return AdaptivePromptManager.create_adaptive_prompt(
        model_name, fields, image_available, ocr_text
    )


def get_model_generation_params(model_name: str) -> Dict[str, any]:
    """Получает оптимальные параметры генерации для модели"""
    return AdaptivePromptManager.get_generation_params(model_name)


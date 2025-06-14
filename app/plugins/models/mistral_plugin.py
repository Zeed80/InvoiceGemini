"""
MistralPlugin для InvoiceGemini
Поддерживает Mistral 7B и 8x7B модели для обработки счетов
"""

import os
import json
from typing import Dict, Any, Optional, List
from ..base_llm_plugin import BaseLLMPlugin

class MistralPlugin(BaseLLMPlugin):
    """
    Плагин для Mistral моделей
    Поддерживает Mistral 7B Instruct v0.3 и Mixtral 8x7B
    """
    
    def __init__(self, model_name: str = "mistral-7b-instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Конфигурация Mistral моделей
        self.supported_models = {
            "mistral-7b-instruct": {
                "path": "mistralai/Mistral-7B-Instruct-v0.3",
                "memory_gb": 16,
                "context_length": 32768,
                "description": "Mistral 7B Instruct v0.3 - эффективная модель для инструкций"
            },
            "mistral-8x7b-instruct": {
                "path": "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                "memory_gb": 96,
                "context_length": 32768,
                "description": "Mixtral 8x7B - мощная модель смеси экспертов"
            },
            "mistral-7b-base": {
                "path": "mistralai/Mistral-7B-v0.3",
                "memory_gb": 14,
                "context_length": 32768,
                "description": "Mistral 7B Base - базовая модель для fine-tuning"
            }
        }
        
        # Текущая модель
        self.current_model_key = "mistral-7b-instruct"
        
        # Конфигурация генерации
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "eos_token_id": None,
            "pad_token_id": None
        }
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Возвращает информацию о плагине"""
        return {
            "name": "Mistral Plugin",
            "version": "1.0.0",
            "description": "Поддержка Mistral и Mixtral моделей для обработки счетов",
            "author": "InvoiceGemini Team",
            "supported_models": list(self.supported_models.keys()),
            "current_model": self.current_model_key,
            "is_loaded": self.is_loaded,
            "capabilities": [
                "Многоязычная обработка документов",
                "Высокая точность извлечения данных", 
                "Поддержка длинных документов (32K токенов)",
                "Быстрые инференс благодаря оптимизации",
                "Понимание структуры счетов"
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        model_config = self.supported_models[self.current_model_key]
        
        return {
            "name": f"Mistral {self.current_model_key}",
            "model_family": "Mistral",
            "model_path": model_config["path"],
            "device": str(self.device) if hasattr(self, 'device') else "Unknown",
            "is_loaded": self.is_loaded,
            "memory_requirements": f"~{model_config['memory_gb']} GB GPU/RAM",
            "context_length": model_config["context_length"],
            "torch_available": self._check_torch_available(),
            "description": model_config["description"]
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию для обучения"""
        model_config = self.supported_models[self.current_model_key]
        
        return {
            "supports_lora": True,
            "supports_qlora": True,
            "default_lora_rank": 16,
            "max_sequence_length": model_config["context_length"],
            "training_args": {
                "batch_size": 2 if "8x7b" in self.current_model_key else 4,
                "learning_rate": "2e-4",
                "num_epochs": 3,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "gradient_accumulation_steps": 8 if "8x7b" in self.current_model_key else 4
            },
            "recommended_memory": model_config["memory_gb"],
            "chat_template": self._get_mistral_chat_template()
        }
    
    def load_model(self, model_key: Optional[str] = None) -> bool:
        """Загружает выбранную Mistral модель"""
        if not self._check_dependencies():
            return False
        
        if model_key and model_key in self.supported_models:
            self.current_model_key = model_key
        
        model_config = self.supported_models[self.current_model_key]
        model_path = model_config["path"]
        
        try:
            print(f"🚀 Загрузка Mistral модели: {model_path}")
            
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            # Определяем устройство
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Используется устройство: {self.device}")
            
            # Настройки квантизации для экономии памяти
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("[WRENCH] Используется 4-bit квантизация для экономии памяти")
            
            # Загружаем токенизатор
            print("📝 Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Устанавливаем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("[WRENCH] Установлен pad_token = eos_token")
            
            # Обновляем конфигурацию генерации
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Загружаем модель
            print("🧠 Загрузка модели...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Если не используем device_map, перемещаем модель на устройство
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"✅ Mistral модель {self.current_model_key} успешно загружена!")
            return True
            
        except ImportError as e:
            print(f"❌ Ошибка импорта библиотек: {e}")
            print("💡 Установите: pip install torch transformers accelerate bitsandbytes")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки Mistral модели: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self) -> bool:
        """Выгружает модель из памяти"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Очистка кэша CUDA
            if hasattr(self, 'device') and 'cuda' in str(self.device):
                import torch
                torch.cuda.empty_cache()
                print("🧹 CUDA кэш очищен")
            
            self.is_loaded = False
            print("✅ Mistral модель выгружена")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка выгрузки Mistral модели: {e}")
            return False
    
    def process_image(self, image_path: str, ocr_lang: str = "rus+eng", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Основной метод обработки изображения."""
        try:
            # Извлекаем текст с помощью OCR
            ocr_text = self.extract_text_from_image(image_path, ocr_lang)
            
            # Обрабатываем счет
            return self.process_invoice(ocr_text, image_path)
        except Exception as e:
            print(f"❌ Ошибка обработки изображения: {e}")
            return {"error": f"Ошибка обработки изображения: {e}"}

    def generate_response(self, prompt: str, image_context: str = "") -> str:
        """Генерирует ответ с помощью Mistral модели."""
        if not self.is_loaded:
            return "Модель не загружена"
        
        try:
            # Объединяем промпт и контекст
            full_prompt = f"{prompt}\n\nТекст документа:\n{image_context}" if image_context else prompt
            
            # Форматируем в Mistral chat формат
            chat_messages = [{"role": "user", "content": full_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Токенизация
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_config, pad_token_id=self.tokenizer.eos_token_id)
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            return f"Ошибка генерации: {e}"

    def process_invoice(self, ocr_text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Обрабатывает счет с помощью Mistral модели"""
        if not self.is_loaded:
            print("❌ Mistral модель не загружена")
            return {"error": "Модель не загружена"}
        
        try:
            # Создаем промпт для Mistral
            system_prompt = self._get_system_prompt()
            user_prompt = self._create_user_prompt(ocr_text)
            
            # Форматируем в Mistral chat формат
            chat_messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
            
            # Применяем chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"🎯 Обработка документа с Mistral ({self.current_model_key})")
            print(f"📝 Длина OCR текста: {len(ocr_text)} символов")
            
            # Токенизация
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(4096, self.supported_models[self.current_model_key]["context_length"])
            ).to(self.device)
            
            # Генерация ответа
            print("🤖 Генерация ответа...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование ответа
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            print(f"📤 Получен ответ длиной: {len(response)} символов")
            
            # Парсинг JSON из ответа
            parsed_result = self._parse_mistral_response(response)
            
            # Добавляем метаинформацию
            parsed_result['note_gemini'] = f"Извлечено с помощью Mistral ({self.current_model_key})"
            parsed_result['raw_response_mistral'] = response
            parsed_result['model_used'] = self.current_model_key
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Ошибка обработки с Mistral: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "note_gemini": f"Ошибка Mistral ({self.current_model_key}): {str(e)}"
            }
    
    def _get_system_prompt(self) -> str:
        """Возвращает системный промпт для Mistral"""
        return """Ты - эксперт по анализу деловых документов и счетов. 
Твоя задача - извлечь структурированные данные из предоставленного текста счета.

ИНСТРУКЦИИ:
1. Анализируй весь текст внимательно
2. Извлеки все доступные поля
3. Верни результат ТОЛЬКО в формате JSON
4. Если поле не найдено, используй "N/A"
5. Суммы указывай числами без символов валют
6. Даты в формате DD.MM.YYYY

ФОРМАТ ОТВЕТА (JSON):
{
    "company": "название компании",
    "invoice_number": "номер счета", 
    "date": "дата счета",
    "total_amount": "общая сумма",
    "currency": "валюта",
    "items": [
        {
            "name": "название товара/услуги",
            "quantity": "количество",
            "price": "цена за единицу",
            "amount": "сумма по позиции"
        }
    ],
    "amount_without_vat_gemini": "сумма без НДС",
    "vat_percent_gemini": "процент НДС",
    "category_gemini": "категория расходов",
    "description_gemini": "описание"
}"""
    
    def _create_user_prompt(self, ocr_text: str) -> str:
        """Создает пользовательский промпт"""
        return f"""Проанализируй следующий текст счета и извлеки данные:

ТЕКСТ СЧЕТА:
{ocr_text}

Верни результат в формате JSON как указано в инструкции."""
    
    def _get_mistral_chat_template(self) -> str:
        """Возвращает шаблон чата для Mistral"""
        return """<s>[INST] {{ system_prompt }}

{{ user_message }} [/INST]"""
    
    def _parse_mistral_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ Mistral и извлекает JSON"""
        try:
            import re
            import json
            
            # Очищаем ответ от лишнего текста
            response = response.strip()
            
            # Ищем JSON в ответе (между { и })
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                # Если JSON не найден, возвращаем ошибку
                print(f"⚠️ JSON не найден в ответе Mistral: {response[:200]}...")
                return {
                    "error": "JSON не найден в ответе",
                    "raw_response": response
                }
                
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON от Mistral: {e}")
            return {
                "error": f"Ошибка парсинга JSON: {e}",
                "raw_response": response
            }
        except Exception as e:
            print(f"❌ Неожиданная ошибка парсинга ответа Mistral: {e}")
            return {
                "error": f"Ошибка парсинга: {e}",
                "raw_response": response
            }
    
    def switch_model(self, model_key: str) -> bool:
        """Переключает на другую модель Mistral"""
        if model_key not in self.supported_models:
            print(f"❌ Неподдерживаемая модель: {model_key}")
            return False
        
        if self.is_loaded:
            print("🔄 Выгрузка текущей модели...")
            self.unload_model()
        
        print(f"🔄 Переключение на модель: {model_key}")
        return self.load_model(model_key)
    
    def get_supported_models(self) -> List[str]:
        """Возвращает список поддерживаемых моделей"""
        return list(self.supported_models.keys())
    
    def _check_dependencies(self) -> bool:
        """Проверяет наличие необходимых зависимостей"""
        try:
            import torch
            import transformers
            return True
        except ImportError as e:
            print(f"❌ Отсутствуют зависимости: {e}")
            print("💡 Установите: pip install torch transformers accelerate")
            return False
    
    def _check_torch_available(self) -> bool:
        """Проверяет доступность PyTorch"""
        try:
            import torch
            return True
        except ImportError:
            return False 
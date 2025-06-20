"""
CodeLlamaPlugin для InvoiceGemini
Специализированный плагин для обработки технических документов и счетов с кодами товаров
"""

import os
import json
from typing import Dict, Any, Optional, List
from ..base_llm_plugin import BaseLLMPlugin

class CodeLlamaPlugin(BaseLLMPlugin):
    """
    Плагин для Code Llama моделей
    Специализирован на обработке технических документов и структурированных данных
    """
    
    def __init__(self, model_name: str = "codellama-7b-instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Конфигурация Code Llama моделей
        self.supported_models = {
            "codellama-7b-instruct": {
                "path": "codellama/CodeLlama-7b-Instruct-hf",
                "memory_gb": 16,
                "context_length": 16384,
                "description": "Code Llama 7B Instruct - специализирован на структурированных данных"
            },
            "codellama-13b-instruct": {
                "path": "codellama/CodeLlama-13b-Instruct-hf",
                "memory_gb": 28,
                "context_length": 16384,
                "description": "Code Llama 13B Instruct - более мощная версия для сложных задач"
            },
            "codellama-34b-instruct": {
                "path": "codellama/CodeLlama-34b-Instruct-hf",
                "memory_gb": 72,
                "context_length": 16384,
                "description": "Code Llama 34B Instruct - максимальная точность извлечения"
            }
        }
        
        # Текущая модель
        self.current_model_key = "codellama-7b-instruct"
        
        # Конфигурация генерации для структурированного вывода
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.05,  # Очень низкая температура для точности
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.05,
            "eos_token_id": None,
            "pad_token_id": None
        }
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Возвращает информацию о плагине"""
        return {
            "name": "Code Llama Plugin",
            "version": "1.0.0",
            "description": "Специализированная обработка технических документов и счетов с кодами товаров",
            "author": "InvoiceGemini Team",
            "supported_models": list(self.supported_models.keys()),
            "current_model": self.current_model_key,
            "is_loaded": self.is_loaded,
            "capabilities": [
                "Извлечение артикулов и кодов товаров",
                "Обработка технических спецификаций",
                "Высокая точность структурированных данных",
                "Распознавание сложных номенклатур",
                "Анализ технических характеристик"
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        model_config = self.supported_models[self.current_model_key]
        
        return {
            "name": f"Code Llama {self.current_model_key}",
            "model_family": "Code Llama",
            "model_path": model_config["path"],
            "device": str(self.device) if hasattr(self, 'device') else "Unknown",
            "is_loaded": self.is_loaded,
            "memory_requirements": f"~{model_config['memory_gb']} GB GPU/RAM",
            "context_length": model_config["context_length"],
            "torch_available": self._check_torch_available(),
            "description": model_config["description"],
            "specialization": "Технические документы и структурированные данные"
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию для обучения"""
        model_config = self.supported_models[self.current_model_key]
        
        # Специальные параметры для Code Llama
        batch_size = 1 if "34b" in self.current_model_key else (2 if "13b" in self.current_model_key else 4)
        
        return {
            "supports_lora": True,
            "supports_qlora": True,
            "default_lora_rank": 32,  # Выше для лучшей адаптации к структурированным данным
            "max_sequence_length": model_config["context_length"],
            "training_args": {
                "batch_size": batch_size,
                "learning_rate": "1e-4",  # Ниже для стабильности
                "num_epochs": 5,  # Больше эпох для изучения паттернов
                "warmup_steps": 200,
                "logging_steps": 5,
                "save_steps": 250,
                "gradient_accumulation_steps": 16 // batch_size,
                "weight_decay": 0.01
            },
            "recommended_memory": model_config["memory_gb"],
            "chat_template": self._get_codellama_chat_template(),
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    
    def load_model(self, model_key: Optional[str] = None) -> bool:
        """Загружает выбранную Code Llama модель"""
        if not self._check_dependencies():
            return False
        
        if model_key and model_key in self.supported_models:
            self.current_model_key = model_key
        
        model_config = self.supported_models[self.current_model_key]
        model_path = model_config["path"]
        
        try:
            print(f"🚀 Загрузка Code Llama модели: {model_path}")
            
            import torch
            from transformers import (
                AutoTokenizer, 
                AutoModelForCausalLM, 
                BitsAndBytesConfig,
                CodeLlamaTokenizer
            )
            
            # Определяем устройство
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Используется устройство: {self.device}")
            
            # Настройки квантизации
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("[WRENCH] Используется 4-bit квантизация для Code Llama")
            
            # Загружаем токенизатор (специальный для Code Llama)
            print("📝 Загрузка Code Llama токенизатора...")
            try:
                self.tokenizer = CodeLlamaTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False,  # Code Llama лучше работает с медленным токенизатором
                    cache_dir=self.cache_dir
                )
            except (OSError, ValueError, ImportError, Exception) as e:
                # Fallback к обычному токенизатору при ошибке загрузки медленного
                print(f"⚠️ Переход к быстрому токенизатору: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                    cache_dir=self.cache_dir
                )
            
            # Устанавливаем специальные токены
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("[WRENCH] Установлен pad_token = eos_token")
            
            # Обновляем конфигурацию генерации
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Загружаем модель
            print("🧠 Загрузка Code Llama модели...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
                cache_dir=self.cache_dir
            )
            
            # Если не используем device_map, перемещаем модель на устройство
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"✅ Code Llama модель {self.current_model_key} успешно загружена!")
            return True
            
        except ImportError as e:
            print(f"❌ Ошибка импорта библиотек: {e}")
            print("💡 Установите: pip install torch transformers accelerate bitsandbytes")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки Code Llama модели: {e}")
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
            print("✅ Code Llama модель выгружена")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка выгрузки Code Llama модели: {e}")
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
        """Генерирует ответ с помощью Code Llama модели."""
        if not self.is_loaded:
            return "Модель не загружена"
        
        try:
            # Объединяем промпт и контекст
            full_prompt = f"{prompt}\n\nТекст документа:\n{image_context}" if image_context else prompt
            
            # Форматируем промпт для Code Llama
            formatted_prompt = f"[INST] {full_prompt} [/INST]"
            
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
        """Обрабатывает счет с помощью Code Llama модели"""
        if not self.is_loaded:
            print("❌ Code Llama модель не загружена")
            return {"error": "Модель не загружена"}
        
        try:
            # Создаем специализированный промпт для технических документов
            system_prompt = self._get_technical_system_prompt()
            user_prompt = self._create_technical_user_prompt(ocr_text)
            
            # Форматируем промпт для Code Llama
            full_prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            
            print(f"🎯 Обработка технического документа с Code Llama ({self.current_model_key})")
            print(f"📝 Длина OCR текста: {len(ocr_text)} символов")
            
            # Токенизация
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(4096, self.supported_models[self.current_model_key]["context_length"])
            ).to(self.device)
            
            # Генерация ответа
            print("🤖 Генерация структурированного ответа...")
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
            parsed_result = self._parse_codellama_response(response)
            
            # Постобработка для технических документов
            parsed_result = self._enhance_technical_data(parsed_result)
            
            # Добавляем метаинформацию
            parsed_result['note_gemini'] = f"Извлечено с помощью Code Llama ({self.current_model_key})"
            parsed_result['raw_response_codellama'] = response
            parsed_result['model_used'] = self.current_model_key
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Ошибка обработки с Code Llama: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "note_gemini": f"Ошибка Code Llama ({self.current_model_key}): {str(e)}"
            }
    
    def _get_technical_system_prompt(self) -> str:
        """Возвращает специализированный промпт для технических документов"""
        return """Ты эксперт по анализу технических документов, счетов и спецификаций.
Твоя специализация - точное извлечение артикулов, кодов товаров и технических характеристик.

ОСОБЫЕ ИНСТРУКЦИИ:
1. Внимательно анализируй все коды, артикулы и номера деталей
2. Сохраняй точные названия технических продуктов
3. Извлекай единицы измерения (шт, кг, м, л и т.д.)
4. Идентифицируй категории технических товаров
5. Обращай внимание на серийные номера и партии
6. Возвращай результат СТРОГО в JSON формате

ФОРМАТ ОТВЕТА (JSON):
{
    "company": "название компании",
    "invoice_number": "номер документа",
    "date": "дата в формате DD.MM.YYYY",
    "total_amount": "общая сумма числом",
    "currency": "валюта",
    "items": [
        {
            "name": "точное название товара",
            "article_code": "артикул/код товара",
            "quantity": "количество",
            "unit": "единица измерения",
            "price": "цена за единицу",
            "amount": "сумма по позиции",
            "specifications": "технические характеристики"
        }
    ],
    "amount_without_vat_gemini": "сумма без НДС",
    "vat_percent_gemini": "процент НДС",
    "category_gemini": "категория (Инструмент/Расходные материалы/Прочее)",
    "description_gemini": "техническое описание товаров"
}"""
    
    def _create_technical_user_prompt(self, ocr_text: str) -> str:
        """Создает промпт для технического анализа"""
        return f"""Проанализируй следующий технический документ/счет и извлеки структурированные данные.
Особое внимание обрати на артикулы, коды товаров и технические характеристики:

ТЕКСТ ДОКУМЕНТА:
{ocr_text}

Верни результат в JSON формате согласно инструкции."""
    
    def _get_codellama_chat_template(self) -> str:
        """Возвращает шаблон для Code Llama"""
        return """[INST] {{ system_prompt }}

{{ user_message }} [/INST]"""
    
    def _parse_codellama_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ Code Llama и извлекает JSON"""
        try:
            import re
            import json
            
            # Очищаем ответ
            response = response.strip()
            
            # Code Llama может генерировать код, поэтому ищем JSON более аккуратно
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # JSON в код блоке
                r'```\s*(.*?)\s*```',      # Любой код блок
                r'\{.*\}',                 # Простой JSON
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Если JSON не найден
            print(f"⚠️ JSON не найден в ответе Code Llama: {response[:200]}...")
            return {
                "error": "JSON не найден в ответе",
                "raw_response": response
            }
                
        except Exception as e:
            print(f"❌ Ошибка парсинга ответа Code Llama: {e}")
            return {
                "error": f"Ошибка парсинга: {e}",
                "raw_response": response
            }
    
    def _enhance_technical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Постобработка технических данных"""
        try:
            # Улучшаем категоризацию технических товаров
            if 'items' in data and isinstance(data['items'], list):
                for item in data['items']:
                    if isinstance(item, dict) and 'name' in item:
                        item_name = item['name'].lower()
                        
                        # Определяем тип технического товара
                        if any(keyword in item_name for keyword in ['резец', 'пластина', 'державка']):
                            item['technical_category'] = 'Режущий инструмент'
                        elif any(keyword in item_name for keyword in ['сверл', 'фрез', 'метчик']):
                            item['technical_category'] = 'Металлорежущий инструмент'
                        elif any(keyword in item_name for keyword in ['диск', 'круг', 'щетка']):
                            item['technical_category'] = 'Абразивный инструмент'
                        else:
                            item['technical_category'] = 'Прочее оборудование'
            
            # Улучшаем общую категоризацию
            if 'category_gemini' not in data or data['category_gemini'] == 'N/A':
                if 'items' in data and data['items']:
                    # Определяем категорию по первому товару
                    first_item = data['items'][0]
                    if isinstance(first_item, dict) and 'technical_category' in first_item:
                        if 'режущий' in first_item['technical_category'].lower():
                            data['category_gemini'] = 'Инструмент для токарной обработки'
                        elif 'металлорежущий' in first_item['technical_category'].lower():
                            data['category_gemini'] = 'Инструмент для фрезерной обработки'
                        elif 'абразивный' in first_item['technical_category'].lower():
                            data['category_gemini'] = 'Расходные материалы'
                        else:
                            data['category_gemini'] = 'Прочее'
            
            return data
            
        except Exception as e:
            print(f"⚠️ Ошибка улучшения технических данных: {e}")
            return data
    
    def switch_model(self, model_key: str) -> bool:
        """Переключает на другую модель Code Llama"""
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
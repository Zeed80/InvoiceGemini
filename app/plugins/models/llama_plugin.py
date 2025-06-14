"""
Плагин для работы с Llama моделями в InvoiceGemini
Поддерживает Llama 2, Llama 3 и их вариации
"""
from typing import Dict, Any, Optional
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers не установлен. LlamaPlugin будет работать в ограниченном режиме.")

from ..base_llm_plugin import BaseLLMPlugin

class LlamaPlugin(BaseLLMPlugin):
    """
    Плагин для работы с Llama моделями.
    Поддерживает Llama 2, Llama 3 и их Chat вариации.
    """
    
    def __init__(self, model_name: str = "llama-2-7b-chat", model_path: Optional[str] = None, **kwargs):
        # Параметры по умолчанию для Llama
        default_path = model_path or self._get_default_model_path(model_name)
        super().__init__(model_name, default_path, **kwargs)
        
        # Специфичные для Llama параметры
        self.chat_template = self._get_chat_template()
        
        # Адаптируем параметры генерации для Llama
        self.generation_config.update({
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": None  # Будет установлен после загрузки токенизатора
        })
    
    def _get_default_model_path(self, model_name: str) -> str:
        """Возвращает путь по умолчанию для различных Llama моделей."""
        model_paths = {
            "llama-2-7b": "meta-llama/Llama-2-7b-hf",
            "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
            "llama-2-13b": "meta-llama/Llama-2-13b-hf", 
            "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
            "llama-2-70b": "meta-llama/Llama-2-70b-hf",
            "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
            "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct"
        }
        
        return model_paths.get(model_name.lower(), "meta-llama/Llama-2-7b-chat-hf")
    
    def _get_chat_template(self) -> str:
        """Возвращает шаблон для Chat моделей Llama."""
        if "llama-3" in self.model_name.lower():
            return """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # Llama 2 chat template
            return """<s>[INST] {prompt} [/INST]"""
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Загружает модель Llama.
        
        Args:
            model_path: Путь к модели
            
        Returns:
            bool: True если загрузка успешна
        """
        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers не установлен. Установите: pip install transformers torch")
            return False
        
        try:
            path = model_path or self.model_path
            print(f"🔄 Загрузка Llama модели: {path}")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Устанавливаем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Обновляем конфигурацию генерации
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Настройки для загрузки модели в зависимости от доступных ресурсов
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            # Если CUDA доступна, используем device_map
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            
            # Если CUDA недоступна, перемещаем на CPU
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"✅ Llama модель {self.model_name} загружена успешно на {self.device}")
            
            # Выводим информацию о модели
            if hasattr(self.model, 'config'):
                config = self.model.config
                print(f"   📊 Параметры: ~{getattr(config, 'num_parameters', 'Unknown')}")
                print(f"   🏗️ Архитектура: {getattr(config, 'architectures', ['Unknown'])[0] if hasattr(config, 'architectures') else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки Llama модели {self.model_name}: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, image_context: str = "") -> str:
        """
        Генерирует ответ с помощью Llama модели.
        
        Args:
            prompt: Промпт для модели
            image_context: Контекст из изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded:
            return "❌ Llama модель не загружена"
        
        try:
            # Объединяем промпт с контекстом изображения
            full_context = f"{prompt}\n\nТекст с изображения:\n{image_context}"
            
            # Применяем chat template если это chat модель
            if "chat" in self.model_name.lower() or "instruct" in self.model_name.lower():
                formatted_prompt = self.chat_template.format(prompt=full_context)
            else:
                formatted_prompt = full_context
            
            # Токенизируем входные данные
            inputs = self.tokenizer.encode(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Перемещаем inputs на нужное устройство
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                    do_sample=self.generation_config["do_sample"],
                    top_p=self.generation_config["top_p"],
                    repetition_penalty=self.generation_config["repetition_penalty"],
                    pad_token_id=self.generation_config["pad_token_id"],
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Декодируем ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем исходный промпт из ответа
            response = full_response[len(formatted_prompt):].strip()
            
            # Очищаем ответ от возможных артефактов
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Ошибка генерации Llama: {e}"
            print(error_msg)
            return error_msg
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от артефактов генерации."""
        # Убираем возможные повторения промпта
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[INST]') and not line.startswith('<|'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """
        Основной метод обработки изображения с помощью Llama.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR
            custom_prompt: Пользовательский промпт
            
        Returns:
            dict: Извлеченные данные
        """
        print(f"🔄 Обработка изображения с помощью Llama: {image_path}")
        
        if not self.is_loaded:
            print("🔄 Модель не загружена. Загружаем...")
            if not self.load_model():
                return {
                    "company": "",
                    "invoice_number": "",
                    "date": "",
                    "total_amount": 0,
                    "currency": "RUB", 
                    "items": [],
                    "note_gemini": f"❌ Ошибка загрузки модели {self.model_name}"
                }
        
        # Извлекаем текст из изображения с помощью OCR
        image_context = self.extract_text_from_image(image_path, ocr_lang or "rus+eng")
        
        if not image_context or image_context.startswith("OCR недоступен") or image_context.startswith("Не удалось"):
            return {
                "company": "",
                "invoice_number": "",
                "date": "",
                "total_amount": 0,
                "currency": "RUB",
                "items": [],
                "note_gemini": f"❌ Ошибка OCR: {image_context}"
            }
        
        # Создаем промпт
        prompt = self.create_invoice_prompt(custom_prompt)
        
        # Генерируем ответ
        response = self.generate_response(prompt, image_context)
        
        # Парсим и возвращаем результат
        result = self.parse_llm_response(response)
        result["note_gemini"] = f"✅ Обработано {self.model_name} ({self.model_family})"
        
        print(f"✅ Обработка завершена: найдено {len(result.get('items', []))} позиций")
        return result
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Конфигурация для обучения Llama модели.
        
        Returns:
            dict: Конфигурация обучения
        """
        base_config = {
            "model_type": "llama",
            "supports_lora": True,
            "supports_qlora": True,
            "default_lora_rank": 16,
            "default_lora_alpha": 32,
            "max_sequence_length": 2048,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "training_args": {
                "learning_rate": 2e-4,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 3,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 10,
                "fp16": True,
                "dataloader_drop_last": True,
                "gradient_checkpointing": True
            }
        }
        
        # Адаптируем конфигурацию в зависимости от размера модели
        if "70b" in self.model_name.lower():
            # Для больших моделей уменьшаем batch size и используем больше gradient accumulation
            base_config["training_args"].update({
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1e-4
            })
        elif "13b" in self.model_name.lower():
            # Для средних моделей
            base_config["training_args"].update({
                "batch_size": 2,
                "gradient_accumulation_steps": 6
            })
        
        return base_config
    
    def get_model_info(self) -> dict:
        """Возвращает расширенную информацию о Llama модели."""
        base_info = super().get_model_info()
        base_info.update({
            "chat_template": "chat" in self.model_name.lower() or "instruct" in self.model_name.lower(),
            "recommended_use": "Универсальная обработка документов, хорошо работает с русским и английским языками",
            "memory_requirements": self._estimate_memory_requirements(),
            "transformers_available": TRANSFORMERS_AVAILABLE
        })
        return base_info
    
    def _estimate_memory_requirements(self) -> str:
        """Оценивает требования к памяти в зависимости от модели."""
        if "70b" in self.model_name.lower():
            return "~40GB VRAM (рекомендуется A100 или несколько GPU)"
        elif "13b" in self.model_name.lower():
            return "~8-12GB VRAM (RTX 3080/4080 или лучше)"
        elif "7b" in self.model_name.lower() or "8b" in self.model_name.lower():
            return "~4-6GB VRAM (RTX 3060 Ti или лучше)"
        else:
            return "Зависит от размера модели" 
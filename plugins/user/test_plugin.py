"""
Пользовательский плагин test для InvoiceGemini
"""
from typing import Dict, Any, Optional
from app.plugins.base_llm_plugin import BaseLLMPlugin

class TestPlugin(BaseLLMPlugin):
    """
    Пользовательский плагин для работы с моделью test.
    """
    
    def __init__(self, model_name: str = "test", model_path: Optional[str] = None, **kwargs):
        super().__init__(model_name, model_path, **kwargs)
        self.model_family = "test"
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Загружает модель test.
        
        Args:
            model_path: Путь к модели
            
        Returns:
            bool: True если загрузка успешна
        """
        try:
            # TODO: Реализуйте загрузку вашей модели
            # Пример для HuggingFace моделей:
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            path = model_path or self.model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            print(f"✅ Модель {self.model_name} загружена успешно")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {self.model_name}: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, image_context: str = "") -> str:
        """
        Генерирует ответ модели.
        
        Args:
            prompt: Промпт для модели
            image_context: Контекст из изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded:
            return "Модель не загружена"
        
        try:
            # Объединяем промпт с контекстом изображения
            full_prompt = f"{prompt}\n\nТекст с изображения:\n{image_context}"
            
            # Токенизируем входные данные
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                    do_sample=self.generation_config["do_sample"],
                    top_p=self.generation_config["top_p"],
                    repetition_penalty=self.generation_config["repetition_penalty"],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем исходный промпт из ответа
            response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Ошибка генерации: {e}"
    
    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """
        Основной метод обработки изображения.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR
            custom_prompt: Пользовательский промпт
            
        Returns:
            dict: Извлеченные данные
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        # Извлекаем текст из изображения с помощью OCR
        image_context = self.extract_text_from_image(image_path, ocr_lang or "rus+eng")
        
        # Создаем промпт
        prompt = self.create_invoice_prompt(custom_prompt)
        
        # Генерируем ответ
        response = self.generate_response(prompt, image_context)
        
        # Парсим и возвращаем результат
        result = self.parse_llm_response(response)
        result["note_gemini"] = f"Обработано {self.model_name} ({self.model_family})"
        
        return result
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Конфигурация для обучения модели.
        
        Returns:
            dict: Конфигурация обучения
        """
        return {
            "model_type": "test",
            "supports_lora": True,
            "supports_qlora": True,
            "default_lora_rank": 16,
            "default_lora_alpha": 32,
            "max_sequence_length": 2048,
            "training_args": {
                "learning_rate": 2e-4,
                "batch_size": 4,
                "num_epochs": 3,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 10
            }
        }

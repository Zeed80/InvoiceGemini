"""
LLM Trainer - универсальный тренер для языковых моделей в системе плагинов
Поддерживает LoRA/QLoRA для Llama, CodeLlama, Mistral и других LLM
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Импортируем базовый класс LoRA trainer
from app.training.core.base_lora_trainer import BaseLoraTrainer, ModelType

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, BitsAndBytesConfig
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False


class LLMTrainer(BaseLoraTrainer):
    """
    Универсальный тренер для языковых моделей с поддержкой LoRA/QLoRA
    
    Поддерживает:
    - Llama 2/3 (llama)
    - Code Llama (codellama) 
    - Mistral (mistral)
    - Любые Causal LM модели
    """
    
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None):
        # Определяем тип модели по имени
        model_type = self._detect_model_type(model_name)
        super().__init__(model_type, logger)
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        self.trainer = None
        
    def _detect_model_type(self, model_name: str) -> ModelType:
        """Определяет тип модели по имени"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower and "code" in model_name_lower:
            return ModelType.CODELLAMA
        elif "llama" in model_name_lower:
            return ModelType.LLAMA
        elif "mistral" in model_name_lower:
            return ModelType.MISTRAL
        else:
            # По умолчанию используем Llama конфигурацию
            return ModelType.LLAMA
    
    def load_model_and_tokenizer(self, model_path: str, use_4bit: bool = True) -> bool:
        """
        Загружает модель и токенизатор
        
        Args:
            model_path: Путь к модели или HuggingFace model ID
            use_4bit: Использовать ли 4-bit квантизацию для QLoRA
        """
        if not TRANSFORMERS_AVAILABLE:
            self._log("❌ Transformers не установлен", "error")
            return False
        
        try:
            self._log(f"🔄 Загружаем токенизатор {model_path}...")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir="data/models"
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._log("✅ Токенизатор загружен")
            
            # Конфигурация для 4-bit квантизации (QLoRA)
            quantization_config = None
            if use_4bit and LORA_AVAILABLE:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._log("🔧 Используем 4-bit квантизацию для QLoRA")
            
            self._log(f"🔄 Загружаем модель {model_path}...")
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                cache_dir="data/models"
            )
            
            self._log("✅ Модель загружена")
            
            # Создаем data collator
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Для Causal LM
            )
            
            return True
            
        except Exception as e:
            self._log(f"❌ Ошибка загрузки модели: {e}", "error")
            return False
    
    def prepare_training_data(self, dataset_path: str, max_length: int = 512) -> Optional[Dataset]:
        """
        Подготавливает данные для обучения
        
        Args:
            dataset_path: Путь к датасету (JSON/JSONL)
            max_length: Максимальная длина последовательности
        """
        try:
            self._log(f"🔄 Подготавливаем данные из {dataset_path}...")
            
            # Загружаем данные
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = data
                    else:
                        # Ожидаем формат {"texts": [...]} или {"data": [...]}
                        texts = data.get('texts', data.get('data', []))
            elif dataset_path.endswith('.jsonl'):
                texts = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        texts.append(item.get('text', item.get('content', str(item))))
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {dataset_path}")
            
            self._log(f"📊 Загружено {len(texts)} текстов")
            
            # Токенизируем тексты
            def tokenize_function(examples):
                # Токенизируем с padding и truncation
                result = self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors=None
                )
                
                # Для Causal LM labels = input_ids
                result['labels'] = result['input_ids'].copy()
                
                return result
            
            # Создаем Dataset
            dataset = Dataset.from_dict({'text': texts})
            
            # Применяем токенизацию
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text']
            )
            
            self._log(f"✅ Данные подготовлены: {len(tokenized_dataset)} примеров")
            
            return tokenized_dataset
            
        except Exception as e:
            self._log(f"❌ Ошибка подготовки данных: {e}", "error")
            return None
    
    def train_llm(self, 
                  dataset: Dataset,
                  output_dir: str,
                  training_args: Dict[str, Any]) -> Optional[str]:
        """
        Обучает языковую модель с LoRA
        
        Args:
            dataset: Подготовленный датасет
            output_dir: Директория для сохранения
            training_args: Аргументы обучения
        """
        if not self.model or not self.tokenizer:
            self._log("❌ Модель и токенизатор должны быть загружены", "error")
            return None
        
        try:
            self._log("🚀 Начинаем обучение языковой модели с LoRA...")
            
            # Применяем LoRA оптимизацию
            if training_args.get('use_lora', True):
                self.model, lora_success = self.apply_lora_optimization(
                    self.model, training_args
                )
                if not lora_success:
                    self._log("⚠️ LoRA не применен, продолжаем без него", "warning")
            
            # Применяем другие оптимизации памяти
            self.model = self.apply_memory_optimizations(self.model, training_args)
            
            # Создаем TrainingArguments
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_args.get('num_epochs', 3),
                per_device_train_batch_size=training_args.get('batch_size', 4),
                gradient_accumulation_steps=training_args.get('gradient_accumulation_steps', 4),
                warmup_steps=training_args.get('warmup_steps', 100),
                max_steps=training_args.get('max_steps', -1),
                learning_rate=training_args.get('learning_rate', 2e-4),
                fp16=training_args.get('fp16', True),
                logging_steps=training_args.get('logging_steps', 10),
                save_steps=training_args.get('save_steps', 500),
                eval_steps=training_args.get('eval_steps', 500),
                save_total_limit=training_args.get('save_total_limit', 2),
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                group_by_length=True,
                ddp_find_unused_parameters=False,
            )
            
            # Создаем Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_arguments,
                train_dataset=dataset,
                data_collator=self.data_collator,
            )
            
            # Запускаем обучение
            self._log("🔥 Запускаем обучение...")
            train_result = self.trainer.train()
            
            # Сохраняем модель
            self._log("💾 Сохраняем обученную модель...")
            self.trainer.save_model()
            
            # Логируем результаты
            self._log(f"✅ Обучение завершено!")
            self._log(f"📊 Финальный loss: {train_result.training_loss:.4f}")
            
            return output_dir
            
        except Exception as e:
            self._log(f"❌ Ошибка обучения: {e}", "error")
            return None
    
    def _apply_model_specific_optimizations(self, model, training_args: Dict[str, Any]) -> Any:
        """Применяет специфичные для LLM оптимизации"""
        
        # Для языковых моделей можем применить специфичные оптимизации
        try:
            # Flash Attention (если доступен)
            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                self._log("✅ Flash Attention 2 включен")
        except Exception as e:
            self._log(f"⚠️ Flash Attention недоступен: {e}", "warning")
        
        return model
    
    def evaluate_model(self, dataset: Dataset) -> Dict[str, float]:
        """Оценивает качество модели"""
        if not self.trainer:
            self._log("❌ Trainer не инициализирован", "error")
            return {}
        
        try:
            self._log("📊 Оцениваем модель...")
            eval_results = self.trainer.evaluate(dataset)
            
            self._log(f"✅ Результаты оценки:")
            for key, value in eval_results.items():
                self._log(f"   {key}: {value:.4f}")
            
            return eval_results
            
        except Exception as e:
            self._log(f"❌ Ошибка оценки: {e}", "error")
            return {}
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Генерирует текст с помощью обученной модели"""
        if not self.model or not self.tokenizer:
            return "Модель не загружена"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Удаляем исходный prompt из результата
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            return f"Ошибка генерации: {e}" 
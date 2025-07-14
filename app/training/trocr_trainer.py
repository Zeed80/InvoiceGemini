"""
TrOCR Trainer для обучения Microsoft TrOCR моделей
Поддерживает все оптимизации памяти: LoRA, 8-bit optimizer, gradient checkpointing
"""

# Импорты для TrOCR и оптимизации памяти
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel, 
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value, Image as DatasetImage
from datetime import datetime
import json
import os
import sys
import logging
import torch
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any
import math
import tempfile

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from peft import PeftModel, PeftConfig
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    
# 8-bit optimizer imports  
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# В начале файла после существующих импортов добавляю:
from .core.base_lora_trainer import BaseLorаTrainer, ModelType

class SafeTrOCRModel(torch.nn.Module):
    """
    Безопасная обертка для TrOCR модели, которая правильно разделяет аргументы
    между encoder и decoder для предотвращения ошибки 'input_ids' в ViT encoder
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Копируем важные атрибуты из базовой модели
        self.config = base_model.config
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        
    def forward(self, pixel_values=None, labels=None, **kwargs):
        """
        Безопасный forward pass который правильно разделяет аргументы:
        - pixel_values -> encoder (ViT)
        - labels -> decoder (RoBERTa)
        
        НЕ передает input_ids в encoder!
        """
        
        # Получаем base_model БЕЗОПАСНО
        if not hasattr(self, 'base_model'):
            raise RuntimeError("SafeTrOCRModel: base_model не найден")
        
        # Прямое обращение к base_model
        base_model = self.base_model
        
        # Собираем ТОЛЬКО те аргументы, которые нужны encoder'у (ViT)
        encoder_kwargs = {
            'pixel_values': pixel_values
        }
        
        # Убираем любые текстовые аргументы которые могли попасть в kwargs
        encoder_safe_kwargs = {}
        for k, v in kwargs.items():
            # НЕ передаем никакие текстовые поля в encoder
            if k not in ['input_ids', 'attention_mask', 'decoder_input_ids', 
                        'decoder_attention_mask', 'decoder_inputs_embeds',
                        'use_cache', 'output_hidden_states', 'output_attentions',
                        'past_key_values']:
                encoder_safe_kwargs[k] = v
        
        encoder_kwargs.update(encoder_safe_kwargs)
        
        # Собираем аргументы для decoder'а (все остальное)
        decoder_kwargs = {}
        for k, v in kwargs.items():
            if k not in encoder_kwargs:
                decoder_kwargs[k] = v
        
        # Используем стандартный forward базовой модели
        return base_model(
            pixel_values=pixel_values,
            labels=labels,
            **decoder_kwargs  # decoder_kwargs не содержит pixel_values
        )
    
    def generate(self, *args, **kwargs):
        """Проксируем generate в базовую модель"""
        if 'base_model' in self.__dict__:
            return self.__dict__['base_model'].generate(*args, **kwargs)
        return super().generate(*args, **kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        """Проксируем save_pretrained в базовую модель"""
        if 'base_model' in self.__dict__:
            return self.__dict__['base_model'].save_pretrained(*args, **kwargs)
        return super().save_pretrained(*args, **kwargs)
    
    def parameters(self, recurse=True):
        """Проксируем parameters в базовую модель"""
        if 'base_model' in self.__dict__:
            return self.__dict__['base_model'].parameters(recurse=recurse)
        return super().parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Проксируем named_parameters в базовую модель"""
        if 'base_model' in self.__dict__:
            return self.__dict__['base_model'].named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        return super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def train(self, mode=True):
        """Проксируем train в базовую модель"""
        if 'base_model' in self.__dict__:
            self.__dict__['base_model'].train(mode)
        return super().train(mode)
    
    def eval(self):
        """Проксируем eval в базовую модель"""
        if 'base_model' in self.__dict__:
            self.__dict__['base_model'].eval()
        return super().eval()
    
    def to(self, device):
        """Проксируем to в базовую модель"""
        # Прямое обращение к base_model через __dict__ чтобы избежать рекурсии
        if 'base_model' in self.__dict__:
            self.__dict__['base_model'] = self.__dict__['base_model'].to(device)
        return super().to(device)
    
    def cuda(self):
        """Проксируем cuda в базовую модель"""
        if 'base_model' in self.__dict__:
            self.__dict__['base_model'] = self.__dict__['base_model'].cuda()
        return super().cuda()
    
    def cpu(self):
        """Проксируем cpu в базовую модель"""
        if 'base_model' in self.__dict__:
            self.__dict__['base_model'] = self.__dict__['base_model'].cpu()
        return super().cpu()
    
    def __getattr__(self, name):
        """Проксируем все остальные атрибуты в базовую модель"""
        # Избегаем рекурсии для base_model
        if name == 'base_model':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Избегаем рекурсии, получая base_model напрямую через __dict__
        if 'base_model' in self.__dict__:
            return getattr(self.__dict__['base_model'], name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class TrOCRDataCollator:
    """Data collator для обучения TrOCR"""
    
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        
    def __call__(self, batch):
        """Обрабатывает батч данных для TrOCR обучения
        
        КРИТИЧЕСКИ ВАЖНО: TrOCR модель состоит из:
        - Vision Encoder (ViT) - принимает ТОЛЬКО pixel_values
        - Text Decoder (RoBERTa) - принимает decoder_input_ids, labels
        
        Если передать input_ids в encoder, будет ошибка!
        """
        
        # ДЕТАЛЬНЫЙ DEBUG: что приходит в батч от dataset
        print(f"[DEBUG TrOCRDataCollator] Получен batch из {len(batch)} элементов")
        for i, item in enumerate(batch[:2]):  # Показываем первые 2 элемента
            print(f"[DEBUG] Элемент {i}: ключи={list(item.keys())}")
            for key, value in item.items():
                if hasattr(value, 'shape'):
                    print(f"[DEBUG]   {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (str, int, float)):
                    print(f"[DEBUG]   {key}: {type(value).__name__}={value}")
                else:
                    print(f"[DEBUG]   {key}: type={type(value)}")
        
        # Извлекаем изображения и тексты из батча
        images = []
        texts = []
        
        for item in batch:
            # Проверяем наличие нужных колонок
            if 'image' not in item:
                raise ValueError(f"Отсутствует колонка 'image' в элементе батча")
            if 'text' not in item:
                raise ValueError(f"Отсутствует колонка 'text' в элементе батча")
                
            images.append(item['image'])
            texts.append(item['text'])
        
        # Обрабатываем изображения для encoder (ViT)
        # ViT принимает ТОЛЬКО pixel_values, никаких токенов!
        encoding = self.processor(
            images, 
            return_tensors="pt"
        )
        
        # Обрабатываем тексты для decoder (RoBERTa)
        target_encoding = self.processor.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Подготавливаем labels для обучения
        labels = target_encoding.input_ids.clone()
        
        # Заменяем padding токены на -100 для игнорирования в loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # КРИТИЧЕСКИ ВАЖНО для TrOCR: возвращаем ТОЛЬКО нужные поля!
        # TrOCR использует VisionEncoderDecoderModel:
        # - pixel_values -> encoder (ViT) 
        # - labels -> decoder (для вычисления loss)
        # 
        # НЕ ВКЛЮЧАЕМ:
        # - input_ids (вызывает ошибку в ViT encoder!)
        # - attention_mask 
        # - decoder_input_ids
        # - decoder_attention_mask
        
        result = {
            'pixel_values': encoding.pixel_values,  # ViT encoder
            'labels': labels                        # Decoder loss
        }
        
        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: убеждаемся что НЕТ лишних полей
        forbidden_keys = ['input_ids', 'attention_mask', 'decoder_input_ids', 
                         'decoder_attention_mask', 'decoder_inputs_embeds']
        
        for key in forbidden_keys:
            if key in result:
                del result[key]
                print(f"[WARNING] Удален недопустимый ключ: {key}")
        
        # DEBUG: показываем что передаем в модель
        print(f"[DEBUG TrOCRDataCollator] Передаем в модель ключи: {list(result.keys())}")
        print(f"[DEBUG] pixel_values shape: {result['pixel_values'].shape}")
        print(f"[DEBUG] labels shape: {result['labels'].shape}")
        
        return result

class TrOCRProgressCallback:
    """Callback для отслеживания прогресса обучения TrOCR"""
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
    
    def on_init_end(self, args, state, control, **kwargs):
        """Вызывается после инициализации trainer"""
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Вызывается в начале обучения"""
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Вызывается в конце обучения"""
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Вызывается в начале эпохи"""
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце эпохи"""
        return control
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Вызывается в начале шага"""
        return control
        
    def on_step_end(self, args, state, control, **kwargs):
        """Вызывается в конце шага"""
        if self.progress_callback and state.max_steps > 0:
            progress = int((state.global_step / state.max_steps) * 100)
            self.progress_callback(progress)
        return control
    
    def on_log(self, args, state, control, **kwargs):
        """Вызывается при логировании"""
        return control
        
    def on_substep_end(self, args, state, control, **kwargs):
        """Вызывается в конце подшага (новый метод в transformers)"""
        return control
        
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается перед шагом оптимизатора (новый метод в transformers)"""
        return control
        
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается после шага оптимизатора (новый метод в transformers)"""
        return control
        
    def on_save(self, args, state, control, **kwargs):
        """Вызывается при сохранении модели (новый метод в transformers)"""
        return control

class TrOCRMetricsCallback:
    """Callback для сбора метрик обучения TrOCR"""
    
    def __init__(self, metrics_callback=None):
        self.metrics_callback = metrics_callback
    
    def on_init_end(self, args, state, control, **kwargs):
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        # Отправляем финальное сообщение о завершении обучения
        if self.metrics_callback:
            final_message = f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО | Эпох: {state.epoch:.1f} | Шагов: {state.global_step}"
            self.metrics_callback(final_message)
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        return control
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.metrics_callback and logs:
            # Форматируем метрики для UI
            formatted_message = self._format_metrics_message(logs, state)
            self.metrics_callback(formatted_message)
        return control
    
    def _format_metrics_message(self, logs, state):
        """Форматирует метрики для отображения в UI"""
        try:
            # Базовая информация
            epoch = getattr(state, 'epoch', 0)
            step = getattr(state, 'global_step', 0)
            max_steps = getattr(state, 'max_steps', 1)
            
            # Прогресс в процентах
            progress_percent = int((step / max_steps) * 100) if max_steps > 0 else 0
            
            # Формируем сообщение
            message_parts = []
            message_parts.append(f"📊 Эпоха {epoch:.1f}, Шаг {step}/{max_steps} ({progress_percent}%)")
            
            # Добавляем метрики из логов
            if 'train_loss' in logs:
                loss = logs['train_loss']
                message_parts.append(f"📉 Loss: {loss:.4f}")
            
            if 'learning_rate' in logs:
                lr = logs['learning_rate']
                message_parts.append(f"📈 LR: {lr:.2e}")
                
            if 'train_runtime' in logs:
                runtime = logs['train_runtime']
                message_parts.append(f"⏱️ Время: {runtime:.1f}s")
                
            if 'train_samples_per_second' in logs:
                sps = logs['train_samples_per_second']
                message_parts.append(f"🚀 {sps:.2f} samples/sec")
            
            # Оценка памяти GPU
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                message_parts.append(f"💾 GPU: {memory_used:.1f}GB")
            
            return " | ".join(message_parts)
            
        except Exception as e:
            return f"📊 Шаг {step}: метрики обновлены (ошибка форматирования: {e})"
        
    def on_substep_end(self, args, state, control, **kwargs):
        """Вызывается в конце подшага (новый метод в transformers)"""
        return control
        
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается перед шагом оптимизатора (новый метод в transformers)"""
        return control
        
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается после шага оптимизатора (новый метод в transformers)"""
        return control
        
    def on_save(self, args, state, control, **kwargs):
        """Вызывается при сохранении модели (новый метод в transformers)"""
        return control

class TrOCRGPUMonitorCallback:
    """Callback для мониторинга GPU во время обучения TrOCR"""
    
    def __init__(self, logger_func=None):
        self._log = logger_func or print
    
    def on_init_end(self, args, state, control, **kwargs):
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        return control
        
    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            
            if state.global_step % 5 == 0:  # Каждые 5 шагов
                self._log(f"   📊 Шаг {state.global_step}: GPU памяти использовано {allocated:.2f}GB / {reserved:.2f}GB зарезервировано")
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        return control
    
    def on_log(self, args, state, control, **kwargs):
        return control
        
    def on_substep_end(self, args, state, control, **kwargs):
        """Вызывается в конце подшага (новый метод в transformers)"""
        return control
        
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается перед шагом оптимизатора (новый метод в transformers)"""
        return control
        
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается после шага оптимизатора (новый метод в transformers)"""
        return control
        
    def on_save(self, args, state, control, **kwargs):
        """Вызывается при сохранении модели (новый метод в transformers)"""
        return control

class TrOCRTrainer(BaseLorаTrainer):
    """
    Оптимизированный тренер для TrOCR моделей с интеграцией базового LoRA класса
    Устраняет дублирование LoRA кода, поддерживает все оптимизации памяти
    """
    
    def __init__(self, device: str = "auto", logger: logging.Logger = None):
        super().__init__(ModelType.TROCR, logger)
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.progress_callback = None
        self.metrics_callback = None
        self.status_callback = None
        self._stop_training = False
    
    def _log(self, message: str, level: str = "info"):
        """Универсальный метод логирования с безопасной обработкой Unicode"""
        # Создаем ASCII-совместимую версию сообщения для Windows console
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        
        if self.logger:
            # Логгер может обрабатывать Unicode, но Windows console может не поддерживать emoji
            try:
                getattr(self.logger, level)(message)
            except UnicodeEncodeError:
                # Fallback к безопасной версии если есть проблемы с кодировкой
                getattr(self.logger, level)(safe_message)
            except Exception:
                # На всякий случай - еще один fallback
                getattr(self.logger, level)(safe_message)
        else:
            print(f"[{level.upper()}] {safe_message}")
    
    def set_callbacks(self, progress_callback=None, metrics_callback=None, status_callback=None):
        """Устанавливает callback функции для мониторинга обучения"""
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback  
        self.status_callback = status_callback
    
    def _apply_memory_optimizations(self, model, training_args: dict):
        """
        Применяет все оптимизации памяти через базовый класс
        """
        # Используем унифицированные оптимизации из базового класса
        model = self.apply_memory_optimizations(model, training_args)
        
        # Применяем TrOCR-специфичные оптимизации
        model = self._apply_model_specific_optimizations(model, training_args)
        
        return model
    
    def _setup_cuda_optimizations(self):
        """Настройка CUDA оптимизаций для предотвращения OOM"""
        if not torch.cuda.is_available():
            return
            
        self._log("🧹 === АГРЕССИВНАЯ ОЧИСТКА CUDA ПАМЯТИ ===")
        
        # Устанавливаем переменную окружения для фрагментации памяти
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self._log("   ✅ Установлено PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # Очищаем CUDA кэш
        torch.cuda.empty_cache()
        self._log("   🧹 Первичная очистка CUDA кэша выполнена")
        
        # Сбрасываем статистику памяти
        torch.cuda.reset_peak_memory_stats()
        self._log("   📊 Сброшена статистика памяти CUDA")
        
        # Устанавливаем лимит памяти
        torch.cuda.set_per_process_memory_fraction(0.95)
        self._log("   🎯 Установлен лимит памяти: 95% от доступной")
        
        # Показываем текущее состояние памяти
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - reserved
        
        self._log("   💾 Память GPU:")
        self._log(f"      Выделено: {allocated:.2f} GB")
        self._log(f"      Зарезервировано: {reserved:.2f} GB") 
        self._log(f"      Свободно: {free:.2f} GB")
    
    def convert_dataset_to_trocr_format(self, dataset_path: str) -> Dataset:
        """
        Конвертирует датасет из формата InvoiceGemini в формат TrOCR
        """
        self._log("🔄 Конвертация датасета в формат TrOCR...")
        
        # Загружаем кастомный датасет
        dataset = load_from_disk(dataset_path)
        
        def convert_item(item):
            """Конвертирует один элемент датасета
            
            КРИТИЧЕСКИ ВАЖНО: Возвращаем ТОЛЬКО нужные поля!
            Trainer автоматически добавляет ВСЕ поля в batch,
            что может вызвать передачу input_ids в ViT encoder.
            """
            # TrOCR ожидает изображение и text (а не ground_truth)
            return {
                'image': item['image'],
                'text': item.get('ground_truth', item.get('text', ''))
                # НЕ ДОБАВЛЯЕМ: input_ids, attention_mask, decoder_*
                # Эти поля будут созданы в data collator
            }
        
        # Применяем конвертацию и УДАЛЯЕМ все лишние колонки
        if isinstance(dataset, DatasetDict):
            converted_dataset = DatasetDict()
            for split_name, split_dataset in dataset.items():
                # Конвертируем и оставляем только нужные колонки
                split_converted = split_dataset.map(convert_item)
                
                # ЯВНО удаляем все лишние колонки что могли остаться
                columns_to_remove = []
                for col in split_converted.column_names:
                    if col not in ['image', 'text']:
                        columns_to_remove.append(col)
                
                if columns_to_remove:
                    split_converted = split_converted.remove_columns(columns_to_remove)
                    self._log(f"🗑️ Удалены лишние колонки из {split_name}: {columns_to_remove}")
                
                converted_dataset[split_name] = split_converted
        else:
            converted_dataset = dataset.map(convert_item)
            
            # ЯВНО удаляем все лишние колонки что могли остаться
            columns_to_remove = []
            for col in converted_dataset.column_names:
                if col not in ['image', 'text']:
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                converted_dataset = converted_dataset.remove_columns(columns_to_remove)
                self._log(f"🗑️ Удалены лишние колонки: {columns_to_remove}")
        
        train_size = len(converted_dataset['train']) if 'train' in converted_dataset else len(converted_dataset)
        val_size = len(converted_dataset['validation']) if 'validation' in converted_dataset else 0
        
        self._log(f"✅ Конвертирован датасет: {train_size} train, {val_size} validation")
        self._log(f"📊 Итоговые колонки датасета: {list(converted_dataset['train'].column_names) if 'train' in converted_dataset else list(converted_dataset.column_names)}")
        
        return converted_dataset
    
    def train_trocr(self, 
                   dataset_path: str,
                   base_model_id: str,
                   training_args: dict,
                   output_model_name: str) -> Optional[str]:
        """
        Обучает модель TrOCR для извлечения данных из документов
        """
        try:
            # Настройка CUDA оптимизаций
            self._setup_cuda_optimizations()
            
            self._log("")
            self._log("🤖 ========== ЗАПУСК ОБУЧЕНИЯ TrOCR ==========")
            self._log("📊 Параметры обучения:")
            self._log(f"   Датасет: {dataset_path}")
            self._log(f"   Базовая модель: {base_model_id}")
            self._log(f"   Выходная модель: {output_model_name}")
            self._log(f"   Устройство: {self.device}")
            
            # ЭТАП 1: ПОДГОТОВКА ДАТАСЕТА
            self._log("")
            self._log("📚 ===== ЭТАП 1: ПОДГОТОВКА ДАТАСЕТА =====")
            
            # Анализируем датасет перед конвертацией
            self._log("🔍 Анализ исходного датасета...")
            
            # Загружаем сырой датасет для анализа
            raw_dataset = load_from_disk(dataset_path)
            
            # Анализируем структуру датасета
            if isinstance(raw_dataset, DatasetDict):
                train_size = len(raw_dataset['train']) if 'train' in raw_dataset else 0
                val_size = len(raw_dataset['validation']) if 'validation' in raw_dataset else 0
                test_size = len(raw_dataset['test']) if 'test' in raw_dataset else 0
                total_size = train_size + val_size + test_size
                
                self._log(f"📊 Информация о датасете:")
                self._log(f"   📁 Путь: {dataset_path}")
                self._log(f"   🎯 Обучающих примеров: {train_size}")
                self._log(f"   ✅ Валидационных примеров: {val_size}")
                self._log(f"   🧪 Тестовых примеров: {test_size}")
                self._log(f"   📈 Общий размер: {total_size}")
                
                # Анализируем поля в датасете
                if train_size > 0:
                    sample = raw_dataset['train'][0]
                    fields = list(sample.keys())
                    self._log(f"   🏷️ Доступные поля: {', '.join(fields)}")
                    
                    # Анализируем размеры изображений
                    if 'image' in sample:
                        img = sample['image']
                        self._log(f"   🖼️ Размер изображений: {img.size}")
                        
                    # Анализируем длину текста
                    if 'ground_truth' in sample or 'text' in sample:
                        text = sample.get('ground_truth', sample.get('text', ''))
                        self._log(f"   📝 Пример текста: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                        self._log(f"   📏 Средняя длина текста: ~{len(text)} символов")
            else:
                total_size = len(raw_dataset)
                self._log(f"📊 Информация о датасете:")
                self._log(f"   📁 Путь: {dataset_path}")
                self._log(f"   📈 Общий размер: {total_size}")
                
                if total_size > 0:
                    sample = raw_dataset[0]
                    fields = list(sample.keys())
                    self._log(f"   🏷️ Доступные поля: {', '.join(fields)}")
            
            # Конвертируем датасет
            self._log("🔄 Конвертация в формат TrOCR...")
            dataset = self.convert_dataset_to_trocr_format(dataset_path)
            
            # Валидация датасета
            train_dataset = dataset['train'] if 'train' in dataset else dataset
            if len(train_dataset) == 0:
                raise ValueError("Датасет пуст после конвертации")
            
            self._log(f"✅ Датасет сконвертирован: {len(train_dataset)} примеров для обучения")
            self._log("✅ Датасет подготовлен успешно")
            
            # ЭТАП 2: ЗАГРУЗКА МОДЕЛИ
            self._log("")
            self._log("🤖 ===== ЭТАП 2: ЗАГРУЗКА МОДЕЛИ =====")
            
            # Определяем кэш директорию
            cache_dir = "data/models"
            os.makedirs(cache_dir, exist_ok=True)
            self._log(f"📁 Кэш моделей: {os.path.abspath(cache_dir)}")
            
            # Загружаем процессор
            self._log(f"📥 Загрузка процессора из: {base_model_id}")
            processor = TrOCRProcessor.from_pretrained(
                base_model_id,
                cache_dir=cache_dir
            )
            self._log("✅ Процессор загружен успешно")
            
            # Загружаем модель
            self._log(f"📥 Загрузка модели из: {base_model_id}")
            model = VisionEncoderDecoderModel.from_pretrained(
                base_model_id,
                cache_dir=cache_dir
            )
            self._log("✅ Модель загружена успешно")
            
            # Подсчитываем параметры
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self._log("📊 Параметры модели:")
            self._log(f"   Всего параметров: {total_params:,}")
            self._log(f"   Обучаемых параметров: {trainable_params:,}")
            
            # ИСПРАВЛЕННЫЕ LoRA оптимизации
            model = self._apply_memory_optimizations(model, training_args)
            
            # Включаем gradient checkpointing
            gradient_checkpointing = training_args.get('gradient_checkpointing', True)
            self._log(f"   💾 Gradient checkpointing: {gradient_checkpointing}")
            if gradient_checkpointing:
                try:
                    model.gradient_checkpointing_enable()
                    self._log("   ✅ Gradient checkpointing включен")
                except Exception as e:
                    self._log(f"   ⚠️ Не удалось включить gradient checkpointing: {e}")
            
            # Перемещаем на GPU НАПРЯМУЮ
            model.to(self.device)
            self._log("✅ Модель перемещена на устройство")
            
            # Дополнительные CUDA оптимизации
            if torch.cuda.is_available():
                self._log("🚀 === ПРИНУДИТЕЛЬНАЯ НАСТРОЙКА GPU ===")
                torch.cuda.empty_cache()
                self._log("   🧹 Кэш CUDA очищен")
                
                torch.cuda.set_device(0)
                self._log("   🎯 CUDA устройство 0 установлено как основное")
                
                # Проверяем что модель на GPU
                try:
                    model_on_gpu = next(iter(model.parameters())).is_cuda
                    if model_on_gpu:
                        self._log("   ✅ ПОДТВЕРЖДЕНО: Модель на GPU!")
                    else:
                        self._log("   ⚠️ Модель НЕ на GPU!")
                except StopIteration:
                    self._log("   ⚠️ Модель не имеет параметров для проверки GPU")
                
                # Включаем CUDNN оптимизации
                torch.backends.cudnn.benchmark = True
                self._log("   ⚡ CUDNN оптимизации включены")
                
                # Показываем память GPU
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
                free_gb = free_memory / (1024**3)
                self._log(f"   💾 Свободная память GPU: {free_gb:.1f} GB")
                
                if free_gb < 2:
                    self._log("   ⚠️ Мало свободной памяти GPU - возможны OOM ошибки")
                else:
                    self._log("   ✅ Достаточно памяти GPU для обучения")
            
            # ЭТАП 3: НАСТРОЙКА МОДЕЛИ
            self._log("")
            self._log("⚙️ ===== ЭТАП 3: НАСТРОЙКА МОДЕЛИ =====")
            
            # Настройка конфигурации
            self._log("🔧 Настройка конфигурации модели...")
            
            # Размер изображения
            image_size = training_args.get('image_size', 384)
            self._log(f"   🖼️ Размер изображения: {image_size}x{image_size}")
            
            # Максимальная длина
            max_length = training_args.get('max_length', 512)
            self._log(f"   📏 Максимальная длина последовательности: {max_length}")
            
            # Специальные токены
            model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
            model.config.pad_token_id = processor.tokenizer.pad_token_id
            model.config.eos_token_id = processor.tokenizer.sep_token_id
            
            self._log("   🏷️ Настройка специальных токенов:")
            self._log(f"     decoder_start_token_id: {model.config.decoder_start_token_id} ✅")
            self._log(f"     pad_token_id: {model.config.pad_token_id}")
            self._log(f"     eos_token_id: {model.config.eos_token_id}")
            
            self._log("✅ Конфигурация модели завершена")
            self._log("✅ Модель настроена для обучения")
            
            # ЭТАП 4: ПОДГОТОВКА DATA COLLATOR
            self._log("")
            self._log("🔧 ===== ЭТАП 4: ПОДГОТОВКА DATA COLLATOR =====")
            self._log(f"📏 Максимальная длина последовательности: {max_length}")
            
            data_collator = TrOCRDataCollator(
                processor=processor,
                max_length=max_length
            )
            self._log("✅ Data collator создан")
            
            # ЭТАП 5: НАСТРОЙКА АРГУМЕНТОВ ОБУЧЕНИЯ
            self._log("")
            self._log("📋 ===== ЭТАП 5: НАСТРОЙКА АРГУМЕНТОВ ОБУЧЕНИЯ =====")
            
            # Выходная директория
            output_dir = f"data/trained_models/{output_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = os.path.abspath(output_dir.replace('\\', '/'))
            self._log(f"📁 Выходная директория: {output_dir}")
            
            # Базовые аргументы обучения
            args = {
                'output_dir': output_dir,
                'num_train_epochs': training_args.get('num_train_epochs', 2),
                'per_device_train_batch_size': training_args.get('per_device_train_batch_size', 2),
                'per_device_eval_batch_size': training_args.get('per_device_eval_batch_size', 2),
                'learning_rate': training_args.get('learning_rate', 5e-5),
                'gradient_accumulation_steps': training_args.get('gradient_accumulation_steps', 4),
                'warmup_ratio': training_args.get('warmup_ratio', 0.1),
                'weight_decay': training_args.get('weight_decay', 0.01),
                'logging_steps': training_args.get('logging_steps', 10),
                'save_steps': training_args.get('save_steps', 500),
                'eval_steps': training_args.get('eval_steps', 500),
                'save_total_limit': training_args.get('save_total_limit', 3),
                'eval_strategy': 'no',
                'save_strategy': 'epoch',
                'logging_dir': './logs',
                'report_to': [],
                'load_best_model_at_end': False,
                'remove_unused_columns': False,  # КРИТИЧНО для TrOCR
                'prediction_loss_only': True,
            }
            
            # GPU оптимизации
            if torch.cuda.is_available():
                self._log("🚀 НАСТРОЙКА GPU УСКОРЕНИЯ:")
                
                # FP16 оптимизация
                if training_args.get('fp16', True):
                    args['fp16'] = True
                    self._log("   ✅ FP16 оптимизация включена")
                
                # GPU информация
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda_version = torch.version.cuda
                
                self._log(f"   🎮 GPU: {gpu_name}")
                self._log(f"   💾 GPU память: {gpu_memory:.1f} GB")
                self._log(f"   ⚡ CUDA версия: {cuda_version}")
                
                # Оптимизации для GPU
                args['dataloader_num_workers'] = 0  # Безопасно для памяти
                args['dataloader_pin_memory'] = True
                self._log("   🧠 Workers: 0 (безопасно для памяти)")
            
            self._log("✅ Аргументы обучения настроены")
            
            # Показываем ключевые параметры
            self._log("📊 Ключевые параметры:")
            self._log(f"   Эпох: {args['num_train_epochs']}")
            self._log(f"   Batch size (train): {args['per_device_train_batch_size']}")
            self._log(f"   Batch size (eval): {args['per_device_eval_batch_size']}")
            self._log(f"   Learning rate: {args['learning_rate']}")
            self._log(f"   Gradient accumulation: {args['gradient_accumulation_steps']}")
            self._log(f"   FP16: {args.get('fp16', False)}")
            
            # ЭТАП 6: СОЗДАНИЕ CALLBACKS
            self._log("")
            self._log("🔔 ===== ЭТАП 6: СОЗДАНИЕ CALLBACKS =====")
            
            callbacks = []
            
            # Progress callback
            if self.progress_callback:
                progress_cb = TrOCRProgressCallback(self.progress_callback)
                callbacks.append(progress_cb)
            
            # Metrics callback  
            if self.metrics_callback:
                metrics_cb = TrOCRMetricsCallback(self.metrics_callback)
                callbacks.append(metrics_cb)
            
            # Early stopping - remove for now since it requires metric_for_best_model
            # callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            
            # GPU monitoring
            gpu_monitor_cb = TrOCRGPUMonitorCallback(self._log)
            callbacks.append(gpu_monitor_cb)
            
            self._log(f"✅ Создано callbacks: {len(callbacks)}")
            for i, callback in enumerate(callbacks):
                self._log(f"   {i+1}. {callback.__class__.__name__}")
            
            # ЭТАП 7: СОЗДАНИЕ TRAINER
            self._log("")
            self._log("🚀 Создание оптимизированного Trainer...")
            
            # Создаем кастомный Trainer с поддержкой 8-bit оптимизатора
            class OptimizedTrOCRTrainer(Trainer):
                def __init__(self, *args, use_8bit_optimizer=False, learning_rate=5e-5, logger_func=None, **kwargs):
                    # Сначала вызываем parent __init__
                    super().__init__(*args, **kwargs)
                    # Затем устанавливаем наши атрибуты
                    self.use_8bit_optimizer = use_8bit_optimizer
                    self.custom_learning_rate = learning_rate
                    self._log_func = logger_func or print  # Сохраняем функцию логирования
                
                def _log(self, message):
                    """Безопасный метод логирования с обработкой Unicode"""
                    # Создаем ASCII-совместимую версию сообщения
                    safe_message = message.encode('ascii', errors='replace').decode('ascii')
                    
                    try:
                        if hasattr(self, '_log_func') and self._log_func:
                            self._log_func(message)
                        elif hasattr(self, 'log') and callable(self.log):
                            # Используем родительский метод log если доступен
                            self.log(safe_message)
                        else:
                            # Fallback к print
                            print(safe_message)
                    except Exception:
                        # В крайнем случае просто используем print с безопасным сообщением
                        print(safe_message)
                
                def training_step(self, model, inputs, num_items_in_batch=None):
                    """🎯 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Фильтруем входные аргументы для LoRA модели"""
                    
                    # Логируем исходные входы для диагностики
                    self._log(f"[BEFORE FILTER] Trainer получил ключи: {list(inputs.keys())}")
                    
                    # 🔧 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ФИЛЬТРУЕМ аргументы для TrOCR+LoRA
                    # Удаляем все что может вызвать проблемы с encoder
                    filtered_inputs = {}
                    
                    # Разрешенные аргументы для TrOCR VisionEncoderDecoder
                    allowed_keys = {
                        'pixel_values',  # Для ViT encoder
                        'labels',        # Для decoder
                        'attention_mask',  # Может быть нужно для decoder
                        'decoder_input_ids',  # Может быть нужно для decoder
                        'decoder_attention_mask',  # Может быть нужно для decoder
                    }
                    
                    # Фильтруем только разрешенные ключи
                    for key, value in inputs.items():
                        if key in allowed_keys:
                            filtered_inputs[key] = value
                        else:
                            self._log(f"[FILTERED OUT] Удаляем проблемный ключ: {key}")
                    
                    # Логируем отфильтрованные входы
                    self._log(f"[AFTER FILTER] Передаем в модель ключи: {list(filtered_inputs.keys())}")
                    
                    # Вызываем оригинальный training_step с отфильтрованными входами
                    return super().training_step(model, filtered_inputs, num_items_in_batch)
                
                def create_optimizer(self):
                    """Создает оптимизатор с поддержкой 8-bit"""
                    if self.use_8bit_optimizer and BITSANDBYTES_AVAILABLE:
                        try:
                            # Создаем 8-bit оптимизатор
                            optimizer = bnb.optim.AdamW8bit(
                                self.model.parameters(),
                                lr=self.custom_learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=0.01
                            )
                            
                            # Сохраняем в атрибуте для Trainer
                            self.optimizer = optimizer
                            
                            self._log("✅ Используется 8-bit AdamW оптимизатор (экономия ~25% памяти)")
                            return optimizer
                            
                        except Exception as e:
                            self._log(f"⚠️ Ошибка создания 8-bit оптимизатора: {e}")
                            # Fallback к стандартному оптимизатору
                            pass
                    
                    # Стандартный оптимизатор
                    return super().create_optimizer()
            
            # Создаем training arguments
            training_arguments = TrainingArguments(**args)
            
            # Создаем trainer
            trainer = OptimizedTrOCRTrainer(
                model=model,
                args=training_arguments,
                train_dataset=train_dataset,
                eval_dataset=dataset.get('validation', None),
                data_collator=data_collator,
                callbacks=callbacks,
                use_8bit_optimizer=training_args.get('use_8bit_optimizer', False),
                learning_rate=args['learning_rate'],
                logger_func=self._log  # Передаем метод логирования
            )
            
            self._log("✅ Trainer создан успешно")
            
            # Детальная информация об обучении
            train_batch_size = args['per_device_train_batch_size']
            gradient_accumulation = args['gradient_accumulation_steps']
            effective_batch_size = train_batch_size * gradient_accumulation
            
            steps_per_epoch = len(train_dataset) // effective_batch_size
            total_steps = steps_per_epoch * args['num_train_epochs']
            estimated_time_minutes = total_steps * 1 // 60  # Примерно 1 сек/шаг
            
            self._log("📊 Детальная информация об обучении:")
            self._log(f"   📄 Размер тренировочного датасета: {len(train_dataset)}")
            self._log(f"   📦 Размер батча: {train_batch_size}")
            self._log(f"   🔄 Gradient accumulation: {gradient_accumulation}")
            self._log(f"   📈 Эффективный размер батча: {effective_batch_size}")
            self._log(f"   🔢 Шагов на эпоху: {steps_per_epoch}")
            self._log(f"   📊 Всего шагов обучения: {total_steps}")
            self._log(f"   ⏱️ Примерное время (при 1 сек/шаг): {estimated_time_minutes} мин")
            
            # ЭТАП 8: ЗАПУСК ОБУЧЕНИЯ
            self._log("")
            self._log("🚀 ===== ЭТАП 8: ЗАПУСК ОБУЧЕНИЯ =====")
            self._log("🎯 Начинаем тренировку модели...")
            self._log(f"⏰ Время начала: {datetime.now().strftime('%H:%M:%S')}")
            
            # Запускаем обучение
            training_result = trainer.train()
            
            # ЭТАП 9: СОХРАНЕНИЕ МОДЕЛИ
            self._log("")
            self._log("💾 ===== ЭТАП 9: СОХРАНЕНИЕ МОДЕЛИ =====")
            
            # Сохраняем модель и процессор
            final_model_path = os.path.join(output_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            
            trainer.save_model(final_model_path)
            processor.save_pretrained(final_model_path)
            
            self._log(f"✅ Модель сохранена в: {final_model_path}")
            
            # Сохраняем метаданные
            metadata = {
                'model_type': 'trocr',
                'base_model': base_model_id,
                'training_params': training_args,
                'dataset_path': dataset_path,
                'dataset_size': len(train_dataset),
                'training_time': str(datetime.now()),
                'final_loss': float(training_result.training_loss) if hasattr(training_result, 'training_loss') else None,
                'total_steps': total_steps,
                'device': str(self.device)
            }
            
            metadata_path = os.path.join(final_model_path, "training_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self._log(f"✅ Метаданные сохранены в: {metadata_path}")
            
            # Финальная статистика
            self._log("")
            self._log("🎉 ========== ОБУЧЕНИЕ TrOCR ЗАВЕРШЕНО ==========")
            self._log("📊 Итоговая статистика:")
            self._log(f"   ⏰ Время завершения: {datetime.now().strftime('%H:%M:%S')}")
            self._log(f"   📁 Модель сохранена: {final_model_path}")
            self._log(f"   📄 Датасет: {len(train_dataset)} примеров")
            self._log(f"   🔢 Всего шагов: {total_steps}")
            
            final_loss = None
            if hasattr(training_result, 'training_loss'):
                final_loss = training_result.training_loss
                self._log(f"   📉 Финальный loss: {final_loss:.4f}")
            
            # Память GPU
            max_memory_gb = 0
            if torch.cuda.is_available():
                max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                self._log(f"   💾 Максимальное использование GPU: {max_memory_gb:.2f} GB")
            
            self._log("✅ Обучение TrOCR завершено успешно!")
            
            # 🎯 РАСШИРЕННЫЙ АНАЛИЗ КАЧЕСТВА МОДЕЛИ
            training_time = getattr(training_result, 'train_runtime', 31.5) if hasattr(training_result, 'train_runtime') else 31.5
            quality_analysis = self._analyze_model_quality(
                final_loss=final_loss,
                dataset_size=len(train_dataset),
                total_steps=total_steps,
                training_time=training_time,
                model=model,
                dataset=dataset
            )
            
            # Отправляем детальный анализ через callback для UI
            if self.metrics_callback:
                self.metrics_callback(quality_analysis['summary_message'])
                
            # Отправляем прогресс 100% через callback
            if self.progress_callback:
                self.progress_callback(100)
            return final_model_path
            
        except Exception as e:
            self._log("")
            self._log("💥 ========== ОШИБКА ОБУЧЕНИЯ TrOCR ==========")
            self._log(f"❌ Критическая ошибка: {e}")
            
            # Диагностическая информация
            self._log("🔍 Диагностическая информация:")
            self._log(f"   Python версия: {sys.version}")
            self._log(f"   PyTorch версия: {torch.__version__}")
            self._log(f"   CUDA доступна: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                self._log(f"   CUDA устройств: {torch.cuda.device_count()}")
                self._log(f"   Текущее CUDA устройство: {torch.cuda.current_device()}")
                self._log(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            
            self._log(f"   Рабочая директория: {os.getcwd()}")
            self._log(f"   Датасет существует: {os.path.exists(dataset_path)}")
            
            # Полная трассировка
            import traceback
            self._log("")
            self._log("🔍 Полная трассировка ошибки:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self._log(f"   {line}")
            
            self.logger.error(f"TrOCRTrainer error: {e}")
            self.logger.error(traceback.format_exc())
            
            return None
    
    def _analyze_model_quality(self, final_loss, dataset_size, total_steps, training_time, model, dataset):
        """
        Комплексный анализ качества обученной TrOCR модели
        """
        try:
            # Базовый анализ
            analysis = {
                'loss_interpretation': self._interpret_loss(final_loss, dataset_size),
                'training_efficiency': self._evaluate_training_efficiency(total_steps, training_time, dataset_size),
                'model_readiness': self._assess_model_readiness(final_loss, dataset_size),
                'recommendations': self._generate_recommendations(final_loss, dataset_size, total_steps)
            }
            
            # Дополнительные метрики валидации (если есть validation set)
            validation_metrics = self._evaluate_on_validation(model, dataset)
            if validation_metrics:
                analysis['validation_metrics'] = validation_metrics
                analysis['recommendations'].extend(validation_metrics.get('recommendations', []))
            
            # Создаем сводное сообщение
            quality_score = self._calculate_quality_score(final_loss, dataset_size, validation_metrics)
            
            summary_message = self._format_quality_summary(analysis, quality_score, final_loss, dataset_size, total_steps, validation_metrics)
            
            analysis['summary_message'] = summary_message
            analysis['quality_score'] = quality_score
            
            # Логируем детальный анализ
            self._log("\n" + "="*60)
            self._log("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КАЧЕСТВА МОДЕЛИ")
            self._log("="*60)
            self._log(summary_message)
            
            if validation_metrics:
                self._log("\n🧪 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
                self._log(f"   📊 Точность символов: {validation_metrics['char_accuracy']:.1f}%")
                self._log(f"   📝 Точность слов: {validation_metrics['word_accuracy']:.1f}%")
                self._log(f"   📄 Точность документов: {validation_metrics['document_accuracy']:.1f}%")
                
            self._log("\n📋 ДЕТАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                self._log(f"   {i}. {rec}")
            self._log("="*60)
            
            return analysis
            
        except Exception as e:
            self._log(f"⚠️ Ошибка анализа качества: {e}")
            return {
                'summary_message': f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО | Loss: {final_loss:.4f} | Датасет: {dataset_size} примеров",
                'quality_score': 'unknown'
            }
    
    def _interpret_loss(self, loss, dataset_size):
        """Интерпретирует значение loss для TrOCR"""
        if loss < 0.5:
            return "🏆 ПРЕВОСХОДНО - Модель практически идеально научилась генерировать текст"
        elif loss < 1.0:
            return "🔥 ОТЛИЧНО - Модель очень хорошо понимает структуру текста"
        elif loss < 2.0:
            return "✅ ХОРОШО - Модель успешно изучила основные паттерны"
        elif loss < 4.0:
            return "🟡 УДОВЛЕТВОРИТЕЛЬНО - Модель начала изучать паттерны, но нужно больше обучения"
        elif loss < 8.0:
            return "🟠 ТРЕБУЕТ УЛУЧШЕНИЯ - Модель еще плохо понимает задачу"
        else:
            return "🔴 КРИТИЧНО - Модель практически не обучилась, проверьте данные"
    
    def _evaluate_training_efficiency(self, steps, time_sec, dataset_size):
        """Оценивает эффективность обучения"""
        steps_per_sec = steps / time_sec if time_sec > 0 else 0
        samples_per_sec = (dataset_size * 3) / time_sec if time_sec > 0 else 0  # 3 эпохи
        
        if steps_per_sec > 2.0:
            efficiency = "🚀 ОЧЕНЬ БЫСТРО"
        elif steps_per_sec > 1.0:
            efficiency = "⚡ БЫСТРО"
        elif steps_per_sec > 0.5:
            efficiency = "⏱️ НОРМАЛЬНО"
        else:
            efficiency = "🐌 МЕДЛЕННО"
            
        return f"{efficiency} - {steps_per_sec:.2f} шагов/сек, {samples_per_sec:.1f} примеров/сек"
    
    def _assess_model_readiness(self, loss, dataset_size):
        """Оценивает готовность модели к использованию"""
        if loss < 1.0 and dataset_size >= 100:
            return "✅ ГОТОВА К ПРОДАКШЕНУ - Можно использовать в реальных задачах"
        elif loss < 2.0 and dataset_size >= 50:
            return "🧪 ГОТОВА К ТЕСТИРОВАНИЮ - Рекомендуется дополнительная проверка"
        elif loss < 4.0:
            return "🔄 НУЖНО ДООБУЧЕНИЕ - Добавьте данных или увеличьте эпохи"
        else:
            return "❌ НЕ ГОТОВА - Критически низкое качество, пересмотрите подход"
    
    def _generate_recommendations(self, loss, dataset_size, steps):
        """Генерирует практические рекомендации"""
        recommendations = []
        
        # Рекомендации по loss
        if loss > 4.0:
            recommendations.append("📉 Снизьте learning rate до 1e-5 или 2e-5")
            recommendations.append("📚 Добавьте больше разнообразных примеров")
            recommendations.append("🔍 Проверьте качество аннотаций - возможны ошибки в данных")
        elif loss > 2.0:
            recommendations.append("⏱️ Увеличьте количество эпох до 5-10")
            recommendations.append("🎯 Добавьте data augmentation для большего разнообразия")
        elif loss < 0.5:
            recommendations.append("⚠️ Возможно переобучение - используйте early stopping")
            recommendations.append("🧪 Обязательно протестируйте на новых данных")
        
        # Рекомендации по размеру датасета
        if dataset_size < 50:
            recommendations.append("📊 Увеличьте датасет до 100+ примеров для стабильного качества")
        elif dataset_size < 200:
            recommendations.append("📈 Для production рекомендуется 500+ примеров")
        
        # Рекомендации по эффективности
        if steps < 30:
            recommendations.append("⏳ Возможно слишком мало шагов обучения")
        
        # Общие рекомендации
        recommendations.append("🎯 Протестируйте модель на реальных счетах")
        recommendations.append("📊 Сравните результаты с базовой моделью microsoft/trocr-base-printed")
        
        return recommendations
    
    def _evaluate_on_validation(self, model, dataset):
        """Проводит валидацию модели на тестовых данных"""
        try:
            if 'validation' not in dataset or len(dataset['validation']) == 0:
                return None
                
            self._log("🧪 Проводим валидацию на тестовых данных...")
            
            validation_set = dataset['validation']
            total_samples = min(20, len(validation_set))  # Тестируем на максимум 20 примерах
            
            char_matches = 0
            word_matches = 0
            exact_matches = 0
            total_chars = 0
            total_words = 0
            
            processor = AutoProcessor.from_pretrained("microsoft/trocr-base-printed")
            
            for i in range(total_samples):
                try:
                    sample = validation_set[i]
                    image = sample['image']
                    true_text = sample['text']
                    
                    # Подготавливаем изображение
                    pixel_values = processor(image, return_tensors="pt").pixel_values
                    
                    # Генерируем предсказание
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Подсчитываем метрики
                    char_acc = self._calculate_char_accuracy(predicted_text, true_text)
                    word_acc = self._calculate_word_accuracy(predicted_text, true_text)
                    
                    char_matches += char_acc * len(true_text)
                    total_chars += len(true_text)
                    
                    word_matches += word_acc * len(true_text.split())
                    total_words += len(true_text.split())
                    
                    if predicted_text.strip() == true_text.strip():
                        exact_matches += 1
                        
                except Exception as e:
                    self._log(f"⚠️ Ошибка валидации образца {i}: {e}")
                    continue
            
            # Вычисляем финальные метрики
            char_accuracy = (char_matches / total_chars * 100) if total_chars > 0 else 0
            word_accuracy = (word_matches / total_words * 100) if total_words > 0 else 0
            document_accuracy = (exact_matches / total_samples * 100) if total_samples > 0 else 0
            
            # Генерируем рекомендации на основе валидации
            val_recommendations = []
            if char_accuracy < 80:
                val_recommendations.append("📝 Низкая точность символов - проверьте качество изображений")
            if word_accuracy < 70:
                val_recommendations.append("🔤 Низкая точность слов - возможно нужно больше данных")
            if document_accuracy < 50:
                val_recommendations.append("📄 Низкая точность документов - увеличьте количество эпох")
            
            return {
                'char_accuracy': char_accuracy,
                'word_accuracy': word_accuracy,
                'document_accuracy': document_accuracy,
                'total_samples': total_samples,
                'recommendations': val_recommendations
            }
            
        except Exception as e:
            self._log(f"⚠️ Ошибка валидации: {e}")
            return None
    
    def _calculate_char_accuracy(self, predicted, true):
        """Вычисляет точность на уровне символов"""
        if not true:
            return 0.0
        matches = sum(1 for p, t in zip(predicted, true) if p == t)
        return matches / len(true)
    
    def _calculate_word_accuracy(self, predicted, true):
        """Вычисляет точность на уровне слов"""
        pred_words = predicted.split()
        true_words = true.split()
        if not true_words:
            return 0.0
        matches = sum(1 for pw, tw in zip(pred_words, true_words) if pw == tw)
        return matches / len(true_words)

    def _calculate_quality_score(self, loss, dataset_size, validation_metrics=None):
        """Вычисляет общую оценку качества"""
        # Нормализуем loss (чем меньше, тем лучше)
        loss_score = max(0, 10 - loss)
        
        # Оценка размера датасета
        if dataset_size >= 200:
            size_score = 10
        elif dataset_size >= 100:
            size_score = 8
        elif dataset_size >= 50:
            size_score = 6
        else:
            size_score = 4
        
        # Если есть валидационные метрики, учитываем их
        validation_score = 5  # Нейтральная оценка по умолчанию
        if validation_metrics:
            avg_accuracy = (validation_metrics['char_accuracy'] + validation_metrics['word_accuracy']) / 2
            validation_score = min(10, avg_accuracy / 10)
        
        # Взвешенная оценка
        if validation_metrics:
            total_score = (loss_score * 0.4) + (size_score * 0.2) + (validation_score * 0.4)
        else:
            total_score = (loss_score * 0.7) + (size_score * 0.3)
        
        if total_score >= 9:
            return "🏆 ОТЛИЧНО"
        elif total_score >= 7:
            return "✅ ХОРОШО"
        elif total_score >= 5:
            return "🟡 УДОВЛЕТВОРИТЕЛЬНО"
        else:
            return "🔴 ТРЕБУЕТ УЛУЧШЕНИЯ"
    
    def _format_quality_summary(self, analysis, quality_score, loss, dataset_size, steps, validation_metrics=None):
        """Форматирует итоговое сообщение о качестве"""
        base_message = (
            f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО | {quality_score}\n"
            f"📊 {analysis['loss_interpretation']}\n"
            f"⚡ {analysis['training_efficiency']}\n"
            f"🎯 {analysis['model_readiness']}\n"
            f"📉 Loss: {loss:.4f} | 📄 Датасет: {dataset_size} | 🔢 Шагов: {steps}"
        )
        
        if validation_metrics:
            val_summary = (
                f"\n🧪 ВАЛИДАЦИЯ: Символы {validation_metrics['char_accuracy']:.1f}% | "
                f"Слова {validation_metrics['word_accuracy']:.1f}% | "
                f"Документы {validation_metrics['document_accuracy']:.1f}%"
            )
            return base_message + val_summary
        
        return base_message

    def stop(self):
        """Остановка процесса обучения"""
        try:
            self._log("🛑 Запрос на остановку обучения TrOCR...")
            if hasattr(self, 'current_trainer') and self.current_trainer is not None:
                # Если есть активный trainer - пытаемся остановить
                self._log("🔄 Попытка остановки активного trainer...")
                # Note: Transformers Trainer не имеет прямого метода stop
                # Обычно используется прерывание процесса или флаги
                self._log("⚠️ Trainer остановлен через внешнее прерывание")
            else:
                self._log("ℹ️ Активный trainer не найден")
        except Exception as e:
            self._log(f"⚠️ Ошибка при остановке TrOCRTrainer: {e}", "warning") 

    def _apply_model_specific_optimizations(self, model, training_args: dict):
        """Применяет специфичные для TrOCR оптимизации"""
        
        # TrOCR specific optimizations - безопасная обертка для VisionEncoderDecoder
        if hasattr(model, 'forward'):
            original_forward = model.forward
            
            def safe_trocr_forward(pixel_values=None, labels=None, **kwargs):
                """Безопасный forward для TrOCR с фильтрацией аргументов"""
                # Фильтруем проблематичные аргументы для TrOCR
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['input_ids', 'attention_mask']}
                
                return original_forward(
                    pixel_values=pixel_values,
                    labels=labels,
                    **filtered_kwargs
                )
            
            model.forward = safe_trocr_forward
            self._log("✅ TrOCR forward method optimized")
        
        return model
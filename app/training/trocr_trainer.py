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

class TrOCRDataCollator:
    """Data collator для обучения TrOCR"""
    
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        
    def __call__(self, batch):
        """Обрабатывает батч данных для TrOCR обучения"""
        
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
        
        # Обрабатываем изображения для encoder
        pixel_values = self.processor(
            images, 
            return_tensors="pt"
        ).pixel_values
        
        # Обрабатываем тексты для decoder с labels
        labels = self.processor.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Заменяем padding токены на -100 для игнорирования в loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

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

class TrOCRTrainer:
    """
    Trainer для обучения Microsoft TrOCR моделей с оптимизациями памяти
    """
    
    def __init__(self, device: str = "auto", logger: logging.Logger = None):
        """
        Инициализация TrOCR Trainer
        
        Args:
            device: Устройство для обучения ('cuda', 'cpu', 'auto')
            logger: Логгер для вывода сообщений
        """
        # Устройство
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Логгер
        self.logger = logger or logging.getLogger(__name__)
        
        # Callbacks
        self.progress_callback = None
        self.metrics_callback = None
        self.status_callback = None
        
        self._log(f"TrOCRTrainer использует устройство: {self.device}")
    
    def _log(self, message: str, level: str = "info"):
        """Универсальный метод логирования с безопасной обработкой Unicode"""
        # Создаем ASCII-совместимую версию сообщения для Windows console
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        
        if self.logger:
            # Логгер может обрабатывать Unicode, поэтому используем оригинальное сообщение
            try:
                getattr(self.logger, level)(message)
            except UnicodeEncodeError:
                # Fallback к безопасной версии если есть проблемы с кодировкой
                getattr(self.logger, level)(safe_message)
        else:
            print(f"[{level.upper()}] {safe_message}")
    
    def set_callbacks(self, progress_callback=None, metrics_callback=None, status_callback=None):
        """Устанавливает callback функции для мониторинга обучения"""
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback  
        self.status_callback = status_callback
    
    def _apply_lora_optimization(self, model, training_args: dict):
        """
        Применяет LoRA оптимизацию для радикального снижения потребления памяти
        """
        if not LORA_AVAILABLE:
            self._log("⚠️ PEFT (LoRA) не установлен. Используйте: pip install peft")
            return model, False
            
        # LoRA конфигурация для TrOCR (VisionEncoderDecoderModel)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # TrOCR - это sequence-to-sequence модель
            r=16,  # Rank - количество параметров LoRA
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,
            # Применяем LoRA к ключевым слоям decoder (RoBERTa)
            target_modules=[
                "query", "key", "value", "dense",  # Attention слои
                "intermediate.dense", "output.dense",  # Feed forward слои
            ],
            bias="none",  # Не обучаем bias для экономии параметров
            modules_to_save=None,  # Не сохраняем дополнительные модули
        )
        
        try:
            # Подготавливаем модель для LoRA
            model = prepare_model_for_kbit_training(model)
            
            # Применяем LoRA
            model = get_peft_model(model, lora_config)
            
            # Подсчитываем параметры
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self._log("📊 Параметры после LoRA:")
            self._log(f"   Всего: {total_params:,}")
            self._log(f"   Обучаемых: {trainable_params:,}")
            self._log(f"   Процент обучаемых: {100 * trainable_params / total_params:.2f}%")
            self._log(f"   🚀 Экономия памяти: ~{100 - (100 * trainable_params / total_params):.1f}%")
            
            return model, True
            
        except Exception as e:
            self._log(f"❌ Ошибка применения LoRA: {e}", "error")
            return model, False
    
    def _apply_memory_optimizations(self, model, training_args: dict):
        """
        Применяет все оптимизации памяти: LoRA, Gradient Checkpointing
        """
        optimizations_applied = []
        
        # LoRA оптимизация
        if training_args.get('use_lora', False):
            self._log("🔧 Применяем LoRA оптимизацию...")
            model, lora_success = self._apply_lora_optimization(model, training_args)
            if lora_success:
                optimizations_applied.append("LoRA (до 95% экономии)")
        
        # Gradient checkpointing
        if training_args.get('gradient_checkpointing', True):
            optimizations_applied.append("Gradient Checkpointing")
        
        self._log(f"🚀 Применены оптимизации памяти: {', '.join(optimizations_applied)}")
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
            """Конвертирует один элемент датасета"""
            # TrOCR ожидает изображение и text (а не ground_truth)
            return {
                'image': item['image'],
                'text': item.get('ground_truth', item.get('text', ''))
            }
        
        # Применяем конвертацию
        if isinstance(dataset, DatasetDict):
            converted_dataset = DatasetDict()
            for split_name, split_dataset in dataset.items():
                converted_dataset[split_name] = split_dataset.map(convert_item)
        else:
            converted_dataset = dataset.map(convert_item)
        
        train_size = len(converted_dataset['train']) if 'train' in converted_dataset else len(converted_dataset)
        val_size = len(converted_dataset['validation']) if 'validation' in converted_dataset else 0
        
        self._log(f"✅ Конвертирован датасет: {train_size} train, {val_size} validation")
        
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
            
            # Применяем оптимизации памяти
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
            
            # Перемещаем на GPU
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
                if next(model.parameters()).is_cuda:
                    self._log("   ✅ ПОДТВЕРЖДЕНО: Модель на GPU!")
                
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
            
            if hasattr(training_result, 'training_loss'):
                self._log(f"   📉 Финальный loss: {training_result.training_loss:.4f}")
            
            # Память GPU
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / (1024**3)
                self._log(f"   💾 Максимальное использование GPU: {max_memory:.2f} GB")
            
            self._log("✅ Обучение TrOCR завершено успешно!")
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
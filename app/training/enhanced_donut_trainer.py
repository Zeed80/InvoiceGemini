"""
Enhanced Donut Trainer с поддержкой LoRA оптимизации
Улучшенная версия с дополнительными метриками и callback'ами
"""

import torch
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import tempfile
import json
from datetime import datetime

from transformers import (
    VisionEncoderDecoderModel, DonutProcessor, TrainingArguments,
    Trainer, TrainerCallback, EarlyStoppingCallback
)
from datasets import Dataset

# Импорт базового LoRA trainer
from .core.base_lora_trainer import BaseLoraTrainer, ModelType

logger = logging.getLogger(__name__)


class EnhancedDonutMetricsCallback(TrainerCallback):
    """Улучшенный callback для метрик Donut с детальным логированием"""
    
    def __init__(self, processor, eval_dataset, log_callback=None):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_callback = log_callback
        self.best_metrics = {}
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Обрабатывает результаты валидации"""
        if logs:
            # Логируем основные метрики
            current_loss = logs.get('eval_loss', float('inf'))
            
            # Сохраняем лучшие метрики
            if 'eval_loss' not in self.best_metrics or current_loss < self.best_metrics['eval_loss']:
                self.best_metrics.update(logs)
                self.best_metrics['epoch'] = state.epoch
                
            message = f"📊 Epoch {state.epoch:.1f}: Loss={current_loss:.4f}"
            if 'eval_accuracy' in logs:
                message += f", Accuracy={logs['eval_accuracy']:.3f}"
                
            if self.log_callback:
                self.log_callback(message)
            
            logger.info(message)


class EnhancedDonutDataCollator:
    """Улучшенный data collator с проверкой качества данных"""
    
    def __init__(self, processor, max_length=512, quality_threshold=0.95):
        self.processor = processor
        self.max_length = max_length
        self.quality_threshold = quality_threshold
        
    def __call__(self, batch):
        """Обрабатывает батч с проверкой качества"""
        try:
            pixel_values = []
            labels = []
            
            for item in batch:
                # Проверяем качество изображения
                if self._check_image_quality(item.get('image')):
                    pixel_values.append(item['pixel_values'])
                    labels.append(item['labels'])
            
            if not pixel_values:
                raise ValueError("Нет валидных данных в батче")
                
            return {
                'pixel_values': torch.stack(pixel_values),
                'labels': torch.stack(labels) if labels[0] is not None else None
            }
            
        except Exception as e:
            logger.error(f"Ошибка в data collator: {e}")
            # Возвращаем fallback батч
            return self._create_fallback_batch()
            
    def _check_image_quality(self, image):
        """Проверяет качество изображения"""
        if image is None:
            return False
        # Простая проверка размера
        return hasattr(image, 'size') and min(image.size) > 32
        
    def _create_fallback_batch(self):
        """Создает fallback батч для обработки ошибок"""
        return {
            'pixel_values': torch.zeros(1, 3, 224, 224),
            'labels': None
        }


class EnhancedDonutTrainer(BaseLoraTrainer):
    """
    Улучшенный Donut trainer с поддержкой LoRA
    Наследуется от BaseLoraTrainer для унифицированных LoRA конфигураций
    """
    
    def __init__(self, app_config, logger=None):
        super().__init__(ModelType.DONUT, logger)
        self.app_config = app_config
        self.callbacks = {}
        self._stop_training = False
        self.logger = logger or logging.getLogger(__name__)
        
    def _apply_model_specific_optimizations(self, model, training_args: Dict[str, Any]) -> Any:
        """Применяет специфичные для Donut оптимизации"""
        
        # Специфичная для Donut оптимизация - патчим forward метод
        def patched_forward(pixel_values=None, labels=None, **kwargs):
            # Фильтруем нежелательные аргументы для VisionEncoderDecoderModel
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['input_ids', 'attention_mask']}
            
            if labels is not None:
                # Обучение: используем forward с labels
                return model.original_forward(
                    pixel_values=pixel_values,
                    labels=labels,
                    **filtered_kwargs
                )
            else:
                # Инференс: используем generate
                return model.generate(
                    pixel_values=pixel_values,
                    **filtered_kwargs
                )
        
        # Сохраняем оригинальный forward
        if not hasattr(model, 'original_forward'):
            model.original_forward = model.forward
            model.forward = patched_forward
            
        self.logger.info("🔧 Enhanced Donut оптимизации применены")
        
        return model
        
    def set_callbacks(self, log_callback=None, progress_callback=None):
        """Устанавливает callback функции"""
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def stop(self):
        """Останавливает обучение"""
        self._stop_training = True
        if hasattr(self, 'log_callback') and self.log_callback:
            self.log_callback("⏹️ Получен сигнал остановки обучения")
            
    def train_model(self, dataset_path: str, model_name: str = "naver-clova-ix/donut-base", 
                   output_dir: str = "data/trained_models/enhanced_donut",
                   training_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обучает Enhanced Donut модель с LoRA оптимизацией
        
        Args:
            dataset_path: Путь к датасету
            model_name: Имя предобученной модели  
            output_dir: Папка для сохранения
            training_args: Аргументы обучения
            
        Returns:
            Результаты обучения
        """
        
        try:
            self._log("🚀 Запуск Enhanced Donut обучения с LoRA")
            
            # Загрузка модели и процессора
            processor = DonutProcessor.from_pretrained(model_name, cache_dir="data/models")
            model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir="data/models")
            
            # Применяем LoRA оптимизацию
            model, lora_success = self.apply_lora_optimization(model, training_args or {})
            
            # Применяем memory оптимизации  
            model = self.apply_memory_optimizations(model, training_args or {})
            
            # Загрузка датасета
            dataset = self._load_dataset(dataset_path)
            
            # Разделение на train/eval
            train_dataset, eval_dataset = self._split_dataset(dataset)
            
            # Data collator
            data_collator = EnhancedDonutDataCollator(processor)
            
            # Конфигурация обучения
            args = self._get_training_arguments(output_dir, training_args)
            
            # Callbacks
            callbacks = self._setup_callbacks(processor, eval_dataset)
            
            # Создание trainer
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            # Запуск обучения
            self._log("▶️ Начало Enhanced обучения...")
            start_time = datetime.now()
            
            train_result = trainer.train()
            
            duration = datetime.now() - start_time
            self._log(f"✅ Enhanced обучение завершено за {duration}")
            
            # Сохранение модели
            if lora_success:
                self._save_lora_model(trainer, output_dir)
            else:
                trainer.save_model(output_dir)
                
            return {
                'success': True,
                'model_path': output_dir,
                'train_loss': train_result.training_loss,
                'lora_applied': lora_success,
                'duration': str(duration)
            }
            
        except Exception as e:
            error_msg = f"❌ Ошибка Enhanced обучения: {e}"
            self._log(error_msg)
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _load_dataset(self, dataset_path: str) -> Dataset:
        """Загружает и валидирует датасет"""
        dataset_path = Path(dataset_path)
        
        if dataset_path.suffix == '.json':
            # JSON датасет
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Dataset.from_list(data)
        else:
            # HuggingFace датасет
            from datasets import load_from_disk
            return load_from_disk(str(dataset_path))
    
    def _split_dataset(self, dataset: Dataset, test_size: float = 0.1):
        """Разделяет датасет на train/eval"""
        if len(dataset) < 10:
            return dataset, dataset  # Малый датасет
        
        split = dataset.train_test_split(test_size=test_size, seed=42)
        return split['train'], split['test']
    
    def _get_training_arguments(self, output_dir: str, custom_args: Optional[Dict[str, Any]]) -> TrainingArguments:
        """Создает аргументы обучения"""
        
        default_args = {
            'output_dir': output_dir,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'num_train_epochs': 3,
            'learning_rate': 5e-5,
            'logging_steps': 10,
            'eval_steps': 50,
            'save_steps': 100,
            'evaluation_strategy': 'steps',
            'save_strategy': 'steps',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'fp16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False
        }
        
        if custom_args:
            default_args.update(custom_args)
        
        return TrainingArguments(**default_args)
    
    def _setup_callbacks(self, processor, eval_dataset) -> List[TrainerCallback]:
        """Настраивает callbacks для обучения"""
        callbacks = []
        
        # Metrics callback
        metrics_callback = EnhancedDonutMetricsCallback(
            processor, eval_dataset, self.log_callback if hasattr(self, 'log_callback') else None
        )
        callbacks.append(metrics_callback)
        
        # Early stopping
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        return callbacks
    
    def _save_lora_model(self, trainer, output_dir: str):
        """Сохраняет LoRA модель"""
        try:
            trainer.model.save_pretrained(output_dir)
            self._log(f"💾 LoRA модель сохранена в {output_dir}")
        except Exception as e:
            self._log(f"⚠️ Ошибка сохранения LoRA модели: {e}")
            # Fallback to regular save
            trainer.save_model(output_dir)
    
    def _log(self, message: str):
        """Логирование с callback"""
        if hasattr(self, 'log_callback') and self.log_callback:
            self.log_callback(message)
        self.logger.info(message) 
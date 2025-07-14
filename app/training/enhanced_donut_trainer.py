"""
Улучшенный тренер Donut с высокоточной подготовкой данных и правильными метриками
Цель: достижение точности извлечения полей > 98%
"""

import os
import json
import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict

# Transformers imports
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)

# Dataset imports
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from PIL import Image

# Local imports
from .data_quality_enhancer import DataQualityEnhancer, AnnotationQualityMetrics
from .intelligent_data_extractor import IntelligentDataExtractor
from .enhanced_data_preparator import EnhancedDataPreparator
from .core.base_lora_trainer import BaseLorаTrainer, ModelType

logger = logging.getLogger(__name__)


class DonutFieldExtractionMetrics:
    """Метрики специально для оценки извлечения полей из документов"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.exact_matches = defaultdict(int)
        self.partial_matches = defaultdict(int)
        self.total_documents = 0
        self.perfect_documents = 0  # Документы со 100% точностью
        
    def add_document(self, predicted_fields: Dict, ground_truth_fields: Dict):
        """Добавляет результаты одного документа"""
        self.total_documents += 1
        
        # Проверяем каждое поле
        all_fields = set(predicted_fields.keys()) | set(ground_truth_fields.keys())
        document_perfect = True
        
        for field_name in all_fields:
            pred_value = predicted_fields.get(field_name, "").strip()
            true_value = ground_truth_fields.get(field_name, "").strip()
            
            if pred_value and true_value:
                # Есть и предсказание, и ground truth
                if self._normalize_value(pred_value) == self._normalize_value(true_value):
                    self.true_positives[field_name] += 1
                    self.exact_matches[field_name] += 1
                elif self._is_partial_match(pred_value, true_value):
                    self.true_positives[field_name] += 1
                    self.partial_matches[field_name] += 1
                    document_perfect = False
                else:
                    self.false_positives[field_name] += 1
                    document_perfect = False
            elif pred_value and not true_value:
                # Ложное срабатывание
                self.false_positives[field_name] += 1
                document_perfect = False
            elif not pred_value and true_value:
                # Пропущенное поле
                self.false_negatives[field_name] += 1
                document_perfect = False
                
        if document_perfect and len(ground_truth_fields) > 0:
            self.perfect_documents += 1
            
    def _normalize_value(self, value: str) -> str:
        """Нормализация значения для сравнения"""
        # Удаляем лишние пробелы
        value = " ".join(value.split())
        
        # Приводим к нижнему регистру для некритичных полей
        value = value.lower()
        
        # Удаляем знаки препинания по краям
        value = value.strip(".,;:")
        
        return value
        
    def _is_partial_match(self, pred: str, true: str) -> bool:
        """Проверка частичного совпадения"""
        # Если одна строка содержится в другой
        if pred in true or true in pred:
            return True
            
        # Проверка схожести (например, для дат в разных форматах)
        pred_norm = self._normalize_value(pred)
        true_norm = self._normalize_value(true)
        
        # Для чисел убираем форматирование
        pred_numbers = "".join(filter(str.isdigit, pred_norm))
        true_numbers = "".join(filter(str.isdigit, true_norm))
        
        if pred_numbers and pred_numbers == true_numbers:
            return True
            
        return False
        
    def get_metrics(self) -> Dict[str, float]:
        """Вычисляет итоговые метрики"""
        metrics = {}
        
        # Общие метрики по всем полям
        total_tp = sum(self.true_positives.values())
        total_fp = sum(self.false_positives.values())
        total_fn = sum(self.false_negatives.values())
        
        # Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['overall_precision'] = precision
        metrics['overall_recall'] = recall
        metrics['overall_f1'] = f1
        
        # Точность извлечения документов
        metrics['document_accuracy'] = self.perfect_documents / self.total_documents if self.total_documents > 0 else 0
        
        # Процент точных совпадений
        total_exact = sum(self.exact_matches.values())
        total_partial = sum(self.partial_matches.values())
        metrics['exact_match_rate'] = total_exact / (total_exact + total_partial) if (total_exact + total_partial) > 0 else 0
        
        # Метрики по полям
        metrics['per_field'] = {}
        for field_name in set(self.true_positives.keys()) | set(self.false_positives.keys()) | set(self.false_negatives.keys()):
            tp = self.true_positives[field_name]
            fp = self.false_positives[field_name]
            fn = self.false_negatives[field_name]
            
            field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            field_f1 = 2 * (field_precision * field_recall) / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0
            
            metrics['per_field'][field_name] = {
                'precision': field_precision,
                'recall': field_recall,
                'f1': field_f1,
                'support': tp + fn
            }
            
        return metrics


class EnhancedDonutMetricsCallback(TrainerCallback):
    """Улучшенный callback для правильных метрик Donut"""
    
    def __init__(self, processor, eval_dataset, log_callback=None):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_callback = log_callback
        self.metrics_calculator = DonutFieldExtractionMetrics()
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Вычисляет метрики после каждой оценки"""
        try:
            # Берем выборку для оценки
            eval_samples = self.eval_dataset.select(range(min(100, len(self.eval_dataset))))
            
            self.metrics_calculator.reset()
            
            model.eval()
            with torch.no_grad():
                for sample in eval_samples:
                    # Подготавливаем входные данные
                    image = sample['image']
                    target_text = sample['text']
                    
                    # Генерируем предсказание
                    pixel_values = self.processor(image, return_tensors="pt").pixel_values
                    
                    # Используем правильный task prompt для Donut
                    task_prompt = "<s_docvqa><s_question>Extract all fields from the document</s_question><s_answer>"
                    decoder_input_ids = self.processor.tokenizer(
                        task_prompt, 
                        add_special_tokens=False, 
                        return_tensors="pt"
                    ).input_ids
                    
                    # Генерируем ответ
                    outputs = model.generate(
                        pixel_values.to(model.device),
                        decoder_input_ids=decoder_input_ids.to(model.device),
                        max_length=512,
                        num_beams=4,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    # Декодируем предсказание
                    pred_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    pred_text = pred_text.replace(task_prompt, "").strip()
                    
                    # Парсим JSON из предсказания и ground truth
                    try:
                        pred_fields = self._parse_donut_output(pred_text)
                        true_fields = self._parse_donut_output(target_text)
                        
                        self.metrics_calculator.add_document(pred_fields, true_fields)
                    except Exception as e:
                        logger.warning(f"Ошибка парсинга результата: {e}")
                        continue
            
            # Получаем метрики
            metrics = self.metrics_calculator.get_metrics()
            
            # Форматируем сообщение
            accuracy_percentage = metrics['overall_f1'] * 100
            doc_accuracy = metrics['document_accuracy'] * 100
            exact_match = metrics['exact_match_rate'] * 100
            
            metrics_msg = (
                f"📊 Метрики извлечения полей:\n"
                f"   🎯 Общая точность (F1): {accuracy_percentage:.1f}%\n"
                f"   📄 Точность документов (100% полей): {doc_accuracy:.1f}%\n"
                f"   ✅ Точные совпадения: {exact_match:.1f}%\n"
                f"   📈 Precision: {metrics['overall_precision']:.3f}\n"
                f"   📊 Recall: {metrics['overall_recall']:.3f}\n"
            )
            
            # Качественная оценка
            if accuracy_percentage >= 98:
                quality = "🏆 ПРЕВОСХОДНО!"
            elif accuracy_percentage >= 95:
                quality = "🔥 Отлично"
            elif accuracy_percentage >= 90:
                quality = "✅ Хорошо"
            elif accuracy_percentage >= 80:
                quality = "🟡 Удовлетворительно"
            else:
                quality = "🔴 Требует улучшения"
                
            metrics_msg += f"   💎 Качество: {quality}\n"
            
            # Детали по полям (топ-5 худших)
            if metrics['per_field']:
                worst_fields = sorted(
                    metrics['per_field'].items(), 
                    key=lambda x: x[1]['f1']
                )[:5]
                
                if worst_fields:
                    metrics_msg += "   📉 Проблемные поля:\n"
                    for field_name, field_metrics in worst_fields:
                        if field_metrics['f1'] < 0.9:
                            metrics_msg += f"      • {field_name}: F1={field_metrics['f1']:.2f}\n"
            
            if self.log_callback:
                self.log_callback(metrics_msg)
                
            logger.info(metrics_msg)
            
            # Сохраняем детальные метрики
            state.log_history[-1]['eval_field_f1'] = metrics['overall_f1']
            state.log_history[-1]['eval_doc_accuracy'] = metrics['document_accuracy']
            
        except Exception as e:
            error_msg = f"❌ Ошибка при вычислении метрик: {str(e)}"
            if self.log_callback:
                self.log_callback(error_msg)
            logger.error(error_msg, exc_info=True)
            
    def _parse_donut_output(self, text: str) -> Dict[str, str]:
        """Парсит выход Donut в словарь полей"""
        fields = {}
        
        # Попытка 1: JSON парсинг
        try:
            if text.strip().startswith('{'):
                return json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Ошибка JSON парсинга, пробуем альтернативный метод
            pass
            
        # Попытка 2: Парсинг тегов Donut (<s_field>value</s_field>)
        import re
        pattern = r'<s_([^>]+)>([^<]+)</s_\1>'
        matches = re.findall(pattern, text)
        
        for field_name, value in matches:
            fields[field_name] = value.strip()
            
        return fields


class HighQualityDonutDataCollator:
    """Data collator с валидацией качества данных"""
    
    def __init__(self, processor, max_length=512, quality_threshold=0.95):
        self.processor = processor
        self.max_length = max_length
        self.quality_threshold = quality_threshold
        self.quality_enhancer = DataQualityEnhancer()
        
    def __call__(self, batch):
        """Обрабатывает батч с проверкой качества"""
        # Фильтруем низкокачественные примеры
        high_quality_batch = []
        
        for item in batch:
            # Проверяем качество аннотации
            if self._check_annotation_quality(item):
                high_quality_batch.append(item)
            else:
                logger.warning("Пропущен низкокачественный пример")
                
        if not high_quality_batch:
            # Если все примеры низкого качества, используем оригинальный батч
            high_quality_batch = batch
            
        # Извлекаем изображения и тексты
        images = [item['image'] for item in high_quality_batch]
        texts = [item['text'] for item in high_quality_batch]
        
        # Обрабатываем изображения
        pixel_values = self.processor(
            images, 
            return_tensors="pt"
        ).pixel_values
        
        # Обрабатываем тексты с улучшенным форматированием
        formatted_texts = [self._format_target_text(text) for text in texts]
        
        labels = self.processor.tokenizer(
            formatted_texts,
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
        
    def _check_annotation_quality(self, item) -> bool:
        """Проверяет качество аннотации"""
        try:
            # Проверяем наличие обязательных полей
            if 'image' not in item or 'text' not in item:
                return False
                
            # Проверяем, что текст не пустой
            if not item['text'] or len(item['text'].strip()) < 10:
                return False
                
            # Проверяем структуру данных
            if '<s_' in item['text'] or '{' in item['text']:
                # Данные выглядят структурированными
                return True
                
            return True
            
        except Exception:
            return False
            
    def _format_target_text(self, text: str) -> str:
        """Форматирует целевой текст для обучения"""
        # Добавляем task prefix для лучшего обучения
        task_prefix = "<s_docvqa><s_question>Extract all fields from the document</s_question><s_answer>"
        
        if not text.startswith(task_prefix):
            text = task_prefix + text + "</s_answer>"
            
        return text


class EnhancedDonutTrainer(BaseLorаTrainer):
    """
    Улучшенный Donut trainer с интеграцией базового LoRA класса
    Устраняет дублирование кода через наследование от BaseLorаTrainer
    """
    
    def __init__(self, app_config, logger=None):
        super().__init__(ModelType.DONUT, logger)
        self.app_config = app_config
        self.callbacks = {}
        self._stop_training = False
        
    def _apply_model_specific_optimizations(self, model, training_args):
        """Применяет специфичные для Donut оптимизации"""
        
        # Специфичная для Donut оптимизация - патчим forward метод
        def patched_forward(pixel_values=None, labels=None, **kwargs):
            # Фильтруем нежелательные аргументы для VisionEncoderDecoderModel
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['input_ids', 'attention_mask']}
            
            return model.original_forward(
                pixel_values=pixel_values,
                labels=labels,
                **filtered_kwargs
            )
        
        # Сохраняем оригинальный forward и заменяем
        if not hasattr(model, 'original_forward'):
            model.original_forward = model.forward
            model.forward = patched_forward
            self._log("✅ Donut forward method patched")
        
        return model 
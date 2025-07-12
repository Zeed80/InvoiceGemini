import logging
import os
import json
import sys
import torch
from datasets import Dataset, DatasetDict, load_from_disk, ClassLabel
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    AutoConfig,
    LayoutLMv3Processor
)
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Предполгается, что seqeval установлен (pip install seqeval)
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time

# Импорт для обучения Donut
from .enhanced_donut_trainer import EnhancedDonutTrainer
from .donut_model_tester import DonutModelTester

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Класс для обучения моделей LayoutLM"""
    
    def __init__(self, app_config):
        """
        Инициализация тренера моделей
        
        Args:
            app_config: Конфигурация приложения
        """
        self.app_config = app_config
        
        # Создаем логгер
        self.logger = logging.getLogger("ModelTrainer")
        
        # Определяем устройство для обучения
        from app.settings_manager import settings_manager
        
        # Проверяем настройки устройства
        use_gpu_if_available = settings_manager.get_use_gpu_if_available()
        selected_device = settings_manager.get_training_device()
        
        # Проверяем доступность CUDA
        import torch
        cuda_available = torch.cuda.is_available()
        
        # Определяем фактическое устройство для использования
        if selected_device == "cuda" and cuda_available:
            self.device = torch.device("cuda")
            self._log(f"ModelTrainer будет использовать устройство: cuda")
            
            # Ограничение памяти GPU, если задано
            memory_limit = settings_manager.get_max_gpu_memory()
            if memory_limit > 0:
                # Устанавливаем лимит памяти CUDA
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(memory_limit / 100, i)
                self._log(f"Установлено ограничение памяти CUDA: {memory_limit} МБ")
                
            # Определяем стратегию для нескольких GPU
            multi_gpu_strategy = settings_manager.get_multi_gpu_strategy()
            if multi_gpu_strategy != "none" and torch.cuda.device_count() > 1:
                self._log(f"Обнаружено несколько GPU ({torch.cuda.device_count()}), используется стратегия: {multi_gpu_strategy}")
                # Здесь будет код для разных стратегий мульти-GPU
        elif use_gpu_if_available and cuda_available:
            self.device = torch.device("cuda")
            self._log(f"ModelTrainer будет использовать устройство: cuda")
        else:
            self.device = torch.device("cpu")
            self._log(f"ModelTrainer будет использовать устройство: cpu")
        
        # Флаг для остановки обучения
        self.stop_requested = False
        self.progress_callback = None
        self.log_callback = None
        
    def _log(self, message):
        """
        Логирование сообщений обучения
        
        Args:
            message: Текст сообщения
        """
        # Использование логгера
        self.logger.info(message)
        
        # Вывод через callback, если он задан
        if hasattr(self, 'log_callback') and self.log_callback is not None:
            self.log_callback(message)
            
    def set_callbacks(self, log_callback=None, progress_callback=None):
        """
        Устанавливает функции обратного вызова для логирования и прогресса
        
        Args:
            log_callback: Функция для логирования сообщений
            progress_callback: Функция для отслеживания прогресса
        """
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def stop(self):
        """
        Отправляет сигнал для остановки обучения
        """
        self.stop_requested = True
        self._log("Получен сигнал для остановки обучения")
    
    def setup_distributed(self, local_rank):
        """
        Настраивает распределенное обучение
        
        Args:
            local_rank: Локальный ранг для распределенного обучения
        """
        dist.init_process_group(backend='nccl')
        self._log(f"Инициализировано распределенное обучение на ранге {local_rank}")

    def _create_training_args(self, base_args, output_dir):
        """
        Создает объект TrainingArguments с заданными параметрами
        
        Args:
            base_args: Базовые аргументы для обучения
            output_dir: Директория для сохранения модели
            
        Returns:
            TrainingArguments: Объект с аргументами для обучения
        """
        from transformers import TrainingArguments
        
        # Базовые аргументы, всегда присутствующие
        args = {
            "output_dir": output_dir,
            "logging_steps": 10,
            "save_total_limit": 2,  # Сохраняем только 2 последние чекпоинта
            "report_to": "none",  # Отключаем отчеты (можно включить "wandb" и т.д.)
        }
        
        # Проверяем и заменяем устаревшие параметры на новые
        if "evaluation_strategy" in base_args:
            args["eval_strategy"] = base_args.pop("evaluation_strategy")
            self.logger.info("Заменен устаревший параметр 'evaluation_strategy' на 'eval_strategy'")
            
        if "save_strategy" in base_args:
            args["save_strategy"] = base_args.pop("save_strategy")
            
        if "logging_strategy" in base_args:
            args["logging_strategy"] = base_args.pop("logging_strategy")
            
        # Обрабатываем другие потенциально устаревшие параметры
        if "metric_for_best_model" in base_args:
            # Проверим, не переименован ли этот параметр в новых версиях
            # Пока оставляем как есть, но логируем для отслеживания
            self.logger.info("Используется параметр 'metric_for_best_model', который может быть устаревшим")
            
        if "load_best_model_at_end" in base_args:
            # Проверим, не переименован ли этот параметр в новых версиях
            # Пока оставляем как есть, но логируем для отслеживания
            self.logger.info("Используется параметр 'load_best_model_at_end', который может быть устаревшим")
            
        # Добавляем пользовательские аргументы
        args.update(base_args)
        
        # Создаем объект TrainingArguments
        training_args = TrainingArguments(**args)
        
        return training_args
    
    def _setup_model_for_device(self, model):
        """
        Настраивает модель для выбранного устройства (CPU или GPU)
        
        Args:
            model: Модель PyTorch
            
        Returns:
            Модель, размещенная на выбранном устройстве
        """
        if self.device == "cuda":
            # Если мульти-GPU и выбрана стратегия data_parallel
            if self.is_multi_gpu and self.multi_gpu_strategy == "data_parallel":
                self._log("Настройка модели для DataParallel")
                model = torch.nn.DataParallel(model)
            
            # Размещаем модель на GPU
            model = model.to(self.device)
            self._log(f"Модель перемещена на {self.device}")
            
            # Информация о использовании памяти GPU
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                self._log(f"Использование памяти GPU: выделено {allocated:.2f} MB, зарезервировано {reserved:.2f} MB")
        else:
            model = model.to(self.device)
            self._log(f"Модель перемещена на {self.device}")
        
        return model
    
    # Метод для перемещения тензоров на нужное устройство
    def _to_device(self, data):
        """
        Перемещает данные (тензоры или словари с тензорами) на выбранное устройство
        
        Args:
            data: Данные для перемещения
            
        Returns:
            Данные на выбранном устройстве
        """
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data
    
    def _compute_metrics(self, eval_pred):
        """
        Вычисляет метрики для оценки модели.
        
        Args:
            eval_pred: Кортеж (predictions, labels)
            
        Returns:
            dict: Метрики
        """
        try:
            # Проверяем, инициализированы ли словари
            if not hasattr(self, 'id2label') or not self.id2label:
                # Если не инициализированы, создаем базовые словари
                self._log("ПРЕДУПРЕЖДЕНИЕ: id2label не инициализирован при вычислении метрик")
                if hasattr(self, 'label_list') and self.label_list:
                    self.id2label = {i: label for i, label in enumerate(self.label_list)}
                else:
                    # Если нет даже label_list, используем заглушку
                    self._log("ОШИБКА: Отсутствует label_list для создания id2label")
                    dummy_labels = ["O", "B-LABEL", "I-LABEL"]
                    self.id2label = {i: label for i, label in enumerate(dummy_labels)}
            
            # Распаковываем предсказания и истинные метки
            predictions, labels = eval_pred

            # Проверяем, что у нас есть валидные предсказания и метки
            if predictions is None or len(predictions) == 0:
                self._log("ОШИБКА: Отсутствуют предсказания для вычисления метрик")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                
            if labels is None or len(labels) == 0:
                self._log("ОШИБКА: Отсутствуют истинные метки для вычисления метрик")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # Проверяем размерности предсказаний и меток
            if predictions.shape[0] != labels.shape[0] or predictions.shape[1] != labels.shape[1]:
                self._log(f"ВНИМАНИЕ: Размеры предсказаний и меток не совпадают! predictions: {predictions.shape}, labels: {labels.shape}")
                # Используем минимальные размеры
                min_batch = min(predictions.shape[0], labels.shape[0])
                min_seq = min(predictions.shape[1], labels.shape[1])
                predictions = predictions[:min_batch, :min_seq]
                labels = labels[:min_batch, :min_seq]
            
            # Проверяем диапазон меток перед вычислением метрик
            max_label = np.max(labels[labels != -100]) if np.any(labels != -100) else 0
            max_id = max(self.id2label.keys()) if self.id2label else 0
            
            if max_label > max_id:
                self._log(f"ВНИМАНИЕ: Найдены метки вне диапазона словаря id2label! max_label={max_label}, max_id={max_id}")
                # Исправляем метки, выходящие за диапазон
                labels = np.where(labels > max_id, max_id, labels)
                
            # Выравниваем предсказания и метки
            true_predictions, true_labels = self._align_predictions(
                predictions=predictions, 
                label_ids=labels,
                id2label=self.id2label
            )
            
            # Проверяем результаты выравнивания
            if not true_predictions or not true_labels:
                self._log("ПРЕДУПРЕЖДЕНИЕ: Нет действительных предсказаний после выравнивания")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                
            if len(true_predictions) != len(true_labels):
                self._log(f"ВНИМАНИЕ: Размеры списков предсказаний и меток не совпадают! preds: {len(true_predictions)}, labels: {len(true_labels)}")
                # Используем минимальную длину
                min_len = min(len(true_predictions), len(true_labels))
                true_predictions = true_predictions[:min_len]
                true_labels = true_labels[:min_len]
            
            # Проверяем, что все списки в true_predictions и true_labels не пустые
            valid_indices = []
            for i, (preds, labels) in enumerate(zip(true_predictions, true_labels)):
                if preds and labels:
                    valid_indices.append(i)
            
            if not valid_indices:
                self._log("ПРЕДУПРЕЖДЕНИЕ: После фильтрации пустых списков нет действительных предсказаний")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # Отбираем только валидные списки
            true_predictions = [true_predictions[i] for i in valid_indices]
            true_labels = [true_labels[i] for i in valid_indices]
            
            # Проверяем длины вложенных списков
            for i, (preds, labels) in enumerate(zip(true_predictions, true_labels)):
                if len(preds) != len(labels):
                    self._log(f"ВНИМАНИЕ: Для примера {i} длины списков предсказаний и меток не совпадают! preds: {len(preds)}, labels: {len(labels)}")
                    # Используем минимальную длину
                    min_len = min(len(preds), len(labels))
                    true_predictions[i] = preds[:min_len]
                    true_labels[i] = labels[:min_len]
            
            # Вычисляем метрики с использованием seqeval
            try:
                from seqeval.metrics import precision_score, recall_score, f1_score
                
                # Проверяем, содержат ли предсказания и метки только строковые значения
                for i, (preds, labels) in enumerate(zip(true_predictions, true_labels)):
                    if not all(isinstance(p, str) for p in preds):
                        self._log(f"ПРЕДУПРЕЖДЕНИЕ: Предсказания {i} содержат нестроковые значения")
                        true_predictions[i] = [str(p) for p in preds]
                    if not all(isinstance(l, str) for l in labels):
                        self._log(f"ПРЕДУПРЕЖДЕНИЕ: Метки {i} содержат нестроковые значения")
                        true_labels[i] = [str(l) for l in labels]
                
                # Вычисляем метрики
                try:
                    precision = precision_score(true_labels, true_predictions)
                    recall = recall_score(true_labels, true_predictions)
                    f1 = f1_score(true_labels, true_predictions)
                except Exception as e:
                    self._log(f"ОШИБКА при вычислении общих метрик: {str(e)}")
                    # Используем значения по умолчанию
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                self._log(f"Вычисленные метрики: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
                
                # Вычисляем метрики для каждого класса
                class_wise_metrics = {}
                try:
                    # Попытка вычислить метрики по классам без вызова исключений
                    from seqeval.metrics import classification_report
                    report = classification_report(true_labels, true_predictions, output_dict=True)
                    
                    # Добавляем агрегированные метрики
                    for key in ['macro avg', 'weighted avg']:
                        if key in report:
                            for metric in ['precision', 'recall', 'f1-score']:
                                if metric in report[key]:
                                    class_wise_metrics[f"{key}_{metric}"] = report[key][metric]
                    
                    # Добавляем метрики по отдельным классам
                    for entity, metrics_dict in report.items():
                        if entity not in ['macro avg', 'weighted avg', 'micro avg']:
                            class_wise_metrics[f"f1_{entity}"] = metrics_dict.get('f1-score', 0.0)
                            
                except Exception as e:
                    self._log(f"ОШИБКА при вычислении метрик по классам: {str(e)}")
                
                # Собираем все метрики в один словарь
                metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    **class_wise_metrics  # Добавляем метрики по классам
                }
                
                return metrics
            except ImportError:
                self._log("ПРЕДУПРЕЖДЕНИЕ: seqeval не установлен, вычисление метрик невозможно")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            except Exception as e:
                self._log(f"ОШИБКА при вычислении метрик: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                
        except Exception as e:
            self._log(f"Ошибка при вычислении метрик: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def _tokenize_and_align_labels(self, examples, tokenizer):
        """
        Токенизация текста и выравнивание меток с поддержкой длинных документов
        
        Args:
            examples: Примеры для токенизации
            tokenizer: Токенизатор
        
        Returns:
            Dict: Токенизированные данные
        """
        # Проверяем, что у нас есть все необходимые данные
        if "words" not in examples or "bboxes" not in examples:
            self.logger.error("Отсутствуют необходимые поля 'words' или 'bboxes' в примерах")
            raise ValueError("Отсутствуют необходимые поля в примерах")
        
        # Проверяем, что у нас есть метки (либо labels, либо ner_tags)
        label_field = None
        if "labels" in examples:
            label_field = "labels"
        elif "ner_tags" in examples:
            label_field = "ner_tags"
        else:
            self.logger.error("Отсутствуют поля меток 'labels' или 'ner_tags' в примерах")
            raise ValueError("Отсутствуют поля меток в примерах")
        
        # Токенизируем слова
        tokenized_inputs = tokenizer(
            examples["words"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Выравниваем метки с токенами
        labels = []
        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=None)):
            label_ids = []
            previous_word_idx = None
            
            # Для каждого токена определяем соответствующую метку
            for word_idx in word_ids:
                if word_idx is None:
                    # Для специальных токенов (CLS, SEP, PAD)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Для первого токена слова берем соответствующую метку
                    try:
                        label_ids.append(examples[label_field][i][word_idx])
                    except (IndexError, TypeError):
                        self.logger.warning(f"Проблема с индексами: i={i}, word_idx={word_idx}, len(examples[{label_field}])={len(examples[label_field])}")
                        label_ids.append(-100)
                else:
                    # Для продолжения слова используем специальную метку
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        
        # Обрабатываем bbox для каждого токена
        bboxes = []
        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=None)):
            bbox_sequence = []
            for word_idx in word_ids:
                if word_idx is None:
                    # Для специальных токенов используем нулевой bbox
                    bbox_sequence.append([0, 0, 0, 0])
                else:
                    try:
                        # Для обычных токенов берем соответствующий bbox
                        bbox_sequence.append(examples["bboxes"][i][word_idx])
                    except (IndexError, TypeError):
                        self.logger.warning(f"Проблема с индексами bbox: i={i}, word_idx={word_idx}, len(examples['bboxes'])={len(examples['bboxes'])}")
                        bbox_sequence.append([0, 0, 0, 0])
            
            bboxes.append(bbox_sequence)
        
        tokenized_inputs["bbox"] = bboxes
        
        return tokenized_inputs
    
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str, metrics: Dict = None) -> bool:
        """
        Сохраняет обученную модель, токенизатор и метрики.
        
        Args:
            model (PreTrainedModel): Обученная модель
            tokenizer (PreTrainedTokenizer): Токенизатор
            output_dir (str): Путь для сохранения
            metrics (Dict, optional): Метрики обучения для сохранения
            
        Returns:
            bool: True если сохранение успешно, False в случае ошибки
        """
        try:
            self._log(f"Сохранение модели в: {output_dir}")
            
            # Создаем директорию, если её нет
            os.makedirs(output_dir, exist_ok=True)
            
            # Сохраняем модель и токенизатор
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Сохраняем метки
            if self.label_list:
                with open(os.path.join(output_dir, "label_list.json"), "w", encoding="utf-8") as f:
                    json.dump(self.label_list, f, ensure_ascii=False, indent=2)
            
            # Сохраняем метрики
            if metrics:
                with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            self._log("Модель успешно сохранена")
            return True
            
        except Exception as e:
            self._log(f"ОШИБКА при сохранении модели: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return False
    
    def load_model(self, model_dir: str) -> tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer], Optional[List[str]]]:
        """
        Загружает сохраненную модель, токенизатор и список меток.
        
        Args:
            model_dir (str): Путь к сохраненной модели
            
        Returns:
            tuple: (model, tokenizer, label_list) или (None, None, None) в случае ошибки
        """
        try:
            self._log(f"Загрузка модели из: {model_dir}")
            
            # Проверяем наличие необходимых файлов
            if not os.path.exists(model_dir):
                self._log(f"Директория модели не существует: {model_dir}")
                return None, None, None
            
            # Загружаем список меток
            label_list_path = os.path.join(model_dir, "label_list.json")
            if os.path.exists(label_list_path):
                with open(label_list_path, "r", encoding="utf-8") as f:
                    self.label_list = json.load(f)
                self._log(f"Загружен список меток: {self.label_list}")
            else:
                self._log("ПРЕДУПРЕЖДЕНИЕ: Файл с метками не найден")
                return None, None, None
            
            # Загружаем модель и токенизатор
            model = AutoModelForTokenClassification.from_pretrained(
                model_dir,
                num_labels=len(self.label_list)
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Переносим модель на нужное устройство
            model = self._setup_model_for_device(model)
            
            self._log("Модель успешно загружена")
            return model, tokenizer, self.label_list
            
        except Exception as e:
            self._log(f"ОШИБКА при загрузке модели: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None, None, None
    
    def evaluate_model(self, model: PreTrainedModel, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Оценивает качество модели на валидационном датасете.
        
        Args:
            model (PreTrainedModel): Модель для оценки
            eval_dataset (Dataset): Валидационный датасет
            
        Returns:
            Dict[str, float]: Метрики качества или пустой словарь в случае ошибки
        """
        try:
            self._log("Оценка качества модели...")
            
            # Создаем Trainer для оценки
            trainer = Trainer(
                model=model,
                compute_metrics=self._compute_metrics,
                eval_dataset=eval_dataset
            )
            
            # Запускаем оценку
            metrics = trainer.evaluate()
            
            self._log(f"Результаты оценки: {metrics}")
            return metrics
            
        except Exception as e:
            self._log(f"ОШИБКА при оценке модели: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return {}
            
    def _normalize_bbox(self, bbox, width, height):
        """
        Нормализует координаты ограничивающего прямоугольника (bbox) в формат, требуемый LayoutLM (0-1000).
        
        Args:
            bbox (list): Координаты [x1, y1, x2, y2] в пикселях
            width (int): Ширина изображения в пикселях
            height (int): Высота изображения в пикселях
            
        Returns:
            list: Нормализованные координаты [x1, y1, x2, y2] в диапазоне 0-1000
        """
        # Сначала проверяем, что у нас валидный bbox
        if not bbox or len(bbox) != 4:
            self._log(f"ВНИМАНИЕ: Некорректный bbox {bbox}. Используем значение по умолчанию [0, 0, 0, 0].")
            return [0, 0, 0, 0]
            
        try:
            # Преобразуем все значения в int
            bbox = [int(coord) if coord is not None else 0 for coord in bbox]
            
            # Проверяем, что координаты не отрицательные
            bbox = [max(0, coord) for coord in bbox]
            
            # Проверяем, что x2 > x1 и y2 > y1
            if bbox[2] <= bbox[0]:
                bbox[2] = bbox[0] + 1
            if bbox[3] <= bbox[1]:
                bbox[3] = bbox[1] + 1
                
            # Проверяем, что координаты не превышают размеры изображения
            bbox[0] = min(bbox[0], width)
            bbox[2] = min(bbox[2], width)
            bbox[1] = min(bbox[1], height)
            bbox[3] = min(bbox[3], height)
            
            # Нормализуем в диапазон 0-1000
            normalized = [
                int(1000 * (bbox[0] / width)) if width > 0 else 0,
                int(1000 * (bbox[1] / height)) if height > 0 else 0,
                int(1000 * (bbox[2] / width)) if width > 0 else 0,
                int(1000 * (bbox[3] / height)) if height > 0 else 0,
            ]
            
            # Проверяем, что все значения в диапазоне 0-1000
            normalized = [max(0, min(1000, coord)) for coord in normalized]
            
            return normalized
        except Exception as e:
            self._log(f"Ошибка при нормализации bbox: {str(e)}. Используем значение по умолчанию [0, 0, 0, 0].")
            return [0, 0, 0, 0]
            
    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Подготавливает датасет для обучения.
        
        Args:
            dataset: Исходный датасет
            
        Returns:
            Dataset: Подготовленный датасет
        """
        try:
            # Проверяем, что label_list инициализирован
            if not hasattr(self, 'label_list') or not self.label_list:
                self._log("ОШИБКА: label_list не инициализирован")
                self.label_list = self._get_label_list(dataset)
                self._log(f"Автоматически определены метки: {self.label_list}")
                
            # Создаем словари для преобразования меток
            if not hasattr(self, 'label2id') or not self.label2id:
                self.label2id = {label: i for i, label in enumerate(self.label_list)}
            if not hasattr(self, 'id2label') or not self.id2label:
                self.id2label = {i: label for i, label in enumerate(self.label_list)}
            
            self._log(f"Используем label2id: {self.label2id}")
            self._log(f"Используем id2label: {self.id2label}")
            
            # Проверяем наличие необходимых колонок в датасете
            required_columns = ['image_path', 'words', 'bboxes', 'labels']
            missing_columns = [col for col in required_columns if col not in dataset.column_names]
            if missing_columns:
                self._log(f"ОШИБКА: В датасете отсутствуют колонки: {missing_columns}")
                return None
            
            # Создаем функцию для подготовки примера
            def prepare_example(examples):
                try:
                    # Проверяем и загружаем изображения
                    images = []
                    image_sizes = []  # Сохраняем размеры изображений
                    
                    for path in examples["image_path"]:
                        try:
                            img = Image.open(path).convert("RGB")
                            images.append(img)
                            image_sizes.append(img.size)  # (width, height)
                        except Exception as e:
                            self._log(f"ОШИБКА загрузки изображения {path}: {str(e)}")
                            # Если не удалось загрузить, используем пустое изображение
                            dummy_img = Image.new('RGB', (100, 100), color='white')
                            images.append(dummy_img)
                            image_sizes.append((100, 100))
                    
                    # Преобразуем строковые метки в числовые ID
                    word_labels = []
                    for example_labels in examples["labels"]:
                        # Анализируем текущий тип меток
                        if all(isinstance(label, int) or (hasattr(label, 'dtype') and 'int' in str(label.dtype)) for label in example_labels):
                            # Если все метки уже числовые, просто конвертируем в int
                            example_id_labels = [int(label) if label != -100 else -100 for label in example_labels]
                        else:
                            # Если метки строковые, преобразуем в числовые ID
                            example_id_labels = []
                            for label in example_labels:
                                if label == "-100" or label == -100:
                                    example_id_labels.append(-100)
                                else:
                                    # Получаем ID метки из словаря label2id или используем 0 (обычно 'O')
                                    label_id = self.label2id.get(label, 0)
                                    example_id_labels.append(label_id)
                        
                        # Проверяем, что все метки в допустимом диапазоне
                        num_labels = len(self.id2label)
                        example_id_labels = [
                            -100 if label == -100 else max(0, min(num_labels-1, label))
                            for label in example_id_labels
                        ]
                        
                        word_labels.append(example_id_labels)
                    
                    # Нормализуем bboxes для обеспечения диапазона 0-1000
                    normalized_bboxes = []
                    for i, bbox_list in enumerate(examples["bboxes"]):
                        if i < len(images):  # проверяем, что у нас есть соответствующее изображение
                            width, height = image_sizes[i]
                            
                            # Проверяем и исправляем структуру bbox_list
                            valid_bbox_list = []
                            for bbox in bbox_list:
                                # Проверяем, что bbox имеет правильную структуру
                                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                                    self._log(f"ВНИМАНИЕ: Некорректный bbox {bbox}. Используем значение по умолчанию [0, 0, 0, 0].")
                                    valid_bbox_list.append([0, 0, 0, 0])
                                else:
                                    # Нормализуем bbox
                                    valid_bbox_list.append(self._normalize_bbox(bbox, width, height))
                            
                            normalized_bboxes.append(valid_bbox_list)
                        else:
                            # Если нет соответствующего изображения, используем нулевые bbox
                            normalized_bboxes.append([[0, 0, 0, 0] for _ in range(len(bbox_list))])
                    
                    # Проверяем, что длины слов, боксов и меток совпадают
                    for i in range(len(examples["words"])):
                        words_len = len(examples["words"][i])
                        bboxes_len = len(normalized_bboxes[i]) if i < len(normalized_bboxes) else 0
                        labels_len = len(word_labels[i]) if i < len(word_labels) else 0
                        
                        if not (words_len == bboxes_len == labels_len):
                            self._log(f"ВНИМАНИЕ: Длины слов ({words_len}), боксов ({bboxes_len}) и меток ({labels_len}) не совпадают для примера {i}")
                            
                            # Выравниваем длины до минимальной
                            min_len = min(words_len, bboxes_len, labels_len)
                            
                            if i < len(examples["words"]):
                                examples["words"][i] = examples["words"][i][:min_len]
                            if i < len(normalized_bboxes):
                                normalized_bboxes[i] = normalized_bboxes[i][:min_len]
                            if i < len(word_labels):
                                word_labels[i] = word_labels[i][:min_len]
                    
                    # Логируем информацию о диапазоне нормализованных bbox для проверки
                    if normalized_bboxes:
                        flat_bboxes = [bbox for sublist in normalized_bboxes for bbox in sublist]
                        if flat_bboxes:
                            min_x = min([bbox[0] for bbox in flat_bboxes])
                            max_x = max([bbox[2] for bbox in flat_bboxes])
                            min_y = min([bbox[1] for bbox in flat_bboxes])
                            max_y = max([bbox[3] for bbox in flat_bboxes])
                            self._log(f"Диапазон нормализованных bbox: X=[{min_x},{max_x}], Y=[{min_y},{max_y}]")
                    
                    # Подготавливаем входные данные
                    encoding = self.processor(
                        images,
                        examples["words"],
                        boxes=normalized_bboxes,  # Используем нормализованные bbox
                        word_labels=word_labels,  # Используем преобразованные числовые ID
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    return encoding
                except Exception as e:
                    self._log(f"ОШИБКА при подготовке примера: {str(e)}")
                    import traceback
                    self._log(traceback.format_exc())
                    # Возвращаем пустой словарь, который будет обработан map
                    return {}
                
            # Применяем подготовку ко всему датасету
            prepared_dataset = dataset.map(
                prepare_example,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            
            # Проверяем, что датасет не пустой
            if len(prepared_dataset) == 0:
                self._log("ОШИБКА: После подготовки датасет пуст")
                return None
            
            # Сохраняем label_list в атрибуте датасета для последующего использования
            prepared_dataset.label_list = self.label_list
            
            return prepared_dataset
            
        except Exception as e:
            self._log(f"Ошибка при подготовке датасета: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None
            
    def _get_label_list(self, dataset: Dataset) -> List[str]:
        """
        Получает список уникальных меток из датасета.
        
        Args:
            dataset: Исходный датасет
            
        Returns:
            List[str]: Список уникальных меток
        """
        try:
            self._log("Определение уникальных меток из датасета")
            
            # Проверяем наличие колонки с метками
            if "labels" not in dataset.column_names:
                self._log("ОШИБКА: В датасете отсутствует колонка 'labels'")
                return ["O", "B-ENTITY", "I-ENTITY"]  # Возвращаем базовый набор меток
            
            # Собираем все уникальные метки из датасета
            unique_labels = set()
            
            # Обрабатываем различные типы данных в колонке 'labels'
            for example_labels in dataset["labels"]:
                for label in example_labels:
                    # Пропускаем специальные токены и пустые значения
                    if label == -100 or label == "-100" or not label:
                        continue
                        
                    # Если метка числовая, конвертируем в строку для сохранения
                    if isinstance(label, int) or (hasattr(label, 'dtype') and 'int' in str(label.dtype)):
                        # Это уже ID, пропускаем
                        continue
                    
                    # Добавляем метку в множество уникальных меток
                    if label.strip():
                        unique_labels.add(label.strip())
            
            # Если уникальных меток нет, возвращаем базовый набор
            if not unique_labels:
                self._log("ПРЕДУПРЕЖДЕНИЕ: Не удалось найти уникальные метки в датасете")
                return ["O", "B-ENTITY", "I-ENTITY"]
            
            # Сортируем и возвращаем список меток
            return sorted(list(unique_labels))
            
        except Exception as e:
            self._log(f"Ошибка при получении списка меток: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return ["O", "B-ENTITY", "I-ENTITY"]  # Возвращаем базовый набор меток при ошибке

    def _align_predictions(self, predictions, label_ids, id2label):
        """
        Выравнивает предсказания и истинные метки, преобразуя числовые ID в строковые метки.
        
        Args:
            predictions: Предсказания модели (логиты)
            label_ids: Числовые ID истинных меток
            id2label: Словарь для преобразования ID в метки
            
        Returns:
            tuple: (предсказанные метки, истинные метки)
        """
        try:
            if predictions is None or len(predictions) == 0:
                self._log("Предсказания пустые, невозможно выровнять")
                return [["O"]], [["O"]]
                
            # Получаем индексы максимальных значений логитов
            preds_argmax = np.argmax(predictions, axis=2)  # (batch_size, seq_len)
            
            # Проверяем размеры массивов
            if preds_argmax.shape[0] != label_ids.shape[0] or preds_argmax.shape[1] != label_ids.shape[1]:
                self._log(f"ВНИМАНИЕ: Размеры массивов не совпадают! preds: {preds_argmax.shape}, labels: {label_ids.shape}")
                # Обрезаем до минимального размера для предотвращения ошибок
                min_batch = min(preds_argmax.shape[0], label_ids.shape[0])
                min_seq = min(preds_argmax.shape[1], label_ids.shape[1])
                preds_argmax = preds_argmax[:min_batch, :min_seq]
                label_ids = label_ids[:min_batch, :min_seq]
            
            batch_size, seq_len = preds_argmax.shape
            
            out_pred_list = [[] for _ in range(batch_size)]
            out_label_list = [[] for _ in range(batch_size)]

            # Логируем размерности для отладки
            self._log(f"Размерности: preds_argmax={preds_argmax.shape}, label_ids={label_ids.shape}")
            
            # Найдем максимальные значения предсказаний и меток для проверки диапазона
            max_pred_id = np.max(preds_argmax)
            max_label_id = np.max(label_ids[label_ids != -100]) if np.any(label_ids != -100) else 0
            max_id = max(max_pred_id, max_label_id)
            
            self._log(f"Максимальный ID предсказаний: {max_pred_id}, максимальный ID меток: {max_label_id}")
            
            # Подготовим словарь id2label, убедившись, что все необходимые ключи присутствуют
            max_expected_id = max(id2label.keys()) if id2label else -1
            
            # Если словарь неполный, расширим его
            if max_id > max_expected_id:
                self._log(f"Расширяем словарь id2label: текущий макс ID = {max_expected_id}, требуемый макс ID = {max_id}")
                for i in range(max_expected_id + 1, max_id + 1):
                    id2label[i] = f"UNK_{i}"
            
            for i in range(batch_size):
                for j in range(seq_len):
                    # Обрабатываем только непадинговые токены
                    if label_ids[i, j] != -100:
                        # Безопасно получаем индексы предсказаний и истинных меток
                        pred_id = int(preds_argmax[i, j])
                        label_id = int(label_ids[i, j])
                        
                        # Проверяем, что индексы находятся в допустимом диапазоне
                        if pred_id >= len(id2label):
                            self._log(f"ВНИМАНИЕ: Индекс предсказания {pred_id} выходит за пределы словаря id2label")
                            # Используем последнюю допустимую метку или "UNK"
                            pred_id = min(pred_id, max(id2label.keys()))
                        
                        if label_id >= len(id2label):
                            self._log(f"ВНИМАНИЕ: Индекс истинной метки {label_id} выходит за пределы словаря id2label")
                            # Используем последнюю допустимую метку или "UNK"
                            label_id = min(label_id, max(id2label.keys()))
                        
                        # Получаем строковые метки из словаря id2label
                        pred_label = id2label.get(pred_id, f"UNK_{pred_id}")
                        true_label = id2label.get(label_id, f"UNK_{label_id}")
                        
                        out_pred_list[i].append(pred_label)
                        out_label_list[i].append(true_label)
            
            # Проверяем, что все списки не пустые
            out_pred_list = [preds for preds in out_pred_list if preds]
            out_label_list = [labels for labels in out_label_list if labels]
            
            if not out_pred_list or not out_label_list:
                self._log("ПРЕДУПРЕЖДЕНИЕ: После выравнивания получены пустые списки меток")
                # Возвращаем минимальные непустые списки для предотвращения ошибок
                return [["O"]], [["O"]]
            
            # Убедимся, что все метки - строковые
            for i, preds in enumerate(out_pred_list):
                out_pred_list[i] = [str(p) for p in preds]
                
            for i, labels in enumerate(out_label_list):
                out_label_list[i] = [str(l) for l in labels]
            
            return out_pred_list, out_label_list
            
        except Exception as e:
            self._log(f"ОШИБКА при выравнивании предсказаний: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            # Возвращаем минимальные непустые списки для предотвращения ошибок
            return [["O"]], [["O"]]

    def _collate_fn(self, features):
        """
        Функция для объединения батчей с поддержкой распределенного обучения
        
        Args:
            features: Список примеров для объединения
            
        Returns:
            dict: Батч данных
        """
        batch = {}
        
        # Собираем все ключи из первого примера
        first = features[0]
        for key in first.keys():
            if key not in ['labels', 'pixel_values']:
                batch[key] = torch.stack([f[key] for f in features])
            elif key == 'labels':
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            else:  # pixel_values
                batch[key] = torch.stack([f[key] for f in features])
                
        return batch

    def train_layoutlm(self, 
                    dataset_path: str,
                    base_model_id: str,
                    training_args: dict,
                    output_model_name: str,
                    local_rank: int = -1,
                    output_model_path: str = None) -> Optional[str]:
        """
        Обучение модели LayoutLM на указанном датасете.
        
        Args:
            dataset_path: Путь к датасету
            base_model_id: ID базовой модели или путь к ней
            training_args: Параметры обучения
            output_model_name: Имя выходной модели
            local_rank: Ранг для распределенного обучения
            
        Returns:
            Путь к обученной модели или None в случае ошибки
        """
        import sys
        
        try:
            start_time = time.time()
            
            # Логируем начало процесса обучения
            self._log("=" * 80)
            self._log("🚀 НАЧАЛО ОБУЧЕНИЯ LAYOUTLM")
            self._log("=" * 80)
            self._log(f"📅 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"📊 Путь к датасету: {dataset_path}")
            self._log(f"🤖 ID базовой модели: {base_model_id}")
            self._log(f"💾 Имя выходной модели: {output_model_name}")
            self._log(f"🔧 Локальный ранг: {local_rank}")
            
            # Системная информация
            self._log("\n🖥️ СИСТЕМНАЯ ИНФОРМАЦИЯ:")
            self._log(f"  🔌 Устройство: {self.device}")
            self._log(f"  🐍 Python версия: {sys.version.split()[0]}")
            self._log(f"  🔥 PyTorch версия: {torch.__version__}")
            self._log(f"  🔥 CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self._log(f"  🎮 GPU устройство: {torch.cuda.get_device_name()}")
                self._log(f"  💾 GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                self._log(f"  💾 GPU свободна: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # Настраиваем устройство и распределенное обучение
            if local_rank != -1:
                self._log(f"\n🌐 Настройка распределенного обучения для ранга: {local_rank}")
                self.setup_distributed(local_rank)
                
            # Обновляем fp16 параметр в зависимости от доступности CUDA
            if "fp16" in training_args:
                training_args["fp16"] = torch.cuda.is_available()
                self._log(f"⚙️ Автоматически установлен параметр fp16={training_args['fp16']} на основе доступности CUDA")
                
            # Преобразуем report_to в список, если это строка
            if "report_to" in training_args and isinstance(training_args["report_to"], str):
                training_args["report_to"] = [training_args["report_to"]]
                self._log(f"⚙️ Преобразован параметр report_to в список: {training_args['report_to']}")
                
            # Заменяем evaluation_strategy на eval_strategy, если необходимо
            if "evaluation_strategy" in training_args:
                training_args["eval_strategy"] = training_args.pop("evaluation_strategy")
                self._log("⚙️ Заменен устаревший параметр 'evaluation_strategy' на 'eval_strategy'")
                
            # Проверяем, указан ли output_dir в training_args
            if "output_dir" not in training_args:
                # Используем относительный путь если указан
                if output_model_path:
                    # Если путь относительный, делаем его абсолютным
                    if not os.path.isabs(output_model_path):
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        output_dir = os.path.join(project_root, output_model_path)
                    else:
                        output_dir = output_model_path
                else:
                    # Используем старую логику для обратной совместимости
                    output_dir = os.path.join(self.app_config.TRAINED_MODELS_PATH, output_model_name)
                    
                training_args["output_dir"] = output_dir
                self._log(f"⚙️ Добавлен параметр output_dir: {output_dir}")
            
            # Подробное логирование параметров обучения
            self._log("\n📋 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
            for key, value in sorted(training_args.items()):
                self._log(f"  {key}: {value}")
            
            # Проверяем доступность ресурсов
            self._log("\n🔍 ПРОВЕРКА РЕСУРСОВ:")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                self._log(f"  💾 GPU память выделена: {memory_allocated:.2f} GB")
                self._log(f"  💾 GPU память закэширована: {memory_cached:.2f} GB")
            
            # Готовим директорию для выходной модели
            output_dir = training_args["output_dir"]
            if os.path.exists(output_dir):
                # Если директория существует, создаем новую с временной меткой
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"{output_dir}_{timestamp}"
                training_args["output_dir"] = output_dir
                self._log(f"Папка для модели уже существует. Новый путь: {output_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Загружаем датасет
            self._log("\n📊 ЭТАП 1: ЗАГРУЗКА ДАТАСЕТА")
            self._log("-" * 50)
            self._log(f"📁 Путь к датасету: {dataset_path}")
            self._log("🔄 Загрузка датасета...")
            
            from datasets import load_from_disk, DatasetDict
            
            try:
                # Пробуем загрузить датасет
                dataset = load_from_disk(dataset_path)
                self._log("✅ Датасет успешно загружен")
                
                # Подробная информация о датасете
                self._log("\n📈 СТАТИСТИКА ДАТАСЕТА:")
                for split_name, split_data in dataset.items():
                    self._log(f"  📄 {split_name}: {len(split_data)} примеров")
                
                # Информация о структуре
                if "train" in dataset:
                    train_features = list(dataset["train"].features.keys())
                    self._log(f"  🏗️ Поля датасета: {train_features}")
                    
                    # Показываем пример данных
                    if len(dataset["train"]) > 0:
                        example = dataset["train"][0]
                        self._log("  🔍 Пример данных:")
                        for key, value in example.items():
                            if key == "words" and isinstance(value, list):
                                preview = str(value[:3]) + "..." if len(value) > 3 else str(value)
                                self._log(f"    {key}: {preview} (всего: {len(value)})")
                            elif key == "bboxes" and isinstance(value, list):
                                self._log(f"    {key}: {len(value)} bounding boxes")
                            elif key == "labels" and isinstance(value, list):
                                self._log(f"    {key}: {len(value)} меток")
                            elif isinstance(value, (str, int, float)):
                                preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                self._log(f"    {key}: {preview}")
                
                # Проверяем структуру датасета и наличие необходимых полей
                required_features = ['words', 'bboxes', 'labels']
                
                # Проверяем наличие необходимых полей в датасете
                features_exist = all(feature in dataset["train"].features for feature in required_features)
                
                if not features_exist:
                    missing_features = [feature for feature in required_features if feature not in dataset["train"].features]
                    self._log(f"❌ В датасете отсутствуют необходимые поля: {missing_features}")
                    raise ValueError(f"В датасете отсутствуют необходимые поля: {missing_features}")
                else:
                    self._log(f"✅ Все необходимые поля присутствуют: {required_features}")
                    
                # Проверяем наличие поля ner_tags (используется для обучения LayoutLM)
                # Если его нет, попробуем конвертировать поле labels
                if 'ner_tags' not in dataset["train"].features:
                    self.logger.info("Поле 'ner_tags' отсутствует в датасете. Конвертируем поле 'labels' в 'ner_tags'...")
                    
                    # Функция для преобразования строковых меток в числовые
                    def convert_labels_to_ner_tags(examples):
                        # Собираем все уникальные метки
                        all_labels = []
                        for split in dataset.keys():
                            for labels_list in dataset[split]["labels"]:
                                # Проверяем, что labels_list - это список
                                if isinstance(labels_list, list):
                                    all_labels.append(labels_list)
                                else:
                                    all_labels.append([labels_list])
                        
                        # Сглаживаем список и удаляем дубликаты
                        all_labels_flat = []
                        for labels_list in all_labels:
                            for label in labels_list:
                                # Проверяем и пропускаем None и другие недопустимые значения
                                if label is not None and label != -100:
                                    all_labels_flat.append(label)
                        
                        # Преобразуем строковые метки в хешируемый тип (tuple)
                        try:
                            # Пытаемся преобразовать все метки в строки
                            unique_labels = sorted(set([str(label) for label in all_labels_flat]))
                        except Exception as e:
                            self.logger.error(f"Ошибка при преобразовании меток в строки: {str(e)}")
                            # В случае ошибки используем простой набор меток
                            unique_labels = ["O", "B-ENTITY", "I-ENTITY"]
                        
                        # Создаем словарь для маппинга строковых меток в числовые
                        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                        self.logger.info(f"Создан маппинг меток: {label_to_id}")
                        
                        # Преобразуем строковые метки в числовые
                        examples["ner_tags"] = []
                        
                        for labels in examples["labels"]:
                            # Проверяем, что labels - это список
                            if not isinstance(labels, list):
                                labels = [labels]
                            
                            # Преобразуем каждую метку в числовой ID
                            numeric_tags = []
                            for label in labels:
                                # Проверяем специальные значения
                                if label is None or label == -100:
                                    numeric_tags.append(-100)
                                else:
                                    # Преобразуем метку в строку для использования в словаре
                                    label_str = str(label)
                                    # Получаем ID из словаря или используем 0 по умолчанию
                                    numeric_tags.append(label_to_id.get(label_str, 0))
                            
                            examples["ner_tags"].append(numeric_tags)
                        
                        return examples
                    
                    # Применяем функцию конвертации к каждому сплиту датасета
                    for split in dataset.keys():
                        dataset[split] = dataset[split].map(convert_labels_to_ner_tags)
                    
                    # Получаем уникальные ID для меток и создаем feature
                    from datasets import ClassLabel
                    
                    # Получаем уникальные метки из всех тегов ner_tags
                    all_ner_tags = []
                    for split in dataset.keys():
                        for tags in dataset[split]["ner_tags"]:
                            # Проверяем, что tags - это список
                            if isinstance(tags, list):
                                # Если это вложенный список, сначала выравниваем его
                                flat_tags = []
                                for item in tags:
                                    if isinstance(item, list):
                                        flat_tags.extend(item)
                                    else:
                                        flat_tags.append(item)
                                all_ner_tags.extend(flat_tags)
                            else:
                                # Если это не список, добавляем элемент напрямую
                                all_ner_tags.append(tags)
                    
                    # Находим максимальное значение для определения количества классов
                    if all_ner_tags:
                        # Преобразуем все элементы в числа, если они еще не являются числами
                        all_ner_tags_numeric = []
                        for tag in all_ner_tags:
                            if tag == -100:
                                # Пропускаем специальные метки -100
                                continue
                            elif isinstance(tag, str) and tag.isdigit():
                                all_ner_tags_numeric.append(int(tag))
                            elif isinstance(tag, (int, float)):
                                all_ner_tags_numeric.append(int(tag))
                            elif hasattr(tag, 'dtype') and 'int' in str(tag.dtype):  # numpy integer types
                                all_ner_tags_numeric.append(int(tag))
                        
                        if all_ner_tags_numeric:
                            # Получаем количество классов как максимальное значение + 1
                            num_labels = max(all_ner_tags_numeric) + 1
                            # Страховка: минимум 2 класса
                            num_labels = max(2, num_labels)
                            self.logger.info(f"Обнаружено {num_labels} уникальных меток (максимальный ID: {max(all_ner_tags_numeric)})")
                        else:
                            # Если после фильтрации список пуст, используем значение по умолчанию
                            self.logger.warning("Список числовых меток пуст! Устанавливаем количество меток по умолчанию: 12")
                            num_labels = 12
                    else:
                        # Если список пуст, устанавливаем значение по умолчанию
                        self.logger.warning("Список меток пуст! Устанавливаем количество меток по умолчанию: 12")
                        num_labels = 12
                    
                    self.logger.info(f"Обнаружено {num_labels} уникальных меток")
                    
                    # Исправляем проблему с преобразованием типов
                    # Сначала убедимся, что все ner_tags содержат только числа, а не вложенные списки
                    for split in dataset.keys():
                        new_ner_tags = []
                        for tags in dataset[split]["ner_tags"]:
                            if isinstance(tags, list):
                                # Выравниваем вложенные списки
                                flat_tags = []
                                for item in tags:
                                    if isinstance(item, list):
                                        # Берем первый элемент из вложенного списка или 0, если список пуст
                                        flat_tags.append(item[0] if item else 0)
                                    else:
                                        flat_tags.append(item)
                                new_ner_tags.append(flat_tags)
                            else:
                                new_ner_tags.append(tags)
                        
                        # Заменяем ner_tags в датасете
                        dataset[split] = dataset[split].remove_columns(["ner_tags"]) if "ner_tags" in dataset[split].column_names else dataset[split]
                        dataset[split] = dataset[split].add_column("ner_tags", new_ner_tags)
                    
                    # Создаем правильный ClassLabel для ner_tags
                    dataset_features = dataset["train"].features.copy()
                    
                    # Используем Sequence вместо ClassLabel для ner_tags, так как это список меток
                    from datasets import Sequence, ClassLabel
                    dataset_features["ner_tags"] = Sequence(ClassLabel(num_classes=num_labels, names=[str(i) for i in range(num_labels)]))
                    
                    # Применяем новые features к датасету
                    for split in dataset.keys():
                        dataset[split] = dataset[split].cast_column("ner_tags", dataset_features["ner_tags"])
                    
                    self.logger.info(f"Конвертация меток завершена. Создано {num_labels} уникальных меток.")
                
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке или подготовке датасета: {str(e)}")
                raise
                
            # 2. Загружаем токенизатор
            self.logger.info("Загрузка токенизатора...")
            
            from transformers import AutoTokenizer
            
            # Сохраняем изначальный ID модели для проверок
            original_base_model_id = base_model_id
            
            # Проверка на локальный путь к модели
            local_model_path = os.path.join(self.app_config.MODELS_PATH, 'layoutlm', base_model_id.split('/')[-1])
            
            try:
                # Сначала пробуем загрузить из локального кеша или указанного пути
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                self.logger.info(f"Токенизатор успешно загружен из {base_model_id}")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить токенизатор из {base_model_id}: {str(e)}")
                
                # Пробуем загрузить из локального пути
                if os.path.exists(local_model_path):
                    self.logger.info(f"Пробуем загрузить токенизатор из локального пути: {local_model_path}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                        self.logger.info(f"Токенизатор успешно загружен из локального пути")
                    except Exception as local_e:
                        self.logger.error(f"Не удалось загрузить токенизатор из локального пути: {str(local_e)}")
                        raise RuntimeError(f"Не удалось загрузить токенизатор ни из {base_model_id}, ни из локального пути {local_model_path}")
                else:
                    # Если локальный путь не существует, предлагаем использовать другую модель
                    self.logger.error(f"Локальный путь {local_model_path} не существует")
                    alternative_model = "microsoft/layoutlmv3-base"
                    self.logger.info(f"Пробуем использовать альтернативную модель: {alternative_model}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(alternative_model)
                        self.logger.info(f"Токенизатор успешно загружен из альтернативной модели: {alternative_model}")
                        # Обновляем базовую модель для дальнейшей загрузки
                        base_model_id = alternative_model
                    except Exception as alt_e:
                        self.logger.error(f"Не удалось загрузить токенизатор из альтернативной модели: {str(alt_e)}")
                        raise RuntimeError(f"Не удалось загрузить токенизатор из всех доступных источников")
            
            # 3. Загружаем модель
            self._log("\n🤖 ЭТАП 3: ЗАГРУЗКА МОДЕЛИ")
            self._log("-" * 50)
            self._log(f"📥 Загрузка базовой модели: {base_model_id}")
            
            from transformers import AutoModelForTokenClassification
            
            num_labels = len(dataset["train"].features["ner_tags"].feature.names)
            self._log(f"🏷️ Количество меток для классификации: {num_labels}")
            
            # Проверяем доступную память перед загрузкой
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3
                self._log(f"💾 Память GPU до загрузки модели: {memory_before:.2f} GB")
            
            try:
                # Сначала пробуем загрузить из указанного пути
                self._log(f"🔄 Попытка загрузки из: {base_model_id}")
                model = AutoModelForTokenClassification.from_pretrained(
                    base_model_id, 
                    num_labels=num_labels,
                    cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache'),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._log(f"✅ Модель успешно загружена из {base_model_id}")
            except Exception as e:
                self._log(f"⚠️ Не удалось загрузить модель из {base_model_id}: {str(e)}")
                
                # Если использовалась альтернативная модель для токенизатора, пробуем её же для модели
                if base_model_id != original_base_model_id:
                    self._log(f"🔄 Используем альтернативную модель: {base_model_id}")
                    try:
                        model = AutoModelForTokenClassification.from_pretrained(
                            base_model_id,
                            num_labels=num_labels,
                            cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache'),
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                        self._log(f"✅ Модель успешно загружена из альтернативной модели: {base_model_id}")
                    except Exception as alt_e:
                        self._log(f"❌ Не удалось загрузить модель из альтернативной модели {base_model_id}: {str(alt_e)}")
                        raise RuntimeError(f"Не удалось загрузить модель ни из {original_base_model_id}, ни из альтернативной модели {base_model_id}")
                else:
                    # Пробуем загрузить из локального пути
                    local_model_path = os.path.join(self.app_config.MODELS_PATH, 'layoutlm', base_model_id.split('/')[-1])
                    if os.path.exists(local_model_path):
                        self._log(f"🔄 Пробуем загрузить модель из локального пути: {local_model_path}")
                        try:
                            model = AutoModelForTokenClassification.from_pretrained(
                                local_model_path,
                                num_labels=num_labels,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                            )
                            self._log(f"✅ Модель успешно загружена из локального пути")
                        except Exception as local_e:
                            self._log(f"⚠️ Не удалось загрузить модель из локального пути: {str(local_e)}")
                            # Пробуем последнюю альтернативную модель
                            alternative_model = "microsoft/layoutlmv3-base"
                            self._log(f"🔄 Пробуем последнюю альтернативную модель: {alternative_model}")
                            try:
                                model = AutoModelForTokenClassification.from_pretrained(
                                    alternative_model,
                                    num_labels=num_labels,
                                    cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache'),
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                                )
                                self._log(f"✅ Модель успешно загружена из альтернативной модели: {alternative_model}")
                                # Обновляем базовую модель для дальнейших операций
                                base_model_id = alternative_model
                            except Exception as final_e:
                                self._log(f"❌ Не удалось загрузить модель из последней альтернативной модели: {str(final_e)}")
                                raise RuntimeError(f"Не удалось загрузить модель из всех доступных источников")
                    else:
                        # Если локальный путь не существует, пробуем альтернативную модель
                        alternative_model = "microsoft/layoutlmv3-base"
                        self._log(f"🔄 Пробуем использовать альтернативную модель: {alternative_model}")
                        try:
                            model = AutoModelForTokenClassification.from_pretrained(
                                alternative_model,
                                num_labels=num_labels,
                                cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache'),
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                            )
                            self._log(f"✅ Модель успешно загружена из альтернативной модели: {alternative_model}")
                            # Обновляем базовую модель для дальнейших операций
                            base_model_id = alternative_model
                        except Exception as alt_e:
                            self._log(f"❌ Не удалось загрузить модель из альтернативной модели: {str(alt_e)}")
                            raise RuntimeError(f"Не удалось загрузить модель из всех доступных источников")
            
            # Проверяем, что модель определена, прежде чем перемещать на устройство
            if 'model' not in locals():
                self._log(f"❌ Не удалось загрузить модель из всех доступных источников")
                raise RuntimeError(f"Не удалось загрузить модель из всех доступных источников")
            
            # Информация о загруженной модели
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._log(f"\n📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
            self._log(f"  📈 Всего параметров: {total_params:,}")
            self._log(f"  🎯 Обучаемых параметров: {trainable_params:,}")
            self._log(f"  📊 Процент обучаемых: {(trainable_params/total_params)*100:.2f}%")
            self._log(f"  🏷️ Количество классов: {num_labels}")
            
            # Перемещаем модель на нужное устройство
            self._log(f"\n🔄 Перемещение модели на устройство: {self.device}")
            model = model.to(self.device)
            
            # Проверяем память после загрузки
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                memory_used = memory_after - memory_before if 'memory_before' in locals() else memory_after
                self._log(f"💾 Память GPU после загрузки модели: {memory_after:.2f} GB")
                self._log(f"💾 Модель заняла памяти: {memory_used:.2f} GB")
            
            self._log("✅ Модель успешно размещена на устройстве")
            
            # 4. Готовим директорию для выходной модели
            output_dir = os.path.join(self.app_config.TRAINED_MODELS_PATH, output_model_name)
            
            # Проверяем, существует ли директория, и создаем её если нет
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Создана директория для модели: {output_dir}")
            
            # 5. Создаем TrainingArguments на основе переданных аргументов
            from transformers import TrainingArguments
            
            # Подготавливаем базовые TrainingArguments
            args = self._create_training_args(training_args, output_dir)
            
            self.logger.info(f"Подготовлены аргументы для обучения: {args}")
            
            # 6. Создаем Trainer
            from transformers import Trainer
            
            # Функция для вычисления метрик
            import numpy as np
            from datasets.features import ClassLabel
            from seqeval.metrics import classification_report
            
            # Получаем имена labels
            label_list = dataset["train"].features["ner_tags"].feature.names
            
            # Определяем функцию для вычисления метрик
            def compute_metrics(p):
                try:
                    predictions, labels = p
                    predictions = np.argmax(predictions, axis=2)

                    # Удаляем ignored index (особенно для padding токенов) с безопасной обработкой индексов
                    true_predictions = []
                    true_labels = []
                    
                    for prediction, label in zip(predictions, labels):
                        pred_sequence = []
                        label_sequence = []
                        
                        for pred_idx, label_idx in zip(prediction, label):
                            if label_idx != -100:  # Игнорируем padding токены
                                # Безопасно получаем метки с проверкой диапазона
                                if 0 <= pred_idx < len(label_list):
                                    pred_label = label_list[pred_idx]
                                else:
                                    pred_label = "O"  # Fallback к "O" для некорректных предсказаний
                                    
                                if 0 <= label_idx < len(label_list):
                                    true_label = label_list[label_idx]
                                else:
                                    true_label = "O"  # Fallback к "O" для некорректных меток
                                
                                pred_sequence.append(pred_label)
                                label_sequence.append(true_label)
                        
                        if pred_sequence and label_sequence:  # Добавляем только непустые последовательности
                            true_predictions.append(pred_sequence)
                            true_labels.append(label_sequence)
                    
                    # Проверяем, что у нас есть данные для оценки
                    if not true_predictions or not true_labels:
                        self.logger.warning("Нет валидных предсказаний для оценки")
                        return {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "macro avg_precision": 0.0,
                            "macro avg_recall": 0.0,
                            "macro avg_f1-score": 0.0,
                            "weighted avg_precision": 0.0,
                            "weighted avg_recall": 0.0,
                            "weighted avg_f1-score": 0.0
                        }
                    
                    # Используем библиотеку seqeval для оценки
                    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
                    
                    # Вычисляем базовые метрики
                    precision = precision_score(true_labels, true_predictions)
                    recall = recall_score(true_labels, true_predictions)
                    f1 = f1_score(true_labels, true_predictions)
                    
                    # Получаем детальный отчет
                    try:
                        results = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
                        
                        # Извлекаем агрегированные метрики
                        final_results = {
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                        }
                        
                        # Добавляем детализированные метрики
                        for key in ['macro avg', 'weighted avg']:
                            if key in results:
                                for metric in ['precision', 'recall', 'f1-score']:
                                    if metric in results[key]:
                                        final_results[f"{key}_{metric}"] = results[key][metric]
                        
                        return final_results
                        
                    except Exception as e:
                        self.logger.warning(f"Ошибка при получении детального отчета seqeval: {e}")
                        # Возвращаем базовые метрики
                        return {
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "macro avg_precision": precision,
                            "macro avg_recall": recall,
                            "macro avg_f1-score": f1,
                            "weighted avg_precision": precision,
                            "weighted avg_recall": recall,
                            "weighted avg_f1-score": f1
                        }
                        
                except Exception as e:
                    self.logger.error(f"Критическая ошибка при вычислении метрик: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    # Возвращаем нулевые метрики в случае критической ошибки
                    return {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "macro avg_precision": 0.0,
                        "macro avg_recall": 0.0,
                        "macro avg_f1-score": 0.0,
                        "weighted avg_precision": 0.0,
                        "weighted avg_recall": 0.0,
                        "weighted avg_f1-score": 0.0
                    }
            
            # Токенизируем данные
            self.logger.info("Токенизация данных...")
            
            # Получаем количество классов для проверки диапазона меток
            num_classes = len(dataset["train"].features["ner_tags"].feature.names)
            self.logger.info(f"Количество классов для меток: {num_classes}")
            
            # Проверяем диапазон меток перед обучением
            for split in dataset.keys():
                all_tags = []
                for tags in dataset[split]["ner_tags"]:
                    all_tags.extend(tags)
                
                # Удаляем игнорируемые метки
                valid_tags = [tag for tag in all_tags if tag != -100]
                
                if valid_tags:
                    min_tag = min(valid_tags)
                    max_tag = max(valid_tags)
                    self.logger.info(f"Диапазон меток в {split}: min={min_tag}, max={max_tag}")
                    
                    # Проверяем соответствие диапазону
                    if min_tag < 0:
                        self.logger.error(f"Найдены отрицательные метки в {split}! Исправляем...")
                        # Исправляем отрицательные метки, заменяя их на 0
                        for i, tags in enumerate(dataset[split]["ner_tags"]):
                            dataset[split]["ner_tags"][i] = [max(0, tag) for tag in tags]
                    
                    if max_tag >= num_classes:
                        self.logger.error(f"Найдены метки вне диапазона в {split}! Максимальная метка {max_tag} >= {num_classes}. Исправляем...")
                        # Исправляем метки, выходящие за диапазон
                        for i, tags in enumerate(dataset[split]["ner_tags"]):
                            dataset[split]["ner_tags"][i] = [min(tag, num_classes-1) if tag != -100 else -100 for tag in tags]
            
            # Функция для токенизации и выравнивания меток
            def tokenize_dataset(examples):
                try:
                    # Преобразуем ner_tags в labels для совместимости с методом _tokenize_and_align_labels
                    examples["labels"] = examples["ner_tags"]
                    
                    # Проверяем диапазон меток в текущем батче
                    for i, labels in enumerate(examples["labels"]):
                        if labels:
                            # Исправляем метки вне диапазона
                            examples["labels"][i] = [
                                -100 if label == -100 else max(0, min(label, num_classes-1))
                                for label in labels
                            ]
                    
                    # Проверяем наличие необходимых полей
                    if "words" not in examples or len(examples["words"]) == 0:
                        raise ValueError("Отсутствует поле 'words' в примерах")
                        
                    if "bboxes" not in examples or len(examples["bboxes"]) == 0:
                        raise ValueError("Отсутствует поле 'bboxes' в примерах")
                    
                    # LayoutLMv3 требует явной передачи bounding boxes и input_ids
                    tokenized_inputs = tokenizer(
                        examples["words"],
                        boxes=examples["bboxes"],
                        word_labels=examples["labels"],
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Проверяем, что токенизация вернула необходимые поля
                    if "input_ids" not in tokenized_inputs:
                        raise ValueError(f"Токенизация не создала поле 'input_ids'. Доступные поля: {list(tokenized_inputs.keys())}")
                        
                    return tokenized_inputs
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при токенизации: {str(e)}")
                    # Создаем минимальный пример, чтобы не прерывать процесс
                    return {
                        "input_ids": torch.zeros((1, 4), dtype=torch.long),
                        "attention_mask": torch.ones((1, 4), dtype=torch.long),
                        "bbox": torch.zeros((1, 4, 4), dtype=torch.long),
                        "labels": torch.ones((1, 4), dtype=torch.long) * -100
                    }
            
            # Применяем токенизацию к обоим сплитам датасета
            self._log("\n🔤 ЭТАП 4: ТОКЕНИЗАЦИЯ ДАННЫХ")
            self._log("-" * 50)
            
            tokenized_dataset = {}
            tokenization_start = time.time()
            
            for split in dataset.keys():
                split_start = time.time()
                self._log(f"🔄 Токенизация сплита '{split}' ({len(dataset[split])} примеров)...")
                
                # Проверяем первый пример перед токенизацией
                if len(dataset[split]) > 0:
                    example = dataset[split][0]
                    self._log(f"  📝 Пример структуры данных в '{split}':")
                    for key, value in example.items():
                        if isinstance(value, list):
                            self._log(f"    {key}: список длиной {len(value)}")
                        else:
                            self._log(f"    {key}: {type(value).__name__}")
                
                tokenized_dataset[split] = dataset[split].map(
                    tokenize_dataset,
                    batched=True,
                    remove_columns=dataset[split].column_names,
                    desc=f"Токенизация {split}"
                )
                
                split_time = time.time() - split_start
                self._log(f"✅ Датасет '{split}' токенизирован за {split_time:.2f}с. Размер: {len(tokenized_dataset[split])}")
                
                # Проверяем структуру токенизированных данных
                if len(tokenized_dataset[split]) > 0:
                    tokenized_example = tokenized_dataset[split][0]
                    self._log(f"  📊 Поля после токенизации: {list(tokenized_example.keys())}")
                    for key, value in tokenized_example.items():
                        if hasattr(value, 'shape'):
                            self._log(f"    {key}: shape {value.shape}")
                        else:
                            self._log(f"    {key}: {type(value).__name__}")
            
            tokenization_time = time.time() - tokenization_start
            self._log(f"⏱️ Общее время токенизации: {tokenization_time:.2f}с")
            
            # Создаем data collator для динамического паддинга
            self._log("\n⚙️ Создание data collator для динамического паддинга...")
            from transformers import DataCollatorForTokenClassification
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
            self._log("✅ Data collator создан")

            # Создаем тренер
            self._log("\n🏋️ ЭТАП 5: СОЗДАНИЕ ТРЕНЕРА")
            self._log("-" * 50)
            
            trainer_start = time.time()
            self._log("🔧 Инициализация Trainer...")
            
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            
            trainer_init_time = time.time() - trainer_start
            self._log(f"✅ Тренер создан за {trainer_init_time:.2f}с")
            
            # Определяем общее количество шагов для прогресс-бара
            num_train_epochs = trainer.args.num_train_epochs
            train_dataset_length = len(tokenized_dataset["train"])
            batch_size = trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count())
            total_steps = int((train_dataset_length / batch_size) * num_train_epochs)
            
            self._log(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
            self._log(f"  📅 Количество эпох: {num_train_epochs}")
            self._log(f"  📄 Примеров в тренировочном наборе: {train_dataset_length}")
            self._log(f"  📦 Размер батча на устройство: {trainer.args.per_device_train_batch_size}")
            self._log(f"  🎮 Количество GPU: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            self._log(f"  📦 Эффективный размер батча: {batch_size}")
            self._log(f"  👣 Общее количество шагов: {total_steps}")
            self._log(f"  💾 Градиентное накопление: {trainer.args.gradient_accumulation_steps}")
            self._log(f"  📝 Логирование каждые: {trainer.args.logging_steps} шагов")
            self._log(f"  💾 Сохранение каждые: {trainer.args.save_steps if hasattr(trainer.args, 'save_steps') else 'N/A'} шагов")
            self._log(f"  📊 Оценка каждые: {trainer.args.eval_steps if hasattr(trainer.args, 'eval_steps') else 'N/A'} шагов")
            
            # Добавляем отслеживание прогресса через callback
            class ProgressCallback(TrainerCallback):
                def __init__(self, progress_callback, log_callback):
                    self.progress_callback = progress_callback
                    self.log_callback = log_callback
                    self.last_logged_step = 0
                    self.last_logged_epoch = 0
                
                def on_step_end(self, args, state, control, **kwargs):
                    # Используем state.global_step вместо самостоятельного подсчета
                    current_step = state.global_step
                    total_steps = state.max_steps
                    
                    # Обновляем прогресс (передаем только процент)
                    if self.progress_callback is not None and total_steps > 0:
                        progress_percent = int((current_step / total_steps) * 100)
                        self.progress_callback(progress_percent)
                    
                    # Логируем каждые 10% или каждые 100 шагов (что меньше)
                    if total_steps > 0:
                        log_interval = min(100, max(1, total_steps // 10))
                        if current_step - self.last_logged_step >= log_interval:
                            if self.log_callback:
                                progress_percent = (current_step / total_steps) * 100
                                self.log_callback(f"🏃 Шаг {current_step}/{total_steps} ({progress_percent:.1f}%)")
                            self.last_logged_step = current_step
                    
                    return control
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    if self.log_callback:
                        epoch = int(state.epoch) + 1
                        self.log_callback(f"📅 Начало эпохи {epoch}/{args.num_train_epochs}")
                
                def on_epoch_end(self, args, state, control, **kwargs):
                    if self.log_callback:
                        epoch = int(state.epoch) + 1  # +1 потому что эпохи начинаются с 0
                        current_step = state.global_step
                        total_steps = state.max_steps
                        
                        # Получаем последние метрики
                        if state.log_history:
                            last_log = state.log_history[-1]
                            loss = last_log.get('train_loss', last_log.get('loss', 'N/A'))
                            lr = last_log.get('learning_rate', 'N/A')
                            
                            # Форматируем loss и lr
                            if isinstance(loss, float):
                                loss_str = f"{loss:.4f}"
                            else:
                                loss_str = str(loss)
                                
                            if isinstance(lr, float):
                                lr_str = f"{lr:.2e}"
                            else:
                                lr_str = str(lr)
                                
                            self.log_callback(f"✅ Эпоха {epoch} завершена. Loss: {loss_str}, LR: {lr_str}")
                        else:
                            self.log_callback(f"✅ Эпоха {epoch} завершена")
                            
                        # Обновляем прогресс по эпохам если доступно
                        if self.progress_callback is not None and total_steps > 0:
                            progress_percent = int((current_step / total_steps) * 100)
                            self.progress_callback(progress_percent)
                
                def on_evaluate(self, args, state, control, **kwargs):
                    if self.log_callback:
                        self.log_callback("📊 Запуск оценки на валидационном наборе...")
                        
                def on_train_begin(self, args, state, control, **kwargs):
                    if self.log_callback:
                        self.log_callback(f"🚀 Начинаем обучение на {state.max_steps} шагов ({args.num_train_epochs} эпох)")
                        
                def on_train_end(self, args, state, control, **kwargs):
                    if self.progress_callback is not None:
                        self.progress_callback(100)  # Завершаем прогресс на 100%
                    if self.log_callback:
                        self.log_callback("🎉 Обучение успешно завершено!")
            
            # Добавляем callback для отслеживания прогресса и логирования
            trainer.add_callback(ProgressCallback(
                self.progress_callback if hasattr(self, 'progress_callback') else None,
                self._log
            ))
            
            # Логируем начало обучения
            self._log("\n🚀 ЭТАП 6: НАЧАЛО ОБУЧЕНИЯ")
            self._log("=" * 80)
            training_start_time = time.time()
            
            # Проверяем состояние памяти перед обучением
            if torch.cuda.is_available():
                memory_before_training = torch.cuda.memory_allocated() / 1024**3
                memory_reserved_before = torch.cuda.memory_reserved() / 1024**3
                self._log(f"💾 Память GPU перед обучением: выделено {memory_before_training:.2f} GB, зарезервировано {memory_reserved_before:.2f} GB")
            
            self._log(f"🎯 Цель обучения: обучить модель LayoutLM для извлечения данных из документов")
            self._log(f"⏰ Время начала обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Запускаем обучение
            try:
                self._log("🏃 Запуск процесса обучения...")
                train_result = trainer.train()
                self._log("✅ Обучение успешно завершено!")
                
                # Сохраняем модель
                self._log(f"💾 Сохранение обученной модели в: {output_dir}")
                save_start = time.time()
                trainer.save_model(output_dir)
                save_time = time.time() - save_start
                self._log(f"✅ Модель сохранена за {save_time:.2f}с")
                
            except Exception as training_error:
                self._log(f"❌ ОШИБКА во время обучения: {str(training_error)}")
                import traceback
                self._log(f"📋 Трассировка ошибки:\n{traceback.format_exc()}")
                raise
            
            # Дополнительная проверка памяти после обучения
            if torch.cuda.is_available():
                memory_after_training = torch.cuda.memory_allocated() / 1024**3
                memory_reserved_after = torch.cuda.memory_reserved() / 1024**3
                self._log(f"💾 Память GPU после обучения: выделено {memory_after_training:.2f} GB, зарезервировано {memory_reserved_after:.2f} GB")
                
                if 'memory_before_training' in locals():
                    memory_increase = memory_after_training - memory_before_training
                    self._log(f"📈 Изменение использования памяти GPU: {memory_increase:+.2f} GB")
            
            # 8. Анализируем результаты обучения
            self._log("\n📊 ЭТАП 7: АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
            self._log("=" * 80)
            
            metrics = train_result.metrics
            self._log("📈 Получение метрик обучения...")
            trainer.log_metrics("train", metrics)
            
            # Подробное логирование метрик
            self._log("\n🎯 МЕТРИКИ ОБУЧЕНИЯ:")
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        self._log(f"  📊 {key}: {value:.6f}")
                    else:
                        self._log(f"  📊 {key}: {value}")
            else:
                self._log("  ⚠️ Метрики обучения недоступны")
            
            # Вычисляем общее время обучения
            total_training_time = time.time() - start_time
            actual_training_time = time.time() - training_start_time
            
            # Форматируем время
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}ч {minutes}м {secs}с"
                elif minutes > 0:
                    return f"{minutes}м {secs}с"
                else:
                    return f"{secs}с"
            
            self._log(f"\n⏱️ ВРЕМЕННАЯ СТАТИСТИКА:")
            self._log(f"  🕐 Общее время выполнения: {format_time(total_training_time)}")
            self._log(f"  🏃 Время непосредственно обучения: {format_time(actual_training_time)}")
            self._log(f"  ⚙️ Время подготовки: {format_time(total_training_time - actual_training_time)}")
            
            if total_steps > 0:
                time_per_step = actual_training_time / total_steps
                self._log(f"  👣 Среднее время на шаг: {time_per_step:.2f}с")
                
                if train_dataset_length > 0:
                    examples_per_second = (train_dataset_length * num_train_epochs) / actual_training_time
                    self._log(f"  📊 Примеров в секунду: {examples_per_second:.2f}")
            
            # Проводим оценку на валидационном наборе (если есть)
            if "validation" in tokenized_dataset and len(tokenized_dataset["validation"]) > 0:
                self._log("\n🔍 ОЦЕНКА НА ВАЛИДАЦИОННОМ НАБОРЕ:")
                self._log("-" * 50)
                
                try:
                    eval_start = time.time()
                    self._log("🔄 Запуск оценки на валидационных данных...")
                    
                    eval_results = trainer.evaluate()
                    eval_time = time.time() - eval_start
                    
                    self._log(f"✅ Оценка завершена за {format_time(eval_time)}")
                    self._log("\n📊 МЕТРИКИ ВАЛИДАЦИИ:")
                    
                    for key, value in eval_results.items():
                        if isinstance(value, float):
                            if 'loss' in key.lower():
                                self._log(f"  📉 {key}: {value:.6f}")
                            elif any(metric in key.lower() for metric in ['precision', 'recall', 'f1', 'accuracy']):
                                self._log(f"  🎯 {key}: {value:.4f} ({value*100:.2f}%)")
                            else:
                                self._log(f"  📊 {key}: {value:.6f}")
                        else:
                            self._log(f"  📊 {key}: {value}")
                    
                    # Анализируем качество модели
                    if 'eval_f1' in eval_results:
                        f1_score = eval_results['eval_f1']
                        if f1_score >= 0.9:
                            self._log("🌟 Отличное качество модели! (F1 >= 90%)")
                        elif f1_score >= 0.8:
                            self._log("👍 Хорошее качество модели (F1 >= 80%)")
                        elif f1_score >= 0.7:
                            self._log("⚠️ Удовлетворительное качество модели (F1 >= 70%)")
                        else:
                            self._log("🔴 Низкое качество модели (F1 < 70%). Рекомендуется дополнительное обучение")
                
                except Exception as eval_error:
                    self._log(f"❌ Ошибка при оценке на валидационном наборе: {str(eval_error)}")
            else:
                self._log("\n⚠️ Валидационный набор недоступен, пропускаем оценку")
            
            # Сохраняем дополнительную информацию об обучении
            self._log(f"\n💾 СОХРАНЕНИЕ ДОПОЛНИТЕЛЬНОЙ ИНФОРМАЦИИ:")
            self._log("-" * 50)
            
            # Сохраняем подробную конфигурацию обучения
            config_path = os.path.join(output_dir, "training_config.json")
            self._log(f"📄 Сохранение конфигурации обучения: {config_path}")
            
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    import json
                    import sys
                    
                    # Сохраняем полную информацию об обучении
                    config_data = {
                        "training_info": {
                            "base_model_id": base_model_id,
                            "dataset_path": dataset_path,
                            "output_model_name": output_model_name,
                            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "total_training_time_seconds": total_training_time,
                            "actual_training_time_seconds": actual_training_time
                        },
                        "training_args": training_args,
                        "training_metrics": metrics,
                        "validation_metrics": eval_results if 'eval_results' in locals() else {},
                        "dataset_info": {
                            "train_size": len(tokenized_dataset["train"]),
                            "validation_size": len(tokenized_dataset["validation"]) if "validation" in tokenized_dataset else 0,
                            "num_labels": num_labels,
                            "total_steps": total_steps,
                            "epochs": num_train_epochs
                        },
                        "system_info": {
                            "device": str(self.device),
                            "cuda_available": torch.cuda.is_available(),
                            "python_version": sys.version.split()[0],
                            "pytorch_version": torch.__version__
                        }
                    }
                    
                    # Добавляем информацию о GPU, если использовалась CUDA
                    if torch.cuda.is_available():
                        config_data["system_info"]["gpu_info"] = {
                            "device_name": torch.cuda.get_device_name(),
                            "device_count": torch.cuda.device_count(),
                            "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2
                        }
                    
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                self._log("✅ Конфигурация обучения сохранена")
            except Exception as save_error:
                self._log(f"❌ Ошибка при сохранении конфигурации: {str(save_error)}")
            
            # Сохраняем список меток
            if hasattr(self, 'label_list') and self.label_list:
                labels_path = os.path.join(output_dir, "label_list.json")
                self._log(f"🏷️ Сохранение списка меток: {labels_path}")
                try:
                    with open(labels_path, 'w', encoding='utf-8') as f:
                        json.dump(self.label_list, f, indent=2, ensure_ascii=False)
                    self._log("✅ Список меток сохранен")
                except Exception as labels_error:
                    self._log(f"❌ Ошибка при сохранении меток: {str(labels_error)}")
            
            # Создаем краткий отчет об обучении
            summary_path = os.path.join(output_dir, "training_summary.txt")
            self._log(f"📋 Создание краткого отчета: {summary_path}")
            
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ LAYOUTLM\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Базовая модель: {base_model_id}\n")
                    f.write(f"Датасет: {dataset_path}\n")
                    f.write(f"Устройство: {self.device}\n")
                    f.write(f"Общее время: {format_time(total_training_time)}\n")
                    f.write(f"Время обучения: {format_time(actual_training_time)}\n\n")
                    
                    f.write("ПАРАМЕТРЫ ОБУЧЕНИЯ:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Эпох: {num_train_epochs}\n")
                    f.write(f"Размер батча: {batch_size}\n")
                    f.write(f"Всего шагов: {total_steps}\n")
                    f.write(f"Примеров в тренировочном наборе: {len(tokenized_dataset['train'])}\n")
                    if "validation" in tokenized_dataset:
                        f.write(f"Примеров в валидационном наборе: {len(tokenized_dataset['validation'])}\n")
                    f.write(f"Количество меток: {num_labels}\n\n")
                    
                    if metrics:
                        f.write("МЕТРИКИ ОБУЧЕНИЯ:\n")
                        f.write("-" * 20 + "\n")
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                f.write(f"{key}: {value:.6f}\n")
                            else:
                                f.write(f"{key}: {value}\n")
                        f.write("\n")
                    
                    if 'eval_results' in locals() and eval_results:
                        f.write("МЕТРИКИ ВАЛИДАЦИИ:\n")
                        f.write("-" * 20 + "\n")
                        for key, value in eval_results.items():
                            if isinstance(value, float):
                                f.write(f"{key}: {value:.6f}\n")
                            else:
                                f.write(f"{key}: {value}\n")
                
                self._log("✅ Краткий отчет создан")
            except Exception as summary_error:
                self._log(f"❌ Ошибка при создании отчета: {str(summary_error)}")
            
            # Финальное сообщение
            self._log("\n🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            self._log("=" * 80)
            self._log(f"✅ Модель сохранена в: {output_dir}")
            self._log(f"⏱️ Общее время выполнения: {format_time(total_training_time)}")
            self._log(f"🎯 Основные файлы:")
            self._log(f"   📄 Конфигурация: {config_path}")
            self._log(f"   🏷️ Метки: {os.path.join(output_dir, 'label_list.json')}")
            self._log(f"   📋 Отчет: {summary_path}")
            self._log("🚀 Модель готова к использованию!")
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def train_donut_high_accuracy(self,
                                 dataset_path: str,
                                 base_model_id: str,
                                 training_args: dict,
                                 output_model_name: str,
                                 ocr_processor=None,
                                 gemini_processor=None) -> Optional[str]:
        """
        Обучает модель Donut с высокой точностью (> 98%)
        
        Args:
            dataset_path: Путь к папке с документами для обучения
            base_model_id: ID базовой модели Donut
            training_args: Аргументы обучения
            output_model_name: Имя выходной модели
            ocr_processor: OCR процессор для извлечения текста
            gemini_processor: Gemini процессор для интеллектуального извлечения
            
        Returns:
            str: Путь к обученной модели или None при ошибке
        """
        try:
            self._log("🎯 ========== ОБУЧЕНИЕ DONUT С ВЫСОКОЙ ТОЧНОСТЬЮ ==========")
            self._log(f"🎯 Целевая точность: > 98%")
            
            # Создаем улучшенный тренер
            enhanced_trainer = EnhancedDonutTrainer(self.app_config)
            
            # Передаем callbacks
            enhanced_trainer.set_callbacks(
                log_callback=self.log_callback,
                progress_callback=self.progress_callback
            )
            
            # Подготавливаем высококачественный датасет
            self._log("📊 Подготовка высококачественного датасета...")
            
            dataset = enhanced_trainer.prepare_high_quality_dataset(
                source_folder=dataset_path,
                ocr_processor=ocr_processor,
                gemini_processor=gemini_processor
            )
            
            if self.stop_requested:
                self._log("⏹️ Обучение остановлено пользователем")
                return None
                
            # Обучаем с оптимальными параметрами
            self._log("🚀 Запуск обучения с оптимизированными параметрами...")
            
            output_path = enhanced_trainer.train_high_accuracy_donut(
                dataset=dataset,
                base_model_id=base_model_id,
                output_model_name=output_model_name,
                training_args=training_args
            )
            
            if output_path and not self.stop_requested:
                # Тестируем модель
                self._log("\n🧪 Тестирование обученной модели...")
                
                tester = DonutModelTester(output_path)
                tester.load_model()
                
                # Если есть тестовые данные, проводим валидацию
                test_data_path = os.path.join(self.app_config.TEST_DATA_PATH, "invoices")
                if os.path.exists(test_data_path):
                    self._log(f"📁 Найдена папка с тестовыми данными: {test_data_path}")
                    
                    ground_truth_path = os.path.join(test_data_path, "ground_truth.json")
                    if not os.path.exists(ground_truth_path):
                        ground_truth_path = None
                        
                    results = tester.test_on_dataset(test_data_path, ground_truth_path)
                    passed = tester.validate_model_quality()
                    
                    if passed:
                        self._log("🎉 Модель успешно прошла валидацию! Точность > 98%")
                    else:
                        self._log("⚠️ Модель не достигла целевой точности 98%")
                else:
                    self._log("ℹ️ Папка с тестовыми данными не найдена, пропускаем валидацию")
                    
                self._log(f"\n✅ Обучение завершено! Модель сохранена в: {output_path}")
                return output_path
            else:
                return None
                
        except Exception as e:
            self._log(f"❌ Ошибка при обучении Donut: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

class TrainingMetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.training_loss = []
        self.eval_loss = []
        self.learning_rates = []
        self.epochs = []
        self.current_epoch = 0
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch += 1
        self.epochs.append(self.current_epoch)
        
        # Сохраняем метрики
        metrics = {
            'epoch': self.current_epoch,
            'training_loss': None,
            'eval_loss': None,
            'learning_rate': None
        }
        
        # Проверяем наличие элементов в log_history
        if state.log_history:
            last_log = state.log_history[-1]
            # Обновляем метрики из последнего лога
            metrics.update({
                'training_loss': last_log.get('loss', None),
                'eval_loss': last_log.get('eval_loss', None),
                'learning_rate': last_log.get('learning_rate', None)
            })
        
        # Добавляем значения только если они не None
        if metrics['training_loss'] is not None:
            self.training_loss.append(metrics['training_loss'])
        if metrics['eval_loss'] is not None:
            self.eval_loss.append(metrics['eval_loss'])
        if metrics['learning_rate'] is not None:
            self.learning_rates.append(metrics['learning_rate'])
            
        # Сохраняем метрики в JSON
        metrics_file = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'epochs': self.epochs,
                'training_loss': self.training_loss,
                'eval_loss': self.eval_loss,
                'learning_rates': self.learning_rates
            }, f, indent=2)
            
        # Генерируем графики
        self._plot_metrics()
        
    def _plot_metrics(self):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Использование не-интерактивного бэкенда для работы в фоновом потоке
            
            plt.figure(figsize=(12, 8))
            
            # График функции потерь
            plt.subplot(2, 1, 1)
            
            # Проверяем наличие данных и их размерности
            if self.training_loss and self.epochs:
                # Убеждаемся, что размерности совпадают
                x_train = self.epochs[:len(self.training_loss)]
                plt.plot(x_train, self.training_loss, label='Training Loss')
                
            if self.eval_loss and self.epochs:
                # Убеждаемся, что размерности совпадают
                x_eval = self.epochs[:len(self.eval_loss)]
                plt.plot(x_eval, self.eval_loss, label='Validation Loss')
                
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # График скорости обучения
            plt.subplot(2, 1, 2)
            
            if self.learning_rates and self.epochs:
                # Убеждаемся, что размерности совпадают
                x_lr = self.epochs[:len(self.learning_rates)]
                plt.plot(x_lr, self.learning_rates, label='Learning Rate')
                
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
            plt.close()
        except Exception as e:
            print(f"Ошибка при построении графиков: {str(e)}")
            import traceback
            print(traceback.format_exc()) 
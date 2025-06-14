import os
import json
import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

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
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Evaluation imports
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

class DonutDataCollator:
    """Кастомный data collator для Donut модели"""
    
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        
    def __call__(self, batch):
        """Обрабатывает батч данных для Donut"""
        # Извлекаем изображения и тексты
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Обрабатываем изображения
        pixel_values = self.processor(
            images, 
            return_tensors="pt"
        ).pixel_values
        
        # Обрабатываем тексты
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

class DonutMetricsCallback(TrainerCallback):
    """Callback для вычисления метрик Donut во время обучения"""
    
    def __init__(self, processor, eval_dataset, log_callback=None):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_callback = log_callback
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Вычисляет метрики после каждой оценки"""
        try:
            # Берем небольшую выборку для быстрой оценки
            eval_samples = self.eval_dataset.select(range(min(50, len(self.eval_dataset))))
            
            predictions = []
            references = []
            
            model.eval()
            with torch.no_grad():
                for sample in eval_samples:
                    # Подготавливаем входные данные
                    image = sample['image']
                    target_text = sample['text']
                    
                    # Генерируем предсказание
                    pixel_values = self.processor(image, return_tensors="pt").pixel_values
                    
                    # Используем task token для генерации
                    task_prompt = "<s_cord-v2>"
                    decoder_input_ids = self.processor.tokenizer(
                        task_prompt, 
                        add_special_tokens=False, 
                        return_tensors="pt"
                    ).input_ids
                    
                    # Генерируем ответ
                    outputs = model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=256,
                        num_beams=1,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    # Декодируем предсказание
                    pred_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    pred_text = pred_text.replace(task_prompt, "").strip()
                    
                    predictions.append(pred_text)
                    references.append(target_text)
            
            # Вычисляем метрики
            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                # BLEU score
                bleu = sentence_bleu([ref.split()], pred.split())
                bleu_scores.append(bleu)
                
                # ROUGE scores
                rouge = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(rouge['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(rouge['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(rouge['rougeL'].fmeasure)
            
            # Средние значения
            avg_bleu = np.mean(bleu_scores)
            avg_rouge1 = np.mean(rouge_scores['rouge1'])
            avg_rouge2 = np.mean(rouge_scores['rouge2'])
            avg_rougeL = np.mean(rouge_scores['rougeL'])
            
            # Дополнительная статистика
            min_bleu = np.min(bleu_scores) if bleu_scores else 0
            max_bleu = np.max(bleu_scores) if bleu_scores else 0
            std_bleu = np.std(bleu_scores) if bleu_scores else 0
            
            # Логируем метрики
            metrics_msg = (
                f"📊 Метрики оценки (на {len(bleu_scores)} примерах):\n"
                f"   BLEU: {avg_bleu:.4f} (мин: {min_bleu:.4f}, макс: {max_bleu:.4f}, σ: {std_bleu:.4f})\n"
                f"   ROUGE-1: {avg_rouge1:.4f}\n"
                f"   ROUGE-2: {avg_rouge2:.4f}\n"
                f"   ROUGE-L: {avg_rougeL:.4f}\n"
                f"   📈 Качество: {'Отлично' if avg_bleu > 0.8 else 'Хорошо' if avg_bleu > 0.6 else 'Удовлетворительно' if avg_bleu > 0.4 else 'Требует улучшения'}"
            )
            
            if self.log_callback:
                self.log_callback(metrics_msg)
                
            logger.info(metrics_msg)
            
        except Exception as e:
            error_msg = f"Ошибка при вычислении метрик: {str(e)}"
            if self.log_callback:
                self.log_callback(error_msg)
            logger.error(error_msg)

class DonutProgressCallback(TrainerCallback):
    """Callback для отслеживания прогресса обучения Donut"""
    
    def __init__(self, progress_callback=None, log_callback=None):
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Начало обучения"""
        self.start_time = datetime.now()
        if self.log_callback:
            self.log_callback("🚀 Начинаем обучение модели Donut...")
            
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Начало эпохи"""
        if self.log_callback:
            self.log_callback(f"📈 Эпоха {state.epoch + 1}/{args.num_train_epochs}")
            
    def on_step_end(self, args, state, control, **kwargs):
        """Конец шага"""
        if self.progress_callback and state.max_steps > 0:
            progress = int((state.global_step / state.max_steps) * 100)
            self.progress_callback(progress)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Логирование"""
        if logs and self.log_callback:
            # Основные метрики
            if 'loss' in logs:
                # Определяем тренд loss
                loss_trend = ""
                if hasattr(self, 'prev_loss'):
                    if logs['loss'] < self.prev_loss:
                        loss_trend = " ⬇️"
                    elif logs['loss'] > self.prev_loss:
                        loss_trend = " ⬆️"
                    else:
                        loss_trend = " ➡️"
                self.prev_loss = logs['loss']
                
                self.log_callback(f"📉 Шаг {state.global_step}: Loss = {logs['loss']:.4f}{loss_trend}")
            
            if 'eval_loss' in logs:
                # Определяем качество eval_loss
                eval_quality = ""
                if logs['eval_loss'] < 0.1:
                    eval_quality = " 🟢"
                elif logs['eval_loss'] < 0.5:
                    eval_quality = " 🟡"
                else:
                    eval_quality = " 🔴"
                    
                self.log_callback(f"📊 Eval Loss = {logs['eval_loss']:.4f}{eval_quality}")
            
            # Дополнительные метрики
            if 'learning_rate' in logs:
                self.log_callback(f"📈 Learning Rate = {logs['learning_rate']:.2e}")
                
            # Информация о прогрессе
            if state.max_steps > 0:
                progress_pct = (state.global_step / state.max_steps) * 100
                remaining_steps = state.max_steps - state.global_step
                self.log_callback(f"⏳ Прогресс: {progress_pct:.1f}% (осталось шагов: {remaining_steps})")
                
            # Время обучения
            if hasattr(self, 'start_time'):
                elapsed = datetime.now() - self.start_time
                elapsed_str = str(elapsed).split('.')[0]  # Убираем микросекунды
                self.log_callback(f"⏱️ Время обучения: {elapsed_str}")
            
            if 'grad_norm' in logs:
                self.log_callback(f"🔄 Gradient Norm = {logs['grad_norm']:.4f}")
            
            # Прогресс эпохи
            if state.max_steps > 0:
                epoch_progress = (state.global_step % (state.max_steps // args.num_train_epochs)) / (state.max_steps // args.num_train_epochs) * 100
                self.log_callback(f"⏳ Прогресс эпохи: {epoch_progress:.1f}%")
            
            # Оценка времени
            if self.start_time and state.global_step > 0:
                elapsed = datetime.now() - self.start_time
                steps_per_second = state.global_step / elapsed.total_seconds()
                remaining_steps = state.max_steps - state.global_step
                eta = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta)))
                self.log_callback(f"⏱️ ETA: {eta_str} (скорость: {steps_per_second:.2f} шагов/сек)")
                
    def on_train_end(self, args, state, control, **kwargs):
        """Конец обучения"""
        if self.start_time and self.log_callback:
            duration = datetime.now() - self.start_time
            self.log_callback(f"✅ Обучение завершено за {duration}")

class DonutTrainer:
    """Класс для обучения модели Donut"""
    
    def __init__(self, app_config):
        self.app_config = app_config
        self.logger = logging.getLogger("DonutTrainer")
        
        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"DonutTrainer использует устройство: {self.device}")
        
        # Callbacks
        self.progress_callback = None
        self.log_callback = None
        self.stop_requested = False
        
    def set_callbacks(self, log_callback=None, progress_callback=None):
        """Устанавливает функции обратного вызова"""
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def stop(self):
        """Остановка обучения"""
        self.stop_requested = True
        if self.log_callback:
            self.log_callback("⏹️ Получен сигнал остановки обучения")
            
    def _log(self, message):
        """Логирование с поддержкой callback"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
            
    def prepare_dataset(self, dataset_path: str, task_type: str = "document_parsing") -> DatasetDict:
        """
        Подготавливает датасет для обучения Donut
        
        Args:
            dataset_path: Путь к датасету
            task_type: Тип задачи (document_parsing, document_vqa, etc.)
            
        Returns:
            DatasetDict: Подготовленный датасет
        """
        self._log(f"📊 Подготовка датасета: {dataset_path}")
        
        try:
            # Загружаем датасет
            if os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
                # Hugging Face DatasetDict
                dataset = load_from_disk(dataset_path)
                self._log(f"✅ Загружен HF DatasetDict с разделами: {list(dataset.keys())}")
            else:
                # Кастомный формат - конвертируем
                dataset = self._convert_custom_dataset(dataset_path, task_type)
                
            # Проверяем структуру датасета
            self._validate_dataset(dataset)
            
            return dataset
            
        except Exception as e:
            error_msg = f"❌ Ошибка подготовки датасета: {str(e)}"
            self._log(error_msg)
            raise RuntimeError(error_msg)
            
    def _convert_custom_dataset(self, dataset_path: str, task_type: str) -> DatasetDict:
        """Конвертирует кастомный датасет в формат для Donut"""
        self._log("🔄 Конвертация кастомного датасета...")
        
        # Ищем изображения и аннотации
        images_dir = os.path.join(dataset_path, "images")
        annotations_file = os.path.join(dataset_path, "annotations.json")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Папка с изображениями не найдена: {images_dir}")
            
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Файл аннотаций не найден: {annotations_file}")
            
        # Загружаем аннотации
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        # Подготавливаем данные
        data = []
        for ann in annotations:
            image_path = os.path.join(images_dir, ann['image'])
            if os.path.exists(image_path):
                # Загружаем изображение
                image = Image.open(image_path).convert('RGB')
                
                # Формируем целевой текст в зависимости от типа задачи
                if task_type == "document_parsing":
                    # Для парсинга документов формируем JSON-подобную структуру
                    target_text = self._format_parsing_target(ann.get('fields', {}))
                elif task_type == "document_vqa":
                    # Для VQA формируем вопрос-ответ
                    target_text = self._format_vqa_target(ann.get('qa_pairs', []))
                else:
                    target_text = json.dumps(ann.get('fields', {}), ensure_ascii=False)
                    
                data.append({
                    'image': image,
                    'text': target_text
                })
                
        # Разделяем на train/validation
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # Создаем DatasetDict
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })
        
        self._log(f"✅ Конвертирован датасет: {len(train_data)} train, {len(val_data)} validation")
        
        return dataset_dict
        
    def _format_parsing_target(self, fields: Dict) -> str:
        """Форматирует поля для задачи парсинга документов"""
        # Формируем структурированный текст для Donut
        formatted_fields = []
        for key, value in fields.items():
            if value:  # Только непустые поля
                formatted_fields.append(f"<s_{key}>{value}</s_{key}>")
                
        return "".join(formatted_fields)
        
    def _format_vqa_target(self, qa_pairs: List[Dict]) -> str:
        """Форматирует пары вопрос-ответ для VQA"""
        formatted_pairs = []
        for qa in qa_pairs:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            if question and answer:
                formatted_pairs.append(f"<s_question>{question}</s_question><s_answer>{answer}</s_answer>")
                
        return "".join(formatted_pairs)
        
    def _validate_dataset(self, dataset: DatasetDict):
        """Проверяет корректность датасета"""
        required_splits = ['train']
        for split in required_splits:
            if split not in dataset:
                raise ValueError(f"Отсутствует обязательный раздел: {split}")
                
        # Проверяем наличие необходимых колонок
        train_dataset = dataset['train']
        required_columns = ['image', 'text']
        
        for col in required_columns:
            if col not in train_dataset.column_names:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")
                
        self._log(f"✅ Датасет валиден: {len(train_dataset)} примеров для обучения")
        
    def train_donut(self, 
                   dataset_path: str,
                   base_model_id: str,
                   training_args: dict,
                   output_model_name: str) -> Optional[str]:
        """
        Обучает модель Donut
        
        Args:
            dataset_path: Путь к датасету
            base_model_id: ID базовой модели
            training_args: Аргументы обучения
            output_model_name: Имя выходной модели
            
        Returns:
            str: Путь к обученной модели или None при ошибке
        """
        try:
            self._log("🍩 ========== НАЧАЛО ОБУЧЕНИЯ DONUT ==========")
            self._log(f"📊 Датасет: {dataset_path}")
            self._log(f"🤖 Базовая модель: {base_model_id}")
            self._log(f"💾 Имя выходной модели: {output_model_name}")
            self._log(f"🖥️ Устройство: {self.device}")
            
            # Логируем параметры обучения
            self._log("⚙️ Параметры обучения:")
            for key, value in training_args.items():
                self._log(f"   {key}: {value}")
            
            # 1. Подготавливаем датасет
            self._log("\n📊 ===== ЭТАП 1: ПОДГОТОВКА ДАТАСЕТА =====")
            task_type = training_args.get('task_type', 'document_parsing')
            self._log(f"🎯 Тип задачи: {task_type}")
            
            dataset = self.prepare_dataset(dataset_path, task_type)
            
            # Подробная информация о датасете
            self._log("📈 Статистика датасета:")
            for split_name, split_data in dataset.items():
                self._log(f"   {split_name}: {len(split_data)} примеров")
                if len(split_data) > 0:
                    # Показываем пример данных
                    example = split_data[0]
                    self._log(f"   Пример {split_name}:")
                    if 'image' in example:
                        img_size = example['image'].size if hasattr(example['image'], 'size') else 'неизвестно'
                        self._log(f"     Изображение: {img_size}")
                    if 'text' in example:
                        text_preview = example['text'][:100] + "..." if len(example['text']) > 100 else example['text']
                        self._log(f"     Текст: {text_preview}")
            
            if self.stop_requested:
                self._log("⏹️ Остановка на этапе подготовки датасета")
                return None
                
            # 2. Загружаем модель и процессор
            self._log("\n🤖 ===== ЭТАП 2: ЗАГРУЗКА МОДЕЛИ =====")
            self._log(f"📥 Загрузка процессора из: {base_model_id}")
            
            cache_dir = os.path.join(self.app_config.MODELS_PATH, 'donut_cache')
            self._log(f"💾 Кэш директория: {cache_dir}")
            
            processor = DonutProcessor.from_pretrained(
                base_model_id,
                cache_dir=cache_dir
            )
            self._log("✅ Процессор загружен успешно")
            
            self._log(f"📥 Загрузка модели из: {base_model_id}")
            model = VisionEncoderDecoderModel.from_pretrained(
                base_model_id,
                cache_dir=cache_dir
            )
            self._log("✅ Модель загружена успешно")
            
            # Информация о модели
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._log(f"📊 Параметры модели:")
            self._log(f"   Всего параметров: {total_params:,}")
            self._log(f"   Обучаемых параметров: {trainable_params:,}")
            
            # Перемещаем модель на устройство
            self._log(f"🔄 Перемещение модели на устройство: {self.device}")
            model.to(self.device)
            self._log("✅ Модель перемещена на устройство")
            
            if self.stop_requested:
                self._log("⏹️ Остановка на этапе загрузки модели")
                return None
                
            # 3. Настраиваем модель для обучения
            self._log("\n⚙️ ===== ЭТАП 3: НАСТРОЙКА МОДЕЛИ =====")
            self._configure_model_for_training(model, processor, training_args)
            self._log("✅ Модель настроена для обучения")
            
            # 4. Подготавливаем data collator
            self._log("\n🔧 ===== ЭТАП 4: ПОДГОТОВКА DATA COLLATOR =====")
            max_length = training_args.get('max_length', 512)
            self._log(f"📏 Максимальная длина последовательности: {max_length}")
            data_collator = DonutDataCollator(processor, max_length)
            self._log("✅ Data collator создан")
            
            # 5. Настраиваем аргументы обучения
            self._log("\n📋 ===== ЭТАП 5: НАСТРОЙКА АРГУМЕНТОВ ОБУЧЕНИЯ =====")
            output_dir = os.path.join(
                self.app_config.TRAINED_MODELS_PATH,
                f"donut_{output_model_name}"
            )
            os.makedirs(output_dir, exist_ok=True)
            self._log(f"📁 Выходная директория: {output_dir}")
            
            train_args = self._create_training_arguments(training_args, output_dir)
            self._log("✅ Аргументы обучения настроены")
            
            # Логируем ключевые параметры обучения
            self._log("📊 Ключевые параметры:")
            self._log(f"   Эпох: {train_args.num_train_epochs}")
            self._log(f"   Batch size (train): {train_args.per_device_train_batch_size}")
            self._log(f"   Batch size (eval): {train_args.per_device_eval_batch_size}")
            self._log(f"   Learning rate: {train_args.learning_rate}")
            self._log(f"   Gradient accumulation: {train_args.gradient_accumulation_steps}")
            self._log(f"   FP16: {getattr(train_args, 'fp16', False)}")
            
            # 6. Создаем callbacks
            self._log("\n🔔 ===== ЭТАП 6: СОЗДАНИЕ CALLBACKS =====")
            callbacks = self._create_callbacks(processor, dataset.get('validation'))
            self._log(f"✅ Создано callbacks: {len(callbacks)}")
            for i, callback in enumerate(callbacks):
                self._log(f"   {i+1}. {callback.__class__.__name__}")
            
            # 7. Создаем тренер
            self._log("\n🏃 ===== ЭТАП 7: СОЗДАНИЕ TRAINER =====")
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset.get('validation'),
                data_collator=data_collator,
                callbacks=callbacks,
            )
            self._log("✅ Trainer создан успешно")
            
            # Информация о тренировочном процессе
            train_dataset_size = len(dataset['train'])
            batch_size = train_args.per_device_train_batch_size
            grad_accum = train_args.gradient_accumulation_steps
            effective_batch_size = batch_size * grad_accum
            steps_per_epoch = train_dataset_size // effective_batch_size
            total_steps = steps_per_epoch * train_args.num_train_epochs
            
            self._log(f"📊 Детальная информация об обучении:")
            self._log(f"   📄 Размер тренировочного датасета: {train_dataset_size}")
            self._log(f"   📦 Размер батча: {batch_size}")
            self._log(f"   🔄 Gradient accumulation: {grad_accum}")
            self._log(f"   📈 Эффективный размер батча: {effective_batch_size}")
            self._log(f"   🔢 Шагов на эпоху: {steps_per_epoch}")
            self._log(f"   📊 Всего шагов обучения: {total_steps}")
            self._log(f"   ⏱️ Примерное время (при 1 сек/шаг): {total_steps // 60} мин")
            
            if self.stop_requested:
                self._log("⏹️ Остановка на этапе создания trainer")
                return None
                
            # 8. Запускаем обучение
            self._log("\n🚀 ===== ЭТАП 8: ЗАПУСК ОБУЧЕНИЯ =====")
            self._log("🎯 Начинаем тренировку модели...")
            
            # Добавляем обработчик остановки
            class StopTrainingCallback(TrainerCallback):
                def __init__(self, donut_trainer):
                    self.donut_trainer = donut_trainer
                    
                def on_step_end(self, args, state, control, **kwargs):
                    if self.donut_trainer.stop_requested:
                        control.should_training_stop = True
                        
            trainer.add_callback(StopTrainingCallback(self))
            
            # Логируем начало обучения
            start_time = datetime.now()
            self._log(f"⏰ Время начала: {start_time.strftime('%H:%M:%S')}")
            
            # Обучаем модель
            training_result = trainer.train()
            
            if self.stop_requested:
                self._log("⏹️ Обучение остановлено пользователем")
                return None
                
            # Логируем результаты обучения
            end_time = datetime.now()
            duration = end_time - start_time
            self._log(f"⏰ Время окончания: {end_time.strftime('%H:%M:%S')}")
            self._log(f"⏱️ Общее время обучения: {duration}")
            
            # Детальная статистика обучения
            if hasattr(training_result, 'training_loss'):
                final_loss = training_result.training_loss
                self._log(f"📉 Финальный training loss: {final_loss:.4f}")
                
                # Оценка качества loss
                if final_loss < 0.1:
                    loss_quality = "🟢 Отличный"
                elif final_loss < 0.5:
                    loss_quality = "🟡 Хороший"
                elif final_loss < 1.0:
                    loss_quality = "🟠 Удовлетворительный"
                else:
                    loss_quality = "🔴 Требует улучшения"
                    
                self._log(f"📊 Качество обучения: {loss_quality}")
                
            # Статистика производительности
            if hasattr(training_result, 'global_step') and training_result.global_step > 0:
                total_seconds = duration.total_seconds()
                steps_per_second = training_result.global_step / total_seconds
                seconds_per_step = total_seconds / training_result.global_step
                
                self._log(f"⚡ Производительность:")
                self._log(f"   🏃 Шагов в секунду: {steps_per_second:.2f}")
                self._log(f"   ⏱️ Секунд на шаг: {seconds_per_step:.2f}")
                
                if seconds_per_step < 1:
                    perf_rating = "🚀 Очень быстро"
                elif seconds_per_step < 5:
                    perf_rating = "⚡ Быстро"
                elif seconds_per_step < 10:
                    perf_rating = "🐎 Нормально"
                else:
                    perf_rating = "🐌 Медленно"
                    
                self._log(f"   📈 Оценка: {perf_rating}")
                
            # 9. Сохраняем модель
            self._log("\n💾 ===== ЭТАП 9: СОХРАНЕНИЕ МОДЕЛИ =====")
            self._log(f"📁 Сохранение в: {output_dir}")
            
            trainer.save_model(output_dir)
            self._log("✅ Модель сохранена")
            
            processor.save_pretrained(output_dir)
            self._log("✅ Процессор сохранен")
            
            # Сохраняем метаданные
            metadata = {
                'base_model': base_model_id,
                'task_type': task_type,
                'training_args': training_args,
                'created_at': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'training_duration': str(duration),
                'final_loss': getattr(training_result, 'training_loss', None),
                'total_steps': getattr(training_result, 'global_step', None)
            }
            
            metadata_path = os.path.join(output_dir, 'training_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self._log(f"✅ Метаданные сохранены: {metadata_path}")
                
            self._log(f"\n🎉 ========== ОБУЧЕНИЕ DONUT ЗАВЕРШЕНО ==========")
            self._log(f"📁 Модель сохранена в: {output_dir}")
            self._log(f"⏱️ Время обучения: {duration}")
            self._log(f"📊 Всего шагов: {getattr(training_result, 'global_step', 'неизвестно')}")
            
            return output_dir
            
        except Exception as e:
            self._log(f"\n💥 ========== ОШИБКА ОБУЧЕНИЯ DONUT ==========")
            error_msg = f"❌ Критическая ошибка: {str(e)}"
            self._log(error_msg)
            
            # Подробная диагностика ошибки
            import traceback
            import sys
            
            self._log("🔍 Диагностическая информация:")
            self._log(f"   Python версия: {sys.version}")
            self._log(f"   PyTorch версия: {torch.__version__}")
            self._log(f"   CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self._log(f"   CUDA устройств: {torch.cuda.device_count()}")
                self._log(f"   Текущее CUDA устройство: {torch.cuda.current_device()}")
                self._log(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            self._log(f"   Рабочая директория: {os.getcwd()}")
            self._log(f"   Датасет существует: {os.path.exists(dataset_path) if 'dataset_path' in locals() else 'неизвестно'}")
            self._log(f"   Модели директория: {self.app_config.MODELS_PATH}")
            self._log(f"   Обученные модели директория: {self.app_config.TRAINED_MODELS_PATH}")
            
            # Полная трассировка ошибки
            self._log("\n🔍 Полная трассировка ошибки:")
            full_traceback = traceback.format_exc()
            for line in full_traceback.split('\n'):
                if line.strip():
                    self._log(f"   {line}")
            
            self.logger.error(f"DonutTrainer error: {error_msg}")
            self.logger.error(full_traceback)
            
            return None
            
    def _configure_model_for_training(self, model, processor, training_args):
        """Настраивает модель для обучения"""
        self._log("🔧 Настройка конфигурации модели...")
        
        # Настраиваем размер изображения
        image_size = training_args.get('image_size', 384)
        self._log(f"   🖼️ Размер изображения: {image_size}x{image_size}")
        if hasattr(model.config, 'encoder') and hasattr(model.config.encoder, 'image_size'):
            model.config.encoder.image_size = [image_size, image_size]
            self._log("   ✅ Размер изображения установлен в encoder")
        else:
            self._log("   ⚠️ Encoder не поддерживает настройку размера изображения")
            
        # Настраиваем максимальную длину
        max_length = training_args.get('max_length', 512)
        self._log(f"   📏 Максимальная длина последовательности: {max_length}")
        if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'max_length'):
            model.config.decoder.max_length = max_length
            self._log("   ✅ Максимальная длина установлена в decoder")
        else:
            self._log("   ⚠️ Decoder не поддерживает настройку максимальной длины")
            
        # Настраиваем специальные токены
        self._log("   🏷️ Настройка специальных токенов:")
        
        pad_token_id = processor.tokenizer.pad_token_id
        model.config.pad_token_id = pad_token_id
        self._log(f"     pad_token_id: {pad_token_id}")
        
        eos_token_id = processor.tokenizer.eos_token_id
        model.config.eos_token_id = eos_token_id
        self._log(f"     eos_token_id: {eos_token_id}")
        
        bos_token_id = processor.tokenizer.bos_token_id
        model.config.bos_token_id = bos_token_id
        self._log(f"     bos_token_id: {bos_token_id}")
        
        # Информация о размере словаря
        vocab_size = len(processor.tokenizer)
        self._log(f"   📚 Размер словаря токенизатора: {vocab_size}")
        
        # Включаем gradient checkpointing для экономии памяти
        gradient_checkpointing = training_args.get('gradient_checkpointing', True)
        self._log(f"   💾 Gradient checkpointing: {gradient_checkpointing}")
        if gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
                self._log("   ✅ Gradient checkpointing включен")
            except Exception as e:
                self._log(f"   ⚠️ Не удалось включить gradient checkpointing: {e}")
        
        # Проверяем совместимость конфигурации
        self._log("   🔍 Проверка совместимости конфигурации:")
        if hasattr(model.config, 'vocab_size'):
            model_vocab_size = model.config.vocab_size
            self._log(f"     Размер словаря модели: {model_vocab_size}")
            if model_vocab_size != vocab_size:
                self._log(f"     ⚠️ Несоответствие размеров словарей!")
        
        self._log("✅ Конфигурация модели завершена")
        
    def _create_training_arguments(self, training_args: dict, output_dir: str) -> TrainingArguments:
        """Создает аргументы для обучения"""
        
        # Базовые аргументы
        args = {
            'output_dir': output_dir,
            'num_train_epochs': training_args.get('num_train_epochs', 5),
            'per_device_train_batch_size': training_args.get('per_device_train_batch_size', 2),
            'per_device_eval_batch_size': training_args.get('per_device_eval_batch_size', 2),
            'gradient_accumulation_steps': training_args.get('gradient_accumulation_steps', 4),
            'learning_rate': training_args.get('learning_rate', 3e-5),
            'weight_decay': training_args.get('weight_decay', 0.01),
            'warmup_ratio': training_args.get('warmup_ratio', 0.1),
            'logging_steps': 10,
            'save_steps': training_args.get('save_steps', 500),
            'eval_steps': training_args.get('eval_steps', 500),
            'eval_strategy': 'steps',  # Исправлено: использую новый параметр
            'save_strategy': 'steps',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'save_total_limit': 3,
            'report_to': 'none',
            'remove_unused_columns': False,
            'dataloader_pin_memory': False,
        }
        
        # FP16 оптимизация
        if training_args.get('fp16', True) and torch.cuda.is_available():
            args['fp16'] = True
            
        # Настройки для GPU
        if torch.cuda.is_available():
            args['dataloader_num_workers'] = 2
        else:
            args['dataloader_num_workers'] = 0
            
        return TrainingArguments(**args)
        
    def _create_callbacks(self, processor, eval_dataset) -> List[TrainerCallback]:
        """Создает callbacks для обучения"""
        callbacks = []
        
        # Progress callback
        if self.progress_callback or self.log_callback:
            callbacks.append(DonutProgressCallback(
                progress_callback=self.progress_callback,
                log_callback=self.log_callback
            ))
            
        # Metrics callback
        if eval_dataset:
            callbacks.append(DonutMetricsCallback(
                processor=processor,
                eval_dataset=eval_dataset,
                log_callback=self.log_callback
            ))
            
        # Early stopping
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        ))
        
        return callbacks 
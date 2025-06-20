"""
Оптимизатор гиперпараметров для TrOCR обучения
Анализирует характеристики датасета и предыдущие результаты для подбора оптимальных параметров
"""

import os
import json
import math
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetCharacteristics:
    """Характеристики датасета для оптимизации"""
    size: int
    avg_text_length: float
    complexity_score: float
    label_diversity: int
    image_dimensions: Tuple[int, int]
    

@dataclass
class OptimizationResults:
    """Результаты оптимизации гиперпараметров"""
    epochs: int
    batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int
    warmup_steps: int
    scheduler_type: str
    optimization_reason: str
    expected_training_time: int  # в минутах
    memory_usage_estimate: float  # в GB


class TrOCRHyperparameterOptimizer:
    """
    Интеллектуальный оптимизатор гиперпараметров для TrOCR
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация оптимизатора
        
        Args:
            logger: Логгер для вывода сообщений
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Базовые конфигурации для разных размеров датасетов
        self.size_based_configs = {
            'tiny': {  # < 20 примеров
                'base_epochs': 10,
                'base_lr': 3e-5,
                'base_batch_size': 2,
                'gradient_accumulation': 4
            },
            'small': {  # 20-50 примеров
                'base_epochs': 8,
                'base_lr': 5e-5,
                'base_batch_size': 4,
                'gradient_accumulation': 2
            },
            'medium': {  # 50-200 примеров
                'base_epochs': 5,
                'base_lr': 5e-5,
                'base_batch_size': 8,
                'gradient_accumulation': 1
            },
            'large': {  # > 200 примеров
                'base_epochs': 3,
                'base_lr': 2e-5,
                'base_batch_size': 16,
                'gradient_accumulation': 1
            }
        }
        
        # GPU memory limits для разных конфигураций
        self.memory_configs = {
            'conservative': {'max_batch': 2, 'memory_limit': 4.0},
            'balanced': {'max_batch': 4, 'memory_limit': 8.0},
            'aggressive': {'max_batch': 8, 'memory_limit': 12.0}
        }
    
    def analyze_dataset(self, dataset_path: str) -> DatasetCharacteristics:
        """
        Анализирует характеристики датасета
        
        Args:
            dataset_path: Путь к датасету
            
        Returns:
            DatasetCharacteristics: Характеристики датасета
        """
        try:
            from datasets import load_from_disk
            
            # Загружаем датасет
            dataset = load_from_disk(dataset_path)
            train_dataset = dataset['train'] if 'train' in dataset else dataset
            
            size = len(train_dataset)
            
            # Анализируем тексты
            texts = [item['text'] for item in train_dataset if 'text' in item]
            avg_text_length = sum(len(text) for text in texts) / len(texts) if texts else 0
            
            # Оценка сложности на основе длины текстов и разнообразия
            complexity_score = min(10.0, avg_text_length / 10.0)
            
            # Анализируем разнообразие меток (если есть)
            all_labels = set()
            for item in train_dataset:
                if 'labels' in item:
                    all_labels.update(item['labels'])
            label_diversity = len(all_labels)
            
            # Анализируем размеры изображений
            image_dims = (384, 384)  # TrOCR стандарт
            if train_dataset and 'image' in train_dataset[0]:
                try:
                    first_image = train_dataset[0]['image']
                    if hasattr(first_image, 'size'):
                        image_dims = first_image.size
                except:
                    pass
            
            return DatasetCharacteristics(
                size=size,
                avg_text_length=avg_text_length,
                complexity_score=complexity_score,
                label_diversity=label_diversity,
                image_dimensions=image_dims
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа датасета: {e}")
            # Возвращаем базовые характеристики
            return DatasetCharacteristics(
                size=10,
                avg_text_length=20.0,
                complexity_score=5.0,
                label_diversity=5,
                image_dimensions=(384, 384)
            )
    
    def analyze_previous_results(self, model_output_dir: str) -> Optional[Dict]:
        """
        Анализирует результаты предыдущего обучения
        
        Args:
            model_output_dir: Директория с результатами обучения
            
        Returns:
            Dict: Метрики предыдущего обучения или None
        """
        try:
            metadata_path = Path(model_output_dir) / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить предыдущие результаты: {e}")
        
        return None
    
    def optimize_hyperparameters(
        self, 
        dataset_path: str,
        gpu_memory_gb: float = 12.0,
        target_training_time_minutes: int = 30,
        previous_results: Optional[Dict] = None
    ) -> OptimizationResults:
        """
        Оптимизирует гиперпараметры на основе характеристик датасета
        
        Args:
            dataset_path: Путь к датасету
            gpu_memory_gb: Доступная GPU память в GB
            target_training_time_minutes: Целевое время обучения в минутах
            previous_results: Результаты предыдущего обучения
            
        Returns:
            OptimizationResults: Оптимизированные параметры
        """
        # Анализируем датасет
        characteristics = self.analyze_dataset(dataset_path)
        
        # Определяем категорию размера датасета
        if characteristics.size < 20:
            size_category = 'tiny'
        elif characteristics.size < 50:
            size_category = 'small'
        elif characteristics.size < 200:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # Базовая конфигурация
        base_config = self.size_based_configs[size_category]
        
        # Определяем конфигурацию памяти
        if gpu_memory_gb >= 10:
            memory_category = 'aggressive'
        elif gpu_memory_gb >= 6:
            memory_category = 'balanced'
        else:
            memory_category = 'conservative'
        
        memory_config = self.memory_configs[memory_category]
        
        # Начальные параметры
        epochs = base_config['base_epochs']
        learning_rate = base_config['base_lr']
        batch_size = min(base_config['base_batch_size'], memory_config['max_batch'])
        gradient_accumulation = base_config['gradient_accumulation']
        
        optimization_reasons = []
        
        # Анализируем предыдущие результаты для адаптации
        if previous_results:
            final_loss = previous_results.get('final_loss', 0)
            prev_epochs = previous_results.get('epochs', 3)
            prev_lr = previous_results.get('learning_rate', 5e-5)
            
            # Если loss слишком высокий - увеличиваем обучение
            if final_loss > 10.0:
                epochs = min(epochs + 5, 15)  # Увеличиваем эпохи
                learning_rate = prev_lr * 0.8  # Снижаем LR
                optimization_reasons.append(f"Высокий loss ({final_loss:.2f}) - увеличены эпохи и снижен LR")
            elif final_loss > 5.0:
                epochs = min(epochs + 2, 10)
                optimization_reasons.append(f"Умеренный loss ({final_loss:.2f}) - увеличены эпохи")
            elif final_loss < 1.0:
                epochs = max(epochs - 1, 2)
                optimization_reasons.append(f"Низкий loss ({final_loss:.2f}) - уменьшены эпохи")
        
        # Адаптация для маленьких датасетов
        if characteristics.size < 20:
            # Для очень маленьких датасетов нужно больше эпох но меньший LR
            epochs = max(epochs, 8)
            learning_rate = min(learning_rate, 2e-5)
            batch_size = min(batch_size, 2)  # Маленький batch для стабильности
            gradient_accumulation = max(gradient_accumulation, 4)
            optimization_reasons.append(f"Маленький датасет ({characteristics.size}) - больше эпох, меньший LR")
        
        # Адаптация для сложных текстов
        if characteristics.avg_text_length > 50:
            learning_rate *= 0.8  # Снижаем LR для сложных текстов
            epochs = min(epochs + 2, 12)
            optimization_reasons.append(f"Длинные тексты (ср. {characteristics.avg_text_length:.1f}) - снижен LR")
        
        # Адаптация под время обучения
        steps_per_epoch = max(1, characteristics.size // (batch_size * gradient_accumulation))
        estimated_time = epochs * steps_per_epoch * 2  # ~2 секунды на шаг
        
        if estimated_time > target_training_time_minutes * 60:
            # Слишком долго - сокращаем эпохи
            max_epochs = max(3, target_training_time_minutes * 30 // steps_per_epoch)
            if epochs > max_epochs:
                epochs = max_epochs
                optimization_reasons.append(f"Сокращены эпохи до {epochs} для укладки в {target_training_time_minutes}мин")
        
        # Настройка scheduler и warmup
        total_steps = epochs * steps_per_epoch
        warmup_steps = min(total_steps // 10, 100)  # 10% или максимум 100 шагов
        scheduler_type = "linear" if total_steps > 50 else "constant"
        
        # Оценка использования памяти
        memory_estimate = self._estimate_memory_usage(batch_size, gradient_accumulation)
        
        # Финальная проверка памяти
        if memory_estimate > gpu_memory_gb * 0.9:  # 90% лимит
            # Снижаем batch_size
            while batch_size > 1 and memory_estimate > gpu_memory_gb * 0.9:
                batch_size = max(1, batch_size - 1)
                gradient_accumulation = min(gradient_accumulation + 1, 8)
                memory_estimate = self._estimate_memory_usage(batch_size, gradient_accumulation)
            optimization_reasons.append(f"Снижен batch_size для экономии памяти")
        
        # Формируем итоговое объяснение
        reason = "; ".join(optimization_reasons) if optimization_reasons else f"Базовая оптимизация для {size_category} датасета"
        
        return OptimizationResults(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation,
            warmup_steps=warmup_steps,
            scheduler_type=scheduler_type,
            optimization_reason=reason,
            expected_training_time=estimated_time // 60,
            memory_usage_estimate=memory_estimate
        )
    
    def _estimate_memory_usage(self, batch_size: int, gradient_accumulation: int) -> float:
        """
        Оценивает использование GPU памяти
        
        Args:
            batch_size: Размер батча
            gradient_accumulation: Шаги накопления градиентов
            
        Returns:
            float: Оценка использования памяти в GB
        """
        # Базовая память модели TrOCR
        base_memory = 1.5  # GB для модели
        
        # Память на батч (приблизительно)
        batch_memory = batch_size * 0.8  # GB на единицу батча
        
        # Градиенты и оптимизатор
        optimizer_memory = 1.0  # GB для AdamW
        
        # Накопление градиентов увеличивает память
        accumulation_factor = 1 + (gradient_accumulation - 1) * 0.3
        
        total_memory = (base_memory + batch_memory + optimizer_memory) * accumulation_factor
        
        return total_memory
    
    def get_learning_rate_schedule_recommendations(self, characteristics: DatasetCharacteristics) -> Dict:
        """
        Рекомендации по планировщику learning rate
        
        Args:
            characteristics: Характеристики датасета
            
        Returns:
            Dict: Рекомендации по LR schedule
        """
        if characteristics.size < 30:
            return {
                'scheduler': 'constant_with_warmup',
                'warmup_ratio': 0.1,
                'reason': 'Маленький датасет - стабильный LR с warmup'
            }
        elif characteristics.complexity_score > 7:
            return {
                'scheduler': 'cosine',
                'warmup_ratio': 0.05,
                'reason': 'Сложный датасет - косинусное затухание'
            }
        else:
            return {
                'scheduler': 'linear',
                'warmup_ratio': 0.1,
                'reason': 'Стандартный случай - линейное затухание'
            }
    
    def generate_training_report(self, optimization: OptimizationResults, characteristics: DatasetCharacteristics) -> str:
        """
        Генерирует отчет об оптимизации
        
        Args:
            optimization: Результаты оптимизации
            characteristics: Характеристики датасета
            
        Returns:
            str: Отчет в виде строки
        """
        report = f"""
🔧 ОТЧЕТ ОБ ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ TrOCR

📊 Характеристики датасета:
   • Размер: {characteristics.size} примеров
   • Средняя длина текста: {characteristics.avg_text_length:.1f} символов
   • Сложность: {characteristics.complexity_score:.1f}/10
   • Разнообразие меток: {characteristics.label_diversity}
   • Размер изображений: {characteristics.image_dimensions[0]}x{characteristics.image_dimensions[1]}

⚙️ Оптимизированные параметры:
   • Эпохи: {optimization.epochs}
   • Batch size: {optimization.batch_size}
   • Learning rate: {optimization.learning_rate:.2e}
   • Gradient accumulation: {optimization.gradient_accumulation_steps}
   • Warmup steps: {optimization.warmup_steps}
   • Scheduler: {optimization.scheduler_type}

📈 Прогнозы:
   • Ожидаемое время обучения: ~{optimization.expected_training_time} минут
   • Использование GPU памяти: ~{optimization.memory_usage_estimate:.1f} GB

💡 Обоснование оптимизации:
   {optimization.optimization_reason}

🎯 Рекомендации:
   • Мониторьте validation loss каждые 2-3 эпохи
   • При переобучении уменьшите learning rate в 2 раза
   • При плато добавьте аугментацию данных
   • Сохраняйте checkpoint'ы для возобновления обучения
"""
        return report.strip() 
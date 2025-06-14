import os
import json
import torch
import logging
from typing import Dict, Optional, Tuple
from transformers import AutoModelForTokenClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ModelValidator:
    """Класс для валидации сохраненных моделей"""
    
    def __init__(self, model_dir: str):
        """
        Инициализация валидатора
        
        Args:
            model_dir: Путь к директории с моделью
        """
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.label_list = None
        
    def load_model(self) -> bool:
        """
        Загружает модель и проверяет ее структуру
        
        Returns:
            bool: True если загрузка успешна, False в противном случае
        """
        try:
            # Проверяем наличие необходимых файлов
            required_files = [
                "config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.txt",
                "label_list.json"
            ]
            
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    logger.error(f"Отсутствует файл: {file}")
                    return False
            
            # Загружаем список меток
            with open(os.path.join(self.model_dir, "label_list.json"), "r") as f:
                self.label_list = json.load(f)
            
            # Загружаем модель и токенизатор
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                num_labels=len(self.label_list)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Переносим модель на нужное устройство
            self.model = self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            return False
            
    def validate_model_outputs(self, test_dataset) -> Tuple[bool, Dict]:
        """
        Проверяет корректность выходов модели
        
        Args:
            test_dataset: Тестовый датасет
            
        Returns:
            Tuple[bool, Dict]: (успех, метрики)
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Модель не загружена")
                return False, {}
            
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
            
            all_preds = []
            all_labels = []
            
            # Проходим по тестовому датасету
            for batch in test_dataset:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Проверяем форму выходов
                if outputs.logits.shape[-1] != len(self.label_list):
                    logger.error(f"Некорректное количество классов в выходе: {outputs.logits.shape[-1]}")
                    return False, {}
                
                # Получаем предсказания
                preds = torch.argmax(outputs.logits, dim=-1)
                
                # Собираем предсказания и метки
                valid_preds = preds[labels != -100].cpu().numpy()
                valid_labels = labels[labels != -100].cpu().numpy()
                
                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)
            
            # Вычисляем метрики
            report = classification_report(
                all_labels,
                all_preds,
                labels=range(len(self.label_list)),
                target_names=self.label_list,
                output_dict=True
            )
            
            # Сохраняем confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            self._plot_confusion_matrix(cm)
            
            # Формируем итоговые метрики
            metrics = {
                'overall_accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'per_class': {
                    label: {
                        'precision': report[label]['precision'],
                        'recall': report[label]['recall'],
                        'f1': report[label]['f1-score']
                    }
                    for label in self.label_list
                }
            }
            
            # Сохраняем метрики
            metrics_path = os.path.join(self.model_dir, "validation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"Ошибка при валидации выходов модели: {str(e)}")
            return False, {}
            
    def validate_model_size(self) -> Tuple[bool, Dict]:
        """
        Проверяет размер модели и ее компонентов
        
        Returns:
            Tuple[bool, Dict]: (успех, информация о размере)
        """
        try:
            size_info = {}
            
            # Проверяем размер файлов модели
            for file in os.listdir(self.model_dir):
                file_path = os.path.join(self.model_dir, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    size_info[file] = f"{size_mb:.2f} MB"
            
            # Проверяем количество параметров модели
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                size_info['total_parameters'] = total_params
                size_info['trainable_parameters'] = trainable_params
                size_info['non_trainable_parameters'] = total_params - trainable_params
            
            # Сохраняем информацию о размере
            size_path = os.path.join(self.model_dir, "model_size_info.json")
            with open(size_path, "w") as f:
                json.dump(size_info, f, indent=2)
            
            return True, size_info
            
        except Exception as e:
            logger.error(f"Ошибка при проверке размера модели: {str(e)}")
            return False, {}
            
    def validate_memory_usage(self, batch_size: int = 1) -> Tuple[bool, float]:
        """
        Проверяет использование памяти при инференсе
        
        Args:
            batch_size: Размер батча для теста
            
        Returns:
            Tuple[bool, float]: (успех, использование памяти в MB)
        """
        try:
            if not self.model:
                return False, 0.0
            
            # Создаем тестовый батч
            test_input = {
                'input_ids': torch.randint(0, 1000, (batch_size, 512)).to(self.device),
                'attention_mask': torch.ones(batch_size, 512).to(self.device),
                'token_type_ids': torch.zeros(batch_size, 512).to(self.device),
                'bbox': torch.randint(0, 1000, (batch_size, 512, 4)).to(self.device),
                'pixel_values': torch.randn(batch_size, 3, 224, 224).to(self.device)
            }
            
            # Замеряем использование памяти
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = self.model(**test_input)
            
            memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            # Сохраняем информацию о памяти
            memory_info = {
                'batch_size': batch_size,
                'memory_used_mb': memory_used
            }
            
            memory_path = os.path.join(self.model_dir, "memory_usage_info.json")
            with open(memory_path, "w") as f:
                json.dump(memory_info, f, indent=2)
            
            return True, memory_used
            
        except Exception as e:
            logger.error(f"Ошибка при проверке использования памяти: {str(e)}")
            return False, 0.0
            
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Создает и сохраняет визуализацию confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_list,
            yticklabels=self.label_list
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(os.path.join(self.model_dir, "confusion_matrix.png"))
        plt.close() 
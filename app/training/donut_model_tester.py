"""
Утилита для тестирования и валидации обученной модели Donut
Позволяет проверить реальную точность на тестовом наборе данных
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict
from datetime import datetime
import time

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)


class DonutModelTester:
    """Класс для тестирования обученной модели Donut"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Путь к обученной модели
            device: Устройство для выполнения (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = None
        self.model = None
        
        # Метрики
        self.test_results = {
            'total_documents': 0,
            'perfect_documents': 0,
            'field_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'processing_times': [],
            'errors': []
        }
        
    def load_model(self):
        """Загружает модель и процессор"""
        logger.info(f"📥 Загрузка модели из {self.model_path}")
        
        try:
            self.processor = DonutProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Модель загружена на устройство: {self.device}")
            
            # Загружаем метаданные если есть
            metadata_path = os.path.join(self.model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"📊 Метаданные модели:")
                logger.info(f"   Базовая модель: {metadata.get('base_model', 'неизвестно')}")
                logger.info(f"   Дата обучения: {metadata.get('created_at', 'неизвестно')}")
                logger.info(f"   Целевая точность: {metadata.get('target_accuracy', 'неизвестно')}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
            
    def extract_fields_from_image(self, image_path: str) -> Dict[str, str]:
        """Извлекает поля из одного изображения"""
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Засекаем время
            start_time = time.time()
            
            # Подготавливаем входные данные
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Task prompt
            task_prompt = "<s_docvqa><s_question>Extract all fields from the document</s_question><s_answer>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Генерируем
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=768,
                    num_beams=4,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Декодируем результат
            prediction = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            prediction = prediction.replace(task_prompt, "").strip()
            
            # Замеряем время
            processing_time = time.time() - start_time
            self.test_results['processing_times'].append(processing_time)
            
            # Парсим результат
            fields = self._parse_donut_output(prediction)
            
            return fields
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {image_path}: {e}")
            self.test_results['errors'].append({
                'file': image_path,
                'error': str(e)
            })
            return {}
            
    def _parse_donut_output(self, text: str) -> Dict[str, str]:
        """Парсит выход Donut"""
        fields = {}
        
        # JSON парсинг
        try:
            if text.strip().startswith('{'):
                return json.loads(text)
        except:
            pass
            
        # Парсинг тегов
        import re
        pattern = r'<s_([^>]+)>([^<]+)</s_\1>'
        matches = re.findall(pattern, text)
        
        for field_name, value in matches:
            fields[field_name] = value.strip()
            
        return fields
        
    def test_on_dataset(self, test_data_path: str, ground_truth_path: str = None):
        """
        Тестирует модель на датасете
        
        Args:
            test_data_path: Путь к папке с тестовыми изображениями
            ground_truth_path: Путь к файлу с ground truth (опционально)
        """
        logger.info(f"🧪 Начинаем тестирование на датасете: {test_data_path}")
        
        # Получаем список файлов
        test_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.pdf']:
            test_files.extend(Path(test_data_path).glob(f'*{ext}'))
            
        logger.info(f"📁 Найдено тестовых файлов: {len(test_files)}")
        
        # Загружаем ground truth если есть
        ground_truth = {}
        if ground_truth_path and os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            logger.info(f"📊 Загружен ground truth для {len(ground_truth)} файлов")
            
        # Тестируем каждый файл
        results = []
        
        for i, file_path in enumerate(test_files):
            logger.info(f"🔍 Обработка {i+1}/{len(test_files)}: {file_path.name}")
            
            # Извлекаем поля
            predicted_fields = self.extract_fields_from_image(str(file_path))
            
            # Сравниваем с ground truth если есть
            if file_path.name in ground_truth:
                true_fields = ground_truth[file_path.name]
                accuracy = self._calculate_accuracy(predicted_fields, true_fields)
                
                results.append({
                    'file': file_path.name,
                    'predicted': predicted_fields,
                    'ground_truth': true_fields,
                    'accuracy': accuracy
                })
                
                logger.info(f"   ✅ Точность: {accuracy:.1%}")
            else:
                results.append({
                    'file': file_path.name,
                    'predicted': predicted_fields,
                    'ground_truth': None,
                    'accuracy': None
                })
                
        # Вычисляем общие метрики
        self._calculate_overall_metrics(results)
        
        # Сохраняем результаты
        self._save_test_results(results)
        
        return results
        
    def _calculate_accuracy(self, predicted: Dict, ground_truth: Dict) -> float:
        """Вычисляет точность для одного документа"""
        if not ground_truth:
            return 0.0
            
        self.test_results['total_documents'] += 1
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        correct_fields = 0
        total_fields = len(all_fields)
        
        document_perfect = True
        
        for field in all_fields:
            pred_value = predicted.get(field, "").strip().lower()
            true_value = ground_truth.get(field, "").strip().lower()
            
            if pred_value and true_value:
                if pred_value == true_value:
                    correct_fields += 1
                    self.test_results['field_metrics'][field]['tp'] += 1
                else:
                    # Проверяем частичное совпадение
                    if self._is_partial_match(pred_value, true_value):
                        correct_fields += 0.5
                        self.test_results['field_metrics'][field]['tp'] += 1
                    else:
                        self.test_results['field_metrics'][field]['fp'] += 1
                        document_perfect = False
            elif pred_value and not true_value:
                self.test_results['field_metrics'][field]['fp'] += 1
                document_perfect = False
            elif not pred_value and true_value:
                self.test_results['field_metrics'][field]['fn'] += 1
                document_perfect = False
                
        if document_perfect and total_fields > 0:
            self.test_results['perfect_documents'] += 1
            
        return correct_fields / total_fields if total_fields > 0 else 0.0
        
    def _is_partial_match(self, pred: str, true: str) -> bool:
        """Проверка частичного совпадения"""
        # Удаляем все кроме цифр для сравнения чисел
        pred_digits = ''.join(filter(str.isdigit, pred))
        true_digits = ''.join(filter(str.isdigit, true))
        
        if pred_digits and pred_digits == true_digits:
            return True
            
        # Проверка вхождения
        if len(pred) > 3 and len(true) > 3:
            if pred in true or true in pred:
                return True
                
        return False
        
    def _calculate_overall_metrics(self, results: List[Dict]):
        """Вычисляет общие метрики"""
        logger.info("\n📊 ОБЩИЕ МЕТРИКИ ТЕСТИРОВАНИЯ:")
        
        # Точность по документам
        accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            logger.info(f"🎯 Средняя точность: {avg_accuracy:.1%}")
            logger.info(f"📄 Идеальных документов: {self.test_results['perfect_documents']}/{self.test_results['total_documents']} ({self.test_results['perfect_documents']/self.test_results['total_documents']*100:.1f}%)")
            
            # Распределение точности
            acc_90_plus = sum(1 for a in accuracies if a >= 0.9)
            acc_95_plus = sum(1 for a in accuracies if a >= 0.95)
            acc_98_plus = sum(1 for a in accuracies if a >= 0.98)
            
            logger.info(f"📈 Распределение точности:")
            logger.info(f"   ≥ 98%: {acc_98_plus} документов ({acc_98_plus/len(accuracies)*100:.1f}%)")
            logger.info(f"   ≥ 95%: {acc_95_plus} документов ({acc_95_plus/len(accuracies)*100:.1f}%)")
            logger.info(f"   ≥ 90%: {acc_90_plus} документов ({acc_90_plus/len(accuracies)*100:.1f}%)")
            
        # Метрики по полям
        logger.info("\n📋 Метрики по полям:")
        field_f1_scores = []
        
        for field, metrics in self.test_results['field_metrics'].items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            field_f1_scores.append(f1)
            
            if f1 < 0.95:  # Показываем только проблемные поля
                logger.info(f"   {field}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
                
        # Общий F1
        if field_f1_scores:
            overall_f1 = np.mean(field_f1_scores)
            logger.info(f"\n🏆 Общий F1-score: {overall_f1:.3f} ({overall_f1*100:.1f}%)")
            
            if overall_f1 >= 0.98:
                logger.info("🎉 ПОЗДРАВЛЯЕМ! Достигнута целевая точность > 98%!")
            elif overall_f1 >= 0.95:
                logger.info("🔥 Отличный результат! Близко к цели.")
            elif overall_f1 >= 0.90:
                logger.info("✅ Хороший результат, но есть что улучшить.")
            else:
                logger.info("⚠️ Требуется дополнительная оптимизация.")
                
        # Производительность
        if self.test_results['processing_times']:
            avg_time = np.mean(self.test_results['processing_times'])
            logger.info(f"\n⚡ Средняя скорость обработки: {avg_time:.2f} сек/документ")
            
        # Ошибки
        if self.test_results['errors']:
            logger.info(f"\n❌ Ошибок при обработке: {len(self.test_results['errors'])}")
            
    def _save_test_results(self, results: List[Dict]):
        """Сохраняет результаты тестирования"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем детальные результаты
        results_path = f"test_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': self.model_path,
                'test_date': timestamp,
                'summary': {
                    'total_documents': self.test_results['total_documents'],
                    'perfect_documents': self.test_results['perfect_documents'],
                    'average_processing_time': np.mean(self.test_results['processing_times']) if self.test_results['processing_times'] else 0
                },
                'detailed_results': results,
                'field_metrics': dict(self.test_results['field_metrics']),
                'errors': self.test_results['errors']
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n💾 Результаты сохранены в: {results_path}")
        
        # Сохраняем CSV для анализа
        if results:
            df_data = []
            for r in results:
                if r['accuracy'] is not None:
                    df_data.append({
                        'file': r['file'],
                        'accuracy': r['accuracy'],
                        'predicted_fields': len(r['predicted']),
                        'ground_truth_fields': len(r['ground_truth']) if r['ground_truth'] else 0
                    })
                    
            if df_data:
                df = pd.DataFrame(df_data)
                csv_path = f"test_results_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"📊 CSV результаты сохранены в: {csv_path}")
                
    def validate_model_quality(self) -> bool:
        """
        Проверяет, соответствует ли модель требованиям качества
        
        Returns:
            True если модель достигла целевой точности > 98%
        """
        if not self.test_results['field_metrics']:
            logger.warning("⚠️ Нет данных для валидации")
            return False
            
        # Вычисляем общий F1
        f1_scores = []
        for field, metrics in self.test_results['field_metrics'].items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
            
        overall_f1 = np.mean(f1_scores) if f1_scores else 0
        
        logger.info(f"\n🎯 Валидация модели:")
        logger.info(f"   Общий F1-score: {overall_f1:.3f}")
        logger.info(f"   Целевой порог: 0.98")
        
        if overall_f1 >= 0.98:
            logger.info("   ✅ Модель ПРОШЛА валидацию!")
            return True
        else:
            logger.info("   ❌ Модель НЕ прошла валидацию")
            logger.info(f"   Необходимо улучшить на: {(0.98 - overall_f1)*100:.1f}%")
            return False


def test_donut_model(model_path: str, test_data_path: str, ground_truth_path: str = None):
    """
    Главная функция для тестирования модели
    
    Args:
        model_path: Путь к обученной модели
        test_data_path: Путь к тестовым данным
        ground_truth_path: Путь к ground truth (опционально)
    """
    tester = DonutModelTester(model_path)
    
    # Загружаем модель
    tester.load_model()
    
    # Тестируем
    results = tester.test_on_dataset(test_data_path, ground_truth_path)
    
    # Валидируем качество
    passed = tester.validate_model_quality()
    
    return results, passed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Использование: python donut_model_tester.py <model_path> <test_data_path> [ground_truth_path]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    ground_truth_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_donut_model(model_path, test_data_path, ground_truth_path) 
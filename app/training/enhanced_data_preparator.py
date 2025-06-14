"""
Улучшенный TrainingDataPreparator с интеллектуальным извлечением данных
Интегрирует IntelligentDataExtractor с существующим функционалом
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import shutil

from .intelligent_data_extractor import IntelligentDataExtractor, ExtractedField
from .data_preparator import TrainingDataPreparator


class EnhancedDataPreparator:
    """
    Улучшенный подготовщик данных с интеллектуальным извлечением
    """
    
    def __init__(self, ocr_processor, gemini_processor, logger=None):
        self.ocr_processor = ocr_processor
        self.gemini_processor = gemini_processor
        self.logger = logger or logging.getLogger(__name__)
        
        # Создаем интеллектуальный экстрактор
        self.intelligent_extractor = IntelligentDataExtractor(
            gemini_processor=gemini_processor,
            logger=self.logger
        )
        
        # Создаем обычный TrainingDataPreparator для совместимости
        self.data_preparator = TrainingDataPreparator(
            ocr_processor=ocr_processor,
            gemini_processor=gemini_processor,
            logger=self.logger
        )
        
        # Статистика обработки
        self.processing_stats = {
            'total_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_fields_extracted': 0,
            'unique_field_types': set(),
            'document_types': {}
        }
    
    def prepare_enhanced_dataset(self, 
                                source_folder: str, 
                                output_path: str, 
                                dataset_type: str = "LayoutLM",
                                max_files: int = None,
                                progress_callback=None,
                                log_callback=None) -> str:
        """
        Подготавливает датасет с использованием интеллектуального извлечения
        
        Args:
            source_folder: Папка с исходными документами
            output_path: Путь для сохранения датасета
            dataset_type: Тип датасета (LayoutLM/Donut)
            max_files: Максимальное количество файлов для обработки
            progress_callback: Функция для отчета о прогрессе
            log_callback: Функция для логирования
            
        Returns:
            Путь к созданному датасету
        """
        try:
            self.logger.info("🚀 Запуск улучшенной подготовки датасета...")
            self._reset_stats()
            
            # Получаем список файлов
            files = self._get_files_to_process(source_folder, max_files)
            if not files:
                raise ValueError("Не найдено файлов для обработки")
            
            self.processing_stats['total_files'] = len(files)
            self._log_message(log_callback, f"📁 Найдено файлов: {len(files)}")
            
            # Создаем выходную папку
            os.makedirs(output_path, exist_ok=True)
            
            # Обрабатываем каждый файл
            enhanced_annotations = []
            
            for i, file_path in enumerate(files):
                try:
                    self._log_message(log_callback, f"📄 Обработка файла {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    # Интеллектуальное извлечение данных
                    annotation = self._process_file_enhanced(file_path, output_path)
                    
                    if annotation:
                        enhanced_annotations.append(annotation)
                        self.processing_stats['successful_extractions'] += 1
                        self._log_message(log_callback, f"✅ Файл обработан успешно")
                    else:
                        self.processing_stats['failed_extractions'] += 1
                        self._log_message(log_callback, f"❌ Ошибка обработки файла")
                    
                    # Обновляем прогресс
                    if progress_callback:
                        progress = int((i + 1) / len(files) * 100)
                        progress_callback(progress)
                        
                except Exception as e:
                    self.logger.error(f"❌ Ошибка обработки файла {file_path}: {e}")
                    self.processing_stats['failed_extractions'] += 1
                    continue
            
            # Создаем финальный датасет
            if enhanced_annotations:
                dataset_path = self._create_final_dataset(
                    enhanced_annotations, 
                    output_path, 
                    dataset_type
                )
                
                # Сохраняем статистику
                self._save_processing_stats(output_path)
                
                self._log_message(log_callback, f"🎉 Датасет создан успешно: {dataset_path}")
                self._log_processing_summary(log_callback)
                
                return dataset_path
            else:
                raise ValueError("Не удалось создать ни одной аннотации")
                
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка подготовки датасета: {e}")
            raise
    
    def _process_file_enhanced(self, file_path: str, output_path: str) -> Optional[Dict]:
        """Обрабатывает один файл с интеллектуальным извлечением"""
        try:
            # Конвертируем в изображения если нужно
            image_paths = self._convert_to_images(file_path, output_path)
            
            annotations = []
            for image_path in image_paths:
                annotation = self._process_single_image_enhanced(image_path)
                if annotation:
                    annotations.append(annotation)
            
            return annotations[0] if annotations else None
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки файла {file_path}: {e}")
            return None
    
    def _process_single_image_enhanced(self, image_path: str) -> Optional[Dict]:
        """Обрабатывает одно изображение с интеллектуальным извлечением"""
        try:
            self.logger.info(f"🔍 Интеллектуальная обработка: {image_path}")
            
            # 1. OCR для получения слов и координат
            ocr_result = self.ocr_processor.process_image(image_path)
            if not ocr_result or 'words' not in ocr_result:
                self.logger.error("❌ OCR не вернул результат")
                return None
            
            # 2. Интеллектуальное извлечение всех полезных данных
            extracted_data = self.intelligent_extractor.extract_all_data(image_path)
            
            if not extracted_data or not extracted_data.get('fields'):
                self.logger.warning("⚠️ Не извлечено полезных данных")
                # Возвращаем базовую аннотацию
                return self._create_basic_annotation(ocr_result, image_path)
            
            # 3. Конвертируем в формат обучения
            training_data = self.intelligent_extractor.convert_to_training_format(
                extracted_data, 
                ocr_result['words']
            )
            
            # 4. Создаем аннотацию
            annotation = self._create_enhanced_annotation(
                training_data, 
                extracted_data, 
                image_path
            )
            
            # Обновляем статистику
            self._update_stats(extracted_data)
            
            return annotation
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки изображения {image_path}: {e}")
            return None
    
    def _create_enhanced_annotation(self, training_data: Dict, extracted_data: Dict, image_path: str) -> Dict:
        """Создает улучшенную аннотацию"""
        try:
            # Подсчитываем статистику меток
            labels = training_data['labels']
            label_stats = {}
            for label in labels:
                label_stats[label] = label_stats.get(label, 0) + 1
            
            # Создаем аннотацию
            annotation = {
                'image_path': image_path,
                'words': training_data['words'],
                'bboxes': training_data['bboxes'],
                'labels': labels,
                'document_type': extracted_data.get('document_type', 'unknown'),
                'extracted_fields_count': extracted_data.get('total_fields', 0),
                'field_mappings': training_data.get('field_mappings', {}),
                'label_statistics': label_stats,
                'categories': extracted_data.get('categories', {}),
                'extraction_method': 'intelligent',
                'quality_score': self._calculate_quality_score(extracted_data, labels)
            }
            
            self.logger.info(f"✅ Создана улучшенная аннотация:")
            self.logger.info(f"   📝 Слов: {len(training_data['words'])}")
            self.logger.info(f"   🏷️ Полезных меток: {len([l for l in labels if l != 'O'])}")
            self.logger.info(f"   📊 Извлечено полей: {extracted_data.get('total_fields', 0)}")
            
            return annotation
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания аннотации: {e}")
            return None
    
    def _create_basic_annotation(self, ocr_result: Dict, image_path: str) -> Dict:
        """Создает базовую аннотацию если интеллектуальное извлечение не сработало"""
        words = ocr_result['words']
        return {
            'image_path': image_path,
            'words': [w['text'] for w in words],
            'bboxes': [w['bbox'] for w in words],
            'labels': ['O'] * len(words),
            'document_type': 'unknown',
            'extracted_fields_count': 0,
            'field_mappings': {},
            'label_statistics': {'O': len(words)},
            'categories': {},
            'extraction_method': 'basic',
            'quality_score': 0.1
        }
    
    def _calculate_quality_score(self, extracted_data: Dict, labels: List[str]) -> float:
        """Вычисляет оценку качества аннотации"""
        try:
            total_labels = len(labels)
            useful_labels = len([l for l in labels if l != 'O'])
            fields_count = extracted_data.get('total_fields', 0)
            
            if total_labels == 0:
                return 0.0
            
            # Базовая оценка по соотношению полезных меток
            label_ratio = useful_labels / total_labels
            
            # Бонус за количество извлеченных полей
            field_bonus = min(fields_count / 20, 0.3)  # Максимум 30% бонуса
            
            # Бонус за разнообразие категорий
            categories_count = len(extracted_data.get('categories', {}))
            category_bonus = min(categories_count / 10, 0.2)  # Максимум 20% бонуса
            
            quality_score = min(label_ratio + field_bonus + category_bonus, 1.0)
            
            return round(quality_score, 3)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка расчета качества: {e}")
            return 0.0
    
    def _convert_to_images(self, file_path: str, output_path: str) -> List[str]:
        """Конвертирует файл в изображения"""
        try:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                return [file_path]
            elif file_path.lower().endswith('.pdf'):
                # Используем существующий метод из DataPreparator
                return self.data_preparator._convert_pdf_to_images(file_path, output_path)
            else:
                self.logger.warning(f"⚠️ Неподдерживаемый формат файла: {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации файла {file_path}: {e}")
            return []
    
    def _get_files_to_process(self, source_folder: str, max_files: int = None) -> List[str]:
        """Получает список файлов для обработки"""
        try:
            supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
            files = []
            
            for root, dirs, filenames in os.walk(source_folder):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        files.append(os.path.join(root, filename))
            
            if max_files:
                files = files[:max_files]
            
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения списка файлов: {e}")
            return []
    
    def _create_final_dataset(self, annotations: List[Dict], output_path: str, dataset_type: str) -> str:
        """Создает финальный датасет"""
        try:
            self.logger.info("📦 Создание финального датасета...")
            
            if dataset_type == "LayoutLM":
                return self._create_layoutlm_dataset(annotations, output_path)
            elif dataset_type == "Donut":
                return self._create_donut_dataset(annotations, output_path)
            else:
                raise ValueError(f"Неподдерживаемый тип датасета: {dataset_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания датасета: {e}")
            raise
    
    def _create_layoutlm_dataset(self, annotations: List[Dict], output_path: str) -> str:
        """Создает датасет для LayoutLM"""
        try:
            # Используем существующий метод из DataPreparator
            # но с нашими улучшенными аннотациями
            
            # Конвертируем наши аннотации в формат DataPreparator
            converted_data = []
            for annotation in annotations:
                converted_data.append({
                    'words': annotation['words'],
                    'bboxes': annotation['bboxes'],
                    'labels': annotation['labels'],
                    'image_path': annotation['image_path']
                })
            
            # Создаем HuggingFace датасет
            dataset_path = os.path.join(output_path, "dataset_dict")
            
            # Используем метод из DataPreparator для создания HF датасета
            self.data_preparator.layoutlm_data = converted_data
            hf_dataset_path = self.data_preparator._create_huggingface_dataset_for_layoutlm(output_path)
            
            return hf_dataset_path
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания LayoutLM датасета: {e}")
            raise
    
    def _create_donut_dataset(self, annotations: List[Dict], output_path: str) -> str:
        """Создает датасет для Donut"""
        try:
            # Для Donut создаем JSON аннотации
            donut_annotations = []
            
            for annotation in annotations:
                # Создаем структуру для Donut
                donut_annotation = {
                    'image': os.path.basename(annotation['image_path']),
                    'ground_truth': self._create_donut_ground_truth(annotation)
                }
                donut_annotations.append(donut_annotation)
            
            # Сохраняем аннотации
            annotations_file = os.path.join(output_path, "annotations.json")
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(donut_annotations, f, ensure_ascii=False, indent=2)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания Donut датасета: {e}")
            raise
    
    def _create_donut_ground_truth(self, annotation: Dict) -> Dict:
        """Создает ground truth для Donut из аннотации"""
        try:
            # Извлекаем поля из field_mappings
            fields = {}
            
            for field_name, indices in annotation.get('field_mappings', {}).items():
                if indices:
                    # Собираем значение из слов
                    words = annotation['words']
                    field_value = ' '.join([words[i] for i in indices if i < len(words)])
                    fields[field_name.lower()] = field_value
            
            return fields
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания ground truth: {e}")
            return {}
    
    def _update_stats(self, extracted_data: Dict):
        """Обновляет статистику обработки"""
        try:
            fields = extracted_data.get('fields', [])
            self.processing_stats['total_fields_extracted'] += len(fields)
            
            # Уникальные типы полей
            for field in fields:
                if hasattr(field, 'field_type'):
                    self.processing_stats['unique_field_types'].add(field.field_type)
            
            # Типы документов
            doc_type = extracted_data.get('document_type', 'unknown')
            self.processing_stats['document_types'][doc_type] = \
                self.processing_stats['document_types'].get(doc_type, 0) + 1
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления статистики: {e}")
    
    def _save_processing_stats(self, output_path: str):
        """Сохраняет статистику обработки"""
        try:
            # Конвертируем set в list для JSON
            stats = self.processing_stats.copy()
            stats['unique_field_types'] = list(stats['unique_field_types'])
            
            stats_file = os.path.join(output_path, "processing_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"📊 Статистика сохранена: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения статистики: {e}")
    
    def _reset_stats(self):
        """Сбрасывает статистику"""
        self.processing_stats = {
            'total_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_fields_extracted': 0,
            'unique_field_types': set(),
            'document_types': {}
        }
    
    def _log_message(self, log_callback, message: str):
        """Отправляет сообщение в лог"""
        self.logger.info(message)
        if log_callback:
            log_callback(message)
    
    def _log_processing_summary(self, log_callback):
        """Логирует итоговую сводку обработки"""
        try:
            stats = self.processing_stats
            
            self._log_message(log_callback, "📊 ИТОГОВАЯ СТАТИСТИКА:")
            self._log_message(log_callback, f"   📁 Всего файлов: {stats['total_files']}")
            self._log_message(log_callback, f"   ✅ Успешно: {stats['successful_extractions']}")
            self._log_message(log_callback, f"   ❌ Ошибок: {stats['failed_extractions']}")
            self._log_message(log_callback, f"   📝 Извлечено полей: {stats['total_fields_extracted']}")
            self._log_message(log_callback, f"   🏷️ Типов полей: {len(stats['unique_field_types'])}")
            
            # Типы документов
            if stats['document_types']:
                self._log_message(log_callback, "   📄 Типы документов:")
                for doc_type, count in stats['document_types'].items():
                    self._log_message(log_callback, f"      • {doc_type}: {count}")
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка логирования сводки: {e}") 
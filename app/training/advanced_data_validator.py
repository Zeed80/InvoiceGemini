"""
Продвинутая система валидации и очистки данных для TrOCR датасетов
Включает детекцию дубликатов, анализ качества изображений и текста, автоматическую очистку
"""

import os
import cv2
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from PIL import Image, ImageStat
import logging
from collections import defaultdict, Counter
import re
import difflib
from datetime import datetime

# Для обнаружения дубликатов изображений
import imagehash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Для анализа качества изображений
try:
    from skimage import measure
    from skimage.filters import laplace
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Для проверки орфографии (опционально)
try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False


@dataclass
class ValidationIssue:
    """Структура для описания проблемы валидации"""
    type: str  # 'duplicate', 'low_quality', 'text_error', 'format_error'
    severity: str  # 'critical', 'warning', 'info'
    item_path: str
    description: str
    suggestion: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ImageQualityMetrics:
    """Метрики качества изображения"""
    width: int
    height: int
    channels: int
    file_size: int
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    aspect_ratio: float
    
    @property
    def resolution_score(self) -> float:
        """Оценка разрешения (0-1)"""
        min_resolution = 224 * 224
        max_resolution = 2048 * 2048
        current_resolution = self.width * self.height
        
        if current_resolution < min_resolution:
            return 0.0
        elif current_resolution > max_resolution:
            return 1.0
        else:
            return (current_resolution - min_resolution) / (max_resolution - min_resolution)
    
    @property
    def overall_score(self) -> float:
        """Общая оценка качества (0-1)"""
        scores = [
            self.resolution_score,
            min(self.brightness / 128.0, 1.0),  # Оптимальная яркость ~128
            min(self.contrast / 50.0, 1.0),     # Оптимальный контраст ~50
            min(self.sharpness / 100.0, 1.0),   # Высокая резкость
            max(0.0, 1.0 - self.noise_level / 50.0)  # Низкий уровень шума
        ]
        return sum(scores) / len(scores)


@dataclass
class TextQualityMetrics:
    """Метрики качества текста"""
    length: int
    word_count: int
    char_diversity: float
    language_confidence: float
    spelling_errors: int
    special_chars_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    
    @property
    def readability_score(self) -> float:
        """Оценка читаемости текста (0-1)"""
        if self.length == 0:
            return 0.0
            
        # Оптимальная длина для TrOCR: 10-100 символов
        length_score = 1.0 if 10 <= self.length <= 100 else max(0.1, 1.0 - abs(self.length - 55) / 100)
        
        # Разнообразие символов
        diversity_score = min(self.char_diversity / 0.8, 1.0)
        
        # Орфография
        spelling_score = max(0.0, 1.0 - self.spelling_errors / max(1, self.word_count))
        
        # Баланс типов символов
        balance_score = 1.0 - abs(self.special_chars_ratio - 0.1) - abs(self.uppercase_ratio - 0.1)
        balance_score = max(0.0, balance_score)
        
        return (length_score + diversity_score + spelling_score + balance_score) / 4


@dataclass
class DuplicateGroup:
    """Группа дубликатов"""
    type: str  # 'exact', 'near_duplicate', 'text_duplicate'
    similarity: float
    items: List[str]
    primary_item: str
    
    @property
    def duplicates_to_remove(self) -> List[str]:
        """Элементы для удаления (все кроме первичного)"""
        return [item for item in self.items if item != self.primary_item]


class AdvancedDataValidator:
    """Продвинутая система валидации и очистки данных"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.image_hashes = {}
        self.text_vectors = None
        self.vectorizer = None
        
        # Инициализация проверки орфографии
        self.spell_checker = None
        if SPELLCHECK_AVAILABLE:
            try:
                self.spell_checker = SpellChecker(language='ru')
                # Добавляем английский язык
                self.spell_checker_en = SpellChecker(language='en')
            except:
                self.spell_checker = None
        
        # Настройки валидации
        self.min_image_size = (64, 64)
        self.max_image_size = (4096, 4096)
        self.min_text_length = 1
        self.max_text_length = 500
        self.similarity_threshold = 0.95
        self.quality_threshold = 0.3
        
    def validate_dataset(self, dataset_path: str, 
                        check_duplicates: bool = True,
                        check_quality: bool = True,
                        check_text: bool = True) -> Dict:
        """Полная валидация датасета"""
        
        self.logger.info(f"🔍 Начинаем валидацию датасета: {dataset_path}")
        
        validation_results = {
            'dataset_path': dataset_path,
            'timestamp': datetime.now().isoformat(),
            'total_items': 0,
            'valid_items': 0,
            'issues': [],
            'duplicates': [],
            'quality_stats': {},
            'recommendations': []
        }
        
        try:
            # Загружаем аннотации
            annotations = self._load_annotations(dataset_path)
            validation_results['total_items'] = len(annotations)
            
            if not annotations:
                validation_results['issues'].append(ValidationIssue(
                    type='format_error',
                    severity='critical',
                    item_path=dataset_path,
                    description='Не найдено аннотаций в датасете',
                    suggestion='Проверьте формат файла аннотаций'
                ))
                return validation_results
            
            # 1. Проверка дубликатов
            if check_duplicates:
                self.logger.info("🔍 Поиск дубликатов...")
                duplicates = self._find_duplicates(annotations)
                validation_results['duplicates'] = [asdict(dup) for dup in duplicates]
                
                # Добавляем issues для дубликатов
                for dup_group in duplicates:
                    for item in dup_group.duplicates_to_remove:
                        validation_results['issues'].append(ValidationIssue(
                            type='duplicate',
                            severity='warning',
                            item_path=item,
                            description=f'Дубликат ({dup_group.type}, similarity: {dup_group.similarity:.3f})',
                            suggestion=f'Удалить, оставить: {dup_group.primary_item}',
                            metadata={'duplicate_group': dup_group.type, 'similarity': dup_group.similarity}
                        ))
            
            # 2. Проверка качества изображений
            if check_quality:
                self.logger.info("🖼️ Анализ качества изображений...")
                image_quality_stats = self._analyze_image_quality(annotations)
                validation_results['quality_stats']['images'] = image_quality_stats
            
            # 3. Проверка качества текста
            if check_text:
                self.logger.info("📝 Анализ качества текста...")
                text_quality_stats = self._analyze_text_quality(annotations)
                validation_results['quality_stats']['text'] = text_quality_stats
            
            # 4. Подсчет валидных элементов
            validation_results['valid_items'] = self._count_valid_items(validation_results)
            
            # 5. Генерация рекомендаций
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
            self.logger.info(f"✅ Валидация завершена. Найдено {len(validation_results['issues'])} проблем")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации: {e}")
            validation_results['issues'].append(ValidationIssue(
                type='format_error',
                severity='critical',
                item_path=dataset_path,
                description=f'Ошибка валидации: {str(e)}',
                suggestion='Проверьте формат датасета и пути к файлам'
            ))
        
        return validation_results
    
    def clean_dataset(self, validation_results, remove_duplicates=True, remove_low_quality=True, quality_threshold=0.3):
        """Автоматическая очистка датасета на основе результатов валидации"""
        try:
            dataset_path = validation_results['dataset_path']
            issues = validation_results['issues']
            duplicates = validation_results['duplicates']
            
            print(f"🧹 Начинаем очистку датасета: {dataset_path}")
            
            # Собираем элементы для удаления
            items_to_remove = set()
            removal_stats = {
                'duplicates_removed': 0,
                'low_quality_removed': 0,
                'total_removed': 0,
                'total_kept': 0,
                'removal_percentage': 0
            }
            
            # 1. Обрабатываем дубликаты
            if remove_duplicates:
                for duplicate_group in duplicates:
                    # Оставляем только первый элемент из группы дубликатов
                    for item_path in duplicate_group[1:]:  # Пропускаем первый элемент
                        items_to_remove.add(item_path)
                        removal_stats['duplicates_removed'] += 1
                        
                print(f"📋 Отмечено для удаления дубликатов: {removal_stats['duplicates_removed']}")
            
            # 2. Обрабатываем низкокачественные элементы
            if remove_low_quality:
                for issue in issues:
                    if issue['type'] in ['low_image_quality', 'low_text_quality']:
                        if issue.get('severity', 'medium') == 'critical' or issue.get('quality_score', 1.0) < quality_threshold:
                            item_path = issue.get('file_path', issue.get('item_path'))
                            if item_path:
                                items_to_remove.add(item_path)
                                removal_stats['low_quality_removed'] += 1
                                
                print(f"📋 Отмечено для удаления низкокачественных: {removal_stats['low_quality_removed']}")
            
            # 3. Выполняем очистку в зависимости от типа датасета
            removal_stats['total_removed'] = len(items_to_remove)
            
            if items_to_remove:
                if os.path.exists(os.path.join(dataset_path, 'metadata.json')):
                    # TrOCR датасет с метаданными
                    self._clean_trocr_dataset(dataset_path, items_to_remove)
                elif os.path.exists(os.path.join(dataset_path, 'dataset_dict')):
                    # Datasets формат
                    self._clean_datasets_format(dataset_path, items_to_remove)
                else:
                    # Стандартная папочная структура
                    self._clean_folder_structure(dataset_path, items_to_remove)
                    
                print(f"✅ Удалено элементов: {removal_stats['total_removed']}")
            else:
                print("ℹ️ Нет элементов для удаления")
            
            # 4. Пересчитываем статистику
            remaining_count = self._count_remaining_items(dataset_path)
            removal_stats['total_kept'] = remaining_count
            
            total_original = removal_stats['total_removed'] + remaining_count
            if total_original > 0:
                removal_stats['removal_percentage'] = (removal_stats['total_removed'] / total_original) * 100
            
            # 5. Обновляем метаданные
            self._update_metadata_after_cleanup(dataset_path, removal_stats)
            
            return {
                'success': True,
                'cleanup_stats': removal_stats,
                'dataset_path': dataset_path
            }
            
        except Exception as e:
            print(f"❌ Ошибка очистки датасета: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'cleanup_stats': {}
            }
    
    def _clean_trocr_dataset(self, dataset_path, items_to_remove):
        """Очистка TrOCR датасета с метаданными"""
        try:
            metadata_file = os.path.join(dataset_path, 'metadata.json')
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Удаляем записи из метаданных
            original_count = len(metadata.get('annotations', []))
            cleaned_annotations = []
            
            for annotation in metadata.get('annotations', []):
                image_path = annotation.get('image_path', '')
                full_image_path = os.path.join(dataset_path, image_path)
                
                if full_image_path not in items_to_remove:
                    cleaned_annotations.append(annotation)
                else:
                    # Удаляем файл изображения
                    if os.path.exists(full_image_path):
                        os.remove(full_image_path)
            
            # Обновляем метаданные
            metadata['annotations'] = cleaned_annotations
            metadata['total_samples'] = len(cleaned_annotations)
            
            # Сохраняем обновленные метаданные
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            print(f"📋 TrOCR метаданные обновлены: {original_count} → {len(cleaned_annotations)}")
            
        except Exception as e:
            print(f"❌ Ошибка очистки TrOCR датасета: {e}")
    
    def _clean_datasets_format(self, dataset_path, items_to_remove):
        """Очистка датасета в формате datasets"""
        try:
            # Для datasets формата нужно пересоздать датасет
            print("🔄 Обнаружен datasets формат - требуется пересоздание...")
            
            # Загружаем существующий датасет
            from datasets import Dataset, load_from_disk
            dataset_dict_path = os.path.join(dataset_path, 'dataset_dict')
            
            if os.path.exists(dataset_dict_path):
                dataset = load_from_disk(dataset_dict_path)
                
                # Фильтруем элементы
                def filter_function(example):
                    image_path = example.get('image_path', '')
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(dataset_path, image_path)
                    return image_path not in items_to_remove
                
                # Применяем фильтр
                cleaned_dataset = dataset.filter(filter_function)
                
                # Сохраняем очищенный датасет
                cleaned_dataset.save_to_disk(dataset_dict_path)
                
                print(f"📋 Datasets формат обновлен: {len(dataset)} → {len(cleaned_dataset)}")
            
        except Exception as e:
            print(f"❌ Ошибка очистки datasets формата: {e}")
    
    def _clean_folder_structure(self, dataset_path, items_to_remove):
        """Очистка датасета с папочной структурой"""
        try:
            removed_count = 0
            
            for item_path in items_to_remove:
                if os.path.exists(item_path):
                    os.remove(item_path)
                    removed_count += 1
                    
                    # Удаляем соответствующий текстовый файл, если есть
                    base_name = os.path.splitext(item_path)[0]
                    txt_path = base_name + '.txt'
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
            
            print(f"📋 Удалено файлов из папочной структуры: {removed_count}")
            
        except Exception as e:
            print(f"❌ Ошибка очистки папочной структуры: {e}")
    
    def _count_remaining_items(self, dataset_path):
        """Подсчитывает оставшиеся элементы в датасете"""
        try:
            # Проверяем тип датасета и считаем соответственно
            if os.path.exists(os.path.join(dataset_path, 'metadata.json')):
                # TrOCR датасет
                with open(os.path.join(dataset_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return len(metadata.get('annotations', []))
                
            elif os.path.exists(os.path.join(dataset_path, 'dataset_dict')):
                # Datasets формат
                from datasets import load_from_disk
                dataset = load_from_disk(os.path.join(dataset_path, 'dataset_dict'))
                return len(dataset)
                
            else:
                # Папочная структура - считаем изображения
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                count = 0
                for root, dirs, files in os.walk(dataset_path):
                    for file in files:
                        if os.path.splitext(file.lower())[1] in image_extensions:
                            count += 1
                return count
                
        except Exception as e:
            print(f"❌ Ошибка подсчета элементов: {e}")
            return 0
    
    def _update_metadata_after_cleanup(self, dataset_path, removal_stats):
        """Обновляет метаданные датасета после очистки"""
        try:
            # Создаем файл со статистикой очистки
            cleanup_log = {
                'cleanup_timestamp': datetime.now().isoformat(),
                'cleanup_stats': removal_stats,
                'cleanup_settings': {
                    'removed_duplicates': removal_stats['duplicates_removed'] > 0,
                    'removed_low_quality': removal_stats['low_quality_removed'] > 0
                }
            }
            
            log_file = os.path.join(dataset_path, 'cleanup_log.json')
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(cleanup_log, f, ensure_ascii=False, indent=2)
                
            print(f"📋 Создан лог очистки: {log_file}")
            
        except Exception as e:
            print(f"❌ Ошибка обновления метаданных: {e}")
    
    def _load_annotations(self, dataset_path: str) -> List[Dict]:
        """Загрузка аннотаций из датасета"""
        annotations_file = Path(dataset_path) / "annotations.json"
        
        if not annotations_file.exists():
            # Ищем другие возможные файлы аннотаций
            possible_files = [
                "annotations.json", "dataset.json", "data.json",
                "labels.json", "annotations.txt"
            ]
            
            for filename in possible_files:
                file_path = Path(dataset_path) / filename
                if file_path.exists():
                    annotations_file = file_path
                    break
            else:
                return []
        
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Нормализуем формат
            if isinstance(data, list):
                return data
            elif 'annotations' in data:
                return data['annotations']
            elif 'data' in data:
                return data['data']
            else:
                return [data]
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки аннотаций: {e}")
            return []
    
    def _find_duplicates(self, annotations: List[Dict]) -> List[DuplicateGroup]:
        """Поиск дубликатов изображений и текста"""
        duplicates = []
        
        # 1. Поиск дубликатов изображений по хешу
        image_duplicates = self._find_image_duplicates(annotations)
        duplicates.extend(image_duplicates)
        
        # 2. Поиск дубликатов текста
        text_duplicates = self._find_text_duplicates(annotations)
        duplicates.extend(text_duplicates)
        
        return duplicates
    
    def _find_image_duplicates(self, annotations: List[Dict]) -> List[DuplicateGroup]:
        """Поиск дубликатов изображений"""
        hash_groups = defaultdict(list)
        
        for ann in annotations:
            image_path = ann.get('image_path', '')
            if not os.path.exists(image_path):
                continue
                
            try:
                # Вычисляем перцептуальный хеш
                image = Image.open(image_path)
                img_hash = imagehash.phash(image)
                hash_groups[str(img_hash)].append(image_path)
                
            except Exception as e:
                self.logger.warning(f"Ошибка хеширования {image_path}: {e}")
                continue
        
        # Группируем дубликаты
        duplicates = []
        for img_hash, paths in hash_groups.items():
            if len(paths) > 1:
                duplicates.append(DuplicateGroup(
                    type='exact',
                    similarity=1.0,
                    items=paths,
                    primary_item=paths[0]  # Первый как основной
                ))
        
        # Поиск похожих изображений (разные хеши, но близкие)
        near_duplicates = self._find_near_duplicate_images(annotations)
        duplicates.extend(near_duplicates)
        
        return duplicates
    
    def _find_near_duplicate_images(self, annotations: List[Dict]) -> List[DuplicateGroup]:
        """Поиск похожих (но не идентичных) изображений"""
        if len(annotations) < 2:
            return []
            
        hashes = []
        paths = []
        
        for ann in annotations:
            image_path = ann.get('image_path', '')
            if not os.path.exists(image_path):
                continue
                
            try:
                image = Image.open(image_path)
                img_hash = imagehash.phash(image)
                hashes.append(img_hash)
                paths.append(image_path)
            except:
                continue
        
        duplicates = []
        processed = set()
        
        for i, hash1 in enumerate(hashes):
            if i in processed:
                continue
                
            similar_group = [paths[i]]
            
            for j, hash2 in enumerate(hashes[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Вычисляем расстояние Хемминга
                hamming_distance = hash1 - hash2
                similarity = 1.0 - (hamming_distance / 64.0)  # Нормализуем к 0-1
                
                if similarity >= self.similarity_threshold:
                    similar_group.append(paths[j])
                    processed.add(j)
            
            if len(similar_group) > 1:
                processed.add(i)
                duplicates.append(DuplicateGroup(
                    type='near_duplicate',
                    similarity=similarity,
                    items=similar_group,
                    primary_item=similar_group[0]
                ))
        
        return duplicates
    
    def _find_text_duplicates(self, annotations: List[Dict]) -> List[DuplicateGroup]:
        """Поиск дубликатов текста"""
        text_groups = defaultdict(list)
        
        # 1. Точные дубликаты текста
        for ann in annotations:
            text = ann.get('text', '').strip()
            if text:
                text_groups[text].append(ann.get('image_path', ''))
        
        exact_duplicates = []
        for text, paths in text_groups.items():
            if len(paths) > 1:
                exact_duplicates.append(DuplicateGroup(
                    type='text_duplicate',
                    similarity=1.0,
                    items=paths,
                    primary_item=paths[0]
                ))
        
        # 2. Похожие тексты (с использованием TF-IDF)
        similar_duplicates = self._find_similar_texts(annotations)
        
        return exact_duplicates + similar_duplicates
    
    def _find_similar_texts(self, annotations: List[Dict]) -> List[DuplicateGroup]:
        """Поиск похожих текстов с помощью TF-IDF"""
        texts = []
        paths = []
        
        for ann in annotations:
            text = ann.get('text', '').strip()
            if text and len(text) > 3:  # Игнорируем очень короткие тексты
                texts.append(text)
                paths.append(ann.get('image_path', ''))
        
        if len(texts) < 2:
            return []
        
        try:
            # Векторизация текстов
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,
                ngram_range=(1, 2),
                max_features=1000
            )
            
            text_vectors = self.vectorizer.fit_transform(texts)
            
            # Вычисляем косинусное сходство
            similarity_matrix = cosine_similarity(text_vectors)
            
            duplicates = []
            processed = set()
            
            for i in range(len(texts)):
                if i in processed:
                    continue
                    
                similar_group = [paths[i]]
                
                for j in range(i+1, len(texts)):
                    if j in processed:
                        continue
                        
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= self.similarity_threshold:
                        similar_group.append(paths[j])
                        processed.add(j)
                
                if len(similar_group) > 1:
                    processed.add(i)
                    duplicates.append(DuplicateGroup(
                        type='text_similarity',
                        similarity=float(similarity),
                        items=similar_group,
                        primary_item=similar_group[0]
                    ))
            
        except Exception as e:
            self.logger.warning(f"Ошибка поиска похожих текстов: {e}")
            return []
        
        return duplicates
    
    def _analyze_image_quality(self, annotations: List[Dict]) -> Dict:
        """Анализ качества изображений"""
        quality_stats = {
            'total_images': 0,
            'analyzed_images': 0,
            'average_quality': 0.0,
            'low_quality_count': 0,
            'resolution_stats': {},
            'quality_distribution': {'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}
        }
        
        quality_scores = []
        resolutions = []
        
        for ann in annotations:
            image_path = ann.get('image_path', '')
            quality_stats['total_images'] += 1
            
            if not os.path.exists(image_path):
                continue
            
            try:
                metrics = self._analyze_single_image(image_path)
                quality_score = metrics.overall_score
                quality_scores.append(quality_score)
                resolutions.append(metrics.width * metrics.height)
                
                # Классификация качества
                if quality_score >= 0.8:
                    quality_stats['quality_distribution']['excellent'] += 1
                elif quality_score >= 0.6:
                    quality_stats['quality_distribution']['good'] += 1
                elif quality_score >= 0.4:
                    quality_stats['quality_distribution']['average'] += 1
                else:
                    quality_stats['quality_distribution']['poor'] += 1
                    quality_stats['low_quality_count'] += 1
                
                quality_stats['analyzed_images'] += 1
                
            except Exception as e:
                self.logger.warning(f"Ошибка анализа изображения {image_path}: {e}")
                continue
        
        if quality_scores:
            quality_stats['average_quality'] = sum(quality_scores) / len(quality_scores)
            quality_stats['resolution_stats'] = {
                'min': min(resolutions),
                'max': max(resolutions),
                'average': sum(resolutions) / len(resolutions)
            }
        
        return quality_stats
    
    def _analyze_single_image(self, image_path: str) -> ImageQualityMetrics:
        """Анализ качества одного изображения"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        height, width, channels = image.shape
        file_size = os.path.getsize(image_path)
        
        # Конвертируем в градации серого для анализа
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Вычисляем метрики
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Резкость (Laplacian variance)
        if SKIMAGE_AVAILABLE:
            sharpness = laplace(gray).var()
        else:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
        
        # Уровень шума (оценка через высокочастотные компоненты)
        noise_level = self._estimate_noise_level(gray)
        
        aspect_ratio = width / height
        
        return ImageQualityMetrics(
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            aspect_ratio=aspect_ratio
        )
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Оценка уровня шума в изображении"""
        # Используем медианный фильтр для удаления шума
        median_filtered = cv2.medianBlur(gray_image, 5)
        
        # Разность между оригиналом и отфильтрованным изображением
        noise = cv2.absdiff(gray_image, median_filtered)
        
        # Среднее значение шума
        return np.mean(noise)
    
    def _analyze_text_quality(self, annotations: List[Dict]) -> Dict:
        """Анализ качества текста"""
        text_stats = {
            'total_texts': 0,
            'analyzed_texts': 0,
            'average_length': 0,
            'average_word_count': 0,
            'total_spelling_errors': 0,
            'language_distribution': defaultdict(int),
            'quality_distribution': {'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}
        }
        
        lengths = []
        word_counts = []
        spelling_errors = []
        
        for ann in annotations:
            text = ann.get('text', '').strip()
            text_stats['total_texts'] += 1
            
            if not text:
                continue
            
            try:
                metrics = self._analyze_single_text(text)
                
                lengths.append(metrics.length)
                word_counts.append(metrics.word_count)
                spelling_errors.append(metrics.spelling_errors)
                
                # Определяем язык (упрощенно)
                if self._is_cyrillic(text):
                    text_stats['language_distribution']['ru'] += 1
                elif self._is_latin(text):
                    text_stats['language_distribution']['en'] += 1
                else:
                    text_stats['language_distribution']['mixed'] += 1
                
                # Классификация качества
                quality_score = metrics.readability_score
                if quality_score >= 0.8:
                    text_stats['quality_distribution']['excellent'] += 1
                elif quality_score >= 0.6:
                    text_stats['quality_distribution']['good'] += 1
                elif quality_score >= 0.4:
                    text_stats['quality_distribution']['average'] += 1
                else:
                    text_stats['quality_distribution']['poor'] += 1
                
                text_stats['analyzed_texts'] += 1
                
            except Exception as e:
                self.logger.warning(f"Ошибка анализа текста '{text[:50]}...': {e}")
                continue
        
        if lengths:
            text_stats['average_length'] = sum(lengths) / len(lengths)
            text_stats['average_word_count'] = sum(word_counts) / len(word_counts)
            text_stats['total_spelling_errors'] = sum(spelling_errors)
        
        return text_stats
    
    def _analyze_single_text(self, text: str) -> TextQualityMetrics:
        """Анализ качества одного текста"""
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Разнообразие символов
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / max(1, length)
        
        # Проверка орфографии
        spelling_errors = 0
        if self.spell_checker and word_count > 0:
            try:
                if self._is_cyrillic(text):
                    unknown_words = self.spell_checker.unknown(words)
                else:
                    unknown_words = self.spell_checker_en.unknown(words) if hasattr(self, 'spell_checker_en') else set()
                spelling_errors = len(unknown_words)
            except:
                spelling_errors = 0
        
        # Соотношения типов символов
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        digits = sum(1 for c in text if c.isdigit())
        uppercase = sum(1 for c in text if c.isupper())
        
        special_chars_ratio = special_chars / max(1, length)
        digit_ratio = digits / max(1, length)
        uppercase_ratio = uppercase / max(1, length)
        
        # Уверенность в языке (упрощенная)
        if self._is_cyrillic(text):
            language_confidence = 0.9
        elif self._is_latin(text):
            language_confidence = 0.8
        else:
            language_confidence = 0.5
        
        return TextQualityMetrics(
            length=length,
            word_count=word_count,
            char_diversity=char_diversity,
            language_confidence=language_confidence,
            spelling_errors=spelling_errors,
            special_chars_ratio=special_chars_ratio,
            digit_ratio=digit_ratio,
            uppercase_ratio=uppercase_ratio
        )
    
    def _is_cyrillic(self, text: str) -> bool:
        """Проверка, содержит ли текст кириллицу"""
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        return cyrillic_chars > len(text) * 0.3
    
    def _is_latin(self, text: str) -> bool:
        """Проверка, содержит ли текст латиницу"""
        latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        return latin_chars > len(text) * 0.3
    
    def _count_valid_items(self, validation_results: Dict) -> int:
        """Подсчет валидных элементов"""
        total_items = validation_results['total_items']
        critical_issues = sum(1 for issue in validation_results['issues'] 
                            if issue['severity'] == 'critical')
        return max(0, total_items - critical_issues)
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Генерация рекомендаций по улучшению датасета"""
        recommendations = []
        
        # Анализ проблем
        issues_by_type = defaultdict(int)
        for issue in validation_results['issues']:
            issues_by_type[issue['type']] += 1
        
        # Рекомендации по дубликатам
        if issues_by_type['duplicate'] > 0:
            recommendations.append(
                f"🔄 Найдено {issues_by_type['duplicate']} дубликатов. "
                "Рекомендуется их удалить для улучшения качества обучения."
            )
        
        # Рекомендации по качеству
        if 'images' in validation_results.get('quality_stats', {}):
            img_stats = validation_results['quality_stats']['images']
            if img_stats.get('low_quality_count', 0) > 0:
                recommendations.append(
                    f"📸 Найдено {img_stats['low_quality_count']} изображений низкого качества. "
                    "Рекомендуется их улучшить или заменить."
                )
        
        # Рекомендации по тексту
        if 'text' in validation_results.get('quality_stats', {}):
            text_stats = validation_results['quality_stats']['text']
            if text_stats.get('total_spelling_errors', 0) > 0:
                recommendations.append(
                    f"📝 Найдено {text_stats['total_spelling_errors']} орфографических ошибок. "
                    "Рекомендуется проверить и исправить тексты."
                )
        
        # Общие рекомендации
        valid_ratio = validation_results['valid_items'] / max(1, validation_results['total_items'])
        if valid_ratio < 0.8:
            recommendations.append(
                "⚠️ Менее 80% данных проходят валидацию. "
                "Рекомендуется серьезная очистка датасета."
            )
        elif valid_ratio < 0.95:
            recommendations.append(
                "✨ Качество датасета можно улучшить, удалив проблемные элементы."
            )
        else:
            recommendations.append(
                "✅ Датасет имеет высокое качество, минимальная очистка требуется."
            )
        
        return recommendations
    
    def _save_clean_dataset(self, dataset_path: str, clean_annotations: List[Dict]):
        """Сохранение очищенного датасета"""
        # Создаем резервную копию
        original_file = Path(dataset_path) / "annotations.json"
        backup_file = Path(dataset_path) / f"annotations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if original_file.exists():
            import shutil
            shutil.copy2(original_file, backup_file)
        
        # Сохраняем очищенный датасет
        with open(original_file, 'w', encoding='utf-8') as f:
            json.dump(clean_annotations, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Очищенный датасет сохранен: {original_file}")
        self.logger.info(f"📋 Резервная копия: {backup_file}")
    
    def generate_quality_report(self, validation_results: Dict, output_path: str = None) -> str:
        """Генерация подробного отчета о качестве датасета"""
        if output_path is None:
            output_path = Path(validation_results['dataset_path']) / "quality_report.html"
        
        # HTML отчет с детальной статистикой
        html_content = self._generate_html_report(validation_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"📊 Отчет о качестве сохранен: {output_path}")
        return str(output_path)
    
    def _generate_html_report(self, validation_results: Dict) -> str:
        """Генерация HTML отчета"""
        # Упрощенный HTML отчет (можно расширить)
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Отчет о качестве датасета</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .critical {{ background: #ffe6e6; }}
        .warning {{ background: #fff3e0; }}
        .success {{ background: #e8f5e8; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .stat-item {{ background: #f8f9fa; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Отчет о качестве датасета</h1>
        <p>Датасет: {validation_results['dataset_path']}</p>
        <p>Время анализа: {validation_results['timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>📈 Общая статистика</h2>
        <div class="stats">
            <div class="stat-item">
                <strong>Всего элементов:</strong> {validation_results['total_items']}
            </div>
            <div class="stat-item">
                <strong>Валидных элементов:</strong> {validation_results['valid_items']}
            </div>
            <div class="stat-item">
                <strong>Найдено проблем:</strong> {len(validation_results['issues'])}
            </div>
            <div class="stat-item">
                <strong>Группы дубликатов:</strong> {len(validation_results['duplicates'])}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>💡 Рекомендации</h2>
        <ul>
        {''.join(f'<li>{rec}</li>' for rec in validation_results['recommendations'])}
        </ul>
    </div>
</body>
</html>
        """
        
        return html 
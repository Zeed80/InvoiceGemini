#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrOCR Dataset Preparator

Полноценный модуль для подготовки датасетов для обучения TrOCR моделей.
Поддерживает различные форматы входных данных и создает оптимизированные датасеты.

Основано на лучших практиках Microsoft Research и сообщества HuggingFace.
"""

import os
import json
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

# Импорты для transformers - с обработкой ошибок
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torchvision.transforms as transforms
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers не установлен. Установите: pip install transformers torch torchvision")


@dataclass
class TrOCRDatasetConfig:
    """Конфигурация для подготовки датасета TrOCR"""
    
    # Основные параметры
    model_name: str = "microsoft/trocr-base-stage1"  # Базовая модель для fine-tuning
    max_target_length: int = 128  # Максимальная длина текста
    image_size: Tuple[int, int] = (384, 384)  # Размер изображений
    
    # Параметры аугментации
    enable_augmentation: bool = True
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    gaussian_blur_prob: float = 0.3
    gaussian_blur_kernel: Tuple[int, int] = (3, 7)
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    
    # Параметры качества
    min_image_size: Tuple[int, int] = (32, 32)  # Минимальный размер изображения
    max_text_length: int = 200  # Максимальная длина исходного текста
    min_text_length: int = 1   # Минимальная длина текста
    
    # Форматы поддерживаемых файлов
    supported_image_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_image_formats is None:
            self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


class TrOCRCustomDataset(Dataset):
    """
    Кастомный Dataset для TrOCR с поддержкой аугментаций
    """
    
    def __init__(self, 
                 data_pairs: List[Tuple[str, str]], 
                 processor,
                 config: TrOCRDatasetConfig,
                 is_training: bool = True):
        """
        Args:
            data_pairs: Список пар (путь_к_изображению, текст)
            processor: TrOCR процессор
            config: Конфигурация датасета
            is_training: Флаг тренировочного режима (для аугментаций)
        """
        self.data_pairs = data_pairs
        self.processor = processor
        self.config = config
        self.is_training = is_training
        
        # Настройка аугментаций для тренировки
        if is_training and config.enable_augmentation and TRANSFORMERS_AVAILABLE:
            self.augmentation_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=config.brightness_range,
                    contrast=config.contrast_range,
                    saturation=config.saturation_range,
                    hue=config.hue_range
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=config.gaussian_blur_kernel,
                        sigma=config.gaussian_blur_sigma
                    )
                ], p=config.gaussian_blur_prob)
            ])
        else:
            self.augmentation_transform = None
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_path, text = self.data_pairs[idx]
        
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Применяем аугментации если это тренировка
            if self.augmentation_transform and self.is_training:
                image = self.augmentation_transform(image)
            
            if not TRANSFORMERS_AVAILABLE:
                # Fallback режим без transformers
                return {
                    "image_path": image_path,
                    "text": text,
                    "image": image
                }
            
            # Обрабатываем изображение через TrOCR процессор
            pixel_values = self.processor(image, return_tensors='pt').pixel_values
            
            # Токенизируем текст
            labels = self.processor.tokenizer(
                text,
                padding='max_length',
                max_length=self.config.max_target_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            # Заменяем padding токены на -100 (игнорируются в loss)
            labels = labels.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "pixel_values": pixel_values.squeeze(),
                "labels": labels
            }
            
        except Exception as e:
            logging.error(f"Ошибка обработки {image_path}: {e}")
            # Возвращаем пустой пример в случае ошибки
            if TRANSFORMERS_AVAILABLE:
                return {
                    "pixel_values": torch.zeros((3, self.config.image_size[0], self.config.image_size[1])),
                    "labels": torch.full((self.config.max_target_length,), -100, dtype=torch.long)
                }
            else:
                return {
                    "image_path": image_path,
                    "text": text,
                    "error": str(e)
                }


class TrOCRDatasetPreparator:
    """
    Основной класс для подготовки датасетов TrOCR
    """
    
    def __init__(self, config: Optional[TrOCRDatasetConfig] = None):
        self.config = config or TrOCRDatasetConfig()
        self.logger = logging.getLogger(__name__)
        self.processor = None
        
        # Инициализируем процессор если transformers доступен
        if TRANSFORMERS_AVAILABLE:
            try:
                self.processor = TrOCRProcessor.from_pretrained(
                    self.config.model_name,
                    cache_dir="data/models"
                )
                self.logger.info(f"Загружен TrOCR процессор: {self.config.model_name}")
            except Exception as e:
                self.logger.error(f"Ошибка загрузки процессора: {e}")
                self.processor = None
        else:
            self.logger.warning("Transformers недоступен - работа в ограниченном режиме")
    
    def prepare_from_folder_structure(self,
                                    source_folder: str,
                                    output_path: str,
                                    train_split: float = 0.8,
                                    val_split: float = 0.1,
                                    test_split: float = 0.1) -> Dict[str, str]:
        """
        Подготавливает датасет из структуры папок.
        
        Ожидаемая структура:
        source_folder/
        ├── images/
        │   ├── img1.jpg
        │   ├── img2.png
        │   └── ...
        └── annotations.txt  # или .json, .csv
        
        Args:
            source_folder: Папка с исходными данными
            output_path: Путь для сохранения готового датасета
            train_split: Доля тренировочных данных
            val_split: Доля валидационных данных
            test_split: Доля тестовых данных
            
        Returns:
            Dict с путями к созданным датасетам
        """
        source_path = Path(source_folder)
        if not source_path.exists():
            raise FileNotFoundError(f"Папка {source_folder} не найдена")
        
        self.logger.info(f"Подготовка датасета из: {source_folder}")
        
        # Ищем файлы аннотаций
        annotation_files = list(source_path.glob("*.txt")) + \
                          list(source_path.glob("*.json")) + \
                          list(source_path.glob("*.csv"))
        
        if not annotation_files:
            raise FileNotFoundError("Не найдены файлы аннотаций (.txt, .json, .csv)")
        
        # Парсим аннотации
        data_pairs = []
        for ann_file in annotation_files:
            pairs = self._parse_annotation_file(ann_file, source_path)
            data_pairs.extend(pairs)
        
        self.logger.info(f"Найдено {len(data_pairs)} пар изображение-текст")
        
        # Фильтруем и валидируем данные
        valid_pairs = self._validate_data_pairs(data_pairs)
        self.logger.info(f"После валидации: {len(valid_pairs)} пар")
        
        # Разделяем на train/val/test
        train_pairs, val_pairs, test_pairs = self._split_data(
            valid_pairs, train_split, val_split, test_split
        )
        
        # Создаем датасеты
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = {}
        if train_pairs:
            datasets['train'] = self._create_dataset_split(
                train_pairs, output_dir / "train", is_training=True
            )
        if val_pairs:
            datasets['validation'] = self._create_dataset_split(
                val_pairs, output_dir / "validation", is_training=False
            )
        if test_pairs:
            datasets['test'] = self._create_dataset_split(
                test_pairs, output_dir / "test", is_training=False
            )
        
        # Сохраняем метаданные
        self._save_dataset_metadata(output_dir, datasets, valid_pairs)
        
        self.logger.info(f"Датасет успешно подготовлен в: {output_path}")
        return datasets
    
    def prepare_from_invoice_annotations(self,
                                       images_folder: str,
                                       annotations_file: str,
                                       output_path: str,
                                       field_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Подготавливает датасет из аннотаций счетов (JSON формат).
        
        Args:
            images_folder: Папка с изображениями счетов
            annotations_file: JSON файл с аннотациями
            output_path: Путь для сохранения датасета
            field_mapping: Маппинг полей для извлечения текста
            
        Returns:
            Dict с путями к созданным датасетам
        """
        if field_mapping is None:
            field_mapping = {
                'invoice_number': 'Номер счета',
                'date': 'Дата',
                'supplier': 'Поставщик',
                'total_amount': 'Сумма',
                'customer': 'Покупатель'
            }
        
        # Загружаем аннотации
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        data_pairs = []
        images_path = Path(images_folder)
        
        for annotation in tqdm(annotations, desc="Обработка аннотаций"):
            image_file = annotation.get('image_file')
            if not image_file:
                continue
                
            image_path = images_path / image_file
            if not image_path.exists():
                self.logger.warning(f"Изображение не найдено: {image_path}")
                continue
            
            # Извлекаем текстовые поля
            extracted_data = annotation.get('extracted_data', {})
            for field, value in extracted_data.items():
                if field in field_mapping and value:
                    # Создаем текстовую метку с контекстом
                    text_label = f"{field_mapping[field]}: {value}"
                    data_pairs.append((str(image_path), text_label))
        
        self.logger.info(f"Извлечено {len(data_pairs)} пар из аннотаций счетов")
        
        # Валидируем и создаем датасет
        valid_pairs = self._validate_data_pairs(data_pairs)
        
        # Разделяем данные
        train_pairs, val_pairs, test_pairs = self._split_data(valid_pairs, 0.7, 0.15, 0.15)
        
        # Создаем выходную структуру
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = {}
        if train_pairs:
            datasets['train'] = self._create_dataset_split(
                train_pairs, output_dir / "train", is_training=True
            )
        if val_pairs:
            datasets['validation'] = self._create_dataset_split(
                val_pairs, output_dir / "validation", is_training=False
            )
        if test_pairs:
            datasets['test'] = self._create_dataset_split(
                test_pairs, output_dir / "test", is_training=False
            )
        
        self._save_dataset_metadata(output_dir, datasets, valid_pairs)
        return datasets
    
    def prepare_synthetic_dataset(self,
                                output_path: str,
                                num_samples: int = 10000,
                                text_sources: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Создает синтетический датасет для TrOCR.
        
        Args:
            output_path: Путь для сохранения датасета
            num_samples: Количество синтетических примеров
            text_sources: Источники текста для генерации
            
        Returns:
            Dict с путями к созданным датасетам
        """
        if text_sources is None:
            # Базовые тексты для генерации (можно расширить)
            text_sources = [
                "ООО \"Рога и копыта\"",
                "Счет-фактура №",
                "от {} г.",
                "Сумма к оплате:",
                "НДС 20%:",
                "Итого:",
                "Покупатель:",
                "Поставщик:",
                "Банковские реквизиты",
                "ИНН/КПП:",
                "Расчетный счет:",
            ]
        
        self.logger.info(f"Генерация {num_samples} синтетических примеров")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Генерируем синтетические данные
        synthetic_pairs = self._generate_synthetic_data(
            text_sources, num_samples, output_dir
        )
        
        # Разделяем данные
        train_pairs, val_pairs, test_pairs = self._split_data(
            synthetic_pairs, 0.8, 0.1, 0.1
        )
        
        # Создаем датасеты
        datasets = {}
        datasets['train'] = self._create_dataset_split(
            train_pairs, output_dir / "train", is_training=True
        )
        datasets['validation'] = self._create_dataset_split(
            val_pairs, output_dir / "validation", is_training=False
        )
        datasets['test'] = self._create_dataset_split(
            test_pairs, output_dir / "test", is_training=False
        )
        
        self._save_dataset_metadata(output_dir, datasets, synthetic_pairs)
        return datasets
    
    def _parse_annotation_file(self, ann_file: Path, source_path: Path) -> List[Tuple[str, str]]:
        """Парсит файл аннотаций в различных форматах"""
        
        pairs = []
        
        if ann_file.suffix == '.txt':
            # Формат: filename.jpg\ttext_content
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Разделяем по первому табу или пробелу
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        parts = line.split(' ', 1)
                    
                    if len(parts) == 2:
                        filename, text = parts
                        image_path = source_path / "images" / filename
                        if image_path.exists():
                            pairs.append((str(image_path), text.strip()))
        
        elif ann_file.suffix == '.json':
            # JSON формат
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if 'image' in item and 'text' in item:
                        image_path = source_path / "images" / item['image']
                        if image_path.exists():
                            pairs.append((str(image_path), item['text']))
            elif isinstance(data, dict):
                for filename, text in data.items():
                    image_path = source_path / "images" / filename
                    if image_path.exists():
                        pairs.append((str(image_path), str(text)))
        
        elif ann_file.suffix == '.csv':
            # CSV формат
            df = pd.read_csv(ann_file)
            
            # Ищем столбцы с изображениями и текстом
            image_cols = [col for col in df.columns if 'image' in col.lower() or 'file' in col.lower()]
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'label' in col.lower()]
            
            if image_cols and text_cols:
                image_col = image_cols[0]
                text_col = text_cols[0]
                
                for _, row in df.iterrows():
                    filename = row[image_col]
                    text = row[text_col]
                    image_path = source_path / "images" / filename
                    if image_path.exists() and pd.notna(text):
                        pairs.append((str(image_path), str(text)))
        
        return pairs
    
    def _validate_data_pairs(self, data_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Валидирует пары данных"""
        
        valid_pairs = []
        
        for image_path, text in tqdm(data_pairs, desc="Валидация данных"):
            try:
                # Проверяем изображение
                image = Image.open(image_path)
                width, height = image.size
                
                # Проверка минимального размера
                if width < self.config.min_image_size[0] or height < self.config.min_image_size[1]:
                    continue
                
                # Проверяем текст
                text = text.strip()
                if (len(text) < self.config.min_text_length or 
                    len(text) > self.config.max_text_length):
                    continue
                
                # Проверяем формат изображения
                image_ext = Path(image_path).suffix.lower()
                if image_ext not in self.config.supported_image_formats:
                    continue
                
                valid_pairs.append((image_path, text))
                
            except Exception as e:
                self.logger.warning(f"Ошибка валидации {image_path}: {e}")
                continue
        
        return valid_pairs
    
    def _split_data(self, data_pairs: List[Tuple[str, str]], 
                   train_split: float, val_split: float, test_split: float) -> Tuple[List, List, List]:
        """Разделяет данные на train/val/test"""
        
        # Проверяем что сумма долей равна 1
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Сумма долей должна быть 1.0, получено: {total}")
        
        # Перемешиваем данные
        np.random.shuffle(data_pairs)
        
        n_total = len(data_pairs)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_pairs = data_pairs[:n_train]
        val_pairs = data_pairs[n_train:n_train + n_val]
        test_pairs = data_pairs[n_train + n_val:]
        
        self.logger.info(f"Разделение данных: train={len(train_pairs)}, "
                        f"val={len(val_pairs)}, test={len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs
    
    def _create_dataset_split(self, data_pairs: List[Tuple[str, str]], 
                            output_dir: Path, is_training: bool) -> str:
        """Создает split датасета"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем изображения и создаем аннотации
        annotations = []
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, (image_path, text) in enumerate(tqdm(data_pairs, desc=f"Создание {output_dir.name}")):
            try:
                # Копируем изображение
                image_name = f"{i:06d}{Path(image_path).suffix}"
                target_image_path = images_dir / image_name
                shutil.copy2(image_path, target_image_path)
                
                # Добавляем аннотацию
                annotations.append({
                    "image": image_name,
                    "text": text
                })
                
            except Exception as e:
                self.logger.error(f"Ошибка копирования {image_path}: {e}")
                continue
        
        # Сохраняем аннотации
        annotations_file = output_dir / "annotations.json"
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        # Создаем и сохраняем PyTorch Dataset если transformers доступен
        if TRANSFORMERS_AVAILABLE and self.processor:
            dataset = TrOCRCustomDataset(
                [(str(images_dir / ann["image"]), ann["text"]) for ann in annotations],
                self.processor,
                self.config,
                is_training=is_training
            )
            
            # Сохраняем dataset
            dataset_file = output_dir / "dataset.pt"
            torch.save({
                'dataset': dataset,
                'config': self.config,
                'annotations': annotations
            }, dataset_file)
        
        return str(output_dir)
    
    def _generate_synthetic_data(self, text_sources: List[str], 
                               num_samples: int, output_dir: Path) -> List[Tuple[str, str]]:
        """Генерирует синтетические данные"""
        
        pairs = []
        synthetic_dir = output_dir / "synthetic_images"
        synthetic_dir.mkdir(exist_ok=True)
        
        from PIL import ImageDraw, ImageFont
        
        # Базовые настройки для генерации
        image_size = (400, 100)
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        try:
            # Пытаемся загрузить системный шрифт
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        for i in tqdm(range(num_samples), desc="Генерация синтетических данных"):
            # Выбираем случайный текст
            text = np.random.choice(text_sources)
            
            # Добавляем случайные числа для разнообразия
            if "№" in text:
                text += str(np.random.randint(1000, 99999))
            elif ":" in text:
                text += f" {np.random.randint(100, 999999)} руб."
            
            # Создаем изображение
            image = Image.new('RGB', image_size, background_color)
            draw = ImageDraw.Draw(image)
            
            # Вычисляем позицию текста
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (image_size[0] - text_width) // 2
            y = (image_size[1] - text_height) // 2
            
            # Рисуем текст
            draw.text((x, y), text, fill=text_color, font=font)
            
            # Добавляем небольшой шум
            if np.random.random() < 0.3:
                # Небольшое размытие
                try:
                    image = image.filter(Image.BLUR)
                except:
                    pass  # Если фильтр не работает, пропускаем
            
            # Сохраняем изображение
            image_name = f"synthetic_{i:06d}.png"
            image_path = synthetic_dir / image_name
            image.save(image_path)
            
            pairs.append((str(image_path), text))
        
        return pairs
    
    def _save_dataset_metadata(self, output_dir: Path, 
                             datasets: Dict[str, str], 
                             all_pairs: List[Tuple[str, str]]):
        """Сохраняет метаданные датасета"""
        
        metadata = {
            "config": {
                "model_name": self.config.model_name,
                "max_target_length": self.config.max_target_length,
                "image_size": self.config.image_size,
                "enable_augmentation": self.config.enable_augmentation
            },
            "statistics": {
                "total_samples": len(all_pairs),
                "splits": {},
                "text_lengths": {
                    "min": min(len(text) for _, text in all_pairs) if all_pairs else 0,
                    "max": max(len(text) for _, text in all_pairs) if all_pairs else 0,
                    "avg": np.mean([len(text) for _, text in all_pairs]) if all_pairs else 0
                }
            },
            "paths": datasets,
            "created_at": pd.Timestamp.now().isoformat(),
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
        
        # Подсчитываем размеры splits
        for name, path in datasets.items():
            try:
                ann_file = Path(path) / "annotations.json"
                if ann_file.exists():
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                    metadata["statistics"]["splits"][name] = len(annotations)
            except:
                metadata["statistics"]["splits"][name] = 0
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Метаданные сохранены в: {metadata_file}")
    
    def load_prepared_dataset(self, dataset_path: str, split: str = "train"):
        """
        Загружает подготовленный датасет
        
        Args:
            dataset_path: Путь к подготовленному датасету
            split: Название split'а ('train', 'validation', 'test')
            
        Returns:
            TrOCRCustomDataset или dict с данными
        """
        dataset_file = Path(dataset_path) / split / "dataset.pt"
        
        if not dataset_file.exists():
            # Если нет .pt файла, загружаем из аннотаций
            ann_file = Path(dataset_path) / split / "annotations.json"
            if ann_file.exists():
                with open(ann_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                images_dir = Path(dataset_path) / split / "images"
                data_pairs = [(str(images_dir / ann["image"]), ann["text"]) for ann in annotations]
                
                if TRANSFORMERS_AVAILABLE and self.processor:
                    return TrOCRCustomDataset(data_pairs, self.processor, self.config, split=="train")
                else:
                    return {"data_pairs": data_pairs, "split": split}
            else:
                raise FileNotFoundError(f"Датасет не найден: {dataset_file} или {ann_file}")
        
        data = torch.load(dataset_file)
        return data['dataset']
    
    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """
        Получает информацию о подготовленном датасете
        
        Args:
            dataset_path: Путь к датасету
            
        Returns:
            Информация о датасете
        """
        metadata_file = Path(dataset_path) / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Метаданные не найдены: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# Функции для интеграции с основным приложением

def create_trocr_dataset_from_invoices(images_folder: str,
                                     annotations_file: str,
                                     output_path: str,
                                     config: Optional[TrOCRDatasetConfig] = None) -> Dict[str, str]:
    """
    Упрощенная функция для создания TrOCR датасета из аннотаций счетов
    
    Args:
        images_folder: Папка с изображениями счетов  
        annotations_file: JSON файл с аннотациями
        output_path: Путь для сохранения датасета
        config: Конфигурация (опционально)
        
    Returns:
        Dict с путями к созданным split'ам
    """
    preparator = TrOCRDatasetPreparator(config)
    return preparator.prepare_from_invoice_annotations(
        images_folder, annotations_file, output_path
    )


def create_synthetic_trocr_dataset(output_path: str,
                                 num_samples: int = 10000,
                                 config: Optional[TrOCRDatasetConfig] = None) -> Dict[str, str]:
    """
    Создает синтетический TrOCR датасет
    
    Args:
        output_path: Путь для сохранения
        num_samples: Количество примеров
        config: Конфигурация (опционально)
        
    Returns:
        Dict с путями к созданным split'ам
    """
    preparator = TrOCRDatasetPreparator(config)
    return preparator.prepare_synthetic_dataset(output_path, num_samples)


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 TrOCR Dataset Preparator")
    print(f"📦 Transformers доступен: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    
    # Создаем конфигурацию
    config = TrOCRDatasetConfig(
        model_name="microsoft/trocr-base-stage1",
        enable_augmentation=True,
        max_target_length=128
    )
    
    # Создаем препаратор
    preparator = TrOCRDatasetPreparator(config)
    
    # Пример создания синтетического датасета
    try:
        datasets = preparator.prepare_synthetic_dataset(
            output_path="data/trocr_synthetic_dataset",
            num_samples=100  # Маленький пример
        )
        
        print("✅ Созданы датасеты:")
        for split_name, split_path in datasets.items():
            print(f"  📁 {split_name}: {split_path}")
    except Exception as e:
        print(f"❌ Ошибка: {e}") 
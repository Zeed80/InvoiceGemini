"""
Универсальный парсер датасетов для TrOCR обучения
Поддерживает множество форматов: COCO, YOLO, PASCAL VOC, JSON, CSV и другие
"""

import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from PIL import Image
import yaml


class DatasetFormat(Enum):
    """Поддерживаемые форматы датасетов"""
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    JSON_SIMPLE = "json_simple"
    CSV = "csv"
    LABELME = "labelme"
    CUSTOM_JSON = "custom_json"
    FOLDER_STRUCTURE = "folder_structure"


@dataclass
class AnnotationData:
    """Структура аннотации текста"""
    image_path: str
    text: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    confidence: float = 1.0
    language: str = "ru"
    category: str = "invoice_text"
    

@dataclass
class DatasetInfo:
    """Информация о датасете"""
    format: DatasetFormat
    total_images: int
    total_annotations: int
    languages: List[str]
    categories: List[str]
    image_formats: List[str]
    path: str


class UniversalDatasetParser:
    """Универсальный парсер для различных форматов датасетов"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.supported_formats = list(DatasetFormat)
        
    def detect_format(self, dataset_path: str) -> DatasetFormat:
        """Автоматическое определение формата датасета"""
        dataset_path = Path(dataset_path)
        
        # Проверяем COCO формат
        if self._is_coco_format(dataset_path):
            return DatasetFormat.COCO
            
        # Проверяем YOLO формат
        if self._is_yolo_format(dataset_path):
            return DatasetFormat.YOLO
            
        # Проверяем PASCAL VOC формат
        if self._is_pascal_voc_format(dataset_path):
            return DatasetFormat.PASCAL_VOC
            
        # Проверяем LabelMe формат
        if self._is_labelme_format(dataset_path):
            return DatasetFormat.LABELME
            
        # Проверяем CSV формат
        if self._is_csv_format(dataset_path):
            return DatasetFormat.CSV
            
        # Проверяем структуру папок
        if self._is_folder_structure(dataset_path):
            return DatasetFormat.FOLDER_STRUCTURE
            
        # По умолчанию JSON
        return DatasetFormat.JSON_SIMPLE
        
    def parse_dataset(self, dataset_path: str, 
                     format_type: Optional[DatasetFormat] = None) -> List[AnnotationData]:
        """Парсинг датасета в универсальный формат"""
        
        if format_type is None:
            format_type = self.detect_format(dataset_path)
            
        self.logger.info(f"Парсинг датасета в формате {format_type.value} из {dataset_path}")
        
        try:
            if format_type == DatasetFormat.COCO:
                return self._parse_coco(dataset_path)
            elif format_type == DatasetFormat.YOLO:
                return self._parse_yolo(dataset_path)
            elif format_type == DatasetFormat.PASCAL_VOC:
                return self._parse_pascal_voc(dataset_path)
            elif format_type == DatasetFormat.LABELME:
                return self._parse_labelme(dataset_path)
            elif format_type == DatasetFormat.CSV:
                return self._parse_csv(dataset_path)
            elif format_type == DatasetFormat.FOLDER_STRUCTURE:
                return self._parse_folder_structure(dataset_path)
            else:
                return self._parse_json_simple(dataset_path)
                
        except Exception as e:
            self.logger.error(f"Ошибка парсинга датасета: {e}")
            return []
    
    def get_dataset_info(self, dataset_path: str, 
                        format_type: Optional[DatasetFormat] = None) -> DatasetInfo:
        """Получение информации о датасете"""
        
        if format_type is None:
            format_type = self.detect_format(dataset_path)
            
        annotations = self.parse_dataset(dataset_path, format_type)
        
        # Анализируем аннотации
        image_paths = list(set([ann.image_path for ann in annotations]))
        languages = list(set([ann.language for ann in annotations]))
        categories = list(set([ann.category for ann in annotations]))
        
        # Определяем форматы изображений
        image_formats = []
        for img_path in image_paths[:10]:  # Проверяем первые 10
            if os.path.exists(img_path):
                ext = Path(img_path).suffix.lower()
                if ext not in image_formats:
                    image_formats.append(ext)
        
        return DatasetInfo(
            format=format_type,
            total_images=len(image_paths),
            total_annotations=len(annotations),
            languages=languages,
            categories=categories,
            image_formats=image_formats,
            path=str(dataset_path)
        )
    
    # ==========================================
    # Детекторы форматов
    # ==========================================
    
    def _is_coco_format(self, dataset_path: Path) -> bool:
        """Проверка COCO формата"""
        # Ищем annotations.json или instances_*.json
        for json_file in dataset_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if all(key in data for key in ['images', 'annotations', 'categories']):
                        return True
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logging.debug(f"Файл {file} не является COCO форматом: {e}")
                continue
        return False
        
    def _is_yolo_format(self, dataset_path: Path) -> bool:
        """Проверка YOLO формата"""
        # Ищем .yaml конфиг и .txt файлы аннотаций
        yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))
        txt_files = list(dataset_path.glob("**/*.txt"))
        
        return len(yaml_files) > 0 and len(txt_files) > 0
        
    def _is_pascal_voc_format(self, dataset_path: Path) -> bool:
        """Проверка PASCAL VOC формата"""
        # Ищем XML аннотации
        xml_files = list(dataset_path.glob("**/*.xml"))
        return len(xml_files) > 0
        
    def _is_labelme_format(self, dataset_path: Path) -> bool:
        """Проверка LabelMe формата"""
        # Ищем JSON с полями shapes, imagePath
        for json_file in dataset_path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shapes' in data and 'imagePath' in data:
                        return True
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logging.debug(f"Файл {file} не является LabelMe форматом: {e}")
                continue
        return False
        
    def _is_csv_format(self, dataset_path: Path) -> bool:
        """Проверка CSV формата"""
        csv_files = list(dataset_path.glob("*.csv"))
        return len(csv_files) > 0
        
    def _is_folder_structure(self, dataset_path: Path) -> bool:
        """Проверка структуры папок"""
        # Ищем папки с изображениями
        image_dirs = []
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                if len(images) > 0:
                    image_dirs.append(subdir)
        return len(image_dirs) > 0
    
    # ==========================================
    # Парсеры по форматам
    # ==========================================
    
    def _parse_coco(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг COCO датасета"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        # Ищем главный JSON файл
        json_files = list(dataset_path.glob("*.json"))
        if not json_files:
            json_files = list(dataset_path.glob("annotations/*.json"))
            
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if not all(key in data for key in ['images', 'annotations']):
                    continue
                    
                # Создаем маппинг изображений
                image_map = {img['id']: img for img in data['images']}
                category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                
                for ann in data['annotations']:
                    if ann['image_id'] not in image_map:
                        continue
                        
                    image_info = image_map[ann['image_id']]
                    image_path = str(dataset_path / "images" / image_info['file_name'])
                    
                    # Извлекаем текст из аннотации
                    text = ""
                    if 'caption' in ann:
                        text = ann['caption']
                    elif 'text' in ann:
                        text = ann['text']
                    elif 'description' in ann:
                        text = ann['description']
                    
                    if not text:
                        continue
                        
                    # Bbox в формате COCO: [x, y, width, height]
                    bbox = None
                    if 'bbox' in ann:
                        bbox = tuple(ann['bbox'])
                        
                    category = category_map.get(ann.get('category_id', 0), 'text')
                    
                    annotations.append(AnnotationData(
                        image_path=image_path,
                        text=text,
                        bbox=bbox,
                        confidence=ann.get('score', 1.0),
                        category=category
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга COCO файла {json_file}: {e}")
                continue
                
        return annotations
    
    def _parse_yolo(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг YOLO датасета"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        # Ищем конфигурацию
        yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))
        config = {}
        
        if yaml_files:
            try:
                with open(yaml_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except (yaml.YAMLError, IOError) as e:
                logging.debug(f"Не удалось прочитать YAML конфиг: {e}")
        
        # Ищем изображения и аннотации
        for img_file in dataset_path.glob("**/*.jpg"):
            txt_file = img_file.with_suffix('.txt')
            if not txt_file.exists():
                continue
                
            try:
                # Читаем размеры изображения
                img = Image.open(img_file)
                img_width, img_height = img.size
                
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Конвертируем YOLO координаты в абсолютные
                        x = int((x_center - width/2) * img_width)
                        y = int((y_center - height/2) * img_height)
                        w = int(width * img_width)
                        h = int(height * img_height)
                        
                        # Текст может быть в остальных частях строки
                        text = " ".join(parts[5:]) if len(parts) > 5 else f"text_{class_id}"
                        
                        annotations.append(AnnotationData(
                            image_path=str(img_file),
                            text=text,
                            bbox=(x, y, w, h),
                            category=f"class_{class_id}"
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга YOLO файла {img_file}: {e}")
                continue
                
        return annotations
    
    def _parse_pascal_voc(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг PASCAL VOC датасета"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        for xml_file in dataset_path.glob("**/*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Получаем путь к изображению
                filename = root.find('filename')
                if filename is None:
                    continue
                    
                image_path = str(dataset_path / "images" / filename.text)
                if not os.path.exists(image_path):
                    image_path = str(xml_file.parent / filename.text)
                
                # Парсим объекты
                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is None:
                        continue
                        
                    # Ищем текст в различных полях
                    text = ""
                    text_elem = obj.find('text')
                    if text_elem is not None:
                        text = text_elem.text
                    else:
                        text = name.text
                    
                    # Bbox координаты
                    bbox_elem = obj.find('bndbox')
                    bbox = None
                    if bbox_elem is not None:
                        xmin = int(bbox_elem.find('xmin').text)
                        ymin = int(bbox_elem.find('ymin').text)
                        xmax = int(bbox_elem.find('xmax').text)
                        ymax = int(bbox_elem.find('ymax').text)
                        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    
                    annotations.append(AnnotationData(
                        image_path=image_path,
                        text=text,
                        bbox=bbox,
                        category=name.text
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга XML файла {xml_file}: {e}")
                continue
                
        return annotations
    
    def _parse_labelme(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг LabelMe датасета"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        for json_file in dataset_path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'shapes' not in data or 'imagePath' not in data:
                    continue
                    
                image_path = str(dataset_path / data['imagePath'])
                if not os.path.exists(image_path):
                    image_path = str(json_file.parent / data['imagePath'])
                
                for shape in data['shapes']:
                    label = shape.get('label', 'text')
                    points = shape.get('points', [])
                    
                    # Рассчитываем bbox из точек
                    bbox = None
                    if points:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        bbox = (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))
                    
                    # Ищем текст в атрибутах
                    text = shape.get('text', label)
                    if 'attributes' in shape:
                        text = shape['attributes'].get('text', text)
                    
                    annotations.append(AnnotationData(
                        image_path=image_path,
                        text=text,
                        bbox=bbox,
                        category=label
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга LabelMe файла {json_file}: {e}")
                continue
                
        return annotations
    
    def _parse_csv(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг CSV датасета"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        for csv_file in dataset_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Стандартизируем названия колонок
                df.columns = df.columns.str.lower()
                
                for _, row in df.iterrows():
                    # Пытаемся найти нужные колонки
                    image_path = ""
                    for col in ['image_path', 'image', 'filename', 'file']:
                        if col in df.columns:
                            image_path = str(dataset_path / str(row[col]))
                            break
                    
                    text = ""
                    for col in ['text', 'caption', 'label', 'description']:
                        if col in df.columns:
                            text = str(row[col])
                            break
                    
                    if not image_path or not text:
                        continue
                    
                    # Ищем bbox
                    bbox = None
                    bbox_cols = ['x', 'y', 'width', 'height']
                    if all(col in df.columns for col in bbox_cols):
                        bbox = (int(row['x']), int(row['y']), 
                               int(row['width']), int(row['height']))
                    
                    category = str(row.get('category', 'text'))
                    confidence = float(row.get('confidence', 1.0))
                    
                    annotations.append(AnnotationData(
                        image_path=image_path,
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        category=category
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга CSV файла {csv_file}: {e}")
                continue
                
        return annotations
    
    def _parse_folder_structure(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг структуры папок"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        for subdir in dataset_path.iterdir():
            if not subdir.is_dir():
                continue
                
            category = subdir.name
            
            for img_file in subdir.glob("*.jpg"):
                # Ищем соответствующий текстовый файл
                txt_file = img_file.with_suffix('.txt')
                text = category  # По умолчанию название папки
                
                if txt_file.exists():
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    except (IOError, UnicodeDecodeError) as e:
                        logging.debug(f"Не удалось прочитать файл {txt_file}: {e}")
                
                annotations.append(AnnotationData(
                    image_path=str(img_file),
                    text=text,
                    category=category
                ))
                
        return annotations
    
    def _parse_json_simple(self, dataset_path: str) -> List[AnnotationData]:
        """Парсинг простого JSON формата"""
        dataset_path = Path(dataset_path)
        annotations = []
        
        for json_file in dataset_path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Если это массив аннотаций
                if isinstance(data, list):
                    items = data
                elif 'annotations' in data:
                    items = data['annotations']
                elif 'data' in data:
                    items = data['data']
                else:
                    items = [data]
                
                for item in items:
                    # Извлекаем основные поля
                    image_path = item.get('image_path', item.get('image', ''))
                    text = item.get('text', item.get('caption', item.get('label', '')))
                    
                    if not image_path or not text:
                        continue
                    
                    # Абсолютный путь
                    if not os.path.isabs(image_path):
                        image_path = str(dataset_path / image_path)
                    
                    # Bbox
                    bbox = None
                    if 'bbox' in item:
                        bbox = tuple(item['bbox'])
                    elif all(k in item for k in ['x', 'y', 'width', 'height']):
                        bbox = (item['x'], item['y'], item['width'], item['height'])
                    
                    annotations.append(AnnotationData(
                        image_path=image_path,
                        text=text,
                        bbox=bbox,
                        confidence=item.get('confidence', 1.0),
                        category=item.get('category', 'text'),
                        language=item.get('language', 'ru')
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга JSON файла {json_file}: {e}")
                continue
                
        return annotations
    
    def convert_to_trocr_format(self, annotations: List[AnnotationData], 
                               output_path: str) -> str:
        """Конвертация в формат TrOCR"""
        trocr_data = []
        
        for ann in annotations:
            if not os.path.exists(ann.image_path):
                continue
                
            trocr_item = {
                "image_path": ann.image_path,
                "text": ann.text
            }
            
            if ann.bbox:
                trocr_item["bbox"] = {
                    "x": ann.bbox[0],
                    "y": ann.bbox[1], 
                    "width": ann.bbox[2],
                    "height": ann.bbox[3]
                }
            
            trocr_item.update({
                "confidence": ann.confidence,
                "category": ann.category,
                "language": ann.language
            })
            
            trocr_data.append(trocr_item)
        
        # Сохраняем в формате TrOCR
        output_file = Path(output_path) / "annotations.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trocr_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Конвертировано {len(trocr_data)} аннотаций в {output_file}")
        return str(output_file) 
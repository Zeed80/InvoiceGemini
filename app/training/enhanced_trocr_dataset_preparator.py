#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced TrOCR Dataset Preparator с автоматической разметкой через LLM Gemini

Полностью автоматизированная система создания высококачественных датасетов TrOCR
с использованием современных LLM для интеллектуальной разметки данных.

Основано на:
- Microsoft TrOCR: https://huggingface.co/docs/transformers/model_doc/trocr
- Gemini Vision API: https://ai.google.dev/docs/vision
- Лучших практиках аннотации данных: https://medium.com/@mshayan38/data-annotation-using-modern-llms-gemini-82f8823a6f12
"""

import os
import json
import logging
import tempfile
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import random
import string

# Импорты для transformers
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torchvision.transforms as transforms
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Импорты для Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Импорты из основного проекта
from .trocr_dataset_preparator import TrOCRDatasetPreparator, TrOCRDatasetConfig, TrOCRCustomDataset


@dataclass
class EnhancedTrOCRConfig(TrOCRDatasetConfig):
    """Расширенная конфигурация с поддержкой LLM автоматизации"""
    
    # LLM настройки
    enable_llm_annotation: bool = True
    llm_provider: str = "gemini"  # gemini, openai, anthropic
    llm_model: str = "models/gemini-2.0-flash-exp"
    max_llm_requests_per_minute: int = 60
    llm_confidence_threshold: float = 0.8
    
    # Настройки для OCR разметки
    ocr_languages: List[str] = field(default_factory=lambda: ["ru", "en"])
    extract_bounding_boxes: bool = True
    min_word_confidence: float = 0.5
    use_layout_analysis: bool = True
    
    # Настройки качества
    enable_quality_filter: bool = True
    min_text_length_chars: int = 3
    max_text_length_chars: int = 500
    filter_non_text_images: bool = True
    
    # Синтетическая генерация
    synthetic_templates: List[str] = field(default_factory=lambda: [
        "invoice", "receipt", "document", "form", "table"
    ])
    synthetic_fonts: List[str] = field(default_factory=lambda: [
        "arial.ttf", "times.ttf", "courier.ttf"
    ])
    synthetic_backgrounds: List[str] = field(default_factory=lambda: [
        "white", "light_gray", "cream", "light_blue"
    ])
    
    # Продвинутая аугментация
    enable_document_layout_augmentation: bool = True
    enable_text_style_variation: bool = True
    enable_background_texture: bool = True


class LLMAnnotationEngine:
    """Движок для автоматической аннотации с помощью LLM"""
    
    def __init__(self, config: EnhancedTrOCRConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.request_count = 0
        self.last_request_time = 0
        
        self._setup_llm_client()
    
    def _setup_llm_client(self):
        """Настройка клиента LLM"""
        if self.config.llm_provider == "gemini" and GEMINI_AVAILABLE:
            try:
                # Получаем API ключ
                from app.settings_manager import settings_manager
                api_key = settings_manager.get_gemini_api_key()
                
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(self.config.llm_model)
                    self.logger.info(f"✅ Gemini клиент настроен: {self.config.llm_model}")
                else:
                    self.logger.error("❌ API ключ Gemini не найден")
            except Exception as e:
                self.logger.error(f"❌ Ошибка настройки Gemini: {e}")
        else:
            self.logger.warning("⚠️ LLM провайдер недоступен, используется fallback")
    
    def _rate_limit_check(self):
        """Проверка лимитов запросов"""
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            if self.request_count >= self.config.max_llm_requests_per_minute:
                sleep_time = 60 - (current_time - self.last_request_time)
                self.logger.info(f"⏱️ Достигнут лимит запросов, ожидаем {sleep_time:.1f}с")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
            self.last_request_time = current_time
    
    def annotate_image_for_ocr(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Создает OCR аннотации для изображения с помощью LLM
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Dict с OCR данными или None
        """
        if not self.client:
            return None
        
        try:
            self._rate_limit_check()
            self.request_count += 1
            
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Создаем специализированный промпт для OCR
            prompt = self._create_ocr_annotation_prompt()
            
            # Настройки генерации для OCR
            generation_config = {
                "temperature": 0.1,  # Низкая температура для точности
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            }
            
            # Отправляем запрос
            response = self.client.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            if response and response.text:
                # Парсим JSON ответ
                try:
                    ocr_data = json.loads(response.text)
                    return self._validate_ocr_annotation(ocr_data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ Ошибка парсинга JSON от LLM: {e}")
                    return self._extract_text_fallback(response.text)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка аннотации изображения {image_path}: {e}")
        
        return None
    
    def _create_ocr_annotation_prompt(self) -> str:
        """Создает специализированный промпт для OCR аннотации"""
        return """
Ты - эксперт по оптическому распознаванию текста (OCR). Проанализируй изображение и извлеки весь видимый текст.

Верни результат в следующем JSON формате:
{
    "text_blocks": [
        {
            "text": "извлеченный текст",
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2],
            "language": "ru",
            "font_size": "medium",
            "text_type": "paragraph"
        }
    ],
    "full_text": "весь текст документа",
    "document_type": "invoice|receipt|document|form|other",
    "layout_structure": {
        "has_table": true,
        "has_header": true,
        "has_footer": false,
        "columns": 1
    },
    "quality_metrics": {
        "image_quality": "high|medium|low",
        "text_clarity": "clear|blurry|unclear",
        "overall_confidence": 0.85
    }
}

ТРЕБОВАНИЯ:
1. Извлекай ВСЕ видимый текст, включая числа, символы, знаки препинания
2. Сохраняй оригинальную орфографию и формат
3. Указывай приблизительные координаты bbox в пикселях
4. Оценивай confidence от 0.0 до 1.0
5. Определяй язык текста (ru, en, mix)
6. Классифицируй тип документа
7. НЕ переводи и НЕ исправляй текст
8. Для плохо видимого текста указывай низкий confidence

Верни ТОЛЬКО JSON, без дополнительного текста.
"""
    
    def _validate_ocr_annotation(self, ocr_data: Dict) -> Optional[Dict]:
        """Валидирует и очищает OCR аннотацию"""
        try:
            if not isinstance(ocr_data, dict):
                return None
            
            # Проверяем обязательные поля
            if "text_blocks" not in ocr_data or "full_text" not in ocr_data:
                return None
            
            # Фильтруем блоки по confidence
            filtered_blocks = []
            for block in ocr_data.get("text_blocks", []):
                if (isinstance(block, dict) and 
                    "confidence" in block and 
                    block["confidence"] >= self.config.min_word_confidence):
                    filtered_blocks.append(block)
            
            ocr_data["text_blocks"] = filtered_blocks
            
            # Проверяем качество
            overall_confidence = ocr_data.get("quality_metrics", {}).get("overall_confidence", 0.0)
            if overall_confidence < self.config.llm_confidence_threshold:
                self.logger.warning(f"⚠️ Низкий общий confidence: {overall_confidence}")
            
            return ocr_data
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации OCR: {e}")
            return None
    
    def _extract_text_fallback(self, response_text: str) -> Optional[Dict]:
        """Fallback извлечение текста из ответа LLM"""
        try:
            # Простое извлечение текста если JSON не получился
            return {
                "text_blocks": [
                    {
                        "text": response_text,
                        "confidence": 0.5,
                        "bbox": [0, 0, 100, 100],
                        "language": "mixed",
                        "font_size": "medium",
                        "text_type": "paragraph"
                    }
                ],
                "full_text": response_text,
                "document_type": "other",
                "layout_structure": {"columns": 1},
                "quality_metrics": {"overall_confidence": 0.5}
            }
        except:
            return None
    
    def enhance_existing_annotation(self, annotation: Dict, image_path: str) -> Dict:
        """Улучшает существующую аннотацию с помощью LLM"""
        if not self.client:
            return annotation
        
        try:
            self._rate_limit_check()
            
            # Создаем промпт для улучшения
            prompt = f"""
Улучши существующую OCR аннотацию. Исходная аннотация:
{json.dumps(annotation, ensure_ascii=False, indent=2)}

Проанализируй изображение и:
1. Исправь ошибки в тексте
2. Добавь пропущенный текст
3. Улучши bbox координаты
4. Повысь точность confidence
5. Дополни метаданные

Верни улучшенную аннотацию в том же JSON формате.
"""
            
            image = Image.open(image_path).convert('RGB')
            response = self.client.generate_content([prompt, image])
            
            if response and response.text:
                try:
                    enhanced = json.loads(response.text)
                    return enhanced if enhanced else annotation
                except:
                    return annotation
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка улучшения аннотации: {e}")
        
        return annotation


class SyntheticDataGenerator:
    """Генератор синтетических данных для TrOCR"""
    
    def __init__(self, config: EnhancedTrOCRConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Шаблоны текстов для разных типов документов
        self.text_templates = {
            "invoice": [
                "ООО «{company}»", "Счет-фактура №{number}", 
                "от {date}", "Сумма: {amount} руб.",
                "НДС 20%: {tax} руб.", "Итого: {total} руб."
            ],
            "receipt": [
                "Чек №{number}", "Дата: {date}",
                "Товар: {item}", "Цена: {price}",
                "Итого: {total}"
            ],
            "document": [
                "Документ №{number}", "Дата создания: {date}",
                "Содержание документа", "Подпись: {signature}"
            ]
        }
        
        # Примеры данных для подстановки
        self.sample_data = {
            "companies": ["ООО Рога и копыта", "ИП Иванов И.И.", "ЗАО Светлое будущее"],
            "items": ["Товар 1", "Услуга консультации", "Оборудование"],
            "names": ["Иванов", "Петров", "Сидоров"],
            "numbers": lambda: str(random.randint(1, 9999)),
            "amounts": lambda: f"{random.randint(100, 50000)},00"
        }
    
    def generate_synthetic_dataset(self, 
                                 output_dir: Path, 
                                 num_samples: int = 1000,
                                 progress_callback: Optional[Callable] = None) -> List[Tuple[str, Dict]]:
        """
        Генерирует синтетический датасет
        
        Args:
            output_dir: Директория для сохранения
            num_samples: Количество примеров
            progress_callback: Callback для отображения прогресса
            
        Returns:
            Список пар (путь_к_изображению, аннотация)
        """
        self.logger.info(f"🎨 Генерация {num_samples} синтетических примеров")
        
        synthetic_pairs = []
        images_dir = output_dir / "synthetic_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(num_samples), desc="Генерация синтетических данных"):
            try:
                # Выбираем тип документа
                doc_type = random.choice(self.config.synthetic_templates)
                
                # Генерируем текст
                text_content = self._generate_text_for_type(doc_type)
                
                # Создаем изображение
                image_path = images_dir / f"synthetic_{i+1:06d}.png"
                image = self._create_synthetic_image(text_content, doc_type)
                image.save(image_path, "PNG", quality=95)
                
                # Создаем аннотацию
                annotation = self._create_synthetic_annotation(text_content, image.size)
                
                synthetic_pairs.append((str(image_path), annotation))
                
                # Обновляем прогресс
                if progress_callback:
                    progress_callback(int((i + 1) / num_samples * 100))
                    
            except Exception as e:
                self.logger.error(f"❌ Ошибка генерации примера {i}: {e}")
                continue
        
        self.logger.info(f"✅ Сгенерировано {len(synthetic_pairs)} синтетических примеров")
        return synthetic_pairs
    
    def _generate_text_for_type(self, doc_type: str) -> str:
        """Генерирует текст для определенного типа документа"""
        templates = self.text_templates.get(doc_type, self.text_templates["document"])
        
        # Заполняем шаблоны случайными данными
        filled_templates = []
        for template in templates:
            try:
                filled = template.format(
                    company=random.choice(self.sample_data["companies"]),
                    number=self.sample_data["numbers"](),
                    date=f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.2024",
                    amount=self.sample_data["amounts"](),
                    tax=self.sample_data["amounts"](),
                    total=self.sample_data["amounts"](),
                    item=random.choice(self.sample_data["items"]),
                    price=self.sample_data["amounts"](),
                    signature=random.choice(self.sample_data["names"])
                )
                filled_templates.append(filled)
            except KeyError:
                # Если шаблон не требует подстановки
                filled_templates.append(template)
        
        return "\n".join(filled_templates)
    
    def _create_synthetic_image(self, text: str, doc_type: str) -> Image.Image:
        """Создает синтетическое изображение с текстом"""
        # Размеры изображения
        width, height = self.config.image_size
        
        # Выбираем цвет фона
        bg_color = random.choice(["white", "#f8f9fa", "#f1f3f4", "#e8eaed"])
        
        # Создаем изображение
        image = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        try:
            # Пытаемся загрузить шрифт
            font_size = random.randint(12, 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback на default шрифт
            font = ImageFont.load_default()
        
        # Цвет текста
        text_color = random.choice(["black", "#333333", "#1a1a1a"])
        
        # Размещаем текст
        lines = text.split('\n')
        y_offset = random.randint(20, 50)
        line_height = font_size + 5
        
        for line in lines:
            if line.strip():
                x_offset = random.randint(20, 60)
                draw.text((x_offset, y_offset), line, fill=text_color, font=font)
                y_offset += line_height
        
        # Добавляем немного шума для реалистичности
        if random.random() < 0.3:
            image = self._add_image_noise(image)
        
        return image
    
    def _add_image_noise(self, image: Image.Image) -> Image.Image:
        """Добавляет легкий шум к изображению"""
        try:
            # Небольшое размытие
            if random.random() < 0.5:
                from PIL import ImageFilter
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Изменение яркости
            if random.random() < 0.3:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.9, 1.1))
            
            # Изменение контраста  
            if random.random() < 0.3:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.95, 1.05))
        except:
            pass
        
        return image
    
    def _create_synthetic_annotation(self, text: str, image_size: Tuple[int, int]) -> Dict:
        """Создает аннотацию для синтетического изображения"""
        width, height = image_size
        
        # Разбиваем текст на блоки
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        text_blocks = []
        y_offset = 30
        line_height = 25
        
        for line in lines:
            # Создаем bbox для строки
            x1 = 30
            y1 = y_offset
            x2 = min(width - 30, x1 + len(line) * 8)  # Примерная ширина
            y2 = y1 + line_height
            
            text_blocks.append({
                "text": line,
                "confidence": random.uniform(0.85, 0.98),
                "bbox": [x1, y1, x2, y2],
                "language": "ru",
                "font_size": "medium",
                "text_type": "line"
            })
            
            y_offset += line_height
        
        return {
            "text_blocks": text_blocks,
            "full_text": text,
            "document_type": "synthetic",
            "layout_structure": {
                "has_table": False,
                "has_header": True,
                "has_footer": False,
                "columns": 1
            },
            "quality_metrics": {
                "image_quality": "high",
                "text_clarity": "clear",
                "overall_confidence": 0.95
            },
            "is_synthetic": True
        }


class EnhancedTrOCRDatasetPreparator(TrOCRDatasetPreparator):
    """
    Улучшенный подготовщик датасетов TrOCR с автоматической LLM аннотацией
    """
    
    def __init__(self, config: Optional[EnhancedTrOCRConfig] = None):
        # Инициализируем базовую конфигурацию
        base_config = config or EnhancedTrOCRConfig()
        super().__init__(base_config)
        
        # Расширенная конфигурация
        self.enhanced_config = base_config
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем LLM движок
        self.llm_engine = None
        if self.enhanced_config.enable_llm_annotation:
            self.llm_engine = LLMAnnotationEngine(self.enhanced_config, self.logger)
        
        # Генератор синтетических данных
        self.synthetic_generator = SyntheticDataGenerator(self.enhanced_config, self.logger)
        
        self.logger.info("🚀 Enhanced TrOCR Dataset Preparator инициализирован")
    
    def prepare_fully_automated_dataset(self,
                                      source_images: List[str],
                                      output_path: str,
                                      num_synthetic: int = 500,
                                      progress_callback: Optional[Callable] = None) -> Dict[str, str]:
        """
        Полностью автоматизированная подготовка датасета
        
        Args:
            source_images: Список путей к исходным изображениям
            output_path: Путь для сохранения датасета  
            num_synthetic: Количество синтетических примеров
            progress_callback: Callback для прогресса
            
        Returns:
            Dict с путями к созданным датасетам
        """
        self.logger.info(f"🤖 Запуск полностью автоматизированной подготовки датасета")
        self.logger.info(f"📁 Исходные изображения: {len(source_images)}")
        self.logger.info(f"🎨 Синтетические примеры: {num_synthetic}")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data_pairs = []
        total_steps = len(source_images) + num_synthetic
        current_step = 0
        
        # Этап 1: Обработка исходных изображений с LLM аннотацией
        if source_images:
            self.logger.info("📝 Этап 1: Автоматическая аннотация исходных изображений")
            
            for i, image_path in enumerate(source_images):
                try:
                    if progress_callback:
                        progress = int(current_step / total_steps * 100)
                        progress_callback(progress)
                    
                    # Создаем аннотацию через LLM
                    annotation = self._create_automated_annotation(image_path)
                    
                    if annotation and self._validate_annotation_quality(annotation):
                        all_data_pairs.append((image_path, annotation))
                        self.logger.debug(f"✅ Обработано: {Path(image_path).name}")
                    else:
                        self.logger.warning(f"⚠️ Пропущено (низкое качество): {Path(image_path).name}")
                    
                    current_step += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка обработки {image_path}: {e}")
                    current_step += 1
                    continue
        
        # Этап 2: Генерация синтетических данных
        if num_synthetic > 0:
            self.logger.info("🎨 Этап 2: Генерация синтетических данных")
            
            def synthetic_progress(progress):
                nonlocal current_step
                synthetic_step = current_step + int(progress / 100 * num_synthetic)
                if progress_callback:
                    progress_callback(int(synthetic_step / total_steps * 100))
            
            synthetic_pairs = self.synthetic_generator.generate_synthetic_dataset(
                output_dir, num_synthetic, synthetic_progress
            )
            all_data_pairs.extend(synthetic_pairs)
            current_step = total_steps
        
        # Этап 3: Создание итогового датасета
        self.logger.info("🏗️ Этап 3: Создание итогового датасета")
        
        if progress_callback:
            progress_callback(95)
        
        # Конвертируем в формат для базового класса
        converted_pairs = self._convert_annotations_to_trocr_format(all_data_pairs)
        
        # Разделяем данные
        train_pairs, val_pairs, test_pairs = self._split_data(converted_pairs, 0.7, 0.15, 0.15)
        
        # Создаем датасеты
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
        
        # Сохраняем расширенные метаданные
        self._save_enhanced_metadata(output_dir, datasets, all_data_pairs)
        
        if progress_callback:
            progress_callback(100)
        
        self.logger.info(f"🎉 Автоматизированный датасет готов: {output_path}")
        self.logger.info(f"📊 Всего примеров: {len(converted_pairs)}")
        self.logger.info(f"🏋️ Тренировочных: {len(train_pairs)}")
        self.logger.info(f"✅ Валидационных: {len(val_pairs)}")
        self.logger.info(f"🧪 Тестовых: {len(test_pairs)}")
        
        return datasets
    
    def _create_automated_annotation(self, image_path: str) -> Optional[Dict]:
        """Создает автоматическую аннотацию для изображения"""
        if not self.llm_engine:
            self.logger.warning("⚠️ LLM движок недоступен, используется fallback")
            return self._create_fallback_annotation(image_path)
        
        try:
            # Используем LLM для аннотации
            annotation = self.llm_engine.annotate_image_for_ocr(image_path)
            
            if annotation:
                self.logger.debug(f"✅ LLM аннотация создана для {Path(image_path).name}")
                return annotation
            else:
                self.logger.warning(f"⚠️ LLM не смог создать аннотацию для {Path(image_path).name}")
                return self._create_fallback_annotation(image_path)
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка LLM аннотации {image_path}: {e}")
            return self._create_fallback_annotation(image_path)
    
    def _create_fallback_annotation(self, image_path: str) -> Dict:
        """Создает базовую аннотацию без LLM"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Базовая аннотация с placeholder текстом
            return {
                "text_blocks": [
                    {
                        "text": f"Document content from {Path(image_path).name}",
                        "confidence": 0.5,
                        "bbox": [0, 0, width, height],
                        "language": "en",
                        "font_size": "medium",
                        "text_type": "document"
                    }
                ],
                "full_text": f"Document content from {Path(image_path).name}",
                "document_type": "document",
                "layout_structure": {"columns": 1},
                "quality_metrics": {"overall_confidence": 0.5},
                "is_fallback": True
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания fallback аннотации: {e}")
            return None
    
    def _validate_annotation_quality(self, annotation: Dict) -> bool:
        """Проверяет качество аннотации"""
        if not annotation:
            return False
        
        try:
            # Проверяем наличие текста
            full_text = annotation.get("full_text", "")
            if len(full_text) < self.enhanced_config.min_text_length_chars:
                return False
            
            if len(full_text) > self.enhanced_config.max_text_length_chars:
                return False
            
            # Проверяем общий confidence
            overall_confidence = annotation.get("quality_metrics", {}).get("overall_confidence", 0.0)
            if overall_confidence < self.enhanced_config.llm_confidence_threshold:
                return False
            
            # Проверяем наличие текстовых блоков
            text_blocks = annotation.get("text_blocks", [])
            if len(text_blocks) == 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации аннотации: {e}")
            return False
    
    def _convert_annotations_to_trocr_format(self, data_pairs: List[Tuple[str, Dict]]) -> List[Tuple[str, str]]:
        """Конвертирует расширенные аннотации в формат TrOCR (изображение, текст)"""
        converted_pairs = []
        
        for image_path, annotation in data_pairs:
            try:
                # Извлекаем полный текст
                full_text = annotation.get("full_text", "")
                
                if full_text.strip():
                    converted_pairs.append((image_path, full_text.strip()))
                    
            except Exception as e:
                self.logger.error(f"❌ Ошибка конвертации аннотации для {image_path}: {e}")
                continue
        
        return converted_pairs
    
    def _save_enhanced_metadata(self, output_dir: Path, datasets: Dict, all_pairs: List):
        """Сохраняет расширенные метаданные датасета"""
        try:
            # Базовые метаданные
            super()._save_dataset_metadata(output_dir, datasets, 
                                         self._convert_annotations_to_trocr_format(all_pairs))
            
            # Расширенные метаданные
            enhanced_metadata = {
                "dataset_type": "enhanced_trocr",
                "creation_method": "llm_automated",
                "llm_provider": self.enhanced_config.llm_provider,
                "llm_model": self.enhanced_config.llm_model,
                "config": {
                    "enable_llm_annotation": self.enhanced_config.enable_llm_annotation,
                    "llm_confidence_threshold": self.enhanced_config.llm_confidence_threshold,
                    "synthetic_templates": self.enhanced_config.synthetic_templates,
                    "image_size": self.enhanced_config.image_size
                },
                "quality_stats": self._calculate_quality_stats(all_pairs),
                "annotation_sources": self._analyze_annotation_sources(all_pairs)
            }
            
            # Сохраняем расширенные метаданные
            enhanced_file = output_dir / "enhanced_metadata.json"
            
            # Конвертируем numpy типы в native Python типы для JSON
            def convert_numpy_types(obj):
                import numpy as np
                if hasattr(obj, 'item'):  # numpy скаляры
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            enhanced_metadata = convert_numpy_types(enhanced_metadata)
            
            with open(enhanced_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Расширенные метаданные сохранены: {enhanced_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения расширенных метаданных: {e}")
    
    def _calculate_quality_stats(self, data_pairs: List[Tuple[str, Dict]]) -> Dict:
        """Вычисляет статистики качества датасета"""
        try:
            confidences = []
            text_lengths = []
            languages = []
            
            for _, annotation in data_pairs:
                # Confidence
                overall_conf = annotation.get("quality_metrics", {}).get("overall_confidence", 0.0)
                confidences.append(overall_conf)
                
                # Длина текста
                text_len = len(annotation.get("full_text", ""))
                text_lengths.append(text_len)
                
                # Языки
                for block in annotation.get("text_blocks", []):
                    lang = block.get("language", "unknown")
                    languages.append(lang)
            
            return {
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "min_confidence": np.min(confidences) if confidences else 0.0,
                "max_confidence": np.max(confidences) if confidences else 0.0,
                "avg_text_length": np.mean(text_lengths) if text_lengths else 0.0,
                "min_text_length": np.min(text_lengths) if text_lengths else 0.0,
                "max_text_length": np.max(text_lengths) if text_lengths else 0.0,
                "language_distribution": {lang: languages.count(lang) for lang in set(languages)}
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка вычисления статистик качества: {e}")
            return {}
    
    def _analyze_annotation_sources(self, data_pairs: List[Tuple[str, Dict]]) -> Dict:
        """Анализирует источники аннотаций"""
        try:
            sources = {
                "llm_generated": 0,
                "synthetic": 0,
                "fallback": 0,
                "manual": 0
            }
            
            for _, annotation in data_pairs:
                if annotation.get("is_synthetic", False):
                    sources["synthetic"] += 1
                elif annotation.get("is_fallback", False):
                    sources["fallback"] += 1
                elif "llm" in str(annotation).lower():
                    sources["llm_generated"] += 1
                else:
                    sources["manual"] += 1
            
            return sources
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа источников: {e}")
            return {}


# Функции для удобного использования
def create_automated_trocr_dataset(source_images: List[str],
                                 output_path: str,
                                 num_synthetic: int = 500,
                                 config: Optional[EnhancedTrOCRConfig] = None,
                                 progress_callback: Optional[Callable] = None) -> Dict[str, str]:
    """
    Создает полностью автоматизированный TrOCR датасет
    
    Args:
        source_images: Список путей к исходным изображениям
        output_path: Путь для сохранения датасета
        num_synthetic: Количество синтетических примеров
        config: Конфигурация (опционально)
        progress_callback: Callback для прогресса
        
    Returns:
        Dict с путями к созданным датасетам
    """
    preparator = EnhancedTrOCRDatasetPreparator(config)
    return preparator.prepare_fully_automated_dataset(
        source_images, output_path, num_synthetic, progress_callback
    )


def create_llm_annotated_dataset_from_folder(images_folder: str,
                                           output_path: str,
                                           config: Optional[EnhancedTrOCRConfig] = None) -> Dict[str, str]:
    """
    Создает датасет с LLM аннотациями из папки с изображениями
    
    Args:
        images_folder: Папка с изображениями
        output_path: Путь для сохранения
        config: Конфигурация
        
    Returns:
        Dict с путями к датасетам
    """
    # Собираем все изображения из папки
    images_folder = Path(images_folder)
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
    
    source_images = []
    for ext in supported_formats:
        source_images.extend(list(images_folder.glob(f"*{ext}")))
        source_images.extend(list(images_folder.glob(f"**/*{ext}")))
    
    source_images = [str(img) for img in source_images]
    
    return create_automated_trocr_dataset(
        source_images=source_images,
        output_path=output_path,
        num_synthetic=0,  # Только реальные изображения
        config=config
    )


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    config = EnhancedTrOCRConfig(
        enable_llm_annotation=True,
        llm_model="models/gemini-2.0-flash-exp",
        max_llm_requests_per_minute=30
    )
    
    # Создаем автоматизированный датасет
    source_images = ["path/to/image1.jpg", "path/to/image2.png"]
    
    datasets = create_automated_trocr_dataset(
        source_images=source_images,
        output_path="data/automated_trocr_dataset",
        num_synthetic=100,
        config=config
    )
    
    print("✅ Автоматизированный TrOCR датасет создан!")
    print(f"📂 Датасеты: {datasets}") 
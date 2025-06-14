#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеллектуальный анализатор PDF файлов
Определяет наличие текстового слоя и извлекает текст с координатами
"""

import os
import logging
from pathlib import Path
import tempfile
from typing import List, Dict, Tuple, Optional, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class PDFTextAnalyzer:
    """Анализатор текстового содержимого PDF файлов"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        if not fitz:
            self.logger.warning("⚠️ PyMuPDF не установлен. Установите: pip install PyMuPDF")
        if not convert_from_path:
            self.logger.warning("⚠️ pdf2image не установлен. Установите: pip install pdf2image")
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Анализирует PDF файл и определяет оптимальный способ обработки
        
        Returns:
        {
            'has_text_layer': bool,      # Есть ли текстовый слой
            'text_quality': float,       # Качество текста (0-1)
            'page_count': int,           # Количество страниц
            'file_size': int,            # Размер файла
            'processing_method': str,    # 'text_extraction' или 'ocr'
            'recommendation': str        # Рекомендация по обработке
        }
        """
        if not fitz:
            return {
                'has_text_layer': False,
                'text_quality': 0.0,
                'page_count': 0,
                'file_size': 0,
                'processing_method': 'ocr',
                'recommendation': 'PyMuPDF не установлен, используется OCR'
            }
        
        try:
            file_size = os.path.getsize(pdf_path)
            
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
                total_chars = 0
                total_words = 0
                pages_with_text = 0
                
                # Анализируем первые несколько страниц
                pages_to_check = min(3, page_count)
                
                for page_num in range(pages_to_check):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    if text.strip():
                        pages_with_text += 1
                        total_chars += len(text)
                        total_words += len(text.split())
                
                # Вычисляем качество текста
                has_text_layer = pages_with_text > 0
                
                if has_text_layer:
                    # Оценка качества: среднее количество символов на страницу
                    avg_chars_per_page = total_chars / pages_to_check
                    text_quality = min(1.0, avg_chars_per_page / 1000)  # Нормализуем к 1000 символов
                    
                    # Проверяем, что текст содержательный (не только пробелы и символы)
                    meaningful_ratio = total_words / max(1, total_chars / 5)  # Примерно 5 символов на слово
                    text_quality *= meaningful_ratio
                    
                    processing_method = 'text_extraction' if text_quality > 0.3 else 'ocr'
                    recommendation = f"Текстовый слой найден (качество: {text_quality:.2f})"
                else:
                    text_quality = 0.0
                    processing_method = 'ocr'
                    recommendation = "Текстовый слой отсутствует, нужен OCR"
                
                return {
                    'has_text_layer': has_text_layer,
                    'text_quality': text_quality,
                    'page_count': page_count,
                    'file_size': file_size,
                    'processing_method': processing_method,
                    'recommendation': recommendation
                }
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа PDF {pdf_path}: {e}")
            return {
                'has_text_layer': False,
                'text_quality': 0.0,
                'page_count': 0,
                'file_size': 0,
                'processing_method': 'ocr',
                'recommendation': f'Ошибка анализа: {e}'
            }
    
    def extract_text_with_coordinates(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Извлекает текст с координатами из указанной страницы PDF
        
        Returns:
        [
            {
                'text': str,        # Текст блока
                'bbox': tuple,      # (x0, y0, x1, y1) координаты
                'confidence': float # Уверенность (1.0 для текстового слоя)
            }
        ]
        """
        if not fitz:
            self.logger.error("❌ PyMuPDF не установлен")
            return []
        
        try:
            with fitz.open(pdf_path) as doc:
                if page_num >= len(doc):
                    self.logger.warning(f"⚠️ Страница {page_num} не найдена в PDF")
                    return []
                
                page = doc[page_num]
                blocks = []
                
                # Получаем текстовые блоки с координатами
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:  # Текстовый блок
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    bbox = span["bbox"]  # (x0, y0, x1, y1)
                                    blocks.append({
                                        'text': text,
                                        'bbox': bbox,
                                        'confidence': 1.0  # Максимальная уверенность для текстового слоя
                                    })
                
                self.logger.info(f"✅ Извлечено {len(blocks)} текстовых блоков из страницы {page_num}")
                return blocks
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка извлечения текста из PDF {pdf_path}: {e}")
            return []
    
    def convert_to_image_if_needed(self, pdf_path: str, output_dir: str = None) -> Optional[str]:
        """
        Конвертирует PDF в изображение только если нет качественного текстового слоя
        
        Returns:
            str: Путь к изображению или None при ошибке
        """
        # Сначала анализируем PDF
        analysis = self.analyze_pdf(pdf_path)
        
        if analysis['processing_method'] == 'text_extraction':
            self.logger.info(f"✅ PDF {Path(pdf_path).name} содержит качественный текст, конвертация не нужна")
            return None
        
        # Конвертируем в изображение
        if not convert_from_path:
            self.logger.error("❌ pdf2image не установлен")
            return None
        
        try:
            if not output_dir:
                output_dir = tempfile.gettempdir()
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Конвертируем первую страницу
            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=1,
                dpi=200,
                fmt='jpeg'
            )
            
            if images:
                image_path = output_dir / f"{Path(pdf_path).stem}_page_1.jpg"
                images[0].save(image_path, 'JPEG', quality=95)
                
                self.logger.info(f"✅ PDF конвертирован в изображение: {image_path}")
                return str(image_path)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации PDF в изображение {pdf_path}: {e}")
        
        return None
    
    def get_processing_strategy(self, pdf_path: str) -> Dict[str, Any]:
        """
        Определяет оптимальную стратегию обработки PDF файла
        
        Returns:
        {
            'method': str,              # 'text_extraction' или 'ocr'
            'text_blocks': List[Dict],  # Если text_extraction
            'image_path': str,          # Если ocr
            'analysis': Dict            # Результат анализа
        }
        """
        analysis = self.analyze_pdf(pdf_path)
        
        if analysis['processing_method'] == 'text_extraction':
            # Извлекаем текст напрямую
            text_blocks = self.extract_text_with_coordinates(pdf_path)
            return {
                'method': 'text_extraction',
                'text_blocks': text_blocks,
                'image_path': None,
                'analysis': analysis
            }
        else:
            # Конвертируем в изображение для OCR
            image_path = self.convert_to_image_if_needed(pdf_path)
            return {
                'method': 'ocr',
                'text_blocks': [],
                'image_path': image_path,
                'analysis': analysis
            }

# Функции для совместимости с существующим кодом
def analyze_pdf_text_quality(pdf_path: str, logger=None) -> float:
    """Анализирует качество текстового слоя PDF (0.0 - 1.0)"""
    analyzer = PDFTextAnalyzer(logger)
    analysis = analyzer.analyze_pdf(pdf_path)
    return analysis['text_quality']

def has_text_layer(pdf_path: str, logger=None) -> bool:
    """Проверяет наличие текстового слоя в PDF"""
    analyzer = PDFTextAnalyzer(logger)
    analysis = analyzer.analyze_pdf(pdf_path)
    return analysis['has_text_layer']

def extract_pdf_text_blocks(pdf_path: str, page_num: int = 0, logger=None) -> List[Dict]:
    """Извлекает текстовые блоки с координатами из PDF"""
    analyzer = PDFTextAnalyzer(logger)
    return analyzer.extract_text_with_coordinates(pdf_path, page_num) 
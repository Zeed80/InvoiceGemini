"""
Модуль содержит классы для обработки изображений с использованием моделей машинного обучения.
"""
import os
import sys
import tempfile
import numpy as np
import re
import time  # Добавляем импорт time
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import pytesseract
import json
import torch
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
import uuid
import logging  # Добавляем импорт модуля logging

# Настраиваем логгер
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    AutoImageProcessor, 
    AutoTokenizer,
    VisionEncoderDecoderModel,
    DonutProcessor as HfDonutProcessor
)

# Импорт библиотеки Google GenAI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from . import config as app_config
from . import utils
from .settings_manager import settings_manager
from .base_processor import BaseProcessor  # Импортируем BaseProcessor из нового файла
from .invoice_formatter import InvoiceFormatter  # Импортируем InvoiceFormatter из нового файла
from app.processing.table_extractor import extract_table_items_from_layoutlm

# Импортируем новые компоненты безопасности и управления ресурсами
try:
    from .core.memory_manager import get_memory_manager
    from .core.resource_manager import get_resource_manager
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False
    logger.warning("Модули управления памятью недоступны")

class ModelManager:
    """
    Класс для управления моделями машинного обучения.
    Отвечает за загрузку, кэширование и доступ к моделям.
    """
    
    def __init__(self):
        self.models = {} # Кэш для основных процессоров (LayoutLM, Donut)
        self.ocr_processor_instance = None
        self.gemini_processor_instance = None # Кэш для GeminiProcessor
        
        # Инициализация плагинной системы LLM
        self.plugin_manager = None
        self._init_llm_plugins()
        
        # Инициализируем менеджеры ресурсов если доступны
        self.memory_manager = get_memory_manager() if MEMORY_MANAGEMENT_AVAILABLE else None
        self.resource_manager = get_resource_manager() if MEMORY_MANAGEMENT_AVAILABLE else None
        
        if self.memory_manager:
            logger.info("Менеджер памяти инициализирован")
        if self.resource_manager:
            logger.info("Менеджер ресурсов инициализирован")
        
        logger.debug("ModelManager.__init__ completed") 
        
    def get_model(self, model_type):
        model_type_lower = model_type.lower()
        
        # Проверяем доступность памяти перед загрузкой
        if self.memory_manager:
            can_load, message = self.memory_manager.can_load_model(model_type_lower)
            if not can_load:
                logger.error(f"Недостаточно памяти для загрузки модели: {message}")
                raise MemoryError(message)
        
        if model_type_lower == 'layoutlm':
            # NEW: Логика для определения, какую модель LayoutLM загружать
            active_type = settings_manager.get_active_layoutlm_model_type()
            model_identifier_to_load = ""
            is_custom_model = False
            
            if active_type == 'custom':
                # Загружаем кастомную локальную модель
                custom_model_name = settings_manager.get_string('Models', 'custom_layoutlm_model_name', app_config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME)
                # Формируем путь к модели в каталоге trained_models
                model_identifier_to_load = os.path.join(app_config.TRAINED_MODELS_PATH, custom_model_name)
                is_custom_model = True  
                logger.debug(f"ModelManager: Загрузка кастомной LayoutLM модели: {model_identifier_to_load}, is_custom={is_custom_model}")
            else:
                # Загружаем модель с Hugging Face
                model_identifier_to_load = settings_manager.get_string('Models', 'layoutlm_id', app_config.LAYOUTLM_MODEL_ID)
                print(f"DEBUG: ModelManager: Загрузка Hugging Face LayoutLM модели: {model_identifier_to_load}, is_custom={is_custom_model}")
            
            # Формируем кэш-ключ, учитывая и тип модели (custom/hf), и идентификатор
            cache_key = f"layoutlm_{is_custom_model}_{model_identifier_to_load.replace(os.sep, '_')}"
            if cache_key not in self.models or self.models[cache_key] is None:
                print(f"DEBUG: ModelManager: Создание нового LayoutLMProcessor с model_id={model_identifier_to_load}, is_custom={is_custom_model}")
                ocr_processor = self.get_ocr_processor()
                self.models[cache_key] = LayoutLMProcessor(ocr_processor, model_identifier_to_load, is_custom_model)
            
            return self.models[cache_key]
            
        elif model_type_lower == 'donut':
            model_id = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
            cache_key = f"donut_{model_id.replace(os.sep, '_')}"
            if cache_key not in self.models or self.models[cache_key] is None:
                print(f"DEBUG: ModelManager: Создание нового DonutProcessor с model_id={model_id}")
                self.models[cache_key] = DonutProcessorImpl(model_id)
            return self.models[cache_key]

        elif model_type_lower == 'gemini':
            # Используем поздний импорт для избежания циклического импорта
            if self.gemini_processor_instance is None or \
               (hasattr(self.gemini_processor_instance, 'model_id') and \
                self.gemini_processor_instance.model_id != settings_manager.get_string('Gemini', 'sub_model_id', app_config.GEMINI_MODEL_ID)):
                # Импортируем GeminiProcessor здесь, чтобы избежать циклического импорта
                from .gemini_processor import GeminiProcessor
                print("DEBUG: ModelManager: (Пере)создание экземпляра GeminiProcessor")
                self.gemini_processor_instance = GeminiProcessor()
            return self.gemini_processor_instance
            
        elif model_type_lower == 'trocr':
            # Поддержка TrOCR моделей
            # Определяем какую модель TrOCR загружать
            model_source = settings_manager.get_string('Models', 'trocr_model_source', 'huggingface')
            
            if model_source == 'custom':
                # Загружаем кастомную дообученную модель
                custom_model_name = settings_manager.get_string('Models', 'custom_trocr_model_name', '')
                if custom_model_name:
                    model_identifier = os.path.join(app_config.TRAINED_MODELS_PATH, custom_model_name)
                else:
                    # Fallback на базовую модель
                    model_identifier = settings_manager.get_string('Models', 'trocr_model_id', 'microsoft/trocr-base-printed')
            else:
                # Загружаем модель с HuggingFace
                model_identifier = settings_manager.get_string('Models', 'trocr_model_id', 'microsoft/trocr-base-printed')
            
            # Формируем кэш-ключ
            cache_key = f"trocr_{model_source}_{model_identifier.replace(os.sep, '_')}"
            
            if cache_key not in self.models or self.models[cache_key] is None:
                print(f"DEBUG: ModelManager: Создание нового TrOCRProcessor с model_id={model_identifier}")
                from .trocr_processor import TrOCRProcessor
                self.models[cache_key] = TrOCRProcessor(model_name=model_identifier)
                
            return self.models[cache_key]
            
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

    def get_ocr_processor(self):
        """Возвращает экземпляр OCRProcessor."""
        if self.ocr_processor_instance is None:
            print("DEBUG: ModelManager creating new OCRProcessor instance")
            self.ocr_processor_instance = OCRProcessor()
        return self.ocr_processor_instance

    def load_layoutlm_model(self, model_id_or_path, is_custom=False):
        """
        Загружает модель LayoutLM с указанным ID или из указанного пути.
        
        Args:
            model_id_or_path (str): ID модели для Hugging Face или путь к локальной модели
            is_custom (bool): Флаг, указывающий, что это локальная пользовательская модель
            
        Returns:
            bool: True, если загрузка прошла успешно, иначе False
        """
        print(f"DEBUG: ModelManager.load_layoutlm_model вызван с model_id_or_path='{model_id_or_path}', is_custom={is_custom}")
        
        # Формируем кэш-ключ для этой модели
        cache_key = f"layoutlm_{is_custom}_{model_id_or_path.replace(os.sep, '_')}"
        
        # Проверяем, нужно ли создавать новый экземпляр процессора
        create_new = False
        if cache_key not in self.models or self.models[cache_key] is None:
            create_new = True
        elif hasattr(self.models[cache_key], 'model_id_loaded') and self.models[cache_key].model_id_loaded != model_id_or_path:
            create_new = True
        
        if create_new:
            # Создаем новый процессор для этой модели
            print(f"DEBUG: ModelManager: Создание нового LayoutLMProcessor с model_id={model_id_or_path}, is_custom={is_custom}")
            ocr_processor = self.get_ocr_processor()
            self.models[cache_key] = LayoutLMProcessor(ocr_processor, model_id_or_path, is_custom)
        
        # Загружаем модель, если еще не загружена
        if not self.models[cache_key].is_loaded:
            success = self.models[cache_key].load_model()
            if success:
                # Если успешно, обновляем настройки в settings_manager
                if is_custom:
                    settings_manager.set_value('Models', 'active_layoutlm_model_type', 'custom')
                    settings_manager.set_value('Models', 'custom_layoutlm_model_name', os.path.basename(model_id_or_path))
                else:
                    settings_manager.set_value('Models', 'active_layoutlm_model_type', 'huggingface')
                    settings_manager.set_value('Models', 'layoutlm_id', model_id_or_path)
                settings_manager.save_settings()
            return success
        else:
            return True  # Уже загружена

    def clear_layoutlm_model(self):
        """
        Выгружает все модели LayoutLM из памяти.
        
        Returns:
            bool: True, если операция успешна
        """
        print("DEBUG: ModelManager.clear_layoutlm_model вызван")
        
        # Находим и удаляем все ключи, связанные с LayoutLM
        keys_to_delete = []
        for key in self.models.keys():
            if key.startswith('layoutlm_'):
                keys_to_delete.append(key)
                
        # Удаляем каждый ключ
        for key in keys_to_delete:
            if self.models[key] is not None:
                # Если в процессоре есть метод unload_model, вызываем его
                if hasattr(self.models[key], 'unload_model') and callable(self.models[key].unload_model):
                    try:
                        self.models[key].unload_model()
                    except Exception as e:
                        print(f"Предупреждение: Ошибка при выгрузке модели LayoutLM: {e}")
                
                # Для ускорения освобождения памяти
                if hasattr(self.models[key], 'model'):
                    try:
                        self.models[key].model = None
                    except Exception:
                        pass
                    
                if hasattr(self.models[key], 'processor'):
                    try:
                        self.models[key].processor = None
                    except Exception:
                        pass
                    
                # Обновляем флаг загрузки
                if hasattr(self.models[key], 'is_loaded'):
                    self.models[key].is_loaded = False
                
            # Удаляем ссылку на процессор из словаря
            del self.models[key]
            
        # Вызываем сборщик мусора для освобождения памяти
        import gc
        gc.collect()
        
        print(f"DEBUG: ModelManager: LayoutLM модели выгружены, удалено {len(keys_to_delete)} экземпляров")
        return True

    def get_gemini_processor(self):
        """Возвращает экземпляр GeminiProcessor."""
        return self.get_model('gemini')
    
    def _init_llm_plugins(self):
        """Инициализирует систему LLM плагинов"""
        try:
            from .plugins.plugin_manager import PluginManager
            self.plugin_manager = PluginManager()
            print("[OK] Система LLM плагинов инициализирована")
        except ImportError as e:
            print(f"[WARN] Система LLM плагинов недоступна: {e}")
            self.plugin_manager = None
        except Exception as e:
            print(f"[ERROR] Ошибка инициализации LLM плагинов: {e}")
            self.plugin_manager = None
    
    def get_llm_plugin_manager(self):
        """Возвращает менеджер LLM плагинов"""
        return self.plugin_manager
    
    def get_llm_plugin(self, plugin_id: str):
        """
        Получает экземпляр LLM плагина
        
        Args:
            plugin_id: ID плагина (например, 'llama', 'mistral')
            
        Returns:
            BaseLLMPlugin или None
        """
        if not self.plugin_manager:
            return None
        
        return self.plugin_manager.get_plugin_instance(plugin_id)
    
    def create_llm_plugin(self, plugin_id: str):
        """
        Создает новый экземпляр LLM плагина
        
        Args:
            plugin_id: ID плагина
            
        Returns:
            BaseLLMPlugin или None
        """
        if not self.plugin_manager:
            return None
        
        return self.plugin_manager.create_plugin_instance(plugin_id)
    
    def get_available_llm_plugins(self) -> List[str]:
        """Возвращает список доступных LLM плагинов"""
        if not self.plugin_manager:
            return []
        
        return self.plugin_manager.get_available_plugins()
    
    def download_model(self, model_type, model_id=None, is_custom=False):
        """
        Загружает модель из Hugging Face Hub или другого источника.
        is_custom здесь больше для информации, т.к. LayoutLMProcessor сам решит, что делать.
        """
        try:
            # Для кастомных моделей LayoutLM загрузка не требуется в этом контексте,
            # так как они уже должны быть локально. Но для HF моделей - да.
            if model_type.lower() == 'layoutlm' and is_custom:
                print(f"INFO: ModelManager: Для кастомной LayoutLM модели '{model_id}' скачивание не выполняется.")
                # Процессор сам попробует загрузить из локального пути
                processor = self.get_model(model_type) # Получаем инстанс, который попытается загрузить
                return processor.is_loaded # Возвращаем статус загрузки процессора
            
            processor = self.get_model(model_type)
            # Для LayoutLM и Donut, model_id для загрузки будет определен внутри get_model
            # и передан в конструктор процессора. Затем процессор вызовет свой load_model.
            # Здесь мы просто инициируем этот процесс, если модель еще не загружена.
            if not processor.is_loaded:
                # model_id для LayoutLM и Donut уже учтен при создании экземпляра в get_model
                # Для Gemini, model_id (sub_model_id) также учтен
                return processor.load_model() # Вызываем load_model без аргументов, т.к. ID уже установлен
            else:
                # Если уже загружена, можно попытаться перегрузить с новым ID, если он предоставлен
                # Но текущая логика get_model должна создавать новый экземпляр, если ID изменился.
                # Поэтому здесь просто возвращаем True, если уже is_loaded.
                if model_id and hasattr(processor, 'model_id_loaded') and processor.model_id_loaded != model_id and not is_custom:
                    print(f"INFO: ModelManager: Модель {model_type} уже загружена, но с другим ID. Перезагрузка с {model_id}...")
                    return processor.load_model(model_id)
                return True # Уже загружена

        except Exception as e:
            print(f"Ошибка при инициации загрузки/проверки модели {model_type} (ID/Path: {model_id}): {str(e)}")
            import traceback
            traceback.print_exc()
            return False


class OCRProcessor:
    """
    Класс для работы с OCR через Tesseract.
    """
    
    def __init__(self):
        # Проверяем настройки Tesseract
        tesseract_path = settings_manager.get_string('Paths', 'tesseract_path', '')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Установлен путь Tesseract: {tesseract_path}")
        else:
            print("Путь Tesseract не указан в настройках (используется системный)")

    def get_tesseract_path(self):
        """
        Возвращает путь к исполняемому файлу Tesseract.
        
        Returns:
            str: Путь к Tesseract или None, если используется системный
        """
        return pytesseract.pytesseract.tesseract_cmd if hasattr(pytesseract.pytesseract, 'tesseract_cmd') else None
        
    @staticmethod
    def validate_tesseract():
        """
        Проверяет, установлен ли Tesseract OCR и настроен ли путь к нему.
        
        Returns:
            bool: True, если Tesseract доступен, иначе False
        """
        if not utils.is_tesseract_installed():
            return False
        
        # Устанавливаем путь к tesseract.exe для pytesseract
        if app_config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = app_config.TESSERACT_PATH
        
        return True
    
    @staticmethod
    def process_image(image_path, lang='eng'):
        """
        Извлекает текст и bounding box'ы из изображения с помощью Tesseract OCR.
        Также поддерживает PDF-файлы, автоматически конвертируя их в изображения.
        
        Args:
            image_path (str): Путь к файлу изображения или PDF
            lang (str): Язык OCR (по умолчанию 'eng')
            
        Returns:
            tuple: (текст, список словарей с bounding box'ами и текстом)
        """
        if not OCRProcessor.validate_tesseract():
            # Возвращаем пустые значения вместо исключения, чтобы вызывающий код мог это обработать
            print("ПРЕДУПРЕЖДЕНИЕ: Tesseract OCR не установлен или не найден. OCR не будет выполнен.")
            return "", [] 
        
        # Проверяем, является ли файл PDF-документом
        if image_path.lower().endswith('.pdf'):
            return OCRProcessor.process_pdf(image_path, lang)
        
        # Читаем изображение
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"ОШИБКА OCR: Файл не найден - {image_path}")
            return "", []
        except Exception as e:
            print(f"ОШИБКА OCR: Не удалось открыть изображение {image_path} - {e}")
            return "", []
        
        # Извлекаем весь текст
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            
            # Извлекаем данные о боксах (с уровнем детализации 5 - символы)
            boxes_data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Формируем список словарей с данными о словах
            words = []
            n_boxes = len(boxes_data['level'])
            for i in range(n_boxes):
                # Фильтруем только распознанные слова (не пустые строки, не символы)
                if boxes_data['text'][i].strip() and int(boxes_data['conf'][i]) > -1: # conf -1 для нераспознанных
                    word = {
                        'text': boxes_data['text'][i],
                        'confidence': float(boxes_data['conf'][i]),
                        'x': int(boxes_data['left'][i]),
                        'y': int(boxes_data['top'][i]),
                        'width': int(boxes_data['width'][i]),
                        'height': int(boxes_data['height'][i]),
                        'page_num': 1  # Всегда 1, так как обрабатываем одну страницу
                    }
                    words.append(word)
            
            print(f"[OCR DEBUG] Распознано {len(words)} слов в файле {os.path.basename(image_path)}")
            print(f"[OCR DEBUG] Первые 5 слов с координатами:")
            for i, word in enumerate(words[:5]):
                print(f"  {i+1}. '{word['text']}' ({word['x']}, {word['y']}, {word['width']}, {word['height']})")
            
            return text, words
        except pytesseract.TesseractNotFoundError:
            print("ОШИБКА OCR: Tesseract не найден. Проверьте путь в настройках.")
            return "", []
        except Exception as e:
            print(f"ОШИБКА OCR: Ошибка во время выполнения Tesseract для {image_path} - {e}")
            return "", []
    
    @staticmethod
    def process_pdf(pdf_path, lang='eng'):
        """
        Обрабатывает PDF-файл, конвертируя его в изображения и выполняя OCR.
        
        Args:
            pdf_path (str): Путь к PDF-файлу
            lang (str): Язык OCR
            
        Returns:
            tuple: (текст, список словарей с bounding box'ами и текстом)
        """
        try:
            print(f"Конвертация PDF в изображения: {pdf_path}")
            
            # Создаем временную директорию для сохранения изображений
            temp_dir = tempfile.mkdtemp()
            
            # Конвертируем PDF в изображения
            try:
                # Проверяем настройки poppler
                poppler_path = app_config.POPPLER_PATH if hasattr(app_config, 'POPPLER_PATH') else None
                
                # Конвертируем PDF в изображения (только первую страницу для начала)
                images = convert_from_path(
                    pdf_path, 
                    dpi=300, 
                    first_page=1, 
                    last_page=1,
                    poppler_path=poppler_path
                )
                
                if not images:
                    print(f"ОШИБКА OCR: Не удалось конвертировать PDF в изображения: {pdf_path}")
                    return "", []
                
                # Сохраняем первое изображение во временный файл
                temp_image_path = os.path.join(temp_dir, "temp_pdf_page.jpg")
                images[0].save(temp_image_path, "JPEG")
                
                # Обрабатываем изображение с помощью OCR
                text, words = OCRProcessor.process_image(temp_image_path, lang)
                
                return text, words
                
            except Exception as e:
                print(f"ОШИБКА OCR: Не удалось конвертировать PDF в изображения: {e}")
                import traceback
                traceback.print_exc()
                return "", []
            
        finally:
            # Удаляем временную директорию
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    @staticmethod
    def get_available_languages():
        """
        Возвращает список доступных языков Tesseract OCR.
        
        Returns:
            list: Список кодов языков, доступных в Tesseract
        """
        if not OCRProcessor.validate_tesseract():
            return ['eng']  # По умолчанию только английский
        
        try:
            # Получаем список языков из Tesseract
            langs = pytesseract.get_languages()
            return langs
        except:
            return ['eng']  # В случае ошибки возвращаем только английский

    @staticmethod
    def process_file(image_path, lang=None):
        """
        Обрабатывает изображение и возвращает результат в формате, совместимом с TrainingDataPreparator.
        
        Args:
            image_path (str): Путь к изображению
            lang (str): Язык OCR (если None, используется значение из настроек)
            
        Returns:
            dict: Словарь с результатами OCR: words, width, height
        """
        if lang is None:
            # Получаем язык OCR из настроек
            lang = settings_manager.get_string('OCR', 'language', 'eng')
        
        # Используем существующий метод process_image
        text, word_data = OCRProcessor.process_image(image_path, lang)
        
        # Преобразуем результат в нужный формат для TrainingDataPreparator
        try:
            # Получаем размеры изображения
            image = Image.open(image_path)
            width, height = image.size
            image.close()
            
            # Преобразуем данные слов в формат, ожидаемый TrainingDataPreparator
            words = []
            for word in word_data:
                try:
                    # Создаем bbox в формате [x1, y1, x2, y2]
                    x1 = word['x']
                    y1 = word['y']
                    x2 = x1 + word['width']
                    y2 = y1 + word['height']
                    
                    words.append({
                        'text': word['text'],
                        'bbox': [x1, y1, x2, y2],
                        'confidence': word.get('confidence', 0)
                    })
                except KeyError as e:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Пропуск слова из-за отсутствия ключа: {e}")
            
            return {
                'words': words,
                'text': text,
                'width': width,
                'height': height
            }
        except Exception as e:
            print(f"ОШИБКА в OCRProcessor.process_file: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'words': [],
                'text': text,
                'width': 0,
                'height': 0
            }


class LayoutLMProcessor(BaseProcessor):
    """
    Процессор для модели LayoutLMv3.
    Использует Tesseract OCR для извлечения текста и его координат.
    """
    def __init__(self, ocr_processor, model_identifier, is_custom):
        print(f"DEBUG: LayoutLMProcessor.__init__ called with identifier: '{model_identifier}', is_custom: {is_custom}") 
        self.model_identifier_to_load = model_identifier # Сохраняем то, что было передано для загрузки
        self.is_custom_model = is_custom
        
        self.processor = None
        self.model = None
        self.is_loaded = False
        self.ocr_processor = ocr_processor 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LayoutLMProcessor будет использовать устройство: {self.device}")
        
        # Эти поля будут установлены после успешной загрузки
        self.model_id_loaded = None # Фактический ID/путь загруженной модели
        self.is_custom_loaded = None # Флаг, что загружена кастомная
        
    def load_model(self, model_id_override=None): # model_id_override здесь для совместимости, но лучше не использовать
        """
        Загружает модель LayoutLM.
        Если self.is_custom_model is True, self.model_identifier_to_load должен быть локальным путем.
        Иначе, это HF ID.
        """
        actual_identifier_to_use = model_id_override if model_id_override else self.model_identifier_to_load
        is_loading_custom = self.is_custom_model if model_id_override is None else False # Если оверрайд, считаем что это HF ID

        if not actual_identifier_to_use:
            self._log("[ОШИБКА LayoutLM] Идентификатор модели (ID или путь) не указан.")
            self.is_loaded = False
            return False

        self._log(f"Загрузка LayoutLM модели: '{actual_identifier_to_use}' (Кастомная: {is_loading_custom})...")
        
        model_cache_dir_for_hf = os.path.join(app_config.MODELS_PATH, 'layoutlm', actual_identifier_to_use.replace("/", "_"))
        # Для кастомных моделей, actual_identifier_to_use уже является путем, и кэш HF не используется в том же смысле
        # Но AutoProcessor.from_pretrained и AutoModelForTokenClassification могут все равно пытаться что-то кэшировать,
        # если им передать путь, который похож на HF ID. Поэтому лучше использовать local_files_only=True для кастомных.

        offline_mode = app_config.OFFLINE_MODE
        token = app_config.HF_TOKEN
        
        source_path_for_load = actual_identifier_to_use # По умолчанию это HF ID или прямой путь к кастомной модели

        try:
            if is_loading_custom:
                # Для кастомной модели, source_path_for_load - это прямой путь к папке
                if not os.path.isdir(source_path_for_load):
                    self._log(f"[ОШИБКА LayoutLM] Путь к кастомной модели не найден или не является директорией: {source_path_for_load}")
                    self.is_loaded = False
                    return False
                self._log(f"Загрузка кастомной модели LayoutLM из локального пути: {source_path_for_load}")
                
                # Проверяем наличие необходимых файлов
                required_files = ['config.json', 'pytorch_model.bin']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(source_path_for_load, f))]
                if missing_files:
                    self._log(f"[ПРЕДУПРЕЖДЕНИЕ] В директории кастомной модели отсутствуют файлы: {', '.join(missing_files)}")
                
                # Проверяем наличие preprocessor_config.json, если его нет - создаем
                preprocessor_config_path = os.path.join(source_path_for_load, 'preprocessor_config.json')
                if not os.path.exists(preprocessor_config_path):
                    self._log(f"[ПРЕДУПРЕЖДЕНИЕ] Файл preprocessor_config.json отсутствует, создаем базовый")
                    # Создаем базовый preprocessor_config.json для LayoutLM
                    base_preprocessor_config = {
                        "apply_ocr": False,
                        "do_resize": True,
                        "do_thumbnail": True,
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_processor_type": "LayoutLMv3ImageProcessor",
                        "image_std": [0.5, 0.5, 0.5],
                        "processor_class": "LayoutLMv3Processor",
                        "size": {"height": 224, "width": 224},
                        "tokenizer_class": "LayoutLMv3Tokenizer"
                    }
                    try:
                        import json
                        with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
                            json.dump(base_preprocessor_config, f, indent=2)
                        self._log(f"Создан базовый preprocessor_config.json в {preprocessor_config_path}")
                    except Exception as e_create:
                        self._log(f"[ОШИБКА] Не удалось создать preprocessor_config.json: {e_create}")
                
                # При загрузке из локального пути, HF кэш не должен использоваться для скачивания
                # local_files_only=True гарантирует, что файлы будут браться только из этого пути.
                try:
                    # Сначала пробуем загрузить процессор
                    self.processor = AutoProcessor.from_pretrained(source_path_for_load, apply_ocr=False, local_files_only=True, token=token)
                except Exception as e_proc:
                    self._log(f"[ОШИБКА] Не удалось загрузить процессор: {e_proc}")
                    # Если не удалось загрузить процессор, пробуем создать его из базовой модели
                    try:
                        from transformers import LayoutLMv3Processor, LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor
                        base_tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base", local_files_only=False, token=token)
                        base_image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
                        self.processor = LayoutLMv3Processor(base_tokenizer, base_image_processor)
                        self._log("Создан базовый процессор LayoutLMv3 из microsoft/layoutlmv3-base")
                    except Exception as e_base_proc:
                        self._log(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось создать базовый процессор: {e_base_proc}")
                        self.is_loaded = False
                        return False
                
                # Загружаем модель
                try:
                    self.model = AutoModelForTokenClassification.from_pretrained(source_path_for_load, local_files_only=True, token=token)
                except Exception as e_model:
                    self._log(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить модель: {e_model}")
                    self.is_loaded = False
                    return False
            else:
                # Загрузка модели с Hugging Face (с возможностью оффлайн режима из кэша HF)
                self._log(f"Загрузка модели LayoutLM с Hugging Face: {source_path_for_load} (Offline: {offline_mode})")
                os.makedirs(model_cache_dir_for_hf, exist_ok=True)
                
                self.processor = AutoProcessor.from_pretrained(
                    source_path_for_load,
                    apply_ocr=False, 
                    cache_dir=model_cache_dir_for_hf, # Явно указываем cache_dir
                    local_files_only=offline_mode,
                    token=token
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    source_path_for_load,
                    cache_dir=model_cache_dir_for_hf, # Явно указываем cache_dir
                    local_files_only=offline_mode,
                    token=token
                )
            
            self.model.to(self.device)
            self._log(f"Модель LayoutLM '{actual_identifier_to_use}' успешно загружена.")
            self.is_loaded = True
            self.model_id_loaded = actual_identifier_to_use # Сохраняем фактический идентификатор загруженной модели
            self.is_custom_loaded = is_loading_custom      # Сохраняем флаг типа загруженной модели
            return True
        
        except Exception as e_load:
            # Если это не кастомная модель и не оффлайн режим, ИЛИ если это оффлайн режим, но загрузка из кэша не удалась
            if not is_loading_custom and not offline_mode:
                self._log(f"Не удалось загрузить LayoutLM ('{actual_identifier_to_use}') напрямую, пробуем принудительно скачать файлы в кэш: {e_load}")
                try:
                    # Это может быть не нужно, если cache_dir правильно работает с from_pretrained
                    # Но оставим как запасной вариант, если from_pretrained не скачивает сама
                    required_files = getattr(self.model.config, '_model_type', 'layoutlm') # Получаем тип модели для файлов
                    # Здесь нужна более умная логика для определения файлов, но для LayoutLM это обычно:
                    config_files = ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'special_tokens_map.json', 'tokenizer_config.json']
                    model_files = ['pytorch_model.bin'] # или .safetensors
                    # Для LayoutLM (не v3) файлы могут быть другими.
                    # Это очень упрощенная логика скачивания, from_pretrained должна справляться лучше.
                    
                    # Удаляем старый кэш для этой модели перед принудительной загрузкой
                    if os.path.exists(model_cache_dir_for_hf):
                        import shutil
                        self._log(f"Удаление старого кэша для {actual_identifier_to_use} в {model_cache_dir_for_hf}")
                        try: shutil.rmtree(model_cache_dir_for_hf)
                        except Exception as e_rm: self._log(f"Ошибка удаления старого кэша: {e_rm}")
                    os.makedirs(model_cache_dir_for_hf, exist_ok=True)

                    for file_name in config_files + model_files:
                        try:
                            hf_hub_download(
                                repo_id=actual_identifier_to_use, 
                                filename=file_name,
                                cache_dir=model_cache_dir_for_hf, # Скачиваем в наш кэш
                                local_files_only=False, # Принудительное скачивание
                                token=token,
                                force_download=True 
                            )
                            self._log(f"Файл {file_name} для LayoutLM принудительно загружен в {model_cache_dir_for_hf}")
                        except Exception as e_file_dl:
                            self._log(f"Не удалось принудительно скачать файл {file_name} для LayoutLM ({actual_identifier_to_use}): {e_file_dl}. Пропускаем, если он не критичен.")
                    
                    # После принудительного скачивания, пробуем загрузить из локального кэша
                    self.processor = AutoProcessor.from_pretrained(model_cache_dir_for_hf, apply_ocr=False, local_files_only=True, token=token)
                    self.model = AutoModelForTokenClassification.from_pretrained(model_cache_dir_for_hf, local_files_only=True, token=token)
                    self.model.to(self.device)
                    self._log(f"Модель LayoutLM '{actual_identifier_to_use}' успешно загружена после принудительного скачивания в кэш.")
                    self.is_loaded = True
                    self.model_id_loaded = actual_identifier_to_use
                    self.is_custom_loaded = False # Это была HF модель
                    return True
                except Exception as e_download_and_load:
                    self._log(f"Критическая ошибка при загрузке LayoutLM ('{actual_identifier_to_use}') после попытки принудительного скачивания: {e_download_and_load}")
                    # import traceback; traceback.print_exc() # Раскомментировать для детальной отладки
                    self.is_loaded = False
                    return False
            else:
                 # Это была либо кастомная модель (и загрузка не удалась), либо оффлайн-режим для HF модели (и кэш не помог)
                self._log(f"Ошибка загрузки LayoutLM ('{actual_identifier_to_use}'). Кастомная: {is_loading_custom}, Оффлайн: {offline_mode}. Ошибка: {e_load}")
                # import traceback; traceback.print_exc()
                self.is_loaded = False
                return False
        return False # Если ни один из путей не привел к успеху
    
    def _log(self, message):
        # Простой логгер, можно заменить на logging.info или передавать callback
        print(f"LayoutLMProcessor: {message}")

    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        if not self.is_loaded:
            print("ОШИБКА: LayoutLMProcessor.process_image вызван, но модель не загружена!")
            error_data = {"note_gemini": "ОШИБКА: Модель LayoutLM не загружена."}
            return InvoiceFormatter.format_invoice_data(error_data)

        lang = ocr_lang if ocr_lang else app_config.DEFAULT_TESSERACT_LANG
        
        # Проверяем, является ли файл PDF-документом
        is_pdf = image_path.lower().endswith('.pdf')
        temp_dir = None
        temp_image_path = None
        
        try:
            if is_pdf:
                # Для PDF файлов используем временный файл изображения
                try:
                    # Создаем временную директорию
                    temp_dir = tempfile.mkdtemp()
                    
                    # Проверяем настройки poppler
                    poppler_path = app_config.POPPLER_PATH if hasattr(app_config, 'POPPLER_PATH') else None
                    
                    print(f"Конвертация PDF в изображение для LayoutLM: {image_path}")
                    
                    # Конвертируем PDF в изображения (только первую страницу)
                    images = convert_from_path(
                        image_path, 
                        dpi=300, 
                        first_page=1, 
                        last_page=1,
                        poppler_path=poppler_path
                    )
                    
                    if not images:
                        print(f"ОШИБКА: Не удалось конвертировать PDF в изображения: {image_path}")
                        error_data = {"note_gemini": f"Ошибка конвертации PDF: {os.path.basename(image_path)}."}
                        return InvoiceFormatter.format_invoice_data(error_data)
                    
                    # Сохраняем первое изображение во временный файл
                    temp_image_path = os.path.join(temp_dir, "temp_layoutlm_page.jpg")
                    images[0].save(temp_image_path, "JPEG")
                    print(f"PDF сконвертирован в изображение: {temp_image_path}")
                    
                    # Используем временный файл изображения для дальнейшей обработки
                    text, words = self.ocr_processor.process_image(temp_image_path, lang=lang)
                except Exception as e:
                    print(f"ОШИБКА: Не удалось обработать PDF файл: {e}")
                    import traceback
                    traceback.print_exc()
                    error_data = {"note_gemini": f"Ошибка обработки PDF: {str(e)}"}
                    return InvoiceFormatter.format_invoice_data(error_data)
            else:
                # Для обычных изображений используем стандартный OCR
                text, words = self.ocr_processor.process_image(image_path, lang=lang)

            if not words:
                print(f"Не удалось распознать слова в изображении: {image_path} с языком {lang}")
                error_data = {"note_gemini": f"OCR не распознал текст на изображении ({os.path.basename(image_path)})."}
                return InvoiceFormatter.format_invoice_data(error_data)
            
            try:
                # Для PDF используем temp_image_path, для обычных изображений - image_path
                actual_image_path = temp_image_path if is_pdf else image_path
                print(f"Открываем изображение для LayoutLM: {actual_image_path}")
                image = Image.open(actual_image_path).convert("RGB")
                width, height = image.size
            except Exception as e:
                print(f"Ошибка открытия изображения {actual_image_path} в LayoutLMProcessor: {e}")
                error_data = {"note_gemini": f"Ошибка открытия изображения: {os.path.basename(image_path)}."}
                return InvoiceFormatter.format_invoice_data(error_data)

            word_texts = [word['text'] for word in words]
            raw_boxes = [[word['x'], word['y'], word['x'] + word['width'], word['y'] + word['height']] for word in words]
            normalized_boxes = [self._normalize_box(box, width, height) for box in raw_boxes]
            
            try:
                encoding = self.processor(
                    image, 
                    word_texts,
                    boxes=normalized_boxes,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length", 
                    max_length=512 
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    
                predictions = outputs.logits.argmax(dim=2)
                id2label = {int(k):v for k,v in self.model.config.id2label.items()} # Убедимся что ключи - int
                word_data_decoded = self._decode_predictions(encoding, predictions, id2label, words, normalized_boxes)

                # Отладочная информация
                print(f"DEBUG LayoutLM: Найдено {len(word_data_decoded)} размеченных слов")
                if word_data_decoded:
                    print("DEBUG LayoutLM: Первые 10 размеченных слов:")
                    for i, (word, label, box) in enumerate(word_data_decoded[:10]):
                        print(f"  {i+1}. '{word}' -> {label} (координаты: {box})")

                # Извлекаем поля с учетом всех возможных меток
                company_name = self._extract_field(word_data_decoded, 'COMPANY_NAME')
                if not company_name:
                    company_name = self._extract_field(word_data_decoded, 'COMPANY')
                
                invoice_number = self._extract_field(word_data_decoded, 'INVOICE_NUMBER')
                
                invoice_date = self._extract_field(word_data_decoded, 'INVOICE_DATE')
                if not invoice_date:
                    invoice_date = self._extract_field(word_data_decoded, 'DATE')
                
                # Извлекаем сумму с учетом русских меток
                total_amount = self._extract_field(word_data_decoded, 'TOTAL')
                if not total_amount:
                    total_amount = self._extract_field(word_data_decoded, 'AMOUNT')
                
                # Извлекаем сумму без НДС (русские метки)
                amount_without_vat = self._extract_field(word_data_decoded, 'СУММА_БЕЗ_НДС')
                
                # Извлекаем НДС (русские метки)
                vat_rate = self._extract_field(word_data_decoded, 'НДС_%')
                
                invoice_data = {
                    'company': company_name,
                    'inn': self._extract_field(word_data_decoded, 'COMPANY_INN'),
                    'invoice_number': invoice_number,
                    'date': invoice_date,
                    'total_amount': total_amount,
                    'amount_without_vat': amount_without_vat,
                    'vat_percent': vat_rate,
                    'currency': 'RUB', 
                    'items': [], 
                    'note_gemini': "Извлечено с помощью LayoutLMv3"
                }

                # Отладочная информация для извлеченных полей
                print(f"DEBUG LayoutLM: Извлеченные поля:")
                for field, value in invoice_data.items():
                    if field not in ['items', 'currency']:
                        print(f"  {field}: '{value}'")
                invoice_data['currency'] = self._extract_currency(invoice_data, word_data_decoded)

                # Подготовка данных для извлечения товаров из таблицы
                try:
                    table_data = {
                        'words': word_texts,
                        'boxes': normalized_boxes,
                        'labels': [id2label.get(idx.item(), 'O') for idx in predictions[0][:len(word_texts)]]
                    }
                    
                    print(f"DEBUG: Подготовлены данные для извлечения товаров: words={len(table_data['words'])}, boxes={len(table_data['boxes'])}, labels={len(table_data['labels'])}")
                    
                    # Извлечение структурированных данных о товарах из подготовленных данных
                    items = extract_table_items_from_layoutlm(table_data)
                    print(f"DEBUG: Извлечено товаров: {len(items)}")
                    logger.info(f"Извлечено {len(items)} товаров из таблицы")
                    logger.debug(f"Извлеченные товары: {items}")
                    
                    # Добавляем извлеченные товары в результат
                    if items:
                        invoice_data['items'] = items
                except Exception as e:
                    print(f"DEBUG: Ошибка при извлечении товаров: {e}")
                    import traceback
                    traceback.print_exc()

                return InvoiceFormatter.format_invoice_data(invoice_data)

            except Exception as e:
                print(f"Ошибка при обработке изображения в LayoutLMProcessor ({self.model_id_loaded}): {e}")
                import traceback
                traceback.print_exc()
                error_data = {"note_gemini": f"Ошибка LayoutLM: {e}"}
                return InvoiceFormatter.format_invoice_data(error_data)
        finally:
            # Удаляем временную директорию в самом конце
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Временная директория удалена: {temp_dir}")
                except Exception as e:
                    print(f"Ошибка при удалении временной директории: {e}")

    def _normalize_box(self, box, width, height):
        x0, y0, x1, y1 = box
        return [
            min(max(0, int(1000 * (x0 / width))), 1000),
            min(max(0, int(1000 * (y0 / height))), 1000),
            min(max(0, int(1000 * (x1 / width))), 1000),
            min(max(0, int(1000 * (y1 / height))), 1000)
        ]

    def _decode_predictions(self, encoding, predictions, id2label, original_ocr_words, original_normalized_boxes):
        word_labels = {}
        for i, pred_index in enumerate(predictions[0].tolist()):
            label = id2label.get(pred_index, 'O') 
            word_idx = encoding.word_ids(batch_index=0)[i]
            if word_idx is None or label == 'O':
                continue
            if word_idx not in word_labels:
                word_labels[word_idx] = []
            word_labels[word_idx].append(label)

        decoded_results = []
        # Используем длину original_ocr_words для итерации, так как word_ids могут ссылаться на эти индексы
        for idx in range(len(original_ocr_words)):
            word_info = original_ocr_words[idx]
            word_text = word_info['text']
            # Бокс берем из original_normalized_boxes по тому же индексу
            word_box = original_normalized_boxes[idx] if idx < len(original_normalized_boxes) else [0,0,0,0]
            
            labels_for_this_word = word_labels.get(idx, ['O'])
            final_label_base = 'O'
            for lbl in labels_for_this_word:
                if lbl.startswith('B-'):
                    final_label_base = lbl.split('-')[-1]
                    break
                elif lbl.startswith('I-') and final_label_base == 'O': 
                    final_label_base = lbl.split('-')[-1]
            
            if final_label_base != 'O':
                decoded_results.append((word_text, final_label_base, word_box))
        return decoded_results

    def _extract_field(self, word_data, target_label_base):
        extracted_words = []
        try:
            word_data_sorted = sorted(word_data, key=lambda item: (item[2][1], item[2][0])) 
        except IndexError:
            print("Предупреждение: Ошибка при сортировке word_data по координатам боксов в _extract_field.")
            word_data_sorted = word_data

        for word_text, label_base, box in word_data_sorted:
            if label_base.upper() == target_label_base.upper():
                extracted_words.append(word_text)
        return " ".join(extracted_words)

    def _extract_currency(self, parsed_fields, word_data):
        total_str = parsed_fields.get('total_amount', '')
        if '₽' in total_str or 'руб' in total_str.lower(): return 'RUB'
        if '$' in total_str: return 'USD'
        if '€' in total_str: return 'EUR'
        currency_from_label = self._extract_field(word_data, 'CURRENCY')
        if currency_from_label.upper() in ['RUB', 'РУБ', '₽'] : return 'RUB'
        return 'RUB' 

    def get_full_prompt(self, custom_prompt_text=None):
        base_prompt = custom_prompt_text if custom_prompt_text else settings_manager.get_string('Prompts', 'layoutlm_prompt', "Распознай структуру документа и извлеки основные поля.")
        full_prompt = f"""
====== СИСТЕМНАЯ ИНФОРМАЦИЯ (НЕ ОТОБРАЖАЕТСЯ ПОЛЬЗОВАТЕЛЮ) ======
Модель: LayoutLMv3 ({self.model_id_loaded})
Использует OCR: Да (Tesseract OCR)
Язык OCR: {settings_manager.get_string('OCR', 'language', app_config.DEFAULT_TESSERACT_LANG)}
Извлечение координат: Да, нормализованные в диапазоне 0-1000
Прямой доступ к тексту: Да
====== КОНЕЦ СИСТЕМНОЙ ИНФОРМАЦИИ ======

====== БАЗОВЫЙ ПРОМПТ ДЛЯ ИЗВЛЕЧЕНИЯ ДАННЫХ ======
{base_prompt}
====== КОНЕЦ БАЗОВОГО ПРОМПТА ======

====== ОЖИДАЕМЫЕ ПОЛЯ ДЛЯ ИЗВЛЕЧЕНИЯ (примерный список, модель может извлекать и другие) ======
- COMPANY (Наименование поставщика/продавца)
- INVOICE_ID (Номер счета/документа)
- DATE (Дата счета/документа)
- TOTAL (Общая сумма по счету)
- CURRENCY (Валюта)
- (Другие поля, если модель их разметит, например, ADDRESS, VAT, ITEM_NAME, ITEM_QUANTITY, ITEM_PRICE и т.д.)
====== КОНЕЦ ОЖИДАЕМЫХ ПОЛЕЙ ======
"""
        return full_prompt


class DonutProcessorImpl(BaseProcessor):
    """
    Процессор для модели Donut.
    Использует Donut для извлечения данных из изображений без OCR.
    """
    
    def __init__(self, model_id):
        print("DEBUG: DonutProcessorImpl.__init__ called")
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.task_start_token = "<s_cord-v2>"
    
    def load_model(self, model_id=None):
        try:
            if model_id:
                self.model_id = model_id
            print(f"Загрузка Donut модели {self.model_id}...")
            model_cache_dir = os.path.join(app_config.MODELS_PATH, 'donut')
            os.makedirs(model_cache_dir, exist_ok=True)
            offline_mode = app_config.OFFLINE_MODE
            token = app_config.HF_TOKEN

            # Проверяем доступность CUDA
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Donut: Используется устройство: {device}")

            try:
                self.processor = HfDonutProcessor.from_pretrained(
                    model_cache_dir if offline_mode and os.path.exists(os.path.join(model_cache_dir, 'processor_config.json')) else self.model_id,
                    cache_dir=model_cache_dir, # Указываем cache_dir здесь
                    local_files_only=offline_mode,
                    token=token
                )
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_cache_dir if offline_mode and os.path.exists(os.path.join(model_cache_dir, 'config.json')) else self.model_id,
                    cache_dir=model_cache_dir, # Указываем cache_dir здесь
                    local_files_only=offline_mode,
                    token=token,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32  # Оптимизация для GPU
                )
                
                # Перемещаем модель на устройство
                self.model.to(device)
                self.device = device
                print(f"Donut: Модель '{self.model_id}' успешно загружена.")
            except Exception as e_direct_load:
                if not offline_mode:
                    print(f"Donut: Не удалось загрузить '{self.model_id}' напрямую ({e_direct_load}), пробуем скачать файлы отдельно...")
                    required_files = ['config.json', 'pytorch_model.bin', 'processor_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model']
                    for file_name in required_files:
                        try:
                            hf_hub_download(
                                repo_id=self.model_id, 
                                filename=file_name,
                                cache_dir=model_cache_dir,
                                token=token,
                                force_download=True
                            )
                            print(f"Файл {file_name} для Donut загружен в {model_cache_dir}")
                        except Exception as e_file:
                            print(f"Не удалось скачать файл {file_name} для Donut ({self.model_id}): {e_file}. Пропускаем, если он не критичен.")
                    
                    self.processor = HfDonutProcessor.from_pretrained(model_cache_dir, local_files_only=True, token=token)
                    self.model = VisionEncoderDecoderModel.from_pretrained(model_cache_dir, local_files_only=True, token=token)
                    print(f"Donut: Модель '{self.model_id}' успешно загружена из кэша после скачивания файлов.")
                else:
                    print(f"Donut: Оффлайн-режим, не удалось загрузить '{self.model_id}' из кэша ({e_direct_load}).")
                    self.is_loaded = False
                    return False
            
            if self.task_start_token not in self.processor.tokenizer.get_vocab():
                print(f"Добавление токена {self.task_start_token} в токенизатор Donut...")
                added_tokens = self.processor.tokenizer.add_tokens(self.task_start_token, special_tokens=True)
                if added_tokens > 0:
                    self.model.resize_token_embeddings(len(self.processor.tokenizer))
                    print(f"Токен {self.task_start_token} добавлен, эмбеддинги модели обновлены.")
                else:
                    print(f"Предупреждение: Токен {self.task_start_token} не был добавлен (уже есть?).")
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Общая ошибка при загрузке модели Donut ({self.model_id}): {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False

    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """
        Обрабатывает изображение с помощью Donut модели.
        ocr_lang и custom_prompt здесь не используются напрямую для генерации,
        но custom_prompt (если это JSON структура) может использоваться для self.task_start_token.
        """
        if not self.is_loaded:
            print("ОШИБКА: DonutProcessor.process_image вызван, но модель не загружена!")
            # Пытаемся загрузить модель, если она еще не загружена
            if not self.load_model(): # Загружаем с текущим self.model_id
                print("ОШИБКА: DonutProcessor.process_image: Не удалось загрузить модель Donut.")
                error_data = {"note_gemini": "ОШИБКА: Модель Donut не загружена / не удалось загрузить."}
                return InvoiceFormatter.format_invoice_data(error_data)

        try:
            # Проверяем, является ли файл PDF-документом
            if image_path.lower().endswith('.pdf'):
                # Для PDF файлов используем конвертацию через pdf2image
                try:
                    from pdf2image import convert_from_path
                    import tempfile
                    
                    # Проверяем настройки poppler
                    poppler_path = settings_manager.get_string('Tools', 'poppler_path', '')
                    if poppler_path and os.path.exists(poppler_path):
                        poppler_path = os.path.join(poppler_path, 'bin') if not poppler_path.endswith('bin') else poppler_path
                    else:
                        poppler_path = None
                    
                    # Конвертируем первую страницу PDF в изображение
                    pages = convert_from_path(
                        image_path, 
                        first_page=1, 
                        last_page=1,
                        dpi=200,  # Хорошее качество для OCR
                        poppler_path=poppler_path
                    )
                    
                    if pages:
                        image = pages[0].convert("RGB")
                        print(f"Donut: PDF конвертирован в изображение {image.size}")
                    else:
                        raise Exception("Не удалось конвертировать PDF")
                        
                except Exception as e:
                    print(f"Ошибка конвертации PDF {image_path} в DonutProcessor: {e}")
                    error_data = {"note_gemini": f"Ошибка конвертации PDF: {os.path.basename(image_path)}"}
                    return InvoiceFormatter.format_invoice_data(error_data)
            else:
                # Для обычных изображений
                image = Image.open(image_path).convert("RGB")
                
        except Exception as e:
            print(f"Ошибка открытия изображения {image_path} в DonutProcessor: {e}")
            error_data = {"note_gemini": f"Ошибка открытия изображения: {os.path.basename(image_path)}."}
            return InvoiceFormatter.format_invoice_data(error_data)

        # Для модели CORD-v2 используем правильный task token
        # Игнорируем custom_prompt, так как модель обучена на специфическом формате
        task_prompt_to_use = self.task_start_token
        
        print(f"DonutProcessor: Используется task_prompt: {task_prompt_to_use}")

        try:
            # Подготовка входов для модели
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            decoder_input_ids = self.processor.tokenizer(task_prompt_to_use, add_special_tokens=False, return_tensors="pt").input_ids
            
            # Перемещаем тензоры на то же устройство, что и модель
            device = getattr(self, 'device', torch.device('cpu'))
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            
            print(f"Donut: Начинаем генерацию на устройстве {device}...")
            start_time = time.time()
            
            # ОПТИМИЗИРОВАННЫЕ параметры генерации для ускорения
            max_length = min(512, self.model.decoder.config.max_position_embeddings)  # Ограничиваем длину
            
            # Генерация вывода с правильными параметрами для CORD-v2
            with torch.no_grad():  # Отключаем градиенты для ускорения
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=max_length,  # Уменьшенная максимальная длина
                    min_length=10,  # Минимальная длина для избежания слишком коротких ответов
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # Greedy decoding для скорости
                    do_sample=False,  # Детерминированная генерация
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]] if self.processor.tokenizer.unk_token_id is not None else None,
                    return_dict_in_generate=True,
                    # Убираем параметры, которые не работают с num_beams=1
                    repetition_penalty=1.1,  # Избегаем повторений
                )
            
            generation_time = time.time() - start_time
            print(f"Donut: Генерация завершена за {generation_time:.2f} сек")
            
            # Декодирование последовательности
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # Убираем task_prompt из начала, если он там есть
            if sequence.startswith(task_prompt_to_use):
                sequence = sequence[len(task_prompt_to_use):]
            
            # Парсинг JSON из строки
            # parsed_json = self._parse_json_output(sequence) # Заменяем на новую реализацию
            cleaned_sequence = self._clean_json_string_for_donut(sequence)
            parsed_json = json.loads(cleaned_sequence)
            
            # Добавляем системную заметку и возможно исходный ответ для отладки
            parsed_json['note_gemini'] = f"Извлечено с помощью Donut ({self.model_id})"
            parsed_json['raw_response_donut'] = cleaned_sequence 
            
            return InvoiceFormatter.format_invoice_data(parsed_json)
            
        except Exception as e:
            print(f"Ошибка при обработке изображения в DonutProcessor ({self.model_id}): {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем словарь с ошибкой, чтобы приложение не падало
            error_data = {
                "note_gemini": f"Ошибка Donut: {str(e)}",
                "raw_response_donut": sequence if 'sequence' in locals() else "Ошибка до декодирования"
            }
            return InvoiceFormatter.format_invoice_data(error_data)

    # НОВЫЙ МЕТОД для очистки JSON, похожий на тот, что в GeminiProcessor
    def _clean_json_string_for_donut(self, s: str) -> str:
        """Очищает строку ответа модели Donut CORD-v2, преобразуя теги в JSON."""
        s = s.strip()
        
        # Сначала убираем все <unk> токены
        s = re.sub(r'<unk>', '', s)
        
        # Модель CORD-v2 возвращает данные в формате тегов, а не JSON
        # Пример: <s_company>ООО "Компания"</s_company><s_total>1000.00</s_total>
        # Нужно преобразовать это в JSON
        
        # Проверяем, есть ли теги CORD-v2
        if '<s_' in s and '</s_' in s:
            return self._parse_cord_tags_to_json(s)
        
        # Если есть обычный JSON, пытаемся его извлечь
        match_md_json = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        if match_md_json:
            return match_md_json.group(1).strip()
        
        # Ищем первый символ { и последний }
        try:
            start_index = s.index('{')
            end_index = s.rindex('}')
            potential_json = s[start_index : end_index + 1]
            
            # Пытаемся исправить распространенные ошибки JSON
            potential_json = self._fix_common_json_errors(potential_json)
            
            json.loads(potential_json) # Проверка на валидность
            return potential_json
        except (ValueError, json.JSONDecodeError) as e:
            # Если не удалось найти/распарсить, пытаемся создать базовую структуру
            print(f"Предупреждение: Не удалось извлечь валидный JSON из ответа Donut: '{s[:200]}...'")
            print(f"Ошибка JSON: {e}")
            
            # Пытаемся извлечь хотя бы некоторые данные с помощью регулярных выражений
            fallback_data = self._extract_fallback_data(s)
            return json.dumps(fallback_data, ensure_ascii=False)
    
    def _parse_cord_tags_to_json(self, s: str) -> str:
        """Преобразует теги CORD-v2 в JSON формат."""
        try:
            # Словарь для маппинга тегов CORD-v2 в поля JSON
            cord_to_json_mapping = {
                's_company': 'company',
                's_nm': 'company',  # альтернативное название компании
                's_total': 'total_amount',
                's_total_price': 'total_amount',
                's_subtotal_price': 'subtotal_amount',
                's_date': 'date',
                's_invoice_number': 'invoice_number',
                's_invoice_id': 'invoice_number',
                's_address': 'address',
                's_menu': 'items',
                's_cnt': 'quantity',
                's_price': 'price',
                's_cashprice': 'cash_amount',
                's_changeprice': 'change_amount',
            }
            
            result = {}
            
            # Находим все теги в формате <s_tag>content</s_tag>
            tag_pattern = r'<s_([^>]+)>(.*?)</s_\1>'
            matches = re.findall(tag_pattern, s, re.DOTALL)
            
            for tag, content in matches:
                # Очищаем содержимое от лишних символов
                content = content.strip()
                content = re.sub(r'<unk>', '', content)
                content = re.sub(r'[^\x20-\x7E\u0400-\u04FF\u0100-\u017F.,\-\d]', '', content)
                
                if content:  # Только если есть содержимое
                    # Маппим тег в JSON поле
                    json_field = cord_to_json_mapping.get(tag, tag)
                    result[json_field] = content
            
            # Если ничего не найдено, создаем базовую структуру
            if not result:
                result = {
                    'error': 'Не удалось извлечь данные из тегов CORD-v2',
                    'raw_text': s[:500] + '...' if len(s) > 500 else s
                }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            print(f"Ошибка парсинга тегов CORD-v2: {e}")
            # Возвращаем fallback структуру
            fallback_data = {
                'error': f'Ошибка парсинга тегов: {str(e)}',
                'raw_text': s[:500] + '...' if len(s) > 500 else s
            }
            return json.dumps(fallback_data, ensure_ascii=False)
    
    def _fix_common_json_errors(self, json_str: str) -> str:
        """Исправляет распространенные ошибки в JSON строках от Donut"""
        # Убираем символы <unk> которые часто появляются в выводе Donut
        json_str = re.sub(r'<unk>', '', json_str)
        
        # Убираем trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Исправляем незакрытые кавычки
        json_str = re.sub(r':\s*"([^"]*)"([^,}\]]*)', r': "\1"', json_str)
        
        # Убираем лишние символы после значений
        json_str = re.sub(r':\s*"([^"]*)"[^,}\]]*([,}\]])', r': "\1"\2', json_str)
        
        # Убираем некорректные символы и последовательности
        json_str = re.sub(r'[^\x20-\x7E\u0400-\u04FF{}",:\[\]0-9.]', '', json_str)
        
        return json_str
    
    def _extract_fallback_data(self, text: str) -> dict:
        """Извлекает данные из текста с помощью регулярных выражений как fallback"""
        # Сначала очищаем текст от символов <unk> и других артефактов
        cleaned_text = re.sub(r'<unk>', '', text)
        cleaned_text = re.sub(r'[^\x20-\x7E\u0400-\u04FF]', ' ', cleaned_text)
        
        fallback_data = {}
        
        # Паттерны для извлечения основных полей
        patterns = {
            'company': [r'company["\s]*:[\s]*["\s]*([^"}\n,]+)', r'поставщик["\s]*:[\s]*["\s]*([^"}\n,]+)'],
            'invoice_number': [r'invoice[_\s]*(?:id|number)["\s]*:[\s]*["\s]*([^"}\n,]+)', r'номер["\s]*:[\s]*["\s]*([^"}\n,]+)'],
            'date': [r'date["\s]*:[\s]*["\s]*([^"}\n,]+)', r'дата["\s]*:[\s]*["\s]*([^"}\n,]+)'],
            'total_amount': [r'total["\s]*:[\s]*["\s]*([^"}\n,]+)', r'сумма["\s]*:[\s]*["\s]*([^"}\n,]+)'],
            'currency': [r'currency["\s]*:[\s]*["\s]*([^"}\n,]+)', r'валюта["\s]*:[\s]*["\s]*([^"}\n,]+)']
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"').strip()
                    # Дополнительная очистка значения
                    value = re.sub(r'<unk>', '', value)
                    value = re.sub(r'[^\x20-\x7E\u0400-\u04FF]', '', value).strip()
                    if value and value != '<unk>':
                        fallback_data[field] = value
                        break
        
        # Если ничего не найдено, возвращаем базовую структуру с ошибкой
        if not fallback_data:
            fallback_data = {
                'error': 'Не удалось извлечь данные из ответа Donut',
                'raw_text': cleaned_text[:500] + '...' if len(cleaned_text) > 500 else cleaned_text
            }
        
        return fallback_data

    # def _parse_json_output(self, output_string):
    #     # ЭТОТ МЕТОД БОЛЬШЕ НЕ НУЖЕН, ЗАМЕНЕН НА _clean_json_string_for_donut и json.loads
    #     try:
    #         # Простая реализация, может потребоваться более сложная логика
    #         # для очистки строки перед парсингом, если модель добавляет лишние символы.
    #         return json.loads(output_string)
    #     except json.JSONDecodeError as e:
    #         print(f"Ошибка декодирования JSON из вывода Donut: {e}")
    #         print(f"Строка, вызвавшая ошибку: {output_string}")
    #         # Возвращаем пустой словарь или специфическую структуру ошибки
    #         return {"error": "JSONDecodeError", "raw_output": output_string}

    def get_full_prompt(self, custom_prompt=None):
        base_prompt = custom_prompt if custom_prompt else settings_manager.get_string('Prompts', 'donut_prompt', "Распознай структуру документа и извлеки основные поля.")
        full_prompt = f"""
====== СИСТЕМНАЯ ИНФОРМАЦИЯ (НЕ ОТОБРАЖАЕТСЯ ПОЛЬЗОВАТЕЛЮ) ======
Модель: Donut ({self.model_id})
Использует OCR: Нет
Язык OCR: {settings_manager.get_string('OCR', 'language', app_config.DEFAULT_TESSERACT_LANG)}
Извлечение координат: Нет
Прямой доступ к тексту: Нет
====== КОНЕЦ СИСТЕМНОЙ ИНФОРМАЦИИ ======

====== БАЗОВЫЙ ПРОМПТ ДЛЯ ИЗВЛЕЧЕНИЯ ДАННЫХ ======
{base_prompt}
====== КОНЕЦ БАЗОВОГО ПРОМПТА ======

====== ОЖИДАЕМЫЕ ПОЛЯ ДЛЯ ИЗВЛЕЧЕНИЯ (примерный список, модель может извлекать и другие) ======
- COMPANY (Наименование поставщика/продавца)
- INVOICE_ID (Номер счета/документа)
- DATE (Дата счета/документа)
- TOTAL (Общая сумма по счету)
- CURRENCY (Валюта)
- (Другие поля, если модель их разметит, например, ADDRESS, VAT, ITEM_NAME, ITEM_QUANTITY, ITEM_PRICE и т.д.)
====== КОНЕЦ ОЖИДАЕМЫХ ПОЛЕЙ ======
"""
        return full_prompt


class InvoiceFormatter:
    """
    Класс для форматирования данных счета в соответствии с требуемым промтом.
    Преобразует исходные данные в нужный формат и структуру.
    """
    
    # Список предопределенных категорий расходов
    EXPENSE_CATEGORIES = [
        "IT and Software Costs",
        "Telephone and Communication",
        "Office Supplies",
        "Travel and Accommodation",
        "Marketing and Advertising",
        "Service Fees",
        "Subscriptions and Memberships",
        "Training and Education",
        "Utilities and Rent",
        "Professional Services"
    ]
    
    # --- NEW: Восстанавливаем статические методы --- 
    @staticmethod
    def format_number_with_comma(number_str, decimal_places=2):
        """
        Преобразует числовое значение к формату с запятой вместо точки.
        Корректно обрабатывает входные строки с запятой или точкой.
        
        Args:
            number_str (str): Строковое представление числа
            decimal_places (int): Количество знаков после запятой (по умолчанию 2)
            
        Returns:
            str: Форматированное число с запятой в качестве десятичного разделителя или 'N/A'
        """
        if not number_str:
            return "N/A"
            
        try:
            normalized_str = str(number_str).replace(',', '.')
            cleaned_str = re.sub(r'[^\d\.]', '', normalized_str)
            if cleaned_str.count('.') > 1:
                parts = cleaned_str.split('.')
                cleaned_str = parts[0] + '.' + ''.join(parts[1:]) 
            
            value = float(cleaned_str)
            format_string = "{:." + str(decimal_places) + "f}"
            return format_string.format(value).replace('.', ',')
        except (ValueError, TypeError):
            print(f"Предупреждение: Не удалось преобразовать '{number_str}' в число.")
            return "N/A" 
    
    @staticmethod
    def format_date(date_str):
        """
        Форматирует дату в формат DD.MM.YYYY.
        """
        if not date_str:
            return "N/A"
            
        date_patterns = [
            r'(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})',  # DD/MM/YYYY
            r'(\d{4})[\/\.\-](\d{1,2})[\/\.\-](\d{1,2})',  # YYYY/MM/DD
            r'(\d{1,2})[\s]([а-яА-Я]+)[\s](\d{2,4})'  # DD месяц YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups[0]) == 4:
                    year, month, day = groups
                else:
                    day, month, year = groups
                    if not month.isdigit():
                        month_map = {
                            'январ': '01', 'феврал': '02', 'март': '03', 'апрел': '04',
                            'ма': '05', 'май': '05', 'июн': '06', 'июл': '07',
                            'август': '08', 'сентябр': '09', 'октябр': '10',
                            'ноябр': '11', 'декабр': '12'
                        }
                        for ru_month, num in month_map.items():
                            if ru_month in month.lower():
                                month = num
                                break
                
                if len(year) == 2:
                    year = '20' + year
                day = day.zfill(2)
                month = str(month).zfill(2)
                return f"{day}.{month}.{year}"
        
        return date_str # Возвращаем как есть, если не распознано
    
    @staticmethod
    def clean_invoice_number(invoice_number):
        """
        Очищает номер счета от известных префиксов (№, Счет и т.д.).
        """
        if not invoice_number or str(invoice_number).upper() == 'N/A':
            return "N/A"
            
        cleaned_number = str(invoice_number).strip()
        prefixes_to_remove = ["счет №", "счет no", "счет n", "счет", "№", "no.", "no", "n"]
        lower_number = cleaned_number.lower()
        
        for prefix in prefixes_to_remove:
            if lower_number.startswith(prefix):
                cleaned_number = cleaned_number[len(prefix):].lstrip()
                break
        
        return cleaned_number if cleaned_number else "N/A"
    
    @staticmethod
    def classify_expense(description, items):
        """
        Классифицирует расход по категориям на основе описания и элементов.
        """
        text = str(description).lower()
        for item in items:
            if isinstance(item, dict) and 'name' in item:
                text += ' ' + str(item['name']).lower()
        
        keywords = {
            "Инструмент для токарной обработки": ['токарн', 'резец', 'пластин', 'державк', 'sbwr', 'ccmt', 'cnga', 'dclnr', 'sbmt', 'tdjx', 'tpgx', 'wnmg'],
            "Инструмент для фрезерной обработки": ['фрез', 'сверл', 'концевая', 'корпусная'],
            "Расходные материалы": ['клейм', 'метчик', 'плашк', 'развертк', 'зенкер', 'щетк', 'диск', 'круг', 'шлиф'],
            "Прочее": ['щуп', 'измерит', 'штанген', 'микрометр']
        }
        
        best_match = "Прочее"
        max_matches = 0
        
        for category, words in keywords.items():
            matches = sum(1 for word in words if word in text)
            if matches > max_matches:
                max_matches = matches
                best_match = category
            elif matches > 0 and matches == max_matches and category != "Прочее": # Небольшой приоритет более конкретным
                 pass # Оставляем первый найденный с тем же числом совпадений
                 
        return best_match
    
    @staticmethod
    def format_description(items):
        """
        Форматирует список товаров в многострочное описание.
        Каждый товар с новой строки и заканчивается точкой с запятой.
        """
        if not items or not isinstance(items, list):
            return "N/A"
            
        descriptions = []
        for item in items:
            if isinstance(item, dict):
                name = item.get('name', '')
                quantity = item.get('quantity', '')
                price = item.get('price', '') # Цена за единицу
                amount = item.get('amount', '') # Сумма по позиции
                
                item_str = str(name)
                if quantity:
                    item_str += f" - {quantity} шт"
                if amount: # Отображаем сумму по позиции, если есть
                    amount_formatted = InvoiceFormatter.format_number_with_comma(amount)
                    item_str += f", {amount_formatted} руб"
                elif price: # Если нет суммы, но есть цена за единицу
                    price_formatted = InvoiceFormatter.format_number_with_comma(price)
                    item_str += f", {price_formatted} руб/шт"
                
                # Добавляем точку с запятой в конце каждого товара
                item_str += ";"
                descriptions.append(item_str)
            elif isinstance(item, str):
                # Для строковых элементов тоже добавляем точку с запятой
                item_str = item.strip()
                if not item_str.endswith(';'):
                    item_str += ";"
                descriptions.append(item_str)
                 
        # Объединяем товары через перенос строки вместо "; "
        return "\n".join(descriptions) if descriptions else "N/A"
    
    @staticmethod
    def calculate_vat_rate(total_amount, amount_without_vat):
        """
        Рассчитывает ставку НДС на основе общей суммы и суммы без НДС.
        """
        try:
            total = float(str(total_amount).replace(',', '.'))
            base = float(str(amount_without_vat).replace(',', '.'))
            
            if base == 0 or total == 0 or abs(total - base) < 0.01: # Добавлена проверка на равенство
                return "0,0"
                
            vat_amount = total - base
            vat_rate = (vat_amount / base) * 100
            
            if 9.5 <= vat_rate < 10.5:
                return "10,0"
            elif 19.5 <= vat_rate < 20.5:
                 return "20,0"

            return f"{vat_rate:.1f}".replace('.', ',')
        except (ValueError, TypeError, ZeroDivisionError):
            return "N/A" 
    # --- Конец восстановленных методов ---

    @staticmethod
    def format_invoice_data(invoice_data):
        """
        Форматирует данные счета в соответствии с требуемым промтом.
        
        Args:
            invoice_data (dict): Исходные данные счета (внутренний формат)
            
        Returns:
            dict: Форматированные данные для отображения (ключи как в таблице)
        """
        if not invoice_data:
            # Возвращаем словарь с ключами N/A, чтобы таблица не ломалась
            return {
                "№ счета": "N/A", "Дата счета": "N/A", "Категория": "N/A",
                "Поставщик": "N/A", "Товары": "N/A", 
                "Сумма без НДС": "N/A", "% НДС": "N/A", "Сумма с НДС": "N/A",
                "Валюта": "N/A", "INN": "N/A", "KPP": "N/A", "Примечание": "N/A"
            }
            
        # Получаем значения из исходных данных (внутренний формат)
        invoice_number = invoice_data.get('invoice_number', '')
        date_str = invoice_data.get('date', '')
        company = invoice_data.get('company', '')
        
        # NEW: Инициализация переменных, которые могут отсутствовать
        category_gemini = invoice_data.get('category_gemini', '')
        description_gemini = invoice_data.get('description_gemini', '')
        amount_without_vat_gemini = invoice_data.get('amount_without_vat_gemini', '')
        vat_percent_gemini = invoice_data.get('vat_percent_gemini', '')
        note_gemini = invoice_data.get('note_gemini', '')
        
        items = invoice_data.get('items', [])
        total_amount_str = invoice_data.get('total_amount', '0.00')
        currency = invoice_data.get('currency', 'RUB')

        # 1. Номер счета
        clean_invoice_number = invoice_number.strip()
        if not clean_invoice_number:
            clean_invoice_number = "N/A"

        # 2. Дата счета
        formatted_date = date_str.strip()
        if not formatted_date:
            formatted_date = "N/A"

        # 3. Наименование компании-отправителя
        if not company:
            company = "N/A"

        # Инициализируем переменные перед использованием
        amount_without_vat_final_str = "N/A"
        vat_percent_final_str = "N/A"

        # 4. Сумма без НДС
        if not amount_without_vat_gemini:
            amount_without_vat_gemini = "N/A"

        # 5. Процент НДС (если указан)
        if not vat_percent_gemini:
            vat_percent_gemini = "N/A"

        # 6. Общая сумма к оплате
        if not total_amount_str:
            total_amount_str = "N/A"

        # 7. Валюта
        if not currency:
            currency = "N/A"

        # 8. Описание товаров/услуг
        if not description_gemini:
            description_gemini = "N/A"

        # 9. Список товаров/услуг
        if not items:
            items = "N/A"

        # 10. Общая сумма
        # total_float = float(total_amount_str.replace(',', '.')) # БЫЛО - падает на N/A
        try:
            if total_amount_str and str(total_amount_str).strip().upper() != 'N/A':
                total_float = float(str(total_amount_str).replace(',', '.'))
            else:
                total_float = 0.0 # Или None, если нужно различать 0 и N/A
        except (ValueError, TypeError):
             total_float = 0.0 # Если преобразование не удалось

        # 11. Общая сумма без НДС
        if amount_without_vat_gemini and amount_without_vat_gemini.upper() != 'N/A':
            amount_without_vat_final_str = InvoiceFormatter.format_number_with_comma(amount_without_vat_gemini, decimal_places=2)
            if not (vat_percent_gemini and vat_percent_gemini.upper() != 'N/A'):
                try:
                    amount_without_vat_gemini_float = float(str(amount_without_vat_gemini).replace(',', '.'))
                    if amount_without_vat_gemini_float > 0:
                         vat_percent_final_str = InvoiceFormatter.calculate_vat_rate(total_float, amount_without_vat_gemini_float)
                    else:
                         vat_percent_final_str = "0,0" # Если сумма без НДС = 0, то и % = 0
                except (ValueError, TypeError):
                    pass # Оставим N/A если не удалось распарсить для расчета
        elif total_float > 0:
            if vat_percent_gemini and vat_percent_gemini.upper() != 'N/A':
                try:
                    vat_rate_str = InvoiceFormatter.format_number_with_comma(vat_percent_gemini.replace('%','').strip(), decimal_places=1)
                    if vat_rate_str != 'N/A':
                        vat_rate_float = float(vat_rate_str.replace(',', '.'))
                        if vat_rate_float >= 0:
                            calculated_amount_without_vat = total_float / (1 + vat_rate_float / 100)
                            amount_without_vat_final_str = InvoiceFormatter.format_number_with_comma(calculated_amount_without_vat, decimal_places=2)
                            vat_percent_final_str = vat_rate_str
                except (ValueError, TypeError):
                     pass 

        if vat_percent_gemini and vat_percent_gemini.upper() != 'N/A':
            vat_percent_final_str = InvoiceFormatter.format_number_with_comma(vat_percent_gemini.replace('%','').strip(), decimal_places=1)
        elif amount_without_vat_final_str != "N/A" and total_float > 0 and vat_percent_final_str == "N/A":
            try:
                amount_without_vat_float = float(amount_without_vat_final_str.replace(',', '.'))
                if amount_without_vat_float > 0:
                    vat_percent_final_str = InvoiceFormatter.calculate_vat_rate(total_float, amount_without_vat_float)
                else:
                     vat_percent_final_str = "0,0"
            except (ValueError, TypeError):
                 vat_percent_final_str = "N/A"

        if total_float > 0 and amount_without_vat_final_str != "N/A":
             amount_without_vat_float_check = float(amount_without_vat_final_str.replace(',', '.'))
             if abs(total_float - amount_without_vat_float_check) < 0.01:
                 if vat_percent_final_str == "N/A": vat_percent_final_str = "0,0"
        
        if vat_percent_final_str == "N/A":
             pass 
        if amount_without_vat_final_str == "N/A":
             pass 

        # ... (остальная часть метода format_invoice_data) ...
        formatted_total = InvoiceFormatter.format_number_with_comma(total_float, decimal_places=2)
        final_currency = currency if currency and currency.upper() != 'N/A' else 'RUB'
        note = note_gemini if note_gemini and note_gemini.upper() != 'N/A' else "N/A"
        
        result = {
            "№ счета": clean_invoice_number,
            "Дата счета": formatted_date,
            "Категория": category_gemini,
            "Поставщик": company,
            "Товары": description_gemini,
            "Сумма без НДС": amount_without_vat_final_str,
            "% НДС": vat_percent_final_str,
            "Сумма с НДС": formatted_total,
            "Валюта": final_currency,
            "INN": invoice_data.get('inn', 'N/A'),
            "KPP": invoice_data.get('kpp', 'N/A'),
            "Примечание": note
        }
        
        return result 
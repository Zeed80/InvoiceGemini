import logging
import os
import sys
import json
import tempfile
import re
import difflib
from PIL import Image
import torch
from pdf2image import convert_from_path
from datetime import datetime
import shutil
import numpy as np
import albumentations as A
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# NEW: Импорты для Hugging Face Datasets и Transformers
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoProcessor, LayoutLMv3Processor, DonutProcessor

# Импорт анализатора PDF
from app.pdf_text_analyzer import PDFTextAnalyzer

def normalize_text_for_matching(text):
    """
    Нормализует текст для сопоставления.
    
    Args:
        text (str): Текст для нормализации
        
    Returns:
        str: Нормализованный текст
    """
    if not isinstance(text, str):
        return ''
        
    # Расширенный список сокращений
    replacements = {
        'ооо': 'общество с ограниченной ответственностью',
        'оао': 'открытое акционерное общество',
        'зао': 'закрытое акционерное общество',
        'ип': 'индивидуальный предприниматель',
        'руб.': 'руб',
        'р.': 'руб',
        '₽': 'руб',
        'рублей': 'руб',
        'рубль': 'руб',
        'рубля': 'руб',
        'коп.': 'коп',
        'копеек': 'коп',
        'копейка': 'коп',
        'копейки': 'коп',
        'тел.': 'телефон',
        'т.': 'телефон',
        'инн': 'инн',
        'кпп': 'кпп',
        'р/с': 'расчетный счет',
        'к/с': 'корреспондентский счет',
        'бик': 'бик',
        'ндс': 'ндс',
        'без ндс': 'без ндс',
        'в т.ч.': 'в том числе',
        'вкл.': 'включая',
        'исх.': 'исходящий',
        'вх.': 'входящий',
    }
    
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Сохраняем числа и даты
    numbers = []
    dates = []
    
    # Находим и сохраняем даты (расширенный список форматов)
    date_patterns = [
        r'\d{2}[./-]\d{2}[./-]\d{4}',  # DD.MM.YYYY
        r'\d{4}[./-]\d{2}[./-]\d{2}',  # YYYY.MM.DD
        r'\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}',  # DD Month YYYY
        r'(?:от\s+)?\d{1,2}[./-]\d{2}[./-]\d{4}',  # "от" DD.MM.YYYY
    ]
    
    for pattern in date_patterns:
        for date in re.finditer(pattern, text):
            date_text = date.group()
            dates.append(date_text)
            text = text.replace(date_text, ' DATE ')
    
    # Находим и сохраняем числа (улучшенная обработка)
    # Сначала ищем числа с валютой
    currency_pattern = r'[\d\s.,]+(?=\s*(?:руб|₽|р\.|коп|%|процент))'
    for number in re.finditer(currency_pattern, text):
        # Очищаем и нормализуем число
        clean_number = _normalize_number(number.group())
        numbers.append(clean_number)
        text = text.replace(number.group(), ' NUMBER ')
    
    # Затем ищем просто числа
    number_pattern = r'(?<!\w)[\d\s.,]+(?!\w)'
    for number in re.finditer(number_pattern, text):
        if ' NUMBER ' not in number.group():  # Проверяем, что это не уже обработанное число
            clean_number = _normalize_number(number.group())
            numbers.append(clean_number)
            text = text.replace(number.group(), ' NUMBER ')
    
    # Удаляем специальные символы, сохраняя важные
    text = re.sub(r'[«»\"\'\(\)\[\]\{\}]', ' ', text)
    text = re.sub(r'[.,;:]+(?!\d)', ' ', text)  # Сохраняем точки и запятые в числах
    
    # Заменяем сокращения
    for abbr, full in replacements.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    
    # Удаляем лишние пробелы
    text = ' '.join(text.split())
    
    # Восстанавливаем даты
    for date in dates:
        text = text.replace('DATE', date, 1)
    
    # Восстанавливаем числа
    for number in numbers:
        text = text.replace('NUMBER', number, 1)
    
    return text

def _normalize_number(number_str):
    """
    Нормализует числовое значение.
    
    Args:
        number_str (str): Строка с числом
        
    Returns:
        str: Нормализованное число
    """
    # Удаляем пробелы
    clean_number = number_str.replace(' ', '')
    
    # Обрабатываем запятые и точки
    if ',' in clean_number and '.' not in clean_number:
        # Если есть запятая и нет точки, считаем запятую десятичным разделителем
        clean_number = clean_number.replace(',', '.')
    elif ',' in clean_number and '.' in clean_number:
        # Если есть и запятая и точка, считаем запятую разделителем тысяч
        clean_number = clean_number.replace(',', '')
    
    # Разбиваем на целую и дробную части
    parts = clean_number.split('.')
    if len(parts) == 2:
        # Обрабатываем целую часть (удаляем разделители)
        whole = parts[0].replace('.', '')
        # Оставляем дробную часть как есть
        fraction = parts[1]
        clean_number = whole + '.' + fraction
    else:
        # Если нет дробной части, просто удаляем все разделители
        clean_number = clean_number.replace('.', '')
    
    return clean_number

def find_best_match_indices(target_text, prepared_words, similarity_threshold):
    """
    Находит индексы слов в OCR тексте, которые наилучшим образом соответствуют целевому тексту.
    
    Args:
        target_text (str): Нормализованный текст для поиска
        prepared_words (list): Список подготовленных слов из OCR
        similarity_threshold (float): Минимальный порог схожести для совпадения
        
    Returns:
        list: Список индексов найденных слов или пустой список, если совпадений не найдено
    """
    if not target_text or not prepared_words:
        return []
        
    # Нормализуем целевой текст
    target_text = normalize_text_for_matching(target_text)
    target_words = target_text.split()
    
    if not target_words:
        return []
    
    # Ищем наилучшие совпадения для каждого слова
    best_matches = []
    used_indices = set()
    
    for target_word in target_words:
        best_match = {
            'index': -1,
            'ratio': 0.0
        }
        
        # Проверяем каждое слово из OCR
        for word in prepared_words:
            if word['index'] in used_indices:
                continue
                
            # Вычисляем схожесть
            ratio = calculate_text_similarity(target_word, word['text'])
            
            # Обновляем лучшее совпадение
            if ratio > best_match['ratio'] and ratio >= similarity_threshold:
                best_match = {
                    'index': word['index'],
                    'ratio': ratio
                }
        
        # Если нашли совпадение, добавляем его
        if best_match['index'] >= 0:
            best_matches.append(best_match)
            used_indices.add(best_match['index'])
    
    # Если не нашли совпадений для всех слов, возвращаем пустой список
    if len(best_matches) < len(target_words) * 0.5:  # Требуем совпадения хотя бы 50% слов
        return []
    
    # Сортируем индексы по порядку их появления в тексте
    return sorted(match['index'] for match in best_matches)

def calculate_text_similarity(text1, text2, log_callback=None):
    """
    Вычисляет схожесть между двумя текстами с учетом различных факторов.
    
    Args:
        text1 (str): Первый текст для сравнения
        text2 (str): Второй текст для сравнения
        log_callback (callable, optional): Функция для логирования
        
    Returns:
        float: Значение схожести от 0.0 до 1.0
    """
    def log_debug(message):
        if log_callback:
            log_callback(message)
        print(f"[DEBUG] {message}")
    
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
        
    # Нормализуем тексты
    text1 = normalize_text_for_matching(text1)
    text2 = normalize_text_for_matching(text2)
    
    if not text1 or not text2:
        return 0.0
        
    # Если тексты идентичны после нормализации
    if text1 == text2:
        log_debug(f"Точное совпадение: '{text1}' == '{text2}'")
        return 1.0
    
    # Проверяем, являются ли тексты числами
    def is_number(text):
        try:
            float(text.replace(' ', '').replace(',', '.'))
            return True
        except ValueError:
            return False
    
    # Специальная обработка для чисел
    if is_number(text1) and is_number(text2):
        try:
            num1 = float(text1.replace(' ', '').replace(',', '.'))
            num2 = float(text2.replace(' ', '').replace(',', '.'))
            # Для чисел используем относительную разницу
            if num1 == num2:
                return 1.0
            diff = abs(num1 - num2) / max(abs(num1), abs(num2))
            similarity = max(0.0, 1.0 - diff)
            log_debug(f"Числовое сравнение: {num1} ~ {num2} = {similarity:.2f}")
            return similarity
        except ValueError:
            pass
    
    # Проверяем, являются ли тексты датами
    def parse_date(text):
        # Пробуем разные форматы дат
        date_formats = [
            '%d.%m.%Y', '%Y.%m.%d',
            '%d/%m/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%Y-%m-%d'
        ]
        text = text.replace(' ', '')
        for fmt in date_formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None
    
    date1 = parse_date(text1)
    date2 = parse_date(text2)
    if date1 and date2:
        # Для дат используем разницу в днях
        days_diff = abs((date1 - date2).days)
        if days_diff == 0:
            return 1.0
        similarity = max(0.0, 1.0 - (days_diff / 30))  # 30 дней как максимальная разница
        log_debug(f"Сравнение дат: {date1.date()} ~ {date2.date()} = {similarity:.2f}")
        return similarity
    
    # Для коротких текстов (до 5 символов) используем более строгое сравнение
    if len(text1) <= 5 or len(text2) <= 5:
        # Используем расстояние Левенштейна для коротких строк
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        similarity = 1.0 - (distance / max_len)
        log_debug(f"Сравнение коротких строк: '{text1}' ~ '{text2}' = {similarity:.2f}")
        return similarity
        
    # Проверяем, является ли один текст подстрокой другого
    if text1 in text2 or text2 in text1:
        shorter = text1 if len(text1) < len(text2) else text2
        longer = text2 if len(text1) < len(text2) else text1
        # Учитываем относительную длину подстроки
        similarity = 0.7 + 0.3 * (len(shorter) / len(longer))
        log_debug(f"Подстрока: '{shorter}' в '{longer}' = {similarity:.2f}")
        return similarity
    
    # Для остальных случаев используем улучшенное нечеткое сравнение
    # Сначала проверяем пословное сходство
    words1 = set(text1.split())
    words2 = set(text2.split())
    if words1 and words2:
        word_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
    else:
        word_similarity = 0.0
    
    # Затем используем sequence matcher для общего сходства
    sequence_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Комбинируем результаты с весами
    similarity = 0.4 * word_similarity + 0.6 * sequence_similarity
    log_debug(f"Комбинированное сходство: '{text1}' ~ '{text2}' = {similarity:.2f}")
    
    return similarity

def levenshtein_distance(s1, s2):
    """
    Вычисляет расстояние Левенштейна между двумя строками.
    
    Args:
        s1 (str): Первая строка
        s2 (str): Вторая строка
        
    Returns:
        int: Расстояние Левенштейна
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class TrainingDataPreparator:
    """Класс для подготовки данных к обучению нейронных сетей."""
    
    def __init__(self, app_config, ocr_processor, gemini_processor):
        """
        Инициализация класса подготовки данных
        
        Args:
            app_config: Конфигурация приложения
            ocr_processor: Процессор OCR
            gemini_processor: Процессор Gemini
        """
        self.app_config = app_config
        self.ocr_processor = ocr_processor
        self.gemini_processor = gemini_processor
        self.stop_requested = False
        self.log_callback = None
        self.progress_callback = None
        
        # Режим интеллектуального извлечения данных
        self.intelligent_mode = False
        self.intelligent_extractor = None
        
        # PDF анализатор для работы с текстовым слоем
        self.pdf_analyzer = PDFTextAnalyzer(logger=self.logger)
        
        # 🚀 GPU ускорение для подготовки данных
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DataPreparator: Будет использовать устройство: {self.device}")
        
        # Проверяем доступность GPU и выводим информацию
        if torch.cuda.is_available():
            print(f"DataPreparator: 🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"DataPreparator: 💾 GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("DataPreparator: ⚠️ GPU недоступна, используется CPU")
        
        # Инициализируем кэш для процессоров чтобы избежать повторной загрузки
        self._processor_cache = {}
        self._model_cache = {}
        
        # Параметры для batch обработки на GPU
        self.batch_size = 8 if torch.cuda.is_available() else 4
        self.use_gpu_augmentation = torch.cuda.is_available()
        
        print(f"DataPreparator: Batch размер: {self.batch_size}")
        print(f"DataPreparator: GPU аугментация: {self.use_gpu_augmentation}")
        
        # 🎯 Интеграция с FieldManager для автоматического формирования промптов
        try:
            from ..field_manager import field_manager
            self.field_manager = field_manager
            self._log("✅ FieldManager подключен для автоматического формирования промптов")
        except ImportError:
            self.field_manager = None
            self._log("⚠️ FieldManager недоступен, будут использоваться статические промпты")

    def _log(self, message):
        """Логирование сообщений"""
        print(f"[DataPreparator] {message}")
        if self.log_callback:
            self.log_callback(message)
    
    def _init_intelligent_extractor(self):
        """Инициализирует интеллектуальный экстрактор если нужно"""
        if self.intelligent_mode and not self.intelligent_extractor:
            try:
                from .intelligent_data_extractor import IntelligentDataExtractor
                self.intelligent_extractor = IntelligentDataExtractor(
                    gemini_processor=self.gemini_processor,
                    logger=None  # Используем наш _log метод
                )
                self._log("🧠 Интеллектуальный экстрактор инициализирован")
            except Exception as e:
                self._log(f"❌ Ошибка инициализации интеллектуального экстрактора: {e}")
                self.intelligent_mode = False

    def _get_cached_processor(self, model_name: str, processor_type: str = "layoutlm"):
        """
        Получает кэшированный процессор или создает новый с GPU поддержкой
        
        Args:
            model_name: Имя модели
            processor_type: Тип процессора (layoutlm, donut)
            
        Returns:
            Процессор модели
        """
        cache_key = f"{processor_type}_{model_name}"
        
        if cache_key not in self._processor_cache:
            try:
                self._log(f"🔄 Загрузка процессора {processor_type} для {model_name}...")
                
                if processor_type == "layoutlm":
                    from transformers import LayoutLMv3Processor
                    processor = LayoutLMv3Processor.from_pretrained(
                        model_name, 
                        apply_ocr=False,
                        cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache')
                    )
                elif processor_type == "donut":
                    from transformers import DonutProcessor
                    processor = DonutProcessor.from_pretrained(
                        model_name,
                        cache_dir=os.path.join(self.app_config.MODELS_PATH, 'cache')
                    )
                else:
                    raise ValueError(f"Неизвестный тип процессора: {processor_type}")
                
                self._processor_cache[cache_key] = processor
                self._log(f"✅ Процессор {processor_type} загружен и кэширован")
                
            except Exception as e:
                self._log(f"❌ Ошибка загрузки процессора {processor_type}: {e}")
                return None
        
        return self._processor_cache[cache_key]

    def _batch_tokenize_layoutlm(self, images, words_list, bboxes_list, labels_list=None, 
                                model_name="microsoft/layoutlmv3-base"):
        """
        Batch токенизация для LayoutLM с GPU ускорением
        
        Args:
            images: Список изображений
            words_list: Список списков слов
            bboxes_list: Список списков bbox
            labels_list: Список списков меток (опционально)
            model_name: Имя модели
            
        Returns:
            dict: Токенизированные данные
        """
        processor = self._get_cached_processor(model_name, "layoutlm")
        if not processor:
            return None
        
        try:
            self._log(f"🔄 Batch токенизация {len(images)} примеров на {self.device}...")
            
            # Результирующие списки
            all_input_ids = []
            all_attention_masks = []
            all_token_type_ids = []
            all_bboxes = []
            all_pixel_values = []
            all_labels = []
            
            # Обрабатываем данные батчами для эффективного использования GPU памяти
            for i in range(0, len(images), self.batch_size):
                batch_end = min(i + self.batch_size, len(images))
                batch_images = images[i:batch_end]
                batch_words = words_list[i:batch_end]
                batch_bboxes = bboxes_list[i:batch_end]
                
                # Токенизация батча
                try:
                    encoding = processor(
                        batch_images,
                        batch_words,
                        boxes=batch_bboxes,
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Перемещаем на GPU если доступна
                    if torch.cuda.is_available():
                        encoding = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in encoding.items()}
                    
                    # Собираем результаты
                    all_input_ids.extend(encoding["input_ids"].cpu().tolist())
                    all_attention_masks.extend(encoding["attention_mask"].cpu().tolist())
                    all_token_type_ids.extend(encoding["token_type_ids"].cpu().tolist())
                    all_bboxes.extend(encoding["bbox"].cpu().tolist())
                    all_pixel_values.extend(encoding["pixel_values"].cpu().tolist())
                    
                    # Обработка меток если есть
                    if labels_list and i < len(labels_list):
                        batch_labels = labels_list[i:batch_end]
                        for labels in batch_labels:
                            # Конвертация меток в ID и добавление специальных токенов
                            if isinstance(labels[0], str):
                                # Если метки в виде строк, конвертируем в ID
                                label_ids = self._convert_string_labels_to_ids(labels)
                            else:
                                label_ids = labels
                            
                            # Добавляем [CLS] и [SEP] токены
                            label_ids = [-100] + label_ids + [-100]
                            # Паддинг до максимальной длины
                            if len(label_ids) < 512:
                                label_ids.extend([-100] * (512 - len(label_ids)))
                            else:
                                label_ids = label_ids[:512]
                            
                            all_labels.append(label_ids)
                    
                    # Прогресс
                    if self.progress_callback:
                        progress = int((batch_end / len(images)) * 100)
                        self.progress_callback(progress)
                        
                except Exception as batch_error:
                    self._log(f"❌ Ошибка в батче {i}-{batch_end}: {batch_error}")
                    continue
            
            result = {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'token_type_ids': all_token_type_ids,
                'bbox': all_bboxes,
                'pixel_values': all_pixel_values
            }
            
            if all_labels:
                result['labels'] = all_labels
            
            self._log(f"✅ Batch токенизация завершена. Обработано {len(all_input_ids)} примеров")
            
            # Очистка GPU памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            self._log(f"❌ Ошибка batch токенизации: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _convert_string_labels_to_ids(self, labels, entity_types=None):
        """
        Конвертирует строковые метки в числовые ID
        
        Args:
            labels: Список строковых меток
            entity_types: Список типов сущностей
            
        Returns:
            list: Список числовых ID
        """
        if not entity_types:
            # Автоматически определяем типы сущностей из меток
            entity_types = set()
            for label in labels:
                if label.startswith(('B-', 'I-')):
                    entity_types.add(label[2:])
            entity_types = sorted(list(entity_types))
        
        # Создаем словарь меток
        label2id = {'O': 0}
        current_id = 1
        for entity_type in entity_types:
            label2id[f"B-{entity_type}"] = current_id
            current_id += 1
            label2id[f"I-{entity_type}"] = current_id
            current_id += 1
        
        return [label2id.get(label, 0) for label in labels]

    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Применяет аугментацию к изображению с возможностью GPU ускорения
        
        Args:
            image: Входное изображение PIL
            
        Returns:
            Image.Image: Аугментированное изображение
        """
        try:
            import albumentations as A
            import numpy as np
            
            # Создаем аугментации оптимизированные для документов
            augmentation = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(0.0, 25.0), p=0.3),
                A.Blur(blur_limit=2, p=0.2),
                A.Rotate(limit=2, p=0.3),  # Небольшие повороты для документов
                A.RandomGamma(gamma_limit=(95, 105), p=0.3),
            ])
            
            # Конвертируем PIL в numpy для albumentations
            image_np = np.array(image)
            
            # Применяем аугментацию
            augmented = augmentation(image=image_np)
            
            # Конвертируем обратно в PIL
            return Image.fromarray(augmented['image'])
            
        except Exception as e:
            self._log(f"⚠️ Ошибка при аугментации: {e}. Возвращаем оригинальное изображение")
            return image

    def apply_batch_augmentation(self, images: List[Image.Image], 
                               augmentation_factor: int = 1) -> List[Image.Image]:
        """
        Применяет batch аугментацию с GPU ускорением если доступно
        
        Args:
            images: Список изображений
            augmentation_factor: Количество аугментированных версий на изображение
            
        Returns:
            List[Image.Image]: Список оригинальных и аугментированных изображений
        """
        try:
            self._log(f"🎨 Batch аугментация {len(images)} изображений (фактор: {augmentation_factor})")
            
            result_images = []
            
            # Добавляем оригинальные изображения
            result_images.extend(images)
            
            # Если аугментация отключена, возвращаем только оригинальные
            if augmentation_factor <= 0:
                return result_images
            
            # Применяем аугментацию
            total_augmentations = len(images) * augmentation_factor
            processed = 0
            
            for i, image in enumerate(images):
                for aug_idx in range(augmentation_factor):
                    try:
                        aug_image = self.apply_augmentation(image)
                        result_images.append(aug_image)
                        processed += 1
                        
                        # Обновляем прогресс
                        if self.progress_callback and processed % 10 == 0:
                            progress = int((processed / total_augmentations) * 100)
                            self.progress_callback(progress)
                            
                    except Exception as aug_error:
                        self._log(f"❌ Ошибка аугментации изображения {i}.{aug_idx}: {aug_error}")
                        continue
            
            self._log(f"✅ Batch аугментация завершена. Создано {len(result_images)} изображений")
            return result_images
            
        except Exception as e:
            self._log(f"❌ Ошибка batch аугментации: {e}")
            return images  # Возвращаем оригинальные изображения

    def _match_gemini_and_ocr(self, gemini_data: Dict, ocr_data: Dict) -> Dict:
        """
        Сопоставляет данные Gemini с результатами OCR для создания обучающих данных.
        
        Args:
            gemini_data: Данные от Gemini API
            ocr_data: Результаты OCR
            
        Returns:
            Dict: Обработанная запись с метками
        """
        try:
            self._log("Сопоставление данных Gemini и OCR")
            
            # Проверяем, что данные существуют и имеют нужный формат
            if not gemini_data or not ocr_data:
                self._log("ОШИБКА: Отсутствуют данные Gemini или OCR")
                return {}
                
            # Проверяем наличие необходимых полей в OCR данных
            if 'words' not in ocr_data:
                self._log("ОШИБКА: OCR данные не содержат необходимого поля 'words'")
                return {}
                
            # Проверяем, что в словаре words содержатся данные о боксах
            if not ocr_data['words'] or not isinstance(ocr_data['words'], list) or not all(isinstance(word, dict) and 'bbox' in word for word in ocr_data['words']):
                self._log("ОШИБКА: OCR данные не содержат правильно структурированных данных о боксах в поле 'words'")
                return {}
                
            # Проверяем, что в gemini_data есть данные
            if not isinstance(gemini_data, dict):
                self._log(f"ОШИБКА: Неверный формат данных Gemini: {type(gemini_data)}")
                return {}
                
            # Получаем слова и их координаты из OCR
            ocr_words = [word.get('text', '') for word in ocr_data['words']]
            
            # Выводим первые 10 слов OCR для отладки
            if ocr_words:
                sample_words = ocr_words[:10]
                self._log(f"Образец OCR данных (первые 10 слов): {sample_words}")
            
            # Проверка формата данных Gemini - может быть либо с полем 'fields', либо напрямую с ключами
            has_fields = 'fields' in gemini_data and isinstance(gemini_data['fields'], list)
            has_direct_keys = any(key in gemini_data for key in ['invoice_number', 'date', 'total_amount', 'supplier_name'])
            
            # Проверка на наличие табличных данных
            table_keys = ['items', 'line_items', 'products', 'services', 'positions', 'table', 'tables']
            has_table_data = any(key in gemini_data for key in table_keys)
            
            if not (has_fields or has_direct_keys or has_table_data):
                self._log("ОШИБКА: Данные Gemini не содержат ни поле 'fields', ни прямые ключи с данными счета, ни табличные данные")
                return {}
                
            # Получаем слова и бокзы из OCR
            words = [word['text'] for word in ocr_data.get('words', [])]
            raw_bboxes = [word['bbox'] for word in ocr_data.get('words', [])]
            
            # Нормализуем bboxes используя размеры изображения из OCR данных
            image_width = ocr_data.get('width', 0)
            image_height = ocr_data.get('height', 0)
            
            if image_width > 0 and image_height > 0:
                self._log(f"Нормализация bboxes для изображения {image_width}x{image_height}")
                bboxes = [self.normalize_bbox(bbox, image_width, image_height) for bbox in raw_bboxes]
                # Выводим пример нормализации для отладки
                if len(raw_bboxes) > 0:
                    self._log(f"Пример нормализации: {raw_bboxes[0]} -> {bboxes[0]}")
            else:
                self._log(f"ПРЕДУПРЕЖДЕНИЕ: Размеры изображения не найдены в OCR данных, используем исходные bboxes")
                bboxes = raw_bboxes
            
            # Проверяем, что количество слов и боксов совпадает
            if len(words) != len(bboxes):
                self._log(f"ПРЕДУПРЕЖДЕНИЕ: Несоответствие количества слов ({len(words)}) и боксов ({len(bboxes)})")
                # Обрезаем до минимальной длины
                min_len = min(len(words), len(bboxes))
                words = words[:min_len]
                bboxes = bboxes[:min_len]
                
            # Если после обрезки данных не осталось, возвращаем пустой словарь
            if not words or not bboxes:
                self._log("ОШИБКА: После проверки согласованности данных не осталось слов или боксов")
                return {}
            
            # Инициализируем метки для всех слов как "O" (outside)
            labels = ["O"] * len(words)
            
            # Результат распознавания
            entities = {}
            
            # Обработка в зависимости от формата данных
            if has_fields:
                # Выводим список полей из Gemini
                self._log("Обработка полей Gemini в формате fields:")
                for idx, field_data in enumerate(gemini_data.get('fields', [])):
                    if isinstance(field_data, dict) and 'field_name' in field_data and 'field_value' in field_data:
                        self._log(f"  Поле {idx+1}: {field_data['field_name']} = '{field_data['field_value']}'")
                    else:
                        self._log(f"  Поле {idx+1}: {field_data} (некорректный формат)")
                
                # Обрабатываем каждое поле из Gemini в формате fields
                for field_data in gemini_data.get('fields', []):
                    # Проверяем, что поле содержит необходимые данные
                    if not isinstance(field_data, dict) or 'field_name' not in field_data or 'field_value' not in field_data:
                        self._log(f"ПРЕДУПРЕЖДЕНИЕ: Пропуск поля с недостаточными данными: {field_data}")
                        continue
                        
                    field_name = field_data['field_name']
                    field_value = field_data['field_value']
                    
                    # Пропускаем пустые значения
                    if not field_value or not field_name:
                        continue
                    
                    # Нормализуем значение поля для сравнения
                    field_value = str(field_value).strip().lower()
                    
                    # Пропускаем слишком короткие значения (менее 2 символов)
                    if len(field_value) < 2:
                        continue
                    
                    # Определяем тип поля для IOB2 тегов
                    field_type = self._normalize_field_type(field_name)
                    
                    # Поиск совпадений в тексте OCR
                    if 'currency' in field_name.lower() or field_type == 'CURRENCY':
                        self._log(f"Поиск частичного совпадения для поля 'CURRENCY': '{field_value}'")
                    elif 'date' in field_name.lower() or field_type == 'DATE':
                        self._log(f"Поиск частичного совпадения для поля 'DATE': '{field_value}'")
                    
                    # Маркируем соответствующие слова в тексте
                    field_matches = self._mark_matching_words(words, field_value, field_type, labels)
                    if field_matches:
                        entities[field_type] = field_value
            elif has_table_data:
                # Обрабатываем табличные данные
                for table_key in table_keys:
                    if table_key in gemini_data and isinstance(gemini_data[table_key], list):
                        table_items = gemini_data[table_key]
                        self._log(f"Обнаружены табличные данные в ключе '{table_key}': {len(table_items)} строк")
                        
                        for item_idx, item in enumerate(table_items):
                            if not isinstance(item, dict):
                                self._log(f"ПРЕДУПРЕЖДЕНИЕ: Элемент таблицы с индексом {item_idx} не является словарем")
                                continue
                                
                            # Обрабатываем каждый элемент таблицы
                            self._process_table_item(item, words, labels, item_idx)
            
            if has_direct_keys:
                # Обрабатываем прямые ключи из словаря (как из GeminiProcessor._convert_to_training_format)
                for key, value in gemini_data.items():
                    # Пропускаем служебные ключи и пустые значения
                    if key.startswith('note_') or key in ['source_image', 'processed_at'] or not value:
                        continue
                        
                    # Пропускаем табличные данные, так как они уже обработаны выше
                    if key in table_keys and isinstance(value, list):
                        continue
                        
                    # Если значение - словарь (например, supplier, customer), обрабатываем его поля
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_value:
                                # Нормализуем ключ для определения типа поля
                                field_name = f"{key}_{sub_key}"
                                field_type = self._normalize_field_type(field_name)
                                
                                # Нормализуем значение
                                field_value = str(sub_value).strip().lower()
                                
                                # Маркируем соответствующие слова
                                field_matches = self._mark_matching_words(words, field_value, field_type, labels)
                                if field_matches:
                                    entities[field_type] = field_value
                    else:
                        # Обрабатываем простое поле
                        field_type = self._normalize_field_type(key)
                        field_value = str(value).strip().lower()
                        
                        # Маркируем соответствующие слова
                        field_matches = self._mark_matching_words(words, field_value, field_type, labels)
                        if field_matches:
                            entities[field_type] = field_value
            
            # Подсчитываем количество сопоставленных полей
            tag_counts = {}
            for label in labels:
                if label != "O":
                    tag_type = label.split('-')[1] if '-' in label else label
                    tag_counts[tag_type] = tag_counts.get(tag_type, 0) + 1
            
            entity_count = len(entities)
            b_count = sum(1 for label in labels if label.startswith('B-'))
            
            self._log(f"Сопоставлено полей: {entity_count}, распределение меток: {tag_counts}")
            
            # Выводим детальную информацию о сопоставлениях для каждого поля
            for field_type, field_value in entities.items():
                b_labels = [i for i, label in enumerate(labels) if label == f'B-{field_type}']
                i_labels = [i for i, label in enumerate(labels) if label == f'I-{field_type}']
                
                if b_labels:
                    matched_words = []
                    for b_idx in b_labels:
                        # Собираем группу слов, начиная с текущего B- и включая все последующие I-
                        group = [words[b_idx]]
                        i = b_idx + 1
                        while i < len(labels) and labels[i].startswith(f'I-{field_type}'):
                            group.append(words[i])
                            i += 1
                        matched_words.append(' '.join(group))
                    
                    self._log(f"Поле '{field_type}': '{field_value}' => {matched_words}")
                else:
                    self._log(f"Поле '{field_type}': '{field_value}' не сопоставлено с текстом OCR")
            
            # Возвращаем словарь с полями 'words', 'bboxes', 'labels' и 'entities'
            return {
                'words': words,
                'bboxes': bboxes,
                'labels': labels,
                'entities': entities
            }
            
        except Exception as e:
            self._log(f"Ошибка при сопоставлении данных Gemini и OCR: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return {}

        # УЛУЧШЕННЫЕ ФУНКЦИИ ДЛЯ DATA_PREPARATOR.PY
    # Скопируйте эти функции в класс TrainingDataPreparator

    def _mark_matching_words(self, words, field_value, field_type, labels):
        """
        Маркирует слова, соответствующие значению поля. УЛУЧШЕННАЯ ВЕРСИЯ.
        """
        if not field_value or not words:
            return False

        # Нормализуем значение поля
        field_value = str(field_value).strip().lower()

        # Пропускаем значения "n/a", "null", "none" и т.п.
        skip_values = {'n/a', 'null', 'none', 'не указано', 'не заполнено', '-', '—', 'нет', 'пусто'}
        if field_value in skip_values:
            self._log(f"Пропускаем служебное значение для поля '{field_type}': '{field_value}'")
            return False

        # ЭТАП 1: Поиск точных числовых совпадений (высокий приоритет)
        if self._is_numeric_field(field_value):
            found_numeric = self._find_numeric_match_improved(words, field_value, field_type, labels)
            if found_numeric:
                return True

        # ЭТАП 2: Поиск точных совпадений фраз
        field_tokens = field_value.split()
        if len(field_tokens) > 1:
            found_exact = self._find_exact_phrase_match_improved(words, field_value, field_type, labels)
            if found_exact:
                return True

        # ЭТАП 3: Поиск по отдельным значимым токенам
        found_tokens = self._find_token_sequence_match_improved(words, field_tokens, field_type, labels)
        if found_tokens:
            return True

        # ЭТАП 4: Улучшенный частичный поиск
        found_partial = self._find_improved_partial_match(words, field_value, field_type, labels)
        if found_partial:
            return True

        # ЭТАП 5: Контекстный поиск
        found_contextual = self._find_contextual_match_improved(words, field_value, field_type, labels)
        if found_contextual:
            return True

        self._log(f"Не найдено совпадений для поля '{field_type}': '{field_value}'")
        return False

    def _find_numeric_match_improved(self, words, field_value, field_type, labels):
        """Улучшенный поиск числовых значений."""
        # Извлекаем все числа из значения поля
        field_numbers = re.findall(r'\d+(?:[.,]\d+)?', field_value)
        if not field_numbers:
            return False

        # Нормализуем числа (запятые в точки)
        field_numbers_norm = [num.replace(',', '.') for num in field_numbers]

        for i, word in enumerate(words):
            if labels[i] != "O":
                continue

            word_numbers = re.findall(r'\d+(?:[.,]\d+)?', word)
            word_numbers_norm = [num.replace(',', '.') for num in word_numbers]

            # Проверяем точные совпадения чисел
            for field_num in field_numbers_norm:
                for word_num in word_numbers_norm:
                    if field_num == word_num and len(field_num) >= 2:  # Значимые числа
                        labels[i] = f"B-{field_type}"
                        self._log(f"✅ ЧИСЛОВОЕ совпадение '{field_type}': '{field_value}' => '{word}' (число: {field_num})")

                        # Ищем продолжение рядом
                        self._extend_numeric_field_improved(words, i, field_value, field_type, labels)
                        return True

        return False

    def _find_exact_phrase_match_improved(self, words, field_value, field_type, labels):
        """Улучшенный поиск точных фраз."""
        field_value_clean = self._normalize_text_for_matching(field_value)
        field_tokens = field_value_clean.split()

        if len(field_tokens) < 2:
            return False

        # Пробуем разные размеры окна
        max_window = min(len(field_tokens) + 1, 6)

        for window_size in range(max_window, 1, -1):
            for i in range(len(words) - window_size + 1):
                if any(labels[j] != "O" for j in range(i, i + window_size)):
                    continue

                window_words = [self._normalize_text_for_matching(words[j]) for j in range(i, i + window_size)]
                window_text = " ".join(window_words)

                # Проверяем соответствие
                if self._is_phrase_match(window_text, field_value_clean):
                    for j in range(i, i + window_size):
                        labels[j] = f"B-{field_type}" if j == i else f"I-{field_type}"

                    matched_words = " ".join(words[i:i+window_size])
                    self._log(f"✅ ФРАЗОВОЕ совпадение '{field_type}': '{field_value}' => '{matched_words}'")
                    return True

        return False

    def _find_token_sequence_match_improved(self, words, field_tokens, field_type, labels):
        """Улучшенный поиск последовательности токенов."""
        if len(field_tokens) < 2:
            return False

        # Фильтруем значимые токены
        significant_tokens = [token for token in field_tokens if len(token) >= 2]
        if len(significant_tokens) < 2:
            return False

        found_positions = []

        for token in significant_tokens:
            token_clean = self._normalize_text_for_matching(token)
            best_match = None
            best_score = 0

            for i, word in enumerate(words):
                if labels[i] != "O":
                    continue

                word_clean = self._normalize_text_for_matching(word)
                score = self._calculate_token_match_score(word_clean, token_clean)

                if score > 0.8 and score > best_score:
                    best_match = i
                    best_score = score

            if best_match is not None:
                found_positions.append(best_match)

        # Требуем найти хотя бы 60% токенов
        if len(found_positions) >= len(significant_tokens) * 0.6:
            # Маркируем найденные позиции
            found_positions.sort()
            for idx, pos in enumerate(found_positions):
                labels[pos] = f"B-{field_type}" if idx == 0 else f"I-{field_type}"

            self._log(f"✅ ТОКЕНЫ '{field_type}': найдено {len(found_positions)}/{len(significant_tokens)} токенов")
            return True

        return False

    def _find_improved_partial_match(self, words, field_value, field_type, labels):
        """Улучшенный частичный поиск."""
        field_clean = self._normalize_text_for_matching(field_value)

        # Повышенные требования к частичному совпадению
        min_length = max(3, len(field_clean) // 3)

        for i, word in enumerate(words):
            if labels[i] != "O":
                continue

            word_clean = self._normalize_text_for_matching(word)

            if len(word_clean) < min_length:
                continue

            # Вычисляем разные виды совпадений
            scores = []

            # Вхождение
            if word_clean in field_clean or field_clean in word_clean:
                min_len = min(len(word_clean), len(field_clean))
                max_len = max(len(word_clean), len(field_clean))
                scores.append(min_len / max_len)

            # Схожесть начала/конца
            if len(word_clean) >= 4 and len(field_clean) >= 4:
                start_match = word_clean[:4] == field_clean[:4]
                end_match = word_clean[-3:] == field_clean[-3:]
                if start_match or end_match:
                    scores.append(0.7)

            # Левенштейн для коротких строк
            if len(word_clean) <= 15 and len(field_clean) <= 15:
                scores.append(self._levenshtein_similarity(word_clean, field_clean))

            best_score = max(scores) if scores else 0

            if best_score >= 0.75:  # Строгий порог
                labels[i] = f"B-{field_type}"
                self._log(f"✅ ЧАСТИЧНОЕ совпадение '{field_type}': '{field_value}' => '{word}' (score: {best_score:.2f})")
                return True

        return False

    def _find_contextual_match_improved(self, words, field_value, field_type, labels):
        """Улучшенный контекстный поиск."""
        # Контекстные маркеры для типов полей
        context_markers = {
            'INVOICE_ID': ['№', 'номер', 'счет', 'invoice', 'number', 'от'],
            'DATE': ['дата', 'от', 'date', 'г.', 'года', 'число'],
            'TOTAL': ['итого', 'сумма', 'всего', 'total', 'sum', 'к', 'оплате'],
            'TAX_ID': ['инн', 'inn', 'налоговый', 'идентификатор'],
            'COMPANY': ['ооо', 'оао', 'зао', 'ип', 'ltd', 'llc', 'inc', 'организация', 'поставщик']
        }

        markers = context_markers.get(field_type, [])
        if not markers:
            return False

        field_clean = self._normalize_text_for_matching(field_value)

        # Ищем подходящие слова рядом с маркерами
        for i, word in enumerate(words):
            if labels[i] != "O":
                continue

            word_clean = self._normalize_text_for_matching(word)

            # Проверяем соответствие слова типу поля
            if not self._is_suitable_for_field_type_improved(word_clean, field_clean, field_type):
                continue

            # Ищем маркеры в окрестности ±4 слова
            context_found = False
            for j in range(max(0, i-4), min(len(words), i+5)):
                if j == i:
                    continue
                context_word = self._normalize_text_for_matching(words[j])
                if any(marker in context_word for marker in markers):
                    context_found = True
                    break

            if context_found:
                labels[i] = f"B-{field_type}"
                self._log(f"✅ КОНТЕКСТНОЕ совпадение '{field_type}': '{word}' рядом с маркерами")
                return True

        return False

    def _is_suitable_for_field_type_improved(self, word, field_value, field_type):
        """Улучшенная проверка соответствия слова типу поля."""
        if len(word) < 2:
            return False

        if field_type == 'TAX_ID':
            # ИНН: 10-12 цифр
            return bool(re.match(r'^\d{10,12}$', word))
        elif field_type == 'INVOICE_ID':
            # Номер счета: содержит цифры, не слишком длинный
            return bool(re.search(r'\d+', word)) and len(word) <= 20
        elif field_type == 'DATE':
            # Дата: цифры или названия месяцев
            date_patterns = r'\d{1,4}|январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр'
            return bool(re.search(date_patterns, word))
        elif field_type == 'TOTAL':
            # Сумма: обязательно должна содержать цифры
            return bool(re.search(r'\d+', word))
        elif field_type == 'COMPANY':
            # Название компании: любой текст длиннее 2 символов
            return len(word) >= 2

        # Общая проверка на схожесть
        return self._calculate_token_match_score(word, field_value) > 0.4

    def _normalize_text_for_matching(self, text):
        """Улучшенная нормализация для сопоставления."""
        if not text:
            return ""

        text = str(text).lower().strip()

        # Убираем только кавычки и скобки, оставляем цифры и важные символы
        remove_chars = ['"', "'", '«', '»', '(', ')', '[', ']', '{', '}']
        for char in remove_chars:
            text = text.replace(char, ' ')

        # Нормализуем пробелы
        text = " ".join(text.split())

        return text

    def _is_phrase_match(self, text1, text2):
        """Проверяет совпадение фраз."""
        return text1 == text2 or (text1 in text2 and len(text1) / len(text2) > 0.7)

    def _calculate_token_match_score(self, word, token):
        """Вычисляет оценку совпадения токена."""
        if word == token:
            return 1.0
        if word in token or token in word:
            return min(len(word), len(token)) / max(len(word), len(token))
        return self._levenshtein_similarity(word, token)

    def _extend_numeric_field_improved(self, words, start_pos, field_value, field_type, labels):
        """Расширяет числовое поле на соседние слова."""
        extensions = ['руб', 'рублей', 'копеек', 'шт', 'штук', '%', 'процентов']

        for offset in [-1, 1]:
            pos = start_pos + offset
            if 0 <= pos < len(words) and labels[pos] == "O":
                word_clean = self._normalize_text_for_matching(words[pos])
                if any(ext in word_clean for ext in extensions):
                    labels[pos] = f"I-{field_type}"
                    self._log(f"  + расширение '{words[pos]}' для '{field_type}'")

    def _is_numeric_field(self, field_value):
        """Проверяет, является ли поле числовым."""
        return bool(re.search(r'\d', field_value))

    def _levenshtein_similarity(self, s1, s2):
        """Схожесть по Левенштейну."""
        if len(s1) < len(s2):
            return self._levenshtein_similarity(s2, s1)
        if len(s2) == 0:
            return 0.0

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        max_len = max(len(s1), len(s2))
        return 1.0 - (previous_row[-1] / max_len)

    def _is_matching_text(self, text1, text2, threshold=0.5):
        """
        Улучшенная проверка соответствия текстов с ПОНИЖЕННЫМ порогом.
        """
        try:
            if text1 == text2:
                return True

            t1 = self._normalize_text_for_matching(text1)
            t2 = self._normalize_text_for_matching(text2)

            if t1 == t2:
                return True

            # Пропускаем служебные значения
            skip_values = {'n/a', 'null', 'none', 'не указано', 'не заполнено', '-', '—', 'нет'}
            if t1 in skip_values or t2 in skip_values:
                return False

            # Проверка вхождения для длинных текстов
            if len(t1) >= 3 and len(t2) >= 3:
                if t1 in t2 or t2 in t1:
                    min_len = min(len(t1), len(t2))
                    max_len = max(len(t1), len(t2))
                    if min_len / max_len >= threshold:
                        return True

            # Общие числа
            if self._has_common_significant_numbers(t1, t2):
                return True

            # Схожесть для коротких текстов
            if len(t1) <= 30 and len(t2) <= 30:
                similarity = self._levenshtein_similarity(t1, t2)
                return similarity >= threshold

            return False

        except Exception as e:
            self._log(f"Ошибка при проверке соответствия текстов: {str(e)}")
            return False

    def _has_common_significant_numbers(self, str1, str2):
        """Проверяет наличие общих значимых чисел."""
        numbers1 = set(re.findall(r'\d+(?:[.,]\d+)?', str1))
        numbers2 = set(re.findall(r'\d+(?:[.,]\d+)?', str2))

        if not numbers1 or not numbers2:
            return False

        # Нормализуем числа
        numbers1_norm = {num.replace(',', '.') for num in numbers1}
        numbers2_norm = {num.replace(',', '.') for num in numbers2}

        # Ищем точные совпадения значимых чисел (длиннее 1 цифры)
        common = numbers1_norm.intersection(numbers2_norm)
        significant_common = {num for num in common if len(num.replace('.', '').replace(',', '')) > 1}

        return bool(significant_common)


    def _merge_bboxes(self, bboxes):
        """
        Объединяет несколько bounding boxes в один.
        
        Args:
            bboxes (List[List[int]]): Список bounding boxes для объединения
            
        Returns:
            List[int]: Объединенный bounding box
        """
        if not bboxes:
            return [0, 0, 0, 0]
            
        x_min = min(bbox[0] for bbox in bboxes)
        y_min = min(bbox[1] for bbox in bboxes)
        x_max = max(bbox[0] + bbox[2] for bbox in bboxes)
        y_max = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def _assign_iob2_labels(self, matched_entities, ocr_words, entity_types, label_other):
        """
        Назначает IOB2 метки для слов OCR на основе сопоставленных сущностей.
        
        Args:
            matched_entities (list): Список сопоставленных сущностей
            ocr_words (list): Список слов из OCR
            entity_types (list): Список типов сущностей из данных Gemini
            label_other (str): Метка для слов, не входящих в сущности
            
        Returns:
            list: Список IOB2 меток для каждого слова
        """
        num_ocr_words = len(ocr_words)
        iob2_labels = [label_other] * num_ocr_words

        if not matched_entities:
            self._log("[_assign_iob2_labels] Нет сопоставленных сущностей.")
            return iob2_labels

        # Сортируем сущности по их первому индексу в OCR
        matched_entities.sort(key=lambda x: min(x.get('word_ids', [9999])))

        for entity in matched_entities:
            label = entity['entity_type']
            word_ids = entity.get('word_ids', [])

            if not word_ids:
                self._log(f"[_assign_iob2_labels] Пропуск сущности '{label}' - нет word_ids.")
                continue
            
            if label not in entity_types:
                self._log(f"[_assign_iob2_labels] Пропуск сущности '{label}' - тип не найден в данных Gemini.")
                continue

            # Проверяем все индексы на валидность
            if any(idx < 0 or idx >= num_ocr_words for idx in word_ids):
                invalid_indices = [idx for idx in word_ids if idx < 0 or idx >= num_ocr_words]
                self._log(f"[_assign_iob2_labels] ОШИБКА: Индексы {invalid_indices} вне диапазона [0, {num_ocr_words-1}] для сущности '{label}'.")
                continue

            # Назначаем B- для первого слова и I- для остальных
            for i, idx in enumerate(word_ids):
                if iob2_labels[idx] == label_other:
                    iob2_labels[idx] = f"B-{label}" if i == 0 else f"I-{label}"
                elif not (iob2_labels[idx].startswith(f"B-{label}") or iob2_labels[idx].startswith(f"I-{label}")):
                    self._log(f"[_assign_iob2_labels] Конфликт меток для OCR слова с индексом {idx}. "
                             f"Текущая метка: {iob2_labels[idx]}, новая: {label}. Оставляем старую.")

        # Проверяем корректность последовательности меток
        for i in range(1, len(iob2_labels)):
            current_label = iob2_labels[i]
            prev_label = iob2_labels[i-1]
            
            # Если текущая метка начинается с I-, но предыдущая не соответствует той же сущности
            if current_label.startswith('I-'):
                entity_type = current_label[2:]  # Получаем тип сущности после 'I-'
                if not (prev_label == f"B-{entity_type}" or prev_label == f"I-{entity_type}"):
                    # Исправляем на B-, так как это начало новой сущности
                    iob2_labels[i] = f"B-{entity_type}"
                    self._log(f"[_assign_iob2_labels] Исправлена некорректная последовательность меток: индекс {i}")

        return iob2_labels

    def _create_huggingface_dataset(self, processed_dataset_path, entity_types, label_other):
        """
        Создает Hugging Face Dataset из обработанных файлов с GPU ускорением.
        
        Args:
            processed_dataset_path (str): Путь к папке с .ocr.json, .iob2.json и изображениями.
            entity_types (list): Список базовых типов сущностей.
            label_other (str): Метка для токенов вне сущностей.
            
        Returns:
            datasets.Dataset или None: Сформированный датасет или None в случае ошибки.
        """
        try:
            self._log(f"🚀 Создание Hugging Face Dataset с GPU ускорением из: {processed_dataset_path}")
            
            # Собираем все данные для batch обработки
            all_images = []
            all_words_list = []
            all_bboxes_list = []
            all_labels_list = []
            
            # Получаем список всех файлов
            all_files = os.listdir(processed_dataset_path)
            ocr_files = [f for f in all_files if f.endswith('.ocr.json')]
            
            self._log(f"📊 Найдено {len(ocr_files)} файлов для обработки")
            
            for i, ocr_file in enumerate(ocr_files):
                try:
                    base_name = ocr_file[:-9]  # Убираем '.ocr.json'
                    image_file = next((f for f in all_files if f.startswith(base_name) and 
                                     any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])), None)
                    iob2_file = base_name + '.iob2.json'
                    
                    if not image_file or not os.path.exists(os.path.join(processed_dataset_path, iob2_file)):
                        self._log(f"⚠️ Пропуск {ocr_file}: отсутствует изображение или IOB2 файл")
                        continue
                    
                    # Загружаем OCR данные
                    with open(os.path.join(processed_dataset_path, ocr_file), 'r', encoding='utf-8') as f:
                        ocr_data = json.load(f)
                    
                    # Загружаем IOB2 метки
                    with open(os.path.join(processed_dataset_path, iob2_file), 'r', encoding='utf-8') as f:
                        iob2_data = json.load(f)
                    
                    # Загружаем и обрабатываем изображение
                    image_path = os.path.join(processed_dataset_path, image_file)
                    image = Image.open(image_path).convert("RGB")
                    
                    # Получаем данные
                    words = [word['text'] for word in ocr_data.get('words', [])]
                    width, height = image.size
                    bboxes = [self.normalize_bbox(word.get('bbox', [0, 0, 0, 0]), width, height) 
                             for word in ocr_data.get('words', [])]
                    labels = iob2_data.get('labels', [])
                    
                    if not words or not bboxes or not labels or len(words) != len(bboxes) or len(words) != len(labels):
                        self._log(f"⚠️ Пропуск {ocr_file}: несоответствие длин данных")
                        continue
                    
                    # Добавляем данные в batch списки
                    all_images.append(image)
                    all_words_list.append(words)
                    all_bboxes_list.append(bboxes)
                    all_labels_list.append(labels)
                    
                    # Обновляем прогресс загрузки
                    if self.progress_callback and i % 5 == 0:
                        progress = int((i / len(ocr_files)) * 50)  # 50% на загрузку данных
                        self.progress_callback(progress)
                        
                except Exception as file_error:
                    self._log(f"❌ Ошибка при загрузке файла {ocr_file}: {file_error}")
                    continue
            
            if not all_images:
                self._log("❌ Нет данных для создания датасета")
                return None
            
            self._log(f"📁 Загружено {len(all_images)} примеров. Начинаем batch токенизацию...")
            
            # Используем batch токенизацию с GPU ускорением
            tokenized_data = self._batch_tokenize_layoutlm(
                images=all_images,
                words_list=all_words_list,
                bboxes_list=all_bboxes_list,
                labels_list=all_labels_list,
                model_name="microsoft/layoutlmv3-base"
            )
            
            if not tokenized_data:
                self._log("❌ Ошибка при batch токенизации")
                return None
            
            # Создаем датасет
            dataset = Dataset.from_dict(tokenized_data)
            
            self._log(f"✅ Датасет создан успешно с GPU ускорением. Размер: {len(dataset)}")
            self._log(f"💾 Поля датасета: {list(dataset.column_names)}")
            
            # Финальный прогресс
            if self.progress_callback:
                self.progress_callback(100)
            
            return dataset
            
        except Exception as e:
            self._log(f"❌ Ошибка при создании датасета: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _convert_labels_to_ids(self, labels, label_other, entity_types):
        """
        Конвертирует текстовые метки в числовые ID.
        
        Args:
            labels (list): Список текстовых меток
            label_other (str): Метка для токенов вне сущностей
            entity_types (list): Список типов сущностей
            
        Returns:
            list: Список числовых ID меток
        """
        # Создаем словарь меток
        label2id = {label_other: 0}
        current_id = 1
        for entity_type in entity_types:
            label2id[f"B-{entity_type}"] = current_id
            current_id += 1
            label2id[f"I-{entity_type}"] = current_id
            current_id += 1
        
        # Конвертируем метки в ID
        return [label2id.get(label, 0) for label in labels]

    def split_and_save_dataset(self, hf_dataset, base_output_folder, split_ratio=0.1):
        """
        Разделяет датасет на train и validation и сохраняет его.
        
        Args:
            hf_dataset (Dataset): Hugging Face Dataset для разделения 
            base_output_folder (str): Базовая папка для сохранения датасета
            split_ratio (float): Доля данных для validation (0.0-1.0)
            
        Returns:
            DatasetDict или None: Разделенный датасет или None в случае ошибки
        """
        try:
            self._log(f"Разделение датасета. train_test_split={1-split_ratio}/{split_ratio}")
            
            # Проверяем размер датасета
            dataset_size = len(hf_dataset)
            if dataset_size < 2:
                self._log("Датасет слишком маленький для разделения")
                return None
            
            # Создаем директории для сохранения
            train_dir = os.path.join(base_output_folder, 'train')
            val_dir = os.path.join(base_output_folder, 'validation')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            # Разделяем датасет
            split_dataset = hf_dataset.train_test_split(test_size=split_ratio)
            
            # Создаем DatasetDict
            dataset_dict = DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            
            # Сохраняем датасет
            dataset_dict.save_to_disk(base_output_folder)
            
            # Копируем изображения в соответствующие папки
            for split_name, split_data in dataset_dict.items():
                output_dir = os.path.join(base_output_folder, split_name, 'images')
                os.makedirs(output_dir, exist_ok=True)
                
                # Проверяем наличие поля 'image_path' в датасете (было 'image_path')
                if 'image_path' in split_data.column_names:
                    image_paths = split_data['image_path']
                    self._log(f"Копирование {len(image_paths)} изображений в {output_dir}")
                    
                    for image_path in image_paths:
                        if os.path.exists(image_path):
                            image_name = os.path.basename(image_path)
                            output_path = os.path.join(output_dir, image_name)
                            shutil.copy2(image_path, output_path)
                else:
                    self._log(f"ПРЕДУПРЕЖДЕНИЕ: В датасете отсутствует поле 'image_path', пропускаем копирование изображений")
            
            self._log(f"Датасет успешно разделен и сохранен в {base_output_folder}")
            self._log(f"Train size: {len(dataset_dict['train'])}")
            self._log(f"Validation size: {len(dataset_dict['validation'])}")
            
            return dataset_dict
            
        except Exception as e:
            self._log(f"Ошибка при разделении датасета: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def validate_image(self, image_path: str) -> Tuple[bool, str, Optional[Image.Image]]:
        """
        Проверяет корректность изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Tuple[bool, str, Optional[Image.Image]]: (успех, сообщение об ошибке, изображение)
        """
        try:
            if not os.path.exists(image_path):
                return False, f"Файл не существует: {image_path}", None
                
            image = Image.open(image_path)
            
            # Проверяем размер
            if image.size[0] < 100 or image.size[1] < 100:
                return False, f"Изображение слишком маленькое: {image.size}", None
                
            # Проверяем формат
            if image.format not in ['JPEG', 'PNG', 'TIFF']:
                return False, f"Неподдерживаемый формат: {image.format}", None
                
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return True, "", image
            
        except Exception as e:
            return False, f"Ошибка при проверке изображения: {str(e)}", None

    def normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        """
        Нормализует координаты bbox для LayoutLM (диапазон [0, 1023]).
        
        Args:
            bbox: Координаты [x1, y1, x2, y2]
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            List[int]: Нормализованные координаты [x1, y1, x2, y2]
        """
        try:
            if not bbox or len(bbox) != 4:
                return [0, 0, 0, 0]
                
            if width <= 0 or height <= 0:
                return [0, 0, 0, 0]
                
            # Проверяем корректность координат
            x1, y1, x2, y2 = bbox
            
            # Клампим координаты к границам изображения
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
                
            # Нормализуем в диапазон [0, 1023] для LayoutLM
            x1_norm = max(0, min(1023, int(1023 * x1 / width)))
            y1_norm = max(0, min(1023, int(1023 * y1 / height)))
            x2_norm = max(0, min(1023, int(1023 * x2 / width)))
            y2_norm = max(0, min(1023, int(1023 * y2 / height)))
            
            # Проверяем порядок координат
            if x2_norm < x1_norm:
                x1_norm, x2_norm = x2_norm, x1_norm
            if y2_norm < y1_norm:
                y1_norm, y2_norm = y2_norm, y1_norm
                
            # Убеждаемся что bbox не нулевой
            if x1_norm == x2_norm:
                x2_norm = min(1023, x1_norm + 1)
            if y1_norm == y2_norm:
                y2_norm = min(1023, y1_norm + 1)
                
            return [x1_norm, y1_norm, x2_norm, y2_norm]
            
        except Exception as e:
            self._log(f"Ошибка нормализации bbox: {str(e)}")
            return [0, 0, 0, 0]

    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Применяет аугментацию к изображению"""
        augmented = self.augmentation(image=np.array(image))
        return Image.fromarray(augmented['image'])

    def compute_class_weights(self, labels):
        """Вычисляет веса классов для балансировки"""
        # Преобразуем IOB2 метки в числовые классы
        label_counts = Counter(label for doc_labels in labels for label in doc_labels)
        unique_labels = sorted(label_counts.keys())
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Преобразуем метки в числовой формат
        numeric_labels = [label_to_id[label] for doc_labels in labels for label in doc_labels]
        
        # Вычисляем веса
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(numeric_labels),
            y=numeric_labels
        )
        
        # Возвращаем словарь весов
        return {label: weight for label, weight in zip(unique_labels, class_weights)}

    def prepare_dataset(self, images, annotations, split_ratio=0.2, augment=True):
        """Подготовка датасета с аугментацией и балансировкой"""
        processed_images = []
        processed_annotations = []
        
        for img, ann in zip(images, annotations):
            # Добавляем оригинальное изображение
            processed_images.append(img)
            processed_annotations.append(ann)
            
            # Применяем аугментацию если включена
            if augment:
                aug_img = self.apply_augmentation(img)
                processed_images.append(aug_img)
                processed_annotations.append(ann)  # Используем те же аннотации
                
        # Вычисляем веса классов
        class_weights = self.compute_class_weights(processed_annotations)
        
        # Разделяем на train/val
        split_idx = int(len(processed_images) * (1 - split_ratio))
        
        train_images = processed_images[:split_idx]
        train_annotations = processed_annotations[:split_idx]
        val_images = processed_images[split_idx:]
        val_annotations = processed_annotations[split_idx:]
        
        return {
            'train': (train_images, train_annotations),
            'val': (val_images, val_annotations),
            'class_weights': class_weights
        }

    def _process_document_page(self, 
                             image: Image.Image,
                             base_name: str,
                             dataset_folder: str,
                             training_prompt: str) -> Optional[Dict]:
        """
        Обрабатывает одну страницу документа.
        
        Args:
            image: Изображение страницы
            base_name: Базовое имя файла
            dataset_folder: Папка датасета
            training_prompt: Промпт для Gemini
            
        Returns:
            Optional[Dict]: Данные для датасета или None при ошибке
        """
        try:
            # Сохраняем изображение
            image_path = os.path.join(dataset_folder, 'images', f"{base_name}.jpg")
            image.save(image_path, 'JPEG')
            
            # Получаем OCR данные
            ocr_data = self.ocr_processor.process_file(image_path)
            if not ocr_data or not ocr_data.get('words'):
                self._log(f"[ОШИБКА] OCR не вернул данные для файла: {base_name}")
                return None
                
            # Получаем данные от Gemini
            gemini_result = self.gemini_processor.process_file(image_path, training_prompt)
            if not gemini_result:
                self._log(f"Ошибка Gemini для {base_name}")
                return None
            
            # Выводим результаты от Gemini для лучшего понимания процесса
            self._log(f"Результат Gemini для {base_name}:")
            self._log(f"Структура ответа: {type(gemini_result)}")
            
            # Форматированный вывод JSON со структурой ответа
            gemini_json = json.dumps(gemini_result, ensure_ascii=False, indent=2)
            self._log(f"Данные Gemini API:\n{gemini_json}")
                
            # Сопоставляем данные
            matched_entities = self._match_gemini_and_ocr(gemini_result, ocr_data)
            if not matched_entities:
                self._log(f"Нет сопоставленных сущностей для {base_name}")
                return None
                
            # Создаем разметку
            words = ocr_data['words']
            width, height = image.size
            
            # Нормализуем bbox и создаем labels
            bboxes = []
            texts = []
            
            for word in words:
                # Получаем bbox из OCR результата
                bbox = word.get('bbox', None)
                if not bbox:
                    # Если bbox не найден, пробуем собрать его из координат
                    bbox = [
                        word.get('x', 0),
                        word.get('y', 0),
                        word.get('x', 0) + word.get('width', 0),
                        word.get('y', 0) + word.get('height', 0)
                    ]
                
                # Проверяем валидность bbox
                if not all(isinstance(coord, (int, float)) for coord in bbox) or len(bbox) != 4:
                    self._log(f"Некорректный bbox для слова '{word.get('text', '')}': {bbox}")
                    bbox = [0, 0, 0, 0]
                
                # Нормализуем bbox
                try:
                    normalized_bbox = self.normalize_bbox(bbox, width, height)
                except Exception as e:
                    self._log(f"Ошибка нормализации bbox {bbox}: {str(e)}")
                    normalized_bbox = [0, 0, 0, 0]
                
                # Проверяем нормализованный bbox
                if normalized_bbox == [0, 0, 0, 0]:
                    self._log(f"Пропуск слова '{word.get('text', '')}' с нулевым bbox")
                    continue
                
                if not (0 <= normalized_bbox[0] <= 1000 and 0 <= normalized_bbox[1] <= 1000 and
                       0 <= normalized_bbox[2] <= 1000 and 0 <= normalized_bbox[3] <= 1000):
                    self._log(f"Некорректный нормализованный bbox для слова '{word.get('text', '')}': {normalized_bbox}")
                    continue
                
                bboxes.append(normalized_bbox)
                texts.append(word['text'])
            
            if not texts or not bboxes:
                self._log(f"Нет валидных слов с bbox для {base_name}")
                return None
            
            # Создаем IOB2 разметку
            iob2_labels = self._assign_iob2_labels(matched_entities, words, list(gemini_result.keys()), "O")
            
            # Проверяем соответствие длин
            if len(texts) != len(bboxes) or len(texts) != len(iob2_labels):
                self._log(f"Несоответствие длин данных: texts={len(texts)}, bboxes={len(bboxes)}, labels={len(iob2_labels)}")
                return None
            
            # Сохраняем промежуточные данные для отладки
            debug_data = {
                'words': texts,
                'bboxes': bboxes,
                'labels': iob2_labels,
                'entities': matched_entities,
                'gemini_result': gemini_result  # Добавляем оригинальный ответ Gemini для сравнения
            }
            debug_path = os.path.join(dataset_folder, f"{base_name}_debug.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            # Выводим статистику по сопоставленным сущностям и меткам
            label_counts = {}
            for label in iob2_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            self._log(f"Статистика меток в датасете для {base_name}:")
            self._log(f"Всего слов: {len(texts)}")
            self._log(f"Всего уникальных меток: {len(label_counts)}")
            self._log(f"Распределение меток: {json.dumps(label_counts, ensure_ascii=False, indent=2)}")
            
            # Сравнение полей Gemini и меток в датасете
            self._log("Сравнение данных Gemini и записей в датасете:")
            gemini_fields = []
            
            # Собираем поля из ответа Gemini в зависимости от формата
            if 'fields' in gemini_result and isinstance(gemini_result['fields'], list):
                for field in gemini_result['fields']:
                    if isinstance(field, dict) and 'field_name' in field and 'field_value' in field:
                        field_type = self._normalize_field_type(field['field_name'])
                        field_value = field['field_value']
                        gemini_fields.append((field_type, field_value))
            else:
                # Обрабатываем прямые ключи
                for key, value in gemini_result.items():
                    if key not in ['source_image', 'processed_at', 'note_gemini']:
                        if isinstance(value, list):
                            # Табличные данные
                            self._log(f"Табличные данные '{key}' с {len(value)} строками")
                        elif isinstance(value, dict):
                            # Вложенные объекты (supplier, customer, etc)
                            for sub_key, sub_value in value.items():
                                gemini_fields.append((f"{key}_{sub_key}", sub_value))
                        else:
                            gemini_fields.append((key, value))
            
            # Оценка качества сопоставления
            b_count = sum(1 for label in iob2_labels if label.startswith('B-'))
            i_count = sum(1 for label in iob2_labels if label.startswith('I-'))
            total_fields = len(gemini_fields)
            
            self._log(f"Всего полей в Gemini: {total_fields}")
            self._log(f"Найдено начал полей (B-): {b_count}")
            self._log(f"Найдено продолжений полей (I-): {i_count}")
            
            if total_fields > 0:
                matching_rate = b_count / total_fields
                self._log(f"Процент сопоставленных полей: {matching_rate:.2%}")
                
                if matching_rate < 0.5:
                    self._log("ПРЕДУПРЕЖДЕНИЕ: Низкий процент сопоставления полей")
                elif matching_rate > 0.8:
                    self._log("ОТЛИЧНО: Высокий процент сопоставления полей")
            
            # Возвращаем словарь с полями 'words', 'bboxes', 'labels' и 'entities'
            return {
                'words': words,
                'bboxes': bboxes,
                'labels': iob2_labels,
                'entities': entities
            }
            
        except Exception as e:
            self._log(f"Ошибка обработки страницы {base_name}: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _create_dataset(self, processed_data, use_augmentation=True):
        """
        Создает датасет из обработанных данных.
        
        Args:
            processed_data: Список обработанных документов
            use_augmentation: Использовать ли аугментации
            
        Returns:
            Optional[Dataset]: Созданный датасет или None при ошибке
        """
        try:
            if not processed_data:
                self._log("ОШИБКА: Нет данных для создания датасета")
                return None
                
            # Подготавливаем данные
            dataset_dict = {
                'image_path': [],
                'words': [],
                'bboxes': [],
                'labels': []
            }
            
            for data_idx, data in enumerate(processed_data):
                try:
                    # Проверяем наличие необходимых полей
                    if not isinstance(data, dict):
                        self._log(f"ОШИБКА: Элемент данных {data_idx} не является словарем: {type(data)}")
                        continue
                        
                    required_fields = ['image_path', 'words', 'bboxes', 'labels']
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self._log(f"ОШИБКА: В элементе данных {data_idx} отсутствуют поля: {missing_fields}")
                        continue
                    
                    # Проверяем, что все массивы данных имеют одинаковую длину
                    words = data['words']
                    bboxes = data['bboxes']
                    labels = data['labels']
                    
                    if not isinstance(words, list) or not isinstance(bboxes, list) or not isinstance(labels, list):
                        self._log(f"ОШИБКА: words, bboxes или labels не являются списками в элементе {data_idx}")
                        continue
                        
                    if not words or not bboxes or not labels:
                        self._log(f"ОШИБКА: Пустые words, bboxes или labels в элементе {data_idx}")
                        continue
                        
                    if len(words) != len(bboxes) or len(words) != len(labels):
                        self._log(f"ОШИБКА: Разная длина words ({len(words)}), bboxes ({len(bboxes)}) и labels ({len(labels)}) в элементе {data_idx}")
                        
                        # Пытаемся выровнять длины массивов
                        min_len = min(len(words), len(bboxes), len(labels))
                        if min_len > 0:
                            self._log(f"Обрезаем массивы до одинаковой длины: {min_len}")
                            words = words[:min_len]
                            bboxes = bboxes[:min_len]
                            labels = labels[:min_len]
                        else:
                            self._log(f"Невозможно выровнять массивы, пропускаем элемент {data_idx}")
                            continue
                    
                    # Добавляем данные в датасет
                    dataset_dict['image_path'].append(data['image_path'])
                    dataset_dict['words'].append(words)
                    dataset_dict['bboxes'].append(bboxes)
                    dataset_dict['labels'].append(labels)
                    
                    # Добавляем аугментированные данные если нужно
                    if use_augmentation:
                        try:
                            image = Image.open(data['image_path'])
                            aug_image = self.apply_augmentation(image)
                            
                            # Сохраняем аугментированное изображение
                            aug_path = data['image_path'].replace('.jpg', '_aug.jpg')
                            aug_image.save(aug_path, 'JPEG')
                            
                            dataset_dict['image_path'].append(aug_path)
                            dataset_dict['words'].append(words)
                            dataset_dict['bboxes'].append(bboxes)
                            dataset_dict['labels'].append(labels)
                            
                        except Exception as e:
                            self._log(f"Ошибка аугментации для элемента {data_idx}: {str(e)}")
                            continue
                except Exception as data_error:
                    self._log(f"Ошибка при обработке элемента данных {data_idx}: {str(data_error)}")
                    continue
            
            # Проверяем, что в датасете есть данные
            if not dataset_dict['image_path']:
                self._log("ОШИБКА: Не удалось подготовить ни одного элемента для датасета")
                return None
                
            # Создаем датасет
            try:
                from datasets import Dataset, Features, Sequence, Value
                
                features = Features({
                    'image_path': Value('string'),
                    'words': Sequence(Value('string')),
                    'bboxes': Sequence(Sequence(Value('int64'), length=4)),
                    'labels': Sequence(Value('string'))
                })
                
                dataset = Dataset.from_dict(dataset_dict, features=features)
                
                self._log(f"Создан датасет размером {len(dataset)} примеров")
                return dataset
            except Exception as dataset_error:
                self._log(f"Ошибка создания объекта Dataset: {str(dataset_error)}")
                return None
            
        except Exception as e:
            self._log(f"Ошибка создания датасета: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _create_iob2_labels(self, words, matched_entities):
        """Создает IOB2 разметку для слов на основе сопоставленных сущностей"""
        labels = ["O"] * len(words)
        for entity in matched_entities:
            label = entity['entity_type']
            indices = entity.get('word_ids', [])
            if not indices:
                continue
            
            for i, idx in enumerate(indices):
                if 0 <= idx < len(labels):
                    labels[idx] = f"{'B' if i == 0 else 'I'}-{label}"
        
        return labels

    def _is_image_file(self, filename):
        """
        Проверяет, является ли файл изображением или PDF на основе расширения.
        
        Args:
            filename (str): Имя файла для проверки
            
        Returns:
            bool: True, если это файл изображения или PDF, иначе False
        """
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.pdf']
        ext = os.path.splitext(filename)[1].lower()
        return ext in extensions

    def _process_single_image(self, image_path, base_name, train_folder, training_prompt):
        """
        Обрабатывает одно изображение или PDF файл для подготовки данных.
        
        Args:
            image_path (str): Путь к файлу изображения или PDF
            base_name (str): Базовое имя файла без расширения
            train_folder (str): Папка для сохранения обработанных данных
            training_prompt (str): Промпт для Gemini
            
        Returns:
            Optional[Dict]: Словарь с обработанными данными или None в случае ошибки
        """
        try:
            # Проверяем, является ли файл PDF
            if image_path.lower().endswith('.pdf'):
                # Настраиваем путь к Poppler
                poppler_path = self.app_config.POPPLER_PATH
                if not poppler_path or not os.path.exists(poppler_path):
                    self._log(f"[ОШИБКА] Путь к Poppler не найден: {poppler_path}")
                    return None
                
                # Конвертируем PDF в изображения
                images = convert_from_path(image_path, poppler_path=poppler_path)
                if not images:
                    self._log(f"[ОШИБКА] Не удалось конвертировать PDF в изображения: {image_path}")
                    return None
                
                # Сохраняем первую страницу как временное изображение
                temp_image_path = os.path.join(train_folder, f"{base_name}_temp.jpg")
                images[0].save(temp_image_path, 'JPEG')
                image_path = temp_image_path
            
            # Получаем OCR данные
            ocr_data = self.ocr_processor.process_file(image_path)
            if not ocr_data:
                self._log(f"[ОШИБКА] OCR не вернул данные для файла: {base_name}")
                return None
                
            # Проверяем, что ocr_data - словарь
            if not isinstance(ocr_data, dict):
                self._log(f"[ОШИБКА] OCR вернул не словарь: {type(ocr_data)}")
                return None
                
            # Проверяем наличие слов
            if 'words' not in ocr_data or not ocr_data.get('words'):
                self._log(f"[ОШИБКА] OCR не вернул слова для файла: {base_name}")
                return None
                
            # Проверяем, что words - список
            if not isinstance(ocr_data['words'], list):
                self._log(f"[ОШИБКА] OCR вернул слова не в виде списка: {type(ocr_data['words'])}")
                return None
            
            # Подготавливаем текст для Gemini
            words = []
            for word in ocr_data.get('words', []):
                if not isinstance(word, dict):
                    continue
                text = word.get('text', '').strip()
                if text:
                    words.append(text)
            
            if not words:
                self._log(f"[ОШИБКА] Не найдено текста для обработки в файле: {base_name}")
                return None
                
            text_for_gemini = ' '.join(words)
            
            # Проверяем длину текста
            if len(text_for_gemini) < 10:  # Минимальная длина для осмысленного текста
                self._log(f"[ОШИБКА] Слишком короткий текст для обработки: {text_for_gemini}")
                return None
            
            self._log(f"[ИНФО] Подготовлен текст для Gemini ({len(words)} слов)")
            
            # Обрабатываем через Gemini
            gemini_data = self.gemini_processor.process_file(text_for_gemini, custom_prompt=training_prompt)
            if not gemini_data:
                self._log(f"[ОШИБКА] Gemini не вернул данные для файла: {base_name}")
                return None
                
            # Проверяем, что gemini_data - словарь
            if not isinstance(gemini_data, dict):
                self._log(f"[ОШИБКА] Gemini вернул не словарь: {type(gemini_data)}")
                if isinstance(gemini_data, str):
                    try:
                        # Пытаемся преобразовать строку в JSON
                        gemini_data = json.loads(gemini_data)
                    except json.JSONDecodeError:
                        self._log(f"[ОШИБКА] Не удалось преобразовать ответ Gemini в JSON")
                        return None
                else:
                    return None
            
            # Сопоставляем сущности из Gemini с OCR данными
            matched_data = self._match_gemini_and_ocr(gemini_data, ocr_data)
            if not matched_data:
                self._log(f"[DataPreparator] [_process_single_image] Нет сопоставленных сущностей для файла: {base_name}")
                return None
                
            # Проверяем, что matched_data - словарь с необходимыми полями
            required_fields = ['words', 'bboxes', 'labels']
            if not isinstance(matched_data, dict) or not all(field in matched_data for field in required_fields):
                self._log(f"[ОШИБКА] _match_gemini_and_ocr вернул неполные данные: {type(matched_data)}")
                return None
                
            # Проверяем, что есть хотя бы одна метка не "O"
            if all(label == "O" for label in matched_data.get('labels', [])):
                self._log(f"[ОШИБКА] В результате сопоставления все метки имеют значение 'O' (нет найденных сущностей)")
                return None
            
            # Сохраняем изображение в папку датасета
            try:
                image = Image.open(image_path)
                output_image_path = os.path.join(train_folder, f"{base_name}.jpg")
                image.save(output_image_path, 'JPEG')
            except Exception as img_error:
                self._log(f"[ОШИБКА] Не удалось обработать изображение {image_path}: {str(img_error)}")
                return None
            
            # Формируем результат
            result = {
                'image_path': output_image_path,
                'words': matched_data['words'],
                'bboxes': matched_data['bboxes'],
                'labels': matched_data['labels']
            }
            
            # Сохраняем отладочную информацию
            try:
                debug_info = {
                    'ocr_words_count': len(ocr_data['words']),
                    'matched_words_count': len(matched_data['words']),
                    'labels_count': len(matched_data['labels']),
                    'gemini_entities': list(gemini_data.keys()) if isinstance(gemini_data, dict) else None
                }
                debug_path = os.path.join(train_folder, f"{base_name}_debug.json")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_info, f, ensure_ascii=False, indent=2)
            except Exception as debug_error:
                self._log(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось сохранить отладочную информацию: {str(debug_error)}")
            
            return result
            
        except Exception as e:
            self._log(f"[ОШИБКА] Ошибка обработки файла {base_name}: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def prepare_for_layoutlm(self, source_folder: str, dataset_name: str, training_prompt: str = None) -> Optional[str]:
        """
        Подготавливает данные для обучения LayoutLM.
        
        Args:
            source_folder (str): Путь к папке с исходными документами
            dataset_name (str): Имя для создаваемого датасета
            training_prompt (str, optional): Промпт для Gemini (если None, генерируется автоматически)
            
        Returns:
            Optional[str]: Путь к подготовленному датасету или None в случае ошибки
        """
        try:
            # 🎯 Автоматическая генерация промпта на основе настроек полей таблицы
            if not training_prompt:
                training_prompt = self.get_training_prompt("layoutlm")
                self._log("🤖 Промпт сгенерирован автоматически на основе настроек полей таблицы")
            else:
                self._log("📝 Используется пользовательский промпт")
            
            self._log(f"📏 Промпт ({len(training_prompt)} символов):")
            # Выводим первые 300 символов промпта для проверки
            preview = training_prompt[:300] + "..." if len(training_prompt) > 300 else training_prompt
            self._log(f"📖 Превью промпта:\n{preview}")
            
            # Проверяем существование папки с исходными документами
            if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
                self._log(f"ОШИБКА: Папка с исходными документами не найдена: {source_folder}")
                return None
                
            # Проверяем наличие файлов для обработки
            files = [f for f in os.listdir(source_folder) if self._is_image_file(f)]
            if not files:
                self._log(f"ОШИБКА: В папке {source_folder} не найдено изображений или PDF-файлов")
                return None
            
            # Создаем папку для датасета
            dataset_folder = os.path.join(self.app_config.TRAINING_DATASETS_PATH, dataset_name)
            os.makedirs(dataset_folder, exist_ok=True)
            
            # Сохраняем настройки датасета включая информацию об источнике полей
            self._save_dataset_metadata(dataset_folder, {
                'source_folder': os.path.relpath(source_folder, self.app_config.PROJECT_ROOT),
                'dataset_name': dataset_name,
                'task_type': 'layoutlm',
                'training_prompt': training_prompt,
                'fields_source': 'field_manager' if self.field_manager else 'static',
                'active_fields': [f.id for f in self.field_manager.get_enabled_fields()] if self.field_manager else None,
                'entity_types': self.get_entity_types_from_fields()
            })
            
            # Обрабатываем каждый файл в исходной папке
            processed_files = []
            self._log(f"Начинаем обработку {len(files)} файлов из {source_folder}")
            
            for idx, filename in enumerate(files):
                self._log(f"Обработка файла {idx+1}/{len(files)}: {filename}")
                if self._is_image_file(filename):
                    image_path = os.path.join(source_folder, filename)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Обрабатываем изображение
                    result = self._process_single_image(image_path, base_name, dataset_folder, training_prompt)
                    if result:
                        processed_files.append(result)
                    else:
                        self._log(f"ОШИБКА: Не удалось обработать файл {filename}")
            
            if not processed_files:
                self._log("ОШИБКА: Нет успешно обработанных файлов для создания датасета")
                return None
                
            self._log(f"Успешно обработано {len(processed_files)} файлов из {len(files)}")
            
            # Создаем и сохраняем датасет
            self._log("Создание датасета из обработанных файлов...")
            dataset = self._create_dataset(processed_files)
            if dataset is None:
                self._log("ОШИБКА: Не удалось создать датасет")
                return None
                
            # Разделяем на train/validation и сохраняем
            output_path = os.path.join(dataset_folder, "dataset_dict")
            self._log(f"Разделение датасета на train/validation и сохранение в {output_path}...")
            self.split_and_save_dataset(dataset, output_path)
            
            self._log(f"Подготовка данных для LayoutLM завершена. Путь к датасету: {output_path}")
            return output_path
            
        except Exception as e:
            self._log(f"ОШИБКА при подготовке данных для LayoutLM: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def create_full_dataset(self, processed_records: List[Dict], output_dir: str = None) -> Optional[Dataset]:
        """
        Создает полный датасет из обработанных записей.
        
        Args:
            processed_records: Список обработанных записей
            output_dir: Директория для сохранения датасета (опционально)
            
        Returns:
            Dataset или None: Датасет Hugging Face или None в случае ошибки
        """
        try:
            self._log(f"Создание полного датасета из {len(processed_records)} записей")
            
            # Проверяем, что records не пустой
            if not processed_records:
                self._log("ОШИБКА: Список обработанных записей пуст")
                return None
            
            # Проверяем структуру записей для обеспечения согласованности
            required_fields = ['image_path', 'words', 'bboxes', 'labels']
            for i, record in enumerate(processed_records):
                missing_fields = [field for field in required_fields if field not in record]
                if missing_fields:
                    self._log(f"ПРЕДУПРЕЖДЕНИЕ: Запись {i} не содержит поля: {missing_fields}")
                    # Добавляем пустые значения для отсутствующих полей
                    for field in missing_fields:
                        if field == 'image_path':
                            record[field] = ""
                        elif field == 'words':
                            record[field] = []
                        elif field == 'bboxes':
                            record[field] = []
                        elif field == 'labels':
                            record[field] = []
            
            # Фильтруем записи с пустыми или отсутствующими путями к изображениям
            valid_records = [record for record in processed_records if record.get('image_path') and os.path.exists(record['image_path'])]
            
            if len(valid_records) == 0:
                self._log("ОШИБКА: Нет записей с действительными путями к изображениям")
                return None
            
            if len(valid_records) < len(processed_records):
                self._log(f"ПРЕДУПРЕЖДЕНИЕ: Отфильтровано {len(processed_records) - len(valid_records)} записей с недействительными путями к изображениям")
            
            # Создаем Hugging Face Dataset
            from datasets import Dataset as HFDataset
            
            # Проверяем согласованность данных
            num_words = [len(record['words']) for record in valid_records]
            num_bboxes = [len(record['bboxes']) for record in valid_records]
            num_labels = [len(record['labels']) for record in valid_records]
            
            if not all(num_words[i] == num_bboxes[i] == num_labels[i] for i in range(len(valid_records))):
                self._log("ПРЕДУПРЕЖДЕНИЕ: Несогласованность в количестве words, bboxes и labels")
                # Исправляем несогласованности
                for record in valid_records:
                    max_len = max(len(record['words']), len(record['bboxes']), len(record['labels']))
                    # Выравниваем длины всех списков
                    if len(record['words']) < max_len:
                        record['words'].extend(["PAD"] * (max_len - len(record['words'])))
                    if len(record['bboxes']) < max_len:
                        record['bboxes'].extend([[0, 0, 0, 0]] * (max_len - len(record['bboxes'])))
                    if len(record['labels']) < max_len:
                        record['labels'].extend(["O"] * (max_len - len(record['labels'])))
                    # Обрезаем если нужно
                    record['words'] = record['words'][:max_len]
                    record['bboxes'] = record['bboxes'][:max_len]
                    record['labels'] = record['labels'][:max_len]
            
            # Создаем датасет
            try:
                dataset = HFDataset.from_dict({
                    'image_path': [record['image_path'] for record in valid_records],
                    'words': [record['words'] for record in valid_records],
                    'bboxes': [record['bboxes'] for record in valid_records],
                    'labels': [record['labels'] for record in valid_records]
                })
                
                # Выводим информацию о созданном датасете
                self._log(f"Создан датасет с {len(dataset)} записями")
                
                # Сохраняем датасет если задана директория
                if output_dir:
                    self._save_dataset(dataset, output_dir)
                
                return dataset
                
            except Exception as e:
                self._log(f"ОШИБКА при создании HFDataset: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
                return None
                
        except Exception as e:
            self._log(f"Ошибка при создании полного датасета: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _is_matching_text(self, text1, text2, threshold=0.5):
        """
        Проверяет, соответствуют ли два текста друг другу с учетом схожести.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            threshold: Порог схожести (0.0-1.0)
            
        Returns:
            bool: True, если тексты соответствуют друг другу
        """
        try:
            # Проверка на прямое соответствие
            if text1 == text2:
                return True
                
            # Нормализация текстов
            t1 = self._normalize_text(text1)
            t2 = self._normalize_text(text2)
            
            # Проверка на соответствие нормализованных текстов
            if t1 == t2:
                return True
                
            # Проверка на вхождение
            if t1 in t2 or t2 in t1:
                # Если один текст является подстрокой другого, проверяем относительную длину
                min_len = min(len(t1), len(t2))
                max_len = max(len(t1), len(t2))
                if min_len / max_len >= threshold:
                    return True
            
            # Проверка на вхождение важных частей (например, числовых последовательностей)
            if self._has_common_numbers(t1, t2):
                return True
                
            # Расчет расстояния Левенштейна для коротких текстов
            if len(t1) <= 20 and len(t2) <= 20:
                # Используем собственную реализацию расстояния Левенштейна вместо внешней библиотеки
                distance = levenshtein_distance(t1, t2)
                max_len = max(len(t1), len(t2))
                if max_len == 0:  # Избегаем деления на ноль
                    return False
                similarity = 1.0 - (distance / max_len)
                return similarity >= threshold
                
            return False
            
        except Exception as e:
            self._log(f"Ошибка при проверке соответствия текстов: {str(e)}")
            return False
    
    def _normalize_text(self, text):
        """
        Нормализует текст для сравнения, удаляя специальные символы и приводя к нижнему регистру.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Нормализованный текст
        """
        if not text:
            return ""
            
        # Преобразуем в строку и приводим к нижнему регистру
        text = str(text).lower()
        
        # Удаляем лишние пробелы
        text = " ".join(text.split())
        
        # Удаляем только некоторые символы, сохраняя важные для чисел и дат
        for char in ['«', '»', '"', "'", '(', ')', '[', ']', '{', '}']:
            text = text.replace(char, '')
        # НЕ удаляем точки, запятые и тире - они важны для чисел и дат
            
        return text
    
    def _has_common_numbers(self, str1, str2):
        """
        Проверяет, содержат ли две строки одинаковые числовые значения.
        
        Args:
            str1: Первая строка
            str2: Вторая строка
            
        Returns:
            bool: True, если строки содержат хотя бы одно общее число
        """
        try:
            # Получаем все числа из первой строки
            import re
            numbers1 = set(re.findall(r'\d+(?:[.,]\d+)?', str1))
            
            # Получаем все числа из второй строки
            numbers2 = set(re.findall(r'\d+(?:[.,]\d+)?', str2))
            
            # Проверяем, есть ли общие числа
            common_numbers = numbers1.intersection(numbers2)
            
            # Если есть хотя бы одно общее число и оно значимо (длиннее 1 цифры)
            significant_numbers = {num for num in common_numbers if len(num.replace('.', '').replace(',', '')) > 1}
            return bool(significant_numbers)
            
        except Exception as e:
            self._log(f"Ошибка при сравнении чисел в строках: {str(e)}")
            return False
    
    def _normalize_field_type(self, field_name):
        """
        Нормализует имя поля для использования в IOB2 тегах.
        
        Args:
            field_name: Исходное имя поля
            
        Returns:
            str: Нормализованное имя поля
        """
        try:
            if not field_name:
                return "ENTITY"
                
            # Приводим к верхнему регистру
            field_name = field_name.upper()
            
            # Заменяем пробелы на подчеркивания
            field_name = field_name.replace(' ', '_')
            
            # Отображаем известные поля на стандартные категории
            field_mapping = {
                # Компания/организация
                'КОМПАНИЯ': 'COMPANY',
                'ОРГАНИЗАЦИЯ': 'COMPANY',
                'ПОСТАВЩИК': 'COMPANY',
                'ПРОДАВЕЦ': 'COMPANY',
                'ПОКУПАТЕЛЬ': 'COMPANY',
                'CONTRACTOR': 'COMPANY',
                'SUPPLIER': 'COMPANY',
                'VENDOR': 'COMPANY',
                'CUSTOMER': 'COMPANY',
                'FIRM': 'COMPANY',
                'SELLER': 'COMPANY',
                'BUYER': 'COMPANY',
                'ORGANIZATION': 'COMPANY',
                
                # Дата
                'ДАТА': 'DATE',
                'ДЕНЬ': 'DATE',
                'ДАТА_ВЫСТАВЛЕНИЯ': 'DATE',
                'ДАТА_ДОКУМЕНТА': 'DATE',
                'INVOICE_DATE': 'DATE',
                'DOCUMENT_DATE': 'DATE',
                
                # Номер счета/инвойса
                'НОМЕР': 'INVOICE_ID',
                'НОМЕР_СЧЕТА': 'INVOICE_ID',
                'НОМЕР_ИНВОЙСА': 'INVOICE_ID',
                'НОМЕР_ДОКУМЕНТА': 'INVOICE_ID',
                'ДОКУМЕНТ_НОМЕР': 'INVOICE_ID',
                'СЧЕТ': 'INVOICE_ID',
                'ИНВОЙС': 'INVOICE_ID',
                'INVOICE': 'INVOICE_ID',
                'INVOICE_NUMBER': 'INVOICE_ID',
                'DOCUMENT_NUMBER': 'INVOICE_ID',
                
                # Сумма
                'СУММА': 'TOTAL',
                'ИТОГО': 'TOTAL',
                'ВСЕГО': 'TOTAL',
                'ИТОГОВАЯ_СУММА': 'TOTAL',
                'СУММА_К_ОПЛАТЕ': 'TOTAL',
                'TOTAL_AMOUNT': 'TOTAL',
                'AMOUNT': 'TOTAL',
                'AMOUNT_DUE': 'TOTAL',
                'TOTAL_SUM': 'TOTAL',
                'SUM': 'TOTAL',
                
                # Адрес
                'АДРЕС': 'ADDRESS',
                'ЮРИДИЧЕСКИЙ_АДРЕС': 'ADDRESS',
                'ФАКТИЧЕСКИЙ_АДРЕС': 'ADDRESS',
                'ADDRESS': 'ADDRESS',
                'LEGAL_ADDRESS': 'ADDRESS',
                'PHYSICAL_ADDRESS': 'ADDRESS',
                
                # ИНН/ОГРН
                'ИНН': 'TAX_ID',
                'ОГРН': 'TAX_ID',
                'TAX_ID': 'TAX_ID',
                'VAT': 'TAX_ID',
                'VAT_NUMBER': 'TAX_ID',
                
                # Табличные данные - наименование товара/услуги
                'НАИМЕНОВАНИЕ': 'NAME',
                'ТОВАР': 'NAME',
                'ПРОДУКТ': 'NAME',
                'УСЛУГА': 'NAME',
                'РАБОТА': 'NAME',
                'ПОЗИЦИЯ': 'NAME',
                'ITEM': 'NAME',
                'PRODUCT': 'NAME',
                'SERVICE': 'NAME',
                'DESCRIPTION': 'NAME',
                'TITLE': 'NAME',
                'NAME': 'NAME',
                
                # Табличные данные - количество
                'КОЛИЧЕСТВО': 'QUANTITY',
                'КОЛ-ВО': 'QUANTITY',
                'КОЛИЧЕСТВО_ЕДИНИЦ': 'QUANTITY',
                'ОБЪЕМ': 'QUANTITY',
                'QUANTITY': 'QUANTITY',
                'QTY': 'QUANTITY',
                'AMOUNT': 'QUANTITY',
                'COUNT': 'QUANTITY',
                'NUMBER': 'QUANTITY',
                
                # Табличные данные - единица измерения
                'ЕД': 'UNIT',
                'ЕДИНИЦА': 'UNIT',
                'ЕДИНИЦА_ИЗМЕРЕНИЯ': 'UNIT',
                'ЕД_ИЗМ': 'UNIT',
                'UNIT': 'UNIT',
                'MEASURE': 'UNIT',
                
                # Табличные данные - цена
                'ЦЕНА': 'PRICE',
                'СТОИМОСТЬ': 'PRICE',
                'ЦЕНА_ЗА_ЕДИНИЦУ': 'PRICE',
                'ЦЕНА_ЗА_ШТ': 'PRICE',
                'PRICE': 'PRICE',
                'PRICE_PER_UNIT': 'PRICE',
                'UNIT_PRICE': 'PRICE',
                'COST': 'PRICE',
                
                # Табличные данные - сумма по позиции
                'СУММА_ПОЗИЦИИ': 'ITEM_TOTAL',
                'СТОИМОСТЬ_ПОЗИЦИИ': 'ITEM_TOTAL',
                'СУММА_ТОВАРА': 'ITEM_TOTAL',
                'ITEM_TOTAL': 'ITEM_TOTAL',
                'LINE_TOTAL': 'ITEM_TOTAL',
                'SUBTOTAL': 'ITEM_TOTAL',
                'ROW_TOTAL': 'ITEM_TOTAL',
                
                # Табличные данные - НДС
                'НДС': 'VAT',
                'НАЛОГ': 'VAT',
                'СУММА_НДС': 'VAT',
                'СТАВКА_НДС': 'VAT_RATE',
                'VAT': 'VAT',
                'VAT_AMOUNT': 'VAT',
                'TAX': 'VAT',
                'VAT_RATE': 'VAT_RATE',
                'TAX_RATE': 'VAT_RATE',
                
                # Общие категории для таблицы
                'ТАБЛИЦА': 'TABLE',
                'ПОЗИЦИИ': 'TABLE',
                'ТОВАРЫ': 'TABLE',
                'УСЛУГИ': 'TABLE',
                'TABLE': 'TABLE',
                'ITEMS': 'TABLE',
                'PRODUCTS': 'TABLE',
                'SERVICES': 'TABLE',
                'POSITIONS': 'TABLE',
                'LINE_ITEMS': 'TABLE',
            }
            
            # Проверяем прямое соответствие
            if field_name in field_mapping:
                return field_mapping[field_name]
                
            # Проверяем частичное соответствие
            for key, value in field_mapping.items():
                if key in field_name or field_name in key:
                    return value
                    
            # Ограничиваем длину поля
            if len(field_name) > 20:
                field_name = field_name[:20]
                
            # Удаляем специальные символы
            import re
            field_name = re.sub(r'[^\w_]', '', field_name)
            
            # Возвращаем поле или значение по умолчанию
            return field_name if field_name else "ENTITY"
            
        except Exception as e:
            self._log(f"Ошибка при нормализации имени поля: {str(e)}")
            return "ENTITY"

    def _process_table_item(self, item, words, labels, item_idx):
        """
        Обрабатывает элемент таблицы и маркирует соответствующие слова в тексте.
        
        Args:
            item (dict): Словарь с данными элемента таблицы
            words (list): Список слов из OCR
            labels (list): Список меток для обновления
            item_idx (int): Индекс элемента таблицы
        """
        if not isinstance(item, dict):
            self._log(f"ПРЕДУПРЕЖДЕНИЕ: Элемент таблицы с индексом {item_idx} не является словарем")
            return
            
        # Логируем обработку
        self._log(f"Обработка элемента таблицы #{item_idx+1}: {item}")
        
        # Получаем основные поля элемента таблицы
        name_fields = ['name', 'description', 'title', 'product', 'service', 'item', 'наименование', 'товар', 'услуга', 'работа']
        quantity_fields = ['quantity', 'qty', 'count', 'number', 'количество', 'кол-во']
        unit_fields = ['unit', 'measure', 'единица', 'ед_изм', 'ед']
        price_fields = ['price', 'unit_price', 'cost', 'цена', 'стоимость', 'цена_за_единицу']
        total_fields = ['total', 'amount', 'sum', 'line_total', 'сумма', 'итого']
        vat_fields = ['vat', 'tax', 'vat_amount', 'нал', 'ндс']
        
        # Поля для поиска в элементе таблицы
        field_groups = {
            'NAME': name_fields,
            'QUANTITY': quantity_fields,
            'UNIT': unit_fields,
            'PRICE': price_fields,
            'ITEM_TOTAL': total_fields,
            'VAT': vat_fields
        }
        
        # Ищем заголовки таблицы, чтобы начать маркировку табличных строк рядом с ними
        header_indices = []
        header_row_found = False
        
        # Слова, которые обычно находятся в заголовках таблиц
        header_keywords = ['наименование', 'товар', 'услуга', 'работа', 'кол-во', 'количество', 
                           'цена', 'стоимость', 'сумма', 'итого', 'ед', 'ндс']
        
        # Ищем возможные заголовки таблицы
        for i, word in enumerate(words):
            word_lower = word.lower().strip()
            if any(keyword in word_lower for keyword in header_keywords):
                header_indices.append(i)
                
                # Если нашли несколько заголовков подряд, это скорее всего строка заголовка
                if len(header_indices) >= 2 and all(abs(header_indices[j] - header_indices[j-1]) <= 3 
                                                 for j in range(1, len(header_indices))):
                    header_row_found = True
                    break
        
        # Обрабатываем каждую группу полей
        for field_type, field_names in field_groups.items():
            value = None
            
            # Ищем значение поля в элементе таблицы
            for field_name in field_names:
                if field_name in item and item[field_name]:
                    value = item[field_name]
                    break
            
            if value is not None:
                # Преобразуем значение в строку
                value_str = str(value).strip().lower()
                
                # Пропускаем слишком короткие значения
                if len(value_str) < 2:
                    continue
                
                # Создаем метку для поля таблицы
                table_field_type = f"TABLE_ITEM_{field_type}"
                
                # Маркируем соответствующие слова в тексте
                marked_indices = self._mark_table_field(words, value_str, table_field_type, labels, item_idx, header_indices)
                
                # Если это поле наименования и мы успешно нашли его в тексте
                if field_type == 'NAME' and marked_indices:
                    # Маркируем начало новой строки таблицы
                    first_idx = min(marked_indices)
                    labels[first_idx] = f"B-TABLE_ROW"
                    
                    # Логируем найденное соответствие
                    self._log(f"Найдено соответствие для поля '{field_type}': '{value_str}' (индексы: {marked_indices})")

    def _mark_table_field(self, words, field_value, field_type, labels, item_idx, header_indices):
        """
        Маркирует слова, соответствующие полю таблицы.
        
        Args:
            words (list): Список слов из OCR
            field_value (str): Значение поля для поиска
            field_type (str): Тип поля для меток IOB2
            labels (list): Список меток для обновления
            item_idx (int): Индекс элемента таблицы
            header_indices (list): Индексы слов, являющихся заголовками таблицы
            
        Returns:
            list: Индексы размеченных слов
        """
        # Нормализуем значение поля
        field_value = field_value.lower().strip()
        
        # Индексы найденных и размеченных слов
        marked_indices = []
        
        # Приоритет для поиска: сначала после заголовков таблицы, потом везде
        search_ranges = []
        
        # Если есть заголовки таблицы, начинаем поиск после них
        if header_indices:
            max_header_idx = max(header_indices)
            start_idx = max_header_idx + 1
            # Предполагаем, что данные таблицы находятся в пределах 50 слов после заголовка
            end_idx = min(start_idx + 50, len(words))
            search_ranges.append((start_idx, end_idx))
        
        # Добавляем полный диапазон для поиска
        search_ranges.append((0, len(words)))
        
        # Поиск по заданным диапазонам
        for start_idx, end_idx in search_ranges:
            # Прямое совпадение
            for i in range(start_idx, end_idx):
                word = words[i].lower().strip()
                
                # Пропускаем уже размеченные слова
                if labels[i] != "O":
                    continue
                    
                # Проверяем на точное совпадение
                if word == field_value:
                    labels[i] = f"B-{field_type}"
                    marked_indices.append(i)
                    return marked_indices
                    
            # Поиск по группам слов
            max_window_size = min(10, len(words) - start_idx)  # Ограничиваем размер окна
            
            for window_size in range(min(len(field_value.split()), max_window_size), 0, -1):
                for i in range(start_idx, end_idx - window_size + 1):
                    # Пропускаем, если первое слово уже размечено
                    if labels[i] != "O":
                        continue
                        
                    # Создаем окно из слов
                    window_words = [words[j].lower().strip() for j in range(i, i + window_size)]
                    window_text = " ".join(window_words)
                    
                    # Проверяем на соответствие
                    if self._is_matching_text(window_text, field_value, threshold=0.7):
                        # Маркируем первое слово как B-{field_type}
                        labels[i] = f"B-{field_type}"
                        marked_indices.append(i)
                        
                        # Маркируем остальные слова как I-{field_type}
                        for j in range(i + 1, i + window_size):
                            labels[j] = f"I-{field_type}"
                            marked_indices.append(j)
                            
                        return marked_indices
            
            # Поиск по наличию общих чисел или частичному совпадению
            for i in range(start_idx, end_idx):
                # Пропускаем уже размеченные слова
                if labels[i] != "O":
                    continue
                    
                word = words[i].lower().strip()
                
                # Пропускаем слишком короткие слова
                if len(word) < 2:
                    continue
                    
                # Проверяем на частичное совпадение или наличие общих чисел
                if (word in field_value or 
                    field_value in word or 
                    self._has_common_numbers(word, field_value)):
                    
                    # Маркируем текущее слово
                    labels[i] = f"B-{field_type}"
                    marked_indices.append(i)
                    
                    # Ищем продолжение в соседних словах
                    j = i + 1
                    while j < end_idx and j < i + 5:  # Ограничиваем поиск 5 следующими словами
                        if labels[j] != "O":
                            j += 1
                            continue
                            
                        next_word = words[j].lower().strip()
                        
                        if (next_word in field_value or 
                            field_value in next_word or 
                            self._has_common_numbers(next_word, field_value)):
                            
                            labels[j] = f"I-{field_type}"
                            marked_indices.append(j)
                            j += 1
                        else:
                            break
                    
                    # Если нашли хотя бы одно слово, возвращаем результат
                    if marked_indices:
                        return marked_indices
        
        # Если не нашли совпадений
        return marked_indices

    # ... (остальной код, если есть)

    # Новые методы для современной подготовки данных
    def set_callbacks(self, log_callback=None, progress_callback=None):
        """Устанавливает функции обратного вызова для GUI"""
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def stop(self):
        """Остановка подготовки данных"""
        self.stop_requested = True
        self._log("⏹️ Получен запрос на остановку подготовки данных")
        
        # Устанавливаем тайм-аут для принудительной остановки
        import threading
        def force_stop():
            import time
            time.sleep(5)  # Ждем 5 секунд
            if self.stop_requested:
                self._log("🔴 Принудительная остановка через 5 секунд")
                import os
                import sys
                # В крайнем случае завершаем процесс
                # os._exit(0)  # Закомментировано, только для экстренных случаев
        
        # Запускаем таймер принудительной остановки
        timer_thread = threading.Thread(target=force_stop, daemon=True)
        timer_thread.start()
        if self.log_callback:
            self.log_callback("⏹️ Получен сигнал остановки подготовки данных")
            
    def _update_progress(self, progress: int):
        """Обновление прогресса"""
        if self.progress_callback:
            self.progress_callback(progress)
            
    def prepare_dataset_for_donut_modern(self,
                                       source_folder: str,
                                       output_path: str,
                                       task_type: str = "document_parsing",
                                       annotation_method: str = "gemini",
                                       max_files: Optional[int] = None) -> Optional[str]:
        """
        Современная подготовка датасета для обучения Donut
        
        Args:
            source_folder: Папка с исходными документами
            output_path: Путь для сохранения датасета
            task_type: Тип задачи (document_parsing, document_vqa)
            annotation_method: Метод аннотации
            max_files: Максимальное количество файлов
            
        Returns:
            str: Путь к подготовленному датасету или None при ошибке
        """
        import traceback
        import sys
        import psutil
        import os
        
        try:
            # Расширенное логирование для отладки
            self._log("=" * 80)
            self._log("🚀 НАЧАЛО ПОДГОТОВКИ ДАТАСЕТА - РАСШИРЕННОЕ ЛОГИРОВАНИЕ")
            self._log("=" * 80)
            self._log(f"🔍 Версия Python: {sys.version}")
            self._log(f"🔍 Рабочая директория: {os.getcwd()}")
            self._log(f"🔍 PID процесса: {os.getpid()}")
            
            # Проверяем память
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                self._log(f"🔍 Использование памяти: {memory_info.rss / 1024 / 1024:.1f} MB")
            except Exception as e:
                self._log(f"⚠️ Не удалось получить информацию о памяти: {e}")
            
            self._log("🍩 Начинаем подготовку датасета для Donut...")
            self._log(f"📁 Исходная папка: {source_folder}")
            self._log(f"🎯 Тип задачи: {task_type}")
            self._log(f"🔧 Метод аннотации: {annotation_method}")
            self._log(f"📊 Макс. файлов: {max_files if max_files else 'без ограничений'}")
            self._log(f"💾 Выходная папка: {output_path}")
            
            # Проверяем есть ли флаг остановки
            if hasattr(self, 'stop_requested') and self.stop_requested:
                self._log("⏹️ Обнаружен флаг остановки в начале метода")
                return None
            
            # Проверяем входные параметры
            if not source_folder or not os.path.exists(source_folder):
                raise ValueError(f"Исходная папка не существует: {source_folder}")
            
            if not output_path:
                raise ValueError("Не указан путь для сохранения датасета")
                
            self._log("✅ Входные параметры проверены")
            
            # Создаем абсолютный путь если путь относительный
            if not os.path.isabs(output_path):
                # Получаем корневую папку проекта
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_path = os.path.join(project_root, output_path)
                self._log(f"📍 Преобразован в абсолютный путь: {output_path}")
            
            # Создаем структуру датасета
            self._log("📂 Создание структуры датасета...")
            from pathlib import Path
            dataset_dir = Path(output_path)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"✅ Создана папка датасета: {dataset_dir}")
            
            images_dir = dataset_dir / "images"
            images_dir.mkdir(exist_ok=True)
            self._log(f"✅ Создана папка изображений: {images_dir}")
            
            # Сохраняем информацию об исходной папке в датасете
            from datetime import datetime
            dataset_info = {
                "created_at": datetime.now().isoformat(),
                "source_folder": os.path.abspath(source_folder),
                "model_type": "donut",
                "task_type": task_type,
                "annotation_method": annotation_method,
                "max_files": max_files,
                "total_files_processed": 0,
                "successful_files": 0,
                "failed_files": 0
            }
            
            # Сохраняем info файл
            info_path = dataset_dir / "dataset_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            self._log(f"📋 Информация о датасете сохранена: {info_path}")
            
            # Находим файлы
            self._log("🔍 Поиск файлов для обработки...")
            files = self._find_files_modern(source_folder, max_files)
            if not files:
                raise ValueError("Не найдено файлов для обработки")
                
            self._log(f"📄 Найдено файлов: {len(files)}")
            for i, file_path in enumerate(files[:5]):  # Показываем первые 5 файлов
                self._log(f"   {i+1}. {file_path.name}")
            if len(files) > 5:
                self._log(f"   ... и еще {len(files) - 5} файлов")
            
            # Проверяем доступность процессоров
            self._log("🔧 Проверка доступности процессоров...")
            if annotation_method == "gemini":
                if not self.gemini_processor:
                    self._log("⚠️ Gemini процессор недоступен, переключаемся на OCR")
                    annotation_method = "ocr"
                else:
                    self._log("✅ Gemini процессор доступен")
            
            if annotation_method == "ocr":
                if not self.ocr_processor:
                    self._log("⚠️ OCR процессор недоступен, используем базовые аннотации")
                    annotation_method = "manual"
                else:
                    self._log("✅ OCR процессор доступен")
            
            # Обрабатываем файлы
            self._log("🔄 Начинаем обработку файлов...")
            annotations = []
            total_files = len(files)
            processed_files = 0
            failed_files = 0
            
            for i, file_path in enumerate(files):
                try:
                    # Детальное логирование каждого шага
                    self._log("=" * 60)
                    self._log(f"🔄 ОБРАБОТКА ФАЙЛА {i+1}/{total_files}")
                    self._log("=" * 60)
                    self._log(f"📄 Файл: {file_path.name}")
                    self._log(f"📍 Полный путь: {file_path}")
                    self._log(f"📊 Размер файла: {file_path.stat().st_size / 1024:.1f} KB")
                    
                    # Проверяем память и состояние
                    try:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self._log(f"🔍 Память перед обработкой: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                        
                    if hasattr(self, 'stop_requested') and self.stop_requested:
                        self._log("⏹️ Получен сигнал остановки")
                        return None
                    
                    try:
                        # Конвертируем в изображения
                        self._log(f"   📷 Начинаем конвертацию в изображения...")
                        self._log(f"   📷 Тип файла: {file_path.suffix}")
                        images = self._convert_to_images_modern(file_path)
                        self._log(f"   ✅ Конвертация завершена. Получено изображений: {len(images)}")
                        
                        if not images:
                            self._log(f"   ⚠️ Не удалось получить изображения из файла")
                            failed_files += 1
                            continue
                        
                        for j, image in enumerate(images):
                            self._log(f"   🖼️ ОБРАБОТКА ИЗОБРАЖЕНИЯ {j+1}/{len(images)}")
                            
                            if hasattr(self, 'stop_requested') and self.stop_requested:
                                self._log("⏹️ Остановка во время обработки изображения")
                                return None
                                
                            try:
                                # Проверяем размер изображения
                                width, height = image.size
                                self._log(f"   📐 Размер изображения: {width}x{height}")
                                self._log(f"   📐 Режим изображения: {image.mode}")
                                
                                # Сохраняем изображение
                                image_name = f"{file_path.stem}_page_{j+1}.png"
                                image_path = images_dir / image_name
                                self._log(f"   💾 Сохранение: {image_name}")
                                self._log(f"   💾 Путь сохранения: {image_path}")
                                
                                image.save(image_path)
                                self._log(f"   ✅ Изображение сохранено успешно")
                                
                                # Проверяем что файл действительно создан
                                if image_path.exists():
                                    file_size = image_path.stat().st_size
                                    self._log(f"   ✅ Файл создан, размер: {file_size / 1024:.1f} KB")
                                else:
                                    self._log(f"   ❌ ОШИБКА: Файл не был создан!")
                                
                                # Создаем аннотацию для Donut
                                self._log(f"   🏷️ Начинаем создание аннотации методом: {annotation_method}")
                                annotation = self._create_donut_annotation_modern(
                                    image,
                                    image_name,
                                    task_type,
                                    annotation_method
                                )
                                
                                if annotation:
                                    annotations.append(annotation)
                                    self._log(f"   ✅ Аннотация создана и добавлена (всего: {len(annotations)})")
                                else:
                                    self._log(f"   ⚠️ Не удалось создать аннотацию")
                                    
                            except Exception as img_error:
                                self._log(f"   ❌ КРИТИЧЕСКАЯ ОШИБКА при обработке изображения {j+1}: {str(img_error)}")
                                self._log(f"   📋 Трассировка изображения: {traceback.format_exc()}")
                                raise img_error
                        
                        processed_files += 1
                        self._log(f"✅ Файл {file_path.name} обработан ПОЛНОСТЬЮ успешно")
                        
                    except Exception as conv_error:
                        self._log(f"   ❌ ОШИБКА конвертации файла {file_path.name}: {str(conv_error)}")
                        self._log(f"   📋 Трассировка конвертации: {traceback.format_exc()}")
                        failed_files += 1
                        continue
                        
                except Exception as file_error:
                    failed_files += 1
                    error_msg = f"❌ КРИТИЧЕСКАЯ ОШИБКА обработки файла {file_path.name}: {str(file_error)}"
                    self._log(error_msg)
                    self._log(f"📋 Полная трассировка ошибки: {traceback.format_exc()}")
                    
                    # Проверяем состояние системы после ошибки
                    try:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self._log(f"🔍 Память после ошибки: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                    continue
                    
                # Обновляем прогресс
                progress = int((i + 1) / total_files * 80)
                self._update_progress(progress)
                
            self._log(f"📊 Статистика обработки:")
            self._log(f"   ✅ Успешно обработано файлов: {processed_files}")
            self._log(f"   ❌ Файлов с ошибками: {failed_files}")
            self._log(f"   📝 Создано аннотаций: {len(annotations)}")
                
            if not annotations:
                raise ValueError("Не удалось создать ни одной аннотации")
                
            # Сохраняем аннотации
            self._log("💾 Сохранение аннотаций...")
            annotations_file = dataset_dir / "annotations.json"
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            self._log(f"✅ Аннотации сохранены в: {annotations_file}")
                
            self._update_progress(90)
            
            # Создаем HuggingFace Dataset для Donut
            self._log("🤗 Создание HuggingFace Dataset...")
            dataset = self._create_donut_dataset_modern(dataset_dir, task_type)
            self._log("✅ HuggingFace Dataset создан")
            
            # Сохраняем датасет
            self._log("💾 Сохранение датасета...")
            dataset_save_path = str(dataset_dir / "dataset_dict")
            dataset.save_to_disk(dataset_save_path)
            self._log(f"✅ Датасет сохранен в: {dataset_save_path}")
            
            self._update_progress(100)
            
            # Обновляем информацию о датасете
            dataset_info.update({
                "total_files_processed": total_files,
                "successful_files": processed_files,
                "failed_files": failed_files,
                "total_annotations": len(annotations),
                "finished_at": datetime.now().isoformat()
            })
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            self._log(f"📋 Информация о датасете обновлена: {info_path}")
            
            self._log(f"🎉 Датасет для Donut подготовлен успешно!")
            self._log(f"📊 Итоговая статистика:")
            self._log(f"   📄 Обработано файлов: {processed_files}/{total_files}")
            self._log(f"   📝 Создано аннотаций: {len(annotations)}")
            self._log(f"   💾 Сохранен в: {output_path}")
            
            return str(dataset_dir)
            
        except Exception as e:
            error_msg = f"❌ Критическая ошибка подготовки датасета Donut: {str(e)}"
            self._log(error_msg)
            import traceback
            full_traceback = traceback.format_exc()
            self._log(f"🔍 Полная трассировка ошибки:")
            for line in full_traceback.split('\n'):
                if line.strip():
                    self._log(f"   {line}")
            
            # Логируем состояние системы
            self._log("🔧 Диагностическая информация:")
            self._log(f"   Python версия: {sys.version}")
            self._log(f"   Рабочая директория: {os.getcwd()}")
            self._log(f"   Доступна ли папка источник: {os.path.exists(source_folder) if source_folder else False}")
            self._log(f"   Gemini процессор: {'Доступен' if self.gemini_processor else 'Недоступен'}")
            self._log(f"   OCR процессор: {'Доступен' if self.ocr_processor else 'Недоступен'}")
            
            return None
    
    def prepare_donut_dataset(self, source_folder: str, output_folder: str, task_type: str = "document_parsing", 
                            annotation_method: str = "gemini", max_files: Optional[int] = None) -> Optional[str]:
        """
        Алиас для prepare_dataset_for_donut_modern для обратной совместимости
        
        Args:
            source_folder: Папка с исходными документами
            output_folder: Папка для сохранения датасета
            task_type: Тип задачи (document_parsing, document_vqa)
            annotation_method: Метод аннотации (gemini, ocr, manual)
            max_files: Максимальное количество файлов
            
        Returns:
            str: Путь к подготовленному датасету или None при ошибке
        """
        self._log("🔄 Вызван prepare_donut_dataset (алиас для prepare_dataset_for_donut_modern)")
        return self.prepare_dataset_for_donut_modern(
            source_folder=source_folder,
            output_path=output_folder,
            task_type=task_type,
            annotation_method=annotation_method,
            max_files=max_files
        )
            
    def _find_files_modern(self, source_folder: str, max_files: Optional[int] = None) -> List:
        """Находит все поддерживаемые файлы в папке"""
        from pathlib import Path
        source_path = Path(source_folder)
        files = []
        
        # Поддерживаемые форматы
        supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        supported_document_formats = {'.pdf'}
        all_formats = supported_image_formats | supported_document_formats
        
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in all_formats:
                files.append(file_path)
                
                if max_files and len(files) >= max_files:
                    break
                    
        return sorted(files)
        
    def _convert_to_images_modern(self, file_path) -> List[Image.Image]:
        """
        Интеллектуальная конвертация файлов:
        - Для PDF с текстовым слоем: извлекает текст напрямую 
        - Для PDF без текста: конвертирует в изображения для OCR
        - Для изображений: загружает как есть
        """
        import traceback
        import os
        
        images = []
        
        try:
            self._log(f"         [CONVERT] НАЧАЛО конвертации файла: {file_path.name}")
            self._log(f"         [CONVERT] Полный путь: {file_path}")
            self._log(f"         [CONVERT] Размер файла: {file_path.stat().st_size / 1024:.1f} KB")
            
            # Проверяем память
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self._log(f"         [CONVERT] Память до конвертации: {memory_info.rss / 1024 / 1024:.1f} MB")
            except:
                pass
            
            supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            if file_path.suffix.lower() in supported_image_formats:
                self._log(f"         [CONVERT] Файл уже является изображением: {file_path.suffix}")
                try:
                    image = Image.open(file_path).convert('RGB')
                    images.append(image)
                    self._log(f"         [CONVERT] Изображение загружено успешно: {image.size}")
                except Exception as img_error:
                    self._log(f"         [ERROR] Ошибка загрузки изображения: {str(img_error)}")
                    self._log(f"         [ERROR] Трассировка: {traceback.format_exc()}")
                    raise img_error
                
            elif file_path.suffix.lower() == '.pdf':
                self._log(f"         [CONVERT] 🧠 Интеллектуальная обработка PDF файла...")
                
                # Сначала анализируем PDF на наличие текстового слоя
                try:
                    analysis = self.pdf_analyzer.analyze_pdf(str(file_path))
                    self._log(f"         [ANALYSIS] {analysis['recommendation']}")
                    self._log(f"         [ANALYSIS] Метод обработки: {analysis['processing_method']}")
                    
                    if analysis['processing_method'] == 'text_extraction':
                        # PDF содержит качественный текстовый слой
                        self._log(f"         [TEXT_EXTRACT] ✅ Извлекаем текст напрямую (качество: {analysis['text_quality']:.2f})")
                        
                        # Получаем стратегию обработки
                        strategy = self.pdf_analyzer.get_processing_strategy(str(file_path))
                        
                        if strategy['text_blocks']:
                            self._log(f"         [TEXT_EXTRACT] Найдено {len(strategy['text_blocks'])} текстовых блоков")
                            
                            # Сохраняем информацию о текстовых блоках для дальнейшей обработки
                            # Это будет использоваться вместо OCR данных
                            self._pdf_text_blocks = strategy['text_blocks']
                            self._pdf_has_text_layer = True
                            
                            # Создаем изображение для визуализации (только для совместимости)
                            # Но основные данные берем из текстового слоя
                            temp_images = self._convert_pdf_to_image_for_layout(file_path)
                            if temp_images:
                                images.extend(temp_images)
                                self._log(f"         [TEXT_EXTRACT] ✅ PDF с текстовым слоем обработан успешно")
                            else:
                                # Fallback к обычной конвертации если не удалось создать изображение
                                self._log(f"         [TEXT_EXTRACT] ⚠️ Fallback к конвертации изображения")
                                self._pdf_has_text_layer = False
                                images.extend(self._convert_pdf_to_images_ocr(file_path))
                        else:
                            self._log(f"         [TEXT_EXTRACT] ⚠️ Текстовые блоки не найдены, используем OCR")
                            self._pdf_has_text_layer = False
                            images.extend(self._convert_pdf_to_images_ocr(file_path))
                    else:
                        # PDF требует OCR
                        self._log(f"         [OCR] 🔍 PDF требует OCR обработки")
                        self._pdf_has_text_layer = False
                        images.extend(self._convert_pdf_to_images_ocr(file_path))
                        
                except Exception as analysis_error:
                    self._log(f"         [ERROR] Ошибка анализа PDF: {analysis_error}")
                    self._log(f"         [FALLBACK] Используем стандартную конвертацию")
                    self._pdf_has_text_layer = False
                    images.extend(self._convert_pdf_to_images_ocr(file_path))
            else:
                self._log(f"         [WARNING] Неподдерживаемый формат файла: {file_path.suffix}")
                
            self._log(f"         [CONVERT] Конвертация завершена успешно, изображений: {len(images)}")
            return images
            
        except Exception as general_error:
            self._log(f"         [ERROR] ОБЩАЯ ОШИБКА конвертации: {str(general_error)}")
            self._log(f"         [ERROR] Тип ошибки: {type(general_error).__name__}")
            self._log(f"         [ERROR] Трассировка:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self._log(f"         [ERROR]   {line}")
            
            # Возвращаем пустой список при ошибке, чтобы не завершать весь процесс
            return []
    
    def _convert_pdf_to_image_for_layout(self, file_path) -> List[Image.Image]:
        """Конвертирует PDF в изображение для layout (когда есть текстовый слой)"""
        try:
            from pdf2image import convert_from_path
            
            poppler_path = self.app_config.POPPLER_PATH if hasattr(self.app_config, 'POPPLER_PATH') else None
            
            if poppler_path and os.path.exists(poppler_path):
                pdf_images = convert_from_path(
                    str(file_path),
                    first_page=1,
                    last_page=1,  # Только первая страница для layout
                    dpi=150,  # Меньшее разрешение, так как текст уже есть
                    fmt='RGB',
                    poppler_path=poppler_path
                )
            else:
                pdf_images = convert_from_path(
                    str(file_path),
                    first_page=1,
                    last_page=1,
                    dpi=150,
                    fmt='RGB'
                )
            
            return pdf_images
            
        except Exception as e:
            self._log(f"         [ERROR] Ошибка конвертации PDF для layout: {e}")
            return []
    
    def _convert_pdf_to_images_ocr(self, file_path) -> List[Image.Image]:
        """Конвертирует PDF в изображения для OCR (когда нет текстового слоя)"""
        images = []
        try:
            # Получаем путь к Poppler
            self._log(f"         [CONVERT] Проверка пути к Poppler...")
                    poppler_path = self.app_config.POPPLER_PATH if hasattr(self.app_config, 'POPPLER_PATH') else None
                    self._log(f"         [CONVERT] Путь к Poppler: {poppler_path}")
                    
                    if poppler_path and os.path.exists(poppler_path):
                        self._log(f"         [CONVERT] Poppler найден, проверка исполняемых файлов...")
                        
                        # Проверяем наличие основных исполняемых файлов Poppler
                        required_files = ['pdftoppm.exe', 'pdfinfo.exe']
                        for req_file in required_files:
                            full_path = os.path.join(poppler_path, req_file)
                            if os.path.exists(full_path):
                                self._log(f"         [CONVERT] Найден: {req_file}")
                            else:
                                self._log(f"         [WARNING] НЕ найден: {req_file} в {full_path}")
                    else:
                        self._log(f"         [WARNING] Путь к Poppler не найден или недоступен: {poppler_path}")
                    
                    # Используем pdf2image для конвертации
                    self._log(f"         [CONVERT] Импорт pdf2image...")
                    from pdf2image import convert_from_path
                    self._log(f"         [CONVERT] pdf2image импортирован успешно")
                    
                    self._log(f"         [CONVERT] Вызов convert_from_path...")
                    self._log(f"         [CONVERT] Параметры: file={file_path}, dpi=200, fmt=RGB")
                    
                    if poppler_path and os.path.exists(poppler_path):
                        self._log(f"         [CONVERT] Конвертация с указанием Poppler...")
                        pdf_images = convert_from_path(
                            str(file_path),
                            dpi=200,  # Высокое качество для лучшего OCR
                            fmt='RGB',
                            poppler_path=poppler_path
                        )
                    else:
                        self._log(f"         [CONVERT] Конвертация без указания Poppler (используем PATH)...")
                        pdf_images = convert_from_path(
                            str(file_path),
                            dpi=200,  # Высокое качество для лучшего OCR
                            fmt='RGB'
                        )
                    
                    self._log(f"         [CONVERT] convert_from_path завершен УСПЕШНО!")
                    self._log(f"         [CONVERT] Получено изображений: {len(pdf_images)}")
                    
                    # Проверяем каждое изображение
                    for i, img in enumerate(pdf_images):
                        self._log(f"         [CONVERT] Изображение {i+1}: размер {img.size}, режим {img.mode}")
                        
                    images.extend(pdf_images)
                    self._log(f"         [OK] PDF конвертирован в {len(pdf_images)} изображений")
                    
                    # Проверяем память после конвертации
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self._log(f"         [CONVERT] Память после конвертации: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                    
                except Exception as pdf_error:
                    self._log(f"         [ERROR] КРИТИЧЕСКАЯ ОШИБКА конвертации PDF: {str(pdf_error)}")
                    self._log(f"         [ERROR] Тип ошибки: {type(pdf_error).__name__}")
                    self._log(f"         [ERROR] Полная трассировка:")
                    for line in traceback.format_exc().split('\n'):
                        if line.strip():
                            self._log(f"         [ERROR]   {line}")
                    
                    # Дополнительная диагностика
                    self._log(f"         [ERROR] Диагностика окружения:")
                    self._log(f"         [ERROR]   Файл существует: {file_path.exists()}")
                    self._log(f"         [ERROR]   Файл читаем: {os.access(file_path, os.R_OK)}")
                    self._log(f"         [ERROR]   Расширение файла: {file_path.suffix}")
                    
                    # Проверяем память при ошибке
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self._log(f"         [ERROR] Память при ошибке: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                        
                    raise pdf_error
            else:
                self._log(f"         [WARNING] Неподдерживаемый формат файла: {file_path.suffix}")
                
            self._log(f"         [CONVERT] Конвертация завершена успешно, изображений: {len(images)}")
            return images
            
        except Exception as general_error:
            self._log(f"         [ERROR] ОБЩАЯ ОШИБКА конвертации: {str(general_error)}")
            self._log(f"         [ERROR] Тип ошибки: {type(general_error).__name__}")
            self._log(f"         [ERROR] Трассировка:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self._log(f"         [ERROR]   {line}")
            
            # Возвращаем пустой список при ошибке, чтобы не завершать весь процесс
            return []
        
    def _create_donut_annotation_modern(self,
                                      image: Image.Image,
                                      image_name: str,
                                      task_type: str,
                                      annotation_method: str) -> Optional[Dict]:
        """Создает аннотацию для Donut"""
        try:
            self._log(f"      🏷️ Создание аннотации для {image_name}")
            self._log(f"      📋 Метод: {annotation_method}, Тип задачи: {task_type}")
            
            fields = {}
            
            if annotation_method == "gemini" and self.gemini_processor:
                self._log(f"      🤖 Извлечение полей с помощью Gemini...")
                # Используем Gemini для извлечения данных
                fields = self._extract_fields_with_gemini_modern(image)
                self._log(f"      ✅ Gemini извлек полей: {len(fields)}")
                
            elif annotation_method == "ocr":
                self._log(f"      👁️ Извлечение полей с помощью OCR...")
                # Используем OCR
                fields = self._extract_fields_with_ocr_modern(image)
                self._log(f"      ✅ OCR извлек полей: {len(fields)}")
                
            else:
                self._log(f"      📝 Создание базовой аннотации...")
                # Базовая аннотация
                fields = self._create_basic_annotation_modern()
                self._log(f"      ✅ Создана базовая аннотация с {len(fields)} полями")
            
            # Логируем извлеченные поля
            if fields:
                self._log(f"      📊 Извлеченные поля:")
                for key, value in fields.items():
                    if value:
                        self._log(f"         {key}: {value}")
                    else:
                        self._log(f"         {key}: (пусто)")
            else:
                self._log(f"      ⚠️ Поля не извлечены")
                
            # Форматируем для Donut в зависимости от типа задачи
            self._log(f"      🔄 Форматирование для Donut...")
            if task_type == "document_parsing":
                target_text = self._format_donut_parsing_target_modern(fields)
                self._log(f"      ✅ Создан parsing target: {target_text[:100]}...")
            elif task_type == "document_vqa":
                target_text = self._format_donut_vqa_target_modern(fields)
                self._log(f"      ✅ Создан VQA target: {target_text[:100]}...")
            else:
                target_text = json.dumps(fields, ensure_ascii=False)
                self._log(f"      ✅ Создан JSON target: {target_text[:100]}...")
                
            annotation = {
                'image': image_name,
                'text': target_text,
                'fields': fields,  # Сохраняем исходные поля для анализа
                'task_type': task_type
            }
            
            self._log(f"      🎉 Аннотация создана успешно")
            return annotation
            
        except Exception as e:
            error_msg = f"⚠️ Ошибка создания аннотации Donut: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"      Детали ошибки: {traceback.format_exc()}")
            return None
            
    def _extract_fields_with_gemini_modern(self, image: Image.Image) -> Dict:
        """Извлекает поля с помощью Gemini"""
        import traceback
        import tempfile
        import os
        import sys
        
        try:
            self._log(f"         🤖 НАЧИНАЕМ извлечение с Gemini...")
            self._log(f"         📐 Размер изображения: {image.size}")
            self._log(f"         📐 Режим изображения: {image.mode}")
            
            # Проверяем состояние памяти
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self._log(f"         🔍 Память до Gemini: {memory_info.rss / 1024 / 1024:.1f} MB")
            except:
                pass
            
            # Проверяем доступность Gemini процессора
            if not self.gemini_processor:
                self._log(f"         ❌ Gemini процессор недоступен")
                return self._create_basic_annotation_modern()
            else:
                self._log(f"         ✅ Gemini процессор доступен: {type(self.gemini_processor)}")
            
            # Сохраняем изображение во временный файл
            self._log(f"         💾 Создание временного файла...")
            tmp_file_path = None
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                self._log(f"         💾 Путь временного файла: {tmp_file_path}")
                
                # Сохраняем изображение
                self._log(f"         💾 Сохранение изображения в файл...")
                image.save(tmp_file_path)
                self._log(f"         ✅ Временный файл создан успешно")
                
                # Проверяем размер файла
                file_size = os.path.getsize(tmp_file_path)
                self._log(f"         📊 Размер временного файла: {file_size / 1024:.1f} KB")
            
            # Используем Gemini для извлечения данных
            self._log(f"         🔄 ВЫЗОВ Gemini API...")
            self._log(f"         🔄 Файл для обработки: {tmp_file_path}")
            
            try:
                result = self.gemini_processor.process(tmp_file_path)
                self._log(f"         ✅ Gemini API ответил УСПЕШНО")
                self._log(f"         📊 Тип результата: {type(result)}")
                
                if result:
                    self._log(f"         📋 Результат не пустой")
                else:
                    self._log(f"         ⚠️ Результат пустой!")
                    
            except Exception as gemini_error:
                self._log(f"         ❌ КРИТИЧЕСКАЯ ОШИБКА Gemini API: {str(gemini_error)}")
                self._log(f"         🔍 Тип ошибки Gemini: {type(gemini_error).__name__}")
                self._log(f"         📋 Трассировка Gemini API: {traceback.format_exc()}")
                
                # Проверяем память после ошибки
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self._log(f"         🔍 Память после ошибки Gemini: {memory_info.rss / 1024 / 1024:.1f} MB")
                except:
                    pass
                
                raise gemini_error
                
                # Безопасно удаляем временный файл
                try:
                    os.unlink(tmp_file.name)
                    self._log(f"         🗑️ Временный файл удален")
                except PermissionError:
                    # Файл может быть заблокирован, попробуем позже
                    self._log(f"         ⚠️ Временный файл заблокирован, будет удален позже")
                    import atexit
                    atexit.register(lambda: self._safe_delete_file(tmp_file.name))
                except Exception as e:
                    self._log(f"         ⚠️ Не удалось удалить временный файл: {e}")
                
            if result:
                self._log(f"         📊 Результат Gemini: {type(result)}")
                
                # Проверяем различные форматы ответа от Gemini
                if 'fields' in result and isinstance(result['fields'], list):
                    # Формат с полем 'fields' (список полей)
                    fields_data = {}
                    for field in result['fields']:
                        if isinstance(field, dict) and 'field_name' in field and 'field_value' in field:
                            field_name = field['field_name'].lower().replace(' ', '_').replace('№_счета', 'invoice_number').replace('дата_счета', 'date').replace('поставщик', 'company').replace('сумма_с_ндс', 'total_amount')
                            field_value = field['field_value']
                            # Пропускаем поля со значением "N/A"
                            if field_value and field_value != "N/A":
                                fields_data[field_name] = field_value
                    
                    self._log(f"         ✅ Извлечены поля из 'fields': {len(fields_data)} полей")
                    self._log(f"         📋 Извлеченные поля: {list(fields_data.keys())}")
                    return fields_data
                    
                elif isinstance(result, dict) and any(key in result for key in ['invoice_number', 'date', 'total_amount', 'company']):
                    # Прямой формат с ключами полей
                    self._log(f"         ✅ Извлечены поля напрямую: {len(result)} полей")
                    return result
                    
                else:
                    self._log(f"         ⚠️ Неизвестный формат результата")
                    self._log(f"         📋 Ключи результата: {list(result.keys()) if isinstance(result, dict) else 'не словарь'}")
                    # Пытаемся извлечь что-то полезное
                    if isinstance(result, dict):
                        return result
                    else:
                        return self._create_basic_annotation_modern()
            else:
                self._log(f"         ❌ Gemini вернул пустой результат")
                    
        except Exception as e:
            error_msg = f"⚠️ Ошибка извлечения полей с Gemini: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"         Детали ошибки: {traceback.format_exc()}")
            
        self._log(f"         🔄 Переход к базовой аннотации...")
        return self._create_basic_annotation_modern()
        
    def _safe_delete_file(self, file_path):
        """Безопасно удаляет файл с повторными попытками"""
        import time
        for attempt in range(3):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    self._log(f"         🗑️ Отложенное удаление файла успешно: {file_path}")
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)  # Ждем полсекунды перед повторной попыткой
                else:
                    self._log(f"         ❌ Не удалось удалить файл после 3 попыток: {file_path}")
        
    def _extract_fields_with_ocr_modern(self, image: Image.Image) -> Dict:
        """Извлекает поля с помощью OCR"""
        try:
            # Получаем текст с помощью Tesseract
            import pytesseract
            text = pytesseract.image_to_string(image, lang='rus+eng')
            
            # Простое извлечение полей на основе паттернов
            fields = {}
            
            # Ищем номер счета
            invoice_patterns = [
                r'(?:счет|invoice|№)\s*:?\s*([A-Za-z0-9\-/]+)',
                r'(?:номер|number)\s*:?\s*([A-Za-z0-9\-/]+)'
            ]
            
            for pattern in invoice_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields['invoice_number'] = match.group(1).strip()
                    break
                    
            # Ищем дату
            date_patterns = [
                r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
                r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields['date'] = match.group(1).strip()
                    break
                    
            # Ищем сумму
            amount_patterns = [
                r'(?:итого|total|сумма|amount)\s*:?\s*([0-9\s,\.]+)',
                r'([0-9\s,\.]+)\s*(?:руб|rub|₽|usd|eur)'
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields['total_amount'] = match.group(1).strip()
                    break
                    
            return fields
            
        except Exception as e:
            self._log(f"⚠️ Ошибка OCR извлечения: {str(e)}")
            return self._create_basic_annotation_modern()
            
    def _create_basic_annotation_modern(self) -> Dict:
        """Создает базовую аннотацию"""
        return {
            'invoice_number': '',
            'date': '',
            'company': '',
            'total_amount': '',
            'currency': 'RUB'
        }
        
    def _format_donut_parsing_target_modern(self, fields: Dict) -> str:
        """Форматирует поля для Donut в формате парсинга"""
        formatted_fields = []
        
        field_mapping = {
            'invoice_number': 'invoice_id',
            'date': 'date',
            'company': 'nm',
            'total_amount': 'total_price'
        }
        
        for key, value in fields.items():
            if value and key in field_mapping:
                donut_key = field_mapping[key]
                formatted_fields.append(f"<s_{donut_key}>{value}</s_{donut_key}>")
                
        return "".join(formatted_fields)
        
    def _format_donut_vqa_target_modern(self, fields: Dict) -> str:
        """Форматирует поля для Donut в формате VQA"""
        qa_pairs = []
        
        questions = {
            'invoice_number': 'Какой номер счета?',
            'date': 'Какая дата счета?',
            'company': 'Какая компания?',
            'total_amount': 'Какая общая сумма?'
        }
        
        for key, value in fields.items():
            if value and key in questions:
                question = questions[key]
                qa_pairs.append(f"<s_question>{question}</s_question><s_answer>{value}</s_answer>")
                
        return "".join(qa_pairs)
        
    def _create_donut_dataset_modern(self, dataset_dir, task_type: str) -> DatasetDict:
        """Создает HuggingFace Dataset для Donut"""
        try:
            self._log(f"🤗 Создание HuggingFace Dataset...")
            from pathlib import Path
            import json
            from PIL import Image
            
            # Проверяем необходимые импорты
            try:
                from datasets import Dataset, DatasetDict, Features, Value, Image as DatasetImage
                self._log(f"✅ Импорт datasets успешен")
            except ImportError as e:
                self._log(f"❌ Ошибка импорта datasets: {e}")
                self._log(f"   Убедитесь, что библиотека datasets установлена: pip install datasets")
                raise ImportError(f"Библиотека datasets не найдена: {e}")
            
            # Проверяем, что dataset_dir является объектом Path
            if not isinstance(dataset_dir, Path):
                dataset_dir = Path(dataset_dir)
                
            # Проверяем существование папки
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Папка датасета не существует: {dataset_dir}")
                
            self._log(f"📁 Рабочая папка: {dataset_dir}")
            
            # Загружаем аннотации
            annotations_file = dataset_dir / "annotations.json"
            self._log(f"📄 Загрузка аннотаций из: {annotations_file}")
            
            if not annotations_file.exists():
                raise FileNotFoundError(f"Файл аннотаций не найден: {annotations_file}")
            
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                    
                if not isinstance(annotations, list):
                    raise ValueError(f"Аннотации должны быть списком, получен: {type(annotations)}")
                    
                if not annotations:
                    raise ValueError("Список аннотаций пустой")
                    
                self._log(f"✅ Загружено аннотаций: {len(annotations)}")
                
                # Проверяем формат первой аннотации
                first_ann = annotations[0]
                required_keys = ['image', 'text']
                missing_keys = [key for key in required_keys if key not in first_ann]
                if missing_keys:
                    raise ValueError(f"В аннотациях отсутствуют обязательные ключи: {missing_keys}")
                    
                self._log(f"✅ Формат аннотаций корректен")
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Ошибка декодирования JSON аннотаций: {e}")
            except Exception as e:
                raise ValueError(f"Ошибка загрузки аннотаций: {e}")
            
            # Подготавливаем данные
            self._log(f"🔄 Подготовка данных для Dataset...")
            data = []
            images_dir = dataset_dir / "images"
            
            if not images_dir.exists():
                raise FileNotFoundError(f"Папка изображений не найдена: {images_dir}")
            
            processed_annotations = 0
            failed_annotations = 0
            
            for i, ann in enumerate(annotations):
                try:
                    # Проверяем структуру аннотации
                    if not isinstance(ann, dict):
                        self._log(f"   ⚠️ Аннотация {i+1} не является словарем: {type(ann)}")
                        failed_annotations += 1
                        continue
                        
                    if 'image' not in ann or 'text' not in ann:
                        self._log(f"   ⚠️ Аннотация {i+1} не содержит обязательных ключей: {list(ann.keys())}")
                        failed_annotations += 1
                        continue
                        
                    image_filename = ann['image']
                    if not image_filename:
                        self._log(f"   ⚠️ Аннотация {i+1} содержит пустое имя файла")
                        failed_annotations += 1
                        continue
                        
                    image_path = images_dir / image_filename
                    if image_path.exists():
                        self._log(f"   📷 Загрузка изображения {i+1}/{len(annotations)}: {image_filename}")
                        
                        # Безопасная загрузка изображения
                        try:
                            with Image.open(image_path) as img:
                                # Создаем копию изображения в памяти
                                image = img.convert('RGB')
                                # Проверяем размер изображения
                                if image.size[0] < 10 or image.size[1] < 10:
                                    self._log(f"   ⚠️ Изображение слишком маленькое: {image.size}")
                                    failed_annotations += 1
                                    continue
                                    
                                # Создаем копию для Dataset
                                image_copy = image.copy()
                                
                        except Exception as img_error:
                            self._log(f"   ❌ Ошибка загрузки изображения {image_filename}: {img_error}")
                            failed_annotations += 1
                            continue
                        
                        # Проверяем текст аннотации
                        text = ann['text']
                        if not isinstance(text, str):
                            self._log(f"   ⚠️ Текст аннотации не является строкой: {type(text)}")
                            text = str(text)
                            
                        data.append({
                            'image': image_copy,
                            'text': text
                        })
                        processed_annotations += 1
                        self._log(f"   ✅ Изображение добавлено в dataset")
                    else:
                        self._log(f"   ⚠️ Изображение не найдено: {image_path}")
                        failed_annotations += 1
                        
                except Exception as e:
                    self._log(f"   ❌ Ошибка обработки аннотации {i+1}: {e}")
                    import traceback
                    self._log(f"   Детали: {traceback.format_exc()}")
                    failed_annotations += 1
                    continue
            
            self._log(f"📊 Статистика подготовки данных:")
            self._log(f"   ✅ Успешно обработано: {processed_annotations}")
            self._log(f"   ❌ Ошибок: {failed_annotations}")
            
            if not data:
                raise ValueError("Не удалось подготовить данные для Dataset")
                
            # Разделяем на train/validation
            self._log(f"🔄 Разделение на train/validation...")
            
            # Обеспечиваем минимум 1 пример в каждом split
            if len(data) == 1:
                # Если только один пример, дублируем его для обоих splits
                train_data = data
                val_data = data
                self._log(f"📊 Разделение данных (малый датасет):")
                self._log(f"   🎓 Train: {len(train_data)} примеров (дублирован)")
                self._log(f"   ✅ Validation: {len(val_data)} примеров")
            else:
                train_size = max(1, int(0.8 * len(data)))  # Минимум 1 пример
                train_data = data[:train_size]
                val_data = data[train_size:] if len(data) > train_size else [data[-1]]  # Минимум 1 для validation
                
                self._log(f"📊 Разделение данных:")
                self._log(f"   🎓 Train: {len(train_data)} примеров")
                self._log(f"   ✅ Validation: {len(val_data)} примеров")
            
            # Создаем Dataset
            self._log(f"🔄 Создание Dataset объектов...")
            try:
                # Определяем схему данных для корректного создания Dataset
                features = Features({
                    'image': DatasetImage(),
                    'text': Value('string')
                })
                self._log(f"✅ Схема данных определена")
                
                # Создаем train dataset с проверками
                self._log(f"🔄 Создание train dataset из {len(train_data)} примеров...")
                if not train_data:
                    raise ValueError("Нет данных для train dataset")
                    
                train_dataset = Dataset.from_list(train_data, features=features)
                self._log(f"✅ Train dataset создан: {len(train_dataset)} примеров")
                
                # Создаем validation dataset с проверками
                self._log(f"🔄 Создание validation dataset из {len(val_data)} примеров...")
                if not val_data:
                    raise ValueError("Нет данных для validation dataset")
                    
                val_dataset = Dataset.from_list(val_data, features=features)
                self._log(f"✅ Validation dataset создан: {len(val_dataset)} примеров")
                
                # Создаем DatasetDict
                self._log(f"🔄 Создание DatasetDict...")
                dataset_dict = DatasetDict({
                    'train': train_dataset,
                    'validation': val_dataset
                })
                self._log(f"✅ DatasetDict создан успешно")
                self._log(f"📊 Итоговая статистика Dataset:")
                self._log(f"   🎓 Train: {len(dataset_dict['train'])} примеров")
                self._log(f"   ✅ Validation: {len(dataset_dict['validation'])} примеров")
                
                return dataset_dict
                
            except Exception as e:
                self._log(f"❌ Ошибка создания Dataset: {e}")
                import traceback
                self._log(f"🔍 Детали ошибки создания Dataset:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        self._log(f"   {line}")
                raise RuntimeError(f"Не удалось создать HuggingFace Dataset: {e}")
                
        except Exception as e:
            error_msg = f"❌ Критическая ошибка создания HuggingFace Dataset: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"🔍 Детали ошибки:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self._log(f"   {line}")
            raise

    def prepare_dataset_for_layoutlm_modern(self,
                                           source_folder: str,
                                           output_path: str,
                                           task_type: str = "token_classification",
                                           annotation_method: str = "gemini",
                                           max_files: Optional[int] = None) -> Optional[str]:
        """
        Современная подготовка датасета для обучения LayoutLM
        
        Args:
            source_folder: Папка с исходными документами
            output_path: Путь для сохранения датасета
            task_type: Тип задачи (token_classification)
            annotation_method: Метод аннотации (gemini, ocr, manual)
            max_files: Максимальное количество файлов
            
        Returns:
            str: Путь к подготовленному датасету или None при ошибке
        """
        import traceback
        import sys
        import psutil
        import os
        from pathlib import Path
        
        try:
            # Расширенное логирование для отладки
            self._log("=" * 80)
            self._log("🎯 НАЧАЛО ПОДГОТОВКИ ДАТАСЕТА ДЛЯ LAYOUTLM - РАСШИРЕННОЕ ЛОГИРОВАНИЕ")
            self._log("=" * 80)
            self._log(f"🔍 Версия Python: {sys.version}")
            self._log(f"🔍 Рабочая директория: {os.getcwd()}")
            self._log(f"🔍 PID процесса: {os.getpid()}")
            
            # Проверяем память
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                self._log(f"🔍 Использование памяти: {memory_info.rss / 1024 / 1024:.1f} MB")
            except Exception as e:
                self._log(f"⚠️ Не удалось получить информацию о памяти: {e}")
            
            self._log("🎯 Начинаем подготовку датасета для LayoutLM...")
            self._log(f"📁 Исходная папка: {source_folder}")
            self._log(f"🎯 Тип задачи: {task_type}")
            self._log(f"🔧 Метод аннотации: {annotation_method}")
            self._log(f"📊 Макс. файлов: {max_files if max_files else 'без ограничений'}")
            self._log(f"💾 Выходная папка: {output_path}")
            
            # Проверяем есть ли флаг остановки
            if hasattr(self, 'stop_requested') and self.stop_requested:
                self._log("⏹️ Обнаружен флаг остановки в начале метода")
                return None
            
            # Проверяем входные параметры
            if not source_folder or not os.path.exists(source_folder):
                raise ValueError(f"Исходная папка не существует: {source_folder}")
            
            if not output_path:
                raise ValueError("Не указан путь для сохранения датасета")
                
            self._log("✅ Входные параметры проверены")
            
            # Создаем абсолютный путь если путь относительный
            if not os.path.isabs(output_path):
                # Получаем корневую папку проекта
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_path = os.path.join(project_root, output_path)
                self._log(f"📍 Преобразован в абсолютный путь: {output_path}")
            
            # Создаем структуру датасета
            self._log("📂 Создание структуры датасета для LayoutLM...")
            dataset_dir = Path(output_path)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"✅ Создана папка датасета: {dataset_dir}")
            
            # Сохраняем информацию об исходной папке в датасете
            from datetime import datetime
            dataset_info = {
                "created_at": datetime.now().isoformat(),
                "source_folder": os.path.abspath(source_folder),
                "model_type": "layoutlm", 
                "task_type": task_type,
                "annotation_method": annotation_method,
                "max_files": max_files,
                "total_files_processed": 0,
                "successful_files": 0,
                "failed_files": 0
            }
            
            # Сохраняем info файл
            info_path = dataset_dir / "dataset_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            self._log(f"📋 Информация о датасете сохранена: {info_path}")
            
            # Находим файлы
            self._log("🔍 Поиск файлов для обработки...")
            files = self._find_files_modern(source_folder, max_files)
            if not files:
                raise ValueError("Не найдено файлов для обработки")
                
            self._log(f"📄 Найдено файлов: {len(files)}")
            for i, file_path in enumerate(files[:5]):  # Показываем первые 5 файлов
                self._log(f"   {i+1}. {file_path.name}")
            if len(files) > 5:
                self._log(f"   ... и еще {len(files) - 5} файлов")
            
            # Проверяем доступность процессоров
            self._log("🔧 Проверка доступности процессоров...")
            if annotation_method == "gemini":
                if not self.gemini_processor:
                    self._log("⚠️ Gemini процессор недоступен, переключаемся на OCR")
                    annotation_method = "ocr"
                else:
                    self._log("✅ Gemini процессор доступен")
            
            if annotation_method == "ocr":
                if not self.ocr_processor:
                    self._log("⚠️ OCR процессор недоступен, используем базовые аннотации")
                    annotation_method = "manual"
                else:
                    self._log("✅ OCR процессор доступен")
            
            # Обрабатываем файлы для LayoutLM
            self._log("🔄 Начинаем обработку файлов для LayoutLM...")
            processed_records = []
            total_files = len(files)
            processed_files = 0
            failed_files = 0
            
            for i, file_path in enumerate(files):
                try:
                    # Детальное логирование каждого шага
                    self._log("=" * 60)
                    self._log(f"🔄 ОБРАБОТКА ФАЙЛА {i+1}/{total_files} ДЛЯ LAYOUTLM")
                    self._log("=" * 60)
                    self._log(f"📄 Файл: {file_path.name}")
                    self._log(f"📍 Полный путь: {file_path}")
                    self._log(f"📊 Размер файла: {file_path.stat().st_size / 1024:.1f} KB")
                    
                    # Проверяем память и состояние
                    try:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self._log(f"🔍 Память перед обработкой: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                        
                    if hasattr(self, 'stop_requested') and self.stop_requested:
                        self._log("⏹️ Получен сигнал остановки")
                        return None
                    
                    try:
                        # Конвертируем в изображения
                        self._log(f"   📷 Начинаем конвертацию в изображения...")
                        self._log(f"   📷 Тип файла: {file_path.suffix}")
                        images = self._convert_to_images_modern(file_path)
                        self._log(f"   ✅ Конвертация завершена. Получено изображений: {len(images)}")
                        
                        if not images:
                            self._log(f"   ⚠️ Не удалось получить изображения из файла")
                            failed_files += 1
                            continue
                        
                        for j, image in enumerate(images):
                            self._log(f"   🖼️ ОБРАБОТКА ИЗОБРАЖЕНИЯ {j+1}/{len(images)} ДЛЯ LAYOUTLM")
                            
                            if hasattr(self, 'stop_requested') and self.stop_requested:
                                self._log("⏹️ Остановка во время обработки изображения")
                                return None
                                
                            try:
                                # Проверяем размер изображения
                                width, height = image.size
                                self._log(f"   📐 Размер изображения: {width}x{height}")
                                self._log(f"   📐 Режим изображения: {image.mode}")
                                
                                # Создаем запись для LayoutLM
                                image_name = f"{file_path.stem}_page_{j+1}.png"
                                self._log(f"   🎯 Обработка для LayoutLM: {image_name}")
                                
                                # Сохраняем изображение в датасет
                                image_path = dataset_dir / image_name
                                image.save(image_path)
                                self._log(f"   💾 Изображение сохранено: {image_path}")
                                
                                # Создаем LayoutLM аннотацию
                                layoutlm_record = self._create_layoutlm_annotation_modern(
                                    image,
                                    str(image_path),  # Передаем реальный путь к сохраненному изображению
                                    annotation_method
                                )
                                
                                if layoutlm_record:
                                    processed_records.append(layoutlm_record)
                                    self._log(f"   ✅ LayoutLM запись создана и добавлена (всего: {len(processed_records)})")
                                else:
                                    self._log(f"   ⚠️ Не удалось создать LayoutLM запись")
                                    
                            except Exception as img_error:
                                self._log(f"   ❌ КРИТИЧЕСКАЯ ОШИБКА при обработке изображения {j+1}: {str(img_error)}")
                                self._log(f"   📋 Трассировка изображения: {traceback.format_exc()}")
                                raise img_error
                        
                        processed_files += 1
                        self._log(f"✅ Файл {file_path.name} обработан ПОЛНОСТЬЮ для LayoutLM")
                        
                        # Обновляем прогресс
                        progress = int((i + 1) / total_files * 100)
                        self._update_progress(progress)
                        
                    except Exception as conv_error:
                        self._log(f"   ❌ ОШИБКА конвертации файла {file_path.name}: {str(conv_error)}")
                        self._log(f"   📋 Трассировка конвертации: {traceback.format_exc()}")
                        failed_files += 1
                        continue
                        
                except Exception as file_error:
                    self._log(f"❌ ОШИБКА при обработке файла {file_path}: {str(file_error)}")
                    self._log(f"📋 Трассировка файла: {traceback.format_exc()}")
                    failed_files += 1
                    continue
            
            # Проверяем результаты
            self._log("📊 ИТОГИ ОБРАБОТКИ ДЛЯ LAYOUTLM:")
            self._log(f"   📄 Всего файлов: {total_files}")
            self._log(f"   ✅ Обработано: {processed_files}")
            self._log(f"   ❌ Неудачно: {failed_files}")
            self._log(f"   📝 Создано записей: {len(processed_records)}")
            
            if not processed_records:
                raise ValueError("Не удалось создать ни одной записи для LayoutLM")
            
            # Создаем HuggingFace датасет для LayoutLM
            self._log("🤗 Создание HuggingFace датасета для LayoutLM...")
            layoutlm_dataset = self.create_full_dataset(processed_records)
            
            if not layoutlm_dataset:
                raise ValueError("Не удалось создать HuggingFace датасет")
            
            # Сохраняем датасет
            dataset_dict_path = dataset_dir / "dataset_dict"
            dataset_dict_path.mkdir(exist_ok=True)
            self._log(f"💾 Сохранение датасета в: {dataset_dict_path}")
            
            self.split_and_save_dataset(layoutlm_dataset, str(dataset_dict_path))
            
            # Обновляем информацию о датасете
            dataset_info.update({
                "total_files_processed": total_files,
                "successful_files": processed_files,
                "failed_files": failed_files,
                "total_records": len(processed_records),
                "finished_at": datetime.now().isoformat()
            })
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            self._log(f"📋 Информация о датасете обновлена: {info_path}")
            
            self._log("🎉 ПОДГОТОВКА ДАТАСЕТА ДЛЯ LAYOUTLM ЗАВЕРШЕНА УСПЕШНО!")
            self._log(f"💾 Путь к датасету: {dataset_dict_path}")
            self._log("=" * 80)
            
            return str(dataset_dict_path)
            
        except Exception as e:
            self._log(f"❌ КРИТИЧЕСКАЯ ОШИБКА при подготовке датасета для LayoutLM: {str(e)}")
            self._log(f"📋 Полная трассировка: {traceback.format_exc()}")
            return None

    def _create_layoutlm_annotation_modern(self,
                                         image: Image.Image,
                                         image_path: str,
                                         annotation_method: str) -> Optional[Dict]:
        """
        Создает аннотацию для LayoutLM в современном формате
        
        Args:
            image: Изображение документа
            image_name: Имя изображения
            annotation_method: Метод аннотации
            
        Returns:
            Dict: Словарь с данными для LayoutLM (words, bboxes, labels, image_path)
        """
        try:
            image_name = os.path.basename(image_path)
            self._log(f"     🎯 Создание LayoutLM аннотации для {image_name}")
            self._log(f"     🔧 Метод: {annotation_method}")
            
            # Сохраняем изображение временно для OCR
            temp_image_path = f"temp_{image_name}"
            image.save(temp_image_path)
            self._log(f"     💾 Временное изображение сохранено: {temp_image_path}")
            
            try:
                # Получаем OCR данные (для координат и слов)
                self._log(f"     🔍 Извлечение OCR данных...")
                if self.ocr_processor:
                    ocr_result = self.ocr_processor.process_image(temp_image_path)
                    # OCR возвращает кортеж (text, words_list), а не словарь
                    if not ocr_result or len(ocr_result) < 2 or not ocr_result[1]:
                        self._log(f"     ⚠️ OCR не вернул слова, используем базовую аннотацию")
                        words = ["DOCUMENT"]
                        bboxes = [[0, 0, image.width, image.height]]
                    else:
                        # Извлекаем данные из OCR результата
                        ocr_text, ocr_words = ocr_result
                        self._log(f"     📝 OCR текст: {len(ocr_text)} символов")
                        self._log(f"     📝 OCR слова: {len(ocr_words)} слов")
                        
                        # Формируем слова и bbox из OCR
                        words = []
                        bboxes = []
                        for word_data in ocr_words:
                            if word_data.get('text', '').strip():
                                words.append(word_data['text'])
                                # Преобразуем координаты x,y,width,height в x1,y1,x2,y2
                                x, y, w, h = word_data['x'], word_data['y'], word_data['width'], word_data['height']
                                bbox = [x, y, x + w, y + h]
                                bboxes.append(bbox)
                        
                        self._log(f"     ✅ OCR получил {len(words)} слов с координатами")
                        if len(words) > 0:
                            self._log(f"     📋 Первые 3 слова: {words[:3]}")
                else:
                    self._log(f"     ⚠️ OCR процессор недоступен, используем базовую аннотацию")
                    words = ["DOCUMENT"]
                    bboxes = [[0, 0, image.width, image.height]]
                
                # Получаем структурированные данные для создания меток
                structured_data = {}
                
                # Проверяем интеллектуальный режим
                if self.intelligent_mode:
                    self._log(f"     🧠 ИНТЕЛЛЕКТУАЛЬНЫЙ РЕЖИМ: Извлечение ВСЕХ полезных данных...")
                    self._init_intelligent_extractor()
                    
                    if self.intelligent_extractor:
                        try:
                            # Интеллектуальное извлечение всех данных
                            extracted_data = self.intelligent_extractor.extract_all_data(temp_image_path)
                            
                            if extracted_data and extracted_data.get('fields'):
                                self._log(f"     ✅ Интеллектуальный экстрактор извлек {len(extracted_data['fields'])} полей")
                                
                                # Конвертируем в формат structured_data
                                for field in extracted_data['fields']:
                                    if hasattr(field, 'name') and hasattr(field, 'value'):
                                        structured_data[field.name] = field.value
                                    elif isinstance(field, dict):
                                        structured_data[field.get('name', 'unknown')] = field.get('value', '')
                                
                                self._log(f"     📊 Интеллектуальный режим: {list(structured_data.keys())}")
                            else:
                                self._log(f"     ⚠️ Интеллектуальный экстрактор не вернул данные, переход к стандартному режиму")
                                
                        except Exception as e:
                            self._log(f"     ❌ Ошибка интеллектуального извлечения: {e}")
                            self._log(f"     🔄 Переход к стандартному режиму...")
                
                # Если интеллектуальный режим не сработал или отключен, используем стандартные методы
                if not structured_data:
                    if annotation_method == "gemini" and self.gemini_processor:
                        self._log(f"     🤖 Извлечение данных через стандартный Gemini...")
                        structured_data = self._extract_fields_with_gemini_modern(image)
                        self._log(f"     ✅ Стандартный Gemini вернул поля: {list(structured_data.keys())}")
                    elif annotation_method == "ocr" and self.ocr_processor:
                        self._log(f"     🔍 Извлечение данных через OCR...")
                        structured_data = self._extract_fields_with_ocr_modern(image)
                        self._log(f"     ✅ OCR вернул поля: {list(structured_data.keys())}")
                    else:
                        self._log(f"     📝 Использование базовых полей...")
                        structured_data = self._create_basic_annotation_modern()
                
                # Создаем метки для LayoutLM (IOB2 формат)
                self._log(f"     🏷️ Создание меток IOB2...")
                labels = self._create_layoutlm_labels(words, structured_data)
                self._log(f"     ✅ Создано меток: {len(labels)}")
                
                # Нормализуем координаты bbox
                self._log(f"     📐 Нормализация координат...")
                normalized_bboxes = []
                for bbox in bboxes:
                    if len(bbox) == 4:
                        normalized_bbox = self.normalize_bbox(bbox, image.width, image.height)
                        normalized_bboxes.append(normalized_bbox)
                    else:
                        self._log(f"     ⚠️ Неверный формат bbox: {bbox}")
                        normalized_bboxes.append([0, 0, 0, 0])
                
                # Проверяем согласованность данных
                min_len = min(len(words), len(normalized_bboxes), len(labels))
                if min_len == 0:
                    self._log(f"     ❌ Пустые данные - невозможно создать запись")
                    return None
                
                if len(words) != len(normalized_bboxes) or len(words) != len(labels):
                    self._log(f"     ⚠️ Несогласованность длин: words={len(words)}, bboxes={len(normalized_bboxes)}, labels={len(labels)}")
                    self._log(f"     🔧 Обрезаем до минимальной длины: {min_len}")
                    words = words[:min_len]
                    normalized_bboxes = normalized_bboxes[:min_len]
                    labels = labels[:min_len]
                
                # Создаем результат
                result = {
                    'image_path': image_path,       # Путь к сохраненному изображению
                    'words': words,                 # Список слов
                    'bboxes': normalized_bboxes,   # Нормализованные координаты
                    'labels': labels               # IOB2 метки
                }
                
                self._log(f"     ✅ LayoutLM аннотация создана успешно:")
                self._log(f"        📝 Слов: {len(result['words'])}")
                self._log(f"        📐 Bbox: {len(result['bboxes'])}")
                self._log(f"        🏷️ Меток: {len(result['labels'])}")
                self._log(f"        🖼️ Изображение: {result['image_path']}")
                
                return result
                
            finally:
                # Очищаем временный файл
                try:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                        self._log(f"     🗑️ Временный файл удален: {temp_image_path}")
                except:
                    pass
                    
        except Exception as e:
            self._log(f"     ❌ ОШИБКА создания LayoutLM аннотации: {str(e)}")
            import traceback
            self._log(f"     📋 Трассировка: {traceback.format_exc()}")
            return None

    def _create_layoutlm_labels(self, words: List[str], structured_data: Dict) -> List[str]:
        """
        Создает IOB2 метки для LayoutLM на основе структурированных данных
        
        Args:
            words: Список слов из OCR
            structured_data: Структурированные данные (поля счета)
            
        Returns:
            List[str]: Список IOB2 меток
        """
        try:
            self._log(f"       🏷️ Создание IOB2 меток для {len(words)} слов")
            
            # Инициализируем все метки как "O" (Outside)
            labels = ["O"] * len(words)
            
            # Определяем типы полей с расширенным маппингом
            field_types = {
                'company': 'COMPANY',
                'поставщик': 'COMPANY',
                'date': 'DATE',
                'дата_счета': 'DATE', 
                'invoice_number': 'INVOICE_NUMBER',
                '№_счета': 'INVOICE_NUMBER',
                'total_amount': 'AMOUNT',
                'сумма_с_ндс': 'AMOUNT',
                'amount': 'AMOUNT',
                'total': 'AMOUNT',
                'валюта': 'CURRENCY',
                'currency': 'CURRENCY',
                'товары': 'ITEMS',
                'категория': 'CATEGORY',
                'инн': 'TAX_ID',
                'кпп': 'TAX_CODE'
            }
            
            # Обрабатываем каждое поле из структурированных данных
            for field_name, field_value in structured_data.items():
                if not field_value or not isinstance(field_value, str):
                    continue
                    
                field_value = str(field_value).strip()
                if not field_value:
                    continue
                    
                # Определяем тип метки
                label_type = field_types.get(field_name.lower(), field_name.upper())
                
                self._log(f"       🔍 Обработка поля '{field_name}': '{field_value}' -> {label_type}")
                
                # Ищем соответствующие слова
                matched_indices = self._find_matching_word_indices(words, field_value)
                
                if matched_indices:
                    self._log(f"       ✅ Найдены соответствия на позициях: {matched_indices}")
                    # Применяем IOB2 разметку
                    for i, word_idx in enumerate(matched_indices):
                        if i == 0:
                            labels[word_idx] = f"B-{label_type}"  # Beginning
                        else:
                            labels[word_idx] = f"I-{label_type}"  # Inside
                else:
                    self._log(f"       ⚠️ Не найдены соответствия для поля '{field_name}'")
            
            # Подсчитываем статистику меток
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            self._log(f"       📊 Статистика меток: {label_counts}")
            
            return labels
            
        except Exception as e:
            self._log(f"       ❌ ОШИБКА создания IOB2 меток: {str(e)}")
            # Возвращаем базовые метки
            return ["O"] * len(words)

    def _find_matching_word_indices(self, words: List[str], field_value: str) -> List[int]:
        """
        Находит индексы слов, которые соответствуют значению поля
        
        Args:
            words: Список слов
            field_value: Значение поля для поиска
            
        Returns:
            List[int]: Список индексов соответствующих слов
        """
        try:
            field_value = field_value.strip().lower()
            field_words = field_value.split()
            
            if not field_words:
                return []
            
            matched_indices = []
            
            # Пробуем точное совпадение последовательности слов
            for start_idx in range(len(words) - len(field_words) + 1):
                match = True
                for i, field_word in enumerate(field_words):
                    word_idx = start_idx + i
                    if word_idx >= len(words):
                        match = False
                        break
                    
                    word = words[word_idx].lower().strip()
                    if not self._is_similar_word(word, field_word):
                        match = False
                        break
                
                if match:
                    matched_indices = list(range(start_idx, start_idx + len(field_words)))
                    break
            
            # Если точное совпадение не найдено, ищем частичные совпадения
            if not matched_indices:
                for i, word in enumerate(words):
                    word_clean = word.lower().strip()
                    for field_word in field_words:
                        if self._is_similar_word(word_clean, field_word):
                            if i not in matched_indices:
                                matched_indices.append(i)
            
            return sorted(matched_indices)
            
        except Exception as e:
            self._log(f"         ❌ ОШИБКА поиска соответствий: {str(e)}")
            return []

    def _is_similar_word(self, word1: str, word2: str, threshold: float = 0.7) -> bool:
        """
        Проверяет похожесть двух слов
        
        Args:
            word1: Первое слово
            word2: Второе слово
            threshold: Порог похожести
            
        Returns:
            bool: True если слова похожи
        """
        try:
            # Точное совпадение
            if word1 == word2:
                return True
            
            # Проверка вхождения
            if word1 in word2 or word2 in word1:
                return True
            
            # Для коротких слов - более строгие критерии
            if len(word1) <= 3 or len(word2) <= 3:
                return word1 == word2
            
            # Вычисляем схожесть
            similarity = calculate_text_similarity(word1, word2)
            return similarity >= threshold
            
        except Exception:
            return False

    def get_training_prompt(self, task_type: str = "layoutlm") -> str:
        """
        Генерирует промпт для обучения на основе настроек полей таблицы
        
        Args:
            task_type: Тип задачи ("layoutlm", "donut", "gemini")
            
        Returns:
            str: Сгенерированный промпт для обучения
        """
        if not self.field_manager:
            # Fallback промпт если FieldManager недоступен
            return self._get_fallback_training_prompt(task_type)
        
        try:
            enabled_fields = self.field_manager.get_enabled_fields()
            
            if not enabled_fields:
                self._log("⚠️ Нет активных полей в FieldManager, используется fallback промпт")
                return self._get_fallback_training_prompt(task_type)
            
            # Генерируем промпт на основе активных полей
            if task_type == "gemini":
                prompt = self._generate_gemini_training_prompt(enabled_fields)
            elif task_type == "layoutlm":
                prompt = self._generate_layoutlm_training_prompt(enabled_fields)
            elif task_type == "donut":
                prompt = self._generate_donut_training_prompt(enabled_fields)
            else:
                prompt = self._generate_generic_training_prompt(enabled_fields)
            
            self._log(f"🎯 Сгенерирован промпт для {task_type} на основе {len(enabled_fields)} активных полей")
            self._log(f"📏 Длина промпта: {len(prompt)} символов")
            
            return prompt
            
        except Exception as e:
            self._log(f"❌ Ошибка генерации промпта: {e}")
            return self._get_fallback_training_prompt(task_type)

    def _generate_gemini_training_prompt(self, enabled_fields) -> str:
        """Генерирует промпт для Gemini на основе активных полей"""
        prompt_parts = [
            "Проанализируй изображение документа и извлеки следующие поля в JSON формате:",
            "",
            "ПОЛЯ ДЛЯ ИЗВЛЕЧЕНИЯ (в порядке приоритета):"
        ]
        
        # Сортируем поля по приоритету и позиции
        sorted_fields = sorted(enabled_fields, key=lambda f: (f.priority, f.position))
        
        for field in sorted_fields:
            # Создаем описание поля с ключевыми словами
            keywords_str = ", ".join(field.gemini_keywords[:3])
            required_marker = " [ОБЯЗАТЕЛЬНОЕ]" if field.required else ""
            
            prompt_parts.append(
                f"- {field.gemini_keywords[0]}: {field.description}{required_marker}"
            )
            prompt_parts.append(f"  Варианты поиска: {keywords_str}")
            
            # Добавляем информацию о типе данных
            if field.data_type == "date":
                prompt_parts.append("  Формат: дата в виде DD.MM.YYYY или текстом")
            elif field.data_type == "currency":
                prompt_parts.append("  Формат: числовое значение с валютой")
            elif field.data_type == "number":
                prompt_parts.append("  Формат: числовое значение")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "ФОРМАТ ОТВЕТА:",
            "Возвращай ответ ТОЛЬКО в формате JSON.",
            "Используй точные названия полей как указано выше (первое ключевое слово).",
            "Если поле не найдено в документе, используй \"N/A\".",
            "Не добавляй никаких дополнительных объяснений или текста вне JSON.",
            "",
            "ПРИОРИТЕТ:",
            "Обязательные поля должны быть заполнены максимально точно.",
            "Если есть несколько вариантов значения, выбирай наиболее точный."
        ])
        
        return "\n".join(prompt_parts)

    def _generate_layoutlm_training_prompt(self, enabled_fields) -> str:
        """Генерирует промпт для LayoutLM на основе активных полей"""
        # LayoutLM использует тот же промпт что и Gemini, но с акцентом на пространственную информацию
        base_prompt = self._generate_gemini_training_prompt(enabled_fields)
        
        layoutlm_addition = [
            "",
            "ДОПОЛНИТЕЛЬНО ДЛЯ LAYOUTLM:",
            "Учитывай пространственное расположение текста в документе.",
            "Связывай логически связанные поля по их расположению.",
            f"Целевые лейблы для разметки: {', '.join(self._get_layoutlm_labels_from_fields(enabled_fields))}"
        ]
        
        return base_prompt + "\n".join(layoutlm_addition)

    def _generate_donut_training_prompt(self, enabled_fields) -> str:
        """Генерирует промпт для Donut на основе активных полей"""
        prompt_parts = [
            "Извлеки структурированные данные из изображения документа.",
            "Сфокусируйся на следующих полях:",
            ""
        ]
        
        # Для Donut используем более компактный формат
        sorted_fields = sorted(enabled_fields, key=lambda f: (f.priority, f.position))
        
        for field in sorted_fields:
            if field.required:
                prompt_parts.append(f"• {field.gemini_keywords[0]}: {field.description} [ОБЯЗАТЕЛЬНО]")
            else:
                prompt_parts.append(f"• {field.gemini_keywords[0]}: {field.description}")
        
        prompt_parts.extend([
            "",
            "Формат ответа: JSON с указанными полями.",
            "Пропущенные поля заполняй \"N/A\"."
        ])
        
        return "\n".join(prompt_parts)

    def _generate_generic_training_prompt(self, enabled_fields) -> str:
        """Генерирует универсальный промпт на основе активных полей"""
        return self._generate_gemini_training_prompt(enabled_fields)

    def _get_layoutlm_labels_from_fields(self, enabled_fields) -> List[str]:
        """Извлекает LayoutLM лейблы из активных полей"""
        labels = []
        for field in enabled_fields:
            labels.extend(field.layoutlm_labels)
        return list(set(labels))  # Убираем дубликаты

    def _get_fallback_training_prompt(self, task_type: str) -> str:
        """Возвращает fallback промпт если FieldManager недоступен"""
        fallback_prompts = {
            "gemini": """
Проанализируй изображение счета или фактуры и извлеки следующие поля в JSON формате:

- Поставщик: название компании-поставщика
- № Счета: номер счета/инвойса
- Дата счета: дата выставления счета
- Категория: тип товаров/услуг
- Товары: описание товаров/услуг
- Сумма без НДС: сумма без налога
- НДС %: ставка НДС в процентах
- Сумма с НДС: итоговая сумма с налогом
- Валюта: валюта документа
- ИНН: ИНН поставщика
- КПП: КПП поставщика
- Комментарии: дополнительные примечания

ВАЖНО: Возвращай ответ ТОЛЬКО в формате JSON. Если поле не найдено, используй "N/A".
            """.strip(),
            
            "layoutlm": """
Извлеки данные из документа, учитывая пространственное расположение элементов.
Сфокусируйся на основных полях: поставщик, номер счета, дата, сумма.
Формат ответа: JSON с указанными полями.
            """.strip(),
            
            "donut": """
Структурированное извлечение данных из документа.
Основные поля: поставщик, номер документа, дата, итоговая сумма.
Ответ в формате JSON.
            """.strip()
        }
        
        return fallback_prompts.get(task_type, fallback_prompts["gemini"])

    def get_entity_types_from_fields(self) -> List[str]:
        """
        Возвращает список типов сущностей на основе активных полей
        
        Returns:
            List[str]: Список типов сущностей для IOB2 разметки
        """
        if not self.field_manager:
            # Fallback типы сущностей
            return [
                "INVOICE_NUMBER", "DATE", "TOTAL_AMOUNT", "VENDOR", "CUSTOMER", 
                "INN", "KPP", "DESCRIPTION", "CATEGORY", "CURRENCY"
            ]
        
        try:
            enabled_fields = self.field_manager.get_enabled_fields()
            entity_types = []
            
            for field in enabled_fields:
                # Используем LayoutLM лейблы как типы сущностей
                entity_types.extend(field.layoutlm_labels)
            
            # Убираем дубликаты и добавляем базовый тип "O" (Outside)
            unique_types = list(set(entity_types))
            if "O" not in unique_types:
                unique_types.append("O")
                
            self._log(f"📋 Извлечено {len(unique_types)} типов сущностей из активных полей")
            return unique_types
            
        except Exception as e:
            self._log(f"❌ Ошибка извлечения типов сущностей: {e}")
            return ["INVOICE_NUMBER", "DATE", "TOTAL_AMOUNT", "VENDOR", "O"]

    def _save_dataset_metadata(self, dataset_folder: str, metadata: Dict):
        """
        Сохраняет метаданные датасета включая информацию о полях и промптах
        
        Args:
            dataset_folder: Папка датасета
            metadata: Словарь с метаданными
        """
        try:
            import json
            from datetime import datetime
            
            # Добавляем временные метки
            metadata['created_at'] = datetime.now().isoformat()
            metadata['version'] = '1.1'  # Версия с поддержкой FieldManager
            
            # Путь к файлу метаданных
            metadata_path = os.path.join(dataset_folder, 'dataset_metadata.json')
            
            # Сохраняем метаданные
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self._log(f"💾 Метаданные датасета сохранены: {metadata_path}")
            
        except Exception as e:
            self._log(f"⚠️ Ошибка сохранения метаданных датасета: {e}")

    def load_dataset_metadata(self, dataset_folder: str) -> Optional[Dict]:
        """
        Загружает метаданные датасета
        
        Args:
            dataset_folder: Папка датасета
            
        Returns:
            Optional[Dict]: Метаданные или None при ошибке
        """
        try:
            import json
            
            metadata_path = os.path.join(dataset_folder, 'dataset_metadata.json')
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self._log(f"📂 Загружены метаданные датасета: {metadata_path}")
            return metadata
            
        except Exception as e:
            self._log(f"⚠️ Ошибка загрузки метаданных датасета: {e}")
            return None
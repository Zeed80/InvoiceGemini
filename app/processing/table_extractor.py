"""
Извлечение табличных данных из предсказаний LayoutLMv3

Данный модуль содержит классы и функции для извлечения структурированных данных о товарах
из предсказаний модели LayoutLMv3, используя графовый подход для определения строк и столбцов таблицы.
"""

import numpy as np
import re
from typing import List, Dict, Any, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)

class TableExtractor:
    """
    Класс для извлечения табличных данных из предсказаний LayoutLMv3 
    с использованием графового подхода
    """
    
    def __init__(self, 
                 vertical_proximity_threshold: float = 0.05,
                 horizontal_overlap_threshold: float = 0.3,
                 price_pattern: str = r'\d+[\s\.,]?\d*',
                 ignore_total_rows: bool = True):
        """
        Инициализация экстрактора таблиц.
        
        Args:
            vertical_proximity_threshold: Пороговое значение для группировки элементов в строки
            horizontal_overlap_threshold: Пороговое значение перекрытия для определения столбцов
            price_pattern: Регулярное выражение для распознавания чисел и цен
            ignore_total_rows: Игнорировать ли строки с итоговыми суммами
        """
        self.vertical_proximity_threshold = vertical_proximity_threshold
        self.horizontal_overlap_threshold = horizontal_overlap_threshold
        self.price_pattern = price_pattern
        self.ignore_total_rows = ignore_total_rows
        
    def extract_items(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлекает структурированные данные о товарах из предсказаний LayoutLMv3.
        
        Args:
            predictions: Словарь с предсказаниями модели LayoutLMv3
                содержащий ключи 'words', 'boxes' и 'labels'
                
        Returns:
            List[Dict[str, Any]]: Список товаров с ключами 'name', 'quantity', 'price', 'amount'
        """
        if not predictions or 'words' not in predictions or 'boxes' not in predictions or 'labels' not in predictions:
            logger.warning("Неполные предсказания: необходимы ключи 'words', 'boxes' и 'labels'")
            return []
            
        words = predictions['words']
        boxes = predictions['boxes']
        labels = predictions['labels']
        
        # Проверяем что все списки имеют одинаковую длину
        if not (len(words) == len(boxes) == len(labels)):
            logger.warning(f"Несоответствие размеров данных: words={len(words)}, boxes={len(boxes)}, labels={len(labels)}")
            return []
            
        # Формируем словарь элементов
        items_data = []
        for i, (word, box, label) in enumerate(zip(words, boxes, labels)):
            items_data.append({
                'id': i,
                'word': word,
                'box': box,
                'label': label,
                'row': -1,  # Будет заполнено при группировке
                'col': -1   # Будет заполнено при группировке
            })
            
        # Группируем элементы по строкам на основе y-координат
        rows = self._group_by_rows(items_data)
        logger.debug(f"Сгруппировано {len(rows)} строк")
        
        # Определяем столбцы для каждой строки
        columns = self._identify_columns(rows)
        logger.debug(f"Определено {len(columns)} столбцов")
        
        # Извлекаем данные о товарах
        items = self._extract_table_items(rows, columns)
        logger.debug(f"Извлечено {len(items)} товаров")
        
        return items
        
    def _group_by_rows(self, items_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Группирует элементы по строкам на основе y-координат.
        
        Args:
            items_data: Список элементов с координатами
            
        Returns:
            List[List[Dict[str, Any]]]: Список строк, каждая строка - список элементов
        """
        if not items_data:
            return []
            
        # Сортируем элементы по вертикальной позиции (y-координата)
        sorted_items = sorted(items_data, key=lambda x: x['box'][1])  # сортировка по y1 (верхняя координата)
        
        # Получаем максимальную высоту документа для нормализации
        max_doc_height = max(item['box'][3] for item in items_data) - min(item['box'][1] for item in items_data)
        threshold_px = self.vertical_proximity_threshold * max_doc_height
        
        # Группируем элементы по строкам
        rows = []
        current_row = []
        last_y = None
        
        for item in sorted_items:
            current_y = item['box'][1]  # верхняя координата y
            
            # Если это первый элемент или элемент находится близко по вертикали к предыдущему
            if last_y is None or abs(current_y - last_y) <= threshold_px:
                current_row.append(item)
            else:
                # Начинаем новую строку
                if current_row:
                    # Сортируем элементы в строке по горизонтальной позиции
                    current_row = sorted(current_row, key=lambda x: x['box'][0])  # сортировка по x1
                    rows.append(current_row)
                current_row = [item]
            
            last_y = current_y
        
        # Добавляем последнюю строку
        if current_row:
            current_row = sorted(current_row, key=lambda x: x['box'][0])  # сортировка по x1
            rows.append(current_row)
        
        # Присваиваем номера строк
        for row_idx, row in enumerate(rows):
            for item in row:
                item['row'] = row_idx
                
        return rows
        
    def _identify_columns(self, rows: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Определяет столбцы на основе горизонтального перекрытия элементов в строках.
        
        Args:
            rows: Список строк, каждая строка - список элементов
            
        Returns:
            List[List[Dict[str, Any]]]: Список столбцов, каждый столбец - список элементов
        """
        if not rows:
            return []
            
        # Находим строку с максимальным количеством элементов (предположительно, заголовок)
        header_row_idx = max(range(len(rows)), key=lambda i: len(rows[i]))
        header_row = rows[header_row_idx]
        
        # Если нашли только одну строку с минимумом элементов, используем ее
        if len(header_row) <= 1:
            # Попробуем разделить элементы на равные столбцы по горизонтали
            all_elements = [item for row in rows for item in row]
            
            # Найдем минимальную и максимальную x-координату для определения диапазона
            if not all_elements:
                return []
            
            min_x = min(item['box'][0] for item in all_elements)
            max_x = max(item['box'][2] for item in all_elements)
            
            # Если у нас есть хотя бы 4 строки, предположим, что у нас 4 столбца
            # (имя, количество, цена, сумма)
            num_cols = 4 if len(rows) >= 4 else len(rows)
            
            # Создаем столбцы
            col_width = (max_x - min_x) / num_cols if num_cols > 0 else 0
            columns = [[] for _ in range(num_cols)]
            
            for item in all_elements:
                item_center_x = (item['box'][0] + item['box'][2]) / 2
                col_idx = min(int((item_center_x - min_x) / col_width) if col_width > 0 else 0, num_cols - 1)
                item['col'] = col_idx
                columns[col_idx].append(item)
            
            return columns
        
        # Определяем начальные столбцы на основе заголовков
        columns = [[] for _ in range(len(header_row))]
        for col_idx, item in enumerate(header_row):
            item['col'] = col_idx
            columns[col_idx].append(item)
        
        # Для каждой строки (кроме строки заголовков)
        for row_idx, row in enumerate(rows):
            if row_idx == header_row_idx:
                continue
                
            for item in row:
                # Находим наиболее подходящий столбец для этого элемента
                best_col_idx = self._find_best_column(item, header_row)
                if best_col_idx != -1:
                    item['col'] = best_col_idx
                    columns[best_col_idx].append(item)
                    
        return columns
    
    def _find_best_column(self, item: Dict[str, Any], header_row: List[Dict[str, Any]]) -> int:
        """
        Находит наиболее подходящий столбец для элемента на основе горизонтального перекрытия.
        
        Args:
            item: Элемент для определения столбца
            header_row: Строка с заголовками
            
        Returns:
            int: Индекс наиболее подходящего столбца или -1, если подходящий столбец не найден
        """
        item_x1, _, item_x2, _ = item['box']
        item_width = item_x2 - item_x1
        
        best_overlap = -1
        best_col_idx = -1
        
        for col_idx, header_item in enumerate(header_row):
            header_x1, _, header_x2, _ = header_item['box']
            
            # Вычисляем перекрытие
            overlap_x1 = max(item_x1, header_x1)
            overlap_x2 = min(item_x2, header_x2)
            
            if overlap_x2 > overlap_x1:
                overlap_width = overlap_x2 - overlap_x1
                overlap_ratio = overlap_width / item_width
                
                if overlap_ratio > self.horizontal_overlap_threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_col_idx = col_idx
        
        return best_col_idx
    
    def _extract_table_items(self, rows: List[List[Dict[str, Any]]], columns: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Извлекает данные о товарах из сгруппированных строк и столбцов.
        
        Args:
            rows: Список строк, каждая строка - список элементов
            columns: Список столбцов, каждый столбец - список элементов
            
        Returns:
            List[Dict[str, Any]]: Список товаров с ключами 'name', 'quantity', 'price', 'amount'
        """
        if not rows or not columns:
            return []
            
        # Определяем столбцы по меткам
        name_col_idx = self._find_column_by_label(columns, ['ITEM_NAME', 'NAME', 'DESCRIPTION'])
        quantity_col_idx = self._find_column_by_label(columns, ['QUANTITY', 'QTY', 'COUNT'])
        price_col_idx = self._find_column_by_label(columns, ['PRICE', 'UNIT_PRICE'])
        amount_col_idx = self._find_column_by_label(columns, ['AMOUNT', 'TOTAL', 'SUM'])
        
        logger.debug(f"Индексы столбцов: name={name_col_idx}, quantity={quantity_col_idx}, price={price_col_idx}, amount={amount_col_idx}")
        
        # Если не нашли ключевые столбцы, пытаемся определить их по позиции и содержанию
        if name_col_idx == -1 or quantity_col_idx == -1 or price_col_idx == -1 or amount_col_idx == -1:
            # Предполагаем, что столбцы идут в порядке Название, Количество, Цена, Сумма
            col_indices = list(range(min(4, len(columns))))
            if name_col_idx == -1 and col_indices:
                name_col_idx = col_indices[0]
            if quantity_col_idx == -1 and len(col_indices) > 1:
                quantity_col_idx = col_indices[1]
            if price_col_idx == -1 and len(col_indices) > 2:
                price_col_idx = col_indices[2]
            if amount_col_idx == -1 and len(col_indices) > 3:
                amount_col_idx = col_indices[3]
                
            logger.debug(f"Скорректированные индексы столбцов: name={name_col_idx}, quantity={quantity_col_idx}, price={price_col_idx}, amount={amount_col_idx}")
        
        # Извлекаем данные о товарах из таблицы
        items = []
        
        # Пропускаем строку заголовков (первую строку)
        for row_idx, row in enumerate(rows[1:], 1):
            # Пропускаем строки без данных или с неполными данными
            if len(row) < 2:
                continue
                
            # Извлекаем данные из соответствующих столбцов
            name = self._get_value_from_row_column(row, name_col_idx, 'word', '')
            
            # Определяем, не является ли строка итоговой
            if self.ignore_total_rows and self._is_total_row(name.lower()):
                logger.debug(f"Пропускаем итоговую строку: {name}")
                continue
                
            # Обрабатываем числовые значения
            quantity_str = self._get_value_from_row_column(row, quantity_col_idx, 'word', '0')
            price_str = self._get_value_from_row_column(row, price_col_idx, 'word', '0')
            amount_str = self._get_value_from_row_column(row, amount_col_idx, 'word', '0')
            
            # Преобразуем строки в числа
            quantity = self._parse_number(quantity_str)
            price = self._parse_number(price_str)
            amount = self._parse_number(amount_str)
            
            # Проверяем согласованность данных (если возможно)
            if quantity > 0 and price > 0 and amount == 0:
                # Если отсутствует сумма, но есть цена и количество, вычисляем сумму
                amount = quantity * price
                logger.debug(f"Вычислена сумма для '{name}': {quantity} * {price} = {amount}")
            elif quantity > 0 and price == 0 and amount > 0:
                # Если отсутствует цена, но есть количество и сумма, вычисляем цену
                price = amount / quantity
                logger.debug(f"Вычислена цена для '{name}': {amount} / {quantity} = {price}")
            elif quantity == 0 and price > 0 and amount > 0:
                # Если отсутствует количество, но есть цена и сумма, вычисляем количество
                quantity = amount / price
                logger.debug(f"Вычислено количество для '{name}': {amount} / {price} = {quantity}")
            
            # В зависимости от метки строки, добавляем её как товар или как итог
            if name.lower() in ['всего', 'итого', 'total', 'сумма', 'итог']:
                # Это строка с итогом
                item = {
                    'name': name,
                    'quantity': 0.0,
                    'price': 0.0,
                    'amount': amount  # Сумма помещается в поле amount для итоговых строк
                }
            else:
                # Обычная строка товара
                item = {
                    'name': name,
                    'quantity': quantity,
                    'price': price,
                    'amount': amount
                }
                
            items.append(item)
            
        return items
    
    def _find_column_by_label(self, columns: List[List[Dict[str, Any]]], label_variants: List[str]) -> int:
        """
        Находит индекс столбца, содержащего элементы с указанными метками.
        
        Args:
            columns: Список столбцов
            label_variants: Список возможных меток для поиска
            
        Returns:
            int: Индекс столбца или -1, если не найден
        """
        for col_idx, column in enumerate(columns):
            for item in column:
                if any(label.upper() in item['label'].upper() for label in label_variants):
                    return col_idx
        return -1

    def _get_value_from_row_column(self, row: List[Dict[str, Any]], col_idx: int, key: str, default: Any) -> Any:
        """
        Извлекает значение из указанного столбца строки.
        
        Args:
            row: Строка (список элементов)
            col_idx: Индекс столбца
            key: Ключ для извлечения значения
            default: Значение по умолчанию
            
        Returns:
            Any: Извлеченное значение или значение по умолчанию
        """
        for item in row:
            if item['col'] == col_idx:
                return item.get(key, default)
        return default
    
    def _is_total_row(self, text: str) -> bool:
        """
        Определяет, является ли строка итоговой.
        
        Args:
            text: Текст для проверки
            
        Returns:
            bool: True, если строка итоговая, иначе False
        """
        total_keywords = ['всего', 'итого', 'total', 'сумма', 'sum', 'итог']
        return any(keyword in text for keyword in total_keywords)
        
    def _parse_number(self, text: str) -> float:
        """
        Преобразует строку с числом в float.
        
        Args:
            text: Строка с числом
            
        Returns:
            float: Извлеченное число или 0.0 в случае ошибки
        """
        if not text:
            return 0.0
            
        # Удаляем все не цифровые символы, кроме точки и запятой
        cleaned_text = re.sub(r'[^\d\.,]', '', text)
        
        # Заменяем запятую на точку (для обработки десятичных чисел)
        cleaned_text = cleaned_text.replace(',', '.')
        
        try:
            return float(cleaned_text)
        except ValueError:
            return 0.0
            
    @staticmethod
    def _get_box_height(box: List[int]) -> int:
        """
        Вычисляет высоту ограничивающей рамки.
        
        Args:
            box: Ограничивающая рамка [x1, y1, x2, y2]
            
        Returns:
            int: Высота рамки
        """
        return box[3] - box[1]

def extract_table_items_from_layoutlm(predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Извлекает данные о товарах из предсказаний LayoutLMv3.
    
    Args:
        predictions: Словарь с предсказаниями модели LayoutLMv3
            содержащий ключи 'words', 'boxes' и 'labels'
            
    Returns:
        List[Dict[str, Any]]: Список товаров с ключами 'name', 'quantity', 'price', 'amount'
    """
    # При использовании функции-обертки важно игнорировать итоговые строки, чтобы не дублировать информацию,
    # которая уже есть в полях invoice_data['total']
    extractor = TableExtractor(ignore_total_rows=True)
    return extractor.extract_items(predictions) 
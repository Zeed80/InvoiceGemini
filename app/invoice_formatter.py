"""
Модуль форматирования данных счетов-фактур.
Обеспечивает преобразование извлеченных данных в стандартный формат.

DONE: Добавлена поддержка ставки НДС по умолчанию из настроек приложения.
"""
import re
import os
from . import config # Добавляем импорт конфига для доступа к DEFAULT_VAT_RATE
from .settings_manager import settings_manager # Для доступа к настройкам

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
        Если расчет невозможен, возвращает ставку НДС по умолчанию из настроек.
        """
        try:
            total = float(str(total_amount).replace(',', '.'))
            base = float(str(amount_without_vat).replace(',', '.'))
            
            if base == 0 or total == 0:
                # Если суммы равны 0 или невалидны, возвращаем ставку по умолчанию
                return settings_manager.get_default_vat_rate()
            
            vat_amount = total - base
            
            if base <= 0 or vat_amount < 0:
                # Если сумма без НДС отрицательная или НДС отрицательный - используем ставку по умолчанию
                return settings_manager.get_default_vat_rate()
            
            vat_rate = (vat_amount / base) * 100  # Ставка НДС в процентах
            
            # Округляем ставку до ближайшего стандартного значения (в России: 0%, 10%, 20%)
            if 0 <= vat_rate < 5:
                return 0
            elif 5 <= vat_rate < 15:
                return 10
            elif 15 <= vat_rate < 25:
                return 20
            else:
                # Если ставка НДС необычно высокая, вероятно, данные неверны, используем ставку по умолчанию
                return settings_manager.get_default_vat_rate()
        except (ValueError, TypeError, AttributeError):
            # В случае любой ошибки при конвертации возвращаем ставку НДС по умолчанию
            return settings_manager.get_default_vat_rate()

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
        
        # Обработка альтернативных полей Gemini
        if not company and 'seller_name' in invoice_data:
            company = invoice_data.get('seller_name', '')
            
        # Получаем прочие поля с разными возможными именами
        items = invoice_data.get('items', [])
        description = invoice_data.get('description', invoice_data.get('description_gemini', ''))
        category = invoice_data.get('category', invoice_data.get('category_gemini', ''))
        total_amount_str = invoice_data.get('total_amount', invoice_data.get('total', '0.00'))
        amount_without_vat = invoice_data.get('amount_without_vat', invoice_data.get('amount_without_vat_gemini', invoice_data.get('subtotal', '')))
        vat_percent = invoice_data.get('vat_percent', invoice_data.get('vat_percent_gemini', invoice_data.get('vat_rate', '')))
        currency = invoice_data.get('currency', 'RUB')
        note = invoice_data.get('note', invoice_data.get('note_gemini', ''))
        
        # Новые поля из модели Gemini
        if not invoice_number:
            invoice_number = invoice_data.get('invoice_num', '')
            
        if not date_str:
            date_str = invoice_data.get('invoice_date', '')
            
        if not description and items:
            # Если есть items, но нет description, форматируем описание из items
            if isinstance(items, list) and len(items) > 0:
                description = InvoiceFormatter.format_description(items)
        
        # 1. Номер счета
        clean_invoice_number = invoice_number.strip() if invoice_number else "N/A"

        # 2. Дата счета
        formatted_date = date_str.strip() if date_str else "N/A"

        # 3. Наименование компании-отправителя
        if not company:
            company = "N/A"

        # Инициализируем переменные перед использованием
        amount_without_vat_final_str = "N/A"
        vat_percent_final_str = "N/A"

        # 4. Сумма без НДС
        if amount_without_vat:
            amount_without_vat_final_str = InvoiceFormatter.format_number_with_comma(amount_without_vat, decimal_places=2)
        else:
            amount_without_vat_final_str = "N/A"

        # 5. Процент НДС (если указан)
        if vat_percent:
            vat_percent_final_str = InvoiceFormatter.format_number_with_comma(vat_percent.replace('%','').strip(), decimal_places=1)
        else:
            vat_percent_final_str = "N/A"

        # 6. Общая сумма к оплате
        if not total_amount_str:
            total_amount_str = "N/A"
            total_float = 0.0
        else:
            try:
                total_float = float(str(total_amount_str).replace(',', '.'))
            except (ValueError, TypeError):
                total_float = 0.0

        # 7. Описание товаров/услуг
        if not description:
            description = "N/A"

        # 11. Общая сумма без НДС
        if amount_without_vat_final_str != "N/A":
            try:
                amount_without_vat_float = float(amount_without_vat_final_str.replace(',', '.'))
                if total_float > 0 and vat_percent_final_str == "N/A":
                    if amount_without_vat_float > 0:
                        vat_percent_final_str = InvoiceFormatter.calculate_vat_rate(total_float, amount_without_vat_float)
                    else:
                        vat_percent_final_str = "0,0"
            except (ValueError, TypeError):
                pass
        elif total_float > 0 and vat_percent_final_str != "N/A":
            try:
                vat_rate_float = float(vat_percent_final_str.replace(',', '.'))
                if vat_rate_float >= 0:
                    calculated_amount_without_vat = total_float / (1 + vat_rate_float / 100)
                    amount_without_vat_final_str = InvoiceFormatter.format_number_with_comma(calculated_amount_without_vat, decimal_places=2)
            except (ValueError, TypeError):
                pass

        # Проверка на равенство сумм с НДС и без НДС (значит НДС 0%)
        if total_float > 0 and amount_without_vat_final_str != "N/A":
            try:
                amount_without_vat_float_check = float(amount_without_vat_final_str.replace(',', '.'))
                if abs(total_float - amount_without_vat_float_check) < 0.01:
                    if vat_percent_final_str == "N/A": 
                        vat_percent_final_str = "0,0"
            except (ValueError, TypeError):
                pass
        
        # Форматирование финальных данных
        formatted_total = InvoiceFormatter.format_number_with_comma(total_float, decimal_places=2)
        final_currency = currency if currency and currency.upper() != 'N/A' else 'RUB'
        final_note = note if note and note.upper() != 'N/A' else "N/A"
        final_category = category if category and category.upper() != 'N/A' else "N/A"
        
        result = {
            "№ счета": clean_invoice_number,
            "Дата счета": formatted_date,
            "Категория": final_category,
            "Поставщик": company,
            "Товары": description,
            "Сумма без НДС": amount_without_vat_final_str,
            "% НДС": vat_percent_final_str,
            "Сумма с НДС": formatted_total,
            "Валюта": final_currency,
            "INN": invoice_data.get('inn', 'N/A'),
            "KPP": invoice_data.get('kpp', 'N/A'),
            "Примечание": final_note
        }
        
        return result 
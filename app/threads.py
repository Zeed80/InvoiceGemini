"""
Модуль содержит классы для выполнения фоновых задач.
"""
import os
# УБИРАЕМ: os.environ["HF_HUB_OFFLINE"] = "0"
# УБИРАЕМ: print("DEBUG threads.py: Принудительно установлен HF_HUB_OFFLINE=0 в начале файла.")

import sys
import time
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from . import config
from . import utils
from .processing_engine import ModelManager
from .settings_manager import settings_manager


class ProcessingThread(QThread):
    """Поток для выполнения обработки изображения в фоновом режиме."""
    progress_signal = pyqtSignal(int)         # Сигнал для обновления прогресса (0-100)
    finished_signal = pyqtSignal(object)      # Сигнал завершения (передает результат или None)
    partial_result_signal = pyqtSignal(dict)  # NEW: Сигнал для частичных результатов (для папки)
    error_signal = pyqtSignal(str)            # Сигнал ошибки

    # ИЗМЕНЯЕМ КОНСТРУКТОР: добавляем model_manager
    def __init__(self, model_type, input_path, ocr_lang=None, is_folder=False, model_manager=None, parent=None):
        super().__init__(parent)
        self.model_type = model_type
        self.input_path = input_path
        self.ocr_lang = ocr_lang
        self.is_folder = is_folder # NEW: Флаг обработки папки
        # NEW: Сохраняем переданный model_manager
        if model_manager is None:
            raise ValueError("ModelManager instance must be provided to ProcessingThread")
        self.model_manager = model_manager

    def run(self):
        """Запуск обработки в потоке."""
        try:
            print(f"DEBUG ProcessingThread: run for model_type = {self.model_type}") # Отладка
            processor = self.model_manager.get_model(self.model_type)
            
            # Проверяем, загружен ли процессор, и пытаемся загрузить, если нет
            if not processor.is_loaded:
                print(f"DEBUG ProcessingThread: Processor for {self.model_type} is not loaded. Attempting download...")
                # Используем self.model_manager для загрузки. model_id будет взят из настроек внутри download_model/load_model
                if not self.model_manager.download_model(self.model_type):
                    # Если загрузка не удалась, выбрасываем исключение или отправляем сигнал ошибки
                    error_msg = f"Не удалось загрузить модель {self.model_type} в потоке."
                    print(f"ERROR ProcessingThread: {error_msg}")
                    self.error_signal.emit(error_msg)
                    return # Завершаем поток
                
                # После попытки загрузки, процессор должен быть загружен.
                # Можно переполучить процессор или проверить флаг is_loaded еще раз, 
                # но download_model должен был обновить состояние is_loaded у существующего экземпляра.
                if not processor.is_loaded:
                    error_msg = f"Модель {self.model_type} все еще не загружена после попытки скачивания в потоке."
                    print(f"ERROR ProcessingThread: {error_msg}")
                    self.error_signal.emit(error_msg)
                    return # Завершаем поток
                else:
                    print(f"DEBUG ProcessingThread: Model {self.model_type} successfully loaded in thread.")
            
            # NEW: Логика для папки
            if self.is_folder:
                print(f"Начало пакетной обработки папки: {self.input_path}")
                
                # NEW: Получаем задержку из настроек
                batch_delay = settings_manager.get_int(
                    'Misc', 
                    'batch_processing_delay', 
                    config.DEFAULT_BATCH_PROCESSING_DELAY
                )
                print(f"Установлена задержка между файлами: {batch_delay} сек.")
                
                supported_files = []
                try:
                    # Используем try-except на случай ошибки доступа к папке
                    for filename in os.listdir(self.input_path):
                        file_path = os.path.join(self.input_path, filename)
                        # Проверяем, что это файл и формат поддерживается
                        if os.path.isfile(file_path) and utils.is_supported_format(file_path):
                            supported_files.append(file_path)
                except OSError as e:
                     print(f"Ошибка чтения директории {self.input_path}: {e}")
                     self.error_signal.emit(f"Ошибка чтения папки: {e}")
                     return
                
                if not supported_files:
                    print("В папке не найдены поддерживаемые файлы.")
                    self.finished_signal.emit(None) # Завершаем, если файлов нет
                    return

                total_files = len(supported_files)
                print(f"Найдено {total_files} файлов для обработки.")

                for i, file_path in enumerate(supported_files):
                    # NEW: Добавляем задержку перед обработкой файла (кроме первого)
                    if i > 0 and batch_delay > 0:
                        print(f"Задержка на {batch_delay} сек. перед следующим файлом...")
                        time.sleep(batch_delay)
                        
                    print(f"Обработка файла {i+1}/{total_files}: {os.path.basename(file_path)}")
                    try:
                        # Получаем пользовательский промпт (если нужен)
                        custom_prompt = settings_manager.get_string('Prompts', f'{self.model_type}_prompt', '')
                        # Обрабатываем файл
                        result = processor.process_image(file_path, self.ocr_lang, custom_prompt=custom_prompt)
                        if result:
                            # Применяем маппинг полей для всех моделей
                            if self.model_type == 'gemini':
                                result = self._map_gemini_fields(result)
                            elif self.model_type == 'donut':
                                result = self._map_donut_fields(result)
                            
                            # Добавляем имя файла к результатам для идентификации
                            # result["__source_file__"] = os.path.basename(file_path) # Убрал, т.к. пока не используется в таблице
                            self.partial_result_signal.emit(result) # Отправляем результат для файла
                        else:
                            print(f"Файл {os.path.basename(file_path)} обработан, но результат пуст.")
                            # Можно отправить пустой результат или маркер ошибки, если нужно отобразить это в таблице
                            # self.partial_result_signal.emit({"__source_file__": os.path.basename(file_path), "error": "Пустой результат"})
                    except Exception as file_error:
                        # Логируем ошибку обработки файла, но продолжаем со следующим
                        print(f"Ошибка при обработке файла {os.path.basename(file_path)}: {str(file_error)}")
                        # Можно отправить сигнал об ошибке для файла, если нужно
                        self.error_signal.emit(f"Ошибка файла {os.path.basename(file_path)}: {str(file_error)}")
                    
                    # Обновляем общий прогресс
                    progress = int((i + 1) * 100 / total_files)
                    self.progress_signal.emit(progress)
                
                print("Пакетная обработка завершена.")
                
                # Очищаем временные файлы, если использовался GeminiProcessor
                if self.model_type == 'gemini' and hasattr(processor, 'cleanup_temp_files'):
                    print("Запускаем очистку временных файлов Gemini...")
                    processor.cleanup_temp_files()
                
                self.finished_signal.emit(None) # Финальный сигнал для папки (результаты уже отправлены)

            # Логика для одного файла (восстановлена и использует custom_prompt)
            else:
                print(f"Начало обработки файла: {self.input_path}")
                self.progress_signal.emit(10) # Начальный прогресс
                
                # Имитация загрузки/подготовки (можно убрать, если не нужно)
                self.msleep(100) 
                self.progress_signal.emit(30)
                
                # Получаем пользовательский промпт из настроек
                custom_prompt = settings_manager.get_string('Prompts', f'{self.model_type}_prompt', '')

                # Обработка изображения
                result = processor.process_image(self.input_path, self.ocr_lang, custom_prompt=custom_prompt)
                
                # Применяем маппинг полей для всех моделей
                if self.model_type == 'gemini' and result:
                    result = self._map_gemini_fields(result)
                elif self.model_type == 'donut' and result:
                    result = self._map_donut_fields(result)
                
                # Имитация завершения (можно убрать)
                self.msleep(100)
                self.progress_signal.emit(100)
                
                print("Обработка файла завершена.")
                
                # Очищаем временные файлы, если использовался GeminiProcessor
                if self.model_type == 'gemini' and hasattr(processor, 'cleanup_temp_files'):
                    print("Запускаем очистку временных файлов Gemini...")
                    processor.cleanup_temp_files()
                
                self.finished_signal.emit(result) # Отправляем результат

        except Exception as e:
            print(f"Критическая ошибка в потоке обработки: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def _map_gemini_fields(self, result):
        """
        Маппит поля из результата Gemini в формат для таблицы используя FieldManager.
        
        Args:
            result (dict): Результат от Gemini в формате {\"fields\": [...]}
        
        Returns:
            dict: Результат с маппингом полей для таблицы
        """
        if not result:
            return result
        
        # Импортируем field_manager здесь, чтобы избежать циклических импортов
        try:
            from .field_manager import field_manager
        except ImportError:
            # Fallback на старую логику, если field_manager недоступен
            return self._map_gemini_fields_fallback(result)
        
        # Проверяем, есть ли структура fields
        if 'fields' in result and isinstance(result['fields'], list):
            # Преобразуем массив fields в плоскую структуру
            flat_result = {}
            
            # Получаем актуальный маппинг полей от FieldManager: keyword -> field_id
            field_id_mapping = field_manager.get_field_id_mapping_for_model('gemini')
            
            # Получаем конфигурацию колонок таблицы для преобразования field_id -> display_name
            table_columns = field_manager.get_table_columns()
            field_id_to_display = {col["id"]: col["name"] for col in table_columns}
            
            for field in result['fields']:
                if isinstance(field, dict) and 'field_name' in field and 'field_value' in field:
                    field_name = field['field_name']
                    field_value = field['field_value']
                    
                    # Ищем соответствие в маппинге (регистронезависимо)
                    field_name_lower = field_name.lower()
                    
                    if field_name_lower in field_id_mapping:
                        # Получаем field_id, затем display_name
                        field_id = field_id_mapping[field_name_lower]
                        display_name = field_id_to_display.get(field_id, field_name)
                        flat_result[display_name] = field_value
                    else:
                        # Если точного соответствия нет, используем оригинальное название
                        flat_result[field_name] = field_value
            
            # Добавляем метаданные, если они есть
            for key in ['source_image', 'processed_at']:
                if key in result:
                    flat_result[key] = result[key]
            
            print(f"ОТЛАДКА: Маппинг полей завершен. Итоговая структура: {list(flat_result.keys())}")
            return flat_result
        else:
            # Если структура не содержит fields, используем старую логику с field_mapping
            mapped_result = {}
            field_id_mapping = field_manager.get_field_id_mapping_for_model('gemini')
            table_columns = field_manager.get_table_columns()
            field_id_to_display = {col["id"]: col["name"] for col in table_columns}
            
            for ru_field, value in result.items():
                # Пропускаем служебные поля
                if ru_field in ['source_image', 'processed_at', 'error']:
                    mapped_result[ru_field] = value
                    continue
                    
                # Ищем соответствие в маппинге
                field_lower = ru_field.lower()
                if field_lower in field_id_mapping:
                    field_id = field_id_mapping[field_lower]
                    display_name = field_id_to_display.get(field_id, ru_field)
                    mapped_result[display_name] = value
                else:
                    # Если соответствия нет, используем оригинальное название
                    mapped_result[ru_field] = value
                    
            return mapped_result
    
    def _map_donut_fields(self, result):
        """
        Маппинг полей для модели Donut.
        Donut возвращает поля с английскими названиями, которые нужно преобразовать в русские.
        """
        if not result:
            return result
            
        try:
            # Импортируем field_manager
            from .field_manager import field_manager
            
            # Получаем конфигурацию колонок таблицы
            table_columns = field_manager.get_table_columns()
            field_id_to_display = {col["id"]: col["name"] for col in table_columns}
            
            # Маппинг английских названий Donut на field_id
            donut_to_field_id = {
                "№ Invoice": "invoice_number",
                "Invoice Date": "invoice_date", 
                "Category": "category",
                "Sender": "sender",
                "Description": "description",
                "Amount (0% VAT)": "amount_no_vat",
                "VAT %": "vat_percent",
                "Total": "total",
                "Currency": "currency",
                "Note": "note",
                # Дополнительные поля, если Donut их вернет
                "INN": "inn",
                "KPP": "kpp"
            }
            
            mapped_result = {}
            
            for field_name, value in result.items():
                # Пропускаем служебные поля
                if field_name in ['source_image', 'processed_at', 'error', 'note_gemini', 'raw_response_donut']:
                    mapped_result[field_name] = value
                    continue
                
                # Ищем соответствие в маппинге
                if field_name in donut_to_field_id:
                    field_id = donut_to_field_id[field_name]
                    # Получаем русское название из field_manager
                    if field_id in field_id_to_display:
                        russian_name = field_id_to_display[field_id]
                        mapped_result[russian_name] = value
                        print(f"ОТЛАДКА: Donut поле '{field_name}' -> '{russian_name}' = '{value}'")
                    else:
                        # Если field_id не найден, используем оригинальное название
                        mapped_result[field_name] = value
                        print(f"ОТЛАДКА: Donut поле '{field_name}' (field_id '{field_id}' не найден) = '{value}'")
                else:
                    # Если точного соответствия нет, используем оригинальное название
                    mapped_result[field_name] = value
                    print(f"ОТЛАДКА: Donut неизвестное поле '{field_name}' = '{value}'")
            
            print(f"ОТЛАДКА: Donut маппинг завершен. Русские поля: {list(mapped_result.keys())}")
            return mapped_result
            
        except Exception as e:
            print(f"ОШИБКА: Ошибка маппинга полей Donut: {e}")
            import traceback
            traceback.print_exc()
            # В случае ошибки возвращаем исходный результат
            return result
    
    def _map_gemini_fields_fallback(self, result):
        """
        Fallback метод маппинга полей если FieldManager недоступен.
        """
        if not result:
            return result
        
        # Проверяем, есть ли структура fields
        if 'fields' in result and isinstance(result['fields'], list):
            # Преобразуем массив fields в плоскую структуру
            flat_result = {}
            
            # Статический маппинг как fallback
            field_mapping = {
                # Поставщик
                'company': 'Sender',
                'supplier_name': 'Sender',
                'vendor': 'Sender',
                'поставщик': 'Sender',
                'название компании': 'Sender',
                
                # Номер счета - ИСПРАВЛЕНО: добавляем все варианты от Gemini
                'invoice_number': '№ Invoice',
                'номер счета': '№ Invoice',
                '№ счета': '№ Invoice',      # НОВОЕ: точное соответствие от Gemini
                'invoice_id': '№ Invoice',
                'счет №': '№ Invoice',
                'invoice no': '№ Invoice',
                
                # Дата счета
                'invoice_date': 'Invoice Date',
                'date': 'Invoice Date',
                'дата счета': 'Invoice Date',
                'дата': 'Invoice Date',
                
                # Сумма с НДС (Total)
                'total_amount': 'Total',
                'total': 'Total',
                'сумма с ндс': 'Total',
                'итого': 'Total',
                'amount': 'Total',
                
                # Валюта
                'currency': 'Currency',
                'валюта': 'Currency',
                
                # Категория
                'category': 'Category',
                'категория': 'Category',
                
                # Товары/Описание
                'description': 'Description',
                'товары': 'Description',
                'услуги': 'Description',
                'items': 'Description',
                
                # НДС %
                'vat_rate': 'VAT %',
                'tax_rate': 'VAT %',
                'ндс %': 'VAT %',
                'ставка ндс': 'VAT %',
                
                # Сумма без НДС
                'amount_no_vat': 'Amount (0% VAT)',
                'сумма без ндс': 'Amount (0% VAT)',
                'net_amount': 'Amount (0% VAT)',
                
                # Комментарии
                'note': 'Note',
                'comment': 'Note',
                'комментарий': 'Note',
                'комментарии': 'Note',  # НОВОЕ: множественное число
                'примечание': 'Note',
                
                # НОВЫЕ поля от современного Gemini API
                'инн': 'INN',
                'кпп': 'KPP',
                'inn': 'INN',
                'kpp': 'KPP',
                'tax_id': 'INN',
                'supplier_inn': 'INN',
                'supplier_kpp': 'KPP',
            }
                    
            for field in result['fields']:
                if isinstance(field, dict) and 'field_name' in field and 'field_value' in field:
                    field_name = field['field_name']
                    field_value = field['field_value']
                    
                    # Ищем соответствие в маппинге (регистронезависимо)
                    mapped_field = None
                    field_name_lower = field_name.lower()
                    
                    if field_name_lower in field_mapping:
                        mapped_field = field_mapping[field_name_lower]
                    else:
                        # Если точного соответствия нет, используем оригинальное название
                        mapped_field = field_name
                    
                    flat_result[mapped_field] = field_value
            
            # Добавляем метаданные, если они есть
            for key in ['source_image', 'processed_at']:
                if key in result:
                    flat_result[key] = result[key]
            
            return flat_result
        else:
            # Если структура не содержит fields, используем старую логику
            mapped_result = {}
            
            for ru_field, value in result.items():
                # Ищем соответствие русского поля в маппинге
                if ru_field in config.FIELD_MAPPING:
                    en_field = config.FIELD_MAPPING[ru_field]
                    mapped_result[en_field] = value
                else:
                    # Если соответствия нет, используем оригинальное название
                    mapped_result[ru_field] = value
                    
            return mapped_result


class ModelDownloadThread(QThread):
    """
    Поток для загрузки моделей машинного обучения.
    """
    
    finished_signal = pyqtSignal(bool)   # Сигнал о завершении загрузки (успех/неуспех)
    progress_signal = pyqtSignal(int)    # Сигнал с процентом выполнения
    error_signal = pyqtSignal(str)       # Сигнал с сообщением об ошибке
    
    # ИЗМЕНЯЕМ КОНСТРУКТОР: добавляем model_manager
    def __init__(self, model_type, model_id=None, token=None, model_manager=None, parent=None):
        """
        Инициализация потока загрузки модели.
        
        Args:
            model_type (str): Тип модели ('layoutlm' или 'donut')
            model_id (str, optional): ID модели из Hugging Face Hub
            token (str, optional): Токен для доступа к Hugging Face Hub
            parent (QObject, optional): Родительский объект
        """
        super().__init__(parent)
        self.model_type = model_type
        self.model_id = model_id
        self.token = token  
        # NEW: Сохраняем переданный model_manager
        if model_manager is None:
            raise ValueError("ModelManager instance must be provided to ModelDownloadThread")
        self.model_manager = model_manager
        self.cache_path = os.path.join(config.MODELS_PATH, model_type.lower())
    
    def run(self):
        """Выполняет загрузку модели в отдельном потоке."""
        try:
            # Проверяем, находимся ли мы в оффлайн-режиме
            if config.OFFLINE_MODE:
                self.error_signal.emit("Загрузка моделей невозможна в оффлайн-режиме. Отключите оффлайн-режим в настройках.")
                self.finished_signal.emit(False)
                return
                
            # Эмулируем прогресс
            self.progress_signal.emit(0)
            
            # Устанавливаем таймаут HTTP-запросов (если нужно, но huggingface_hub может сам управлять)
            # os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(config.HTTP_TIMEOUT)
            
            # Пробуем использовать hf_transfer
            try:
                import hf_transfer
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                print("Включено ускорение загрузки с использованием hf_transfer")
            except ImportError:
                print("hf_transfer не найден, используется стандартный метод загрузки")
            
            # Эмуляция начального прогресса
            initial_progress_limit = 30
            progress_step = 5
            for progress in range(0, initial_progress_limit, progress_step):
                self.progress_signal.emit(progress)
                time.sleep(0.1) 
            self.progress_signal.emit(initial_progress_limit)
            
            # Реальная загрузка модели через ПЕРЕДАННЫЙ ModelManager
            # Передаем ID и токен (хотя download_model токен не использует)
            success = self.model_manager.download_model(self.model_type, self.model_id)
            
            # Обновляем прогресс до 80% после завершения download_model
            self.progress_signal.emit(80)
            time.sleep(0.1)
            
            # Имитация завершающих этапов (распаковка, проверка)
            for progress in range(80, 101, 5):
                self.progress_signal.emit(progress)
                time.sleep(0.05)
            
            # Отправляем сигнал о завершении
            self.finished_signal.emit(success)
            
            if not success:
                print(f"ModelDownloadThread: download_model для {self.model_type} ({self.model_id}) вернул False")
                # Ошибка уже должна была быть залогирована и показана пользователю ниже по стеку
                # (например, в layoutlm_download_error или donut_download_error)
                
        except Exception as e:
            # Ловим ошибки, которые могли произойти внутри run()
            error_message = f"Критическая ошибка в потоке загрузки ({self.model_type}): {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            self.error_signal.emit(error_message)
            self.finished_signal.emit(False)


class TesseractCheckThread(QThread):
    """
    Поток для проверки установки Tesseract OCR.
    """
    
    finished_signal = pyqtSignal(bool, str)  # Сигнал о завершении (успех, версия)
    error_signal = pyqtSignal(str)           # Сигнал с сообщением об ошибке
    
    def __init__(self, path=None):
        """
        Инициализация потока проверки Tesseract.
        
        Args:
            path (str, optional): Путь к исполняемому файлу Tesseract
        """
        super().__init__()
        self.path = path
    
    def run(self):
        """Выполнение проверки Tesseract OCR."""
        try:
            # Проверяем путь, если указан
            if self.path:
                if not os.path.exists(self.path):
                    self.error_signal.emit("Указанный файл не существует")
                    self.finished_signal.emit(False, "")
                    return
                
                # Проверяем, что это действительно Tesseract
                import subprocess
                result = subprocess.run([self.path, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "tesseract" in result.stdout.lower():
                    version = result.stdout.strip().split()[1]
                    self.finished_signal.emit(True, version)
                else:
                    self.error_signal.emit("Файл не является исполняемым файлом Tesseract")
                    self.finished_signal.emit(False, "")
            else:
                # Проверяем в системе
                if utils.is_tesseract_installed():
                    path = utils.find_tesseract_in_path()
                    if path:
                        import subprocess
                        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            version = result.stdout.strip().split()[1]
                            self.finished_signal.emit(True, version)
                            return
                    
                    # Если не удалось получить версию, но Tesseract найден
                    self.finished_signal.emit(True, "неизвестная версия")
                else:
                    self.error_signal.emit("Tesseract OCR не найден в системе")
                    self.finished_signal.emit(False, "")
                    
        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False, "") 
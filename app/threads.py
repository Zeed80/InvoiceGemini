"""
Модуль содержит классы для выполнения фоновых задач.
"""
import os
# УБИРАЕМ: os.environ["HF_HUB_OFFLINE"] = "0"
# УБИРАЕМ: print("DEBUG threads.py: Принудительно установлен HF_HUB_OFFLINE=0 в начале файла.")

import sys
import time
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from . import config
from . import utils
from .processing_engine import ModelManager
from .settings_manager import settings_manager

# Настройка логирования
logger = logging.getLogger(__name__)

# Импортируем менеджер кэша
try:
    from .core.cache_manager import get_cache_manager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("CacheManager недоступен, кэширование отключено")


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
        self._should_stop = False  # Флаг для корректной остановки

    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.model_manager = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке ProcessingThread: {e}")

    def run(self):
        """Запуск обработки в потоке."""
        try:
            logger.info(f"Начинаем обработку файла {self.input_path} с моделью {self.model_type}")
            self.progress_signal.emit(10)
            
            processor = self.model_manager.get_model(self.model_type)
            
            # Проверяем, загружен ли процессор, и пытаемся загрузить, если нет
            if not processor.is_loaded:
                logger.info(f"DEBUG ProcessingThread: Processor for {self.model_type} is not loaded. Attempting download...")
                # Используем self.model_manager для загрузки. model_id будет взят из настроек внутри download_model/load_model
                if not self.model_manager.download_model(self.model_type):
                    # Если загрузка не удалась, выбрасываем исключение или отправляем сигнал ошибки
                    error_msg = f"Не удалось загрузить модель {self.model_type} в потоке."
                    logger.error(f"ERROR ProcessingThread: {error_msg}")
                    self.error_signal.emit(error_msg)
                    return # Завершаем поток
                
                # После попытки загрузки, процессор должен быть загружен.
                # Можно переполучить процессор или проверить флаг is_loaded еще раз, 
                # но download_model должен был обновить состояние is_loaded у существующего экземпляра.
                if not processor.is_loaded:
                    error_msg = f"Модель {self.model_type} все еще не загружена после попытки скачивания в потоке."
                    logger.error(f"ERROR ProcessingThread: {error_msg}")
                    self.error_signal.emit(error_msg)
                    return # Завершаем поток
                else:
                    logger.info(f"DEBUG ProcessingThread: Model {self.model_type} successfully loaded in thread.")
            
            # NEW: Логика для папки
            if self.is_folder:
                logger.info(f"Начало пакетной обработки папки: {self.input_path}")
                
                # NEW: Получаем задержку из настроек
                batch_delay = settings_manager.get_int(
                    'Misc', 
                    'batch_processing_delay', 
                    config.DEFAULT_BATCH_PROCESSING_DELAY
                )
                logger.info(f"Установлена задержка между файлами: {batch_delay} сек.")
                
                supported_files = []
                try:
                    # Используем try-except на случай ошибки доступа к папке
                    for filename in os.listdir(self.input_path):
                        file_path = os.path.join(self.input_path, filename)
                        # Проверяем, что это файл и формат поддерживается
                        if os.path.isfile(file_path) and utils.is_supported_format(file_path):
                            supported_files.append(file_path)
                except OSError as e:
                     logger.error(f"Ошибка чтения директории {self.input_path}: {e}")
                     self.error_signal.emit(f"Ошибка чтения папки: {e}")
                     return
                
                if not supported_files:
                    logger.info("В папке не найдены поддерживаемые файлы.")
                    self.finished_signal.emit(None) # Завершаем, если файлов нет
                    return

                total_files = len(supported_files)
                logger.info(f"Найдено {total_files} файлов для обработки.")

                for i, file_path in enumerate(supported_files):
                    # NEW: Добавляем задержку перед обработкой файла (кроме первого)
                    if i > 0 and batch_delay > 0:
                        logger.info(f"Задержка на {batch_delay} сек. перед следующим файлом...")
                        time.sleep(batch_delay)
                        
                    logger.info(f"Обработка файла {i+1}/{total_files}: {os.path.basename(file_path)}")
                    try:
                        # Проверяем кэш для текущего файла
                        if CACHE_AVAILABLE and settings_manager.get_bool('Cache', 'enabled', True):
                            cache_manager = get_cache_manager()
                            file_hash = cache_manager.calculate_file_hash(file_path)
                            cached_result = cache_manager.get_cached_result(file_hash, self.model_type)
                            
                            if cached_result:
                                logger.info(f"Результат получен из кэша для {os.path.basename(file_path)}")
                                self.partial_result_signal.emit(cached_result)
                                # Обновляем прогресс и переходим к следующему файлу
                                progress = int((i + 1) * 100 / total_files)
                                self.progress_signal.emit(progress)
                                continue
                        
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
                            
                            # Сохраняем результат в кэш
                            if CACHE_AVAILABLE and settings_manager.get_bool('Cache', 'enabled', True):
                                try:
                                    cache_manager = get_cache_manager()
                                    file_hash = cache_manager.calculate_file_hash(file_path)
                                    cache_manager.cache_result(file_hash, self.model_type, result, file_path)
                                except Exception as e:
                                    logger.error(f"Ошибка сохранения в кэш для {file_path}: {e}")
                            
                            # Добавляем имя файла к результатам для идентификации
                            # result["__source_file__"] = os.path.basename(file_path) # Убрал, т.к. пока не используется в таблице
                            self.partial_result_signal.emit(result) # Отправляем результат для файла
                        else:
                            logger.info(f"Файл {os.path.basename(file_path)} обработан, но результат пуст.")
                            # Можно отправить пустой результат или маркер ошибки, если нужно отобразить это в таблице
                            # self.partial_result_signal.emit({"__source_file__": os.path.basename(file_path), "error": "Пустой результат"})
                    except Exception as file_error:
                        # Логируем ошибку обработки файла, но продолжаем со следующим
                        logger.error(f"Ошибка при обработке файла {os.path.basename(file_path)}: {str(file_error)}")
                        # Можно отправить сигнал об ошибке для файла, если нужно
                        self.error_signal.emit(f"Ошибка файла {os.path.basename(file_path)}: {str(file_error)}")
                    
                    # Обновляем общий прогресс
                    progress = int((i + 1) * 100 / total_files)
                    self.progress_signal.emit(progress)
                
                logger.info("Пакетная обработка завершена.")
                
                # Очищаем временные файлы, если использовался GeminiProcessor
                if self.model_type == 'gemini' and hasattr(processor, 'cleanup_temp_files'):
                    logger.info("Запускаем очистку временных файлов Gemini...")
                    processor.cleanup_temp_files()
                
                self.finished_signal.emit(None) # Финальный сигнал для папки (результаты уже отправлены)

            # Логика для одного файла (восстановлена и использует custom_prompt)
            else:
                logger.info(f"Начало обработки файла: {self.input_path}")
                self.progress_signal.emit(10) # Начальный прогресс
                
                # Проверяем кэш
                if CACHE_AVAILABLE and settings_manager.get_bool('Cache', 'enabled', True):
                    cached_result = self._check_cache()
                    if cached_result:
                        logger.info(f"Результат получен из кэша для {self.input_path}")
                        self.progress_signal.emit(100)
                        self.finished_signal.emit(cached_result)
                        return
                
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
                
                # Сохраняем результат в кэш
                if CACHE_AVAILABLE and settings_manager.get_bool('Cache', 'enabled', True) and result:
                    self._save_to_cache(result)
                
                # Имитация завершения (можно убрать)
                self.msleep(100)
                self.progress_signal.emit(100)
                
                logger.info("Обработка файла завершена.")
                
                # Очищаем временные файлы, если использовался GeminiProcessor
                if self.model_type == 'gemini' and hasattr(processor, 'cleanup_temp_files'):
                    logger.info("Запускаем очистку временных файлов Gemini...")
                    processor.cleanup_temp_files()
                
                self.finished_signal.emit(result) # Отправляем результат

        except Exception as e:
            logger.error(f"Критическая ошибка в потоке обработки: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def _map_gemini_fields(self, result):
        """
        Маппит поля из результата Gemini в формат для таблицы используя FieldManager.
        
        Args:
            result (dict): Результат от Gemini в формате {"fields": [...]}
        
        Returns:
            dict: Результат с маппингом полей для таблицы
        """
        if not result:
            return result
        
        # Импортируем field_manager здесь, чтобы избежать циклических импортов
        try:
            from .field_manager import field_manager
        except ImportError:
            logger.warning("FieldManager недоступен, используем fallback метод")
            # Fallback на старую логику, если field_manager недоступен
            return self._map_gemini_fields_fallback(result)
        
        try:
            # ИСПРАВЛЕНИЕ: Используем правильный метод get_enabled_fields()
            table_fields = field_manager.get_enabled_fields()
            if not table_fields:
                # Если поля не настроены, используем fallback
                return self._map_gemini_fields_fallback(result)
            
            # ИСПРАВЛЕНИЕ: Создаем маппинг используя правильную структуру TableField
            field_mapping = {}
            for field in table_fields:
                # field - это объект TableField, используем его атрибуты
                field_name = field.display_name
                field_id = field.id
                # Добавляем маппинг по display_name и ключевым словам Gemini
                if field_name:
                    field_mapping[field_name.lower()] = field_name
                # Добавляем маппинг по Gemini keywords
                for keyword in field.gemini_keywords:
                    field_mapping[keyword.lower()] = field_name
            
            # Если Gemini вернул структуру с полем 'fields'
            if 'fields' in result and isinstance(result['fields'], list):
                mapped_result = {}
                
                for field_data in result['fields']:
                    if isinstance(field_data, dict):
                        # ИСПРАВЛЕНИЕ: Gemini возвращает 'field_name' и 'field_value', а не 'name' и 'value'
                        field_name = field_data.get('field_name', field_data.get('name', '')).lower()
                        field_value = field_data.get('field_value', field_data.get('value', ''))
                        
                        # Ищем соответствующее название колонки
                        if field_name in field_mapping:
                            column_name = field_mapping[field_name]
                            mapped_result[column_name] = field_value
                        else:
                            # Если точного совпадения нет, пробуем частичное
                            for key, col in field_mapping.items():
                                if key in field_name or field_name in key:
                                    mapped_result[col] = field_value
                                    break
                            else:
                                # Если совпадения не найдено, используем оригинальное имя
                                original_name = field_data.get('field_name', field_data.get('name', field_name))
                                mapped_result[original_name] = field_value
                
                logger.debug(f"Маппинг Gemini полей завершен: {len(mapped_result)} полей")
                return mapped_result
            else:
                # Если результат уже плоский, пробуем маппить ключи
                mapped_result = {}
                for key, value in result.items():
                    key_lower = key.lower()
                    if key_lower in field_mapping:
                        mapped_result[field_mapping[key_lower]] = value
                    else:
                        # Пробуем частичное совпадение
                        mapped = False
                        for field_key, col in field_mapping.items():
                            if field_key in key_lower or key_lower in field_key:
                                mapped_result[col] = value
                                mapped = True
                                break
                        if not mapped:
                            mapped_result[key] = value
                
                return mapped_result
                
        except Exception as e:
            logger.error(f"Ошибка маппинга полей Gemini через FieldManager: {e}", exc_info=True)
            # В случае ошибки используем fallback
            return self._map_gemini_fields_fallback(result)
    
    def _map_donut_fields(self, result):
        """
        Маппит поля из результата Donut в формат для таблицы используя FieldManager.
        
        Args:
            result (dict): Результат от Donut
        
        Returns:
            dict: Результат с маппингом полей для таблицы
        """
        if not result:
            return result
            
        try:
            # Пытаемся использовать FieldManager для маппинга
            try:
                from .field_manager import field_manager
                # ИСПРАВЛЕНИЕ: Используем правильный метод get_enabled_fields()
                table_fields = field_manager.get_enabled_fields()
                
                if table_fields:
                    # ИСПРАВЛЕНИЕ: Создаем маппинг используя правильную структуру TableField
                    field_mapping = {}
                    for field in table_fields:
                        # field - это объект TableField, используем его атрибуты
                        field_name = field.display_name
                        # Для Donut часто используются английские названия
                        field_mapping[field_name.lower()] = field_name
                        
                        # Добавляем возможные английские варианты
                        if field_name == 'Поставщик':
                            field_mapping['supplier'] = field_name
                            field_mapping['vendor'] = field_name
                        elif field_name == 'ИНН поставщика':
                            field_mapping['supplier_inn'] = field_name
                            field_mapping['vendor_inn'] = field_name
                        elif field_name == 'Покупатель':
                            field_mapping['customer'] = field_name
                            field_mapping['buyer'] = field_name
                        elif 'дата' in field_name.lower():
                            field_mapping['date'] = field_name
                            field_mapping['invoice_date'] = field_name
                        elif 'номер' in field_name.lower() or '№' in field_name:
                            field_mapping['number'] = field_name
                            field_mapping['invoice_number'] = field_name
                            field_mapping['invoice_no'] = field_name
                        elif 'сумма' in field_name.lower():
                            if 'ндс' in field_name.lower():
                                field_mapping['vat_amount'] = field_name
                                field_mapping['tax_amount'] = field_name
                            else:
                                field_mapping['total'] = field_name
                                field_mapping['amount'] = field_name
                                field_mapping['total_amount'] = field_name
                    
                    # Маппим результат
                    mapped_result = {}
                    for key, value in result.items():
                        key_lower = key.lower()
                        if key_lower in field_mapping:
                            mapped_result[field_mapping[key_lower]] = value
                        else:
                            # Пробуем частичное совпадение
                            mapped = False
                            for field_key, col in field_mapping.items():
                                if field_key in key_lower or key_lower in field_key:
                                    mapped_result[col] = value
                                    mapped = True
                                    break
                            if not mapped:
                                mapped_result[key] = value
                    
                    return mapped_result
                    
            except ImportError:
                logger.warning("FieldManager недоступен для Donut маппинга")
            
            # Fallback: возвращаем результат как есть
            return result
            
        except Exception as e:
            logger.error(f"Ошибка маппинга полей Donut: {e}", exc_info=True)
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
            for field in result['fields']:
                if isinstance(field, dict):
                    # ИСПРАВЛЕНИЕ: Поддерживаем оба формата ключей
                    field_name = field.get('field_name', field.get('name', ''))
                    field_value = field.get('field_value', field.get('value', ''))
                    if field_name:
                        flat_result[field_name] = field_value
            return flat_result
        
        # Если результат уже плоский, возвращаем как есть
        return result
    
    def _check_cache(self):
        """Проверяет наличие результата в кэше"""
        if not CACHE_AVAILABLE:
            return None
            
        try:
            cache_manager = get_cache_manager()
            file_hash = cache_manager.calculate_file_hash(self.input_path)
            return cache_manager.get_cached_result(file_hash, self.model_type)
        except Exception as e:
            logger.error(f"Ошибка при проверке кэша: {e}")
            return None
    
    def _save_to_cache(self, result):
        """Сохраняет результат в кэш"""
        if not CACHE_AVAILABLE:
            return
            
        try:
            cache_manager = get_cache_manager()
            file_hash = cache_manager.calculate_file_hash(self.input_path)
            cache_manager.cache_result(file_hash, self.model_type, result, self.input_path)
        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}")


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
        self._should_stop = False  # Флаг для корректной остановки
    
    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.model_manager = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке ModelDownloadThread: {e}")

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
                logger.info("Включено ускорение загрузки с использованием hf_transfer")
            except ImportError:
                logger.info("hf_transfer не найден, используется стандартный метод загрузки")
            
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
                logger.error(f"ModelDownloadThread: download_model для {self.model_type} ({self.model_id}) вернул False")
                # Ошибка уже должна была быть залогирована и показана пользователю ниже по стеку
                # (например, в layoutlm_download_error или donut_download_error)
                
        except Exception as e:
            # Ловим ошибки, которые могли произойти внутри run()
            error_message = f"Критическая ошибка в потоке загрузки ({self.model_type}): {str(e)}"
            logger.error(error_message)
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
        self._should_stop = False  # Флаг для корректной остановки
    
    def stop(self):
        """Безопасная остановка потока."""
        self._should_stop = True
        self.quit()
        self.wait(5000)  # Ждем до 5 секунд завершения
    
    def cleanup(self):
        """Очистка ресурсов потока."""
        try:
            self.stop()
            # Очищаем ссылки
            self.path = None
            self.deleteLater()
        except Exception as e:
            logger.error(f"Ошибка при очистке TesseractCheckThread: {e}")

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
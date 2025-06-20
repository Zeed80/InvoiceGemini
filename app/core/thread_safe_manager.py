"""
Потокобезопасная обертка для ModelManager.
"""

import logging
from threading import RLock
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)


def thread_safe_method(func):
    """Декоратор для потокобезопасных методов."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)
    return wrapper


class ThreadSafeModelManager:
    """Потокобезопасная обертка для ModelManager."""
    
    def __init__(self, model_manager):
        """
        Инициализация потокобезопасного менеджера.
        
        Args:
            model_manager: Экземпляр ModelManager
        """
        self._manager = model_manager
        self._lock = RLock()  # Реентерабельная блокировка
        logger.info("ThreadSafeModelManager инициализирован")
    
    @thread_safe_method
    def get_model(self, model_type: str):
        """
        Потокобезопасное получение модели.
        
        Args:
            model_type: Тип модели
            
        Returns:
            Экземпляр процессора модели
        """
        logger.debug(f"[Thread-Safe] Получение модели типа: {model_type}")
        return self._manager.get_model(model_type)
    
    @thread_safe_method
    def download_model(self, model_type: str, model_id: Optional[str] = None, is_custom: bool = False) -> bool:
        """
        Потокобезопасная загрузка модели.
        
        Args:
            model_type: Тип модели
            model_id: ID модели
            is_custom: Флаг кастомной модели
            
        Returns:
            True если загрузка успешна
        """
        logger.info(f"[Thread-Safe] Загрузка модели: {model_type}, id={model_id}, custom={is_custom}")
        return self._manager.download_model(model_type, model_id, is_custom)
    
    @thread_safe_method
    def load_layoutlm_model(self, model_id_or_path: str, is_custom: bool = False) -> bool:
        """
        Потокобезопасная загрузка LayoutLM модели.
        
        Args:
            model_id_or_path: ID или путь к модели
            is_custom: Флаг кастомной модели
            
        Returns:
            True если загрузка успешна
        """
        logger.info(f"[Thread-Safe] Загрузка LayoutLM: {model_id_or_path}, custom={is_custom}")
        return self._manager.load_layoutlm_model(model_id_or_path, is_custom)
    
    @thread_safe_method
    def clear_layoutlm_model(self) -> bool:
        """Потокобезопасная выгрузка LayoutLM моделей."""
        logger.info("[Thread-Safe] Выгрузка LayoutLM моделей")
        return self._manager.clear_layoutlm_model()
    
    @thread_safe_method
    def get_ocr_processor(self):
        """Потокобезопасное получение OCR процессора."""
        return self._manager.get_ocr_processor()
    
    @thread_safe_method
    def get_gemini_processor(self):
        """Потокобезопасное получение Gemini процессора."""
        return self._manager.get_gemini_processor()
    
    @thread_safe_method
    def get_llm_plugin_manager(self):
        """Потокобезопасное получение менеджера LLM плагинов."""
        return self._manager.get_llm_plugin_manager()
    
    @thread_safe_method
    def get_llm_plugin(self, plugin_id: str):
        """Потокобезопасное получение LLM плагина."""
        return self._manager.get_llm_plugin(plugin_id)
    
    @thread_safe_method
    def create_llm_plugin(self, plugin_id: str):
        """Потокобезопасное создание LLM плагина."""
        return self._manager.create_llm_plugin(plugin_id)
    
    @thread_safe_method
    def get_available_llm_plugins(self) -> list:
        """Потокобезопасное получение списка доступных плагинов."""
        return self._manager.get_available_llm_plugins()
    
    def process_with_model(self, model_type: str, image_path: str, ocr_lang: Optional[str] = None, 
                          custom_prompt: Optional[str] = None) -> Optional[dict]:
        """
        Потокобезопасная обработка изображения с использованием модели.
        
        Args:
            model_type: Тип модели
            image_path: Путь к изображению
            ocr_lang: Язык OCR
            custom_prompt: Кастомный промпт
            
        Returns:
            Результат обработки или None
        """
        try:
            # Получаем модель в потокобезопасном режиме
            with self._lock:
                processor = self._manager.get_model(model_type)
                
                if not processor.is_loaded:
                    logger.warning(f"Модель {model_type} не загружена, пытаемся загрузить...")
                    if not self._manager.download_model(model_type):
                        logger.error(f"Не удалось загрузить модель {model_type}")
                        return None
            
            # Обработка может выполняться параллельно
            # так как каждый процессор должен быть потокобезопасным
            result = processor.process_image(image_path, ocr_lang, custom_prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке с моделью {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_memory_usage_info(self) -> dict:
        """Получает информацию об использовании памяти моделями."""
        with self._lock:
            info = {
                "loaded_models": []
            }
            
            # Собираем информацию о загруженных моделях
            if hasattr(self._manager, 'models'):
                for model_key, processor in self._manager.models.items():
                    if processor and hasattr(processor, 'is_loaded') and processor.is_loaded:
                        model_info = {
                            "key": model_key,
                            "type": processor.__class__.__name__,
                            "loaded": True
                        }
                        
                        # Пытаемся получить размер модели
                        if hasattr(processor, 'model') and processor.model is not None:
                            try:
                                import sys
                                model_size = sys.getsizeof(processor.model) / 1024 / 1024
                                model_info["size_mb"] = model_size
                            except (AttributeError, OSError, Exception) as e:
                                # Ошибка получения размера модели - не критично
                                pass
                        
                        info["loaded_models"].append(model_info)
            
            return info
    
    def cleanup_resources(self):
        """Очищает ресурсы всех моделей."""
        with self._lock:
            logger.info("[Thread-Safe] Начало очистки ресурсов всех моделей")
            
            try:
                # Очищаем LayoutLM модели
                self._manager.clear_layoutlm_model()
                
                # Очищаем другие модели если есть методы cleanup
                if hasattr(self._manager, 'models'):
                    for model_key, processor in list(self._manager.models.items()):
                        if processor and hasattr(processor, 'cleanup'):
                            try:
                                processor.cleanup()
                                logger.info(f"Очищены ресурсы модели: {model_key}")
                            except Exception as e:
                                logger.warning(f"Ошибка при очистке модели {model_key}: {e}")
                
                # Вызываем сборщик мусора
                import gc
                gc.collect()
                
                # Очищаем GPU память если доступна
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU память очищена")
                except (ImportError, RuntimeError, AttributeError, Exception) as e:
                    # Ошибка очистки GPU памяти - не критично
                    pass
                
                logger.info("[Thread-Safe] Очистка ресурсов завершена")
                
            except Exception as e:
                logger.error(f"Ошибка при очистке ресурсов: {e}")
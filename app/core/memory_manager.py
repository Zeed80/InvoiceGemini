"""
Менеджер памяти для контроля использования ресурсов при загрузке ML моделей.
"""

import os
import gc
import logging
import psutil
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Информация о состоянии памяти."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    gpu_available_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None


class MemoryManager:
    """Менеджер для контроля использования памяти при загрузке моделей."""
    
    # Размеры моделей в МБ (примерные)
    MODEL_SIZES = {
        'layoutlm': {
            'base': 500,
            'large': 870,
            'custom': 600  # Среднее значение для кастомных моделей
        },
        'donut': {
            'base': 680,
            'large': 1200
        },
        'gemini': 0,  # Облачная модель, не требует локальной памяти
        'llm_openai': 0,  # Облачная
        'llm_anthropic': 0,  # Облачная
        'llm_google': 0,  # Облачная
        'llm_ollama': {
            'llama3.2-vision:11b': 8000,
            'llama3.2:3b': 2000,
            'llama3.1:8b': 5000,
            'llama3.1:70b': 40000,
            'mistral:7b': 4000,
            'qwen2.5:7b': 4000
        }
    }
    
    # Минимальный запас памяти в МБ
    MIN_FREE_MEMORY = 1024  # 1GB
    
    # Коэффициент запаса (модель может использовать больше памяти при инференсе)
    SAFETY_FACTOR = 1.5
    
    def __init__(self):
        """Инициализация менеджера памяти."""
        self._lock = Lock()
        self._loaded_models: Dict[str, int] = {}  # model_id -> размер в МБ
        self._gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Проверяет доступность GPU."""
        try:
            if torch.cuda.is_available():
                logger.info(f"GPU доступен: {torch.cuda.get_device_name(0)}")
                return True
        except:
            pass
        return False
    
    def get_memory_info(self) -> MemoryInfo:
        """Получает текущую информацию о памяти."""
        # Информация о RAM
        mem = psutil.virtual_memory()
        info = MemoryInfo(
            total_mb=mem.total / 1024 / 1024,
            available_mb=mem.available / 1024 / 1024,
            used_mb=mem.used / 1024 / 1024,
            percent=mem.percent
        )
        
        # Информация о GPU если доступен
        if self._gpu_available:
            try:
                gpu_mem = torch.cuda.mem_get_info(0)
                info.gpu_available_mb = gpu_mem[0] / 1024 / 1024
                info.gpu_total_mb = gpu_mem[1] / 1024 / 1024
                if info.gpu_total_mb is not None and info.gpu_available_mb is not None:
                    info.gpu_used_mb = info.gpu_total_mb - info.gpu_available_mb
                else:
                    info.gpu_used_mb = None
            except Exception as e:
                logger.warning(f"Не удалось получить информацию о GPU памяти: {e}")
        
        return info
    
    def can_load_model(self, model_type: str, model_id: str = None) -> Tuple[bool, str]:
        """
        Проверяет, можно ли загрузить модель без риска исчерпания памяти.
        
        Args:
            model_type: Тип модели (layoutlm, donut, etc.)
            model_id: ID конкретной модели
            
        Returns:
            Tuple[bool, str]: (можно_загрузить, сообщение)
        """
        with self._lock:
            # Получаем размер модели
            model_size = self._estimate_model_size(model_type, model_id)
            
            if model_size == 0:
                # Облачная модель, не требует локальной памяти
                return True, "Облачная модель не требует локальной памяти"
            
            # Получаем информацию о памяти
            mem_info = self.get_memory_info()
            
            # Расчет требуемой памяти с учетом коэффициента безопасности
            required_memory = model_size * self.SAFETY_FACTOR
            
            # Проверяем RAM
            if mem_info.available_mb - required_memory < self.MIN_FREE_MEMORY:
                return False, (
                    f"Недостаточно оперативной памяти. "
                    f"Требуется: {required_memory:.0f}MB, "
                    f"Доступно: {mem_info.available_mb:.0f}MB"
                )
            
            # Проверяем GPU если используется
            if self._gpu_available and self._should_use_gpu(model_type):
                if mem_info.gpu_available_mb and mem_info.gpu_available_mb < required_memory:
                    # Попробуем освободить GPU память
                    self._cleanup_gpu_memory()
                    mem_info = self.get_memory_info()
                    
                    if mem_info.gpu_available_mb and mem_info.gpu_available_mb < required_memory:
                        return False, (
                            f"Недостаточно GPU памяти. "
                            f"Требуется: {required_memory:.0f}MB, "
                            f"Доступно: {mem_info.gpu_available_mb:.0f}MB"
                        )
            
            return True, f"Достаточно памяти для загрузки модели (требуется {required_memory:.0f}MB)"
    
    def _estimate_model_size(self, model_type: str, model_id: str = None) -> int:
        """Оценивает размер модели в МБ."""
        if model_type not in self.MODEL_SIZES:
            logger.warning(f"Неизвестный тип модели: {model_type}, используем размер по умолчанию")
            return 1000  # По умолчанию 1GB
        
        model_info = self.MODEL_SIZES[model_type]
        
        # Если это словарь с вариантами
        if isinstance(model_info, dict):
            if model_id:
                # Ищем конкретную модель
                for key, size in model_info.items():
                    if key in model_id.lower():
                        return size
            # Возвращаем размер базовой модели
            return model_info.get('base', 1000)
        else:
            # Если это просто число
            return int(model_info)
    
    def _should_use_gpu(self, model_type: str) -> bool:
        """Определяет, должна ли модель использовать GPU."""
        # Облачные модели не используют локальный GPU
        cloud_models = ['gemini', 'llm_openai', 'llm_anthropic', 'llm_google']
        return model_type not in cloud_models
    
    def register_loaded_model(self, model_id: str, size_mb: int):
        """Регистрирует загруженную модель."""
        with self._lock:
            self._loaded_models[model_id] = size_mb
            logger.info(f"Зарегистрирована модель {model_id} размером {size_mb}MB")
    
    def unregister_model(self, model_id: str):
        """Удаляет модель из реестра."""
        with self._lock:
            if model_id in self._loaded_models:
                size = self._loaded_models.pop(model_id)
                logger.info(f"Модель {model_id} удалена из реестра (освобождено {size}MB)")
    
    def get_loaded_models_info(self) -> Dict[str, int]:
        """Возвращает информацию о загруженных моделях."""
        with self._lock:
            return self._loaded_models.copy()
    
    def cleanup_memory(self):
        """Выполняет очистку памяти."""
        logger.info("Запуск очистки памяти...")
        
        # Сборка мусора Python
        collected = gc.collect()
        logger.info(f"Сборщик мусора собрал {collected} объектов")
        
        # Очистка GPU памяти если доступен
        if self._gpu_available:
            self._cleanup_gpu_memory()
        
        # Обновляем информацию о памяти
        mem_info = self.get_memory_info()
        log_msg = f"Память после очистки - RAM: {mem_info.available_mb:.0f}MB доступно"
        if mem_info.gpu_available_mb is not None:
            log_msg += f", GPU: {mem_info.gpu_available_mb:.0f}MB доступно"
        logger.info(log_msg)
    
    def _cleanup_gpu_memory(self):
        """Очищает GPU память."""
        if not self._gpu_available:
            return
            
        try:
            # Очищаем кэш CUDA
            torch.cuda.empty_cache()
            
            # Синхронизация
            torch.cuda.synchronize()
            
            logger.info("GPU память очищена")
        except Exception as e:
            logger.warning(f"Ошибка при очистке GPU памяти: {e}")
    
    def suggest_models_to_unload(self, required_mb: int) -> list:
        """
        Предлагает модели для выгрузки чтобы освободить память.
        
        Args:
            required_mb: Требуемый объем памяти в МБ
            
        Returns:
            Список ID моделей для выгрузки
        """
        with self._lock:
            if not self._loaded_models:
                return []
            
            # Сортируем модели по размеру (от больших к маленьким)
            sorted_models = sorted(
                self._loaded_models.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            models_to_unload = []
            freed_memory = 0
            
            for model_id, size in sorted_models:
                models_to_unload.append(model_id)
                freed_memory += size
                
                if freed_memory >= required_mb:
                    break
            
            return models_to_unload
    
    def get_memory_usage_report(self) -> str:
        """Генерирует отчет об использовании памяти."""
        mem_info = self.get_memory_info()
        loaded_models = self.get_loaded_models_info()
        
        report = [
            "=== Отчет об использовании памяти ===",
            f"RAM: {mem_info.used_mb:.0f}/{mem_info.total_mb:.0f}MB ({mem_info.percent:.1f}%)",
            f"Доступно: {mem_info.available_mb:.0f}MB"
        ]
        
        if mem_info.gpu_available_mb is not None:
            report.extend([
                f"GPU: {mem_info.gpu_used_mb:.0f}/{mem_info.gpu_total_mb:.0f}MB",
                f"GPU доступно: {mem_info.gpu_available_mb:.0f}MB"
            ])
        
        if loaded_models:
            report.append("\nЗагруженные модели:")
            total_size = 0
            for model_id, size in loaded_models.items():
                report.append(f"  - {model_id}: {size}MB")
                total_size += size
            report.append(f"Всего используется моделями: {total_size}MB")
        else:
            report.append("\nНет загруженных моделей")
        
        return "\n".join(report)


# Глобальный экземпляр менеджера памяти
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Возвращает глобальный экземпляр менеджера памяти."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
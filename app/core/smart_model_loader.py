"""
Умная система загрузки моделей с предсказательным кэшированием
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread

logger = logging.getLogger(__name__)


@dataclass
class ModelUsageStats:
    """Статистика использования модели"""
    model_id: str
    usage_count: int
    last_used: float
    avg_load_time: float
    memory_usage_mb: int
    success_rate: float
    user_preference_score: float = 0.0


@dataclass
class PredictiveLoadTask:
    """Задача предсказательной загрузки"""
    model_id: str
    priority: int
    file_types: Set[str]
    estimated_load_time: float
    confidence: float


class SmartModelLoader(QObject):
    """
    Умная система загрузки моделей с:
    - Предсказательным кэшированием
    - Приоритизацией по статистике использования
    - Адаптивным управлением памятью
    - Асинхронной загрузкой
    """
    
    # Сигналы
    model_loading_started = pyqtSignal(str)  # model_id
    model_loading_progress = pyqtSignal(str, int)  # model_id, progress
    model_loaded = pyqtSignal(str, float)  # model_id, load_time
    model_unloaded = pyqtSignal(str)  # model_id
    memory_warning = pyqtSignal(int)  # free_memory_mb
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Конфигурация
        self.max_concurrent_loads = 2
        self.memory_threshold_mb = 1024  # 1GB резерв
        self.prediction_window_files = 3  # Предсказываем для следующих 3 файлов
        
        # Статистика и аналитика
        self.usage_stats: Dict[str, ModelUsageStats] = {}
        self.load_history: List[Tuple[str, float, bool]] = []  # model_id, load_time, success
        self.current_file_queue: List[str] = []
        self.last_model_sequence: List[str] = []
        
        # Состояние
        self.loaded_models: Dict[str, object] = {}
        self.loading_tasks: Dict[str, QThread] = {}
        self.load_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_loads)
        
        # Предсказательный кэш
        self.prediction_cache: List[PredictiveLoadTask] = []
        self.prediction_timer = QTimer()
        self.prediction_timer.timeout.connect(self._update_predictions)
        self.prediction_timer.start(5000)  # Обновляем предсказания каждые 5 сек
        
        # Мониторинг ресурсов
        self.memory_monitor_timer = QTimer()
        self.memory_monitor_timer.timeout.connect(self._monitor_memory)
        self.memory_monitor_timer.start(10000)  # Каждые 10 сек
        
        logger.info("🤖 SmartModelLoader инициализирован")
    
    def analyze_file_queue(self, file_paths: List[str]) -> None:
        """Анализирует очередь файлов для предсказательной загрузки"""
        self.current_file_queue = file_paths[:self.prediction_window_files]
        
        # Анализируем типы файлов
        file_types = set()
        for file_path in self.current_file_queue:
            ext = Path(file_path).suffix.lower()
            file_types.add(ext)
        
        # Предсказываем нужные модели
        predicted_models = self._predict_needed_models(file_types)
        
        # Планируем загрузку
        self._schedule_predictive_loading(predicted_models)
    
    def _predict_needed_models(self, file_types: Set[str]) -> List[PredictiveLoadTask]:
        """Предсказывает нужные модели на основе типов файлов и статистики"""
        predictions = []
        
        # Базовые предсказания по типам файлов
        model_preferences = {
            '.pdf': ['gemini', 'layoutlm', 'donut'],
            '.png': ['gemini', 'donut', 'layoutlm'],
            '.jpg': ['gemini', 'donut', 'layoutlm'],
            '.jpeg': ['gemini', 'donut', 'layoutlm']
        }
        
        model_scores = {}
        
        # Оценка по типам файлов
        for file_type in file_types:
            if file_type in model_preferences:
                for i, model_id in enumerate(model_preferences[file_type]):
                    base_score = 100 - (i * 20)  # Убывающий приоритет
                    model_scores[model_id] = model_scores.get(model_id, 0) + base_score
        
        # Добавляем статистику использования
        for model_id, score in model_scores.items():
            if model_id in self.usage_stats:
                stats = self.usage_stats[model_id]
                # Бонусы за частое использование и успешность
                usage_bonus = min(stats.usage_count * 5, 50)
                success_bonus = stats.success_rate * 30
                preference_bonus = stats.user_preference_score * 20
                
                final_score = score + usage_bonus + success_bonus + preference_bonus
                
                predictions.append(PredictiveLoadTask(
                    model_id=model_id,
                    priority=int(final_score),
                    file_types=file_types,
                    estimated_load_time=stats.avg_load_time,
                    confidence=min(stats.usage_count / 10, 1.0)
                ))
        
        # Сортируем по приоритету
        predictions.sort(key=lambda x: x.priority, reverse=True)
        return predictions[:3]  # Топ-3 предсказания
    
    def _schedule_predictive_loading(self, predictions: List[PredictiveLoadTask]):
        """Планирует предсказательную загрузку моделей"""
        self.prediction_cache = predictions
        
        for task in predictions:
            if task.model_id not in self.loaded_models and task.model_id not in self.loading_tasks:
                # Проверяем доступную память
                if self._has_sufficient_memory(task.model_id):
                    logger.info(f"🔮 Предсказательная загрузка: {task.model_id} "
                              f"(приоритет: {task.priority}, уверенность: {task.confidence:.1f})")
                    self._start_background_loading(task.model_id)
    
    def _start_background_loading(self, model_id: str):
        """Запускает фоновую загрузку модели"""
        if model_id in self.loading_tasks:
            return
        
        # Создаем поток загрузки
        loading_thread = ModelLoadingThread(model_id, self)
        loading_thread.loading_progress.connect(
            lambda progress: self.model_loading_progress.emit(model_id, progress)
        )
        loading_thread.loading_completed.connect(self._on_model_loaded)
        loading_thread.loading_failed.connect(self._on_model_load_failed)
        
        self.loading_tasks[model_id] = loading_thread
        self.model_loading_started.emit(model_id)
        
        loading_thread.start()
    
    def _has_sufficient_memory(self, model_id: str) -> bool:
        """Проверяет достаточность памяти для загрузки модели"""
        from .memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        memory_info = memory_manager.get_memory_info()
        
        # Получаем ожидаемый размер модели
        expected_size = memory_manager.get_model_size(model_id, 'base')
        
        # Проверяем RAM
        required_memory = expected_size * 1.5 + self.memory_threshold_mb  # Запас
        has_ram = memory_info.available_mb >= required_memory
        
        # Проверяем GPU память если нужно
        has_gpu = True
        if model_id in ['layoutlm', 'donut'] and torch.cuda.is_available():
            gpu_required = expected_size * 1.2  # GPU требует меньше запаса
            has_gpu = (memory_info.gpu_available_mb or 0) >= gpu_required
        
        return has_ram and has_gpu
    
    def _on_model_loaded(self, model_id: str, model_instance: object, load_time: float):
        """Обработка успешной загрузки модели"""
        # Сохраняем модель
        self.loaded_models[model_id] = model_instance
        
        # Убираем из загружающихся
        if model_id in self.loading_tasks:
            self.loading_tasks[model_id].deleteLater()
            del self.loading_tasks[model_id]
        
        # Обновляем статистику
        self._update_usage_stats(model_id, load_time, True)
        
        # Эмитируем сигнал
        self.model_loaded.emit(model_id, load_time)
        
        logger.info(f"✅ Модель {model_id} загружена за {load_time:.1f}с")
    
    def _on_model_load_failed(self, model_id: str, error: str):
        """Обработка неудачной загрузки модели"""
        # Убираем из загружающихся
        if model_id in self.loading_tasks:
            self.loading_tasks[model_id].deleteLater()
            del self.loading_tasks[model_id]
        
        # Обновляем статистику
        self._update_usage_stats(model_id, 0, False)
        
        logger.error(f"❌ Не удалось загрузить модель {model_id}: {error}")
    
    def _update_usage_stats(self, model_id: str, load_time: float, success: bool):
        """Обновляет статистику использования модели"""
        current_time = time.time()
        
        if model_id not in self.usage_stats:
            self.usage_stats[model_id] = ModelUsageStats(
                model_id=model_id,
                usage_count=0,
                last_used=current_time,
                avg_load_time=0.0,
                memory_usage_mb=0,
                success_rate=0.0
            )
        
        stats = self.usage_stats[model_id]
        stats.usage_count += 1
        stats.last_used = current_time
        
        if success and load_time > 0:
            # Обновляем среднее время загрузки
            if stats.avg_load_time == 0:
                stats.avg_load_time = load_time
            else:
                stats.avg_load_time = (stats.avg_load_time + load_time) / 2
        
        # Обновляем rate успешности
        total_attempts = len([h for h in self.load_history if h[0] == model_id])
        successful_attempts = len([h for h in self.load_history if h[0] == model_id and h[2]])
        if total_attempts > 0:
            stats.success_rate = successful_attempts / total_attempts
        
        # Добавляем в историю
        self.load_history.append((model_id, load_time, success))
        
        # Ограничиваем размер истории
        if len(self.load_history) > 1000:
            self.load_history = self.load_history[-500:]
    
    def _update_predictions(self):
        """Периодическое обновление предсказаний"""
        if self.current_file_queue:
            # Анализируем текущую очередь файлов
            self.analyze_file_queue(self.current_file_queue)
    
    def _monitor_memory(self):
        """Мониторинг памяти и выгрузка моделей при необходимости"""
        from .memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        memory_info = memory_manager.get_memory_info()
        
        if memory_info.available_mb < self.memory_threshold_mb:
            # Критический уровень памяти - выгружаем модели
            self._emergency_unload_models()
            self.memory_warning.emit(int(memory_info.available_mb))
    
    def _emergency_unload_models(self):
        """Экстренная выгрузка моделей для освобождения памяти"""
        # Сортируем модели по времени последнего использования
        models_by_usage = []
        for model_id in self.loaded_models:
            if model_id in self.usage_stats:
                last_used = self.usage_stats[model_id].last_used
                models_by_usage.append((model_id, last_used))
        
        models_by_usage.sort(key=lambda x: x[1])  # Сначала старые
        
        # Выгружаем до 50% моделей
        models_to_unload = models_by_usage[:len(models_by_usage)//2 + 1]
        
        for model_id, _ in models_to_unload:
            self.unload_model(model_id)
            logger.info(f"🧹 Экстренная выгрузка модели: {model_id}")
    
    def get_model(self, model_id: str, blocking: bool = True) -> Optional[object]:
        """
        Получает модель (с загрузкой при необходимости)
        
        Args:
            model_id: ID модели
            blocking: Ждать ли завершения загрузки
            
        Returns:
            Экземпляр модели или None
        """
        # Если модель уже загружена
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # Если модель загружается
        if model_id in self.loading_tasks:
            if blocking:
                # Ждем завершения загрузки
                thread = self.loading_tasks[model_id]
                thread.wait()
                return self.loaded_models.get(model_id)
            else:
                return None
        
        # Запускаем загрузку
        if blocking:
            self._start_background_loading(model_id)
            if model_id in self.loading_tasks:
                thread = self.loading_tasks[model_id]
                thread.wait()
                return self.loaded_models.get(model_id)
        else:
            self._start_background_loading(model_id)
            return None
    
    def unload_model(self, model_id: str):
        """Выгружает модель из памяти"""
        if model_id in self.loaded_models:
            # Освобождаем ресурсы модели
            model = self.loaded_models[model_id]
            if hasattr(model, 'cleanup'):
                model.cleanup()
            
            del self.loaded_models[model_id]
            self.model_unloaded.emit(model_id)
            
            # Принудительная сборка мусора
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику работы системы"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'loading_models': list(self.loading_tasks.keys()),
            'usage_stats': {k: {
                'usage_count': v.usage_count,
                'avg_load_time': v.avg_load_time,
                'success_rate': v.success_rate,
                'last_used': v.last_used
            } for k, v in self.usage_stats.items()},
            'prediction_cache': [
                {
                    'model_id': task.model_id,
                    'priority': task.priority,
                    'confidence': task.confidence
                } for task in self.prediction_cache
            ],
            'total_loads': len(self.load_history),
            'memory_warnings': 0  # TODO: добавить счетчик
        }
    
    def cleanup(self):
        """Очистка ресурсов"""
        # Останавливаем таймеры
        self.prediction_timer.stop()
        self.memory_monitor_timer.stop()
        
        # Выгружаем все модели
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)
        
        # Останавливаем загрузки
        for thread in self.loading_tasks.values():
            thread.quit()
            thread.wait(5000)
        
        self.load_executor.shutdown(wait=True)
        
        logger.info("🧹 SmartModelLoader очищен")


class ModelLoadingThread(QThread):
    """Поток для загрузки модели в фоне"""
    
    loading_progress = pyqtSignal(int)  # progress 0-100
    loading_completed = pyqtSignal(str, object, float)  # model_id, model, load_time
    loading_failed = pyqtSignal(str, str)  # model_id, error
    
    def __init__(self, model_id: str, loader: SmartModelLoader):
        super().__init__()
        self.model_id = model_id
        self.loader = loader
        self._should_stop = False
    
    def run(self):
        """Выполняет загрузку модели"""
        start_time = time.time()
        
        try:
            self.loading_progress.emit(10)
            
            if self._should_stop:
                return
            
            # Создаем процессор напрямую, избегая циклических зависимостей
            model = self._create_model_processor(self.model_id)
            
            if self._should_stop:
                return
            
            self.loading_progress.emit(80)
            
            if model:
                load_time = time.time() - start_time
                self.loading_progress.emit(100)
                self.loading_completed.emit(self.model_id, model, load_time)
            else:
                self.loading_failed.emit(self.model_id, "Не удалось загрузить модель")
                
        except Exception as e:
            self.loading_failed.emit(self.model_id, str(e))
    
    def _create_model_processor(self, model_id: str):
        """Создает процессор модели напрямую"""
        from ..settings_manager import settings_manager
        from .. import config as app_config
        
        self.loading_progress.emit(30)
        
        if model_id == 'layoutlm':
            from ..processing_engine import LayoutLMProcessor, OCRProcessor
            ocr_processor = OCRProcessor()
            
            # Получаем настройки LayoutLM
            active_type = settings_manager.get_active_layoutlm_model_type()
            if active_type == 'custom':
                custom_model_name = settings_manager.get_string('Models', 'custom_layoutlm_model_name', app_config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME)
                model_path = os.path.join(app_config.TRAINED_MODELS_PATH, custom_model_name)
                processor = LayoutLMProcessor(ocr_processor, model_path, True)
            else:
                model_id_hf = settings_manager.get_string('Models', 'layoutlm_id', app_config.LAYOUTLM_MODEL_ID)
                processor = LayoutLMProcessor(ocr_processor, model_id_hf, False)
            
            self.loading_progress.emit(60)
            # Загружаем модель
            if processor.load_model():
                return processor
            else:
                return None
                
        elif model_id == 'donut':
            from ..processing_engine import DonutProcessorImpl
            donut_model_id = settings_manager.get_string('Models', 'donut_id', app_config.DONUT_MODEL_ID)
            processor = DonutProcessorImpl(donut_model_id)
            
            self.loading_progress.emit(60)
            if processor.load_model():
                return processor
            else:
                return None
                
        elif model_id == 'gemini':
            from ..gemini_processor import GeminiProcessor
            self.loading_progress.emit(60)
            processor = GeminiProcessor()
            return processor  # GeminiProcessor не требует явной загрузки
            
        elif model_id == 'trocr':
            from ..trocr_processor import TrOCRProcessor
            model_identifier = settings_manager.get_string('Models', 'trocr_model_id', 'microsoft/trocr-base-printed')
            self.loading_progress.emit(60)
            processor = TrOCRProcessor(model_name=model_identifier)
            return processor
            
        else:
            raise ValueError(f"Неизвестный тип модели: {model_id}")
            
        return None
    
    def stop(self):
        """Останавливает загрузку"""
        self._should_stop = True


# Глобальный экземпляр
_smart_loader: Optional[SmartModelLoader] = None


def get_smart_model_loader() -> SmartModelLoader:
    """Возвращает глобальный экземпляр умного загрузчика моделей"""
    global _smart_loader
    if _smart_loader is None:
        _smart_loader = SmartModelLoader()
    return _smart_loader 
"""
Система мониторинга производительности для отслеживания оптимизаций
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import json
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Метрика производительности"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str
    details: Dict[str, Any] = None


@dataclass
class SystemSnapshot:
    """Снимок состояния системы"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    active_threads: int
    open_files: int


class PerformanceMonitor(QObject):
    """
    Система мониторинга производительности:
    - Отслеживание времени операций
    - Мониторинг системных ресурсов
    - Анализ эффективности кэшей
    - Генерация отчетов
    """
    
    metric_recorded = pyqtSignal(str, float)  # name, value
    threshold_exceeded = pyqtSignal(str, float, float)  # metric, value, threshold
    
    def __init__(self, reports_dir: str = "data/performance"):
        super().__init__()
        
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Хранение метрик
        self.metrics: List[PerformanceMetric] = []
        self.system_snapshots: List[SystemSnapshot] = []
        self.operation_times: Dict[str, List[float]] = {}
        
        # Пороговые значения
        self.thresholds = {
            'startup_time': 5.0,        # секунды
            'model_load_time': 10.0,    # секунды
            'ui_response_time': 0.1,    # секунды
            'memory_usage_percent': 80, # процент
            'cache_hit_rate': 0.7       # 70%
        }
        
        # Блокировка для потокобезопасности
        self.lock = Lock()
        
        # Таймер для системного мониторинга
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self._collect_system_metrics)
        self.system_timer.start(5000)  # Каждые 5 секунд
        
        # Таймер для генерации отчетов
        self.report_timer = QTimer()
        self.report_timer.timeout.connect(self._generate_periodic_report)
        self.report_timer.start(300000)  # Каждые 5 минут
        
        logger.info("📊 PerformanceMonitor инициализирован")
    
    def record_operation_time(self, operation: str, duration: float, 
                             details: Dict[str, Any] = None):
        """Записывает время выполнения операции"""
        with self.lock:
            metric = PerformanceMetric(
                name=f"{operation}_time",
                value=duration,
                unit="seconds",
                timestamp=time.time(),
                category="operation",
                details=details or {}
            )
            
            self.metrics.append(metric)
            
            # Группируем по операциям
            if operation not in self.operation_times:
                self.operation_times[operation] = []
            
            self.operation_times[operation].append(duration)
            
            # Ограничиваем размер истории
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-500:]
            
            # Проверяем пороги
            threshold_key = f"{operation}_time"
            if threshold_key in self.thresholds:
                threshold = self.thresholds[threshold_key]
                if duration > threshold:
                    self.threshold_exceeded.emit(operation, duration, threshold)
            
            self.metric_recorded.emit(f"{operation}_time", duration)
            
            logger.debug(f"⏱️ {operation}: {duration:.3f}с")
    
    def record_cache_stats(self, cache_name: str, hits: int, misses: int, 
                          size_mb: float, details: Dict[str, Any] = None):
        """Записывает статистику кэша"""
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        with self.lock:
            # Запись hit rate
            metric_hit_rate = PerformanceMetric(
                name=f"{cache_name}_hit_rate",
                value=hit_rate,
                unit="ratio",
                timestamp=time.time(),
                category="cache",
                details={
                    'hits': hits,
                    'misses': misses,
                    'size_mb': size_mb,
                    **(details or {})
                }
            )
            
            # Запись размера
            metric_size = PerformanceMetric(
                name=f"{cache_name}_size",
                value=size_mb,
                unit="mb",
                timestamp=time.time(),
                category="cache",
                details=details or {}
            )
            
            self.metrics.extend([metric_hit_rate, metric_size])
            
            # Проверяем пороги
            if hit_rate < self.thresholds.get('cache_hit_rate', 0.7):
                self.threshold_exceeded.emit(
                    f"{cache_name}_hit_rate", hit_rate, self.thresholds['cache_hit_rate']
                )
            
            self.metric_recorded.emit(f"{cache_name}_hit_rate", hit_rate)
            self.metric_recorded.emit(f"{cache_name}_size", size_mb)
    
    def record_memory_usage(self, component: str, memory_mb: float, 
                           details: Dict[str, Any] = None):
        """Записывает использование памяти компонентом"""
        with self.lock:
            metric = PerformanceMetric(
                name=f"{component}_memory",
                value=memory_mb,
                unit="mb",
                timestamp=time.time(),
                category="memory",
                details=details or {}
            )
            
            self.metrics.append(metric)
            self.metric_recorded.emit(f"{component}_memory", memory_mb)
    
    def _collect_system_metrics(self):
        """Собирает системные метрики"""
        try:
            # CPU и память
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Информация о процессе
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # GPU память (если доступно)
            gpu_memory_used = None
            gpu_memory_total = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            except ImportError:
                pass
            
            snapshot = SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024**2,
                memory_available_mb=memory.available / 1024**2,
                gpu_memory_used_mb=gpu_memory_used,
                gpu_memory_total_mb=gpu_memory_total,
                active_threads=process.num_threads(),
                open_files=len(process.open_files())
            )
            
            with self.lock:
                self.system_snapshots.append(snapshot)
                
                # Ограничиваем историю (последние 2 часа при интервале 5 сек)
                max_snapshots = 1440  # 2 часа
                if len(self.system_snapshots) > max_snapshots:
                    self.system_snapshots = self.system_snapshots[-max_snapshots//2:]
                
                # Проверяем пороги памяти
                if memory.percent > self.thresholds.get('memory_usage_percent', 80):
                    self.threshold_exceeded.emit(
                        'system_memory', memory.percent, self.thresholds['memory_usage_percent']
                    )
            
        except Exception as e:
            logger.error(f"Ошибка сбора системных метрик: {e}")
    
    def _generate_periodic_report(self):
        """Генерирует периодический отчет"""
        report = self.generate_performance_report()
        
        # Сохраняем отчет
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"performance_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 Отчет о производительности сохранен: {report_file}")
            
            # Очищаем старые отчеты (старше 7 дней)
            self._cleanup_old_reports()
            
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")
    
    def _cleanup_old_reports(self):
        """Очищает старые отчеты"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for report_file in self.reports_dir.glob("performance_report_*.json"):
            try:
                file_date = datetime.fromtimestamp(report_file.stat().st_mtime)
                if file_date < cutoff_date:
                    report_file.unlink()
            except Exception as e:
                logger.error(f"Ошибка удаления старого отчета {report_file}: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Генерирует полный отчет о производительности"""
        with self.lock:
            current_time = time.time()
            
            # Анализ операций
            operation_stats = {}
            for operation, times in self.operation_times.items():
                if times:
                    operation_stats[operation] = {
                        'count': len(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'last_10_avg': sum(times[-10:]) / min(len(times), 10)
                    }
            
            # Анализ кэшей
            cache_stats = {}
            cache_metrics = [m for m in self.metrics if m.category == 'cache']
            
            for metric in cache_metrics:
                cache_name = metric.name.split('_')[0]
                if cache_name not in cache_stats:
                    cache_stats[cache_name] = {}
                
                if 'hit_rate' in metric.name:
                    cache_stats[cache_name]['hit_rate'] = metric.value
                elif 'size' in metric.name:
                    cache_stats[cache_name]['size_mb'] = metric.value
            
            # Системные метрики (последние 30 минут)
            recent_snapshots = [
                s for s in self.system_snapshots 
                if current_time - s.timestamp <= 1800  # 30 минут
            ]
            
            system_stats = {}
            if recent_snapshots:
                system_stats = {
                    'avg_cpu_percent': sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
                    'avg_memory_percent': sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots),
                    'peak_memory_mb': max(s.memory_used_mb for s in recent_snapshots),
                    'avg_threads': sum(s.active_threads for s in recent_snapshots) / len(recent_snapshots),
                    'current_memory_mb': recent_snapshots[-1].memory_used_mb,
                    'current_cpu_percent': recent_snapshots[-1].cpu_percent
                }
                
                if any(s.gpu_memory_used_mb for s in recent_snapshots):
                    gpu_snapshots = [s for s in recent_snapshots if s.gpu_memory_used_mb]
                    system_stats.update({
                        'avg_gpu_memory_mb': sum(s.gpu_memory_used_mb for s in gpu_snapshots) / len(gpu_snapshots),
                        'peak_gpu_memory_mb': max(s.gpu_memory_used_mb for s in gpu_snapshots),
                        'current_gpu_memory_mb': recent_snapshots[-1].gpu_memory_used_mb
                    })
            
            # Анализ превышений порогов
            threshold_violations = []
            recent_metrics = [
                m for m in self.metrics 
                if current_time - m.timestamp <= 3600  # Последний час
            ]
            
            for metric in recent_metrics:
                threshold_key = metric.name
                if threshold_key in self.thresholds:
                    if metric.value > self.thresholds[threshold_key]:
                        threshold_violations.append({
                            'metric': metric.name,
                            'value': metric.value,
                            'threshold': self.thresholds[threshold_key],
                            'timestamp': metric.timestamp,
                            'severity': 'high' if metric.value > self.thresholds[threshold_key] * 1.5 else 'medium'
                        })
            
            return {
                'generated_at': current_time,
                'period_start': current_time - 3600,  # Последний час
                'operation_performance': operation_stats,
                'cache_performance': cache_stats,
                'system_metrics': system_stats,
                'threshold_violations': threshold_violations,
                'total_metrics_collected': len(self.metrics),
                'optimization_recommendations': self._generate_recommendations(
                    operation_stats, cache_stats, system_stats, threshold_violations
                )
            }
    
    def _generate_recommendations(self, operation_stats: Dict, cache_stats: Dict, 
                                 system_stats: Dict, violations: List) -> List[str]:
        """Генерирует рекомендации по оптимизации"""
        recommendations = []
        
        # Анализ операций
        for operation, stats in operation_stats.items():
            if stats['avg_time'] > 5.0:  # Операции дольше 5 секунд
                recommendations.append(
                    f"⚠️ Операция '{operation}' выполняется медленно (среднее: {stats['avg_time']:.2f}с)"
                )
        
        # Анализ кэшей
        for cache_name, stats in cache_stats.items():
            hit_rate = stats.get('hit_rate', 0)
            if hit_rate < 0.5:  # Hit rate ниже 50%
                recommendations.append(
                    f"📊 Низкий hit rate кэша '{cache_name}': {hit_rate:.1%}. Рекомендуется увеличить размер кэша"
                )
        
        # Анализ системных ресурсов
        if system_stats.get('avg_memory_percent', 0) > 80:
            recommendations.append(
                "🧠 Высокое использование памяти. Рекомендуется оптимизация или увеличение RAM"
            )
        
        if system_stats.get('avg_cpu_percent', 0) > 80:
            recommendations.append(
                "⚡ Высокая нагрузка на CPU. Рекомендуется оптимизация алгоритмов"
            )
        
        # Анализ превышений порогов
        critical_violations = [v for v in violations if v['severity'] == 'high']
        if critical_violations:
            recommendations.append(
                f"🚨 Обнаружено {len(critical_violations)} критических превышений порогов"
            )
        
        if not recommendations:
            recommendations.append("✅ Система работает в оптимальном режиме")
        
        return recommendations
    
    def get_operation_statistics(self, operation: str) -> Dict[str, float]:
        """Возвращает статистику по конкретной операции"""
        with self.lock:
            times = self.operation_times.get(operation, [])
            
            if not times:
                return {}
            
            return {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'median_time': sorted(times)[len(times) // 2],
                'p95_time': sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
                'recent_avg': sum(times[-10:]) / min(len(times), 10)
            }
    
    def set_threshold(self, metric: str, value: float):
        """Устанавливает пороговое значение для метрики"""
        self.thresholds[metric] = value
        logger.info(f"📏 Установлен порог для {metric}: {value}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.system_timer.stop()
        self.report_timer.stop()
        
        # Генерируем финальный отчет
        final_report = self.generate_performance_report()
        
        report_file = self.reports_dir / "final_performance_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения финального отчета: {e}")
        
        logger.info("📊 PerformanceMonitor очищен")


# Глобальный экземпляр
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Возвращает глобальный экземпляр монитора производительности"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Декоратор для автоматического измерения времени операций
def monitor_performance(operation_name: str):
    """Декоратор для автоматического мониторинга времени операций"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation_time(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation_time(
                    f"{operation_name}_failed", duration, {'error': str(e)}
                )
                raise
        
        return wrapper
    return decorator 
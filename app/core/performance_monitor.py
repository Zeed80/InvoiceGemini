"""
PerformanceMonitor - Система комплексного мониторинга производительности
Версия: 4.0 - Финальная фаза мониторинга и оптимизации
"""

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from queue import Queue, Empty
import gc
import sys
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication


@dataclass
class PerformanceMetric:
    """Метрика производительности"""
    name: str
    value: float
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Отчет о производительности"""
    timestamp: datetime
    metrics: List[PerformanceMetric]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceMonitor(QObject):
    """Комплексный мониторинг производительности системы"""
    
    # Сигналы для уведомлений
    metric_updated = pyqtSignal(str, float)  # metric_name, value
    issue_detected = pyqtSignal(str)  # issue_description
    report_generated = pyqtSignal(object)  # PerformanceReport
    
    def __init__(self, app_instance=None):
        super().__init__()
        self.app_instance = app_instance
        self.logger = logging.getLogger(__name__)
        
        # Настройка мониторинга
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_queue = Queue()
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        
        # Настройка отчетов
        self.reports_dir = Path("logs/performance_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Метрики для отслеживания
        self.startup_time = None
        self.startup_stages = {}
        self.ui_response_times = []
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self.model_load_times = {}
        
        # Пороговые значения для проблем
        self.thresholds = {
            'startup_time': 4.0,  # секунды
            'ui_response_time': 0.1,  # секунды
            'memory_usage': 1024,  # МБ
            'cpu_usage': 80.0,  # процент
            'model_load_time': 10.0  # секунды
        }
        
        # Таймер для периодического мониторинга
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self._collect_system_metrics)
        
        self.logger.info("PerformanceMonitor инициализирован")
    
    def start_monitoring(self, interval: float = 1.0):
        """Запуск мониторинга"""
        if self.monitoring_active:
            self.logger.warning("Мониторинг уже активен")
            return
            
        self.monitoring_active = True
        self.monitoring_timer.start(int(interval * 1000))  # Конвертируем в миллисекунды
        
        # Запуск потока для обработки метрик
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Мониторинг запущен с интервалом {interval} сек")
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self.monitoring_timer.stop()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            
        self.logger.info("Мониторинг остановлен")
    
    def record_startup_stage(self, stage_name: str, duration: float):
        """Запись времени этапа запуска"""
        self.startup_stages[stage_name] = duration
        self.logger.info(f"Этап запуска '{stage_name}': {duration:.2f}с")
        
        # Обновление общего времени запуска
        if self.startup_time is None:
            self.startup_time = sum(self.startup_stages.values())
        else:
            self.startup_time = sum(self.startup_stages.values())
        
        # Проверка на проблемы
        if self.startup_time > self.thresholds['startup_time']:
            issue = f"Время запуска превышает порог: {self.startup_time:.2f}с > {self.thresholds['startup_time']}с"
            self.issue_detected.emit(issue)
    
    def record_ui_response_time(self, action: str, duration: float):
        """Запись времени отклика UI"""
        self.ui_response_times.append({
            'action': action,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # Ограничиваем историю
        if len(self.ui_response_times) > 100:
            self.ui_response_times = self.ui_response_times[-100:]
        
        # Проверка на проблемы
        if duration > self.thresholds['ui_response_time']:
            issue = f"Медленный отклик UI '{action}': {duration:.3f}с > {self.thresholds['ui_response_time']}с"
            self.issue_detected.emit(issue)
    
    def record_model_load_time(self, model_name: str, duration: float):
        """Запись времени загрузки модели"""
        self.model_load_times[model_name] = {
            'duration': duration,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"Модель '{model_name}' загружена за {duration:.2f}с")
        
        # Проверка на проблемы
        if duration > self.thresholds['model_load_time']:
            issue = f"Медленная загрузка модели '{model_name}': {duration:.2f}с > {self.thresholds['model_load_time']}с"
            self.issue_detected.emit(issue)
    
    def _collect_system_metrics(self):
        """Сбор системных метрик"""
        try:
            # Получение метрик системы
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / 1024 / 1024
            
            # Получение метрик процесса
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024
            
            # Сохранение метрик
            now = datetime.now()
            self.cpu_usage_history.append({'value': cpu_percent, 'timestamp': now})
            self.memory_usage_history.append({'value': process_memory, 'timestamp': now})
            
            # Ограничиваем историю
            if len(self.cpu_usage_history) > 300:  # 5 минут при интервале 1 сек
                self.cpu_usage_history = self.cpu_usage_history[-300:]
            if len(self.memory_usage_history) > 300:
                self.memory_usage_history = self.memory_usage_history[-300:]
            
            # Создание метрик для очереди
            metrics = [
                PerformanceMetric("cpu_usage", cpu_percent, now, "system"),
                PerformanceMetric("memory_usage", process_memory, now, "system"),
                PerformanceMetric("system_memory", memory_mb, now, "system")
            ]
            
            # Добавление метрик в очередь
            for metric in metrics:
                try:
                    self.metrics_queue.put_nowait(metric)
                except:
                    pass  # Игнорируем ошибки переполнения очереди
            
            # Проверка на проблемы
            if cpu_percent > self.thresholds['cpu_usage']:
                issue = f"Высокая загрузка CPU: {cpu_percent:.1f}% > {self.thresholds['cpu_usage']}%"
                self.issue_detected.emit(issue)
            
            if process_memory > self.thresholds['memory_usage']:
                issue = f"Высокое потребление памяти: {process_memory:.1f}МБ > {self.thresholds['memory_usage']}МБ"
                self.issue_detected.emit(issue)
                
        except Exception as e:
            self.logger.error(f"Ошибка при сборе метрик: {e}")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring_active:
            try:
                # Обработка метрик из очереди
                while not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        self.current_metrics[metric.name] = metric
                        self.metric_updated.emit(metric.name, metric.value)
                    except Empty:
                        break
                
                # Небольшая пауза для уменьшения нагрузки
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(1.0)
    
    def generate_report(self) -> PerformanceReport:
        """Генерация отчета о производительности"""
        now = datetime.now()
        metrics = list(self.current_metrics.values())
        
        # Добавление специальных метрик
        if self.startup_time:
            metrics.append(PerformanceMetric("startup_time", self.startup_time, now, "startup"))
        
        # Средние значения UI
        if self.ui_response_times:
            avg_ui_response = sum(item['duration'] for item in self.ui_response_times) / len(self.ui_response_times)
            metrics.append(PerformanceMetric("avg_ui_response_time", avg_ui_response, now, "ui"))
        
        # Анализ проблем
        issues = []
        recommendations = []
        
        # Проверка времени запуска
        if self.startup_time and self.startup_time > self.thresholds['startup_time']:
            issues.append(f"Медленный запуск: {self.startup_time:.2f}с")
            recommendations.append("Рассмотреть ленивую загрузку компонентов")
        
        # Проверка памяти
        if self.memory_usage_history:
            avg_memory = sum(item['value'] for item in self.memory_usage_history[-10:]) / min(10, len(self.memory_usage_history))
            if avg_memory > self.thresholds['memory_usage']:
                issues.append(f"Высокое потребление памяти: {avg_memory:.1f}МБ")
                recommendations.append("Проверить утечки памяти и оптимизировать кэширование")
        
        # Проверка UI
        if self.ui_response_times:
            slow_actions = [item for item in self.ui_response_times if item['duration'] > self.thresholds['ui_response_time']]
            if slow_actions:
                issues.append(f"Медленные действия UI: {len(slow_actions)} из {len(self.ui_response_times)}")
                recommendations.append("Оптимизировать медленные UI операции")
        
        report = PerformanceReport(
            timestamp=now,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
        
        # Сохранение отчета
        self._save_report(report)
        
        # Отправка сигнала
        self.report_generated.emit(report)
        
        return report
    
    def _save_report(self, report: PerformanceReport):
        """Сохранение отчета в файл"""
        try:
            timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = self.reports_dir / f"performance_report_{timestamp_str}.json"
            
            # Конвертация в JSON
            report_data = {
                'timestamp': report.timestamp.isoformat(),
                'metrics': [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'category': m.category,
                        'metadata': m.metadata
                    }
                    for m in report.metrics
                ],
                'issues': report.issues,
                'recommendations': report.recommendations
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Отчет сохранен: {filename}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении отчета: {e}")
    
    def get_startup_analysis(self) -> Dict[str, Any]:
        """Анализ времени запуска по этапам"""
        if not self.startup_stages:
            return {}
        
        total_time = sum(self.startup_stages.values())
        analysis = {
            'total_time': total_time,
            'stages': self.startup_stages,
            'slowest_stage': max(self.startup_stages, key=self.startup_stages.get),
            'optimization_potential': total_time - self.thresholds['startup_time'] if total_time > self.thresholds['startup_time'] else 0
        }
        
        return analysis
    
    def get_ui_performance_summary(self) -> Dict[str, Any]:
        """Сводка производительности UI"""
        if not self.ui_response_times:
            return {}
        
        durations = [item['duration'] for item in self.ui_response_times]
        
        summary = {
            'total_actions': len(self.ui_response_times),
            'avg_response_time': sum(durations) / len(durations),
            'max_response_time': max(durations),
            'min_response_time': min(durations),
            'slow_actions_count': len([d for d in durations if d > self.thresholds['ui_response_time']]),
            'actions_by_type': {}
        }
        
        # Группировка по типам действий
        for item in self.ui_response_times:
            action_type = item['action']
            if action_type not in summary['actions_by_type']:
                summary['actions_by_type'][action_type] = []
            summary['actions_by_type'][action_type].append(item['duration'])
        
        return summary
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Анализ тренда потребления памяти"""
        if not self.memory_usage_history:
            return {}
        
        recent_data = self.memory_usage_history[-60:]  # Последние 60 измерений
        values = [item['value'] for item in recent_data]
        
        trend = {
            'current_usage': values[-1] if values else 0,
            'avg_usage': sum(values) / len(values),
            'max_usage': max(values),
            'min_usage': min(values),
            'trend_direction': 'stable'
        }
        
        # Определение тренда
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            diff_percent = (avg_second - avg_first) / avg_first * 100
            
            if diff_percent > 5:
                trend['trend_direction'] = 'increasing'
            elif diff_percent < -5:
                trend['trend_direction'] = 'decreasing'
        
        return trend
    
    def optimize_performance(self):
        """Автоматическая оптимизация производительности"""
        optimizations_applied = []
        
        try:
            # Оптимизация 1: Очистка памяти
            if self.memory_usage_history:
                current_memory = self.memory_usage_history[-1]['value']
                if current_memory > self.thresholds['memory_usage'] * 0.8:
                    gc.collect()
                    optimizations_applied.append("Выполнена очистка памяти")
            
            # Оптимизация 2: Настройка Qt приложения
            if self.app_instance:
                app = self.app_instance
                if hasattr(app, 'processEvents'):
                    app.processEvents()
                    optimizations_applied.append("Обработаны события Qt")
            
            # Оптимизация 3: Системные настройки
            if sys.platform == 'win32':
                # Увеличение приоритета процесса на Windows
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetPriorityClass(
                        ctypes.windll.kernel32.GetCurrentProcess(), 
                        0x00000020  # NORMAL_PRIORITY_CLASS
                    )
                    optimizations_applied.append("Установлен нормальный приоритет процесса")
                except:
                    pass
            
            self.logger.info(f"Применены оптимизации: {optimizations_applied}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации: {e}")
        
        return optimizations_applied


# Глобальный экземпляр для использования в приложении
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Получение глобального экземпляра PerformanceMonitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_performance_monitor(app_instance=None):
    """Инициализация системы мониторинга"""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(app_instance)
    return _performance_monitor 
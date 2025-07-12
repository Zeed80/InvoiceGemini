"""
AutoReporting - Система автоматических отчетов о производительности
Версия: 4.0 - Финальная фаза мониторинга
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading
import time
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .performance_monitor import get_performance_monitor, PerformanceReport


@dataclass
class ReportSchedule:
    """Расписание генерации отчетов"""
    name: str
    interval_hours: int
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class AutoReporting(QObject):
    """Система автоматических отчетов"""
    
    # Сигналы
    report_scheduled = pyqtSignal(str)  # report_name
    report_generated = pyqtSignal(str, str)  # report_name, file_path
    alert_triggered = pyqtSignal(str, str)  # alert_type, message
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Настройка директорий
        self.reports_dir = Path("logs/auto_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.alerts_dir = Path("logs/alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Расписание отчетов
        self.schedules: Dict[str, ReportSchedule] = {
            'hourly_summary': ReportSchedule(
                name='Hourly Performance Summary',
                interval_hours=1
            ),
            'daily_report': ReportSchedule(
                name='Daily Performance Report',
                interval_hours=24
            ),
            'startup_analysis': ReportSchedule(
                name='Startup Analysis Report',
                interval_hours=6
            )
        }
        
        # Настройка таймера
        self.scheduler_timer = QTimer()
        self.scheduler_timer.timeout.connect(self._check_schedules)
        self.scheduler_timer.start(60000)  # Проверка каждую минуту
        
        # Настройка алертов
        self.alert_thresholds = {
            'high_memory': 800,  # МБ
            'slow_startup': 6,   # секунды
            'high_cpu': 75,      # процент
            'slow_ui': 0.2       # секунды
        }
        
        # История алертов
        self.alert_history: List[Dict[str, Any]] = []
        
        self.logger.info("AutoReporting система инициализирована")
    
    def start_reporting(self):
        """Запуск системы автоматических отчетов"""
        # Обновление расписания
        now = datetime.now()
        for schedule in self.schedules.values():
            if schedule.next_run is None:
                schedule.next_run = now + timedelta(hours=schedule.interval_hours)
        
        self.logger.info("Автоматические отчеты запущены")
    
    def stop_reporting(self):
        """Остановка системы отчетов"""
        self.scheduler_timer.stop()
        self.logger.info("Автоматические отчеты остановлены")
    
    def _check_schedules(self):
        """Проверка расписания отчетов"""
        now = datetime.now()
        
        for schedule_id, schedule in self.schedules.items():
            if not schedule.enabled:
                continue
                
            if schedule.next_run and now >= schedule.next_run:
                self._generate_scheduled_report(schedule_id)
                
                # Обновление расписания
                schedule.last_run = now
                schedule.next_run = now + timedelta(hours=schedule.interval_hours)
    
    def _generate_scheduled_report(self, schedule_id: str):
        """Генерация запланированного отчета"""
        try:
            schedule = self.schedules[schedule_id]
            self.report_scheduled.emit(schedule.name)
            
            if schedule_id == 'hourly_summary':
                report_path = self._generate_hourly_summary()
            elif schedule_id == 'daily_report':
                report_path = self._generate_daily_report()
            elif schedule_id == 'startup_analysis':
                report_path = self._generate_startup_analysis()
            else:
                return
            
            self.report_generated.emit(schedule.name, report_path)
            self.logger.info(f"Создан отчет: {schedule.name} -> {report_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета {schedule_id}: {e}")
    
    def _generate_hourly_summary(self) -> str:
        """Генерация почасового резюме"""
        monitor = get_performance_monitor()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сбор данных
        current_metrics = monitor.current_metrics
        ui_summary = monitor.get_ui_performance_summary()
        memory_trend = monitor.get_memory_trend()
        
        # Формирование отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'type': 'hourly_summary',
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'category': metric.category
                }
                for name, metric in current_metrics.items()
            },
            'ui_performance': ui_summary,
            'memory_trend': memory_trend,
            'issues_detected': self._detect_current_issues()
        }
        
        # Сохранение отчета
        file_path = self.reports_dir / f"hourly_summary_{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def _generate_daily_report(self) -> str:
        """Генерация ежедневного отчета"""
        monitor = get_performance_monitor()
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Сбор данных за день
        startup_analysis = monitor.get_startup_analysis()
        ui_summary = monitor.get_ui_performance_summary()
        memory_trend = monitor.get_memory_trend()
        
        # Статистика по алертам
        today_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] >= datetime.now() - timedelta(days=1)
        ]
        
        # Формирование отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'type': 'daily_report',
            'period': 'last_24_hours',
            'startup_analysis': startup_analysis,
            'ui_performance': ui_summary,
            'memory_analysis': memory_trend,
            'alerts_summary': {
                'total_alerts': len(today_alerts),
                'alerts_by_type': self._group_alerts_by_type(today_alerts),
                'most_frequent_issue': self._get_most_frequent_issue(today_alerts)
            },
            'recommendations': self._generate_recommendations(startup_analysis, ui_summary, memory_trend)
        }
        
        # Сохранение отчета
        file_path = self.reports_dir / f"daily_report_{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def _generate_startup_analysis(self) -> str:
        """Генерация анализа запуска"""
        monitor = get_performance_monitor()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        startup_analysis = monitor.get_startup_analysis()
        
        # Дополнительный анализ
        if startup_analysis:
            # Анализ узких мест
            bottlenecks = []
            for stage, duration in startup_analysis.get('stages', {}).items():
                if duration > 2.0:  # Этапы длиннее 2 секунд
                    bottlenecks.append({
                        'stage': stage,
                        'duration': duration,
                        'percentage': (duration / startup_analysis['total_time']) * 100
                    })
            
            startup_analysis['bottlenecks'] = bottlenecks
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'type': 'startup_analysis',
            'startup_data': startup_analysis,
            'optimization_suggestions': self._generate_startup_optimizations(startup_analysis)
        }
        
        # Сохранение отчета
        file_path = self.reports_dir / f"startup_analysis_{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def _detect_current_issues(self) -> List[Dict[str, Any]]:
        """Обнаружение текущих проблем"""
        issues = []
        monitor = get_performance_monitor()
        
        # Проверка памяти
        memory_trend = monitor.get_memory_trend()
        if memory_trend and memory_trend.get('current_usage', 0) > self.alert_thresholds['high_memory']:
            issues.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f"Высокое потребление памяти: {memory_trend['current_usage']:.1f}МБ",
                'timestamp': datetime.now().isoformat()
            })
        
        # Проверка CPU
        cpu_metrics = monitor.current_metrics.get('cpu_usage')
        if cpu_metrics and cpu_metrics.value > self.alert_thresholds['high_cpu']:
            issues.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"Высокая загрузка CPU: {cpu_metrics.value:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Проверка UI производительности
        ui_summary = monitor.get_ui_performance_summary()
        if ui_summary and ui_summary.get('avg_response_time', 0) > self.alert_thresholds['slow_ui']:
            issues.append({
                'type': 'slow_ui',
                'severity': 'info',
                'message': f"Медленный отклик UI: {ui_summary['avg_response_time']:.3f}с",
                'timestamp': datetime.now().isoformat()
            })
        
        return issues
    
    def _generate_recommendations(self, startup_analysis: Dict, ui_summary: Dict, memory_trend: Dict) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        # Рекомендации по запуску
        if startup_analysis and startup_analysis.get('total_time', 0) > 4:
            recommendations.append("Рассмотрите ленивую загрузку компонентов для ускорения запуска")
            
            if startup_analysis.get('slowest_stage'):
                recommendations.append(f"Оптимизируйте этап '{startup_analysis['slowest_stage']}' - самый медленный")
        
        # Рекомендации по UI
        if ui_summary and ui_summary.get('slow_actions_count', 0) > 0:
            recommendations.append("Оптимизируйте медленные UI операции")
            
            if ui_summary.get('max_response_time', 0) > 0.5:
                recommendations.append("Рассмотрите асинхронное выполнение длительных операций")
        
        # Рекомендации по памяти
        if memory_trend and memory_trend.get('trend_direction') == 'increasing':
            recommendations.append("Проверьте на утечки памяти - потребление растет")
            
        if memory_trend and memory_trend.get('current_usage', 0) > 800:
            recommendations.append("Рассмотрите оптимизацию кэширования данных")
        
        return recommendations
    
    def _generate_startup_optimizations(self, startup_analysis: Dict) -> List[str]:
        """Генерация предложений по оптимизации запуска"""
        optimizations = []
        
        if not startup_analysis:
            return optimizations
        
        # Анализ узких мест
        bottlenecks = startup_analysis.get('bottlenecks', [])
        for bottleneck in bottlenecks:
            stage = bottleneck['stage']
            duration = bottleneck['duration']
            
            if 'model' in stage.lower():
                optimizations.append(f"Ленивая загрузка моделей в '{stage}' ({duration:.1f}с)")
            elif 'ui' in stage.lower():
                optimizations.append(f"Отложенная инициализация UI в '{stage}' ({duration:.1f}с)")
            elif 'plugin' in stage.lower():
                optimizations.append(f"Асинхронная загрузка плагинов в '{stage}' ({duration:.1f}с)")
            else:
                optimizations.append(f"Оптимизация этапа '{stage}' ({duration:.1f}с)")
        
        return optimizations
    
    def _group_alerts_by_type(self, alerts: List[Dict]) -> Dict[str, int]:
        """Группировка алертов по типам"""
        groups = {}
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            groups[alert_type] = groups.get(alert_type, 0) + 1
        return groups
    
    def _get_most_frequent_issue(self, alerts: List[Dict]) -> Optional[str]:
        """Получение наиболее частой проблемы"""
        if not alerts:
            return None
        
        groups = self._group_alerts_by_type(alerts)
        if not groups:
            return None
        
        return max(groups, key=groups.get)
    
    def trigger_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Создание алерта"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert)
        
        # Ограничиваем историю
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Сохранение алерта
        self._save_alert(alert)
        
        # Отправка сигнала
        self.alert_triggered.emit(alert_type, message)
        
        self.logger.warning(f"Алерт {alert_type}: {message}")
    
    def _save_alert(self, alert: Dict):
        """Сохранение алерта в файл"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            alerts_file = self.alerts_dir / f"alerts_{timestamp}.json"
            
            # Загрузка существующих алертов
            alerts_data = []
            if alerts_file.exists():
                with open(alerts_file, 'r', encoding='utf-8') as f:
                    alerts_data = json.load(f)
            
            # Добавление нового алерта
            alerts_data.append(alert)
            
            # Сохранение
            with open(alerts_file, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении алерта: {e}")
    
    def get_recent_reports(self, hours: int = 24) -> List[str]:
        """Получение списка недавних отчетов"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_reports = []
        
        for report_file in self.reports_dir.glob("*.json"):
            try:
                if report_file.stat().st_mtime > cutoff_time.timestamp():
                    recent_reports.append(str(report_file))
            except:
                continue
        
        return sorted(recent_reports, reverse=True)
    
    def get_report_summary(self, report_path: str) -> Dict[str, Any]:
        """Получение краткого содержания отчета"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            return {
                'type': report_data.get('type', 'unknown'),
                'timestamp': report_data.get('timestamp', ''),
                'issues_count': len(report_data.get('issues_detected', [])),
                'recommendations_count': len(report_data.get('recommendations', [])),
                'file_path': report_path
            }
        except Exception as e:
            self.logger.error(f"Ошибка при чтении отчета {report_path}: {e}")
            return {}


# Глобальный экземпляр
_auto_reporting = None


def get_auto_reporting() -> AutoReporting:
    """Получение глобального экземпляра AutoReporting"""
    global _auto_reporting
    if _auto_reporting is None:
        _auto_reporting = AutoReporting()
    return _auto_reporting


def initialize_auto_reporting():
    """Инициализация системы автоматических отчетов"""
    global _auto_reporting
    _auto_reporting = AutoReporting()
    return _auto_reporting 
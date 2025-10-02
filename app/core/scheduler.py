"""
Планировщик задач для автоматической синхронизации
Поддерживает периодическую синхронизацию с Paperless и другими интеграциями
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class ScheduleInterval(Enum):
    """Интервалы для расписания"""
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


@dataclass
class ScheduledTask:
    """Запланированная задача"""
    task_id: str
    name: str
    func: Callable
    interval: ScheduleInterval
    interval_value: int
    next_run: datetime
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Планировщик задач для автоматической синхронизации"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        interval: ScheduleInterval,
        interval_value: int,
        enabled: bool = True,
        **kwargs
    ) -> bool:
        """
        Добавляет задачу в расписание
        
        Args:
            task_id: Уникальный ID задачи
            name: Название задачи
            func: Функция для выполнения
            interval: Тип интервала (minutes, hours, days, weeks)
            interval_value: Значение интервала
            enabled: Включена ли задача
            **kwargs: Дополнительные аргументы для функции
        """
        try:
            next_run = self._calculate_next_run(interval, interval_value)
            
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                func=func,
                interval=interval,
                interval_value=interval_value,
                next_run=next_run,
                enabled=enabled,
                kwargs=kwargs
            )
            
            with self._lock:
                self.tasks[task_id] = task
            
            self.logger.info(
                f"Добавлена задача '{name}' (ID: {task_id}), "
                f"интервал: {interval_value} {interval.value}, "
                f"следующий запуск: {next_run}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления задачи {task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Удаляет задачу из расписания"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks.pop(task_id)
                self.logger.info(f"Удалена задача '{task.name}' (ID: {task_id})")
                return True
            return False
    
    def enable_task(self, task_id: str) -> bool:
        """Включает задачу"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].enabled = True
                self.logger.info(f"Задача {task_id} включена")
                return True
            return False
    
    def disable_task(self, task_id: str) -> bool:
        """Отключает задачу"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].enabled = False
                self.logger.info(f"Задача {task_id} отключена")
                return True
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Получает статус задачи"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    "task_id": task.task_id,
                    "name": task.name,
                    "enabled": task.enabled,
                    "interval": f"{task.interval_value} {task.interval.value}",
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_error": task.last_error
                }
            return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Получает статусы всех задач"""
        with self._lock:
            return [
                self.get_task_status(task_id) 
                for task_id in self.tasks.keys()
            ]
    
    def start(self):
        """Запускает планировщик"""
        if self.running:
            self.logger.warning("Планировщик уже запущен")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info("Планировщик задач запущен")
    
    def stop(self):
        """Останавливает планировщик"""
        if not self.running:
            return
        
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        self.logger.info("Планировщик задач остановлен")
    
    def _run_loop(self):
        """Основной цикл планировщика"""
        while self.running:
            try:
                now = datetime.now()
                
                with self._lock:
                    tasks_to_run = [
                        task for task in self.tasks.values()
                        if task.enabled and task.next_run <= now
                    ]
                
                for task in tasks_to_run:
                    self._execute_task(task)
                
                # Проверка каждые 10 секунд
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле планировщика: {e}")
    
    def _execute_task(self, task: ScheduledTask):
        """Выполняет задачу"""
        try:
            self.logger.info(f"Выполнение задачи '{task.name}'...")
            
            # Выполнение в отдельном потоке
            thread = threading.Thread(
                target=self._task_wrapper,
                args=(task,),
                daemon=True
            )
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи '{task.name}': {e}")
            task.error_count += 1
            task.last_error = str(e)
    
    def _task_wrapper(self, task: ScheduledTask):
        """Обертка для выполнения задачи"""
        try:
            # Выполнение функции
            task.func(**task.kwargs)
            
            # Обновление статуса
            with self._lock:
                task.last_run = datetime.now()
                task.run_count += 1
                task.next_run = self._calculate_next_run(
                    task.interval,
                    task.interval_value
                )
            
            self.logger.info(
                f"Задача '{task.name}' выполнена успешно. "
                f"Следующий запуск: {task.next_run}"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка в задаче '{task.name}': {e}", exc_info=True)
            
            with self._lock:
                task.error_count += 1
                task.last_error = str(e)
                # Все равно планируем следующий запуск
                task.next_run = self._calculate_next_run(
                    task.interval,
                    task.interval_value
                )
    
    def _calculate_next_run(
        self,
        interval: ScheduleInterval,
        interval_value: int
    ) -> datetime:
        """Вычисляет время следующего запуска"""
        now = datetime.now()
        
        if interval == ScheduleInterval.MINUTES:
            return now + timedelta(minutes=interval_value)
        elif interval == ScheduleInterval.HOURS:
            return now + timedelta(hours=interval_value)
        elif interval == ScheduleInterval.DAYS:
            return now + timedelta(days=interval_value)
        elif interval == ScheduleInterval.WEEKS:
            return now + timedelta(weeks=interval_value)
        else:
            raise ValueError(f"Неизвестный интервал: {interval}")


# Глобальный экземпляр планировщика
_scheduler_instance = None


def get_scheduler() -> TaskScheduler:
    """Получает глобальный экземпляр планировщика"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TaskScheduler()
    return _scheduler_instance


# Вспомогательные функции для быстрого планирования

def schedule_every_minutes(
    task_id: str,
    name: str,
    func: Callable,
    minutes: int,
    **kwargs
) -> bool:
    """Планирует задачу каждые N минут"""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_id, name, func,
        ScheduleInterval.MINUTES, minutes,
        **kwargs
    )


def schedule_every_hours(
    task_id: str,
    name: str,
    func: Callable,
    hours: int,
    **kwargs
) -> bool:
    """Планирует задачу каждые N часов"""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_id, name, func,
        ScheduleInterval.HOURS, hours,
        **kwargs
    )


def schedule_daily(
    task_id: str,
    name: str,
    func: Callable,
    **kwargs
) -> bool:
    """Планирует ежедневную задачу"""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_id, name, func,
        ScheduleInterval.DAYS, 1,
        **kwargs
    )


def schedule_weekly(
    task_id: str,
    name: str,
    func: Callable,
    **kwargs
) -> bool:
    """Планирует еженедельную задачу"""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_id, name, func,
        ScheduleInterval.WEEKS, 1,
        **kwargs
    )


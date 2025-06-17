"""
Менеджер повторных попыток для API запросов.
"""

import time
import logging
import random
from typing import Callable, Any, Optional, Tuple, List, Type
from functools import wraps

logger = logging.getLogger(__name__)

# Опциональные импорты
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Опциональные импорты для специфичных исключений
try:
    from openai import RateLimitError as OpenAIRateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIRateLimitError = None

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicRateLimitError = None


class RetryConfig:
    """Конфигурация для повторных попыток"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        retriable_status_codes: Optional[List[int]] = None
    ):
        """
        Инициализация конфигурации повторных попыток.
        
        Args:
            max_attempts: Максимальное количество попыток
            initial_delay: Начальная задержка в секундах
            max_delay: Максимальная задержка в секундах
            exponential_base: База для экспоненциального увеличения задержки
            jitter: Добавлять ли случайную задержку
            retriable_exceptions: Исключения, при которых нужно повторить
            retriable_status_codes: HTTP коды, при которых нужно повторить
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        # Стандартные исключения для повтора
        default_exceptions = [
            ConnectionError,
            TimeoutError,
        ]
        
        # Добавляем исключения requests если доступны
        if REQUESTS_AVAILABLE:
            default_exceptions.extend([
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ])
            
        # Добавляем исключения httpx если доступны
        if HTTPX_AVAILABLE:
            default_exceptions.extend([
                httpx.TimeoutException,
                httpx.ConnectError,
            ])
        
        # Добавляем исключения rate limit от различных провайдеров
        if OPENAI_AVAILABLE and OpenAIRateLimitError:
            default_exceptions.append(OpenAIRateLimitError)
            
        if ANTHROPIC_AVAILABLE and AnthropicRateLimitError:
            default_exceptions.append(AnthropicRateLimitError)
            
        default_exceptions = tuple(default_exceptions)
        
        self.retriable_exceptions = retriable_exceptions or default_exceptions
        
        # Стандартные HTTP коды для повтора
        self.retriable_status_codes = retriable_status_codes or [
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        ]


class RetryManager:
    """Менеджер для выполнения операций с повторными попытками"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Инициализация менеджера.
        
        Args:
            config: Конфигурация для повторных попыток
        """
        self.config = config or RetryConfig()
        
    def calculate_delay(self, attempt: int) -> float:
        """
        Вычисляет задержку перед следующей попыткой.
        
        Args:
            attempt: Номер попытки (начиная с 0)
            
        Returns:
            float: Задержка в секундах
        """
        # Экспоненциальная задержка
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        # Добавляем джиттер для предотвращения "thundering herd"
        if self.config.jitter:
            delay *= (0.5 + random.random())
            
        return delay
        
    def should_retry(self, exception: Exception) -> bool:
        """
        Определяет, нужно ли повторить попытку для данного исключения.
        
        Args:
            exception: Исключение
            
        Returns:
            bool: True если нужно повторить
        """
        # Проверяем тип исключения
        if isinstance(exception, self.config.retriable_exceptions):
            return True
            
        # Проверяем HTTP статус код
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            return exception.response.status_code in self.config.retriable_status_codes
            
        return False
        
    def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> Any:
        """
        Выполняет функцию с повторными попытками.
        
        Args:
            func: Функция для выполнения
            *args: Позиционные аргументы для функции
            on_retry: Колбэк, вызываемый при повторной попытке
            **kwargs: Именованные аргументы для функции
            
        Returns:
            Any: Результат выполнения функции
            
        Raises:
            Exception: Последнее исключение если все попытки исчерпаны
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Пытаемся выполнить функцию
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Проверяем, нужно ли повторить
                if not self.should_retry(e) or attempt == self.config.max_attempts - 1:
                    logger.error(
                        f"Ошибка выполнения {func.__name__}: {str(e)}. "
                        f"Попытка {attempt + 1}/{self.config.max_attempts} не удалась."
                    )
                    raise
                    
                # Вычисляем задержку
                delay = self.calculate_delay(attempt)
                
                # Логируем попытку
                logger.warning(
                    f"Ошибка выполнения {func.__name__}: {str(e)}. "
                    f"Попытка {attempt + 1}/{self.config.max_attempts}. "
                    f"Повтор через {delay:.1f} секунд..."
                )
                
                # Вызываем колбэк если есть
                if on_retry:
                    on_retry(attempt, e)
                    
                # Ждем перед следующей попыткой
                time.sleep(delay)
                
        # Если мы здесь, все попытки исчерпаны
        raise last_exception


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    retriable_status_codes: Optional[List[int]] = None
):
    """
    Декоратор для выполнения функций с повторными попытками.
    
    Args:
        max_attempts: Максимальное количество попыток
        initial_delay: Начальная задержка в секундах
        max_delay: Максимальная задержка в секундах
        exponential_base: База для экспоненциального увеличения задержки
        jitter: Добавлять ли случайную задержку
        retriable_exceptions: Исключения, при которых нужно повторить
        retriable_status_codes: HTTP коды, при которых нужно повторить
        
    Returns:
        Callable: Декорированная функция
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retriable_exceptions=retriable_exceptions,
                retriable_status_codes=retriable_status_codes
            )
            
            manager = RetryManager(config)
            return manager.execute_with_retry(func, *args, **kwargs)
            
        return wrapper
    return decorator


# Предустановленные конфигурации для разных случаев использования

class RetryProfiles:
    """Предустановленные профили для повторных попыток"""
    
    @staticmethod
    def aggressive() -> RetryConfig:
        """Агрессивный профиль: много попыток, короткие задержки"""
        return RetryConfig(
            max_attempts=10,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=True
        )
        
    @staticmethod
    def conservative() -> RetryConfig:
        """Консервативный профиль: мало попыток, длинные задержки"""
        return RetryConfig(
            max_attempts=3,
            initial_delay=5.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=True
        )
        
    @staticmethod
    def api_default() -> RetryConfig:
        """Стандартный профиль для API запросов"""
        return RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True
        )
        
    @staticmethod
    def no_retry() -> RetryConfig:
        """Профиль без повторных попыток"""
        return RetryConfig(
            max_attempts=1,
            initial_delay=0,
            max_delay=0,
            exponential_base=1,
            jitter=False
        )


# Глобальный менеджер с настройками по умолчанию
default_retry_manager = RetryManager(RetryProfiles.api_default()) 
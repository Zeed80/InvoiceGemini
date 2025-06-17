"""
Unified logging configuration for the application.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self._context: Dict[str, Any] = {}
        
    def set_context(self, **kwargs):
        """Set context variables."""
        self._context.update(kwargs)
        
    def clear_context(self):
        """Clear context variables."""
        self._context.clear()
        
    def filter(self, record):
        """Add context to log record."""
        for key, value in self._context.items():
            setattr(record, key, value)
        return True


class PerformanceFilter(logging.Filter):
    """Add performance metrics to log records."""
    
    def filter(self, record):
        """Add performance data to log record."""
        import psutil
        
        # Add memory usage
        process = psutil.Process()
        record.memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Add CPU usage
        record.cpu_percent = process.cpu_percent()
        
        return True


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key in ['memory_mb', 'cpu_percent', 'user_id', 'session_id', 'model_type']:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
                
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    enable_rotation: bool = True,
    enable_performance: bool = False,
    enable_json: bool = False
):
    """
    Setup unified logging configuration.
    
    Args:
        log_dir: Directory for log files (default: logs/)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_rotation: Enable log file rotation
        enable_performance: Enable performance metrics in logs
        enable_json: Enable JSON formatting for structured logging
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_rotation:
        # Main log file with rotation
        main_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=10_485_760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    else:
        # Simple file handler
        main_handler = logging.FileHandler(
            log_dir / 'app.log',
            encoding='utf-8'
        )
    
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(formatter)
    root_logger.addHandler(main_handler)
    
    # Error log file
    error_handler = logging.FileHandler(
        log_dir / 'errors.log',
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Add filters
    context_filter = ContextFilter()
    root_logger.addFilter(context_filter)
    
    if enable_performance:
        perf_filter = PerformanceFilter()
        root_logger.addFilter(perf_filter)
    
    # Configure specific loggers
    configure_module_loggers()
    
    return context_filter


def configure_module_loggers():
    """Configure logging levels for specific modules."""
    # Reduce noise from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Set specific levels for app modules
    logging.getLogger('app.processing_engine').setLevel(logging.INFO)
    logging.getLogger('app.security').setLevel(logging.INFO)
    logging.getLogger('app.ui').setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(func):
    """
    Decorator to log function performance.
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"Performance: {func.__name__} completed in {elapsed:.3f}s",
                extra={'elapsed_time': elapsed, 'function': func.__name__}
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Performance: {func.__name__} failed after {elapsed:.3f}s",
                extra={'elapsed_time': elapsed, 'function': func.__name__},
                exc_info=True
            )
            raise
            
    return wrapper


def log_exceptions(logger_name: Optional[str] = None):
    """
    Decorator to log exceptions.
    
    Args:
        logger_name: Optional logger name, defaults to module name
        
    Usage:
        @log_exceptions()
        def my_function():
            pass
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    extra={'function': func.__name__},
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._filter = None
        
    def __enter__(self):
        """Add context to logger."""
        for handler in self.logger.handlers:
            for filter in handler.filters:
                if isinstance(filter, ContextFilter):
                    filter.set_context(**self.context)
                    self._filter = filter
                    break
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove context from logger."""
        if self._filter:
            self._filter.clear_context() 
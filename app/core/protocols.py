"""
Protocol definitions for type checking and interface documentation.
"""
from typing import Protocol, Dict, Any, Optional, List, Union, runtime_checkable
from pathlib import Path
from datetime import datetime

from app.core.models import ExtractionResult, ModelConfig, ProcessingStatus


@runtime_checkable
class Processor(Protocol):
    """Protocol for document processors."""
    
    def process(self, file_path: Union[str, Path], **kwargs) -> ExtractionResult:
        """Process a document and extract invoice data."""
        ...
        
    def is_available(self) -> bool:
        """Check if processor is available and ready."""
        ...
        
    def get_config(self) -> ModelConfig:
        """Get processor configuration."""
        ...
        
    def cleanup(self) -> None:
        """Cleanup resources."""
        ...


@runtime_checkable
class Plugin(Protocol):
    """Protocol for plugins."""
    
    @property
    def plugin_id(self) -> str:
        """Unique plugin identifier."""
        ...
        
    @property
    def plugin_name(self) -> str:
        """Human-readable plugin name."""
        ...
        
    @property
    def plugin_type(self) -> str:
        """Plugin type identifier."""
        ...
        
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        ...
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with configuration."""
        ...
        
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...


@runtime_checkable
class LLMPlugin(Plugin, Protocol):
    """Protocol for LLM plugins."""
    
    def load_model(self) -> bool:
        """Load the LLM model."""
        ...
        
    def extract_invoice_data(
        self, 
        image_path: Union[str, Path], 
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract invoice data from image."""
        ...
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        ...


@runtime_checkable
class SecureStorage(Protocol):
    """Protocol for secure storage implementations."""
    
    def store_secret(self, key: str, value: str) -> bool:
        """Store a secret securely."""
        ...
        
    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        ...
        
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        ...
        
    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        ...


@runtime_checkable
class CryptoProvider(Protocol):
    """Protocol for cryptographic operations."""
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        ...
        
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        ...
        
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        ...


@runtime_checkable
class MemoryMonitor(Protocol):
    """Protocol for memory monitoring."""
    
    def get_available_memory(self) -> int:
        """Get available system memory in MB."""
        ...
        
    def get_used_memory(self) -> int:
        """Get used memory in MB."""
        ...
        
    def check_memory_available(self, required_mb: int) -> bool:
        """Check if required memory is available."""
        ...
        
    def get_gpu_memory(self) -> Optional[Dict[str, int]]:
        """Get GPU memory information if available."""
        ...


@runtime_checkable
class ResourceManager(Protocol):
    """Protocol for resource management."""
    
    def acquire_resource(self, resource_id: str) -> Any:
        """Acquire a resource."""
        ...
        
    def release_resource(self, resource_id: str) -> bool:
        """Release a resource."""
        ...
        
    def cleanup_all(self) -> None:
        """Cleanup all resources."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Protocol for logging."""
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ...
        
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...
        
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ...
        
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...
        
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event emitters."""
    
    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event."""
        ...
        
    def on(self, event: str, handler: callable) -> None:
        """Register event handler."""
        ...
        
    def off(self, event: str, handler: callable) -> None:
        """Unregister event handler."""
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...
        
    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        ...
        
    def save(self) -> bool:
        """Save configuration."""
        ...
        
    def reload(self) -> bool:
        """Reload configuration."""
        ...


@runtime_checkable
class ExportFormatter(Protocol):
    """Protocol for export formatters."""
    
    def format(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        template: Optional[str] = None
    ) -> bytes:
        """Format data for export."""
        ...
        
    def get_mime_type(self) -> str:
        """Get MIME type for the format."""
        ...
        
    def get_file_extension(self) -> str:
        """Get file extension for the format."""
        ...


@runtime_checkable
class ValidationRule(Protocol):
    """Protocol for validation rules."""
    
    def validate(self, value: Any) -> bool:
        """Validate a value."""
        ...
        
    def get_error_message(self) -> str:
        """Get validation error message."""
        ...


@runtime_checkable
class TaskQueue(Protocol):
    """Protocol for task queues."""
    
    def enqueue(self, task: Dict[str, Any]) -> str:
        """Enqueue a task and return task ID."""
        ...
        
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Dequeue next task."""
        ...
        
    def get_status(self, task_id: str) -> Optional[ProcessingStatus]:
        """Get task status."""
        ...
        
    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        ...
        
    def get_queue_size(self) -> int:
        """Get number of tasks in queue."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for caching."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with optional TTL in seconds."""
        ...
        
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        ...
        
    def clear(self) -> bool:
        """Clear all cached values."""
        ...
        
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        ... 
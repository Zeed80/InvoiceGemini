"""
Плагинная система для InvoiceGemini
Поддержка локальных LLM моделей
"""

__version__ = "1.0.0"
__author__ = "InvoiceGemini Team"

# Импорты основных компонентов плагинной системы
try:
    from .plugin_manager import PluginManager
    from .base_llm_plugin import BaseLLMPlugin
except ImportError:
    # Плагинная система еще не полностью реализована
    PluginManager = None
    BaseLLMPlugin = None

__all__ = ['PluginManager', 'BaseLLMPlugin'] 
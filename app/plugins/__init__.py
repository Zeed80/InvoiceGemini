"""
Плагинная система для InvoiceGemini
Поддержка локальных LLM моделей
"""

__version__ = "1.0.0"
__author__ = "InvoiceGemini Team"

# Импорты основных компонентов плагинной системы
try:
    from .unified_plugin_manager import UnifiedPluginManager, get_unified_plugin_manager
    # Алиас для обратной совместимости
    PluginManager = get_unified_plugin_manager
    from .base_llm_plugin import BaseLLMPlugin
except ImportError:
    # Плагинная система еще не полностью реализована
    PluginManager = None
    BaseLLMPlugin = None

__all__ = ['PluginManager', 'UnifiedPluginManager', 'get_unified_plugin_manager', 'BaseLLMPlugin'] 
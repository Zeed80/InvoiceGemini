"""
Встроенные LLM плагины для InvoiceGemini
"""

# Список доступных встроенных плагинов
BUILTIN_PLUGINS = [
    # Специализированные плагины для основных провайдеров
    'gemini_plugin',
    'openai_plugin',
    'anthropic_plugin',
    'universal_llm_plugin',
    # Старые плагины (для совместимости)
    'llama_plugin',
    'mistral_plugin', 
    'codellama_plugin'
]

__all__ = BUILTIN_PLUGINS

# Импорты специализированных плагинов
try:
    from .gemini_plugin import GeminiPlugin
except ImportError:
    GeminiPlugin = None

try:
    from .openai_plugin import OpenAIPlugin
except ImportError:
    OpenAIPlugin = None

try:
    from .anthropic_plugin import AnthropicPlugin
except ImportError:
    AnthropicPlugin = None

try:
    from .universal_llm_plugin import UniversalLLMPlugin
except ImportError:
    UniversalLLMPlugin = None

# Импорты старых плагинов (для совместимости)
try:
    from .llama_plugin import LlamaPlugin
except ImportError:
    LlamaPlugin = None

try:
    from .mistral_plugin import MistralPlugin
except ImportError:
    MistralPlugin = None

try:
    from .codellama_plugin import CodeLlamaPlugin
except ImportError:
    CodeLlamaPlugin = None 
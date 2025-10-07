"""
Утилиты для работы с Ollama - централизованные методы без дублирования.
"""
import requests
from typing import List, Tuple, Optional


class OllamaUtils:
    """Утилитарный класс для работы с Ollama без дублирования кода"""
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 5
    
    @staticmethod
    def check_availability(base_url: str = None, timeout: int = None) -> bool:
        """
        Быстрая проверка доступности Ollama сервера.
        
        Args:
            base_url: URL сервера Ollama (по умолчанию http://localhost:11434)
            timeout: Таймаут запроса в секундах (по умолчанию 5)
            
        Returns:
            bool: True если сервер доступен
        """
        base_url = base_url or OllamaUtils.DEFAULT_BASE_URL
        timeout = timeout or OllamaUtils.DEFAULT_TIMEOUT
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except (requests.RequestException, requests.ConnectionError, requests.Timeout):
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_models(base_url: str = None, timeout: int = None) -> List[str]:
        """
        Получает список доступных моделей из Ollama.
        
        Args:
            base_url: URL сервера Ollama
            timeout: Таймаут запроса в секундах
            
        Returns:
            List[str]: Список названий моделей
        """
        base_url = base_url or OllamaUtils.DEFAULT_BASE_URL
        timeout = timeout or OllamaUtils.DEFAULT_TIMEOUT
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []
    
    @staticmethod
    def check_status(base_url: str = None, timeout: int = None) -> Tuple[bool, str]:
        """
        Проверяет статус Ollama с кодом состояния.
        
        Args:
            base_url: URL сервера Ollama
            timeout: Таймаут запроса в секундах
            
        Returns:
            Tuple[bool, str]: (доступен, код_статуса)
            Коды статуса:
            - "OK": Работает корректно
            - "CFG": Нет моделей (требуется настройка)
            - "ERR": Ошибка подключения
            - "TMO": Превышено время ожидания
        """
        base_url = base_url or OllamaUtils.DEFAULT_BASE_URL
        timeout = timeout or OllamaUtils.DEFAULT_TIMEOUT
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                if available_models:
                    return True, "OK"
                else:
                    return False, "CFG"  # Нет моделей
            else:
                return False, "ERR"
                
        except requests.exceptions.ConnectionError:
            return False, "ERR"
        except requests.exceptions.Timeout:
            return False, "TMO"
        except Exception:
            return False, "ERR"
    
    @staticmethod
    def is_model_available(model_name: str, base_url: str = None, timeout: int = None) -> bool:
        """
        Проверяет, доступна ли конкретная модель в Ollama.
        
        Args:
            model_name: Название модели для проверки
            base_url: URL сервера Ollama
            timeout: Таймаут запроса в секундах
            
        Returns:
            bool: True если модель найдена
        """
        models = OllamaUtils.get_models(base_url, timeout)
        return model_name in models
    
    @staticmethod
    def get_server_version(base_url: str = None, timeout: int = None) -> Optional[str]:
        """
        Получает версию сервера Ollama.
        
        Args:
            base_url: URL сервера Ollama
            timeout: Таймаут запроса в секундах
            
        Returns:
            Optional[str]: Версия сервера или None
        """
        base_url = base_url or OllamaUtils.DEFAULT_BASE_URL
        timeout = timeout or OllamaUtils.DEFAULT_TIMEOUT
        
        try:
            response = requests.get(f"{base_url}/api/version", timeout=timeout)
            if response.status_code == 200:
                version_data = response.json()
                return version_data.get("version", "unknown")
            return None
        except Exception:
            return None
    
    @staticmethod
    def is_vision_model(model_name: str) -> bool:
        """
        Проверяет, поддерживает ли модель обработку изображений (vision).
        
        Args:
            model_name: Название модели для проверки
            
        Returns:
            bool: True если модель поддерживает vision
        """
        # Импортируем список из диагностики
        try:
            from .ollama_diagnostic import OllamaDiagnostic
            
            model_lower = model_name.lower()
            
            # Проверка по списку известных vision моделей
            for vision_model in OllamaDiagnostic.VISION_MODELS:
                if vision_model.lower() in model_lower:
                    return True
            
            # Проверка по маркерам в названии
            vision_markers = [
                "vision",     # llama3.2-vision
                "vl",         # qwen2.5vl
                "llava",      # llava модели
                "gemma3",     # gemma3 - мультимодальные
                "cogvlm",     # cogvlm
                "minicpm-v",  # minicpm-v
                "moondream",  # moondream
                "bakllava"    # bakllava
            ]
            
            return any(marker in model_lower for marker in vision_markers)
            
        except ImportError:
            # Если не удалось импортировать, используем упрощенную проверку
            return any(marker in model_name.lower() for marker in 
                      ["vision", "vl", "llava", "gemma3", "cogvlm"])


# Удобные функции для быстрого доступа
def check_ollama_availability(base_url: str = None, timeout: int = None) -> bool:
    """Проверяет доступность Ollama сервера"""
    return OllamaUtils.check_availability(base_url, timeout)


def get_ollama_models(base_url: str = None, timeout: int = None) -> List[str]:
    """Получает список моделей из Ollama"""
    return OllamaUtils.get_models(base_url, timeout)


def check_ollama_status(base_url: str = None, timeout: int = None) -> Tuple[bool, str]:
    """Проверяет статус Ollama с кодом состояния"""
    return OllamaUtils.check_status(base_url, timeout)


def is_vision_model(model_name: str) -> bool:
    """Проверяет, поддерживает ли модель vision"""
    return OllamaUtils.is_vision_model(model_name)


"""
Утилита расширенной диагностики для Ollama.
Предоставляет детальные проверки доступности и состояния Ollama сервера и моделей.
"""
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OllamaModelInfo:
    """Информация о модели Ollama"""
    name: str
    size: int
    digest: str
    modified_at: str
    details: Dict
    
    def get_size_mb(self) -> float:
        """Размер модели в MB"""
        return self.size / (1024 * 1024)
    
    def get_size_gb(self) -> float:
        """Размер модели в GB"""
        return self.size / (1024 * 1024 * 1024)


@dataclass
class OllamaDiagnosticResult:
    """Результат диагностики Ollama"""
    server_available: bool
    server_version: Optional[str]
    models_available: List[OllamaModelInfo]
    vision_models: List[str]
    recommended_models: List[str]
    base_url: str
    error_message: Optional[str]
    timestamp: str


class OllamaDiagnostic:
    """Класс для комплексной диагностики Ollama"""
    
    # Модели с поддержкой vision (обновлено 03.10.2025)
    VISION_MODELS = [
        # Llama Vision модели
        "llama3.2-vision:11b",
        "llama3.2-vision:90b",
        "llama3.2-vision",
        
        # LLaVA модели
        "llava:7b", 
        "llava:13b",
        "llava:34b",
        "bakllava:7b",
        
        # Gemma 3 - мультимодальные модели с visual поддержкой
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
        "gemma3",
        
        # Qwen Vision-Language модели
        "qwen2.5vl:3b",
        "qwen2.5vl:7b",
        "qwen2.5vl:14b",
        "qwen2.5vl",
        "qwen2-vl",
        
        # Другие visual модели
        "cogvlm",
        "minicpm-v",
        "moondream"
    ]
    
    # Рекомендуемые модели для обработки счетов (обновлено 03.10.2025)
    RECOMMENDED_INVOICE_MODELS = [
        # Vision модели (лучший выбор для счетов с изображениями)
        "llama3.2-vision:11b",  # Лучший выбор - полная vision поддержка
        "qwen2.5vl:7b",          # Быстрая vision модель
        "gemma3:12b",            # Мультимодальная модель от Google
        "gemma3:4b",             # Компактная мультимодальная
        
        # Text-only модели (для OCR-обработанных счетов)
        "llama3.1:8b",           # Хороший баланс
        "qwen2.5:7b",            # Быстрая модель
        "mistral:7b"             # Альтернатива
    ]
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Инициализация диагностики.
        
        Args:
            base_url: URL Ollama сервера
        """
        self.base_url = base_url.rstrip('/')
        
    def run_full_diagnostic(self, timeout: int = 10) -> OllamaDiagnosticResult:
        """
        Выполняет полную диагностику Ollama.
        
        Args:
            timeout: Таймаут для запросов в секундах
            
        Returns:
            OllamaDiagnosticResult: Результаты диагностики
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Проверяем доступность сервера
            server_available, version, error = self._check_server_availability(timeout)
            
            if not server_available:
                return OllamaDiagnosticResult(
                    server_available=False,
                    server_version=None,
                    models_available=[],
                    vision_models=[],
                    recommended_models=[],
                    base_url=self.base_url,
                    error_message=error,
                    timestamp=timestamp
                )
            
            # Получаем список моделей
            models = self._get_available_models(timeout)
            
            # Определяем модели с vision
            vision_models = [m.name for m in models if any(vm in m.name for vm in self.VISION_MODELS)]
            
            # Определяем рекомендуемые модели
            recommended = [m.name for m in models if m.name in self.RECOMMENDED_INVOICE_MODELS]
            
            return OllamaDiagnosticResult(
                server_available=True,
                server_version=version,
                models_available=models,
                vision_models=vision_models,
                recommended_models=recommended,
                base_url=self.base_url,
                error_message=None,
                timestamp=timestamp
            )
            
        except Exception as e:
            return OllamaDiagnosticResult(
                server_available=False,
                server_version=None,
                models_available=[],
                vision_models=[],
                recommended_models=[],
                base_url=self.base_url,
                error_message=f"Ошибка диагностики: {str(e)}",
                timestamp=timestamp
            )
    
    def _check_server_availability(self, timeout: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Проверяет доступность Ollama сервера.
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (доступен, версия, сообщение об ошибке)
        """
        try:
            # Проверяем основной endpoint
            response = requests.get(f"{self.base_url}/api/version", timeout=timeout)
            
            if response.status_code == 200:
                version_data = response.json()
                version = version_data.get("version", "unknown")
                return True, version, None
            else:
                return False, None, f"Сервер вернул код {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, None, f"Не удается подключиться к {self.base_url}. Убедитесь, что Ollama запущен (ollama serve)"
        except requests.exceptions.Timeout:
            return False, None, f"Превышено время ожидания ({timeout}с)"
        except Exception as e:
            return False, None, f"Ошибка: {str(e)}"
    
    def _get_available_models(self, timeout: int) -> List[OllamaModelInfo]:
        """
        Получает список доступных моделей.
        
        Returns:
            List[OllamaModelInfo]: Список моделей
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model_info = OllamaModelInfo(
                    name=model_data.get("name", ""),
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=model_data.get("modified_at", ""),
                    details=model_data.get("details", {})
                )
                models.append(model_info)
            
            return models
            
        except Exception as e:
            print(f"❌ Ошибка получения списка моделей: {e}")
            return []
    
    def test_model_response(self, model_name: str, timeout: int = 15) -> Tuple[bool, Optional[str]]:
        """
        Тестирует способность модели генерировать ответы.
        
        Args:
            model_name: Название модели
            timeout: Таймаут запроса
            
        Returns:
            Tuple[bool, Optional[str]]: (успешно, сообщение)
        """
        try:
            test_data = {
                "model": model_name,
                "prompt": "Hello! Respond with: OK",
                "stream": False,
                "options": {"num_predict": 5}
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                if response_text:
                    return True, f"✅ Модель отвечает корректно: '{response_text}'"
                else:
                    return False, "❌ Модель вернула пустой ответ"
            else:
                return False, f"❌ Ошибка: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, f"❌ Превышено время ожидания ({timeout}с)"
        except Exception as e:
            return False, f"❌ Ошибка: {str(e)}"
    
    def get_model_recommendations(self, models: List[OllamaModelInfo]) -> Dict[str, str]:
        """
        Анализирует модели и дает рекомендации.
        
        Args:
            models: Список доступных моделей
            
        Returns:
            Dict[str, str]: Рекомендации по моделям
        """
        recommendations = {}
        model_names = [m.name for m in models]
        
        # Рекомендация для обработки счетов с vision
        vision_available = [m for m in model_names if any(vm in m for vm in self.VISION_MODELS)]
        if vision_available:
            recommendations['best_vision'] = vision_available[0]
        else:
            recommendations['install_vision'] = "llama3.2-vision:11b"
        
        # Рекомендация для быстрой обработки
        fast_models = [m for m in model_names if any(x in m for x in ["7b", "3b"])]
        if fast_models:
            recommendations['fastest'] = fast_models[0]
        
        # Рекомендация для лучшего качества
        quality_models = [m for m in model_names if any(x in m for x in ["70b", "34b", "13b"])]
        if quality_models:
            recommendations['best_quality'] = quality_models[0]
        
        return recommendations
    
    def format_diagnostic_report(self, result: OllamaDiagnosticResult) -> str:
        """
        Форматирует результаты диагностики в читаемый отчет.
        
        Args:
            result: Результаты диагностики
            
        Returns:
            str: Отформатированный отчет
        """
        lines = []
        lines.append("="*60)
        lines.append("📋 ОТЧЕТ ДИАГНОСТИКИ OLLAMA")
        lines.append("="*60)
        lines.append(f"⏰ Время: {result.timestamp}")
        lines.append(f"🌐 URL: {result.base_url}")
        lines.append("")
        
        if not result.server_available:
            lines.append("❌ СЕРВЕР НЕДОСТУПЕН")
            lines.append(f"   Ошибка: {result.error_message}")
            lines.append("")
            lines.append("💡 Рекомендации:")
            lines.append("   1. Установите Ollama: https://ollama.com/download")
            lines.append("   2. Запустите сервер: ollama serve")
            lines.append("   3. Скачайте модель: ollama pull llama3.2-vision:11b")
        else:
            lines.append("✅ СЕРВЕР ДОСТУПЕН")
            if result.server_version:
                lines.append(f"   Версия: {result.server_version}")
            lines.append("")
            
            lines.append(f"📦 УСТАНОВЛЕННЫЕ МОДЕЛИ: {len(result.models_available)}")
            if result.models_available:
                for model in result.models_available:
                    size_str = f"{model.get_size_gb():.2f} GB" if model.get_size_gb() >= 1 else f"{model.get_size_mb():.0f} MB"
                    vision_mark = "👁️" if model.name in result.vision_models else "📝"
                    rec_mark = "⭐" if model.name in result.recommended_models else ""
                    lines.append(f"   {vision_mark} {model.name} ({size_str}) {rec_mark}")
            else:
                lines.append("   ⚠️ Модели не установлены")
            
            lines.append("")
            
            if result.vision_models:
                lines.append(f"👁️ МОДЕЛИ С VISION: {len(result.vision_models)}")
                for vm in result.vision_models:
                    lines.append(f"   ✅ {vm}")
            else:
                lines.append("⚠️ МОДЕЛИ С VISION НЕ НАЙДЕНЫ")
                lines.append("   Рекомендуем: ollama pull llama3.2-vision:11b")
            
            lines.append("")
            
            if result.recommended_models:
                lines.append(f"⭐ РЕКОМЕНДУЕМЫЕ ДЛЯ СЧЕТОВ: {len(result.recommended_models)}")
                for rm in result.recommended_models:
                    lines.append(f"   ✅ {rm}")
            else:
                lines.append("💡 РЕКОМЕНДУЕМ УСТАНОВИТЬ:")
                for rm in self.RECOMMENDED_INVOICE_MODELS[:3]:
                    lines.append(f"   ollama pull {rm}")
        
        lines.append("="*60)
        return "\n".join(lines)


def quick_diagnostic(base_url: str = "http://localhost:11434") -> str:
    """
    Быстрая диагностика Ollama с выводом отчета.
    
    Args:
        base_url: URL Ollama сервера
        
    Returns:
        str: Отчет о диагностике
    """
    diagnostic = OllamaDiagnostic(base_url)
    result = diagnostic.run_full_diagnostic()
    return diagnostic.format_diagnostic_report(result)


# Пример использования
if __name__ == "__main__":
    print(quick_diagnostic())


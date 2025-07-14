"""
LoRA Configuration Manager - централизованное управление конфигурациями LoRA
Содержит предустановленные профили для всех типов моделей
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from peft import LoraConfig, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    # Заглушки для аннотаций типов
    LoraConfig = Any
    TaskType = Any


@dataclass
class LoRAProfile:
    """Профиль LoRA с метаданными"""
    name: str
    description: str
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: Any
    memory_usage: str  # "low", "medium", "high"
    recommended_for: List[str]


class LoRAConfigManager:
    """Менеджер LoRA конфигураций с предустановленными профилями"""
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
    
    def _initialize_profiles(self) -> Dict[str, LoRAProfile]:
        """Инициализирует предустановленные профили LoRA"""
        if not LORA_AVAILABLE:
            return {}
        
        profiles = {}
        
        # Donut профили
        profiles["donut_standard"] = LoRAProfile(
            name="donut_standard",
            description="Стандартная конфигурация для Donut моделей",
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            memory_usage="medium",
            recommended_for=["general_use", "production"]
        )
        
        profiles["donut_lightweight"] = LoRAProfile(
            name="donut_lightweight",
            description="Облегченная конфигурация для экономии памяти",
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            memory_usage="low",
            recommended_for=["memory_limited", "testing"]
        )
        
        profiles["donut_high_precision"] = LoRAProfile(
            name="donut_high_precision",
            description="Высокоточная конфигурация для продакшена",
            r=32,
            lora_alpha=64,
            target_modules=["query", "value", "key", "dense", "intermediate"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            memory_usage="high",
            recommended_for=["high_accuracy", "production"]
        )
        
        # TrOCR профили
        profiles["trocr_safe"] = LoRAProfile(
            name="trocr_safe",
            description="Безопасная конфигурация для TrOCR",
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            memory_usage="low",
            recommended_for=["stable_training", "testing"]
        )
        
        profiles["trocr_extended"] = LoRAProfile(
            name="trocr_extended",
            description="Расширенная конфигурация для лучшей производительности",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            memory_usage="medium",
            recommended_for=["performance", "production"]
        )
        
        # LayoutLM профили
        profiles["layoutlm_standard"] = LoRAProfile(
            name="layoutlm_standard",
            description="Стандартная конфигурация для LayoutLM",
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
            memory_usage="medium",
            recommended_for=["document_ai", "token_classification"]
        )
        
        # Llama профили
        profiles["llama_standard"] = LoRAProfile(
            name="llama_standard",
            description="Стандартная LoRA конфигурация для Llama",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            memory_usage="medium",
            recommended_for=["text_generation", "fine_tuning"]
        )
        
        profiles["llama_qlora"] = LoRAProfile(
            name="llama_qlora",
            description="QLoRA конфигурация с 4-bit квантизацией",
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            memory_usage="low",
            recommended_for=["memory_limited", "4bit_quantization"]
        )
        
        # CodeLlama профили
        profiles["codellama_coding"] = LoRAProfile(
            name="codellama_coding",
            description="Специализированная конфигурация для генерации кода",
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            memory_usage="high",
            recommended_for=["code_generation", "instruction_following"]
        )
        
        # Mistral профили
        profiles["mistral_efficient"] = LoRAProfile(
            name="mistral_efficient",
            description="Эффективная конфигурация для Mistral",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            memory_usage="medium",
            recommended_for=["efficient_training", "general_use"]
        )
        
        return profiles
    
    def get_profile(self, profile_name: str) -> Optional[LoRAProfile]:
        """Получает профиль по имени"""
        return self.profiles.get(profile_name)
    
    def get_lora_config(self, profile_name: str, custom_params: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """
        Создает LoraConfig из профиля
        
        Args:
            profile_name: Имя профиля
            custom_params: Кастомные параметры для переопределения
            
        Returns:
            LoraConfig объект или None если PEFT недоступен
        """
        if not LORA_AVAILABLE:
            print("⚠️ PEFT не установлен. LoRA конфигурации недоступны. Установите: pip install peft")
            return None
        
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Профиль '{profile_name}' не найден")
        
        config_params = {
            "r": profile.r,
            "lora_alpha": profile.lora_alpha,
            "target_modules": profile.target_modules,
            "lora_dropout": profile.lora_dropout,
            "bias": profile.bias,
            "task_type": profile.task_type
        }
        
        # Применяем кастомные параметры
        if custom_params:
            config_params.update(custom_params)
        
        return LoraConfig(**config_params)
    
    def list_profiles(self, filter_by_memory: Optional[str] = None, 
                     filter_by_recommended: Optional[str] = None) -> List[LoRAProfile]:
        """
        Список доступных профилей с фильтрацией
        
        Args:
            filter_by_memory: Фильтр по использованию памяти ("low", "medium", "high")
            filter_by_recommended: Фильтр по рекомендации
            
        Returns:
            Список профилей
        """
        profiles = list(self.profiles.values())
        
        if filter_by_memory:
            profiles = [p for p in profiles if p.memory_usage == filter_by_memory]
        
        if filter_by_recommended:
            profiles = [p for p in profiles if filter_by_recommended in p.recommended_for]
        
        return profiles
    
    def create_custom_profile(self, name: str, description: str, **kwargs) -> LoRAProfile:
        """Создает кастомный профиль"""
        if not LORA_AVAILABLE:
            raise RuntimeError("PEFT не установлен")
        
        # Значения по умолчанию
        defaults = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.FEATURE_EXTRACTION,
            "memory_usage": "medium",
            "recommended_for": ["custom"]
        }
        
        defaults.update(kwargs)
        
        profile = LoRAProfile(name=name, description=description, **defaults)
        self.profiles[name] = profile
        
        return profile
    
    def export_profile(self, profile_name: str, file_path: str):
        """Экспортирует профиль в JSON файл"""
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Профиль '{profile_name}' не найден")
        
        data = {
            "name": profile.name,
            "description": profile.description,
            "r": profile.r,
            "lora_alpha": profile.lora_alpha,
            "target_modules": profile.target_modules,
            "lora_dropout": profile.lora_dropout,
            "bias": profile.bias,
            "task_type": str(profile.task_type),
            "memory_usage": profile.memory_usage,
            "recommended_for": profile.recommended_for
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_profile(self, file_path: str) -> LoRAProfile:
        """Импортирует профиль из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Конвертируем task_type обратно
        task_type_map = {
            "TaskType.FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
            "TaskType.TOKEN_CLS": TaskType.TOKEN_CLS,
            "TaskType.CAUSAL_LM": TaskType.CAUSAL_LM,
        }
        
        data["task_type"] = task_type_map.get(data["task_type"], TaskType.FEATURE_EXTRACTION)
        
        profile = LoRAProfile(**data)
        self.profiles[profile.name] = profile
        
        return profile


# Глобальный экземпляр менеджера
lora_config_manager = LoRAConfigManager()


# Удобные функции для быстрого доступа
def get_donut_config(profile: str = "standard", custom_params: Optional[Dict[str, Any]] = None) -> LoraConfig:
    """Получает LoRA конфигурацию для Donut модели"""
    if not LORA_AVAILABLE:
        print("⚠️ PEFT не установлен. Возвращаю None для Donut конфигурации.")
        return None
    return lora_config_manager.get_lora_config(f"donut_{profile}", custom_params)


def get_trocr_config(profile: str = "safe", custom_params: Optional[Dict[str, Any]] = None) -> LoraConfig:
    """Получает LoRA конфигурацию для TrOCR модели"""
    if not LORA_AVAILABLE:
        print("⚠️ PEFT не установлен. Возвращаю None для TrOCR конфигурации.")
        return None
    return lora_config_manager.get_lora_config(f"trocr_{profile}", custom_params)


def get_llm_config(model_type: str, profile: str = "standard", custom_params: Optional[Dict[str, Any]] = None) -> LoraConfig:
    """
    Получает LoRA конфигурацию для языковых моделей
    
    Args:
        model_type: Тип модели ("llama", "codellama", "mistral")
        profile: Профиль конфигурации
        custom_params: Кастомные параметры
        
    Returns:
        LoraConfig объект или None если PEFT недоступен
    """
    if not LORA_AVAILABLE:
        print(f"⚠️ PEFT не установлен. Возвращаю None для {model_type} конфигурации.")
        return None
    profile_name = f"{model_type}_{profile}"
    return lora_config_manager.get_lora_config(profile_name, custom_params) 
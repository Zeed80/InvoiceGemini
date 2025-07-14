"""
Унифицированные конфигурации LoRA для всех типов моделей в InvoiceGemini
Централизованное место для настройки LoRA параметров
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

try:
    from peft import TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Fallback TaskType enum для совместимости
    class TaskType:
        SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
        TOKEN_CLS = "TOKEN_CLS"
        CAUSAL_LM = "CAUSAL_LM"


@dataclass
class LoRAProfile:
    """Профиль LoRA конфигурации для определенного типа задач"""
    name: str
    description: str
    task_type: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    bias: str
    modules_to_save: Optional[list] = None
    memory_usage: str = "medium"  # low, medium, high
    recommended_for: list = None


class LoRAConfigManager:
    """Менеджер LoRA конфигураций с предустановленными профилями"""
    
    # Предустановленные профили LoRA
    PROFILES = {
        # Профили для Document AI моделей
        "donut_standard": LoRAProfile(
            name="Donut Standard",
            description="Стандартная LoRA конфигурация для Donut моделей",
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
            memory_usage="medium",
            recommended_for=["document_parsing", "invoice_extraction"]
        ),
        
        "donut_lightweight": LoRAProfile(
            name="Donut Lightweight", 
            description="Облегченная LoRA конфигурация для Donut (экономия памяти)",
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            memory_usage="low",
            recommended_for=["small_datasets", "limited_memory"]
        ),
        
        "donut_high_precision": LoRAProfile(
            name="Donut High Precision",
            description="Высокоточная LoRA конфигурация для критичных задач",
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.15,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="lora_only",
            modules_to_save=["embed_tokens"],
            memory_usage="high",
            recommended_for=["high_accuracy_requirements", "production"]
        ),
        
        # Профили для TrOCR
        "trocr_safe": LoRAProfile(
            name="TrOCR Safe",
            description="Безопасная LoRA конфигурация для TrOCR (минимальные слои)",
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "decoder.model.decoder.layers.0.self_attn.q_proj",
                "decoder.model.decoder.layers.0.self_attn.v_proj"
            ],
            bias="none",
            memory_usage="low",
            recommended_for=["trocr_models", "stability_focused"]
        ),
        
        "trocr_extended": LoRAProfile(
            name="TrOCR Extended",
            description="Расширенная LoRA конфигурация для TrOCR (больше слоев)",
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "decoder.model.decoder.layers.0.self_attn.q_proj",
                "decoder.model.decoder.layers.0.self_attn.v_proj",
                "decoder.model.decoder.layers.1.self_attn.q_proj",
                "decoder.model.decoder.layers.1.self_attn.v_proj",
                "decoder.model.decoder.layers.0.self_attn.out_proj",
                "decoder.model.decoder.layers.1.self_attn.out_proj"
            ],
            bias="none",
            memory_usage="medium",
            recommended_for=["better_performance", "sufficient_memory"]
        ),
        
        # Профили для LayoutLM
        "layoutlm_standard": LoRAProfile(
            name="LayoutLM Standard",
            description="Стандартная LoRA конфигурация для LayoutLM моделей",
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
            modules_to_save=["classifier"],
            memory_usage="medium",
            recommended_for=["token_classification", "layout_understanding"]
        ),
        
        # Профили для LLM моделей
        "llama_standard": LoRAProfile(
            name="Llama Standard",
            description="Стандартная LoRA конфигурация для Llama моделей",
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            memory_usage="medium",
            recommended_for=["general_llm", "chat", "instruction_following"]
        ),
        
        "llama_qlora": LoRAProfile(
            name="Llama QLoRA",
            description="QLoRA конфигурация для Llama с 4-bit квантизацией",
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            memory_usage="low",  # За счет квантизации
            recommended_for=["limited_memory", "consumer_gpu", "4bit_quantization"]
        ),
        
        "codellama_coding": LoRAProfile(
            name="CodeLlama Coding",
            description="Специализированная LoRA для Code Llama (программирование)",
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            memory_usage="high",
            recommended_for=["code_generation", "code_completion", "programming"]
        ),
        
        "mistral_efficient": LoRAProfile(
            name="Mistral Efficient",
            description="Эффективная LoRA конфигурация для Mistral моделей",
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            memory_usage="medium",
            recommended_for=["mistral_models", "balanced_performance"]
        )
    }
    
    @classmethod
    def get_profile(cls, profile_name: str) -> Optional[LoRAProfile]:
        """Получает профиль LoRA по имени"""
        return cls.PROFILES.get(profile_name)
    
    @classmethod
    def list_profiles(cls, model_type: Optional[str] = None) -> Dict[str, LoRAProfile]:
        """Возвращает список доступных профилей"""
        if model_type is None:
            return cls.PROFILES.copy()
        
        # Фильтруем профили по типу модели
        filtered = {}
        for name, profile in cls.PROFILES.items():
            if model_type.lower() in name.lower():
                filtered[name] = profile
        
        return filtered
    
    @classmethod
    def get_config_dict(cls, profile_name: str, **overrides) -> Dict[str, Any]:
        """
        Возвращает LoRA конфигурацию в виде словаря для PEFT
        
        Args:
            profile_name: Имя профиля
            **overrides: Параметры для переопределения
        """
        profile = cls.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Профиль '{profile_name}' не найден")
        
        config = {
            "task_type": profile.task_type,
            "r": profile.r,
            "lora_alpha": profile.lora_alpha,
            "lora_dropout": profile.lora_dropout,
            "target_modules": profile.target_modules.copy(),
            "bias": profile.bias,
        }
        
        if profile.modules_to_save:
            config["modules_to_save"] = profile.modules_to_save.copy()
        
        # Применяем переопределения
        config.update(overrides)
        
        return config
    
    @classmethod
    def create_custom_profile(cls, 
                            name: str,
                            base_profile: str,
                            **modifications) -> LoRAProfile:
        """
        Создает кастомный профиль на основе существующего
        
        Args:
            name: Имя нового профиля
            base_profile: Базовый профиль для модификации
            **modifications: Изменения
        """
        base = cls.get_profile(base_profile)
        if not base:
            raise ValueError(f"Базовый профиль '{base_profile}' не найден")
        
        # Копируем базовый профиль
        profile_dict = {
            "name": name,
            "description": modifications.get("description", f"Custom profile based on {base_profile}"),
            "task_type": base.task_type,
            "r": base.r,
            "lora_alpha": base.lora_alpha,
            "lora_dropout": base.lora_dropout,
            "target_modules": base.target_modules.copy(),
            "bias": base.bias,
            "modules_to_save": base.modules_to_save.copy() if base.modules_to_save else None,
            "memory_usage": base.memory_usage,
            "recommended_for": base.recommended_for.copy() if base.recommended_for else []
        }
        
        # Применяем модификации
        for key, value in modifications.items():
            if key in profile_dict:
                profile_dict[key] = value
        
        return LoRAProfile(**profile_dict)
    
    @classmethod
    def get_recommended_profiles(cls, 
                               memory_constraint: str = "medium",
                               task_type: Optional[str] = None) -> List[str]:
        """
        Возвращает рекомендуемые профили на основе ограничений
        
        Args:
            memory_constraint: "low", "medium", "high"
            task_type: Тип задачи для фильтрации
        """
        recommended = []
        
        for name, profile in cls.PROFILES.items():
            # Фильтр по памяти
            if profile.memory_usage != memory_constraint:
                continue
            
            # Фильтр по типу задачи
            if task_type and task_type not in profile.recommended_for:
                continue
            
            recommended.append(name)
        
        return recommended
    
    @classmethod
    def export_profiles(cls, file_path: str):
        """Экспортирует профили в JSON файл"""
        import json
        
        export_data = {}
        for name, profile in cls.PROFILES.items():
            export_data[name] = {
                "name": profile.name,
                "description": profile.description,
                "task_type": profile.task_type,
                "r": profile.r,
                "lora_alpha": profile.lora_alpha,
                "lora_dropout": profile.lora_dropout,
                "target_modules": profile.target_modules,
                "bias": profile.bias,
                "modules_to_save": profile.modules_to_save,
                "memory_usage": profile.memory_usage,
                "recommended_for": profile.recommended_for
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    @classmethod 
    def import_profiles(cls, file_path: str):
        """Импортирует профили из JSON файла"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for name, profile_data in data.items():
            cls.PROFILES[name] = LoRAProfile(**profile_data)


# Convenience функции для быстрого доступа
def get_donut_config(precision: str = "standard", **overrides) -> Dict[str, Any]:
    """Быстрый доступ к Donut LoRA конфигурации"""
    profile_map = {
        "standard": "donut_standard",
        "lightweight": "donut_lightweight", 
        "high": "donut_high_precision"
    }
    profile_name = profile_map.get(precision, "donut_standard")
    return LoRAConfigManager.get_config_dict(profile_name, **overrides)


def get_trocr_config(safety: str = "safe", **overrides) -> Dict[str, Any]:
    """Быстрый доступ к TrOCR LoRA конфигурации"""
    profile_map = {
        "safe": "trocr_safe",
        "extended": "trocr_extended"
    }
    profile_name = profile_map.get(safety, "trocr_safe")
    return LoRAConfigManager.get_config_dict(profile_name, **overrides)


def get_llm_config(model_type: str = "llama", mode: str = "standard", **overrides) -> Dict[str, Any]:
    """Быстрый доступ к LLM LoRA конфигурации"""
    profile_map = {
        ("llama", "standard"): "llama_standard",
        ("llama", "qlora"): "llama_qlora",
        ("codellama", "coding"): "codellama_coding",
        ("mistral", "efficient"): "mistral_efficient"
    }
    
    profile_name = profile_map.get((model_type, mode))
    if not profile_name:
        # Fallback к стандартной Llama конфигурации
        profile_name = "llama_standard"
    
    return LoRAConfigManager.get_config_dict(profile_name, **overrides) 
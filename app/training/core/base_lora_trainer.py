"""
Базовый класс для LoRA тренеров с унифицированными конфигурациями
Устраняет дублирование кода между DonutTrainer, TrOCRTrainer и другими тренерами
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from enum import Enum
import logging

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    # Заглушки для аннотаций типов
    LoraConfig = Any
    TaskType = Any

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


class ModelType(Enum):
    """Типы моделей для LoRA оптимизации"""
    DONUT = "donut"
    TROCR = "trocr" 
    LAYOUTLM = "layoutlm"
    LLAMA = "llama"
    CODELLAMA = "codellama"
    MISTRAL = "mistral"


class BaseLoraTrainer(ABC):
    """
    Базовый класс для всех LoRA тренеров в системе InvoiceGemini
    
    Предоставляет:
    - Унифицированные LoRA конфигурации для разных типов моделей
    - Общие методы применения LoRA оптимизации
    - Стандартизированное логирование и обработку ошибок
    - Поддержку 8-bit оптимизаторов
    """
    
    def __init__(self, model_type: ModelType, logger: Optional[logging.Logger] = None):
        self.model_type = model_type
        self.logger = logger or logging.getLogger(__name__)
        self.lora_applied = False
        self.memory_optimizations = []
        
        if not LORA_AVAILABLE:
            self.logger.warning("PEFT не установлен. LoRA оптимизации недоступны.")
        if not BITSANDBYTES_AVAILABLE:
            self.logger.warning("BitsAndBytes не установлен. 8-bit оптимизации недоступны.")
    
    def get_lora_config(self, custom_config: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """
        Получает LoRA конфигурацию для текущего типа модели
        
        Args:
            custom_config: Кастомные параметры конфигурации
            
        Returns:
            LoraConfig: Конфигурация LoRA для модели
        """
        if not LORA_AVAILABLE:
            raise RuntimeError("PEFT не установлен. Установите: pip install peft")
        
        # Базовые конфигурации для разных типов моделей
        configs = {
            ModelType.DONUT: {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["query", "value", "key", "dense"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION
            },
            ModelType.TROCR: {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION
            },
            ModelType.LAYOUTLM: {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["query", "value", "key", "dense"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.TOKEN_CLS
            },
            ModelType.LLAMA: {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            },
            ModelType.CODELLAMA: {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            },
            ModelType.MISTRAL: {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
        }
        
        base_config = configs.get(self.model_type, configs[ModelType.DONUT])
        
        # Применяем кастомные параметры если есть
        if custom_config:
            base_config.update(custom_config)
        
        return LoraConfig(**base_config)
    
    def apply_lora_optimization(self, model, training_args: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Применяет LoRA оптимизацию к модели
        
        Args:
            model: Модель для оптимизации
            training_args: Аргументы обучения
            
        Returns:
            Tuple[model, success]: Оптимизированная модель и флаг успеха
        """
        if not LORA_AVAILABLE:
            self.logger.warning("PEFT недоступен. Пропускаем LoRA оптимизацию.")
            return model, False
        
        try:
            # Получаем кастомную конфигурацию из аргументов обучения
            lora_config_params = training_args.get('lora_config', {})
            lora_config = self.get_lora_config(lora_config_params)
            
            # Подготавливаем модель для kbit training если используется квантизация
            if training_args.get('use_8bit', False) or training_args.get('use_4bit', False):
                model = prepare_model_for_kbit_training(model)
            
            # Применяем LoRA
            model = get_peft_model(model, lora_config)
            
            # Применяем специфичные для модели оптимизации
            model = self._apply_model_specific_optimizations(model, training_args)
            
            self.lora_applied = True
            self.memory_optimizations.append("LoRA")
            
            self.logger.info(f"✅ LoRA применен успешно для {self.model_type.value}")
            self.logger.info(f"LoRA параметры: r={lora_config.r}, alpha={lora_config.lora_alpha}")
            
            return model, True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка применения LoRA: {e}")
            return model, False
    
    def apply_memory_optimizations(self, model, training_args: Dict[str, Any]) -> Any:
        """
        Применяет дополнительные оптимизации памяти
        
        Args:
            model: Модель для оптимизации
            training_args: Аргументы обучения
            
        Returns:
            Оптимизированная модель
        """
        optimizations = []
        
        # Gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                optimizations.append("Gradient Checkpointing")
            except Exception as e:
                self.logger.warning(f"Не удалось включить gradient checkpointing: {e}")
        
        # Mixed precision
        if training_args.get('fp16', False):
            optimizations.append("FP16")
        elif training_args.get('bf16', False):
            optimizations.append("BF16")
        
        self.memory_optimizations.extend(optimizations)
        
        if optimizations:
            self.logger.info(f"✅ Применены оптимизации памяти: {', '.join(optimizations)}")
        
        return model
    
    def get_8bit_optimizer(self, parameters, lr: float = 5e-5):
        """
        Создает 8-bit оптимизатор AdamW для экономии памяти
        
        Args:
            parameters: Параметры модели
            lr: Learning rate
            
        Returns:
            8-bit оптимизатор или None если недоступен
        """
        if not BITSANDBYTES_AVAILABLE:
            self.logger.warning("BitsAndBytes недоступен. Используется стандартный оптимизатор.")
            return None
        
        try:
            optimizer = bnb.optim.AdamW8bit(parameters, lr=lr)
            self.memory_optimizations.append("8-bit Optimizer")
            self.logger.info("✅ Используется 8-bit AdamW оптимизатор")
            return optimizer
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания 8-bit оптимизатора: {e}")
            return None
    
    @abstractmethod
    def _apply_model_specific_optimizations(self, model, training_args: Dict[str, Any]) -> Any:
        """
        Применяет специфичные для типа модели оптимизации
        Должен быть реализован в каждом дочернем классе
        
        Args:
            model: Модель для оптимизации
            training_args: Аргументы обучения
            
        Returns:
            Оптимизированная модель
        """
        pass
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку использования памяти и примененных оптимизаций
        
        Returns:
            Словарь с информацией об оптимизациях
        """
        return {
            "model_type": self.model_type.value,
            "lora_applied": self.lora_applied,
            "optimizations": self.memory_optimizations,
            "estimated_memory_saving": "80-95%" if self.lora_applied else "0%"
        } 
"""
Менеджер для управления LLM плагинами
"""
import os
import importlib
import importlib.util
import json
from typing import Dict, List, Type, Optional, Any
from .base_llm_plugin import BaseLLMPlugin

class PluginManager:
    """
    Менеджер для управления LLM плагинами.
    Обеспечивает загрузку, создание и управление экземплярами плагинов.
    """
    
    def __init__(self, plugins_dir: str = None):
        """
        Инициализация менеджера плагинов.
        
        Args:
            plugins_dir: Директория с пользовательскими плагинами
        """
        self.builtin_plugins_dir = os.path.join(os.path.dirname(__file__), "models")
        self.user_plugins_dir = plugins_dir or os.path.join(os.getcwd(), "plugins", "user")
        
        # Реестры плагинов
        self.plugin_classes: Dict[str, Type[BaseLLMPlugin]] = {}
        self.plugin_instances: Dict[str, BaseLLMPlugin] = {}
        self.plugin_configs: Dict[str, Dict] = {}
        
        # Создаем директорию для пользовательских плагинов
        os.makedirs(self.user_plugins_dir, exist_ok=True)
        
        print(f"[WRENCH] Инициализация PluginManager...")
        print(f"[FOLDER] Встроенные плагины: {self.builtin_plugins_dir}")
        print(f"[FOLDER] Пользовательские плагины: {self.user_plugins_dir}")
        
        # Загружаем все доступные плагины
        self._load_all_plugins()
    
    def _load_all_plugins(self):
        """Загружает все доступные плагины."""
        print("[REFRESH] Загрузка плагинов...")
        
        # Загружаем встроенные плагины
        self._load_builtin_plugins()
        
        # Загружаем пользовательские плагины
        self._load_user_plugins()
        
        print(f"[OK] Загружено плагинов: {len(self.plugin_classes)}")
        if self.plugin_classes:
            for plugin_id in self.plugin_classes:
                print(f"   - {plugin_id}")
        else:
            print("[WARN] Плагины не найдены")
    
    def _load_builtin_plugins(self):
        """Загружает встроенные плагины."""
        builtin_plugins = [
            ("llama_plugin", "LlamaPlugin"),
            ("mistral_plugin", "MistralPlugin"),
            ("codellama_plugin", "CodeLlamaPlugin"),
        ]
        
        for module_name, class_name in builtin_plugins:
            try:
                # Пытаемся импортировать встроенный плагин
                module_path = f"app.plugins.models.{module_name}"
                module = importlib.import_module(module_path)
                plugin_class = getattr(module, class_name)
                
                if issubclass(plugin_class, BaseLLMPlugin):
                    plugin_id = class_name.lower().replace("plugin", "")
                    self.plugin_classes[plugin_id] = plugin_class
                    print(f"[OK] Загружен встроенный плагин: {class_name}")
                
            except ImportError as e:
                print(f"[WARN] Встроенный плагин {class_name} не найден: {e}")
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки встроенного плагина {class_name}: {e}")
    
    def _load_user_plugins(self):
        """Загружает пользовательские плагины."""
        if not os.path.exists(self.user_plugins_dir):
            print(f"[FOLDER] Директория пользовательских плагинов не существует: {self.user_plugins_dir}")
            return
        
        for filename in os.listdir(self.user_plugins_dir):
            if filename.endswith('_plugin.py') and not filename.startswith('__'):
                self._load_plugin_file(filename, self.user_plugins_dir)
    
    def _load_plugin_file(self, filename: str, plugins_dir: str):
        """
        Загружает плагин из файла.
        
        Args:
            filename: Имя файла плагина
            plugins_dir: Директория с плагинами
        """
        try:
            module_name = filename[:-3]  # убираем .py
            file_path = os.path.join(plugins_dir, filename)
            
            # Загружаем модуль из файла
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                print(f"[WARN] Не удалось создать spec для {filename}")
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Ищем классы, наследующие от BaseLLMPlugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseLLMPlugin) and 
                    attr != BaseLLMPlugin):
                    
                    plugin_id = attr_name.lower().replace("plugin", "")
                    self.plugin_classes[plugin_id] = attr
                    print(f"[OK] Загружен пользовательский плагин: {attr_name}")
                    
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки плагина {filename}: {e}")
    
    def get_available_plugins(self) -> List[str]:
        """
        Возвращает список ID доступных плагинов.
        
        Returns:
            list: Список ID плагинов
        """
        return list(self.plugin_classes.keys())
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о плагине.
        
        Args:
            plugin_id: ID плагина
            
        Returns:
            dict: Информация о плагине или None
        """
        if plugin_id not in self.plugin_classes:
            return None
        
        plugin_class = self.plugin_classes[plugin_id]
        return {
            "id": plugin_id,
            "name": plugin_class.__name__,
            "module": plugin_class.__module__,
            "doc": plugin_class.__doc__ or "Описание отсутствует",
            "is_loaded": plugin_id in self.plugin_instances
        }
    
    def create_plugin_instance(self, plugin_id: str, **kwargs) -> Optional[BaseLLMPlugin]:
        """
        Создает экземпляр плагина.
        
        Args:
            plugin_id: ID плагина
            **kwargs: Параметры для инициализации плагина
            
        Returns:
            BaseLLMPlugin: Экземпляр плагина или None
        """
        if plugin_id not in self.plugin_classes:
            print(f"[ERROR] Плагин {plugin_id} не найден")
            return None
        
        try:
            # Создаем экземпляр только если его еще нет
            if plugin_id not in self.plugin_instances:
                plugin_class = self.plugin_classes[plugin_id]
                
                # Параметры по умолчанию для разных типов плагинов
                default_params = self._get_default_params(plugin_id)
                default_params.update(kwargs)
                
                self.plugin_instances[plugin_id] = plugin_class(**default_params)
                print(f"[OK] Создан экземпляр плагина: {plugin_id}")
            
            return self.plugin_instances[plugin_id]
            
        except Exception as e:
            print(f"[ERROR] Ошибка создания экземпляра плагина {plugin_id}: {e}")
            return None
    
    def _get_default_params(self, plugin_id: str) -> Dict[str, Any]:
        """Возвращает параметры по умолчанию для плагина."""
        defaults = {
            "llama": {
                "model_name": "llama-7b-chat",
                "model_path": "meta-llama/Llama-2-7b-chat-hf"
            },
            "mistral": {
                "model_name": "mistral-7b-instruct",
                "model_path": "mistralai/Mistral-7B-Instruct-v0.2"
            },
            "codellama": {
                "model_name": "codellama-7b-instruct",
                "model_path": "codellama/CodeLlama-7b-Instruct-hf"
            }
        }
        
        return defaults.get(plugin_id, {"model_name": plugin_id})
    
    def get_plugin_instance(self, plugin_id: str) -> Optional[BaseLLMPlugin]:
        """
        Возвращает существующий экземпляр плагина.
        
        Args:
            plugin_id: ID плагина
            
        Returns:
            BaseLLMPlugin: Экземпляр плагина или None
        """
        return self.plugin_instances.get(plugin_id)
    
    def remove_plugin_instance(self, plugin_id: str) -> bool:
        """
        Удаляет экземпляр плагина из памяти.
        
        Args:
            plugin_id: ID плагина
            
        Returns:
            bool: True если удаление успешно
        """
        try:
            if plugin_id in self.plugin_instances:
                # Вызываем cleanup если есть
                instance = self.plugin_instances[plugin_id]
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                
                del self.plugin_instances[plugin_id]
                print(f"[OK] Удален экземпляр плагина: {plugin_id}")
                return True
            return False
            
        except Exception as e:
            print(f"[ERROR] Ошибка удаления экземпляра плагина {plugin_id}: {e}")
            return False
    
    def reload_plugins(self):
        """Перезагружает все плагины."""
        print("🔄 Перезагрузка плагинов...")
        
        # Очищаем существующие экземпляры
        for plugin_id in list(self.plugin_instances.keys()):
            self.remove_plugin_instance(plugin_id)
        
        # Очищаем классы
        self.plugin_classes.clear()
        
        # Загружаем заново
        self._load_all_plugins()
    
    def install_plugin_from_file(self, file_path: str) -> bool:
        """
        Устанавливает плагин из файла.
        
        Args:
            file_path: Путь к файлу плагина
            
        Returns:
            bool: True если установка успешна
        """
        try:
            import shutil
            filename = os.path.basename(file_path)
            
            # Проверяем расширение файла
            if not filename.endswith('_plugin.py'):
                print("[ERROR] Файл плагина должен заканчиваться на '_plugin.py'")
                return False
            
            try:
                # Копируем файл в директорию пользовательских плагинов
                destination = os.path.join(self.user_plugins_dir, filename)
                shutil.copy2(file_path, destination)
                
                # Перезагружаем плагины
                self.reload_plugins()
                
                print(f"[OK] Плагин {filename} установлен успешно")
                return True
                
            except Exception as e:
                print(f"[ERROR] Ошибка установки плагина: {e}")
                return False
            
        except Exception as e:
            print(f"[ERROR] Ошибка установки плагина: {e}")
            return False
    
    def create_plugin_template(self, plugin_name: str) -> str:
        """
        Создает шаблон плагина для пользователя.
        
        Args:
            plugin_name: Название плагина
            
        Returns:
            str: Путь к созданному файлу шаблона
        """
        template_content = f'''"""
Пользовательский плагин {plugin_name} для InvoiceGemini
"""
from typing import Dict, Any, Optional
from app.plugins.base_llm_plugin import BaseLLMPlugin

class {plugin_name.title()}Plugin(BaseLLMPlugin):
    """
    Пользовательский плагин для работы с моделью {plugin_name}.
    """
    
    def __init__(self, model_name: str = "{plugin_name}", model_path: Optional[str] = None, **kwargs):
        super().__init__(model_name, model_path, **kwargs)
        self.model_family = "{plugin_name.lower()}"
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Загружает модель {plugin_name}.
        
        Args:
            model_path: Путь к модели
            
        Returns:
            bool: True если загрузка успешна
        """
        try:
            # TODO: Реализуйте загрузку вашей модели
            # Пример для HuggingFace моделей:
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            path = model_path or self.model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            print(f"✅ Модель {{self.model_name}} загружена успешно")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {{self.model_name}}: {{e}}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, image_context: str = "") -> str:
        """
        Генерирует ответ модели.
        
        Args:
            prompt: Промпт для модели
            image_context: Контекст из изображения
            
        Returns:
            str: Ответ модели
        """
        if not self.is_loaded:
            return "Модель не загружена"
        
        try:
            # Объединяем промпт с контекстом изображения
            full_prompt = f"{{prompt}}\\n\\nТекст с изображения:\\n{{image_context}}"
            
            # Токенизируем входные данные
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                    do_sample=self.generation_config["do_sample"],
                    top_p=self.generation_config["top_p"],
                    repetition_penalty=self.generation_config["repetition_penalty"],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем исходный промпт из ответа
            response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Ошибка генерации: {{e}}"
    
    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """
        Основной метод обработки изображения.
        
        Args:
            image_path: Путь к изображению
            ocr_lang: Язык OCR
            custom_prompt: Пользовательский промпт
            
        Returns:
            dict: Извлеченные данные
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        # Извлекаем текст из изображения с помощью OCR
        image_context = self.extract_text_from_image(image_path, ocr_lang or "rus+eng")
        
        # Создаем промпт
        prompt = self.create_invoice_prompt(custom_prompt)
        
        # Генерируем ответ
        response = self.generate_response(prompt, image_context)
        
        # Парсим и возвращаем результат
        result = self.parse_llm_response(response)
        result["note_gemini"] = f"Обработано {{self.model_name}} ({{self.model_family}})"
        
        return result
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Конфигурация для обучения модели.
        
        Returns:
            dict: Конфигурация обучения
        """
        return {{
            "model_type": "{plugin_name.lower()}",
            "supports_lora": True,
            "supports_qlora": True,
            "default_lora_rank": 16,
            "default_lora_alpha": 32,
            "max_sequence_length": 2048,
            "training_args": {{
                "learning_rate": 2e-4,
                "batch_size": 4,
                "num_epochs": 3,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 10
            }}
        }}
'''
        
        filename = f"{plugin_name.lower()}_plugin.py"
        filepath = os.path.join(self.user_plugins_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"[OK] Создан шаблон плагина: {filepath}")
        return filepath
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по плагинам."""
        return {
            "total_plugins": len(self.plugin_classes),
            "loaded_instances": len(self.plugin_instances),
            "available_plugins": list(self.plugin_classes.keys()),
            "loaded_plugins": list(self.plugin_instances.keys()),
            "builtin_plugins_dir": self.builtin_plugins_dir,
            "user_plugins_dir": self.user_plugins_dir
        } 
"""
Модуль для управления настройками приложения.
Обеспечивает загрузку и сохранение настроек в файл конфигурации.
"""
import os
import configparser
from pathlib import Path

from . import config

class SettingsManager:
    """
    Класс для управления настройками приложения.
    Позволяет загружать настройки из файла и сохранять их автоматически.
    """
    def __init__(self, settings_file=config.SETTINGS_FILE):
        """Инициализация менеджера настроек."""
        # Гарантируем, что путь к файлу настроек находится в директории APP_DATA_PATH
        # Это важно, если SETTINGS_FILE - это просто имя файла
        if not os.path.isabs(settings_file):
            self.settings_file_path = os.path.join(config.APP_DATA_PATH, settings_file)
        else:
            self.settings_file_path = settings_file
            
        self.config = configparser.ConfigParser()
        self._ensure_data_dir_exists() # Убедимся, что директория APP_DATA_PATH существует
        self.load_settings()
    
    def _ensure_data_dir_exists(self):
        # Создаем директорию для файла настроек, если ее нет
        data_dir = os.path.dirname(self.settings_file_path)
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                print(f"Создана директория для настроек: {data_dir}")
            except OSError as e:
                print(f"Ошибка при создании директории {data_dir}: {e}")
                # Если не удалось создать директорию, возможно, стоит кинуть исключение
                # или использовать временный файл, но пока просто выведем ошибку.
    
    def load_settings(self):
        """Загрузка настроек из файла."""
        if os.path.exists(self.settings_file_path):
            try:
                self.config.read(self.settings_file_path, encoding='utf-8')
                
                # Загружаем общие настройки
                if 'General' in self.config:
                    general = self.config['General']
                    config.APP_NAME = general.get('app_name', config.APP_NAME)
                    config.APP_VERSION = general.get('app_version', config.APP_VERSION)
                
                # Загружаем настройки Tesseract
                if 'Tesseract' in self.config:
                    tesseract = self.config['Tesseract']
                    config.TESSERACT_PATH = tesseract.get('path', config.TESSERACT_PATH)
                    config.DEFAULT_TESSERACT_LANG = tesseract.get('lang', config.DEFAULT_TESSERACT_LANG)
                
                # Загружаем настройки моделей
                if 'Models' in self.config:
                    models = self.config['Models']
                    config.LAYOUTLM_MODEL_ID = models.get('layoutlm_id', config.LAYOUTLM_MODEL_ID)
                    config.DONUT_MODEL_ID = models.get('donut_id', config.DONUT_MODEL_ID)
                    config.ACTIVE_LAYOUTLM_MODEL_TYPE = models.get('active_layoutlm_model_type', config.ACTIVE_LAYOUTLM_MODEL_TYPE)
                    config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME = models.get('custom_layoutlm_model_name', config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME)
                
                # Загружаем настройки Gemini
                if 'Gemini' in self.config:
                    gemini = self.config['Gemini']
                    config.GEMINI_API_KEY = gemini.get('api_key', config.GEMINI_API_KEY)
                    config.GEMINI_MODEL_ID = gemini.get('sub_model_id', config.GEMINI_MODEL_ID)
                    config.DEFAULT_GEMINI_TEMPERATURE = float(gemini.get('temperature', str(config.DEFAULT_GEMINI_TEMPERATURE)))
                    config.DEFAULT_GEMINI_MAX_TOKENS = int(gemini.get('max_tokens', str(config.DEFAULT_GEMINI_MAX_TOKENS)))
                    config.GEMINI_PDF_DPI = int(gemini.get('pdf_dpi', str(config.GEMINI_PDF_DPI)))
                    config.GEMINI_AVAILABLE_MODELS = gemini.get('available_models_json', config.GEMINI_AVAILABLE_MODELS)
                
                # Загружаем настройки сети
                if 'Network' in self.config:
                    network = self.config['Network']
                    config.OFFLINE_MODE = network.getboolean('offline_mode', config.OFFLINE_MODE)
                    config.HTTP_TIMEOUT = network.getint('http_timeout', config.HTTP_TIMEOUT)
                
                # Загружаем настройки путей
                if 'Paths' in self.config:
                    paths = self.config['Paths']
                    config.TESSERACT_PATH = paths.get('tesseract_path', config.TESSERACT_PATH)
                    config.POPPLER_PATH = paths.get('poppler_path', config.POPPLER_PATH)
                    config.TRAINING_DATASETS_PATH = paths.get('training_datasets_path', config.TRAINING_DATASETS_PATH)
                    config.TRAINED_MODELS_PATH = paths.get('trained_models_path', config.TRAINED_MODELS_PATH)
                
                # Загружаем настройки обучения
                if 'Training' in self.config:
                    training = self.config['Training']
                    config.LAYOUTLM_MODEL_ID_FOR_TRAINING = training.get('layoutlm_base_model_for_training', config.LAYOUTLM_MODEL_ID_FOR_TRAINING)
                    config.DEFAULT_TRAIN_EPOCHS = training.getint('default_train_epochs', config.DEFAULT_TRAIN_EPOCHS)
                    config.DEFAULT_TRAIN_BATCH_SIZE = training.getint('default_train_batch_size', config.DEFAULT_TRAIN_BATCH_SIZE)
                    config.DEFAULT_LEARNING_RATE = float(training.get('default_learning_rate', str(config.DEFAULT_LEARNING_RATE)))
                    config.TRAINING_DATASETS_PATH = training.get('training_datasets_path', config.TRAINING_DATASETS_PATH)
                    config.TRAINED_MODELS_PATH = training.get('trained_models_path', config.TRAINED_MODELS_PATH)
                
                # Загружаем настройки интерфейса
                if 'Interface' not in self.config:
                    self.config['Interface'] = {
                        'active_model': 'layoutlm',  # Активная модель по умолчанию
                        'last_export_path': '',      # Последний путь экспорта
                        'last_open_path': '',        # Последний путь открытия файла
                        'show_preview': 'True'       # Показывать предпросмотр
                    }
                
                # Загружаем настройки обработки
                if 'Processing' not in self.config:
                    self.config['Processing'] = {
                        'preprocess_images': 'True',  # Предобработка изображений
                        'denoise_level': '0',         # Уровень шумоподавления (0-100)
                        'contrast_enhance': 'False',  # Увеличение контрастности
                        'image_resize': '1280'        # Размер для изменения размера изображения
                    }
                
                # Загружаем настройки OCR
                if 'OCR' not in self.config:
                    self.config['OCR'] = {
                        'use_osd': 'False',           # Использовать OSD (ориентацию страницы)
                        'psm_mode': '3',              # Page Segmentation Mode (3 - авто)
                        'oem_mode': '3'               # OCR Engine Mode (3 - по умолчанию)
                    }
                
                print(f"Настройки загружены из {self.settings_file_path}")
                return True
            except Exception as e:
                print(f"Ошибка при загрузке настроек: {str(e)}")
                return False
        else:
            print("Файл настроек не найден, будут использованы настройки по умолчанию")
            # Добавляем разделы по умолчанию
            for section, values in {
                'General': {
                    'active_model': 'layoutlm',
                    'theme': 'light',
                    'auto_load_last_file': 'false',
                    'last_opened_file': ''
                },
                'Paths': {
                    'tesseract_path': config.TESSERACT_PATH,
                    'poppler_path': config.POPPLER_PATH,
                    'last_open_path': '',
                    'last_export_path': ''
                },
                'OCR': {
                    'language': config.DEFAULT_TESSERACT_LANG
                },
                'Models': {
                    'layoutlm_id': config.LAYOUTLM_MODEL_ID,
                    'donut_id': config.DONUT_MODEL_ID,
                    'gemini_id': config.GEMINI_MODEL_ID,
                    'active_layoutlm_model_type': config.ACTIVE_LAYOUTLM_MODEL_TYPE,
                    'custom_layoutlm_model_name': config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME
                },
                'Prompts': {
                    'layoutlm_prompt': config.LAYOUTLM_PROMPT_DEFAULT,
                    'donut_prompt': config.DONUT_PROMPT_DEFAULT,
                    'gemini_prompt': config.GEMINI_PROMPT_DEFAULT,
                    'gemini_annotation_prompt': config.GEMINI_ANNOTATION_PROMPT_DEFAULT
                },
                'Gemini': {
                    'api_key': '',
                    'sub_model_id': config.GEMINI_MODEL_ID,
                    'available_models_json': '[]'
                },
                'Training': {
                    'training_datasets_path': config.TRAINING_DATASETS_PATH,
                    'trained_models_path': config.TRAINED_MODELS_PATH,
                    'layoutlm_base_model_for_training': config.LAYOUTLM_MODEL_ID_FOR_TRAINING,
                    'default_train_epochs': str(config.DEFAULT_TRAIN_EPOCHS),
                    'default_train_batch_size': str(config.DEFAULT_TRAIN_BATCH_SIZE),
                    'default_learning_rate': str(config.DEFAULT_LEARNING_RATE)
                },
                'Network': {
                    'offline_mode': str(config.OFFLINE_MODE).lower(),
                    'http_timeout': str(config.HTTP_TIMEOUT),
                    'hf_token': config.HF_TOKEN if config.HF_TOKEN else ''
                },
                'Invoice': {
                    'default_vat_rate': str(config.DEFAULT_VAT_RATE)
                }
            }.items():
                if section not in self.config:
                    self.config[section] = {}
                    for key, value in values.items():
                        self.config[section][key] = value
            
            # Сразу сохраняем настройки по умолчанию
            self.save_settings()
            return False
    
    def save_settings(self):
        """Сохранение текущего состояния настроек (self.config) в файл."""
        try:
            # Просто сохраняем текущее содержимое self.config в файл
            # Убедимся, что основные секции существуют, но не перезаписываем их из config
            required_sections = ['General', 'Tesseract', 'Models', 'Network', 'Prompts', 
                                 'Interface', 'Processing', 'OCR', 'Paths', 'Gemini', 
                                 'ModelsStatus', 'Misc', 'Training', 'Invoice'] # Добавляем все ожидаемые секции
            for section in required_sections:
                if section not in self.config:
                    self.config.add_section(section)
            
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(self.settings_file_path), exist_ok=True)
            
            with open(self.settings_file_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
            
            print(f"Настройки сохранены в {self.settings_file_path}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении настроек: {str(e)}")
            return False
    
    def get_string(self, section, key, default=""):
        """Получение строкового значения из настроек."""
        # Для раздела Prompts используем отдельные файлы
        if section == 'Prompts':
            return self.get_prompt(key, default)
            
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default
    
    def get_prompt(self, key, default=""):
        """Специальный метод для чтения промтов из отдельных файлов."""
        # Промты хранятся в отдельных файлах в папке prompts
        prompts_dir = os.path.join(os.path.dirname(self.settings_file_path), "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # Имя файла - это ключ промта с расширением .txt
        prompt_file = os.path.join(prompts_dir, f"{key}.txt")
        
        if not os.path.exists(prompt_file):
            return default
            
        try:
            # Читаем файл с промтом
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Ошибка при чтении промта {key}: {str(e)}")
            return default
    
    def get_model_prompt(self, model_name):
        """Получение промпта для указанной модели."""
        # Определяем дефолтные значения для каждой модели
        defaults = {
            'layoutlm': config.LAYOUTLM_PROMPT_DEFAULT,
            'donut': config.DONUT_PROMPT_DEFAULT,
            'gemini': config.GEMINI_PROMPT_DEFAULT
        }
        
        default_prompt = defaults.get(model_name, "")
        
        # Используем существующий метод get_prompt для чтения из файла
        return self.get_prompt(f"{model_name}_prompt", default_prompt)
    
    def get_int(self, section, key, default=0):
        """Получение целочисленного значения из настроек."""
        try:
            if section in self.config and key in self.config[section]:
                return self.config[section].getint(key)
        except ValueError:
            pass
        return default
    
    def get_float(self, section, key, default=0.0):
        """Получение значения с плавающей точкой из настроек."""
        try:
            if section in self.config and key in self.config[section]:
                return self.config[section].getfloat(key)
        except ValueError:
            pass
        return default
    
    def get_bool(self, section, key, default=False):
        """Получение булева значения из настроек."""
        try:
            if section in self.config and key in self.config[section]:
                return self.config[section].getboolean(key)
        except ValueError:
            pass
        return default
    
    def set_value(self, section, key, value):
        """Установка значения в настройках, обновление config и сохранение."""
        if section not in self.config:
            self.config.add_section(section)
        
        str_value = str(value)

        # Обновляем значение в config для текущей сессии
        try:
            if section == 'Models' and key == 'layoutlm_id':
                config.LAYOUTLM_MODEL_ID = str_value
                print(f"Config updated: config.LAYOUTLM_MODEL_ID = {str_value}")
            elif section == 'Models' and key == 'donut_id':
                config.DONUT_MODEL_ID = str_value
                print(f"Config updated: config.DONUT_MODEL_ID = {str_value}")
            elif section == 'Network' and key == 'offline_mode':
                config.OFFLINE_MODE = value if isinstance(value, bool) else (str_value.lower() == 'true')
                print(f"Config updated: config.OFFLINE_MODE = {config.OFFLINE_MODE}")
            elif section == 'Network' and key == 'http_timeout':
                config.HTTP_TIMEOUT = int(str_value)
                print(f"Config updated: config.HTTP_TIMEOUT = {config.HTTP_TIMEOUT}")
            elif section == 'Tesseract' and key == 'path':
                config.TESSERACT_PATH = str_value
                print(f"Config updated: config.TESSERACT_PATH = {str_value}")
            elif section == 'Tesseract' and key == 'lang':
                config.DEFAULT_TESSERACT_LANG = str_value
                print(f"Config updated: config.DEFAULT_TESSERACT_LANG = {str_value}")
            elif section == 'Paths' and key == 'poppler_path':
                config.POPPLER_PATH = str_value
                print(f"Config updated: config.POPPLER_PATH = {str_value}")
            elif section == 'Gemini' and key == 'api_key':
                config.GOOGLE_API_KEY = str_value
                print(f"Config updated: config.GOOGLE_API_KEY set (value hidden)")
            elif section == 'Gemini' and key == 'temperature':
                config.DEFAULT_GEMINI_TEMPERATURE = float(str_value)
                print(f"Config updated: config.DEFAULT_GEMINI_TEMPERATURE = {config.DEFAULT_GEMINI_TEMPERATURE}")
            elif section == 'Gemini' and key == 'max_tokens':
                config.DEFAULT_GEMINI_MAX_TOKENS = int(str_value)
                print(f"Config updated: config.DEFAULT_GEMINI_MAX_TOKENS = {config.DEFAULT_GEMINI_MAX_TOKENS}")
            elif section == 'Gemini' and key == 'sub_model_id':
                config.GEMINI_MODEL_ID = str_value
                print(f"Config updated: config.GEMINI_MODEL_ID = {config.GEMINI_MODEL_ID}")
            elif section == 'Misc' and key == 'batch_processing_delay':
                config.DEFAULT_BATCH_PROCESSING_DELAY = int(str_value)
                print(f"Config updated: config.DEFAULT_BATCH_PROCESSING_DELAY = {config.DEFAULT_BATCH_PROCESSING_DELAY}")
            # Добавить другие настройки из config по мере необходимости
        except Exception as e:
            print(f"Предупреждение: Не удалось обновить соответствующую переменную в config: {e}")

        # Для значений в разделе Prompts используем отдельные файлы
        if section == 'Prompts':
            try:
                # Создаем директорию для промтов, если не существует
                prompts_dir = os.path.join(os.path.dirname(self.settings_file_path), "prompts")
                os.makedirs(prompts_dir, exist_ok=True)
                
                # Сохраняем промт в отдельный файл
                prompt_file = os.path.join(prompts_dir, f"{key}.txt")
                
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(str_value)
                
                print(f"Промт {key} сохранен в файл {prompt_file}")
                return True
            except Exception as e:
                print(f"Ошибка при сохранении промта {key}: {str(e)}")
                return False
        else:
            # Для обычных значений используем стандартный механизм
            self.config[section][key] = str_value
            self.save_settings()
    
    def remove_value(self, section, key):
        """Удаление значения из настроек."""
        if section in self.config and key in self.config[section]:
            del self.config[section][key]
            self.save_settings()
    
    def save_interface_setting(self, key, value):
        """Сохранение настройки интерфейса."""
        self.set_value('Interface', key, value)
    
    def save_processing_setting(self, key, value):
        """Сохранение настройки обработки."""
        self.set_value('Processing', key, value)
    
    def save_ocr_setting(self, key, value):
        """Сохранение настройки OCR."""
        self.set_value('OCR', key, value)
    
    def get_active_model(self):
        """Получение активной модели из настроек."""
        return self.get_string('Interface', 'active_model', 'layoutlm')
    
    def get_gemini_api_key(self):
        """
        Получение API ключа Gemini из настроек или новой системы безопасности.
        Этот метод унифицирует доступ к ключу, чтобы использовался только один источник.
        """
        # Получаем ключ из настроек
        api_key = self.get_string('Gemini', 'api_key', '')
        
        # Если ключа нет в настройках, пытаемся получить из новой системы безопасности
        if not api_key:
            try:
                from config.secrets import SecretsManager
                secrets_manager = SecretsManager()
                api_key = secrets_manager.get_secret("GOOGLE_API_KEY")
                
                # Если нашли ключ в системе безопасности, сохраняем его в настройки для совместимости
                if api_key:
                    self.set_value('Gemini', 'api_key', api_key)
                    print("API ключ Gemini перенесен из системы безопасности в настройки для совместимости")
                    
            except ImportError:
                # Система безопасности недоступна, это нормально
                pass
            except Exception as e:
                print(f"Ошибка при попытке получения API ключа Gemini из системы безопасности: {e}")
        
        return api_key
    
    def set_active_model(self, model_name):
        """Установка активной модели в настройках."""
        self.set_value('Interface', 'active_model', model_name)

    def get_tesseract_path(self):
        """Получение пути к Tesseract."""
        return self.get_string('Paths', 'tesseract_path', config.TESSERACT_PATH)

    def set_tesseract_path(self, path):
        """Установка пути к Tesseract."""
        self.set_value('Paths', 'tesseract_path', path)
        config.TESSERACT_PATH = path # Обновляем и в текущей сессии

    def get_poppler_path(self):
        """Получение пути к Poppler."""
        return self.get_string('Paths', 'poppler_path', config.POPPLER_PATH)

    def set_poppler_path(self, path):
        """Установка пути к Poppler."""
        self.set_value('Paths', 'poppler_path', path)
        config.POPPLER_PATH = path # Обновляем и в текущей сессии

    def get_active_layoutlm_model_type(self):
        """Получение типа активной LayoutLM модели."""
        return self.get_string('Models', 'active_layoutlm_model_type', config.ACTIVE_LAYOUTLM_MODEL_TYPE)

    def set_active_layoutlm_model_type(self, model_type):
        """Установка типа активной LayoutLM модели."""
        if model_type not in ["huggingface", "custom"]:
            model_type = "huggingface" # Фоллбэк на безопасное значение
            print(f"Предупреждение: Недопустимый тип модели LayoutLM '{model_type}'. Установлен 'huggingface'.")
        self.set_value('Models', 'active_layoutlm_model_type', model_type)
        config.ACTIVE_LAYOUTLM_MODEL_TYPE = model_type # Обновляем глобальную переменную

    def get_custom_layoutlm_model_name(self):
        """Получение имени кастомной LayoutLM модели."""
        return self.get_string('Models', 'custom_layoutlm_model_name', config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME)

    def set_custom_layoutlm_model_name(self, model_name):
        """Установка имени кастомной LayoutLM модели."""
        self.set_value('Models', 'custom_layoutlm_model_name', model_name)
        config.DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME = model_name

    def get_huggingface_token(self):
        """Получение токена Hugging Face из настроек."""
        # Сначала пытаемся получить токен из новой секции 'HuggingFace'
        token = self.get_string('HuggingFace', 'token', '')
        
        # Если токен не найден, пытаемся получить из старой секции 'Network' для обратной совместимости
        if not token:
            token = self.get_string('Network', 'hf_token', '')
            
            # Если нашли токен в старой секции, сохраняем его в новую и удаляем из старой
            if token:
                print("Перемещаем токен HF из 'Network' в 'HuggingFace'")
                self.set_value('HuggingFace', 'token', token)
                self.remove_value('Network', 'hf_token')
                self.save_settings()
                
        return token
        
    def get_default_vat_rate(self):
        """Получение ставки НДС по умолчанию."""
        try:
            if 'Invoice' in self.config and 'default_vat_rate' in self.config['Invoice']:
                return float(self.config['Invoice']['default_vat_rate'])
            return config.DEFAULT_VAT_RATE
        except (ValueError, TypeError):
            return config.DEFAULT_VAT_RATE
            
    def set_default_vat_rate(self, rate):
        """Устанавливает ставку НДС по умолчанию."""
        if isinstance(rate, (int, float)) and 0 <= rate <= 100:
            # Округляем до 2 знаков после запятой
            rate_str = str(round(float(rate), 2))
            # Сохраняем в настройки
            self.set_value('Invoice', 'default_vat_rate', rate_str)
            # Обновляем значение в config
            from . import config
            config.DEFAULT_VAT_RATE = float(rate)
            return True
        return False

    # Методы для работы с настраиваемыми полями таблицы
    def get_table_fields(self):
        """Возвращает настраиваемые поля таблицы результатов."""
        try:
            import json
            # Получаем JSON строку из настроек
            fields_json = self.config.get('TableFields', 'fields_json', fallback=None)
            
            if fields_json:
                # Преобразуем строку JSON в список полей
                fields = json.loads(fields_json)
                if isinstance(fields, list):
                    return fields
                
            # Если поля не найдены или некорректны, возвращаем поля по умолчанию
            return [
                {"id": "seller_name", "name": "Поставщик", "visible": True, "order": 0},
                {"id": "seller_inn", "name": "ИНН поставщика", "visible": True, "order": 1},
                {"id": "invoice_number", "name": "№ счета", "visible": True, "order": 2},
                {"id": "invoice_date", "name": "Дата счета", "visible": True, "order": 3},
                {"id": "subtotal", "name": "Сумма без НДС", "visible": True, "order": 4},
                {"id": "vat_amount", "name": "Сумма НДС", "visible": True, "order": 5},
                {"id": "total_amount", "name": "Итого к оплате", "visible": True, "order": 6}
            ]
        except Exception as e:
            print(f"Ошибка при загрузке полей таблицы: {e}")
            return []
    
    def save_table_fields(self, fields):
        """Сохраняет настраиваемые поля таблицы результатов."""
        if not isinstance(fields, list):
            return False
        
        try:
            import json
            # Создаем секцию TableFields, если её нет
            if 'TableFields' not in self.config:
                self.config['TableFields'] = {}
            
            # Сохраняем поля как строку JSON с экранированием специальных символов
            fields_json = json.dumps(fields, ensure_ascii=False)
            fields_json = fields_json.replace('%', '%%')  # Экранируем % для configparser
            self.config['TableFields']['fields_json'] = fields_json
            
            # Сохраняем настройки в файл
            with open(self.settings_file_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении полей таблицы: {e}")
            return False
            
    def get_training_prompt(self, model_type=None):
        """Получает промпт для обучения модели."""
        from . import config
        
        # Если модель не указана, возвращаем все промпты для обучения
        if model_type is None:
            prompts = {}
            for key in ['gemini_annotation']:
                prompts[key] = self.get_prompt(f'{key}_prompt')
            return prompts
        
        # Иначе возвращаем промпт для указанной модели
        if model_type == 'gemini_annotation':
            return self.get_prompt('gemini_annotation_prompt')
            
        return ""
        
    def save_training_prompt(self, model_type, prompt):
        """Сохраняет промпт для обучения в отдельный файл."""
        # Промты хранятся в отдельных файлах в папке prompts
        prompts_dir = os.path.join(os.path.dirname(self.settings_file_path), "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # Имя файла - это ключ промта с расширением .txt
        prompt_file = os.path.join(prompts_dir, f"{model_type}_prompt.txt")
        
        try:
            # Записываем промт в файл
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"Промт {model_type} сохранен в файл {prompt_file}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении промта {model_type}: {str(e)}")
            return False

    def refresh_layoutlm_models(self):
        """Обновляет список доступных моделей LayoutLM в ModelManagementDialog."""
        trained_models_path = self.get_string('Training', 'trained_models_path', config.TRAINED_MODELS_PATH)
        try:
            from transformers import AutoConfig, AutoTokenizer
            
            trained_models = []
            if os.path.exists(trained_models_path):
                for model_folder in os.listdir(trained_models_path):
                    model_path = os.path.join(trained_models_path, model_folder)
                    # Проверяем, похоже ли это на папку с моделью Hugging Face
                    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
                        try:
                            # Пробуем загрузить конфигурацию для проверки
                            model_config = AutoConfig.from_pretrained(model_path)
                            if hasattr(model_config, 'model_type') and model_config.model_type == 'layoutlmv3':
                                trained_models.append(model_folder)
                        except Exception as e:
                            print(f"Ошибка при проверке модели {model_folder}: {str(e)}")
                            continue
            
            # Обновляем список моделей в настройках
            self.config['ModelsStatus'] = self.config.get('ModelsStatus', {})
            self.config['ModelsStatus']['trained_layoutlm_models'] = ','.join(trained_models)
            self.save_settings()
            return trained_models
        except ImportError:
            print("Не удалось импортировать transformers для проверки моделей")
            return []

    # Методы для настройки GPU

    def get_training_device(self):
        """
        Возвращает устройство для обучения (cpu, cuda)
        
        Returns:
            str: Название устройства
        """
        try:
            # Проверяем, доступна ли CUDA
            import torch
            cuda_available = torch.cuda.is_available()
            
            # Если CUDA недоступна, принудительно возвращаем cpu
            if not cuda_available:
                return "cpu"
                
            # Иначе возвращаем настройку пользователя
            return self.get_string('GPU', 'training_device', 'cuda' if cuda_available else 'cpu')
        except Exception:
            # По умолчанию используем CPU
            return 'cpu'
        
    def set_training_device(self, device):
        """
        Устанавливает устройство для обучения
        
        Args:
            device: 'cpu' или 'cuda'
        """
        if device not in ['cpu', 'cuda']:
            device = 'cpu'
            
        # Проверка доступности CUDA перед сохранением
        if device == 'cuda' and not self._is_cuda_available():
            self.logger.warning("CUDA недоступна, выбрана CPU")
            device = 'cpu'
            
        self.set_value('GPU', 'training_device', device)

    def get_use_gpu_if_available(self):
        """
        Возвращает флаг, нужно ли использовать GPU, если доступно
        
        Returns:
            bool: True, если нужно использовать GPU
        """
        try:
            return self.get_boolean('GPU', 'use_if_available', True)
        except Exception:
            # По умолчанию используем GPU, если доступна
            return True
        
    def set_use_gpu_if_available(self, value):
        """
        Устанавливает настройку автоматического использования GPU
        
        Args:
            value: True или False
        """
        self.set_value('GPU', 'use_if_available', value)
        
    def get_max_gpu_memory(self):
        """
        Возвращает максимально разрешенный объем памяти GPU в процентах (0-100)
        0 означает без ограничений
        
        Returns:
            int: Процент ограничения памяти GPU (0-100)
        """
        try:
            # Получаем значение из настроек или используем 0 (без ограничений) по умолчанию
            return self.get_int('GPU', 'max_memory_percent', 0)
        except Exception:
            # В случае ошибки используем значение по умолчанию (без ограничений)
            return 0
            
    def set_max_gpu_memory(self, value):
        """
        Устанавливает максимальное количество памяти GPU для использования
        
        Args:
            value: Процент ограничения памяти GPU (0-100)
        """
        # Ограничиваем значение в диапазоне 0-100
        value = max(0, min(100, int(value)))
        self.set_value('GPU', 'max_memory_percent', value)
        
    def get_multi_gpu_strategy(self):
        """
        Возвращает стратегию для мульти-GPU (none, data_parallel)
        
        Returns:
            str: Название стратегии
        """
        try:
            return self.get_string('GPU', 'multi_gpu_strategy', 'none')
        except Exception:
            # По умолчанию не используем мульти-GPU
            return 'none'
            
    def set_multi_gpu_strategy(self, strategy):
        """
        Устанавливает стратегию для использования нескольких GPU
        
        Args:
            strategy: 'none', 'data_parallel'
        """
        if strategy not in ['none', 'data_parallel']:
            strategy = 'none'
        self.set_value('GPU', 'multi_gpu_strategy', strategy)

    def _is_cuda_available(self):
        """Проверяет доступность CUDA."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False
    
    def get_cuda_device_info(self):
        """Получает информацию об устройствах CUDA."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
                
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory
            }
        except Exception as e:
            print(f"Ошибка получения информации о CUDA: {e}")
            return {}
    
    # NEW: Universal methods for LLM providers settings
    def get_setting(self, key: str, default=None):
        """
        Универсальный метод для получения настроек.
        
        Args:
            key: Ключ настройки (может содержать точки для вложенных значений)
            default: Значение по умолчанию
        """
        try:
            import json
            
            # Поддержка вложенных ключей через точку
            if '.' in key:
                parts = key.split('.')
                section = parts[0]
                sub_key = '.'.join(parts[1:])
                
                if section in self.config:
                    section_data = self.config[section]
                    if sub_key in section_data:
                        # Пытаемся распарсить JSON если это строка
                        value = section_data[sub_key]
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                return json.loads(value)
                            except:
                                return value
                        return value
                        
                return default
            else:
                # Простой ключ - ищем во всех секциях
                for section_name, section_data in self.config.items():
                    if key in section_data:
                        value = section_data[key]
                        # Пытаемся распарсить JSON если это строка
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                return json.loads(value)
                            except:
                                return value
                        return value
                return default
                
        except Exception as e:
            print(f"Ошибка получения настройки {key}: {e}")
            return default
    
    def save_setting(self, key: str, value):
        """
        Универсальный метод для сохранения настроек.
        
        Args:
            key: Ключ настройки (может содержать точки для вложенных значений)
            value: Значение для сохранения
        """
        try:
            import json
            
            # Поддержка вложенных ключей через точку
            if '.' in key:
                parts = key.split('.')
                section = parts[0]
                sub_key = '.'.join(parts[1:])
                
                # Создаем секцию если её нет
                if section not in self.config:
                    self.config[section] = {}
                
                # Сериализуем сложные объекты в JSON
                if isinstance(value, (dict, list)):
                    self.config[section][sub_key] = json.dumps(value, ensure_ascii=False)
                else:
                    self.config[section][sub_key] = str(value)
            else:
                # Простой ключ - сохраняем в секции General
                if 'General' not in self.config:
                    self.config['General'] = {}
                
                if isinstance(value, (dict, list)):
                    self.config['General'][key] = json.dumps(value, ensure_ascii=False)
                else:
                    self.config['General'][key] = str(value)
            
            # Сохраняем настройки
            self.save_settings()
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения настройки {key}: {e}")
            return False
    
    def get_encrypted_setting(self, key: str, default=None):
        """
        Получает зашифрованную настройку через систему секретов.
        
        Args:
            key: Ключ секрета
            default: Значение по умолчанию
        """
        try:
            from config.secrets import SecretsManager
            secrets = SecretsManager()
            return secrets.get_secret(key, default)
        except Exception as e:
            print(f"Ошибка получения зашифрованной настройки {key}: {e}")
            return default
    
    def save_encrypted_setting(self, key: str, value: str):
        """
        Сохраняет зашифрованную настройку через систему секретов.
        
        Args:
            key: Ключ секрета
            value: Значение для сохранения
        """
        try:
            from config.secrets import SecretsManager
            secrets = SecretsManager()
            return secrets.set_secret(key, value)
        except Exception as e:
            print(f"Ошибка сохранения зашифрованной настройки {key}: {e}")
            return False

# Создаем глобальный экземпляр менеджера настроек
settings_manager = SettingsManager() 
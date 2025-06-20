"""
Файл конфигурации приложения, содержащий пути и настройки по умолчанию.
"""
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from PyQt6.QtGui import QColor
import configparser

# Определение GENAI_AVAILABLE
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Загрузка переменных окружения из .env файла
load_dotenv()

# Общие настройки приложения
APP_NAME = "InvoiceExtractorGUI"
APP_VERSION = "0.9.6"
ORGANIZATION_NAME = "ТехноИнновации"
SETTINGS_FILE = "settings.ini"

# APP Description for better documentation
APP_DESCRIPTION = "Приложение для автоматического извлечения данных из счетов-фактур с использованием ИИ"

# Настройки компании-получателя по умолчанию
DEFAULT_COMPANY_RECEIVER_NAME = 'ООО "Рога и копыта"'

# Загружаем переменные окружения из .env файла безопасно
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(app_dir, ".env")
if os.path.exists(env_path):
    try:
        load_dotenv(env_path)
    except Exception as e:
        print(f"Предупреждение: Ошибка загрузки .env файла: {e}")

# Определение путей для хранения данных приложения (портативный режим)
def get_app_data_path():
    """Возвращает путь для хранения данных приложения в зависимости от режима."""
    # Если приложение должно быть портативным, используем директорию приложения
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_data = os.path.join(app_dir, "data")
    
    # Создаем директорию, если она не существует
    os.makedirs(app_data, exist_ok=True)
    return app_data

# Пути для хранения данных
APP_DATA_PATH = get_app_data_path()
MODELS_PATH = os.path.join(APP_DATA_PATH, "models")
TEMP_PATH = os.path.join(APP_DATA_PATH, "temp")
TRAINING_DATASETS_PATH = os.path.join(APP_DATA_PATH, "training_datasets")
TRAINED_MODELS_PATH = os.path.join(APP_DATA_PATH, "trained_models")

# Модели ML
# LAYOUTLM_MODEL_ID = "microsoft/layoutlmv3-base-finetuned-funsd"
LAYOUTLM_MODEL_ID = "naver-clova-ix/cord-layoutlmv3"
ACTIVE_LAYOUTLM_MODEL_TYPE = "base"  # Тип активной модели LayoutLM: 'base' или 'custom'
DONUT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
GEMINI_MODEL_ID = "models/gemini-1.5-flash-latest" # ID модели Gemini по умолчанию
GEMINI_API_KEY = None  # API ключ будет загружен из файла или переменной окружения
# Удалено: GEMINI_API_KEY_FILE - теперь используется система зашифрованного хранения секретов
GEMINI_AVAILABLE_MODELS = [
    "models/gemini-1.5-pro-vision-latest",
    "models/gemini-pro-vision",
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-001-tuning",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-002",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001"
]

# Настройки аутентификации Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Сначала проверяем переменную окружения
HF_TOKEN_FILE = os.path.join(APP_DATA_PATH, "hf_token.txt")  # Файл для хранения токена

# Настройки аутентификации Google API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)  # API ключ для Google Gemini
GOOGLE_API_KEY_FILE = os.path.join(APP_DATA_PATH, "google_api_key.txt")  # Файл для хранения API ключа

# Данные о моделях
MODELS_INFO = {
    "layoutlm": {
        "id": LAYOUTLM_MODEL_ID,
        "name": "LayoutLMv3",
        "description": "Мощная модель для распознавания структуры документов, с предобучением на изображениях и тексте.",
        "task": "token-classification",
        "size_mb": 870,  # Приблизительный размер модели в МБ (large вариант)
        "requires_ocr": True,
        "languages": ["eng", "rus", "multi"],
        "version": "large"
    },
    "donut": {
        "id": DONUT_MODEL_ID,
        "name": "Donut",
        "description": "Document understanding transformer для извлечения данных из документов без OCR.",
        "task": "document-question-answering",
        "size_mb": 680,  # Приблизительный размер модели в МБ
        "requires_ocr": False,
        "languages": ["eng", "multi"],
        "version": "base"
    },
    "gemini": {
        "id": GEMINI_MODEL_ID,
        "name": "Gemini 2.0 Flash",
        "description": "Мощная мультимодальная модель от Google для понимания изображений и текста.",
        "task": "multimodal-understanding",
        "size_mb": 0,  # Облачная модель
        "requires_ocr": False,
        "languages": ["eng", "rus", "multi"],
        "version": "flash"
    }
}

# Настройки Tesseract OCR
DEFAULT_TESSERACT_LANG = os.getenv("DEFAULT_TESSERACT_LANG", "rus+eng")  # Языки по умолчанию для OCR
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "") # Путь к tesseract.exe (автопоиск если пустой)

# Настройки Poppler (для обработки PDF) - используем относительный путь к bundled версии
def get_default_poppler_path():
    """Получает правильный путь к Poppler на основе текущего расположения проекта."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "bin", "poppler", "poppler-24.08.0", "Library", "bin")

DEFAULT_POPPLER_PATH = get_default_poppler_path()
POPPLER_PATH = os.getenv("POPPLER_PATH", DEFAULT_POPPLER_PATH) # Путь к папке Poppler

def update_paths_from_settings():
    """Обновляет пути из настроек, если они отличаются от значений по умолчанию."""
    global POPPLER_PATH, TRAINING_DATASETS_PATH, TRAINED_MODELS_PATH
    try:
        from .settings_manager import SettingsManager
        settings = SettingsManager()
        
        # Обновляем путь к Poppler из настроек
        saved_poppler_path = settings.get_poppler_path()
        if saved_poppler_path and os.path.exists(saved_poppler_path):
            POPPLER_PATH = saved_poppler_path
        else:
            # Если сохраненный путь не существует, используем текущий по умолчанию
            POPPLER_PATH = get_default_poppler_path()
            if os.path.exists(POPPLER_PATH):
                # Обновляем в настройках правильный путь
                settings.set_poppler_path(POPPLER_PATH)
        
        # Обновляем пути для обучения из настроек Paths (не Training)
        training_datasets_path = settings.get_string('Paths', 'training_datasets_path', TRAINING_DATASETS_PATH)
        if training_datasets_path and os.path.exists(os.path.dirname(training_datasets_path)):
            TRAINING_DATASETS_PATH = training_datasets_path
        else:
            # Используем текущий путь по умолчанию
            current_training_path = os.path.join(APP_DATA_PATH, "training_datasets")
            os.makedirs(current_training_path, exist_ok=True)
            TRAINING_DATASETS_PATH = current_training_path
            settings.set_value('Paths', 'training_datasets_path', TRAINING_DATASETS_PATH)
        
        trained_models_path = settings.get_string('Paths', 'trained_models_path', TRAINED_MODELS_PATH)
        if trained_models_path and os.path.exists(os.path.dirname(trained_models_path)):
            TRAINED_MODELS_PATH = trained_models_path
        else:
            # Используем текущий путь по умолчанию
            current_models_path = os.path.join(APP_DATA_PATH, "trained_models")
            os.makedirs(current_models_path, exist_ok=True)
            TRAINED_MODELS_PATH = current_models_path
            settings.set_value('Paths', 'trained_models_path', TRAINED_MODELS_PATH)
            
        print(f"Обновлены пути обучения:")
        print(f"  TRAINING_DATASETS_PATH: {TRAINING_DATASETS_PATH}")
        print(f"  TRAINED_MODELS_PATH: {TRAINED_MODELS_PATH}")
            
    except Exception as e:
        print(f"Предупреждение: Не удалось обновить пути из настроек: {e}")
        POPPLER_PATH = get_default_poppler_path()

# NEW: Настройки генерации Gemini по умолчанию
DEFAULT_GEMINI_TEMPERATURE = 0.2
DEFAULT_GEMINI_MAX_TOKENS = 8192 # Увеличено с 4096
GEMINI_PDF_DPI = 200 # NEW: DPI для конвертации PDF для Gemini
DEFAULT_BATCH_PROCESSING_DELAY = 2 # NEW: Задержка в секундах между файлами при пакетной обработке

# Промпты для моделей
LAYOUTLM_PROMPT_DEFAULT = "Извлеки структурированные данные из этого документа с учетом его визуального макета."
DONUT_PROMPT_DEFAULT = "Извлеки все поля из счета и представь их в формате JSON."
GEMINI_PROMPT_DEFAULT = """Действуй как эксперт по распознаванию счетов-фактур. Проанализируй данное изображение и извлеки из него все ключевые данные в формате JSON.
Формат должен включать следующие поля (включай только если они присутствуют в документе):
{
  "Поставщик": "название организации-поставщика",
  "ИНН поставщика": "ИНН в формате 10 или 12 цифр",
  "КПП поставщика": "КПП в формате 9 цифр",
  "Адрес поставщика": "полный юридический адрес",
  
  "Покупатель": "название организации-покупателя",
  "ИНН покупателя": "ИНН в формате 10 или 12 цифр",
  "КПП покупателя": "КПП в формате 9 цифр",
  "Адрес покупателя": "полный юридический адрес",
  
  "№ Счета": "номер счета точно как в документе",
  "Дата счета": "дата в формате DD.MM.YYYY",
  "Дата оплаты": "срок оплаты в формате DD.MM.YYYY, если указан",
  
  "Категория": "определи основную категорию товаров/услуг",
  "Товары": "список всех товаров/услуг с количеством и ценами",
  
  "Сумма без НДС": "сумма до НДС числом",
  "НДС %": "ставка НДС числом",
  "Сумма НДС": "сумма НДС числом",
  "Сумма с НДС": "итоговая сумма числом",
  "Валюта": "RUB/USD/EUR и т.д.",
  
  "Банк": "название банка, если указано",
  "БИК": "БИК банка в формате 9 цифр",
  "Р/с": "расчетный счет в формате 20 цифр",
  "К/с": "корреспондентский счет в формате 20 цифр",
  
  "Комментарии": "любая дополнительная информация"
}

Важно: 
1. Представь результат ТОЛЬКО в виде JSON, без лишнего текста до и после.
2. Сохраняй точное форматирование и орфографию из оригинала.
3. Вычисли категорию товаров/услуг на основе их описания.
4. Убедись, что числа форматированы корректно, без лишних пробелов.
5. Для полей с числами (суммы, ИНН, КПП, счета) удали все пробелы и используй точку как разделитель для дробных чисел.
6. Даты всегда приводи к формату DD.MM.YYYY.
"""

GEMINI_ANNOTATION_PROMPT_DEFAULT = """Действуй как эксперт по распознаванию счетов-фактур. Проанализируй данное изображение и извлеки из него все ключевые данные в формате JSON.
Формат должен включать следующие поля (включай только если они присутствуют в документе):
{
  "Поставщик": "название организации-поставщика",
  "ИНН поставщика": "ИНН в формате 10 или 12 цифр",
  "КПП поставщика": "КПП в формате 9 цифр",
  "Адрес поставщика": "полный юридический адрес",
  
  "Покупатель": "название организации-покупателя",
  "ИНН покупателя": "ИНН в формате 10 или 12 цифр",
  "КПП покупателя": "КПП в формате 9 цифр",
  "Адрес покупателя": "полный юридический адрес",
  
  "№ Счета": "номер счета точно как в документе",
  "Дата счета": "дата в формате DD.MM.YYYY",
  "Дата оплаты": "срок оплаты в формате DD.MM.YYYY, если указан",
  
  "Категория": "определи основную категорию товаров/услуг",
  "Товары": "список всех товаров/услуг с количеством и ценами",
  
  "Сумма без НДС": "сумма до НДС числом",
  "НДС %": "ставка НДС числом",
  "Сумма НДС": "сумма НДС числом",
  "Сумма с НДС": "итоговая сумма числом",
  "Валюта": "RUB/USD/EUR и т.д.",
  
  "Банк": "название банка, если указано",
  "БИК": "БИК банка в формате 9 цифр",
  "Р/с": "расчетный счет в формате 20 цифр",
  "К/с": "корреспондентский счет в формате 20 цифр",
  
  "Комментарии": "любая дополнительная информация"
}

Важно: 
1. Представь результат ТОЛЬКО в виде JSON, без лишнего текста до и после.
2. Сохраняй точное форматирование и орфографию из оригинала.
3. Вычисли категорию товаров/услуг на основе их описания.
4. Убедись, что числа форматированы корректно, без лишних пробелов.
5. Для полей с числами (суммы, ИНН, КПП, счета) удали все пробелы и используй точку как разделитель для дробных чисел.
6. Даты всегда приводи к формату DD.MM.YYYY.
"""

# Настройки обработки результатов
EXPORT_FORMATS = ["json", "html", "csv", "txt"]  # Поддерживаемые форматы экспорта

# Настройка НДС
DEFAULT_VAT_RATE = 20.0  # Ставка НДС по умолчанию (%)

# Настройки для обучения моделей
DEFAULT_CUSTOM_LAYOUTLM_MODEL_NAME = ""  # Пустая строка = не выбрана пользовательская модель

# Настройки GPU для обучения
DEFAULT_TRAINING_DEVICE = "cpu"  # Устройство по умолчанию для обучения моделей: 'cpu' или 'cuda'
USE_GPU_IF_AVAILABLE = True      # Автоматически использовать GPU, если он доступен
MAX_GPU_MEMORY_MB = 0            # Лимит памяти GPU (0 = без ограничений)
MULTI_GPU_STRATEGY = "none"      # Стратегия для мульти-GPU: 'none', 'data_parallel', 'model_parallel'

# Настройки для работы с Hugging Face Hub
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "False").lower() == "true"
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", 30)) # Таймаут для HTTP запросов (сек)

# Если токен не найден в переменных окружения, пытаемся загрузить его из файла
if not HF_TOKEN and os.path.exists(HF_TOKEN_FILE):
    try:
        with open(HF_TOKEN_FILE, 'r') as f:
            HF_TOKEN = f.read().strip()
    except Exception:
        pass

# Если Google API ключ не найден в переменных окружения, пытаемся загрузить его из файла
if not GOOGLE_API_KEY and os.path.exists(GOOGLE_API_KEY_FILE):
    try:
        with open(GOOGLE_API_KEY_FILE, 'r') as f:
            GOOGLE_API_KEY = f.read().strip()
    except Exception:
        pass

# Создаем необходимые директории
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(TRAINING_DATASETS_PATH, exist_ok=True)
os.makedirs(TRAINED_MODELS_PATH, exist_ok=True)

# Поддерживаемые форматы файлов
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
SUPPORTED_PDF_FORMAT = [".pdf"]
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS + SUPPORTED_PDF_FORMAT 

# Инициализируем SettingsManager для загрузки настроек
# Эта строка будет добавлена в импорты settings_manager, так что
# к этому моменту модуль settings_manager уже загружен
# GOOGLE_API_KEY будет загружен из settings.ini вместо файла

# NEW: Список типов сущностей для разметки IOB2 (ключи из Gemini JSON)
# Эти имена должны соответствовать ключам, которые Gemini возвращает в JSON.
# Если Gemini возвращает вложенные структуры (например, для позиций), 
# нужно будет решить, как их размечать (например, item_description, item_quantity и т.д. как отдельные метки)
DEFAULT_ENTITY_TYPES = [
    # Поля поставщика
    "COMPANY", "SELLER_INN", "SELLER_KPP", "SELLER_ADDRESS",
    
    # Поля покупателя
    "BUYER_NAME", "BUYER_INN", "BUYER_KPP", "BUYER_ADDRESS",
    
    # Поля счета
    "INVOICE_ID", "DATE", "PAYMENT_DUE_DATE",
    
    # Суммы и НДС
    "SUBTOTAL", "VAT_RATE", "VAT_AMOUNT", "TOTAL",
    
    # Категории и описания
    "CATEGORY", "ITEMS",
    
    # Валюта и дополнительно
    "CURRENCY", "NOTE",
    
    # Банковские реквизиты
    "BANK_NAME", "BANK_BIC", "BANK_ACCOUNT", "BANK_CORR_ACCOUNT",
    
    # Позиции счета (для табличной части)
    "ITEM_DESCRIPTION", "ITEM_QUANTITY", "ITEM_UNIT_PRICE", "ITEM_TOTAL_PRICE"
]

# NEW: Специальные метки для токенизатора и модели
LABEL_OTHER = "O" # Метка для токенов вне сущностей
# Метки B- и I- будут добавляться к DEFAULT_ENTITY_TYPES при формировании списка всех меток

# (Опционально) Параметры для Hugging Face LayoutLM Tokenizer/Processor, если нужны специфичные
# LAYOUTLM_TOKENIZER_NAME = "microsoft/layoutlmv3-base" # или layoutlm-base-uncased и т.д.

# NEW: Имя/путь к токенизатору LayoutLM, используемому для подготовки обучающих данных
# Это должен быть токенизатор, совместимый с моделью, которую планируется обучать.
# Например, если обучаем "microsoft/layoutlmv3-base", то и токенизатор должен быть от него.
LAYOUTLM_TOKENIZER_FOR_TRAINING = "microsoft/layoutlm-base-uncased"

# NEW: Соотношение для разделения датасета на обучающую и валидационную выборки
TRAIN_EVAL_SPLIT_RATIO = 0.1 # 10% на валидацию

# NEW: Параметры для TrainingArguments по умолчанию
TRAINING_LOGGING_STEPS = 50
TRAINING_SAVE_TOTAL_LIMIT = 2 # Сколько лучших чекпоинтов хранить

# NEW: Дополнительные настройки для обучения
LAYOUTLM_MODEL_ID_FOR_TRAINING = "microsoft/layoutlmv3-base" # Базовая модель для дообучения
LAYOUTLM_MAX_SEQ_LENGTH = 512 # Максимальная длина последовательности для LayoutLM

DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_TRAIN_BATCH_SIZE = 4 # Подбирается в зависимости от GPU RAM
DEFAULT_LEARNING_RATE = 5e-5 # 0.00005

# Список типов сущностей, которые Gemini будет пытаться извлечь и на которых будет обучаться LayoutLM
# Эти имена должны совпадать с ключами в GEMINI_ANNOTATION_PROMPT_DEFAULT
DEFAULT_ENTITY_TYPES = [
    # Поля поставщика
    "COMPANY", "SELLER_INN", "SELLER_KPP", "SELLER_ADDRESS",
    
    # Поля покупателя
    "BUYER_NAME", "BUYER_INN", "BUYER_KPP", "BUYER_ADDRESS",
    
    # Поля счета
    "INVOICE_ID", "DATE", "PAYMENT_DUE_DATE",
    
    # Суммы и НДС
    "SUBTOTAL", "VAT_RATE", "VAT_AMOUNT", "TOTAL",
    
    # Категории и описания
    "CATEGORY", "ITEMS",
    
    # Валюта и дополнительно
    "CURRENCY", "NOTE",
    
    # Банковские реквизиты
    "BANK_NAME", "BANK_BIC", "BANK_ACCOUNT", "BANK_CORR_ACCOUNT",
    
    # Позиции счета (для табличной части)
    "ITEM_DESCRIPTION", "ITEM_QUANTITY", "ITEM_UNIT_PRICE", "ITEM_TOTAL_PRICE"
]
LABEL_OTHER = "O" # Метка для токенов, не относящихся к сущностям

MATCHING_SIMILARITY_THRESHOLD = 0.7 # Порог схожести для сопоставления Gemini и OCR

# --- Цвета приложения (NEW) ---
APP_COLORS = {
    "primary": QColor("#3498db"),      # Синий
    "secondary": QColor("#2ecc71"),    # Зеленый
    "background_light": QColor("#ecf0f1"), # Светло-серый
    "background_dark": QColor("#2c3e50"),  # Темно-синий
    "text_light": QColor("#ffffff"),       # Белый
    "text_dark": QColor("#34495e"),        # Темно-серый
    "error": QColor("#e74c3c"),          # Красный
    "warning": QColor("#f39c12"),        # Оранжевый
    "success": QColor("#27ae60"),        # Темно-зеленый
    "info": QColor("#3498db")            # Синий (как primary)
}

# DPI для конвертации PDF при подготовке данных для Gemini
GEMINI_PDF_DPI = 200
# DPI для отображения PDF в UI (может быть ниже для скорости)
UI_PDF_DISPLAY_DPI = 150

# NEW: Настройки отображения
MAX_TABLE_ROWS_DISPLAY = 100 # Макс. строк в таблице результатов для одного файла
DEFAULT_IMAGE_DISPLAY_WIDTH = 600 # Ширина области просмотра изображения

# Проверка и создание необходимых директорий
def create_dirs():
    paths_to_create = [
        TEMP_PATH,
        TRAINING_DATASETS_PATH, TRAINED_MODELS_PATH, # NEW: Добавлено
        MODELS_PATH
    ]
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)

create_dirs()

# Удалено: Загрузка API ключа Google Gemini из файла - теперь используется система зашифрованного хранения
GOOGLE_API_KEY = None  # Будет загружен через новую систему безопасности в Config._check_api_keys()

# Вывод состояния оффлайн-режима и ключа Gemini при запуске
print(f"Оффлайн-режим: {OFFLINE_MODE}")
print(f"Poppler Path: {POPPLER_PATH}")  # Обновлено для всегда показывать путь
print(f"Tesseract Path: {TESSERACT_PATH if TESSERACT_PATH else 'Не указан (используется системный)'}")
print(f"Ключ Google API: {'Загружен' if GOOGLE_API_KEY else 'Не загружен или отсутствует'}")
if not GENAI_AVAILABLE and not GOOGLE_API_KEY: # Убрано, так как GENAI_AVAILABLE здесь еще не определен
     pass # print("Для использования Gemini API установите 'google-generativeai' и укажите API ключ.")

# ... (остальной код config.py) ... 

class Config:
    def __init__(self):
        # Базовые пути
        self.BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.DATA_PATH = os.path.join(self.BASE_PATH, 'data')
        self.MODELS_PATH = os.path.join(self.DATA_PATH, 'models')
        self.PROMPTS_PATH = os.path.join(self.DATA_PATH, 'prompts')
        self.TEMP_PATH = os.path.join(self.DATA_PATH, 'temp')
        self.TRAINING_DATASETS_PATH = os.path.join(self.DATA_PATH, 'training_datasets')
        self.TRAINED_MODELS_PATH = os.path.join(self.DATA_PATH, 'trained_models')
        self.TEST_INVOICES_PATH = os.path.join(self.DATA_PATH, 'test_invoices')
        
        # Пути к внешним инструментам
        self.TESSERACT_PATH = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
        self.POPPLER_PATH = os.path.join(self.DATA_PATH, 'poppler', 'bin')
        
        # Настройки моделей
        self.LAYOUTLM_MODEL_PATH = os.path.join(self.MODELS_PATH, 'layoutlm')
        self.DONUT_MODEL_PATH = os.path.join(self.MODELS_PATH, 'donut')
        
        # Загружаем настройки из файла
        self.settings_file = os.path.join(self.DATA_PATH, 'settings.ini')
        self._load_settings()
        
        # Стандартные типы сущностей для извлечения
        self.DEFAULT_ENTITY_TYPES = [
            "seller_name",
            "seller_inn",
            "buyer_name", 
            "buyer_inn",
            "invoice_number",
            "invoice_date",
            "total_amount",
            "vat",
            "category", 
            "comment"
        ]
        
        # Проверяем и создаем необходимые директории
        self._ensure_directories()
        
        # Проверяем наличие необходимых файлов и путей
        self._check_requirements()
        
    def _load_settings(self):
        """Загружает настройки из файла settings.ini"""
        self.config = configparser.ConfigParser()
        
        if os.path.exists(self.settings_file):
            self.config.read(self.settings_file, encoding='utf-8')
            print(f"Настройки загружены из {self.settings_file}")
        else:
            print(f"Файл настроек не найден: {self.settings_file}")
            self._create_default_settings()
            
    def _create_default_settings(self):
        """Создает файл настроек с значениями по умолчанию"""
        self.config['Paths'] = {
            'TesseractPath': self.TESSERACT_PATH,
            'PopplerPath': self.POPPLER_PATH
        }
        
        self.config['Processing'] = {
            'OfflineMode': 'True',
            'UseCache': 'True'
        }
        
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            self.config.write(f)
            
        print(f"Создан файл настроек по умолчанию: {self.settings_file}")
        
    def _ensure_directories(self):
        """Проверяет и создает необходимые директории"""
        directories = [
            self.DATA_PATH,
            self.MODELS_PATH,
            self.PROMPTS_PATH,
            self.TEMP_PATH,
            self.TRAINING_DATASETS_PATH,
            self.TRAINED_MODELS_PATH,
            self.TEST_INVOICES_PATH,
            os.path.join(self.DATA_PATH, 'poppler', 'bin')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _check_requirements(self):
        """Проверяет наличие необходимых файлов и путей"""
        # Проверяем Tesseract
        tesseract_path = self.config.get('Paths', 'TesseractPath', fallback=self.TESSERACT_PATH)
        if os.path.exists(tesseract_path):
            print(f"Установлен путь Tesseract: {tesseract_path}")
        else:
            print(f"ВНИМАНИЕ: Tesseract не найден по пути: {tesseract_path}")
            
        # Проверяем Poppler
        poppler_path = self.config.get('Paths', 'PopplerPath', fallback=self.POPPLER_PATH)
        if os.path.exists(poppler_path):
            print(f"Установлен путь Poppler: {poppler_path}")
        else:
            print(f"ВНИМАНИЕ: Poppler не найден по пути: {poppler_path}")
            
        # Проверяем оффлайн режим
        offline_mode = self.config.getboolean('Processing', 'OfflineMode', fallback=True)
        print(f"Оффлайн-режим: {offline_mode}")
        
        # Проверяем API ключи
        self._check_api_keys()
        
    def _check_api_keys(self):
        """Проверяет наличие API ключей через новую систему безопасности"""
        # Используем новую систему управления секретами
        try:
            from config.secrets import SecretsManager
            secrets_manager = SecretsManager()
            
            # Проверяем Google Gemini API ключ
            gemini_key = secrets_manager.get_secret("GOOGLE_API_KEY")
            if gemini_key:
                os.environ['GOOGLE_API_KEY'] = gemini_key
                print("Google Gemini API ключ: Загружен из зашифрованного хранилища")
            else:
                print("Google Gemini API ключ: Не настроен (используйте вкладку '🔐 API Ключи' в настройках)")
            
            # Проверяем Hugging Face токен
            hf_token = secrets_manager.get_secret("HF_TOKEN")
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                print("Hugging Face токен: Загружен из зашифрованного хранилища")
            else:
                print("Hugging Face токен: Не настроен (используйте вкладку '🔐 API Ключи' в настройках)")
                
        except ImportError:
            print("Система безопасности недоступна. Используется fallback на settings_manager.")
            # Fallback на старую систему через settings_manager
            try:
                from .settings_manager import settings_manager
                
                gemini_key = settings_manager.get_gemini_api_key()
                if gemini_key:
                    os.environ['GOOGLE_API_KEY'] = gemini_key
                    print("Google Gemini API ключ: Загружен из настроек")
                else:
                    print("Google Gemini API ключ: Не настроен")
                
                hf_token = settings_manager.get_huggingface_token()
                if hf_token:
                    os.environ['HF_TOKEN'] = hf_token
                    print("Hugging Face токен: Загружен из настроек")
                else:
                    print("Hugging Face токен: Не настроен")
                    
            except Exception as e:
                print(f"Ошибка при загрузке API ключей: {e}")
        except Exception as e:
            print(f"Ошибка при работе с системой безопасности: {e}")
    
    def get_gemini_api_key(self):
        """Получает Google Gemini API ключ из системы безопасности или настроек"""
        try:
            from config.secrets import SecretsManager
            secrets_manager = SecretsManager()
            return secrets_manager.get_secret("GOOGLE_API_KEY")
        except ImportError:
            # Fallback на старую систему
            try:
                from .settings_manager import settings_manager
                return settings_manager.get_gemini_api_key()
            except Exception:
                return os.environ.get('GOOGLE_API_KEY', None)
        except Exception:
            return os.environ.get('GOOGLE_API_KEY', None)

# Настраиваемые поля для таблицы результатов
DEFAULT_TABLE_FIELDS = [
    {"id": "Sender", "name": "Поставщик", "visible": True, "order": 0},
    {"id": "№ Invoice", "name": "№ Счета", "visible": True, "order": 1},
    {"id": "Invoice Date", "name": "Дата счета", "visible": True, "order": 2},
    {"id": "Category", "name": "Категория", "visible": True, "order": 3},
    {"id": "Description", "name": "Товары", "visible": True, "order": 4},
    {"id": "Amount (0% VAT)", "name": "Сумма без НДС", "visible": True, "order": 5},
    {"id": "VAT %", "name": "НДС %", "visible": True, "order": 6},
    {"id": "Total", "name": "Сумма с НДС", "visible": True, "order": 7},
    {"id": "Currency", "name": "Валюта", "visible": True, "order": 8},
    {"id": "Note", "name": "Комментарии", "visible": True, "order": 9}
]

# Маппинг между русскими и английскими названиями полей Gemini
FIELD_MAPPING = {
    "Поставщик": "Sender",
    "Получатель": "Receiver",
    "Номер счета": "№ Invoice", 
    "№ счета": "№ Invoice",
    "Дата счета": "Invoice Date",
    "Сумма без НДС": "Amount (0% VAT)",
    "Сумма с НДС": "Total",
    "НДС %": "VAT %",
    "Ставка НДС": "VAT %",
    "Валюта": "Currency",
    "Описание товаров": "Description",
    "Товары": "Description",
    "Товар": "Description",
    "Услуги": "Description",
    "Категория": "Category",
    "Комментарий": "Note",
    "Примечание": "Note",
    "Комментарии": "Note"
} 

# Пути к директориям
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')
PROMPTS_DIR = os.path.join(DATA_DIR, 'prompts')
TRAINING_DATASETS_FOLDER = os.path.join(DATA_DIR, 'training_datasets')

# Настройки OCR
OCR_LANGUAGE = 'eng+rus'

# Настройки Gemini
GOOGLE_API_KEY = None
GEMINI_MODEL_ID = 'models/gemini-2.0-flash'
GEMINI_AVAILABLE_MODELS = [
    'models/gemini-2.0-flash',
    'models/gemini-2.0-pro',
    'models/gemini-2.0-vision'
]

# Настройки LayoutLM
ACTIVE_LAYOUTLM_MODEL_TYPE = 'microsoft/layoutlmv3-base'

# Настройки таблицы полей
DEFAULT_ENTITY_TYPES = [
    'invoice_number',
    'invoice_date',
    'total_amount',
    'seller_name',
    'seller_address',
    'seller_inn',
    'seller_kpp',
    'buyer_name',
    'buyer_address',
    'buyer_inn',
    'buyer_kpp'
]

# Настройки для обучения
TRAINING_BATCH_SIZE = 4
TRAINING_EPOCHS = 3
TRAINING_LEARNING_RATE = 2e-5
TRAINING_WEIGHT_DECAY = 0.01
TRAINING_WARMUP_RATIO = 0.1
TRAINING_EVAL_STEPS = 100
TRAINING_SAVE_STEPS = 500
TRAINING_LOGGING_STEPS = 10 
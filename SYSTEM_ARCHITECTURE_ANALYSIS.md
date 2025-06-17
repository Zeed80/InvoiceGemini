# Анализ архитектуры InvoiceGemini

## Обзор системы

InvoiceGemini - это десктопное приложение на PyQt6 для автоматического извлечения данных из счетов-фактур с использованием различных ML/LLM моделей и расширяемой системы плагинов.

## Основной функционал программы

### 1. Обработка документов

#### Поддерживаемые форматы:
- **Изображения**: PNG, JPG, JPEG
- **PDF документы**: Автоматическая конвертация в изображения

#### Процесс обработки:
1. **Загрузка файла** → Выбор изображения или PDF
2. **OCR извлечение** → Tesseract OCR извлекает текст и координаты (bounding boxes)
3. **Выбор модели** → Пользователь выбирает ML/LLM модель
4. **Обработка** → Модель анализирует текст и изображение
5. **Нормализация** → Приведение данных к единому формату
6. **Отображение** → Результаты в табличном виде

### 2. Поддерживаемые модели

#### Локальные ML модели:
- **LayoutLMv3** - Transformer модель от Microsoft для понимания документов
  - Поддержка custom моделей (обученных пользователем)
  - Работает с bounding boxes из OCR
  - Token classification для извлечения полей

- **Donut** - Document Understanding Transformer
  - End-to-end модель без OCR
  - Прямое извлечение структурированных данных из изображений
  - Поддержка кастомных промптов

#### Облачные API:
- **Google Gemini** - Мультимодальная модель
  - Поддержка Gemini 2.0 Flash, 1.5 Pro/Flash
  - Vision capabilities для анализа изображений
  - Настраиваемые промпты

#### LLM плагины:
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude 3.5, Claude 3)
- **Mistral AI**
- **DeepSeek**
- **xAI (Grok)**
- **Ollama** (локальные LLM: Llama, Mistral, Qwen)

### 3. Извлекаемые поля

Стандартные поля счета-фактуры:
- Данные поставщика (название, ИНН, КПП, адрес)
- Данные покупателя (название, ИНН, КПП, адрес)
- Номер и дата счета
- Список товаров/услуг
- Суммы (без НДС, НДС, итого)
- Банковские реквизиты
- Дополнительная информация

## Система плагинов

### 1. Архитектура плагинов

```
BasePlugin (ABC)
├── ProcessorPlugin      # Обработка данных
│   └── LLMPlugin       # Специализация для LLM
├── ViewerPlugin        # Визуализация данных
├── ExporterPlugin      # Экспорт в различные форматы
├── ImporterPlugin      # Импорт из источников
├── ValidatorPlugin     # Валидация данных
├── TransformerPlugin   # Трансформация форматов
├── IntegrationPlugin   # Интеграция с внешними системами
├── WorkflowPlugin      # Автоматизация процессов
└── NotificationPlugin  # Уведомления и алерты
```

### 2. Типы плагинов

#### PluginType (Enum):
- `LLM` - Языковые модели для извлечения данных
- `PROCESSOR` - Обработка документов
- `VIEWER` - Просмотр и редактирование
- `EXPORTER` - Экспорт данных
- `IMPORTER` - Импорт данных
- `VALIDATOR` - Валидация
- `TRANSFORMER` - Трансформация данных
- `INTEGRATION` - Интеграции
- `WORKFLOW` - Рабочие процессы
- `NOTIFICATION` - Уведомления

#### PluginCapability (Enum):
- `VISION` - Работа с изображениями
- `TEXT` - Работа с текстом
- `TRAINING` - Поддержка обучения
- `STREAMING` - Потоковая обработка
- `BATCH` - Пакетная обработка
- `ASYNC` - Асинхронная работа
- `REALTIME` - Реальное время
- `API` - API интеграции
- `DATABASE` - Работа с БД
- `CLOUD` - Облачные возможности

### 3. LLM плагины

#### Базовый класс BaseLLMPlugin:
```python
class BaseLLMPlugin(BaseProcessor):
    def __init__(self, provider_name, model_name, api_key)
    def load_model() -> bool
    def generate_response(prompt, image_path, image_context) -> str
    def process_image(image_path, ocr_lang, custom_prompt)
    def create_invoice_prompt(custom_prompt, include_context_fields) -> str
    def parse_llm_response(response) -> Dict[str, Any]
```

#### Поддерживаемые провайдеры:
```python
LLM_PROVIDERS = {
    "openai": LLMProviderConfig(...),      # ChatGPT
    "anthropic": LLMProviderConfig(...),   # Claude
    "google": LLMProviderConfig(...),      # Gemini
    "mistral": LLMProviderConfig(...),     # Mistral AI
    "deepseek": LLMProviderConfig(...),    # DeepSeek
    "xai": LLMProviderConfig(...),         # Grok
    "ollama": LLMProviderConfig(...)       # Локальные модели
}
```

### 4. Универсальный менеджер плагинов

**UniversalPluginManager** обеспечивает:
- Автоматическое обнаружение плагинов
- Загрузка и инициализация
- Управление жизненным циклом
- Обработка зависимостей
- Приоритеты выполнения

**Функции:**
- `discover_plugins()` - Поиск плагинов в директориях
- `register_plugin()` - Регистрация плагина
- `get_plugins_by_type()` - Получение плагинов по типу
- `execute_plugin()` - Выполнение плагина
- `get_plugin_metadata()` - Метаданные плагина

### 5. Примеры плагинов

#### Плагины обработки:
- **OCR Enhancement** - Улучшение качества OCR
- **Image Preprocessing** - Предобработка изображений
- **Multi-page Processing** - Обработка многостраничных документов

#### Плагины экспорта:
- **Excel Exporter** - Экспорт в Excel с форматированием
- **PDF Generator** - Создание PDF отчетов
- **API Exporter** - Отправка данных через API

#### Плагины интеграции:
- **1C Integration** - Интеграция с 1С
- **SAP Connector** - Подключение к SAP
- **Database Sync** - Синхронизация с БД

#### Плагины валидации:
- **Tax Validator** - Проверка налоговых данных
- **Format Validator** - Проверка форматов полей
- **Business Rules** - Бизнес-правила

## Процесс обработки счета

### 1. Инициализация
```python
ModelManager() → Управление моделями
├── LayoutLMProcessor
├── DonutProcessor
├── GeminiProcessor
└── LLM Plugin Manager
```

### 2. OCR обработка
```python
OCRProcessor.process_image()
├── Tesseract OCR
├── Extract text
├── Extract bounding boxes
└── Return (text, words_with_coordinates)
```

### 3. Обработка моделью
```python
Selected Model Processing:
├── LayoutLM: Token classification с boxes
├── Donut: End-to-end extraction
├── Gemini: Multimodal analysis
└── LLM Plugin: Prompt-based extraction
```

### 4. Нормализация данных
```python
Data Normalization:
├── Parse JSON response
├── Clean numeric values
├── Format dates
├── Validate fields
└── Map to table structure
```

### 5. Отображение результатов
```python
Results Display:
├── Table widget with configured columns
├── Field validation indicators
├── Export options
└── Preview capabilities
```

## Ключевые компоненты

### 1. ModelManager
- Централизованное управление ML/LLM моделями
- Кэширование загруженных моделей
- Контроль памяти через MemoryManager
- Поддержка custom моделей

### 2. ProcessingEngine
- Координация процесса обработки
- Выбор подходящего процессора
- Обработка ошибок
- Управление потоками

### 3. PluginManager
- Обнаружение и загрузка плагинов
- Управление зависимостями
- Выполнение плагинов
- API для расширения

### 4. UI Components
- FileSelectorWidget - выбор файлов
- ModelSelectorWidget - выбор моделей
- ResultsViewerWidget - отображение результатов
- ProcessingController - контроль обработки

## Расширяемость системы

### 1. Добавление новой ML модели:
1. Наследовать от `BaseProcessor`
2. Реализовать методы `load_model()` и `process_image()`
3. Зарегистрировать в `ModelManager`

### 2. Добавление LLM провайдера:
1. Добавить конфигурацию в `LLM_PROVIDERS`
2. Создать класс плагина, наследующий `BaseLLMPlugin`
3. Реализовать методы `load_model()` и `generate_response()`

### 3. Создание custom плагина:
1. Выбрать базовый класс (ViewerPlugin, ExporterPlugin и т.д.)
2. Реализовать абстрактные методы
3. Добавить метаданные плагина
4. Поместить в директорию плагинов

## Преимущества архитектуры

1. **Модульность** - Четкое разделение компонентов
2. **Расширяемость** - Легко добавлять новые модели и функции
3. **Гибкость** - Поддержка различных типов моделей
4. **Масштабируемость** - От локальных до облачных решений
5. **Универсальность** - Обработка различных форматов документов

## Области применения

1. **Автоматизация бухгалтерии** - Обработка входящих счетов
2. **Документооборот** - Извлечение данных из документов
3. **Интеграция с ERP** - Автоматический ввод данных
4. **Аналитика** - Сбор данных для анализа
5. **Архивирование** - Структурированное хранение 
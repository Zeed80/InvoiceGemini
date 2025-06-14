# 🧾 InvoiceGemini
Проект целиком и полностью создан в Cursor AI с автоматическим подтверждением всех изменений.
**AI-Powered Invoice Data Extraction Desktop Application**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

> **Автоматизированное извлечение данных из счетов с использованием передовых ML-моделей и LLM**

## 🌟 Особенности

### 🤖 Множественные AI-модели
- **LayoutLMv3** - Специализированная модель для понимания структуры документов
- **Donut** - Transformer-модель для OCR без предварительной обработки
- **Google Gemini 2.0 Flash** - Мощная LLM для интеллектуального анализа
- **Расширяемая система плагинов** для интеграции новых моделей

### 📊 Интеллектуальное извлечение данных
- Автоматическое распознавание 12+ полей счетов
- Поддержка русского и английского языков
- Обработка PDF и изображений (PNG, JPG, JPEG)
- Экспорт в JSON, CSV, Excel форматы

### 🎯 Профессиональный интерфейс
- Современный PyQt6 интерфейс
- Drag & Drop загрузка файлов
- Предварительный просмотр документов
- Редактирование извлеченных данных
- Система шаблонов для экспорта

### 🔧 Система обучения
- Встроенные инструменты для создания датасетов
- Автоматическая разметка с помощью OCR + LLM
- Дообучение моделей на пользовательских данных
- Анализ качества и сложности задач

## 🚀 Быстрый старт

### Системные требования
- Python 3.8+
- 8GB+ RAM (рекомендуется 16GB)
- CUDA-совместимая GPU (опционально, для ускорения)
- Windows 10/11, Linux, или macOS

### Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/Zeed80/InvoiceGemini.git
cd InvoiceGemini
```

2. **Создайте виртуальное окружение:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

4. **Настройте API ключи:**
```bash
cp .env.example .env
# Отредактируйте .env файл, добавив ваши API ключи
```

5. **Запустите приложение:**
```bash
python main.py
```

## 📖 Использование

### Базовая обработка документов

1. **Загрузите документ** - перетащите файл в окно приложения
2. **Выберите модель** - LayoutLMv3, Donut, или Gemini
3. **Запустите обработку** - нажмите "Обработать"
4. **Проверьте результаты** - отредактируйте при необходимости
5. **Экспортируйте данные** - выберите формат и сохраните

### Обучение собственной модели

1. Подготовьте датасет счетов
2. Используйте встроенные инструменты разметки
3. Запустите процесс обучения через интерфейс
4. Протестируйте обученную модель

## 🏗️ Архитектура

```
InvoiceGemini/
├── app/                    # Основное приложение
│   ├── gui/               # PyQt6 интерфейс
│   ├── processing/        # Движок обработки
│   ├── plugins/           # Система плагинов LLM
│   ├── training/          # Модули обучения
│   └── utils/             # Утилиты
├── data/                  # Данные и конфигурация
│   ├── models/           # Загруженные модели
│   ├── templates/        # Шаблоны экспорта
│   └── prompts/          # Промпты для LLM
├── resources/            # Ресурсы интерфейса
└── test_data/           # Тестовые данные
```

## 🔌 Система плагинов

InvoiceGemini поддерживает расширяемую архитектуру плагинов для интеграции новых LLM:

```python
from app.plugins.base_llm_plugin import BaseLLMPlugin

class CustomLLMPlugin(BaseLLMPlugin):
    def extract_invoice_data(self, image_path, prompt=None):
        # Ваша реализация
        return extracted_data
```

### Поддерживаемые плагины:
- **GeminiPlugin** - Google Gemini 2.0 Flash
- **OpenAIPlugin** - GPT-4 Vision (планируется)
- **AnthropicPlugin** - Claude Vision (планируется)

## 📊 Поддерживаемые поля

| Поле | Описание | Языки |
|------|----------|-------|
| Номер счета | Уникальный номер документа | RU/EN |
| Дата счета | Дата выставления | RU/EN |
| Название компании | Поставщик услуг | RU/EN |
| ИНН компании | Налоговый номер | RU |
| Сумма без НДС | Сумма до налогов | RU/EN |
| НДС % | Процент налога | RU/EN |
| Общая сумма | Итоговая сумма | RU/EN |
| Товары/услуги | Список позиций | RU/EN |

## 🛠️ Разработка

### Настройка среды разработки

```bash
# Установка в режиме разработки
pip install -e .

# Запуск тестов
python -m pytest tests/

# Проверка кода
flake8 app/
black app/
```

### Структура кода

- **Следуйте PEP 8** для стиля кода
- **Используйте type hints** для всех функций
- **Документируйте** публичные методы
- **Покрывайте тестами** новую функциональность

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за предобученные модели
- [Google](https://ai.google.dev/) за Gemini API
- [PyQt](https://www.riverbankcomputing.com/software/pyqt/) за GUI фреймворк
- Сообщество разработчиков за вклад и обратную связь

## 📞 Поддержка

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/InvoiceGemini/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/InvoiceGemini/discussions)

---

<div align="center">



</div> 

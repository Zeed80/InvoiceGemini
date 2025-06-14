# 🤝 Руководство по участию в проекте InvoiceGemini

Спасибо за интерес к проекту InvoiceGemini! Мы приветствуем любой вклад в развитие проекта.

## 🚀 Как начать

### Настройка среды разработки

1. **Форкните репозиторий** на GitHub
2. **Клонируйте ваш форк:**
   ```bash
   git clone https://github.com/yourusername/InvoiceGemini.git
   cd InvoiceGemini
   ```

3. **Создайте виртуальное окружение:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

4. **Установите зависимости для разработки:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # если есть
   ```

5. **Настройте pre-commit хуки:**
   ```bash
   pre-commit install
   ```

## 📋 Типы вклада

### 🐛 Сообщения об ошибках
- Используйте шаблон issue для багов
- Включите подробное описание проблемы
- Приложите скриншоты если возможно
- Укажите версию Python и ОС

### 💡 Предложения функций
- Опишите желаемую функциональность
- Объясните, почему это будет полезно
- Предложите возможную реализацию

### 🔧 Исправления кода
- Создайте ветку для каждого исправления
- Следуйте стандартам кодирования проекта
- Добавьте тесты для новой функциональности
- Обновите документацию при необходимости

## 📝 Стандарты кодирования

### Python код
- Следуйте **PEP 8**
- Используйте **type hints** для всех функций
- Максимальная длина строки: **88 символов** (Black formatter)
- Используйте **docstrings** для всех публичных методов

### Пример хорошего кода:
```python
from typing import Optional, Dict, Any
from pathlib import Path

def extract_invoice_data(
    image_path: Path, 
    model_type: str = "layoutlm"
) -> Optional[Dict[str, Any]]:
    """
    Извлекает данные из изображения счета.
    
    Args:
        image_path: Путь к изображению
        model_type: Тип модели для обработки
        
    Returns:
        Словарь с извлеченными данными или None при ошибке
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если неподдерживаемый тип модели
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    
    # Реализация...
    return extracted_data
```

### PyQt6 код
- Используйте **сигналы и слоты** для связи компонентов
- Выносите длительные операции в **QThread**
- Используйте **self.tr()** для всех пользовательских строк
- Освобождайте ресурсы с помощью **deleteLater()**

## 🧪 Тестирование

### Запуск тестов
```bash
# Все тесты
python -m pytest

# Конкретный модуль
python -m pytest tests/test_processing.py

# С покрытием кода
python -m pytest --cov=app tests/
```

### Написание тестов
- Создавайте тесты для каждой новой функции
- Используйте **pytest** фреймворк
- Мокайте внешние зависимости (API, файлы)
- Тестируйте граничные случаи

## 🌐 Локализация

### Добавление переводов
1. Обновите файлы `.ts` в `resources/translations/`
2. Используйте Qt Linguist для перевода
3. Сгенерируйте `.qm` файлы
4. Протестируйте интерфейс на разных языках

### Правила локализации
- Все пользовательские строки через `self.tr()`
- Учитывайте длину переведенного текста
- Используйте плейсхолдеры для динамических значений

## 🔌 Разработка плагинов

### Создание LLM плагина
```python
from app.plugins.base_llm_plugin import BaseLLMPlugin

class MyLLMPlugin(BaseLLMPlugin):
    def __init__(self, ocr_service=None, logger=None):
        super().__init__(ocr_service=ocr_service, logger=logger)
        self.model_name = "my-model"
    
    def load_model(self) -> bool:
        # Загрузка модели
        return True
    
    def extract_invoice_data(self, image_path: str, prompt: str = None) -> dict:
        # Извлечение данных
        return {}
    
    def cleanup(self):
        # Очистка ресурсов
        pass
```

## 📦 Процесс Pull Request

### Перед отправкой PR
1. **Синхронизируйте с upstream:**
   ```bash
   git remote add upstream https://github.com/original/InvoiceGemini.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Создайте ветку для функции:**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Проверьте код:**
   ```bash
   # Форматирование
   black app/
   
   # Линтинг
   flake8 app/
   
   # Тесты
   python -m pytest
   ```

### Требования к PR
- ✅ Описательное название и описание
- ✅ Связанные issues (если есть)
- ✅ Все тесты проходят
- ✅ Код отформатирован
- ✅ Документация обновлена
- ✅ Нет конфликтов с main веткой

### Шаблон описания PR
```markdown
## Описание
Краткое описание изменений

## Тип изменения
- [ ] Исправление бага
- [ ] Новая функция
- [ ] Критическое изменение
- [ ] Обновление документации

## Тестирование
- [ ] Добавлены новые тесты
- [ ] Все существующие тесты проходят
- [ ] Протестировано вручную

## Скриншоты (если применимо)
```

## 🏷️ Соглашения о коммитах

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: добавить поддержку GPT-4 Vision
fix: исправить ошибку загрузки LayoutLM
docs: обновить README с примерами
style: форматирование кода
refactor: рефакторинг системы плагинов
test: добавить тесты для OCR модуля
chore: обновить зависимости
```

## 🎯 Приоритетные области

### Высокий приоритет
- 🔧 Новые LLM плагины (OpenAI, Anthropic)
- 🐛 Исправления критических багов
- 🔒 Улучшения безопасности
- 📱 Оптимизация производительности

### Средний приоритет
- 🌐 Локализация на новые языки
- 📊 Поддержка новых форматов документов
- 🎨 Улучшения UI/UX
- 📚 Расширение документации

### Низкий приоритет
- 🧹 Рефакторинг кода
- 📈 Метрики и аналитика
- 🎯 Дополнительные функции

## 💬 Общение

- **GitHub Issues** - для багов и предложений
- **GitHub Discussions** - для общих вопросов
- **Email** - для приватных вопросов

## 📜 Лицензия

Участвуя в проекте, вы соглашаетесь с тем, что ваш вклад будет лицензирован под MIT License.

---

**Спасибо за ваш вклад в InvoiceGemini! 🚀** 
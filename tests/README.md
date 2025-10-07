# 🧪 Тесты InvoiceGemini

## Структура тестов

```
tests/
├── conftest.py           # Общие fixtures для всех тестов
├── unit/                 # Unit тесты
│   ├── test_backup_manager.py
│   └── test_cache_manager.py
├── integration/          # Integration тесты
└── test_data/           # Тестовые данные
```

## Запуск тестов

### Установка зависимостей для тестирования

```bash
pip install pytest pytest-cov pytest-mock
```

### Запуск всех тестов

```bash
pytest tests/
```

### Запуск unit тестов

```bash
pytest tests/unit/
```

### Запуск integration тестов

```bash
pytest tests/integration/
```

### Запуск с покрытием кода

```bash
pytest tests/ --cov=app --cov-report=html
```

Отчет будет доступен в `htmlcov/index.html`

### Запуск конкретного теста

```bash
pytest tests/unit/test_backup_manager.py
```

### Запуск с подробным выводом

```bash
pytest tests/ -v
```

## Написание тестов

### Unit тесты

Unit тесты проверяют отдельные модули изолированно:

```python
import pytest
from app.core.my_module import MyClass

class TestMyClass:
    @pytest.fixture
    def my_instance(self):
        return MyClass()
    
    def test_method(self, my_instance):
        result = my_instance.method()
        assert result == expected_value
```

### Integration тесты

Integration тесты проверяют взаимодействие компонентов:

```python
import pytest
from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin

class TestPaperlessIntegration:
    @pytest.fixture
    def plugin(self):
        return PaperlessNGXPlugin()
    
    def test_connection(self, plugin):
        # Тест подключения к Paperless
        result = plugin.test_connection()
        assert isinstance(result, bool)
```

## Fixtures

Общие fixtures определены в `conftest.py`:

- `test_data_dir` - директория с тестовыми данными
- `sample_invoice_image` - путь к тестовому изображению
- `mock_settings` - мок настроек приложения
- `mock_invoice_data` - пример данных счета
- `mock_ocr_result` - пример результата OCR

## Текущее покрытие

**Цель:** >70% code coverage

**Текущее состояние:**
- Unit тесты: 2 модуля
- Integration тесты: 0 модулей
- Coverage: ~5% (начальный этап)

## Добавление новых тестов

1. Создайте файл `test_<module_name>.py` в `tests/unit/` или `tests/integration/`
2. Импортируйте тестируемый модуль
3. Создайте класс `TestModuleName`
4. Добавьте тестовые методы с префиксом `test_`
5. Используйте fixtures из `conftest.py` или создайте свои

## Best Practices

1. **Один тест - одна проверка**: Каждый тест должен проверять одну вещь
2. **Изолируйте тесты**: Используйте моки и fixtures для изоляции
3. **Понятные имена**: `test_backup_settings_returns_bool` лучше чем `test1`
4. **Arrange-Act-Assert**: Структурируйте тесты по этому паттерну
5. **Не зависьте от порядка**: Тесты должны работать в любом порядке

## TODO

### Unit тесты (приоритет):
- [ ] test_retry_manager.py
- [ ] test_memory_manager.py
- [ ] test_smart_model_loader.py
- [ ] test_storage_adapter.py

### Integration тесты:
- [ ] test_paperless_integration.py
- [ ] test_ocr_pipeline.py
- [ ] test_model_loading.py

### E2E тесты:
- [ ] test_invoice_processing_flow.py
- [ ] test_training_workflow.py


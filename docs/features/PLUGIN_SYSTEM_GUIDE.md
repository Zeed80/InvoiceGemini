# 🔧 Руководство по универсальной системе плагинов InvoiceGemini

## 📋 Обзор

Универсальная система плагинов InvoiceGemini предоставляет расширяемую архитектуру для добавления новых функций обработки, просмотра, экспорта и валидации данных счетов.

## 🎯 Типы плагинов

### 1. **LLM плагины** (`PluginType.LLM`)
- Обработка документов с помощью языковых моделей
- Поддержка облачных и локальных моделей
- Автоматическая адаптация существующих плагинов

### 2. **Плагины просмотра** (`PluginType.VIEWER`)
- Отображение данных в различных форматах
- Встроенные: табличный просмотр, предварительный просмотр
- Интеграция с PyQt6

### 3. **Плагины экспорта** (`PluginType.EXPORTER`)
- Экспорт данных в различные форматы
- Встроенные: JSON, Excel, CSV, PDF
- Поддержка пользовательских форматов

### 4. **Плагины валидации** (`PluginType.VALIDATOR`)
- Проверка корректности данных
- Встроенные: валидация счетов, общая валидация данных
- Настраиваемые правила валидации

### 5. **Плагины обработки** (`PluginType.PROCESSOR`)
- Предварительная и постобработка данных
- Трансформация и очистка данных

### 6. **Плагины трансформации** (`PluginType.TRANSFORMER`)
- Преобразование данных между форматами
- Нормализация и стандартизация

## 🚀 Использование в приложении

### Доступ к системе плагинов

```python
from app.plugins.universal_plugin_manager import UniversalPluginManager

# Инициализация менеджера
manager = UniversalPluginManager()

# Получение статистики
stats = manager.get_plugin_statistics()
print(f"Доступно плагинов: {stats}")
```

### Экспорт данных

```python
# Экспорт в JSON
data = {"invoice_number": "INV-001", "total": 1500.50}
success = manager.export_data(data, "output.json", "json")

# Экспорт в Excel
success = manager.export_data(data, "output.xlsx", "excel")
```

### Валидация данных

```python
# Валидация счета
result = manager.validate_data(data, "invoice")
if result["errors"]:
    print("Ошибки валидации:", result["errors"])

# Общая валидация
result = manager.validate_data(data, "data")
```

### Создание просмотрщиков

```python
# Табличный просмотрщик
table_widget = manager.create_viewer(data, "table", parent_widget)

# Предварительный просмотр
preview_widget = manager.create_viewer(data, "preview", parent_widget)
```

## 🔧 Создание пользовательских плагинов

### Структура плагина

```python
from app.plugins.base_plugin import ExporterPlugin, PluginMetadata, PluginType

class MyExporterPlugin(ExporterPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="My Custom Exporter",
            version="1.0.0",
            description="Экспорт в пользовательский формат",
            author="Your Name",
            plugin_type=PluginType.EXPORTER,
            capabilities=[PluginCapability.TEXT],
            dependencies=["custom_lib"],
            config_schema={
                "required": ["output_format"],
                "optional": {"encoding": "utf-8"}
            }
        )
    
    def initialize(self) -> bool:
        # Инициализация плагина
        return True
    
    def cleanup(self):
        # Очистка ресурсов
        pass
    
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        # Логика экспорта
        return True
    
    def get_supported_formats(self) -> List[str]:
        return ["myformat"]
```

### Размещение плагинов

1. **Встроенные плагины**: `app/plugins/models/`
2. **Пользовательские плагины**: `plugins/user/`

Файлы должны заканчиваться на `_plugin.py`

## 📊 Встроенные плагины

### Экспортеры
- **JSON** (`json`) - Экспорт в JSON формат
- **Excel** (`excel`) - Экспорт в Excel файлы (требует openpyxl)
- **CSV** (`csv`) - Экспорт в CSV формат
- **PDF** (`pdf`) - Экспорт в PDF документы (требует reportlab)

### Просмотрщики
- **Таблица** (`table`) - Табличное отображение данных
- **Предпросмотр** (`preview`) - Текстовый предпросмотр

### Валидаторы
- **Счета** (`invoice`) - Валидация данных счетов
- **Данные** (`data`) - Общая валидация данных

## 🔄 Система адаптеров LLM

Для обеспечения совместимости с существующими LLM плагинами используется система адаптеров:

```python
from app.plugins.llm_plugin_adapter import adapt_all_llm_plugins

# Создание адаптеров для всех LLM плагинов
adapters = adapt_all_llm_plugins(old_plugin_manager)
```

## 🎛️ Интеграция с UI

### Новые кнопки в главном окне

1. **"Валидировать данные"** - Проверка текущих данных
2. **"Просмотр данных"** - Создание просмотрщика
3. **"Экспорт через плагины"** - Диалог экспорта

### Меню "Универсальная система плагинов"

- Статистика плагинов
- Управление плагинами
- Настройки системы

## 📈 Мониторинг и отладка

### Логирование

Система плагинов использует подробное логирование:

```
🔧 Инициализация UniversalPluginManager...
✅ Загружено плагинов: 8
📋 viewer: ['table', 'preview']
📋 exporter: ['json', 'excel', 'csv', 'pdf']
📋 validator: ['invoice', 'data']
```

### Статистика

```python
# Получение детальной статистики
stats = manager.get_statistics()

# Упрощенная статистика
simple_stats = manager.get_plugin_statistics()
```

## 🔒 Безопасность и валидация

### Валидация конфигурации

Все плагины проходят валидацию конфигурации при загрузке:

```python
def validate_plugin_config(self, config: Dict[str, Any]) -> bool:
    # Проверка обязательных полей
    # Валидация типов данных
    # Проверка зависимостей
```

### Изоляция плагинов

- Каждый плагин работает в изолированном контексте
- Автоматическая очистка ресурсов
- Обработка исключений

## 🚨 Устранение неполадок

### Частые проблемы

1. **Плагин не загружается**
   - Проверьте имя файла (должно заканчиваться на `_plugin.py`)
   - Убедитесь, что класс наследует правильный базовый класс
   - Проверьте зависимости

2. **Ошибки экспорта**
   - Убедитесь, что установлены необходимые библиотеки
   - Проверьте права доступа к файлу
   - Валидируйте входные данные

3. **Проблемы с GUI**
   - Убедитесь, что PyQt6 установлен
   - Проверьте передачу parent widget

### Отладка

```python
# Включение подробного логирования
import logging
logging.basicConfig(level=logging.DEBUG)

# Проверка состояния плагина
plugin_info = manager.get_plugin_info('exporter', 'json')
print(f"Плагин загружен: {plugin_info['is_loaded']}")
```

## 📚 Примеры

### Полный пример использования

```python
from app.plugins.universal_plugin_manager import UniversalPluginManager

# Инициализация
manager = UniversalPluginManager()

# Тестовые данные
data = {
    'invoice_number': 'INV-2024-001',
    'date': '2024-01-15',
    'total': 1500.50,
    'vendor': 'Test Company'
}

# Валидация
result = manager.validate_data(data, "invoice")
if not result.get("errors"):
    print("✅ Данные валидны")
    
    # Экспорт в разные форматы
    manager.export_data(data, "invoice.json", "json")
    manager.export_data(data, "invoice.xlsx", "excel")
    manager.export_data(data, "invoice.csv", "csv")
    
    print("✅ Данные экспортированы")
else:
    print("❌ Ошибки валидации:", result["errors"])
```

## 🔮 Будущие возможности

- Поддержка плагинов на других языках (через API)
- Веб-интерфейс для управления плагинами
- Автоматическое обновление плагинов
- Marketplace плагинов
- Расширенная система разрешений

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи в `logs/debug_session_*.log`
2. Запустите тесты: `python test_plugin_system.py`
3. Используйте отладочный режим: `python main.py --debug`

---

**Система плагинов InvoiceGemini** - мощный инструмент для расширения функциональности приложения. Следуйте этому руководству для эффективного использования и создания собственных плагинов. 
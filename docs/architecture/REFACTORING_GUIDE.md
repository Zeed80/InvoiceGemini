# 🔧 Гид по архитектурному рефакторингу InvoiceGemini

## Обзор изменений

InvoiceGemini прошел значительный архитектурный рефакторинг, основанный на современных принципах разработки ПО. Основная цель - улучшение maintainability, testability и extensibility кода.

## Основные архитектурные улучшения

### 1. Dependency Injection (DI) Container
- **Файл**: `app/core/di_container.py`
- **Назначение**: Централизованное управление зависимостями
- **Преимущества**: Упрощает тестирование, повышает гибкость

```python
# Использование DI контейнера
container = get_container()
secrets_manager = container.get('secrets_manager')
```

### 2. Компонентная архитектура UI
- **Директория**: `app/ui/components/`
- **Компоненты**:
  - `FileSelectorWidget` - выбор файлов
  - `ImageViewerWidget` - просмотр изображений
  - `ModelSelectorWidget` - выбор моделей
  - `ResultsViewerWidget` - отображение результатов
  - `ProcessingController` - контроль обработки

### 3. Structured Logging
- **Замена**: `print()` → `logging`
- **Файлы логов**: `logs/app_debug.log`
- **Форматы**: Детальный для файлов, простой для консоли

### 4. Enhanced Error Handling
- **Graceful degradation**: Автоматический fallback к оригинальному UI
- **Comprehensive logging**: Подробные логи ошибок
- **User-friendly messages**: Понятные сообщения об ошибках

## Обратная совместимость

### Переключение между интерфейсами
```python
# В app/config.py
USE_REFACTORED_UI = True   # Рефакторированный UI
USE_REFACTORED_UI = False  # Оригинальный UI
```

### Автоматический fallback
Если рефакторированный UI недоступен, приложение автоматически использует оригинальный:

```python
def choose_main_window():
    if use_refactored and REFACTORED_WINDOW_AVAILABLE:
        return MainWindowRefactored()
    else:
        return MainWindow()  # Fallback
```

## Преимущества рефакторинга

### Для разработчиков
- **Читаемость**: Код стал понятнее благодаря модульной структуре
- **Тестируемость**: DI упрощает создание unit-тестов
- **Расширяемость**: Легко добавлять новые компоненты
- **Отладка**: Подробные логи упрощают поиск проблем

### Для пользователей
- **Стабильность**: Улучшенная обработка ошибок
- **Производительность**: Оптимизированная инициализация
- **Функциональность**: Все существующие функции сохранены
- **Удобство**: Более отзывчивый интерфейс

## Миграция

### Автоматическая миграция
При первом запуске рефакторированной версии:
1. Создается резервная копия `main_original_backup.py`
2. Новый `main.py` заменяет оригинальный
3. Настройки и данные сохраняются

### Ручная миграция
```bash
# Запуск рефакторированной версии
python main.py

# Запуск оригинальной версии (если нужно)
python main_original_backup.py
```

## Новые возможности

### Logging
```python
# Вместо print()
logger = logging.getLogger(__name__)
logger.info("Сообщение для пользователя")
logger.debug("Отладочная информация")
logger.error("Ошибка", exc_info=True)
```

### DI Container
```python
# Регистрация сервиса
container.register_service('my_service', MyService(), singleton=True)

# Получение сервиса
service = container.get('my_service')
```

### Компонентная архитектура
```python
# Создание компонента
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.signals = MySignals()
        
    def emit_signal(self, data):
        self.signals.data_changed.emit(data)
```

## Рекомендации

### Для разработки
1. **Используйте DI**: Регистрируйте новые сервисы в контейнере
2. **Создавайте компоненты**: Выносите UI логику в отдельные классы
3. **Логируйте правильно**: Используйте соответствующие уровни логирования
4. **Тестируйте**: Используйте DI для создания mock-объектов

### Для пользователей
1. **Попробуйте рефакторированный UI**: Установите `USE_REFACTORED_UI = True`
2. **Проверьте логи**: Смотрите `logs/app_debug.log` при проблемах
3. **Сообщайте об ошибках**: Включайте логи в отчеты об ошибках

## Файловая структура

```
app/
├── core/
│   ├── di_container.py          # DI контейнер
│   ├── logging_config.py        # Конфигурация логирования
│   └── ...
├── ui/
│   ├── components/              # UI компоненты
│   │   ├── file_selector.py
│   │   ├── image_viewer.py
│   │   └── ...
│   └── main_window_refactored.py # Рефакторированное окно
├── config.py                    # Включает USE_REFACTORED_UI
└── ...

main.py                         # Новая точка входа
main_original_backup.py         # Резервная копия
main_refactored.py              # Оригинальная рефакторированная версия
```

## Совместимость

### Поддерживаемые функции
- ✅ Все существующие функции сохранены
- ✅ Поддержка всех форматов файлов
- ✅ Работа с моделями ИИ
- ✅ Система плагинов
- ✅ Экспорт данных

### Системные требования
- Python 3.8+
- PyQt6
- Все зависимости из requirements.txt

## Обратная связь

Если у вас возникли проблемы с рефакторированной версией:
1. Проверьте логи в `logs/app_debug.log`
2. Попробуйте переключиться на оригинальный UI (`USE_REFACTORED_UI = False`)
3. Создайте issue с подробным описанием проблемы

---

*Этот рефакторинг следует [принципам качественного кода](https://www.freecodecamp.org/news/best-practices-for-refactoring-code/) и [лучшим практикам Python-разработки](https://www.codesee.io/learning-center/python-refactoring).* 
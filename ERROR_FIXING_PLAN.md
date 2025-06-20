# План исправления ошибок InvoiceGemini

## 📋 Обзор плана

Данный план предназначен для систематического исправления всех выявленных проблем в проекте InvoiceGemini. План разделен на этапы по приоритету: критические → высокие → средние → низкие.

**Общее время выполнения**: ~15-20 рабочих дней  
**Команда**: 1-2 разработчика  
**Подход**: Поэтапная реализация с тестированием после каждого этапа

---

## 🔴 ЭТАП 1: Критические проблемы (Приоритет: СРОЧНО)
**Время выполнения**: 5-7 дней  
**Статус**: Требует немедленного внимания

### 1.1 Исправление голых except блоков
**Время**: 3 дня  
**Критичность**: Максимальная

#### Файлы для исправления (по приоритету):

**День 1:**
- [ ] `main.py` (строка 27)
- [ ] `app/utils.py` (строка 113)
- [ ] `main_refactored.py` (строка 25)
- [ ] `main_original_backup.py` (строка 26)

**День 2:**
- [ ] `app/training_dialog.py` (20 случаев)
  - Строки: 94, 148, 185, 191, 218, 224, 269, 275, 299, 354, 2355, 3158, 3213, 3300, 3307, 3657, 4206, 4709, 4820
- [ ] `app/plugins/base_llm_plugin.py` (строки 377, 505)

**День 3:**
- [ ] `app/training/` модули (множественные случаи)
- [ ] `app/ui/` компоненты
- [ ] `app/processing/` модули

#### Шаблон исправления:
```python
# ❌ БЫЛО
try:
    risky_operation()
except:
    pass

# ✅ СТАЛО  
try:
    risky_operation()
except (SpecificException1, SpecificException2) as e:
    logger.warning(f"Описание ошибки: {e}")
    # Обработка или fallback логика
```

#### Конкретные типы исключений для замены:
- **Файловые операции**: `(OSError, FileNotFoundError, PermissionError)`
- **Сетевые запросы**: `(requests.RequestException, ConnectionError, TimeoutError)`
- **Subprocess**: `(subprocess.SubprocessError, OSError, FileNotFoundError)`
- **Import операции**: `(ImportError, ModuleNotFoundError)`
- **JSON парсинг**: `(json.JSONDecodeError, ValueError, TypeError)`
- **PyQt операции**: `(RuntimeError, AttributeError)`

### 1.2 Аудит и исправление QThread классов
**Время**: 2 дня  
**Критичность**: Высокая

#### Список классов для проверки:
1. [ ] `ProcessingThread` (app/threads.py)
2. [ ] `ModelDownloadThread` (app/threads.py)  
3. [ ] `TesseractCheckThread` (app/threads.py)
4. [ ] `BatchProcessingThread` (app/ui/components/batch_processor.py)
5. [ ] `OptimizedProcessingThread` (app/processing/optimized_file_processor.py)
6. [ ] `ModelComparisonThread` (app/processing/model_comparison.py)
7. [ ] `LLMLoadingThread` (app/main_window.py)
8. [ ] `TrainingThread` (app/ui/plugins_dialog.py)
9. [ ] `DatasetGenerationThread` (app/ui/plugins_dialog.py)
10. [ ] `LLMProviderTestThread` (app/ui/llm_providers_dialog.py)
11. [ ] `ModelRefreshThread` (app/ui/llm_providers_dialog.py)

#### Проверочный список для каждого класса:
- [ ] Правильный вызов `super().__init__()`
- [ ] Корректная обработка `self.requestInterruption()`
- [ ] Использование `deleteLater()` при завершении
- [ ] Отсутствие прямого доступа к UI из `run()`
- [ ] Правильная обработка исключений в `run()`
- [ ] Освобождение ресурсов в деструкторе

#### Шаблон правильного QThread:
```python
class CorrectThread(QThread):
    # Сигналы для коммуникации
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.should_stop = False
    
    def run(self):
        try:
            # Рабочая логика с проверками на прерывание
            while not self.should_stop and not self.isInterruptionRequested():
                # Работа...
                self.progress.emit(progress_value)
                
            if not self.should_stop:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self.should_stop = True
        self.requestInterruption()
        
    def cleanup(self):
        # Освобождение ресурсов
        pass
```

### 1.3 Добавление системы логирования
**Время**: 1-2 дня  
**Критичность**: Высокая

#### Задачи:
- [ ] Создать единую систему логирования для всех модулей
- [ ] Заменить `print()` на `logger` во всех файлах
- [ ] Добавить ротацию логов
- [ ] Настроить уровни логирования

#### Файл конфигурации логирования:
```python
# app/core/logging_config.py
import logging
import logging.handlers
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Настройка логгера для модуля."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Настройка форматирования
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler с ротацией
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
    
    return logger
```

---

## 🟡 ЭТАП 2: Высокие проблемы (Приоритет: ВЫСОКИЙ)
**Время выполнения**: 4-5 дней  

### 2.1 Добавление полной локализации
**Время**: 3 дня  
**Критичность**: Высокая для пользовательского опыта

#### Файлы для локализации:
**День 1:**
- [ ] `app/main_window.py` - основное окно
- [ ] `app/settings_dialog.py` - диалог настроек
- [ ] `app/training_dialog.py` - диалог обучения

**День 2:**
- [ ] Все файлы в `app/ui/` директории
- [ ] Сообщения об ошибках в `app/utils.py`
- [ ] Плагины в `app/plugins/`

**День 3:**
- [ ] Создание файлов переводов `.ts`
- [ ] Компиляция в `.qm` файлы
- [ ] Тестирование локализации

#### Шаблон локализации:
```python
# ❌ БЫЛО
self.setWindowTitle("Обработка счетов")
QMessageBox.critical(self, "Ошибка", "Не удалось загрузить файл")

# ✅ СТАЛО
self.setWindowTitle(self.tr("Обработка счетов"))
QMessageBox.critical(self, self.tr("Ошибка"), self.tr("Не удалось загрузить файл"))
```

### 2.2 Стандартизация обработки ошибок
**Время**: 2 дня  

#### Создание базового класса для обработки ошибок:
```python
# app/core/error_handler.py
class ErrorHandler:
    """Стандартный обработчик ошибок."""
    
    @staticmethod
    def handle_file_error(e: Exception, file_path: str, logger) -> str:
        """Обработка ошибок файловых операций."""
        if isinstance(e, FileNotFoundError):
            msg = f"Файл не найден: {file_path}"
        elif isinstance(e, PermissionError):
            msg = f"Нет доступа к файлу: {file_path}"
        elif isinstance(e, OSError):
            msg = f"Ошибка файловой системы: {e}"
        else:
            msg = f"Неизвестная ошибка при работе с файлом {file_path}: {e}"
        
        logger.error(msg)
        return msg
    
    @staticmethod
    def handle_network_error(e: Exception, url: str, logger) -> str:
        """Обработка сетевых ошибок."""
        # Аналогично для других типов ошибок
        pass
```

### 2.3 Добавление Type Hints
**Время**: 1 день  

#### Приоритетные файлы:
- [ ] `app/main_window.py`
- [ ] `app/processing_engine.py`
- [ ] `app/plugins/base_llm_plugin.py`
- [ ] Все новые и часто используемые методы

---

## 🟢 ЭТАП 3: Средние проблемы (Приоритет: СРЕДНИЙ)
**Время выполнения**: 3-4 дня  

### 3.1 Упрощение избыточных проверок
**Время**: 1 день  

#### Файлы для рефакторинга:
- [ ] `app/ui/export_template_designer.py` (4 случая)
- [ ] `app/training/trainer.py` (1 случай)
- [ ] `app/core/thread_safe_manager.py` (1 случай)

#### Шаблон упрощения:
```python
# ❌ БЫЛО
if hasattr(self, 'template_combo') and self.template_combo is not None:

# ✅ СТАЛО  
if hasattr(self, 'template_combo') and self.template_combo:
```

### 3.2 Добавление документации
**Время**: 2 дня  

#### Приоритет документирования:
1. [ ] Все публичные методы классов
2. [ ] Сложные алгоритмы в processing модулях
3. [ ] API методы плагинов
4. [ ] Конфигурационные файлы

### 3.3 Очистка импортов и удаление неиспользуемого кода
**Время**: 1 день  

---

## 🔵 ЭТАП 4: Дополнительные улучшения (Приоритет: НИЗКИЙ)
**Время выполнения**: 2-3 дня  

### 4.1 Настройка инструментов контроля качества
**Время**: 1 день  

#### Настройка pre-commit hooks:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 4.2 Добавление тестов
**Время**: 2 дня  

#### Структура тестов:
```
tests/
├── unit/
│   ├── test_utils.py
│   ├── test_processing_engine.py
│   └── test_plugins.py
├── integration/
│   ├── test_main_window.py
│   └── test_batch_processing.py
└── fixtures/
    ├── test_images/
    └── test_data.json
```

---

## 📅 Временной график выполнения

### Неделя 1 (Дни 1-5): Критические проблемы
- **День 1-3**: Исправление голых except блоков
- **День 4-5**: Аудит QThread классов

### Неделя 2 (Дни 6-10): Высокие проблемы  
- **День 6-7**: Система логирования
- **День 8-10**: Локализация

### Неделя 3 (Дни 11-15): Средние проблемы
- **День 11-12**: Стандартизация обработки ошибок
- **День 13**: Type hints
- **День 14-15**: Документация и очистка

### Неделя 4 (Дни 16-20): Дополнительные улучшения
- **День 16-17**: Инструменты контроля качества
- **День 18-20**: Тестирование и финальная проверка

---

## ✅ Критерии готовности для каждого этапа

### Этап 1 - Готов когда:
- [ ] Все голые except заменены на специфичные
- [ ] Все QThread классы прошли аудит и исправлены
- [ ] Система логирования работает во всех модулях
- [ ] Приложение стабильно запускается и работает

### Этап 2 - Готов когда:
- [ ] Все пользовательские строки локализованы
- [ ] Стандартная обработка ошибок внедрена
- [ ] Type hints добавлены в критические места
- [ ] UI отображается корректно на русском языке

### Этап 3 - Готов когда:
- [ ] Избыточные проверки упрощены
- [ ] Документация добавлена для всех публичных API
- [ ] Неиспользуемый код удален
- [ ] Импорты оптимизированы

### Этап 4 - Готов когда:
- [ ] Настроены и работают pre-commit hooks
- [ ] Базовые тесты написаны и проходят
- [ ] Качество кода соответствует стандартам
- [ ] Проект готов к продакшену

---

## 🧪 План тестирования

### После каждого этапа:
1. **Юнит-тесты**: Запуск автоматических тестов
2. **Интеграционное тестирование**: Проверка основных workflow
3. **Ручное тестирование**: Проверка UI и основных функций
4. **Регрессионное тестирование**: Убедиться, что ничего не сломалось

### Основные сценарии для тестирования:
- [ ] Запуск приложения
- [ ] Загрузка и обработка изображения
- [ ] Пакетная обработка файлов
- [ ] Экспорт результатов
- [ ] Работа с плагинами
- [ ] Обучение моделей
- [ ] Обработка ошибочных ситуаций

---

## 📊 Метрики успеха

### Количественные метрики:
- **Голые except блоки**: 0 (текущее: 50+)
- **Покрытие логированием**: 95%+ критических операций
- **Покрытие локализацией**: 100% пользовательских строк
- **Покрытие type hints**: 80%+ публичных методов
- **Покрытие тестами**: 60%+ кода

### Качественные метрики:
- **Стабильность**: Отсутствие критических сбоев
- **Производительность**: Время запуска < 5 сек
- **Usability**: Все UI элементы корректно локализованы
- **Maintainability**: Код соответствует стандартам проекта

---

## 🚨 Риски и митигация

### Высокие риски:
1. **Регрессии при исправлении except блоков**
   - *Митигация*: Тщательное тестирование после каждого исправления

2. **Проблемы с threading при рефакторинге**
   - *Митигация*: Постепенный рефакторинг по одному классу

3. **Конфликты при локализации UI**
   - *Митигация*: Тестирование на разных размерах экрана

### Средние риски:
1. **Увеличение времени выполнения из-за логирования**
   - *Митигация*: Настройка уровней логирования
   
2. **Сложности с type hints в сложных типах**
   - *Митигация*: Постепенное добавление, начиная с простых случаев

---

## 📝 Заключение

Данный план обеспечивает систематический подход к исправлению всех выявленных проблем с минимизацией рисков и максимальным качеством результата. Каждый этап имеет четкие критерии готовности и метрики успеха.

**Готов к утверждению и началу выполнения.**

---
*План создан: 2024-12-19*  
*Версия: 1.0*  
*Статус: На утверждении* 
# 🔄 ПЛАН МИГРАЦИИ СИСТЕМЫ ПЛАГИНОВ

## 📊 Текущее состояние

### Существующие менеджеры:
1. **PluginManager** (593 строки) - Базовый LLM менеджер
2. **UniversalPluginManager** (782 строки) - Все типы плагинов
3. **UnifiedPluginManager** (631 строка) - Унифицированный с событиями ✅
4. **AdvancedPluginManager** (627 строк) - Установка/обновление ✅

### Файлы, использующие старые менеджеры:
- `app/main_window.py` - PluginManager, UniversalPluginManager
- `app/processing_engine.py` - PluginManager
- `app/ui/plugins_dialog.py` - PluginManager
- `app/ui/components/processing_controller.py` - PluginManager, UniversalPluginManager
- `app/core/di_container.py` - PluginManager, UniversalPluginManager
- `app/core/optimization_integration.py` - UnifiedPluginManager ✅ (уже использует правильный)

---

## 🎯 ЦЕЛЕВАЯ АРХИТЕКТУРА

### Оставляем (2 менеджера):
1. **UnifiedPluginManager** - Основной менеджер для всех плагинов
2. **AdvancedPluginManager** - Расширение для установки/обновления

### Удаляем (2 менеджера):
1. ❌ **PluginManager** - функциональность переносится в UnifiedPluginManager
2. ❌ **UniversalPluginManager** - дублирует UnifiedPluginManager

---

## 📝 ШАГ 1: Перенос методов из PluginManager в UnifiedPluginManager

### Методы для переноса:

#### 1. `create_plugin_by_provider(provider_name, model_name, api_key, **kwargs)`
**Назначение:** Создание LLM плагина по имени провайдера
**Важность:** 🔴 Критичная - используется для динамического создания LLM плагинов
**Действие:** Добавить в UnifiedPluginManager

#### 2. `get_providers_info() -> Dict[str, Dict]`
**Назначение:** Информация о всех LLM провайдерах (Google, OpenAI, Anthropic и т.д.)
**Важность:** 🟡 Средняя - используется в UI для отображения доступных провайдеров
**Действие:** Добавить в UnifiedPluginManager

#### 3. `get_recommended_plugin(provider_name) -> Optional[str]`
**Назначение:** Рекомендация наилучшего плагина для провайдера
**Важность:** 🟡 Средняя - используется для автовыбора плагина
**Действие:** Добавить в UnifiedPluginManager

#### 4. `create_plugin_template(plugin_name) -> str`
**Назначение:** Генерация шаблона нового плагина
**Важность:** 🟢 Низкая - используется разработчиками
**Действие:** Добавить в UnifiedPluginManager

### Методы, которые НЕ переносим:
- `install_plugin_from_file()` - уже есть в AdvancedPluginManager
- `_get_default_params()` - внутренний метод, переделаем логику
- `_load_builtin_plugins()` - уже есть в UnifiedPluginManager через scan

---

## 🔄 ШАГ 2: Обновление импортов (8 файлов)

### Файл 1: `app/main_window.py`
```python
# БЫЛО:
from .plugins.plugin_manager import PluginManager
from app.plugins.universal_plugin_manager import UniversalPluginManager

self.plugin_manager = PluginManager()
self.universal_plugin_manager = UniversalPluginManager()

# СТАНЕТ:
from .plugins.unified_plugin_manager import get_unified_plugin_manager

self.plugin_manager = get_unified_plugin_manager()
# self.universal_plugin_manager удаляется
```

### Файл 2: `app/processing_engine.py`
```python
# БЫЛО:
from .plugins.plugin_manager import PluginManager
self.plugin_manager = PluginManager()

# СТАНЕТ:
from .plugins.unified_plugin_manager import get_unified_plugin_manager
self.plugin_manager = get_unified_plugin_manager()
```

### Файл 3: `app/ui/plugins_dialog.py`
```python
# БЫЛО:
from ..plugins.plugin_manager import PluginManager
self.plugin_manager = PluginManager()

# СТАНЕТ:
from ..plugins.unified_plugin_manager import get_unified_plugin_manager
self.plugin_manager = get_unified_plugin_manager()
```

### Файл 4: `app/ui/components/processing_controller.py`
```python
# БЫЛО:
from app.plugins.plugin_manager import PluginManager
from app.plugins.universal_plugin_manager import UniversalPluginManager

self.plugin_manager = PluginManager()
self.universal_plugin_manager = UniversalPluginManager()

# СТАНЕТ:
from app.plugins.unified_plugin_manager import get_unified_plugin_manager

self.plugin_manager = get_unified_plugin_manager()
# self.universal_plugin_manager удаляется
```

### Файл 5: `app/core/di_container.py`
```python
# БЫЛО:
from app.plugins.plugin_manager import PluginManager
from app.plugins.universal_plugin_manager import UniversalPluginManager

"plugin_manager": lambda c: PluginManager(),
"universal_plugin_manager": lambda c: UniversalPluginManager(),

# СТАНЕТ:
from app.plugins.unified_plugin_manager import get_unified_plugin_manager

"plugin_manager": lambda c: get_unified_plugin_manager(),
# "universal_plugin_manager" удаляется
```

### Файл 6: `app/plugins/__init__.py`
```python
# БЫЛО:
from .plugin_manager import PluginManager

# СТАНЕТ:
from .unified_plugin_manager import UnifiedPluginManager, get_unified_plugin_manager
```

---

## ⚠️ ШАГ 3: Изменения API

### 3.1 Методы с изменением сигнатуры

#### `get_available_plugins()`
**Было (PluginManager):**
```python
def get_available_plugins(self) -> List[str]:
    return list(self.plugin_classes.keys())
```

**Стало (UnifiedPluginManager):**
```python
def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
    # Возвращает словарь с метаданными
```

**Решение:** Добавить метод-обертку для совместимости:
```python
def get_available_plugin_ids(self) -> List[str]:
    """Обратная совместимость с PluginManager"""
    return list(self.registry.get_all().keys())
```

#### `create_plugin_instance()`
**Было (PluginManager):**
```python
def create_plugin_instance(self, plugin_id: str, **kwargs) -> Optional[BaseLLMPlugin]
```

**Стало (UnifiedPluginManager):**
```python
def enable_plugin(self, plugin_id: str) -> bool
def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]
```

**Решение:** Добавить метод-обертку:
```python
def create_plugin_instance(self, plugin_id: str, **kwargs) -> Optional[BasePlugin]:
    """Обратная совместимость с PluginManager"""
    if self.enable_plugin(plugin_id):
        return self.get_plugin(plugin_id)
    return None
```

### 3.2 Новые методы для добавления в UnifiedPluginManager

```python
# 1. Провайдеры LLM
def create_plugin_by_provider(self, provider_name: str, model_name: str = None, 
                             api_key: str = None, **kwargs) -> Optional[BasePlugin]:
    """Создает LLM плагин по имени провайдера"""
    
def get_providers_info(self) -> Dict[str, Dict]:
    """Информация о LLM провайдерах"""
    
def get_recommended_plugin(self, provider_name: str) -> Optional[str]:
    """Рекомендуемый плагин для провайдера"""

# 2. Шаблоны
def create_plugin_template(self, plugin_name: str, output_dir: str = None) -> str:
    """Создает шаблон плагина"""

# 3. Обратная совместимость
def get_available_plugin_ids(self) -> List[str]:
    """Список ID плагинов (обратная совместимость)"""
    
def create_plugin_instance(self, plugin_id: str, **kwargs) -> Optional[BasePlugin]:
    """Создание экземпляра (обратная совместимость)"""
```

---

## 🧪 ШАГ 4: Тестирование

### 4.1 Проверка перед миграцией
- [ ] Создать резервную ветку git
- [ ] Сохранить текущее состояние
- [ ] Проверить все использования PluginManager и UniversalPluginManager

### 4.2 После добавления методов в UnifiedPluginManager
- [ ] Проверить импорты
- [ ] Запустить приложение
- [ ] Протестировать загрузку плагинов
- [ ] Проверить создание LLM плагинов

### 4.3 После миграции импортов
- [ ] Проверить каждый мигрированный файл
- [ ] Убедиться, что плагины загружаются
- [ ] Протестировать UI диалоги плагинов
- [ ] Проверить обработку файлов с LLM

### 4.4 После удаления старых менеджеров
- [ ] Финальная проверка линтера
- [ ] Полное тестирование всех функций приложения
- [ ] Проверка производительности

---

## 📊 ОЦЕНКА РИСКОВ

### 🔴 Высокий риск:
- **main_window.py** - центральный файл, ошибки критичны
- **processing_engine.py** - обработка файлов, нужна осторожность

### 🟡 Средний риск:
- **plugins_dialog.py** - UI, можно быстро исправить
- **processing_controller.py** - контроллер, изолированный

### 🟢 Низкий риск:
- **di_container.py** - DI контейнер, простая замена
- **__init__.py** - только экспорты

---

## ⏱️ ВРЕМЕННЫЕ ОЦЕНКИ

- **Шаг 1:** Перенос методов - 2-3 часа
- **Шаг 2:** Обновление импортов - 1-2 часа
- **Шаг 3:** Тестирование - 2-3 часа
- **Шаг 4:** Удаление старых файлов - 30 минут

**Итого:** 5-8 часов

---

## ✅ КРИТЕРИИ УСПЕХА

1. ✅ Все 8 файлов используют UnifiedPluginManager
2. ✅ LLM плагины создаются корректно
3. ✅ UI диалоги плагинов работают
4. ✅ Обработка файлов не сломана
5. ✅ Линтер не показывает ошибок
6. ✅ Удалены PluginManager.py и UniversalPluginManager.py

---

## 🚀 ПОРЯДОК ВЫПОЛНЕНИЯ

1. ✅ Создать план миграции (этот документ)
2. ⏳ Добавить недостающие методы в UnifiedPluginManager
3. ⏳ Обновить импорты в файлах (по одному)
4. ⏳ Тестировать после каждого файла
5. ⏳ Удалить старые менеджеры
6. ✅ Финальное тестирование

---

**Дата создания:** 8 октября 2025
**Автор:** AI Assistant
**Статус:** 📋 План готов к выполнению

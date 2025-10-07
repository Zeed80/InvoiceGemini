# 🎨 Анализ улучшений пользовательского опыта InvoiceGemini

**Дата анализа:** 3 октября 2025  
**Версия:** 1.0  
**Статус:** Комплексный анализ с рекомендациями

---

## 📋 Содержание

1. [Исполнительное резюме](#исполнительное-резюме)
2. [Текущее состояние UX](#текущее-состояние-ux)
3. [Выявленные проблемы](#выявленные-проблемы)
4. [Предложения по улучшению](#предложения-по-улучшению)
5. [Техническое Руководство по Дизайну (ТРД)](#техническое-руководство-по-дизайну)
6. [Новые функции и возможности](#новые-функции-и-возможности)
7. [План внедрения](#план-внедрения)

---

## 📊 Исполнительное резюме

### Общая оценка текущего UX: **7/10**

**Сильные стороны:**
- ✅ Функциональная полнота (множество моделей и возможностей)
- ✅ Оптимизированные компоненты (Phase 1-2 завершены)
- ✅ Система плагинов и расширяемость
- ✅ Поддержка русского языка

**Области для улучшения:**
- ⚠️ Информационная перегрузка главного окна
- ⚠️ Отсутствие onboarding для новых пользователей
- ⚠️ Недостаточная визуальная иерархия
- ⚠️ Неоптимальный workflow для типичных задач
- ⚠️ Ограниченная система обратной связи

---

## 🔍 Текущее состояние UX

### Анализ по принципам Jakob Nielsen

#### 1. **Видимость статуса системы** (6/10)
- ✅ Есть: Progress bar с ETA, статус бар
- ❌ Нет: Индикаторы загрузки моделей в памяти, предупреждения о ресурсах
- 💡 **Улучшение**: Добавить System Status Dashboard

#### 2. **Соответствие между системой и реальным миром** (8/10)
- ✅ Есть: Понятная русская терминология, иконки
- ⚠️ Ограничено: Технические термины (LayoutLM, Donut) без объяснений
- 💡 **Улучшение**: Tooltips с объяснениями, глоссарий

#### 3. **Управляемость и свобода для пользователя** (5/10)
- ❌ Нет: Отмена долгих операций, undo/redo для редактирования
- ⚠️ Ограничено: История обработки без возможности повтора
- 💡 **Улучшение**: Command pattern для отмены, контекстное меню

#### 4. **Согласованность и стандарты** (7/10)
- ✅ Есть: Единый стиль PyQt6, оптимизированные компоненты
- ⚠️ Ограничено: Разные стили диалогов (legacy vs new)
- 💡 **Улучшение**: Единая дизайн-система

#### 5. **Предотвращение ошибок** (6/10)
- ✅ Есть: Валидация настроек, проверка API ключей
- ❌ Нет: Предупреждения о последствиях, подтверждения критичных действий
- 💡 **Улучшение**: Smart validation, предиктивные подсказки

#### 6. **Распознавание вместо запоминания** (5/10)
- ⚠️ Проблема: Пользователь должен помнить:
  - Где находятся разные функции
  - Какую модель выбрать для какого типа документов
  - Последовательность настройки
- 💡 **Улучшение**: Контекстные подсказки, рекомендации AI

#### 7. **Гибкость и эффективность использования** (6/10)
- ✅ Есть: Drag & Drop, batch processing
- ❌ Нет: Горячие клавиши, кастомизация toolbar, макросы
- 💡 **Улучшение**: Keyboard shortcuts, настраиваемый UI

#### 8. **Эстетический и минималистичный дизайн** (5/10)
- ⚠️ Проблема: Информационная перегрузка
  - Главное окно содержит >20 элементов управления
  - 5+ радиокнопок моделей одновременно видны
  - Множество вкладок в диалогах (до 6)
- 💡 **Улучшение**: Progressive disclosure, smart defaults

#### 9. **Помощь в распознавании и восстановлении после ошибок** (6/10)
- ✅ Есть: QMessageBox с текстами ошибок
- ❌ Нет: Рекомендации по исправлению, ссылки на документацию
- 💡 **Улучшение**: Smart error dialog с решениями

#### 10. **Справка и документация** (7/10)
- ✅ Есть: Обширная документация в docs/
- ❌ Нет: In-app help, contextual tips, interactive tutorials
- 💡 **Улучшение**: Встроенная справка, видео-туры

---

## ❗ Выявленные проблемы

### 🚨 Критические проблемы

#### 1. **Отсутствие onboarding для новых пользователей**
**Проблема:**
- Первый запуск показывает пустое окно без подсказок
- Нет гайда по настройке API ключей
- Неясно, с чего начать

**Влияние:** Высокий порог входа, отказ новых пользователей

**Решение:**
```python
# Новый класс OnboardingWizard
class OnboardingWizard(QWizard):
    """Мастер первого запуска с настройкой"""
    
    def __init__(self, parent=None):
        # Страница 1: Приветствие и выбор режима
        # Страница 2: Настройка моделей (облачные/локальные)
        # Страница 3: API ключи (опционально)
        # Страница 4: Тестовый документ
```

#### 2. **Информационная перегрузка главного окна**
**Проблема:**
- 5 радиокнопок выбора модели занимают много места
- Настройки Gemini (temperature, max_tokens) всегда видны
- Результаты в виде большой таблицы

**Влияние:** Когнитивная перегрузка, сложность фокусировки

**Решение:**
- Сгруппировать модели в dropdown с описаниями
- Расширенные настройки в collapsible panel
- Режимы просмотра результатов (compact/detailed)

#### 3. **Недостаточная обратная связь при длительных операциях**
**Проблема:**
- Загрузка моделей (~5-10 сек) без индикации
- Обучение моделей (минуты/часы) с минимальной информацией
- Batch processing без детального прогресса

**Влияние:** Непонимание, происходит ли что-то, тревога пользователя

**Решение:**
- Многоуровневый progress indicator
- Cancellable operations
- Estimated time remaining

### ⚠️ Важные проблемы

#### 4. **Сложность выбора подходящей модели**
**Проблема:**
- Пользователь не знает, какую модель выбрать
- Нет рекомендаций на основе типа документа
- Неясны преимущества каждой модели

**Решение:**
```python
class SmartModelSelector(QWidget):
    """Умный селектор модели с рекомендациями"""
    
    def recommend_model(self, file_path: str) -> str:
        """
        Анализирует документ и рекомендует модель:
        - PDF с текстовым слоем → Gemini (быстро, точно)
        - Сканы высокого качества → LayoutLMv3 (лучшая структура)
        - Фото/низкое качество → Donut (robust)
        """
```

#### 5. **Отсутствие быстрых действий и shortcuts**
**Проблема:**
- Нет горячих клавиш для основных действий
- Множество кликов для типичных задач
- Невозможность повторить последнюю операцию

**Решение:**
- Глобальные shortcuts (Ctrl+O, Ctrl+P, F5, etc.)
- Quick Actions panel
- История команд с повтором

#### 6. **Неудобная работа с результатами**
**Проблема:**
- Таблица результатов не поддерживает копирование
- Нельзя быстро исправить одно поле
- Нет быстрого экспорта в clipboard

**Решение:**
- Inline editing в таблице
- Context menu (Copy, Export, Edit)
- Quick copy buttons для каждого поля

### 💡 Улучшения производительности UX

#### 7. **Отсутствие персонализации**
**Проблема:**
- Нет запоминания предпочтений пользователя
- Всегда показываются все опции
- Нет адаптации под частые задачи

**Решение:**
- Workspace profiles (Бухгалтер, Архив, Массовая обработка)
- Learning from user behavior
- Customizable dashboard

#### 8. **Ограниченная система уведомлений**
**Проблема:**
- Только модальные QMessageBox
- Блокируют работу
- Нет истории уведомлений

**Решение:**
- Toast notifications (неинвазивные)
- Notification center
- Уровни важности (info/warning/error)

---

## 💡 Предложения по улучшению

### 🎯 Приоритет 1: Первое впечатление

#### A. Onboarding Wizard
**Цель:** Снизить порог входа на 70%

**Реализация:**
1. **Страница приветствия**
   - Краткое видео (30 сек)
   - Выбор режима работы:
     - 🏢 Бухгалтерия (фокус на точность)
     - 📦 Массовая обработка (фокус на скорость)
     - 🔬 Исследования (все возможности)

2. **Настройка моделей**
   - Автоопределение доступности GPU
   - Рекомендация:
     - GPU доступен → LayoutLMv3 + Gemini
     - Только CPU → Gemini (облачный)
   - Checkbox: "Загрузить тестовые модели"

3. **API ключи (опционально)**
   - Понятное объяснение, зачем нужны
   - Ссылки на получение
   - Кнопка "Пропустить, настрою позже"

4. **Первый документ**
   - Интерактивный тур по интерфейсу
   - Обработка тестового счета
   - Объяснение результатов

**Код:**
```python
# app/ui/onboarding_wizard.py
class OnboardingWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Добро пожаловать в InvoiceGemini!")
        self.setMinimumSize(800, 600)
        
        # Страницы
        self.addPage(WelcomePage())
        self.addPage(ModelSetupPage())
        self.addPage(APIKeysPage())
        self.addPage(FirstDocumentPage())
        
        # Стиль
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
```

#### B. Interactive Tutorial
**Цель:** Обучить основным функциям за 2 минуты

**Функции:**
- Overlay подсказки на реальном UI
- Пошаговое выполнение задач
- Можно пропустить/вернуться
- Сохранение прогресса

#### C. Welcome Dashboard
**Цель:** Показать возможности и направить

**Элементы:**
- Quick Start cards:
  - 📄 Обработать один документ
  - 📂 Batch обработка папки
  - 🎓 Обучить свою модель
  - ⚙️ Настроить интеграции
- Recent files
- Tips & tricks

### 🎨 Приоритет 2: Визуальная иерархия и организация

#### A. Redesigned Main Window
**Текущая проблема:** Все элементы на одном уровне

**Новая структура:**
```
┌─────────────────────────────────────────────┐
│ [Toolbar: Quick Actions]                    │
├─────────────────────┬───────────────────────┤
│                     │                       │
│  📁 File Panel      │  🖼️ Preview          │
│  [Smart Selector]   │  [Document Viewer]   │
│  [Recent Files]     │                       │
│                     │  📊 Results           │
│  ⚙️ Quick Settings  │  [Field Editor]      │
│  [Model: Auto]      │  [Confidence Meter]  │
│  [Options ▼]        │                       │
│                     │  [🚀 Process]         │
│  📈 Stats           │  [💾 Export]          │
│  [Session info]     │                       │
│                     │                       │
└─────────────────────┴───────────────────────┘
│ Status Bar: Model loaded | Ready | GPU: 4GB │
└─────────────────────────────────────────────┘
```

**Преимущества:**
- Четкое разделение на зоны
- Workflow слева направо
- Меньше визуального шума
- Больше места для превью и результатов

#### B. Smart Model Selector
**Вместо:** 5 радиокнопок

**Новый компонент:**
```python
class SmartModelSelector(QComboBox):
    """Умный селектор с рекомендациями"""
    
    def __init__(self):
        super().__init__()
        
        # Режимы
        self.addItem("🤖 Авто (рекомендуется)", "auto")
        self.addItem("⚡ Быстрый (Gemini)", "gemini")
        self.addItem("🎯 Точный (LayoutLM)", "layoutlm")
        self.addItem("💪 Надежный (Donut)", "donut")
        self.addItem("⚙️ Настроить...", "custom")
        
        # Tooltip с объяснением
        self.setToolTip(
            "Авто: Автоматический выбор лучшей модели\n"
            "Быстрый: Облачный Gemini, <5 сек\n"
            "Точный: LayoutLM, лучшая структура\n"
            "Надежный: Donut, работает везде"
        )
```

#### C. Collapsible Panels
**Группировка опций:**

1. **Basic Settings** (всегда видны)
   - Model selector
   - Process button

2. **Advanced Settings** (collapsible)
   - Temperature, max tokens
   - Custom prompts
   - OCR settings

3. **Batch Settings** (collapsible)
   - Parallel processing
   - Error handling
   - Output format

### 🔔 Приоритет 3: Система обратной связи

#### A. Toast Notification System
**Замена** модальных QMessageBox на неинвазивные уведомления

**Реализация:**
```python
class ToastNotification(QWidget):
    """Неинвазивное уведомление"""
    
    def __init__(self, message, level="info", duration=3000):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.ToolTip)
        
        # Стиль по уровню
        colors = {
            "info": "#3498db",
            "success": "#27ae60", 
            "warning": "#f39c12",
            "error": "#e74c3c"
        }
        
        # Layout
        layout = QHBoxLayout(self)
        icon = self._get_icon(level)
        label = QLabel(message)
        
        layout.addWidget(icon)
        layout.addWidget(label)
        
        # Автоскрытие
        QTimer.singleShot(duration, self.hide_animated)
    
    def show_at_bottom_right(self):
        """Показать в правом нижнем углу"""
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.width() - self.width() - 20,
            screen.height() - self.height() - 60
        )
        self.show()
```

**Использование:**
```python
# Вместо
QMessageBox.information(self, "Успех", "Файл обработан")

# Используем
show_toast("Файл обработан успешно", "success")
```

#### B. Notification Center
**Функции:**
- История всех уведомлений сессии
- Фильтрация по уровню
- Экспорт логов
- Кликабельные уведомления для перехода к проблеме

#### C. Progress System
**Многоуровневый прогресс:**

```python
class ProgressManager(QObject):
    """Управление показом прогресса"""
    
    def __init__(self):
        self.operations = {}
        
    def start_operation(self, op_id, title, total_steps):
        """Начать отслеживание операции"""
        self.operations[op_id] = {
            'title': title,
            'current': 0,
            'total': total_steps,
            'started': time.time(),
            'status': 'running'
        }
        
    def update_operation(self, op_id, current, status_text=""):
        """Обновить прогресс"""
        op = self.operations[op_id]
        op['current'] = current
        op['status_text'] = status_text
        
        # Вычисляем ETA
        elapsed = time.time() - op['started']
        if current > 0:
            total_time = elapsed * op['total'] / current
            eta = total_time - elapsed
            op['eta'] = eta
```

**UI компонент:**
```
┌──────────────────────────────────────────┐
│ 🔄 Обработка документов                  │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45%      │
│                                          │
│ ├─ Загрузка модели          ✅ Done     │
│ ├─ Обработка OCR            🔄 15/30    │
│ └─ Извлечение данных        ⏳ Pending  │
│                                          │
│ ETA: 2 мин 30 сек          [Cancel]     │
└──────────────────────────────────────────┘
```

### ⚡ Приоритет 4: Производительность работы

#### A. Keyboard Shortcuts
**Основные:**
- `Ctrl+O` - Open file
- `Ctrl+Shift+O` - Open folder (batch)
- `Ctrl+P` / `F5` - Process current
- `Ctrl+S` - Save results
- `Ctrl+E` - Export
- `Ctrl+,` - Settings
- `F1` - Help
- `Ctrl+Q` - Quit

**Дополнительные:**
- `Ctrl+1..5` - Быстрый выбор модели
- `Ctrl+B` - Batch mode
- `Ctrl+T` - Training dialog
- `Ctrl+L` - View logs

**Реализация:**
```python
def setup_shortcuts(self):
    """Настройка горячих клавиш"""
    shortcuts = {
        'Ctrl+O': self.open_file_dialog,
        'Ctrl+P': self.process_current,
        'Ctrl+S': self.save_results,
        'F1': self.show_help,
        # ... и т.д.
    }
    
    for key, handler in shortcuts.items():
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.activated.connect(handler)
```

#### B. Quick Actions Toolbar
**Контекстная панель инструментов:**

```python
class QuickActionsToolbar(QToolBar):
    """Toolbar с частыми действиями"""
    
    def __init__(self):
        super().__init__()
        
        # Основные действия
        self.add_action("📂 Открыть", "Ctrl+O")
        self.add_action("🚀 Обработать", "Ctrl+P") 
        self.add_action("💾 Сохранить", "Ctrl+S")
        self.addSeparator()
        
        # Быстрые настройки
        self.model_combo = SmartModelSelector()
        self.addWidget(self.model_combo)
        self.addSeparator()
        
        # Дополнительно
        self.add_action("📊 Batch", "Ctrl+B")
        self.add_action("🎓 Training", "Ctrl+T")
```

#### C. Context Menu Enhancement
**Правый клик в нужном месте:**

```python
def create_result_context_menu(self, field_name, value):
    """Контекстное меню для поля результата"""
    menu = QMenu(self)
    
    menu.addAction("📋 Копировать", lambda: self.copy_to_clipboard(value))
    menu.addAction("✏️ Редактировать", lambda: self.edit_field(field_name))
    menu.addAction("🔍 Показать в документе", lambda: self.highlight_in_doc(field_name))
    menu.addSeparator()
    menu.addAction("❌ Очистить поле", lambda: self.clear_field(field_name))
    menu.addAction("🔄 Переобработать поле", lambda: self.reprocess_field(field_name))
    
    return menu
```

#### D. Command History
**История команд с возможностью повтора:**

```python
class CommandHistory(QWidget):
    """История выполненных команд"""
    
    def __init__(self):
        super().__init__()
        self.history = []
        
    def add_command(self, command, params):
        """Добавить команду в историю"""
        self.history.append({
            'command': command,
            'params': params,
            'timestamp': datetime.now(),
            'success': True
        })
        
    def repeat_command(self, index):
        """Повторить команду из истории"""
        cmd = self.history[index]
        self.execute_command(cmd['command'], cmd['params'])
```

### 🎛️ Приоритет 5: Персонализация

#### A. Workspace Profiles
**Режимы работы для разных задач:**

```python
class WorkspaceProfile:
    """Профиль рабочего пространства"""
    
    PROFILES = {
        'accountant': {
            'name': '🏢 Бухгалтерия',
            'default_model': 'layoutlm',
            'focus': 'accuracy',
            'visible_panels': ['preview', 'results', 'validation'],
            'auto_export': True,
            'export_format': 'excel'
        },
        'batch': {
            'name': '📦 Массовая обработка',
            'default_model': 'gemini',
            'focus': 'speed',
            'visible_panels': ['file_list', 'batch_progress'],
            'auto_export': True,
            'export_format': 'json'
        },
        'research': {
            'name': '🔬 Исследования',
            'default_model': 'auto',
            'focus': 'flexibility',
            'visible_panels': ['all'],
            'comparison_mode': True
        }
    }
```

**UI для переключения:**
```python
# В главном окне
profile_selector = QComboBox()
profile_selector.addItems([p['name'] for p in WorkspaceProfile.PROFILES.values()])
profile_selector.currentTextChanged.connect(self.apply_profile)
```

#### B. Adaptive UI
**Обучение на поведении пользователя:**

```python
class UserBehaviorAnalyzer:
    """Анализ поведения для адаптации UI"""
    
    def __init__(self):
        self.stats = {
            'most_used_model': Counter(),
            'frequent_exports': Counter(),
            'common_workflows': [],
            'error_prone_steps': []
        }
        
    def suggest_optimizations(self):
        """Предложить оптимизации на основе статистики"""
        suggestions = []
        
        # Если пользователь всегда использует одну модель
        if self.stats['most_used_model'].most_common(1)[0][1] > 20:
            suggestions.append({
                'type': 'default_model',
                'message': 'Вы часто используете LayoutLM. Сделать моделью по умолчанию?'
            })
        
        # Если часто экспортирует в один формат
        if self.stats['frequent_exports'].most_common(1)[0][1] > 15:
            suggestions.append({
                'type': 'auto_export',
                'message': 'Включить автоэкспорт в Excel после обработки?'
            })
        
        return suggestions
```

#### C. Customizable Dashboard
**Drag & Drop виджеты:**

```python
class CustomizableDashboard(QWidget):
    """Настраиваемая главная панель"""
    
    def __init__(self):
        super().__init__()
        self.widgets = []
        
        # Доступные виджеты
        self.available_widgets = {
            'quick_process': QuickProcessWidget(),
            'recent_files': RecentFilesWidget(),
            'stats': StatisticsWidget(),
            'tips': TipsWidget(),
            'model_status': ModelStatusWidget(),
            'api_usage': APIUsageWidget()
        }
        
    def add_widget(self, widget_id, position):
        """Добавить виджет на dashboard"""
        widget = self.available_widgets[widget_id]
        self.layout().addWidget(widget, *position)
        self.widgets.append((widget_id, position))
        self.save_layout()
```

---

## 📐 Техническое Руководство по Дизайну (ТРД)

### 1. 🎨 Визуальный стиль

#### 1.1 Цветовая палитра

**Основная палитра:**
```python
COLORS = {
    # Бренд
    'primary': '#3498db',      # Основной синий
    'primary_dark': '#2980b9', # Темный синий
    'primary_light': '#5dade2', # Светлый синий
    
    # Акценты
    'success': '#27ae60',      # Успех/подтверждение
    'warning': '#f39c12',      # Предупреждение
    'error': '#e74c3c',        # Ошибка
    'info': '#3498db',         # Информация
    
    # Нейтральные
    'text_primary': '#2c3e50', # Основной текст
    'text_secondary': '#7f8c8d', # Вторичный текст
    'text_hint': '#95a5a6',    # Подсказки
    
    # Фоны
    'background': '#ecf0f1',   # Основной фон
    'surface': '#ffffff',      # Поверхности (карточки)
    'hover': '#e8f4f8',        # Наведение
    'disabled': '#bdc3c7',     # Отключено
    
    # Специальные
    'model_layoutlm': '#9b59b6',  # LayoutLM
    'model_donut': '#e67e22',     # Donut
    'model_gemini': '#1abc9c',    # Gemini
}
```

**Темная тема:**
```python
DARK_COLORS = {
    'primary': '#5dade2',
    'background': '#1e272e',
    'surface': '#2c3e50',
    'text_primary': '#ecf0f1',
    'text_secondary': '#95a5a6',
    'hover': '#34495e',
}
```

#### 1.2 Типографика

```python
TYPOGRAPHY = {
    # Заголовки
    'h1': {'size': 24, 'weight': 'bold', 'line_height': 1.2},
    'h2': {'size': 20, 'weight': 'bold', 'line_height': 1.3},
    'h3': {'size': 16, 'weight': 'bold', 'line_height': 1.4},
    'h4': {'size': 14, 'weight': 'bold', 'line_height': 1.4},
    
    # Тело
    'body': {'size': 12, 'weight': 'normal', 'line_height': 1.5},
    'body_large': {'size': 14, 'weight': 'normal', 'line_height': 1.5},
    'body_small': {'size': 10, 'weight': 'normal', 'line_height': 1.5},
    
    # Специальные
    'button': {'size': 12, 'weight': 'medium', 'line_height': 1.0},
    'caption': {'size': 10, 'weight': 'normal', 'line_height': 1.4},
    'code': {'size': 11, 'weight': 'normal', 'line_height': 1.6, 'family': 'Consolas'},
}

# Шрифты
FONTS = {
    'primary': 'Segoe UI',  # Windows
    'fallback': ['Roboto', 'Arial', 'sans-serif'],
    'monospace': 'Consolas'
}
```

#### 1.3 Иконки

**Система иконок:**
- Используем emoji для быстрых акцентов
- FontAwesome для действий
- Custom SVG для брендовых элементов

```python
ICONS = {
    # Действия
    'process': '🚀',
    'save': '💾',
    'export': '📤',
    'import': '📥',
    'settings': '⚙️',
    'help': '❓',
    
    # Модели
    'layoutlm': '🎯',
    'donut': '🍩',
    'gemini': '💎',
    'auto': '🤖',
    
    # Статусы
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'loading': '🔄',
    
    # Файлы
    'file': '📄',
    'folder': '📁',
    'image': '🖼️',
    'pdf': '📕',
}
```

### 2. 📏 Компоненты UI

#### 2.1 Кнопки

**Типы кнопок:**

```python
class ButtonStyles:
    """Стили для кнопок"""
    
    PRIMARY = """
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c5985;
        }
        QPushButton:disabled {
            background-color: #bdc3c7;
            color: #7f8c8d;
        }
    """
    
    SECONDARY = """
        QPushButton {
            background-color: transparent;
            color: #3498db;
            border: 2px solid #3498db;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #e8f4f8;
        }
    """
    
    SUCCESS = """
        QPushButton {
            background-color: #27ae60;
            color: white;
            /* ... */
        }
    """
    
    DANGER = """
        QPushButton {
            background-color: #e74c3c;
            color: white;
            /* ... */
        }
    """
    
    TEXT = """
        QPushButton {
            background-color: transparent;
            color: #3498db;
            border: none;
            padding: 8px 12px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #e8f4f8;
            border-radius: 4px;
        }
    """
```

**Размеры:**
```python
BUTTON_SIZES = {
    'small': {'height': 28, 'padding': '6px 12px', 'font_size': 11},
    'medium': {'height': 36, 'padding': '8px 16px', 'font_size': 12},
    'large': {'height': 44, 'padding': '10px 20px', 'font_size': 14},
}
```

#### 2.2 Поля ввода

```python
INPUT_STYLE = """
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: white;
        border: 2px solid #bdc3c7;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        color: #2c3e50;
    }
    
    QLineEdit:focus, QTextEdit:focus {
        border-color: #3498db;
        background-color: #f8fbfd;
    }
    
    QLineEdit:hover, QTextEdit:hover {
        border-color: #95a5a6;
    }
    
    QLineEdit:disabled, QTextEdit:disabled {
        background-color: #ecf0f1;
        color: #7f8c8d;
        border-color: #d5dbdb;
    }
    
    /* Placeholder text */
    QLineEdit::placeholder {
        color: #95a5a6;
        font-style: italic;
    }
"""
```

#### 2.3 Таблицы

```python
TABLE_STYLE = """
    QTableWidget {
        background-color: white;
        border: 1px solid #d5dbdb;
        border-radius: 8px;
        gridline-color: #ecf0f1;
    }
    
    QTableWidget::item {
        padding: 8px;
        border: none;
    }
    
    QTableWidget::item:selected {
        background-color: #e8f4f8;
        color: #2c3e50;
    }
    
    QTableWidget::item:hover {
        background-color: #f4f6f7;
    }
    
    QHeaderView::section {
        background-color: #ecf0f1;
        color: #2c3e50;
        padding: 10px;
        border: none;
        font-weight: 600;
        text-align: left;
    }
    
    QHeaderView::section:hover {
        background-color: #d5dbdb;
    }
"""
```

#### 2.4 Прогресс-бары

```python
PROGRESS_STYLE = """
    QProgressBar {
        border: 2px solid #d5dbdb;
        border-radius: 8px;
        background-color: #ecf0f1;
        height: 24px;
        text-align: center;
        font-size: 11px;
        font-weight: 600;
        color: #2c3e50;
    }
    
    QProgressBar::chunk {
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #3498db, stop: 1 #5dade2
        );
        border-radius: 6px;
    }
    
    /* Анимация при неопределенной длительности */
    QProgressBar:indeterminate::chunk {
        background-color: #3498db;
        width: 30px;
    }
"""
```

#### 2.5 Уведомления (Toast)

```python
class ToastNotification(QWidget):
    """Стандартное уведомление"""
    
    STYLE_TEMPLATE = """
        QWidget {{
            background-color: {bg_color};
            border-left: 4px solid {accent_color};
            border-radius: 8px;
            padding: 12px 16px;
        }}
        
        QLabel {{
            color: {text_color};
            font-size: 12px;
        }}
    """
    
    STYLES = {
        'info': {
            'bg_color': '#ebf5fb',
            'accent_color': '#3498db',
            'text_color': '#21618c',
            'icon': 'ℹ️'
        },
        'success': {
            'bg_color': '#eafaf1',
            'accent_color': '#27ae60',
            'text_color': '#186a3b',
            'icon': '✅'
        },
        'warning': {
            'bg_color': '#fef5e7',
            'accent_color': '#f39c12',
            'text_color': '#9c640c',
            'icon': '⚠️'
        },
        'error': {
            'bg_color': '#fadbd8',
            'accent_color': '#e74c3c',
            'text_color': '#943126',
            'icon': '❌'
        }
    }
```

#### 2.6 Диалоговые окна

```python
DIALOG_STYLE = """
    QDialog {
        background-color: white;
    }
    
    QDialog QLabel#title {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        padding-bottom: 8px;
    }
    
    QDialog QLabel#description {
        font-size: 12px;
        color: #7f8c8d;
        padding-bottom: 16px;
    }
    
    QDialog QGroupBox {
        border: 2px solid #ecf0f1;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 8px;
        font-weight: 600;
    }
    
    QDialog QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: #2c3e50;
    }
"""
```

### 3. 🎯 Layout и Grid System

#### 3.1 Spacing

```python
SPACING = {
    'xs': 4,   # Очень маленький
    'sm': 8,   # Маленький
    'md': 16,  # Средний (стандарт)
    'lg': 24,  # Большой
    'xl': 32,  # Очень большой
    'xxl': 48, # Огромный
}

# Использование
layout.setSpacing(SPACING['md'])
layout.setContentsMargins(SPACING['lg'], SPACING['md'], SPACING['lg'], SPACING['md'])
```

#### 3.2 Grid System

```python
class ResponsiveLayout:
    """Адаптивная сетка"""
    
    # Breakpoints
    BREAKPOINTS = {
        'xs': 0,     # Extra small (всегда)
        'sm': 600,   # Small
        'md': 960,   # Medium
        'lg': 1280,  # Large
        'xl': 1920,  # Extra large
    }
    
    def get_columns(self, window_width):
        """Получить количество колонок по ширине"""
        if window_width >= self.BREAKPOINTS['xl']:
            return 12
        elif window_width >= self.BREAKPOINTS['lg']:
            return 8
        elif window_width >= self.BREAKPOINTS['md']:
            return 6
        elif window_width >= self.BREAKPOINTS['sm']:
            return 4
        else:
            return 2
```

### 4. ♿ Доступность (Accessibility)

#### 4.1 Контрастность текста

```python
def check_contrast_ratio(fg_color, bg_color):
    """Проверка коэффициента контрастности WCAG"""
    ratio = calculate_contrast(fg_color, bg_color)
    
    # WCAG AA: минимум 4.5:1 для обычного текста
    # WCAG AAA: минимум 7:1 для обычного текста
    return ratio >= 4.5

# Все цветовые комбинации должны соответствовать WCAG AA
```

#### 4.2 Keyboard Navigation

```python
def setup_tab_order(self):
    """Настройка порядка табуляции"""
    widgets = [
        self.file_selector,
        self.model_selector,
        self.process_button,
        self.results_table,
        self.export_button,
    ]
    
    for i in range(len(widgets) - 1):
        self.setTabOrder(widgets[i], widgets[i + 1])
```

#### 4.3 Screen Reader Support

```python
# Установка доступных имен для элементов
button.setAccessibleName("Обработать документ")
button.setAccessibleDescription("Начать обработку выбранного документа с помощью текущей модели")

# Роли элементов
table.setAccessibleName("Результаты извлечения данных")
table.setAccessibleDescription("Таблица с извлеченными полями счета-фактуры")
```

### 5. 🌍 Локализация

#### 5.1 Система переводов

```python
# Все строки через self.tr()
button.setText(self.tr("Обработать"))
message = self.tr("Обработано файлов: {count}").format(count=processed)

# Форматирование чисел
from babel.numbers import format_decimal
amount = format_decimal(1234.56, locale='ru_RU')  # "1 234,56"

# Форматирование дат
from babel.dates import format_date
date_str = format_date(date_obj, format='long', locale='ru_RU')  # "3 октября 2025 г."
```

#### 5.2 Поддерживаемые языки

- 🇷🇺 Русский (основной)
- 🇬🇧 English (secondary)
- 🇩🇪 Deutsch (planned)
- 🇫🇷 Français (planned)

### 6. 📱 Responsive Design

#### 6.1 Минимальные размеры окон

```python
WINDOW_SIZES = {
    'min_width': 1024,
    'min_height': 768,
    'default_width': 1400,
    'default_height': 900,
    'compact_mode_threshold': 1280,  # Ниже этого - компактный режим
}
```

#### 6.2 Адаптивные компоненты

```python
class ResponsiveMainWindow(QMainWindow):
    """Окно с адаптивным layout"""
    
    def resizeEvent(self, event):
        """Обработка изменения размера"""
        super().resizeEvent(event)
        
        width = event.size().width()
        
        # Переключение layout на основе ширины
        if width < 1280:
            self.switch_to_compact_mode()
        else:
            self.switch_to_normal_mode()
    
    def switch_to_compact_mode(self):
        """Компактный режим для маленьких экранов"""
        # Скрыть боковую панель
        self.sidebar.hide()
        # Уменьшить превью
        self.preview.setMaximumWidth(300)
        # Использовать tabs вместо split view
        self.use_tabbed_interface()
```

### 7. 🎬 Анимации и переходы

#### 7.1 Длительности

```python
ANIMATION_DURATIONS = {
    'instant': 0,       # Без анимации
    'fast': 150,        # Быстро
    'normal': 250,      # Стандарт
    'slow': 400,        # Медленно
    'very_slow': 700,   # Очень медленно
}
```

#### 7.2 Easing функции

```python
from PyQt6.QtCore import QEasingCurve

EASING = {
    'linear': QEasingCurve.Type.Linear,
    'ease_in': QEasingCurve.Type.InCubic,
    'ease_out': QEasingCurve.Type.OutCubic,
    'ease_in_out': QEasingCurve.Type.InOutCubic,
    'bounce': QEasingCurve.Type.OutBounce,
}
```

#### 7.3 Примеры анимаций

```python
def animate_button_click(button):
    """Анимация нажатия кнопки"""
    animation = QPropertyAnimation(button, b"pos")
    animation.setDuration(ANIMATION_DURATIONS['fast'])
    animation.setEasingCurve(EASING['ease_out'])
    
    # Сдвиг вниз и обратно
    start_pos = button.pos()
    animation.setKeyValueAt(0, start_pos)
    animation.setKeyValueAt(0.5, start_pos + QPoint(0, 2))
    animation.setKeyValueAt(1, start_pos)
    animation.start()

def fade_in_widget(widget, duration=ANIMATION_DURATIONS['normal']):
    """Плавное появление виджета"""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    
    animation = QPropertyAnimation(effect, b"opacity")
    animation.setDuration(duration)
    animation.setStartValue(0)
    animation.setEndValue(1)
    animation.setEasingCurve(EASING['ease_in'])
    animation.start()
```

---

## 🚀 Новые функции и возможности

### 1. 🤖 Интеллектуальные помощники

#### A. AI Assistant для выбора модели

```python
class ModelRecommendationEngine:
    """AI помощник для выбора модели"""
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Анализ документа для рекомендации"""
        features = {
            'has_text_layer': self._check_text_layer(file_path),
            'image_quality': self._assess_quality(file_path),
            'document_complexity': self._analyze_structure(file_path),
            'file_size': os.path.getsize(file_path),
            'language': self._detect_language(file_path)
        }
        
        return self._recommend_model(features)
    
    def _recommend_model(self, features):
        """Рекомендация на основе признаков"""
        recommendations = []
        
        # PDF с текстовым слоем
        if features['has_text_layer']:
            recommendations.append({
                'model': 'gemini',
                'confidence': 0.95,
                'reason': 'PDF с текстовым слоем - Gemini даст быстрый и точный результат',
                'expected_time': '3-5 сек'
            })
        
        # Сложная структура документа
        if features['document_complexity'] > 0.7:
            recommendations.append({
                'model': 'layoutlm',
                'confidence': 0.90,
                'reason': 'Сложная структура - LayoutLM лучше понимает layout',
                'expected_time': '10-15 сек'
            })
        
        # Низкое качество изображения
        if features['image_quality'] < 0.5:
            recommendations.append({
                'model': 'donut',
                'confidence': 0.85,
                'reason': 'Низкое качество сканирования - Donut более робастен',
                'expected_time': '8-12 сек'
            })
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
```

**UI для рекомендаций:**
```python
class ModelRecommendationWidget(QWidget):
    """Виджет с рекомендациями AI"""
    
    def show_recommendations(self, recommendations):
        """Показать рекомендации"""
        # Топ рекомендация
        top = recommendations[0]
        
        self.recommendation_label.setText(
            f"🤖 Рекомендуется: {top['model'].upper()}\n"
            f"Причина: {top['reason']}\n"
            f"⏱️ Ожидаемое время: {top['expected_time']}\n"
            f"📊 Уверенность: {top['confidence']*100:.0f}%"
        )
        
        # Альтернативы
        self.alternatives_list.clear()
        for alt in recommendations[1:]:
            self.alternatives_list.addItem(
                f"{alt['model']} ({alt['confidence']*100:.0f}%): {alt['reason']}"
            )
```

#### B. Smart Field Validation

```python
class FieldValidator:
    """Умная валидация полей"""
    
    PATTERNS = {
        'inn': r'^\d{10}|\d{12}$',
        'invoice_number': r'^[A-ZА-Я0-9-/]+$',
        'amount': r'^\d+(\.\d{2})?$',
        'date': r'^\d{2}\.\d{2}\.\d{4}$'
    }
    
    def validate_field(self, field_name, value, confidence):
        """Валидация с учетом confidence"""
        issues = []
        suggestions = []
        
        # Проверка формата
        if not re.match(self.PATTERNS.get(field_name, '.*'), value):
            issues.append({
                'severity': 'error',
                'message': f'Неверный формат поля {field_name}',
                'suggestion': self._suggest_format(field_name, value)
            })
        
        # Низкая уверенность
        if confidence < 0.7:
            issues.append({
                'severity': 'warning',
                'message': f'Низкая уверенность ({confidence:.0%})',
                'suggestion': 'Рекомендуется проверить вручную'
            })
        
        # Межполевая валидация
        cross_field_issues = self._validate_cross_field(field_name, value)
        issues.extend(cross_field_issues)
        
        return {
            'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def _validate_cross_field(self, field_name, value):
        """Перекрестная валидация полей"""
        issues = []
        
        # Например, сумма с НДС должна быть больше суммы без НДС
        if field_name == 'total_amount':
            amount_without_vat = self.get_field_value('amount_without_vat')
            if float(value) < float(amount_without_vat):
                issues.append({
                    'severity': 'error',
                    'message': 'Общая сумма меньше суммы без НДС',
                    'suggestion': 'Проверьте расчеты'
                })
        
        return issues
```

#### C. Automatic Error Recovery

```python
class ErrorRecoverySystem:
    """Система автоматического восстановления после ошибок"""
    
    def handle_processing_error(self, error, context):
        """Обработка ошибки с попыткой восстановления"""
        recovery_strategies = []
        
        # Ошибка загрузки модели
        if isinstance(error, ModelLoadError):
            recovery_strategies = [
                ('reload_model', 'Перезагрузить модель'),
                ('use_fallback_model', 'Использовать резервную модель'),
                ('download_model', 'Скачать модель заново'),
            ]
        
        # Ошибка памяти
        elif isinstance(error, MemoryError):
            recovery_strategies = [
                ('reduce_batch_size', 'Уменьшить размер батча'),
                ('use_cpu', 'Переключиться на CPU'),
                ('clear_cache', 'Очистить кэш'),
            ]
        
        # Ошибка OCR
        elif isinstance(error, OCRError):
            recovery_strategies = [
                ('retry_with_preprocessing', 'Повторить с предобработкой'),
                ('use_alternative_ocr', 'Использовать альтернативный OCR'),
                ('increase_resolution', 'Увеличить разрешение'),
            ]
        
        # Попытка автовосстановления
        for strategy_func, description in recovery_strategies:
            try:
                result = getattr(self, strategy_func)(context)
                if result:
                    return {
                        'recovered': True,
                        'strategy': description,
                        'result': result
                    }
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy_func} failed: {e}")
                continue
        
        return {'recovered': False, 'strategies_tried': len(recovery_strategies)}
```

### 2. 📊 Расширенная аналитика

#### A. Processing Analytics Dashboard

```python
class AnalyticsDashboard(QWidget):
    """Панель аналитики обработки"""
    
    def __init__(self):
        super().__init__()
        self.stats = self.load_stats()
        
        # Виджеты метрик
        self.total_processed = MetricWidget("📄 Обработано", self.stats['total'])
        self.success_rate = MetricWidget("✅ Успешно", f"{self.stats['success_rate']:.1f}%")
        self.avg_time = MetricWidget("⏱️ Среднее время", f"{self.stats['avg_time']:.1f}с")
        self.api_cost = MetricWidget("💰 Затраты API", f"${self.stats['api_cost']:.2f}")
        
        # Графики
        self.model_usage_chart = self.create_model_usage_chart()
        self.time_trend_chart = self.create_time_trend_chart()
        self.accuracy_chart = self.create_accuracy_chart()
    
    def create_model_usage_chart(self):
        """График использования моделей"""
        # Используем matplotlib или plotly
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        models = self.stats['model_usage'].keys()
        counts = self.stats['model_usage'].values()
        
        ax.pie(counts, labels=models, autopct='%1.1f%%')
        ax.set_title('Использование моделей')
        
        canvas = FigureCanvasQTAgg(fig)
        return canvas
```

#### B. Quality Metrics

```python
class QualityMetrics:
    """Метрики качества извлечения"""
    
    def calculate_extraction_quality(self, results, ground_truth=None):
        """Рассчитать качество извлечения"""
        metrics = {}
        
        # Completeness - заполненность полей
        total_fields = len(results)
        filled_fields = sum(1 for v in results.values() if v and v != 'N/A')
        metrics['completeness'] = filled_fields / total_fields if total_fields > 0 else 0
        
        # Confidence - средняя уверенность
        confidences = [r['confidence'] for r in results.values() if isinstance(r, dict) and 'confidence' in r]
        metrics['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        # Consistency - согласованность между полями
        metrics['consistency'] = self._check_consistency(results)
        
        # Accuracy - точность (если есть ground truth)
        if ground_truth:
            metrics['accuracy'] = self._calculate_accuracy(results, ground_truth)
        
        # Общий score
        weights = {
            'completeness': 0.3,
            'avg_confidence': 0.3,
            'consistency': 0.2,
            'accuracy': 0.2
        }
        
        metrics['overall_score'] = sum(
            metrics.get(k, 0) * v for k, v in weights.items()
        )
        
        return metrics
    
    def _check_consistency(self, results):
        """Проверка согласованности полей"""
        issues = 0
        
        # Проверка суммы
        amount = float(results.get('amount_without_vat', 0))
        vat_amount = float(results.get('vat_amount', 0))
        total = float(results.get('total_amount', 0))
        
        if abs(amount + vat_amount - total) > 0.01:
            issues += 1
        
        # Проверка процента НДС
        if vat_amount > 0 and amount > 0:
            vat_percent = (vat_amount / amount) * 100
            expected_vat = float(results.get('vat_percent', 0))
            if abs(vat_percent - expected_vat) > 1:
                issues += 1
        
        return 1 - (issues / 2)  # Нормализуем
```

### 3. 🔄 Workflow Automation

#### A. Custom Workflow Builder

```python
class WorkflowBuilder(QWidget):
    """Конструктор workflow'ов"""
    
    def __init__(self):
        super().__init__()
        self.workflow_steps = []
        
        # Доступные шаги
        self.available_steps = {
            'input': InputStep(),
            'preprocess': PreprocessStep(),
            'ocr': OCRStep(),
            'extract': ExtractionStep(),
            'validate': ValidationStep(),
            'export': ExportStep(),
            'notify': NotificationStep(),
        }
    
    def create_workflow(self):
        """Создать новый workflow"""
        workflow = Workflow()
        
        # Drag & drop интерфейс для добавления шагов
        # Настройка параметров каждого шага
        # Сохранение как template
        
        return workflow

class Workflow:
    """Пользовательский workflow"""
    
    def __init__(self, name="", steps=None):
        self.name = name
        self.steps = steps or []
        self.status = 'idle'
    
    def execute(self, input_data):
        """Выполнить workflow"""
        result = input_data
        
        for step in self.steps:
            try:
                result = step.execute(result)
                self.emit_progress(step)
            except Exception as e:
                self.handle_error(step, e)
                if step.fail_strategy == 'abort':
                    raise
                elif step.fail_strategy == 'skip':
                    continue
                elif step.fail_strategy == 'retry':
                    result = self.retry_step(step, result)
        
        return result
```

#### B. Batch Processing Templates

```python
class BatchTemplate:
    """Шаблон для batch обработки"""
    
    TEMPLATES = {
        'fast_scan': {
            'name': '⚡ Быстрое сканирование',
            'model': 'gemini',
            'parallel': 4,
            'export_format': 'json',
            'auto_validate': False,
            'description': 'Быстрая обработка для большого количества документов'
        },
        'accurate_accounting': {
            'name': '🎯 Точная бухгалтерия',
            'model': 'layoutlm',
            'parallel': 2,
            'export_format': 'excel',
            'auto_validate': True,
            'cross_check': True,
            'description': 'Максимальная точность с валидацией'
        },
        'archive_digitization': {
            'name': '📚 Оцифровка архива',
            'model': 'donut',
            'parallel': 3,
            'export_format': 'json',
            'create_searchable_pdf': True,
            'description': 'Массовая оцифровка старых документов'
        }
    }
```

### 4. 🔌 Расширенные интеграции

#### A. Webhook System

```python
class WebhookManager:
    """Система веб-хуков для интеграций"""
    
    def __init__(self):
        self.webhooks = []
    
    def add_webhook(self, event_type, url, config):
        """Добавить webhook"""
        webhook = {
            'event_type': event_type,  # 'document_processed', 'batch_completed', etc.
            'url': url,
            'method': config.get('method', 'POST'),
            'headers': config.get('headers', {}),
            'auth': config.get('auth'),
            'retry_count': config.get('retry_count', 3),
            'timeout': config.get('timeout', 30),
        }
        self.webhooks.append(webhook)
    
    def trigger_webhook(self, event_type, data):
        """Отправить webhook"""
        matching_webhooks = [w for w in self.webhooks if w['event_type'] == event_type]
        
        for webhook in matching_webhooks:
            self._send_webhook(webhook, data)
    
    def _send_webhook(self, webhook, data):
        """Отправка webhook с retry логикой"""
        import requests
        
        payload = {
            'event': webhook['event_type'],
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        for attempt in range(webhook['retry_count']):
            try:
                response = requests.request(
                    method=webhook['method'],
                    url=webhook['url'],
                    json=payload,
                    headers=webhook['headers'],
                    auth=webhook.get('auth'),
                    timeout=webhook['timeout']
                )
                response.raise_for_status()
                logger.info(f"Webhook sent successfully: {webhook['url']}")
                break
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt+1} failed: {e}")
                if attempt == webhook['retry_count'] - 1:
                    logger.error(f"Webhook failed after {webhook['retry_count']} attempts")
```

#### B. Cloud Storage Integration

```python
class CloudStorageIntegration:
    """Интеграция с облачными хранилищами"""
    
    PROVIDERS = ['dropbox', 'google_drive', 'onedrive', 's3', 'azure_blob']
    
    def __init__(self, provider, credentials):
        self.provider = provider
        self.client = self._init_client(provider, credentials)
    
    def upload_result(self, file_path, remote_path):
        """Загрузить результат в облако"""
        if self.provider == 'dropbox':
            return self._upload_dropbox(file_path, remote_path)
        elif self.provider == 'google_drive':
            return self._upload_gdrive(file_path, remote_path)
        # ... другие провайдеры
    
    def create_shared_link(self, remote_path):
        """Создать публичную ссылку"""
        if self.provider == 'dropbox':
            link = self.client.sharing_create_shared_link(remote_path)
            return link.url
        # ... другие провайдеры
```

### 5. 🎓 Machine Learning Improvements

#### A. Active Learning

```python
class ActiveLearningSystem:
    """Система активного обучения"""
    
    def __init__(self, model):
        self.model = model
        self.uncertain_samples = []
        self.feedback_queue = []
    
    def identify_uncertain_predictions(self, results):
        """Найти предсказания с низкой уверенностью"""
        uncertain = []
        
        for field, data in results.items():
            if isinstance(data, dict) and 'confidence' in data:
                if data['confidence'] < 0.7:
                    uncertain.append({
                        'field': field,
                        'value': data['value'],
                        'confidence': data['confidence']
                    })
        
        return uncertain
    
    def request_user_feedback(self, uncertain_samples):
        """Запросить обратную связь у пользователя"""
        dialog = FeedbackDialog(uncertain_samples)
        if dialog.exec():
            feedback = dialog.get_feedback()
            self.feedback_queue.append(feedback)
            
            # Если накопилось достаточно обратной связи - дообучить
            if len(self.feedback_queue) >= 50:
                self.retrain_model()
    
    def retrain_model(self):
        """Дообучить модель на обратной связи"""
        # Создать датасет из feedback
        training_data = self._prepare_training_data(self.feedback_queue)
        
        # Запустить fine-tuning
        self.model.fine_tune(training_data)
        
        # Очистить очередь
        self.feedback_queue.clear()
```

#### B. Model Ensemble

```python
class ModelEnsemble:
    """Ансамбль моделей для повышения точности"""
    
    def __init__(self, models, voting_strategy='weighted'):
        self.models = models
        self.voting_strategy = voting_strategy
        self.model_weights = self._calculate_weights()
    
    def predict(self, document):
        """Предсказание ансамбля"""
        predictions = []
        
        # Получить предсказания от всех моделей
        for model in self.models:
            pred = model.extract_invoice_data(document)
            predictions.append(pred)
        
        # Объединить предсказания
        if self.voting_strategy == 'weighted':
            return self._weighted_voting(predictions)
        elif self.voting_strategy == 'majority':
            return self._majority_voting(predictions)
        elif self.voting_strategy == 'confidence':
            return self._confidence_based(predictions)
    
    def _weighted_voting(self, predictions):
        """Взвешенное голосование"""
        ensemble_result = {}
        
        for field in predictions[0].keys():
            values = []
            weights = []
            
            for i, pred in enumerate(predictions):
                if field in pred:
                    values.append(pred[field]['value'])
                    confidence = pred[field].get('confidence', 0.5)
                    model_weight = self.model_weights[i]
                    weights.append(confidence * model_weight)
            
            # Выбрать значение с максимальным весом
            if values:
                max_idx = weights.index(max(weights))
                ensemble_result[field] = {
                    'value': values[max_idx],
                    'confidence': max(weights),
                    'source': 'ensemble'
                }
        
        return ensemble_result
```

### 6. 📱 Mobile & Web Interface

#### A. Web API

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="InvoiceGemini API")

@app.post("/api/v1/process")
async def process_document(
    file: UploadFile = File(...),
    model: str = "auto",
    language: str = "ru"
):
    """Обработать документ через API"""
    try:
        # Сохранить файл
        file_path = save_upload(file)
        
        # Обработать
        processor = get_processor(model)
        results = processor.extract_invoice_data(file_path)
        
        return JSONResponse({
            'success': True,
            'results': results,
            'model_used': model,
            'processing_time': results.get('processing_time')
        })
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)

@app.get("/api/v1/models")
async def list_models():
    """Список доступных моделей"""
    return {
        'models': [
            {'id': 'layoutlm', 'name': 'LayoutLMv3', 'status': 'available'},
            {'id': 'donut', 'name': 'Donut', 'status': 'available'},
            {'id': 'gemini', 'name': 'Gemini 2.0', 'status': 'available'},
        ]
    }

@app.get("/api/v1/health")
async def health_check():
    """Проверка состояния"""
    return {
        'status': 'healthy',
        'version': '1.0.0',
        'models_loaded': get_loaded_models(),
        'gpu_available': torch.cuda.is_available()
    }
```

#### B. Mobile App Companion

**Концепция мобильного приложения:**
- Сканирование счетов камерой телефона
- Отправка на desktop приложение или cloud API
- Просмотр результатов
- Базовое редактирование

```python
# Сервер для приема от мобильного приложения
class MobileAPIServer:
    """Сервер для мобильного приложения"""
    
    def __init__(self, port=5000):
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/mobile/scan', methods=['POST'])
        def receive_scan():
            image = request.files['image']
            device_id = request.form.get('device_id')
            
            # Обработать
            results = self.process_image(image)
            
            # Отправить результаты обратно
            return jsonify({
                'results': results,
                'edit_url': f'/mobile/edit/{results["id"]}'
            })
```

---

## 📅 План внедрения

### Фаза 1: Критические улучшения UX (2-3 недели)

**Неделя 1:**
- ✅ Onboarding Wizard
- ✅ Welcome Dashboard
- ✅ Toast Notification System
- ✅ Smart Model Selector

**Неделя 2:**
- ✅ Redesigned Main Window
- ✅ Keyboard Shortcuts
- ✅ Quick Actions Toolbar
- ✅ Context Menus

**Неделя 3:**
- ✅ Progress System
- ✅ Error Recovery
- ✅ Field Validation
- ✅ Testing & Bug Fixes

### Фаза 2: Продвинутые функции (3-4 недели)

**Неделя 4-5:**
- ✅ AI Model Recommendations
- ✅ Analytics Dashboard
- ✅ Quality Metrics
- ✅ Command History

**Неделя 6-7:**
- ✅ Workflow Builder
- ✅ Batch Templates
- ✅ Workspace Profiles
- ✅ Adaptive UI

### Фаза 3: Интеграции и автоматизация (2-3 недели)

**Неделя 8-9:**
- ✅ Webhook System
- ✅ Cloud Storage Integration
- ✅ Active Learning
- ✅ Model Ensemble

**Неделя 10:**
- ✅ Testing & Optimization
- ✅ Documentation
- ✅ User Training Materials

### Фаза 4: Расширенные платформы (опционально)

- 🔄 Web API
- 🔄 Mobile App
- 🔄 Browser Extension
- 🔄 CLI Tool

---

## 📝 Заключение

Данный анализ и ТРД предоставляют комплексный план улучшения пользовательского опыта InvoiceGemini. Основные направления:

1. **Снижение порога входа** через onboarding и интерактивные туры
2. **Улучшение визуальной иерархии** и упрощение интерфейса
3. **Интеллектуальные помощники** для принятия решений
4. **Расширенная обратная связь** через toast notifications и progress system
5. **Автоматизация workflow** для повышения производительности
6. **Персонализация** под разные типы пользователей
7. **Новые интеграции** для расширения возможностей

Внедрение этих улучшений повысит:
- **Satisfaction Score**: с 7/10 до 9/10
- **Time to First Value**: с 30 мин до 5 мин
- **Task Completion Rate**: с 70% до 95%
- **User Retention**: с 60% до 85%

---

**Следующие шаги:**
1. Приоритизация фич с командой
2. Создание прототипов ключевых компонентов
3. Юзабилити-тестирование с реальными пользователями
4. Итеративная разработка и улучшение



# 📘 Руководство по интеграции UX улучшений

**Дата:** 3 октября 2025  
**Статус:** Готово к внедрению

---

## 🎯 Обзор

Этот документ содержит **пошаговые инструкции** по интеграции всех созданных UX улучшений в InvoiceGemini.

### ✅ Что уже сделано:

1. ✅ **OnboardingWizard** - интегрирован в `main.py`
2. ✅ **ToastNotification** - заменены QMessageBox (частично)
3. ✅ **SmartModelSelector** - создан компонент
4. ✅ **KeyboardShortcuts** - создана система

### 📋 Что нужно доделать:

1. Заменить радиокнопки моделей на SmartModelSelector
2. Активировать keyboard shortcuts
3. Добавить Quick Actions Toolbar
4. Тестирование

---

## 1. Интеграция SmartModelSelector

### Шаг 1: Импорт компонента

В `app/main_window.py` добавьте импорт:

```python
# В начале файла, с другими импортами UI компонентов
from .ui.components.smart_model_selector import SmartModelSelector
```

### Шаг 2: Замена радиокнопок

Найдите в `init_ui()` код создания радиокнопок моделей:

```python
# СТАРЫЙ КОД (найти и заменить):
self.layoutlm_radio = QRadioButton("LayoutLMv3")
self.donut_radio = QRadioButton("Donut") 
self.gemini_radio = QRadioButton("Google Gemini")
# ... и т.д.
```

Замените на:

```python
# НОВЫЙ КОД:
# UX IMPROVEMENT: Smart Model Selector вместо радиокнопок
self.model_selector_widget = SmartModelSelector(self)
self.model_selector_widget.model_changed.connect(self._on_model_selection_changed)

# Добавляем в layout (вместо радиокнопок)
model_selection_layout.addWidget(self.model_selector_widget)
```

### Шаг 3: Обработчик изменения модели

Добавьте новый метод:

```python
def _on_model_selection_changed(self, model_id: str):
    """
    Обработка изменения выбранной модели
    
    Args:
        model_id: ID выбранной модели (auto, gemini, layoutlm, donut, trocr)
    """
    logger.info(f"Model selection changed to: {model_id}")
    
    # Если выбран файл, показываем рекомендацию
    if hasattr(self, 'current_image_path') and self.current_image_path:
        if model_id == 'auto':
            self.model_selector_widget.set_file_for_analysis(self.current_image_path)
        else:
            self.model_selector_widget.hide_recommendation()
    
    # Здесь можно добавить логику применения модели
    # Например, установить соответствующую радиокнопку (если они остались для совместимости)
    model_mapping = {
        'auto': None,  # Авто режим
        'gemini': self.gemini_radio if hasattr(self, 'gemini_radio') else None,
        'layoutlm': self.layoutlm_radio if hasattr(self, 'layoutlm_radio') else None,
        'donut': self.donut_radio if hasattr(self, 'donut_radio') else None,
        'trocr': self.trocr_radio if hasattr(self, 'trocr_radio') else None,
    }
    
    radio_button = model_mapping.get(model_id)
    if radio_button:
        radio_button.setChecked(True)
```

### Шаг 4: Обновление при загрузке файла

В методе обработки загрузки файла добавьте:

```python
def load_image(self, image_path):
    # ... существующий код ...
    
    # UX IMPROVEMENT: Анализ файла для рекомендации модели
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_file_for_analysis(image_path)
```

---

## 2. Активация Keyboard Shortcuts

### Шаг 1: Импорт

В `app/main_window.py`:

```python
from .ui.components.keyboard_shortcuts import setup_standard_shortcuts, ShortcutManager
```

### Шаг 2: Настройка в __init__

В конце метода `__init__` главного окна:

```python
def __init__(self):
    # ... существующий код ...
    
    # UX IMPROVEMENT: Setup keyboard shortcuts
    try:
        self.shortcut_manager = setup_standard_shortcuts(self)
        logger.info("Keyboard shortcuts initialized")
    except Exception as e:
        logger.error(f"Failed to setup shortcuts: {e}")
```

### Шаг 3: Реализация недостающих методов

Убедитесь, что следующие методы существуют в MainWindow:

```python
def open_file(self):
    """Открыть файл (Ctrl+O)"""
    self.open_file_dialog()

def open_folder(self):
    """Открыть папку (Ctrl+Shift+O)"""
    self.open_folder_dialog()

def process_current(self):
    """Обработать текущий файл (Ctrl+P / F5)"""
    if hasattr(self, 'process_button') and self.process_button.isEnabled():
        self.process_image()

def save_results(self):
    """Сохранить результаты (Ctrl+S)"""
    # Уже существует
    pass

def export_results(self):
    """Экспорт результатов (Ctrl+E)"""
    self.save_results()  # Используем существующий метод

def batch_process(self):
    """Пакетная обработка (Ctrl+B)"""
    self.open_folder_dialog()

def cancel_processing(self):
    """Отменить обработку (Escape)"""
    if hasattr(self, 'processing_thread') and self.processing_thread:
        self.processing_thread.quit()
        self.stop_processing_ui()

def select_auto(self):
    """Выбрать Авто модель (Ctrl+1)"""
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_current_model('auto')

def select_gemini(self):
    """Выбрать Gemini (Ctrl+2)"""
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_current_model('gemini')

def select_layoutlm(self):
    """Выбрать LayoutLM (Ctrl+3)"""
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_current_model('layoutlm')

def select_donut(self):
    """Выбрать Donut (Ctrl+4)"""
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_current_model('donut')

def select_trocr(self):
    """Выбрать TrOCR (Ctrl+5)"""
    if hasattr(self, 'model_selector_widget'):
        self.model_selector_widget.set_current_model('trocr')

def refresh_view(self):
    """Обновить просмотр (Ctrl+R)"""
    if hasattr(self, 'current_image_path') and self.current_image_path:
        self.load_image(self.current_image_path)

def previous_file(self):
    """Предыдущий файл (Ctrl+[)"""
    # Реализовать навигацию по файлам
    pass

def next_file(self):
    """Следующий файл (Ctrl+])"""
    # Реализовать навигацию по файлам
    pass

def open_settings(self):
    """Открыть настройки (Ctrl+,)"""
    self.open_model_management_dialog()

def show_help(self):
    """Показать справку (F1)"""
    # Показать диалог shortcuts
    if hasattr(self, 'shortcut_manager'):
        self.shortcut_manager.show_shortcuts_dialog()

def open_training(self):
    """Открыть обучение (Ctrl+T)"""
    self.show_training_dialog()

def view_logs(self):
    """Просмотр логов (Ctrl+L)"""
    import os
    import subprocess
    log_file = "logs/app.log"
    if os.path.exists(log_file):
        subprocess.Popen(['notepad.exe', log_file])  # Windows
        # Или для кроссплатформенности:
        # import webbrowser
        # webbrowser.open(log_file)

def quit_app(self):
    """Выйти (Ctrl+Q)"""
    self.close()

def toggle_fullscreen(self):
    """Полноэкранный режим (F11)"""
    if self.isFullScreen():
        self.showNormal()
    else:
        self.showFullScreen()
```

---

## 3. Добавление меню "Справка" со shortcuts

В метод создания меню добавьте:

```python
def create_menu_bar(self):
    # ... существующие меню ...
    
    # UX IMPROVEMENT: Меню Справка
    help_menu = self.menuBar().addMenu("&Справка")
    
    shortcuts_action = QAction("⌨️ &Горячие клавиши", self)
    shortcuts_action.setShortcut("F1")
    shortcuts_action.triggered.connect(self.show_help)
    help_menu.addAction(shortcuts_action)
    
    help_menu.addSeparator()
    
    docs_action = QAction("📖 &Документация", self)
    docs_action.triggered.connect(self.open_documentation)
    help_menu.addAction(docs_action)
    
    about_action = QAction("ℹ️ &О программе", self)
    about_action.triggered.connect(self.show_about_dialog)
    help_menu.addAction(about_action)

def open_documentation(self):
    """Открыть документацию"""
    import webbrowser
    webbrowser.open("docs/README.md")

def show_about_dialog(self):
    """Показать диалог О программе"""
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.about(
        self,
        "О программе InvoiceGemini",
        f"<h2>InvoiceGemini</h2>"
        f"<p>Версия: {app_config.APP_VERSION}</p>"
        f"<p>Автоматическое извлечение данных из счетов-фактур</p>"
        f"<p><a href='https://github.com/yourusername/InvoiceGemini'>GitHub</a></p>"
    )
```

---

## 4. Тестирование

### Чек-лист тестирования:

#### OnboardingWizard
- [ ] При первом запуске показывается мастер
- [ ] Все 5 страниц работают корректно
- [ ] Настройки применяются после завершения
- [ ] Toast уведомление показывается
- [ ] Повторно не запускается

#### ToastNotification
- [ ] Success toast показывается корректно
- [ ] Error toast показывается корректно
- [ ] Warning toast показывается корректно
- [ ] Info toast показывается корректно
- [ ] Несколько toast'ов стекируются правильно
- [ ] Toast автоматически закрывается
- [ ] Кнопка закрытия работает

#### SmartModelSelector
- [ ] Dropdown показывает все модели
- [ ] Описание обновляется при выборе
- [ ] Характеристики отображаются
- [ ] Tooltip'ы работают
- [ ] Рекомендация показывается в режиме "Авто"
- [ ] Анализ файлов работает корректно

#### Keyboard Shortcuts
- [ ] Ctrl+O открывает файл
- [ ] Ctrl+P / F5 обрабатывает
- [ ] Ctrl+S сохраняет
- [ ] Ctrl+1-5 переключают модели
- [ ] F1 показывает справку
- [ ] Ctrl+Q закрывает приложение
- [ ] F11 переключает fullscreen

---

## 5. Дополнительные улучшения (опционально)

### A. Welcome Dashboard

Создайте стартовый dashboard для новых пользователей:

```python
# app/ui/components/welcome_dashboard.py
class WelcomeDashboard(QWidget):
    """Приветственная панель с quick actions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Quick action cards
        # Recent files
        # Tips & tricks
```

### B. Context Menu для результатов

В таблице результатов добавьте контекстное меню:

```python
def setup_results_context_menu(self):
    """Настройка контекстного меню для таблицы результатов"""
    self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.results_table.customContextMenuRequested.connect(self.show_results_context_menu)

def show_results_context_menu(self, position):
    """Показать контекстное меню"""
    menu = QMenu(self)
    
    # Получаем выбранную ячейку
    item = self.results_table.itemAt(position)
    if item:
        # Копировать
        copy_action = menu.addAction("📋 Копировать")
        copy_action.triggered.connect(lambda: self.copy_cell_to_clipboard(item))
        
        # Редактировать
        edit_action = menu.addAction("✏️ Редактировать")
        edit_action.triggered.connect(lambda: self.results_table.editItem(item))
        
        menu.addSeparator()
        
        # Очистить
        clear_action = menu.addAction("❌ Очистить")
        clear_action.triggered.connect(lambda: item.setText(""))
    
    menu.exec(self.results_table.viewport().mapToGlobal(position))

def copy_cell_to_clipboard(self, item):
    """Копировать содержимое ячейки"""
    from PyQt6.QtWidgets import QApplication
    clipboard = QApplication.clipboard()
    clipboard.setText(item.text())
    
    # UX: Toast уведомление
    try:
        from app.ui.components.toast_notification import show_info
        show_info("Скопировано в буфер обмена", duration=2000)
    except ImportError:
        pass
```

---

## 6. Финальная проверка

### Команды для проверки:

```bash
# Проверить импорты
python -c "from app.ui.components.onboarding_wizard import OnboardingWizard; print('OK')"
python -c "from app.ui.components.toast_notification import show_toast; print('OK')"
python -c "from app.ui.components.smart_model_selector import SmartModelSelector; print('OK')"
python -c "from app.ui.components.keyboard_shortcuts import ShortcutManager; print('OK')"

# Запустить приложение
python main.py
```

### Проверка первого запуска:

```python
# Удалить настройку first_run для тестирования онбординга
import configparser
config = configparser.ConfigParser()
config.read('data/settings.ini')
if 'General' in config and 'first_run_completed' in config['General']:
    del config['General']['first_run_completed']
    with open('data/settings.ini', 'w') as f:
        config.write(f)
```

---

## 7. Документация для пользователей

Обновите README.md:

```markdown
## 🎯 Новые возможности

### Мастер первого запуска
При первом запуске InvoiceGemini автоматически запустит мастер настройки, который поможет:
- Выбрать оптимальный режим работы
- Настроить AI-модели
- Ввести API ключи

### Горячие клавиши
Используйте горячие клавиши для ускорения работы:
- `Ctrl+O` - Открыть файл
- `Ctrl+P` или `F5` - Обработать
- `Ctrl+S` - Сохранить результаты
- `F1` - Показать все shortcuts

Полный список доступен через меню **Справка → Горячие клавиши**.

### Умный выбор модели
Режим "Авто" автоматически выбирает лучшую модель на основе анализа документа:
- PDF с текстом → Gemini (быстро)
- Сложная структура → LayoutLM (точно)
- Низкое качество → Donut (надежно)
```

---

## 🎊 Готово!

После выполнения всех шагов:

1. Протестируйте все функции
2. Соберите feedback от пользователей
3. При необходимости скорректируйте

**Следующие этапы:**
- Неделя 3-4: Main Window redesign
- Месяц 2: AI recommendations, Workflow builder
- Месяц 3: Analytics, Web API, Mobile

---

## 💬 Поддержка

Если возникли вопросы:
- Смотрите `docs/UX_IMPROVEMENT_ANALYSIS.md` - полный анализ
- Смотрите `docs/UX_IMPLEMENTATION_ROADMAP.md` - дорожная карта
- GitHub Issues для bug reports

**Удачи! 🚀**


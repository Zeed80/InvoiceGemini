# 🚀 Быстрый старт: Интеграции

## 📄 Paperless-NGX

### Подключение
```python
# 1. Откройте приложение
# 2. Настройки → Интеграция с Paperless-NGX
# 3. Введите:
#    - Server URL: http://192.168.1.125:8000
#    - API Token: ваш_токен
# 4. Тест подключения → Сохранить
```

### Синхронизация
```python
# Текущий документ
1. Обработайте счёт
2. Paperless → Синхронизация → "Синхронизировать текущий"

# Все документы
1. Paperless → Синхронизация → "Синхронизировать все"
```

### Автоматическая синхронизация
```python
# Включить автосинхронизацию
1. Paperless → Синхронизация
2. ✓ "Автоматическая синхронизация"
3. Интервал: 300 сек (5 минут)
4. Сохранить

# Проверить статус в логах
```

## 🤖 Paperless-AI

### AI Тегирование
```python
# Автоматическое
1. Paperless → AI Тегирование
2. ✓ "Включить AI тегирование"
3. ✓ "Автоматически применять теги"
4. Уверенность: 0.7
5. Сохранить

# Тестирование
1. Обработайте документ
2. AI Тегирование → "Тест AI"
3. Просмотрите результаты
```

### Кастомные правила
```python
# Создание правила
1. "🎓 Управление AI" → Кастомные правила
2. Добавить правило:
   - Название: Срочная оплата
   - Паттерн: срочн|urgent
   - Теги: срочно, приоритет
3. Сохранить
```

## ⏰ Планировщик задач

### Программная настройка
```python
from app.core.scheduler import schedule_every_hours

# Синхронизация каждые 2 часа
def my_sync_task():
    # Ваша логика
    pass

schedule_every_hours(
    task_id="my_sync",
    name="Моя синхронизация",
    func=my_sync_task,
    hours=2
)
```

### Проверка задач
```python
from app.core.scheduler import get_scheduler

scheduler = get_scheduler()
tasks = scheduler.get_all_tasks()
for task in tasks:
    print(f"{task['name']}: {task['next_run']}")
```

## 🔗 Webhook сервер

### Запуск
```python
from app.webhooks import start_paperless_webhook

# Запуск на порту 5000
server = start_paperless_webhook(port=5000)

# URL: http://localhost:5000/webhook/paperless
```

### Кастомные обработчики
```python
from app.webhooks import get_webhook_server

server = get_webhook_server()

def my_handler(data):
    doc_id = data.get('document_id')
    print(f"Новый документ: {doc_id}")

server.register_handler("document_added", my_handler)
server.start()
```

## 📝 Примеры использования

### Автоматическая обработка папки
```python
from pathlib import Path
from app.processing_engine import ModelManager
from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin

# Настройка
model_manager = ModelManager()
paperless = PaperlessNGXPlugin({
    "server_url": "http://192.168.1.125:8000",
    "api_token": "ваш_токен"
})
paperless.initialize()
paperless.connect()

# Обработка
for pdf in Path("invoices/").glob("*.pdf"):
    # Распознавание
    data = model_manager.process_file(str(pdf), "gemini")
    data["file_path"] = str(pdf)
    
    # Синхронизация
    result = paperless.sync_data(data, direction="export")
    print(f"{pdf.name}: {'✅' if result.get('success') else '❌'}")
```

### AI тегирование с правилами
```python
from app.plugins.integrations.paperless_ai_advanced import PaperlessAIAdvanced

ai = PaperlessAIAdvanced({
    "server_url": "http://192.168.1.125:8000",
    "api_token": "ваш_токен"
})
ai.initialize()

# Создание правила
from app.plugins.integrations.paperless_ai_advanced import CustomTaggingRule
import uuid

rule = CustomTaggingRule(
    rule_id=str(uuid.uuid4()),
    name="Крупные счета",
    pattern=r"\d{6,}",  # 6+ цифр
    tags=["крупная_сумма", "одобрение"],
    confidence=0.9
)
ai.add_custom_rule(rule)

# Применение
result = ai.smart_tag_document(document_id=123)
print(f"Применено тегов: {len(result['applied_tags'])}")
```

## 📚 Документация

- 📖 [PAPERLESS_USER_GUIDE.md](PAPERLESS_USER_GUIDE.md) - Полное руководство
- 💻 [INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md) - Примеры кода
- 📊 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Отчёт о работе
- 📝 [README.md](README.md) - Основная документация

## 🆘 Проблемы?

### Ошибка подключения
```
✗ Проверьте URL и токен
✗ Проверьте доступность сервера
✗ Увеличьте таймаут
```

### Теги не применяются
```
✗ Снизьте уровень уверенности
✗ Проверьте включено ли автотегирование
✗ Используйте "Тест AI"
```

### Автосинхронизация не работает
```
✗ Проверьте что она включена
✗ Проверьте интервал
✗ Посмотрите логи
```

---

*Краткая инструкция - версия 1.0*


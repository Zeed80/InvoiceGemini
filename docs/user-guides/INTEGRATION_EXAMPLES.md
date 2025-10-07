# 🔌 Примеры использования интеграций

Этот документ содержит практические примеры использования различных интеграций в InvoiceGemini.

## 📋 Содержание

1. [Paperless-NGX API](#paperless-ngx-api)
2. [Paperless-AI API](#paperless-ai-api)
3. [Кастомные плагины](#кастомные-плагины)
4. [Автоматизация](#автоматизация)
5. [Расширенные сценарии](#расширенные-сценарии)

---

## 📄 Paperless-NGX API

### Пример 1: Базовая синхронизация документа

```python
from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin

# Инициализация плагина
config = {
    "server_url": "http://192.168.1.125:8000",
    "api_token": "your_api_token_here",
    "timeout": 30,
    "ssl_verify": True
}

plugin = PaperlessNGXPlugin(config)
plugin.initialize()
plugin.connect()

# Данные обработанного счета
invoice_data = {
    "vendor": "ООО 'Поставщик'",
    "invoice_number": "INV-2024-001",
    "date": "15.01.2024",
    "total": "125000.00",
    "tax": "25000.00",
    "category": "Услуги",
    "file_path": "/path/to/invoice.pdf"
}

# Синхронизация
result = plugin.sync_data(invoice_data, direction="export")
if result.get("success"):
    print(f"Документ синхронизирован: {result.get('document_id')}")
else:
    print(f"Ошибка: {result.get('error')}")
```

### Пример 2: Создание корреспондента

```python
# Автоматическое создание корреспондента из данных поставщика
correspondent_name = invoice_data.get("vendor")
correspondent_id = plugin.create_correspondent(correspondent_name)

if correspondent_id:
    print(f"Создан корреспондент ID: {correspondent_id}")
```

### Пример 3: Загрузка документа с метаданными

```python
# Подготовка метаданных
metadata = {
    "correspondent_id": 42,
    "document_type_id": 5,
    "title": f"Счет {invoice_data['invoice_number']}",
    "created": invoice_data["date"],
    "tags": [10, 15, 23],  # ID тегов
    "custom_fields": [
        {
            "field": 1,
            "value": invoice_data["invoice_number"]
        },
        {
            "field": 2,
            "value": invoice_data["total"]
        }
    ]
}

# Загрузка документа
doc_id = plugin.upload_document(
    invoice_data["file_path"], 
    metadata
)

print(f"Документ загружен с ID: {doc_id}")
```

### Пример 4: Массовая синхронизация

```python
# Получение всех обработанных счетов
from app.core.storage_integration import get_storage_integration

storage = get_storage_integration()
all_invoices = storage.get_all_invoices()

# Синхронизация каждого
for invoice in all_invoices:
    result = plugin.sync_data(invoice, direction="export")
    if result.get("success"):
        print(f"✅ {invoice.get('file_name')}")
    else:
        print(f"❌ {invoice.get('file_name')}: {result.get('error')}")
```

---

## 🤖 Paperless-AI API

### Пример 1: Анализ документа для тегов

```python
from app.plugins.integrations.paperless_ai_plugin import PaperlessAIPlugin

config = {
    "server_url": "http://192.168.1.125:8000",
    "api_token": "your_api_token",
    "confidence_threshold": 0.7,
    "auto_tag": True
}

ai_plugin = PaperlessAIPlugin(config)
ai_plugin.initialize()
ai_plugin.connect()

# Анализ данных счета
invoice_data = {
    "vendor": "ООО 'Поставщик'",
    "category": "Услуги",
    "total": "50000",
    "tax": "10000",
    "date": "15.01.2024"
}

tags = ai_plugin.analyze_document_for_tags(invoice_data)

for tag_name, confidence in tags:
    print(f"• {tag_name}: {confidence:.1%}")
```

**Вывод:**
```
• Категория: Услуги: 90.0%
• Поставщик: ООО 'Поставщик': 85.0%
• С НДС: 95.0%
• Год: 2024: 80.0%
• Январь: 75.0%
• Средняя сумма: 80.0%
```

### Пример 2: Создание кастомного правила

```python
from app.plugins.integrations.paperless_ai_advanced import (
    PaperlessAIAdvanced, CustomTaggingRule
)
import uuid

# Инициализация расширенного AI
advanced_ai = PaperlessAIAdvanced(config)
advanced_ai.initialize()

# Создание правила для срочных документов
rule = CustomTaggingRule(
    rule_id=str(uuid.uuid4()),
    name="Срочная оплата",
    pattern="срочн|urgent|asap",
    tags=["срочно", "высокий_приоритет"],
    enabled=True,
    confidence=0.95
)

# Добавление правила
if advanced_ai.add_custom_rule(rule):
    print("✅ Правило добавлено")
```

### Пример 3: Умное тегирование с правилами

```python
# Применение AI + кастомных правил
result = advanced_ai.smart_tag_document(
    document_id=123,
    content="Срочный счет на оплату услуг..."
)

if result["status"] == "success":
    print(f"AI теги: {result['ai_tags_count']}")
    print(f"Правила: {result['custom_rules_count']}")
    print(f"Применено: {len(result['applied_tags'])}")
    
    for tag in result['suggested_tags']:
        print(f"  • {tag['name']} ({tag['source']}) - {tag['confidence']:.1%}")
```

### Пример 4: Статистика и обучение

```python
# Получение статистики
stats = advanced_ai.get_statistics()

print(f"Документов обработано: {stats['total_documents']}")
print(f"Тегов предложено: {stats['total_tags_suggested']}")
print(f"Принятие тегов: {stats['acceptance_rate']:.1%}")

# Топ-10 тегов
for tag_name, score in stats['top_tags']:
    print(f"  • {tag_name}: {score:.2f}")

# Экспорт данных для обучения
from pathlib import Path

export_path = Path("data/ai_learning_data.json")
if advanced_ai.export_learning_data(export_path):
    print(f"✅ Данные экспортированы: {export_path}")
```

---

## 🔧 Кастомные плагины

### Пример 1: Создание простого плагина интеграции

```python
from app.plugins.base_plugin import (
    IntegrationPlugin, PluginMetadata, PluginType, 
    PluginCapability, PluginPriority
)

class CustomIntegrationPlugin(IntegrationPlugin):
    """Кастомный плагин интеграции"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Custom Integration",
            version="1.0.0",
            description="Интеграция с внешней системой",
            author="Your Name",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[PluginCapability.API],
            priority=PluginPriority.MEDIUM,
            config_schema={
                "required": ["api_url", "api_key"],
                "optional": {"timeout": 30}
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            self.api_url = self.config["api_url"]
            self.api_key = self.config["api_key"]
            return True
        except KeyError as e:
            self.set_error(f"Отсутствует параметр: {e}")
            return False
    
    def sync_data(self, data, direction="export"):
        """Синхронизация данных"""
        try:
            # Ваша логика синхронизации
            response = self._send_to_api(data)
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _send_to_api(self, data):
        """Отправка данных в API"""
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.api_url}/documents",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
```

### Пример 2: Использование кастомного плагина

```python
# Конфигурация
config = {
    "api_url": "https://api.example.com",
    "api_key": "your_secret_key",
    "timeout": 60
}

# Создание и использование
plugin = CustomIntegrationPlugin(config)

if plugin.initialize():
    result = plugin.sync_data(invoice_data)
    if result["success"]:
        print("✅ Синхронизация успешна")
else:
    print(f"❌ Ошибка: {plugin.last_error}")
```

---

## ⚙️ Автоматизация

### Пример 1: Автоматическая обработка и синхронизация

```python
import os
from pathlib import Path
from app.processing_engine import ModelManager
from app.plugins.integrations.paperless_ngx_plugin import PaperlessNGXPlugin

# Настройка
invoice_dir = Path("data/new_invoices")
paperless_config = {
    "server_url": "http://192.168.1.125:8000",
    "api_token": os.getenv("PAPERLESS_API_TOKEN")
}

# Инициализация
model_manager = ModelManager()
paperless = PaperlessNGXPlugin(paperless_config)
paperless.initialize()
paperless.connect()

# Обработка всех счетов
for invoice_file in invoice_dir.glob("*.pdf"):
    print(f"Обработка: {invoice_file.name}")
    
    # 1. Распознавание
    result = model_manager.process_file(
        str(invoice_file),
        model_type="gemini"
    )
    
    # 2. Добавление пути к файлу
    result["file_path"] = str(invoice_file)
    
    # 3. Синхронизация
    sync_result = paperless.sync_data(result, direction="export")
    
    if sync_result.get("success"):
        print(f"✅ {invoice_file.name} синхронизирован")
        # Перемещение в обработанные
        invoice_file.rename(f"data/processed/{invoice_file.name}")
    else:
        print(f"❌ Ошибка: {sync_result.get('error')}")
```

### Пример 2: Планировщик задач

```python
import schedule
import time

def sync_all_documents():
    """Функция для синхронизации всех документов"""
    # Ваша логика
    print("Синхронизация документов...")
    # ...
    print("Синхронизация завершена")

# Расписание
schedule.every().day.at("02:00").do(sync_all_documents)  # Каждый день в 2:00
schedule.every().hour.do(sync_all_documents)              # Каждый час
schedule.every(30).minutes.do(sync_all_documents)         # Каждые 30 минут

# Запуск
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Пример 3: Обработка с AI тегированием

```python
def process_and_tag(invoice_path):
    """Полный цикл: обработка + тегирование + синхронизация"""
    
    # 1. Обработка документа
    invoice_data = model_manager.process_file(invoice_path, "gemini")
    
    # 2. AI анализ для тегов
    tags = ai_plugin.analyze_document_for_tags(invoice_data)
    tag_names = [tag[0] for tag in tags if tag[1] >= 0.7]
    
    # 3. Загрузка в Paperless
    metadata = {
        "title": f"Счет {invoice_data.get('invoice_number', 'N/A')}",
        "tags": tag_names
    }
    
    invoice_data["file_path"] = invoice_path
    result = paperless.sync_data(invoice_data, direction="export")
    
    return result

# Использование
result = process_and_tag("invoice.pdf")
print(f"Статус: {'✅' if result.get('success') else '❌'}")
```

---

## 🚀 Расширенные сценарии

### Сценарий 1: Интеграция с Email

```python
import imaplib
import email
from email.header import decode_header
import tempfile

def process_email_invoices():
    """Обработка счетов из email"""
    
    # Подключение к почте
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login("your_email@gmail.com", "your_password")
    mail.select("inbox")
    
    # Поиск писем со счетами
    _, messages = mail.search(None, 'SUBJECT "Invoice" UNSEEN')
    
    for msg_num in messages[0].split():
        _, msg_data = mail.fetch(msg_num, "(RFC822)")
        email_body = msg_data[0][1]
        message = email.message_from_bytes(email_body)
        
        # Обработка вложений
        for part in message.walk():
            if part.get_content_type() == "application/pdf":
                filename = part.get_filename()
                
                # Сохранение во временный файл
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(part.get_payload(decode=True))
                    temp_path = f.name
                
                # Обработка
                result = process_and_tag(temp_path)
                print(f"Email invoice processed: {filename}")
    
    mail.close()
    mail.logout()
```

### Сценарий 2: Webhook для real-time обновлений

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/paperless', methods=['POST'])
def paperless_webhook():
    """Обработка webhook от Paperless"""
    
    data = request.json
    event_type = data.get("type")
    
    if event_type == "document_added":
        doc_id = data.get("document_id")
        print(f"Новый документ в Paperless: {doc_id}")
        
        # Импорт в InvoiceGemini
        # ...
        
    elif event_type == "document_updated":
        doc_id = data.get("document_id")
        print(f"Документ обновлен: {doc_id}")
        
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

### Сценарий 3: Batch обработка с прогрессом

```python
from tqdm import tqdm
import concurrent.futures

def batch_process_with_progress(file_paths, max_workers=4):
    """Параллельная обработка с прогресс-баром"""
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создание задач
        futures = {
            executor.submit(process_and_tag, path): path 
            for path in file_paths
        }
        
        # Обработка с прогрессом
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append((path, result))
                    pbar.set_description(f"Обработан: {Path(path).name}")
                except Exception as e:
                    print(f"Ошибка {path}: {e}")
                finally:
                    pbar.update(1)
    
    return results

# Использование
files = list(Path("invoices").glob("*.pdf"))
results = batch_process_with_progress(files, max_workers=4)

# Статистика
successful = sum(1 for _, r in results if r.get("success"))
print(f"Успешно: {successful}/{len(results)}")
```

---

## 📚 Дополнительные ресурсы

- 📖 [PAPERLESS_USER_GUIDE.md](PAPERLESS_USER_GUIDE.md) - Руководство пользователя
- 📖 [PAPERLESS_INTEGRATION_GUIDE.md](PAPERLESS_INTEGRATION_GUIDE.md) - Техническая документация
- 🔌 [PLUGIN_SYSTEM_GUIDE.md](PLUGIN_SYSTEM_GUIDE.md) - Система плагинов
- 🤖 [API Reference](docs/api/) - Справочник API

---

*Документ обновлен: 02.10.2024*


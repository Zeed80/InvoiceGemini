# 📊 Руководство по подготовке датасетов TrOCR

## 🎯 Обзор

Модуль подготовки датасетов TrOCR предоставляет полноценный инструментарий для создания высококачественных датасетов для обучения Microsoft TrOCR (Transformer-based Optical Character Recognition) моделей. 

## ✨ Возможности

### 🔧 Источники данных
- **📄 Из аннотаций счетов** - Использует JSON файлы с аннотациями InvoiceGemini
- **📁 Из структуры папок** - Автоматический парсинг папок с изображениями и аннотациями
- **🎨 Синтетический датасет** - Генерация искусственных данных для обучения
- **📋 Из готовых аннотаций** - Импорт внешних датасетов

### ⚙️ Конфигурация
- Поддержка всех моделей TrOCR (base/large, printed/handwritten)
- Настраиваемые аугментации изображений
- Гибкое разделение на train/validation/test
- Автоматическая валидация данных

### 🔍 Качество
- Проверка целостности изображений
- Валидация текстовых аннотаций
- Статистика датасета
- Рекомендации по улучшению

## 🚀 Быстрый старт

### 1. Запуск через интерфейс

1. Откройте InvoiceGemini
2. Перейдите в **Меню → Обучение → 📊 TrOCR Датасет**
3. Выберите тип источника данных
4. Настройте параметры
5. Нажмите **"🚀 Создать датасет"**

### 2. Программное использование

```python
from app.training.trocr_dataset_preparator import (
    TrOCRDatasetPreparator, 
    TrOCRDatasetConfig
)

# Создание конфигурации
config = TrOCRDatasetConfig(
    model_name="microsoft/trocr-base-printed",
    max_target_length=128,
    image_size=(384, 384),
    enable_augmentation=True
)

# Создание препаратора
preparator = TrOCRDatasetPreparator(config)

# Создание синтетического датасета
datasets = preparator.prepare_synthetic_dataset(
    output_path="data/my_trocr_dataset",
    num_samples=10000
)
```

## 📋 Подробное руководство

### Тип 1: Из аннотаций счетов

Использует существующие аннотации InvoiceGemini для создания TrOCR датасета.

**Требования:**
- Папка с изображениями счетов
- JSON файл с аннотациями в формате InvoiceGemini

**Пример структуры:**
```
invoices/
├── images/
│   ├── invoice_001.png
│   ├── invoice_002.jpg
│   └── ...
└── annotations.json
```

**Формат аннотаций:**
```json
[
  {
    "image_file": "invoice_001.png",
    "extracted_data": {
      "invoice_number": "INV-2023-001",
      "date": "2023-01-15",
      "supplier": "ООО Поставщик",
      "total_amount": "15000.00"
    }
  }
]
```

**Процесс:**
1. Выберите "Из аннотаций счетов (JSON)"
2. Укажите папку с изображениями
3. Выберите JSON файл с аннотациями
4. Система создаст пары изображение-текст для каждого поля

### Тип 2: Из структуры папок

Автоматически парсит папку с данными в стандартном формате.

**Ожидаемая структура:**
```
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
└── annotations.txt    # или .json, .csv
```

**Форматы аннотаций:**

**TXT формат:**
```
img001.jpg    Счет-фактура № 12345
img002.png    ООО "Ромашка"
```

**JSON формат:**
```json
{
  "img001.jpg": "Счет-фактура № 12345",
  "img002.png": "ООО \"Ромашка\""
}
```

**CSV формат:**
```csv
image,text
img001.jpg,"Счет-фактура № 12345"
img002.png,"ООО ""Ромашка"""
```

### Тип 3: Синтетический датасет

Генерирует искусственные изображения с текстом для обучения.

**Особенности:**
- Различные шрифты и размеры
- Случайные фоны и цвета  
- Искажения и шум
- Поддержка русского текста

**Настройки:**
- Количество примеров (100 - 100,000)
- Кастомные тексты для генерации
- Параметры аугментаций

**Пример кастомных текстов:**
```python
russian_texts = [
    "ООО \"Ромашка\"",
    "Счет-фактура №",
    "Дата:",
    "Поставщик:",
    "НДС 20%:",
    "Итого:"
]
```

## ⚙️ Конфигурация

### Основные параметры

```python
config = TrOCRDatasetConfig(
    # Базовая модель
    model_name="microsoft/trocr-base-stage1",  # Рекомендуется для fine-tuning
    
    # Параметры текста
    max_target_length=128,           # Максимальная длина токенов
    min_text_length=1,               # Минимальная длина символов
    max_text_length=200,             # Максимальная длина символов
    
    # Параметры изображений
    image_size=(384, 384),           # Размер изображений
    min_image_size=(32, 32),         # Минимальный размер
    
    # Аугментации
    enable_augmentation=True,
    brightness_range=(0.7, 1.3),
    contrast_range=(0.8, 1.2),
    gaussian_blur_prob=0.3
)
```

### Модели TrOCR

| Модель | Описание | Рекомендации |
|--------|----------|--------------|
| `trocr-base-stage1` | 🎯 **Базовая** для fine-tuning | Лучший выбор для обучения на своих данных |
| `trocr-base-printed` | 📄 Печатный текст | Для документов и счетов |
| `trocr-base-handwritten` | ✍️ Рукописный текст | Для рукописных форм |
| `trocr-large-printed` | 📄 Большая для печатного | Более точная, но медленная |
| `trocr-large-handwritten` | ✍️ Большая для рукописного | Максимальная точность |

### Размеры изображений

| Размер | Применение | Память | Скорость |
|--------|------------|--------|----------|
| 224×224 | 🏃 Быстрое обучение | Низкая | Высокая |
| 384×384 | ⚖️ **Рекомендуемый** | Средняя | Средняя |
| 448×448 | 🔍 Детализированный текст | Высокая | Низкая |
| 512×512 | 📊 Сложные документы | Очень высокая | Очень низкая |

## 🔍 Валидация и качество

### Автоматические проверки

- ✅ Существование файлов изображений
- ✅ Читаемость изображений
- ✅ Размеры изображений (минимальные требования)
- ✅ Форматы изображений (поддерживаемые)
- ✅ Длина текстовых аннотаций
- ✅ Кодировка текста (UTF-8)

### Рекомендации по качеству

**Изображения:**
- Разрешение не менее 32×32 пикселей
- Четкие, контрастные изображения
- Поддерживаемые форматы: JPG, PNG, BMP, TIFF

**Тексты:**
- Длина от 1 до 200 символов
- Корректная кодировка UTF-8
- Соответствие содержимому изображения

**Датасет:**
- Минимум 100 примеров для обучения
- Сбалансированное распределение длин текста
- Разнообразие стилей и шрифтов

## 📊 Структура выходных данных

### Организация папок

```
output_dataset/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   ├── 000002.png
│   │   └── ...
│   ├── annotations.json
│   └── dataset.pt         # PyTorch Dataset
├── validation/
│   ├── images/
│   ├── annotations.json
│   └── dataset.pt
├── test/
│   ├── images/
│   ├── annotations.json
│   └── dataset.pt
└── metadata.json          # Метаданные датасета
```

### Формат метаданных

```json
{
  "config": {
    "model_name": "microsoft/trocr-base-stage1",
    "max_target_length": 128,
    "image_size": [384, 384],
    "enable_augmentation": true
  },
  "statistics": {
    "total_samples": 1000,
    "splits": {
      "train": 800,
      "validation": 100, 
      "test": 100
    },
    "text_lengths": {
      "min": 5,
      "max": 45,
      "avg": 22.3
    }
  },
  "created_at": "2025-01-17T10:30:00",
  "transformers_available": true
}
```

## 🔧 Программное использование

### Основные классы

```python
from app.training.trocr_dataset_preparator import (
    TrOCRDatasetPreparator,      # Основной класс
    TrOCRDatasetConfig,          # Конфигурация
    TrOCRCustomDataset,          # PyTorch Dataset
    create_synthetic_trocr_dataset,  # Упрощенная функция
    create_trocr_dataset_from_invoices  # Из аннотаций счетов
)
```

### Примеры использования

#### Создание из аннотаций счетов

```python
datasets = create_trocr_dataset_from_invoices(
    images_folder="data/invoices/images",
    annotations_file="data/invoices/annotations.json",
    output_path="data/trocr_invoice_dataset"
)
```

#### Создание синтетического датасета

```python
config = TrOCRDatasetConfig(
    model_name="microsoft/trocr-base-printed",
    enable_augmentation=True
)

datasets = create_synthetic_trocr_dataset(
    output_path="data/synthetic_trocr",
    num_samples=5000,
    config=config
)
```

#### Загрузка готового датасета

```python
preparator = TrOCRDatasetPreparator()

# Загрузка split'а
train_dataset = preparator.load_prepared_dataset(
    "data/my_trocr_dataset", 
    split="train"
)

# Информация о датасете
info = preparator.get_dataset_info("data/my_trocr_dataset")
print(f"Всего примеров: {info['statistics']['total_samples']}")
```

#### Использование в PyTorch

```python
from torch.utils.data import DataLoader

# Создание DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True,
    num_workers=2
)

# Итерация по батчам
for batch in train_loader:
    pixel_values = batch["pixel_values"]  # [B, C, H, W]
    labels = batch["labels"]              # [B, seq_len]
    
    # Ваш код обучения...
```

## 🎨 Аугментации

### Настройка аугментаций

```python
config = TrOCRDatasetConfig(
    enable_augmentation=True,
    
    # Цветовые искажения
    brightness_range=(0.7, 1.3),      # Яркость ±30%
    contrast_range=(0.8, 1.2),        # Контраст ±20%
    saturation_range=(0.8, 1.2),      # Насыщенность ±20%
    hue_range=(-0.1, 0.1),            # Оттенок ±10%
    
    # Размытие
    gaussian_blur_prob=0.3,           # 30% вероятность
    gaussian_blur_kernel=(3, 7),      # Размер ядра 3-7
    gaussian_blur_sigma=(0.1, 2.0)    # Сигма 0.1-2.0
)
```

### Рекомендации по аугментациям

**Для печатного текста:**
- Умеренные цветовые искажения
- Небольшое размытие (имитация сканирования)
- Без поворотов (текст всегда горизонтальный)

**Для рукописного текста:**
- Более агрессивные искажения
- Имитация различного освещения
- Легкие деформации

**Для счетов и документов:**
- Имитация сканирования и фотографирования
- Различные условия освещения
- Легкие перспективные искажения

## 🔧 Отладка и устранение проблем

### Частые ошибки

#### 1. Ошибка импорта transformers

```
⚠️ Transformers не установлен. Установите: pip install transformers torch torchvision
```

**Решение:**
```bash
pip install transformers torch torchvision pillow
```

#### 2. Недостаточно памяти

```
RuntimeError: CUDA out of memory
```

**Решение:**
- Уменьшите размер изображений
- Используйте меньший batch_size
- Отключите аугментации при создании

#### 3. Некорректные аннотации

```
ValueError: Текст слишком длинный: 250 символов (макс: 200)
```

**Решение:**
- Увеличьте `max_text_length` в конфигурации
- Очистите аннотации от лишних символов

#### 4. Отсутствующие изображения

```
FileNotFoundError: Изображение не найдено: image.jpg
```

**Решение:**
- Проверьте пути к изображениям
- Убедитесь в корректности аннотаций

### Режим отладки

```python
import logging

# Включение детального логирования
logging.basicConfig(level=logging.DEBUG)

# Создание препаратора с отладкой
preparator = TrOCRDatasetPreparator(config)
```

### Проверка созданного датасета

```python
# Загрузка и проверка
dataset = preparator.load_prepared_dataset("path/to/dataset", "train")

print(f"Размер датасета: {len(dataset)}")

# Проверка первого примера
sample = dataset[0]
print(f"Форма изображения: {sample['pixel_values'].shape}")
print(f"Форма меток: {sample['labels'].shape}")

# Статистика
info = preparator.get_dataset_info("path/to/dataset")
print(info['statistics'])
```

## 🚀 Интеграция с обучением

### Использование в TrOCR Trainer

```python
from app.training.trocr_trainer import TrOCRTrainer

# Загрузка датасета
train_dataset = preparator.load_prepared_dataset("path/to/dataset", "train")
val_dataset = preparator.load_prepared_dataset("path/to/dataset", "validation")

# Создание тренера
trainer = TrOCRTrainer(
    model_name="microsoft/trocr-base-stage1",
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Обучение
trainer.train(
    output_dir="models/my_trocr_model",
    num_epochs=5,
    batch_size=8
)
```

### Мониторинг обучения

Датасеты TrOCR полностью совместимы с системой мониторинга InvoiceGemini:

- 📊 Метрики в реальном времени
- 📈 Графики loss и accuracy
- 📝 Детальные логи обучения
- 💾 Автоматическое сохранение моделей

## 📚 Дополнительные ресурсы

### Документация
- [TrOCR Paper](https://arxiv.org/abs/2109.10282) - Оригинальная статья Microsoft
- [Hugging Face TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) - Официальная документация
- [Fine-tuning Guide](https://learnopencv.com/fine-tuning-trocr-training-trocr-to-recognize-curved-text/) - Руководство по fine-tuning

### Примеры датасетов
- SCUT-CTW1500 - изогнутый текст
- IAM Handwriting Database - рукописный текст  
- FUNSD - понимание форм
- CORD - receipts (чеки)

### Сообщество
- [Hugging Face Forums](https://discuss.huggingface.co/c/transformers/9) - Обсуждения TrOCR
- [Microsoft Research](https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/) - Исследования Microsoft

---

## ⭐ Заключение

Модуль подготовки датасетов TrOCR предоставляет все необходимые инструменты для создания высококачественных датасетов и успешного обучения TrOCR моделей. 

**Ключевые преимущества:**
- 🚀 Простота использования
- 🔧 Гибкая конфигурация  
- 📊 Автоматическая валидация
- 🎨 Продвинутые аугментации
- 💾 Совместимость с PyTorch
- 🔍 Подробная статистика

Успешного обучения! 🎉 
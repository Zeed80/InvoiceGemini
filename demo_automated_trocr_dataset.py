#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Демонстрация автоматизированного создания TrOCR датасетов с LLM Gemini

Этот скрипт демонстрирует новые возможности InvoiceGemini:
- Автоматическая разметка изображений с помощью LLM Gemini
- Создание синтетических примеров высокого качества
- Полностью автоматизированный пайплайн создания датасетов TrOCR

Запуск:
    python demo_automated_trocr_dataset.py

Требования:
    - API ключ Google Gemini (настроенный в InvoiceGemini)
    - Достаточно места на диске (рекомендуется 2-5 ГБ)
    - Python 3.8+, PyQt6, Transformers
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Настройка логирования для демо"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_automated_trocr.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def check_requirements():
    """Проверка требований для демо"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Проверка требований...")
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        logger.error("❌ Требуется Python 3.8+")
        return False
    
    # Проверяем необходимые модули
    required_modules = [
        'PyQt6', 'torch', 'transformers', 'PIL', 'numpy', 'pandas'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} доступен")
        except ImportError:
            logger.error(f"❌ {module} не установлен")
            return False
    
    # Проверяем доступность Gemini
    try:
        import google.generativeai as genai
        logger.info("✅ Google Generative AI доступен")
    except ImportError:
        logger.error("❌ google-generativeai не установлен")
        return False
    
    # Проверяем Enhanced TrOCR модуль
    try:
        from app.training.enhanced_trocr_dataset_preparator import (
            EnhancedTrOCRDatasetPreparator, 
            EnhancedTrOCRConfig
        )
        logger.info("✅ Enhanced TrOCR Dataset Preparator доступен")
    except ImportError as e:
        logger.error(f"❌ Enhanced TrOCR Dataset Preparator недоступен: {e}")
        return False
    
    logger.info("✅ Все требования выполнены!")
    return True

def check_api_key():
    """Проверка API ключа Gemini"""
    logger = logging.getLogger(__name__)
    
    try:
        from app.settings_manager import settings_manager
        api_key = settings_manager.get_gemini_api_key()
        
        if api_key:
            logger.info("✅ API ключ Gemini найден")
            return True
        else:
            logger.error("❌ API ключ Gemini не настроен")
            logger.info("💡 Настройте API ключ в InvoiceGemini: Настройки → LLM провайдеры")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка проверки API ключа: {e}")
        return False

def create_demo_images():
    """Создает демонстрационные изображения для тестирования"""
    logger = logging.getLogger(__name__)
    
    logger.info("🎨 Создание демонстрационных изображений...")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Создаем временную папку для демо изображений
        demo_images_dir = project_root / "demo_images"
        demo_images_dir.mkdir(exist_ok=True)
        
        # Шаблоны для создания демо изображений
        demo_templates = [
            {
                "filename": "invoice_001.png",
                "content": [
                    "ООО \"ТехноСервис\"",
                    "ИНН: 7722334455",
                    "Счет-фактура №SF-2024-001",
                    "от 15 января 2024 г.",
                    "",
                    "Покупатель: ООО \"СтройМир\"",
                    "ИНН: 7733445566",
                    "",
                    "Наименование: Крепежные элементы М8",
                    "Количество: 100 шт.",
                    "Цена: 15,00 руб.",
                    "Сумма: 1 500,00 руб.",
                    "",
                    "Итого без НДС: 1 500,00 руб.",
                    "НДС 20%: 300,00 руб.",
                    "Всего к оплате: 1 800,00 руб."
                ]
            },
            {
                "filename": "receipt_001.png",
                "content": [
                    "Магазин \"Продукты\"",
                    "г. Москва, ул. Ленина, 15",
                    "Чек №000123456",
                    "15.01.2024 14:25",
                    "",
                    "Хлеб белый          45,90",
                    "Молоко 1л           89,50",
                    "Сыр Российский     245,00",
                    "Итого:             380,40",
                    "",
                    "Наличными:         400,00",
                    "Сдача:              19,60",
                    "",
                    "Спасибо за покупку!"
                ]
            },
            {
                "filename": "document_001.png",
                "content": [
                    "ДОКУМЕНТ №DOC-2024-001",
                    "Дата создания: 15.01.2024",
                    "",
                    "ТЕХНИЧЕСКОЕ ЗАДАНИЕ",
                    "",
                    "1. Общие требования:",
                    "   - Разработка системы",
                    "   - Тестирование функций",
                    "   - Документирование",
                    "",
                    "2. Сроки выполнения:",
                    "   Начало: 16.01.2024",
                    "   Окончание: 30.01.2024",
                    "",
                    "Подпись: _________________",
                    "Дата: ___________________"
                ]
            }
        ]
        
        created_images = []
        
        for template in demo_templates:
            # Создаем изображение
            width, height = 600, 800
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            
            # Пытаемся загрузить шрифт
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # Размещаем текст
            y_offset = 50
            line_height = 25
            
            for line in template["content"]:
                if line.strip():
                    x_offset = 30
                    draw.text((x_offset, y_offset), line, fill='black', font=font)
                y_offset += line_height
            
            # Сохраняем изображение
            image_path = demo_images_dir / template["filename"]
            image.save(image_path, "PNG", quality=95)
            created_images.append(str(image_path))
            
            logger.info(f"✅ Создано: {template['filename']}")
        
        logger.info(f"🎨 Создано {len(created_images)} демонстрационных изображений")
        return created_images
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания демо изображений: {e}")
        return []

def run_automated_demo():
    """Запуск демонстрации автоматизированного создания датасета"""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Запуск демонстрации автоматизированного создания TrOCR датасета")
    
    try:
        from app.training.enhanced_trocr_dataset_preparator import (
            EnhancedTrOCRDatasetPreparator,
            EnhancedTrOCRConfig,
            create_automated_trocr_dataset
        )
        
        # Создаем демо изображения
        demo_images = create_demo_images()
        if not demo_images:
            logger.error("❌ Не удалось создать демо изображения")
            return False
        
        # Конфигурация для демо
        config = EnhancedTrOCRConfig(
            enable_llm_annotation=True,
            llm_model="models/gemini-2.0-flash-exp",
            max_llm_requests_per_minute=30,  # Консервативный лимит для демо
            llm_confidence_threshold=0.7,
            image_size=(384, 384),
            enable_augmentation=True,
            max_target_length=256,
            # Настройки для демо
            min_text_length_chars=5,
            max_text_length_chars=1000,
            enable_quality_filter=True
        )
        
        # Выходная папка для демо
        output_path = project_root / "demo_trocr_dataset"
        
        logger.info("🔧 Конфигурация демо:")
        logger.info(f"   • Исходных изображений: {len(demo_images)}")
        logger.info(f"   • Синтетических примеров: 100")
        logger.info(f"   • LLM модель: {config.llm_model}")
        logger.info(f"   • Выходная папка: {output_path}")
        
        # Callback для отображения прогресса
        def progress_callback(progress):
            if progress % 10 == 0:  # Выводим каждые 10%
                logger.info(f"📊 Прогресс: {progress}%")
        
        # Запускаем автоматизированное создание
        logger.info("🤖 Начинаем автоматизированное создание датасета...")
        
        datasets = create_automated_trocr_dataset(
            source_images=demo_images,
            output_path=str(output_path),
            num_synthetic=100,  # Небольшое количество для демо
            config=config,
            progress_callback=progress_callback
        )
        
        # Результаты
        logger.info("🎉 Автоматизированное создание завершено!")
        logger.info("📊 Созданные датасеты:")
        
        for split_name, split_path in datasets.items():
            logger.info(f"   • {split_name}: {split_path}")
            
            # Подсчитываем файлы в датасете
            try:
                split_dir = Path(split_path)
                if split_dir.exists():
                    files_count = len(list(split_dir.glob("*.json")))
                    logger.info(f"     Файлов: {files_count}")
            except:
                pass
        
        # Анализ созданного датасета
        logger.info("🔍 Анализ созданного датасета:")
        
        try:
            metadata_file = output_path / "enhanced_metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                quality_stats = metadata.get("quality_stats", {})
                if quality_stats:
                    logger.info(f"   • Средний confidence: {quality_stats.get('avg_confidence', 0):.2f}")
                    logger.info(f"   • Средняя длина текста: {quality_stats.get('avg_text_length', 0):.0f} символов")
                    
                annotation_sources = metadata.get("annotation_sources", {})
                if annotation_sources:
                    logger.info("   • Источники аннотаций:")
                    for source, count in annotation_sources.items():
                        logger.info(f"     - {source}: {count}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось проанализировать метаданные: {e}")
        
        logger.info("✅ Демонстрация успешно завершена!")
        logger.info(f"📁 Результаты сохранены в: {output_path}")
        logger.info("💡 Созданный датасет готов для обучения TrOCR модели")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время демонстрации: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Главная функция демо"""
    print("🤖 InvoiceGemini - Демонстрация автоматизированного создания TrOCR датасетов")
    print("=" * 80)
    
    # Настройка логирования
    logger = setup_logging()
    
    # Проверка требований
    if not check_requirements():
        print("❌ Не выполнены требования для запуска демо")
        print("💡 Установите недостающие зависимости и попробуйте снова")
        return 1
    
    # Проверка API ключа
    if not check_api_key():
        print("❌ API ключ Gemini не настроен")
        print("💡 Настройте API ключ в InvoiceGemini и попробуйте снова")
        return 1
    
    # Запуск демо
    print("\n🚀 Запуск демонстрации...")
    success = run_automated_demo()
    
    if success:
        print("\n🎉 Демонстрация успешно завершена!")
        print("📊 Автоматизированный TrOCR датасет создан с помощью LLM Gemini")
        print("💡 Теперь вы можете использовать созданный датасет для обучения модели")
        return 0
    else:
        print("\n❌ Демонстрация завершилась с ошибками")
        print("📋 Проверьте логи для получения подробной информации")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
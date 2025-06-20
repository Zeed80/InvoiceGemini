#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Демонстрация модуля подготовки датасетов TrOCR

Этот скрипт показывает как использовать TrOCRDatasetPreparator
для создания различных типов датасетов.
"""

import os
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trocr_dataset_demo.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Главная функция демонстрации"""
    
    print("🚀 Демонстрация TrOCR Dataset Preparator")
    print("=" * 50)
    
    try:
        # Импортируем модуль
        from app.training.trocr_dataset_preparator import (
            TrOCRDatasetPreparator, 
            TrOCRDatasetConfig,
            create_synthetic_trocr_dataset
        )
        
        print("✅ Модуль TrOCR Dataset Preparator успешно загружен")
        
        # Создаем конфигурацию
        config = TrOCRDatasetConfig(
            model_name="microsoft/trocr-base-stage1",
            max_target_length=128,
            image_size=(384, 384),
            enable_augmentation=True
        )
        
        print(f"📋 Конфигурация создана:")
        print(f"   • Модель: {config.model_name}")
        print(f"   • Макс. длина текста: {config.max_target_length}")
        print(f"   • Размер изображения: {config.image_size}")
        print(f"   • Аугментации: {config.enable_augmentation}")
        
        # Демонстрация 1: Создание синтетического датасета
        print("\n🎨 Демонстрация 1: Синтетический датасет")
        print("-" * 30)
        
        output_dir = "data/demo_trocr_dataset"
        
        # Создаем папку если не существует
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Создаем препаратор
        preparator = TrOCRDatasetPreparator(config)
        
        # Русские тексты для генерации
        russian_texts = [
            "ООО \"Ромашка\"",
            "Счет-фактура №",
            "Дата:",
            "Поставщик:",
            "Покупатель:", 
            "Сумма к оплате:",
            "НДС 20%:",
            "Итого:",
            "ИНН:",
            "КПП:",
            "Банк:",
            "БИК:",
            "Расчетный счет:",
            "Корр. счет:",
            "Директор",
            "Главный бухгалтер"
        ]
        
        logger.info("Создание синтетического датасета...")
        
        # Создаем небольшой синтетический датасет
        synthetic_datasets = preparator.prepare_synthetic_dataset(
            output_path=output_dir + "/synthetic",
            num_samples=50,  # Небольшой размер для демо
            text_sources=russian_texts
        )
        
        print("✅ Синтетический датасет создан:")
        for split, path in synthetic_datasets.items():
            print(f"   • {split}: {path}")
        
        # Демонстрация 2: Информация о датасете
        print("\n📊 Демонстрация 2: Информация о датасете")
        print("-" * 40)
        
        try:
            dataset_info = preparator.get_dataset_info(output_dir + "/synthetic")
            print("📈 Статистика датасета:")
            print(f"   • Всего примеров: {dataset_info['statistics']['total_samples']}")
            print(f"   • Splits: {list(dataset_info['statistics']['splits'].keys())}")
            print(f"   • Длина текста (мин/макс/сред): {dataset_info['statistics']['text_lengths']['min']}/{dataset_info['statistics']['text_lengths']['max']}/{dataset_info['statistics']['text_lengths']['avg']:.1f}")
            print(f"   • Создан: {dataset_info['created_at']}")
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о датасете: {e}")
        
        # Демонстрация 3: Загрузка датасета
        print("\n💾 Демонстрация 3: Загрузка датасета")
        print("-" * 35)
        
        try:
            # Загружаем тренировочный датасет
            train_dataset = preparator.load_prepared_dataset(
                output_dir + "/synthetic", 
                split="train"
            )
            
            if hasattr(train_dataset, '__len__'):
                print(f"✅ Тренировочный датасет загружен: {len(train_dataset)} примеров")
                
                # Показываем первый пример
                if len(train_dataset) > 0:
                    try:
                        first_sample = train_dataset[0]
                        print("📋 Первый пример:")
                        if isinstance(first_sample, dict):
                            for key, value in first_sample.items():
                                if key == "pixel_values" and hasattr(value, 'shape'):
                                    print(f"   • {key}: tensor{value.shape}")
                                elif key == "labels" and hasattr(value, 'shape'):
                                    print(f"   • {key}: tensor{value.shape}")
                                else:
                                    print(f"   • {key}: {str(value)[:100]}...")
                    except Exception as e:
                        logger.warning(f"Ошибка получения примера: {e}")
            else:
                print(f"✅ Датасет загружен (формат: {type(train_dataset)})")
                
        except Exception as e:
            logger.warning(f"Не удалось загрузить датасет: {e}")
        
        # Демонстрация 4: Простая функция создания
        print("\n🛠️ Демонстрация 4: Упрощенная функция")
        print("-" * 40)
        
        simple_datasets = create_synthetic_trocr_dataset(
            output_path=output_dir + "/simple_synthetic",
            num_samples=20,
            config=config
        )
        
        print("✅ Простой синтетический датасет создан:")
        for split, path in simple_datasets.items():
            print(f"   • {split}: {path}")
        
        print("\n🎉 Демонстрация завершена успешно!")
        print("\n📁 Созданные файлы:")
        print(f"   • Логи: trocr_dataset_demo.log")
        print(f"   • Датасеты: {output_dir}")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("   Убедитесь, что модуль находится в правильном месте")
    except Exception as e:
        logger.error(f"Ошибка демонстрации: {e}")
        print(f"❌ Ошибка: {e}")


def demo_configuration():
    """Демонстрация различных конфигураций"""
    
    print("\n⚙️ Демонстрация конфигураций TrOCR")
    print("-" * 40)
    
    try:
        from app.training.trocr_dataset_preparator import TrOCRDatasetConfig
        
        # Конфигурация для печатного текста
        printed_config = TrOCRDatasetConfig(
            model_name="microsoft/trocr-base-printed",
            max_target_length=64,
            image_size=(224, 224),
            enable_augmentation=False
        )
        
        print("📄 Конфигурация для печатного текста:")
        print(f"   • Модель: {printed_config.model_name}")
        print(f"   • Размер изображения: {printed_config.image_size}")
        print(f"   • Аугментации: {printed_config.enable_augmentation}")
        
        # Конфигурация для рукописного текста
        handwritten_config = TrOCRDatasetConfig(
            model_name="microsoft/trocr-base-handwritten",
            max_target_length=256,
            image_size=(448, 448),
            enable_augmentation=True,
            brightness_range=(0.5, 1.5),
            gaussian_blur_prob=0.5
        )
        
        print("\n✍️ Конфигурация для рукописного текста:")
        print(f"   • Модель: {handwritten_config.model_name}")
        print(f"   • Размер изображения: {handwritten_config.image_size}")
        print(f"   • Яркость: {handwritten_config.brightness_range}")
        print(f"   • Вероятность размытия: {handwritten_config.gaussian_blur_prob}")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта конфигурации: {e}")


if __name__ == "__main__":
    main()
    demo_configuration() 
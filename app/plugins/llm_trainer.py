"""
LLM Trainer для обучения локальных LLM моделей в InvoiceGemini
Поддерживает LoRA/QLoRA fine-tuning с качественным мониторингом
"""
import os
import json
import torch
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time
from datetime import datetime

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling
    )
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import accelerate
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("[WARN] Обучающие библиотеки не установлены. Установите: pip install transformers datasets peft accelerate")

@dataclass
class TrainingMetrics:
    """Класс для хранения метрик обучения"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    gpu_memory_used: float = 0.0
    samples_per_second: float = 0.0

class LLMTrainer:
    """
    Тренер для обучения LLM моделей с LoRA/QLoRA
    """
    
    def __init__(self, plugin_instance, progress_callback: Optional[Callable] = None):
        """
        Инициализация тренера
        
        Args:
            plugin_instance: Экземпляр LLM плагина
            progress_callback: Функция для обновления прогресса
        """
        self.plugin = plugin_instance
        self.progress_callback = progress_callback
        self.training_active = False
        self.should_stop = False
        
        # Инициализация компонентов обучения
        self.trainer = None
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Метрики
        self.current_metrics = TrainingMetrics()
        self.training_history = []
        
        if not TRAINING_AVAILABLE:
            raise ImportError("Требуются библиотеки для обучения: transformers, datasets, peft, accelerate")
    
    def prepare_dataset_with_gemini(self, image_paths: List[str], gemini_processor, output_path: str) -> str:
        """
        Подготавливает датасет для обучения с помощью Gemini API
        
        Args:
            image_paths: Пути к изображениям счетов
            gemini_processor: Экземпляр GeminiProcessor для генерации аннотаций
            output_path: Путь для сохранения датасета
            
        Returns:
            str: Путь к подготовленному датасету
        """
        from datetime import datetime
        import time
        
        start_time = datetime.now()
        self._log("=" * 70)
        self._log("📊 ПОДГОТОВКА ДАТАСЕТА С GEMINI")
        self._log("=" * 70)
        self._log(f"📅 Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"📁 Выходная директория: {output_path}")
        self._log(f"🖼️ Количество изображений: {len(image_paths)}")
        
        # Валидация входных данных
        if not gemini_processor:
            self._log("❌ ОШИБКА: GeminiProcessor не предоставлен")
            raise ValueError("GeminiProcessor не предоставлен")
        
        if not image_paths:
            self._log("❌ ОШИБКА: Список изображений пуст")
            raise ValueError("Список изображений не может быть пустым")
        
        # Проверяем существование изображений
        valid_images = []
        invalid_images = []
        total_size = 0
        
        self._log(f"\n🔍 ПРОВЕРКА ВХОДНЫХ ИЗОБРАЖЕНИЙ:")
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    size = os.path.getsize(image_path)
                    total_size += size
                    valid_images.append(image_path)
                    self._log(f"  ✅ {os.path.basename(image_path)} ({size/1024:.1f} KB)")
                except Exception as e:
                    invalid_images.append((image_path, str(e)))
                    self._log(f"  ❌ {os.path.basename(image_path)} - ошибка чтения: {e}")
            else:
                invalid_images.append((image_path, "файл не найден"))
                self._log(f"  ❌ {os.path.basename(image_path)} - файл не найден")
        
        self._log(f"\n📊 СТАТИСТИКА ВХОДНЫХ ДАННЫХ:")
        self._log(f"  ✅ Валидных изображений: {len(valid_images)}")
        self._log(f"  ❌ Невалидных изображений: {len(invalid_images)}")
        self._log(f"  📏 Общий размер: {total_size/1024/1024:.1f} MB")
        
        if len(valid_images) == 0:
            self._log("❌ КРИТИЧЕСКАЯ ОШИБКА: Нет валидных изображений для обработки")
            raise ValueError("Нет валидных изображений для обработки")
        
        # Создаем выходную директорию
        os.makedirs(output_path, exist_ok=True)
        self._log(f"✅ Выходная директория создана/проверена: {output_path}")
        
        # Инициализируем счетчики
        training_data = []
        successful_count = 0
        failed_count = 0
        processing_times = []
        gemini_errors = []
        
        self._log(f"\n🔄 НАЧАЛО ОБРАБОТКИ ИЗОБРАЖЕНИЙ:")
        
        for i, image_path in enumerate(valid_images, 1):
            try:
                if self.should_stop:
                    self._log(f"🛑 Получен сигнал остановки на изображении {i}")
                    break
                
                process_start = time.time()
                
                self._log(f"\n📷 [{i}/{len(valid_images)}] {os.path.basename(image_path)}")
                self._log(f"   📁 Путь: {image_path}")
                
                # Обновляем прогресс
                progress = (i / len(valid_images)) * 100
                self._update_progress(i, len(valid_images), f"Обработка {os.path.basename(image_path)}")
                self._log(f"   📈 Прогресс: {progress:.1f}%")
                
                # Дополнительная информация о файле
                file_size = os.path.getsize(image_path) / 1024  # KB
                self._log(f"   📏 Размер: {file_size:.1f} KB")
                
                # Обрабатываем изображение с помощью Gemini
                self._log(f"   🤖 Отправка в Gemini...")
                result = gemini_processor.process_image(image_path)
                
                process_time = time.time() - process_start
                processing_times.append(process_time)
                
                if result and isinstance(result, dict):
                    # Проверяем качество результата
                    if 'extracted_data' in result or len(result) > 0:
                        # Создаем обучающий пример
                        training_example = self._create_training_example(image_path, result)
                        training_data.append(training_example)
                        successful_count += 1
                        
                        # Анализируем результат
                        if isinstance(result, dict):
                            fields_count = len(result.get('extracted_data', result))
                            self._log(f"   📋 Извлечено полей: {fields_count}")
                            
                            # Проверяем ключевые поля
                            key_fields = ['invoice_number', 'total_amount', 'invoice_date', 'supplier_name']
                            if 'extracted_data' in result:
                                extracted = result['extracted_data']
                            else:
                                extracted = result
                                
                            found_keys = [k for k in key_fields if k in extracted and extracted[k]]
                            if found_keys:
                                self._log(f"   🔑 Найдены ключевые поля: {', '.join(found_keys)}")
                            else:
                                self._log(f"   ⚠️ Ключевые поля не найдены")
                        
                        self._log(f"   ⏱️ Время обработки: {process_time:.2f}с")
                        self._log(f"   ✅ УСПЕШНО обработано!")
                    else:
                        self._log(f"   ⚠️ Пустой результат от Gemini")
                        failed_count += 1
                        gemini_errors.append((image_path, "Пустой результат"))
                else:
                    self._log(f"   ❌ Неверный формат ответа от Gemini")
                    self._log(f"   📝 Тип результата: {type(result)}")
                    failed_count += 1
                    gemini_errors.append((image_path, f"Неверный тип результата: {type(result)}"))
                
            except Exception as e:
                process_time = time.time() - process_start
                processing_times.append(process_time)
                
                error_msg = str(e)
                self._log(f"   💥 ОШИБКА: {error_msg}")
                self._log(f"   ⏱️ Время до ошибки: {process_time:.2f}с")
                
                failed_count += 1
                gemini_errors.append((image_path, error_msg))
                continue
        
        # Финальная статистика
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        self._log(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        self._log(f"  ✅ Успешно обработано: {successful_count}")
        self._log(f"  ❌ Ошибок: {failed_count}")
        self._log(f"  📊 Всего попыток: {successful_count + failed_count}")
        
        if successful_count + failed_count > 0:
            success_rate = (successful_count / (successful_count + failed_count)) * 100
            self._log(f"  📈 Процент успеха: {success_rate:.1f}%")
            
            if success_rate < 50:
                self._log(f"  ⚠️ НИЗКИЙ процент успеха! Проверьте Gemini API и изображения")
            elif success_rate < 80:
                self._log(f"  🟡 Средний процент успеха")
            else:
                self._log(f"  🟢 Высокий процент успеха")
        
        self._log(f"  ⏱️ Общее время: {total_duration}")
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            self._log(f"  ⚡ Среднее время на изображение: {avg_time:.2f}с")
            self._log(f"  🏃 Минимальное время: {min_time:.2f}с")
            self._log(f"  🐌 Максимальное время: {max_time:.2f}с")
            
            if avg_time > 30:
                self._log(f"  ⚠️ Медленная обработка (>{avg_time:.1f}с на изображение)")
        
        # Сохраняем датасет
        self._log(f"\n💾 СОХРАНЕНИЕ ДАТАСЕТА:")
        dataset_file = os.path.join(output_path, "training_data.json")
        
        try:
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            # Проверяем созданный файл
            file_size = os.path.getsize(dataset_file) / 1024  # KB
            self._log(f"  ✅ Датасет сохранен: {dataset_file}")
            self._log(f"  📏 Размер файла: {file_size:.1f} KB")
            self._log(f"  📋 Количество примеров: {len(training_data)}")
            
            # Анализируем качество данных
            if len(training_data) > 0:
                # Средняя длина входных данных
                avg_input_length = sum(len(ex.get('input', '')) for ex in training_data) / len(training_data)
                avg_output_length = sum(len(ex.get('output', '')) for ex in training_data) / len(training_data)
                
                self._log(f"  📝 Средняя длина input: {avg_input_length:.0f} символов")
                self._log(f"  📝 Средняя длина output: {avg_output_length:.0f} символов")
                
                # Предупреждения о качестве
                if avg_input_length < 50:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень короткие входные данные")
                elif avg_input_length > 5000:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень длинные входные данные")
                else:
                    self._log(f"  ✅ Длина входных данных оптимальна")
                
                if avg_output_length < 20:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень короткие выходные данные")
                elif avg_output_length > 2000:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень длинные выходные данные")
                else:
                    self._log(f"  ✅ Длина выходных данных оптимальна")
            
            # Сохраняем отчет об ошибках если есть
            if gemini_errors:
                error_report_file = os.path.join(output_path, "processing_errors.json")
                error_report = {
                    "timestamp": datetime.now().isoformat(),
                    "total_errors": len(gemini_errors),
                    "errors": [{"image": img, "error": err} for img, err in gemini_errors]
                }
                
                with open(error_report_file, 'w', encoding='utf-8') as f:
                    json.dump(error_report, f, ensure_ascii=False, indent=2)
                    
                self._log(f"  📋 Отчет об ошибках: {error_report_file}")
            
        except Exception as e:
            self._log(f"  💥 ОШИБКА сохранения: {str(e)}")
            raise
        
        self._log(f"\n🎉 ПОДГОТОВКА ДАТАСЕТА ЗАВЕРШЕНА!")
        self._log("=" * 70)
        
        return dataset_file
    
    def _create_training_example(self, image_path: str, gemini_result: Dict) -> Dict:
        """Создает пример для обучения на основе результата Gemini"""
        # Извлекаем текст из изображения
        try:
            image_text = self.plugin.extract_text_from_image(image_path)
        except (AttributeError, IOError, OSError, Exception) as e:
            # Ошибка извлечения текста - используем заглушку
            image_text = "Не удалось извлечь текст"
        
        # Создаем промпт
        prompt = self.plugin.create_invoice_prompt()
        
        # Создаем правильный ответ на основе результата Gemini
        target_response = json.dumps(gemini_result, ensure_ascii=False, indent=2)
        
        # Формируем полный диалог
        full_prompt = f"{prompt}\n\nТекст с изображения:\n{image_text}"
        
        return {
            "input": full_prompt,
            "output": target_response,
            "image_path": image_path,
            "source": "gemini_generated"
        }
    
    def prepare_training_dataset(self, training_data_path: str):
        """
        Подготавливает датасет для обучения из JSON файла
        
        Args:
            training_data_path: Путь к JSON файлу с обучающими данными
            
        Returns:
            DatasetDict: Подготовленный датасет
        """
        from datetime import datetime
        
        start_time = datetime.now()
        self._log("=" * 60)
        self._log("📋 ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
        self._log("=" * 60)
        self._log(f"📅 Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"📁 Путь к датасету: {training_data_path}")
        
        # Проверяем существование файла
        if not os.path.exists(training_data_path):
            self._log(f"❌ ОШИБКА: Файл датасета не найден: {training_data_path}")
            raise FileNotFoundError(f"Файл датасета не найден: {training_data_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(training_data_path) / 1024  # KB
        self._log(f"📏 Размер файла: {file_size:.1f} KB")
        
        if file_size > 100000:  # > 100MB
            self._log(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень большой файл ({file_size:.1f} KB)")
        
        # Загружаем данные
        self._log(f"\n📖 ЗАГРУЗКА ДАННЫХ...")
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self._log(f"❌ ОШИБКА: Неверный формат JSON: {e}")
            raise
        except Exception as e:
            self._log(f"❌ ОШИБКА загрузки файла: {e}")
            raise
        
        self._log(f"✅ Данные загружены успешно")
        self._log(f"📊 Количество примеров: {len(data)}")
        
        if len(data) == 0:
            self._log(f"❌ ОШИБКА: Пустой датасет")
            raise ValueError("Датасет не содержит примеров")
        
        # Анализируем структуру данных
        self._log(f"\n🔍 АНАЛИЗ СТРУКТУРЫ ДАННЫХ:")
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if isinstance(sample, dict):
                keys = list(sample.keys())
                self._log(f"  📋 Ключи в примерах: {keys}")
                
                # Проверяем обязательные поля
                required_fields = ['input', 'output']
                missing_fields = [field for field in required_fields if field not in keys]
                
                if missing_fields:
                    self._log(f"  ❌ Отсутствуют обязательные поля: {missing_fields}")
                    raise ValueError(f"Отсутствуют обязательные поля: {missing_fields}")
                else:
                    self._log(f"  ✅ Все обязательные поля присутствуют")
                
                # Анализируем длины текстов
                input_lengths = [len(ex.get('input', '')) for ex in data if isinstance(ex, dict)]
                output_lengths = [len(ex.get('output', '')) for ex in data if isinstance(ex, dict)]
                
                if input_lengths:
                    avg_input = sum(input_lengths) / len(input_lengths)
                    min_input = min(input_lengths)
                    max_input = max(input_lengths)
                    self._log(f"  📝 Input длины - средн: {avg_input:.0f}, мин: {min_input}, макс: {max_input}")
                
                if output_lengths:
                    avg_output = sum(output_lengths) / len(output_lengths)
                    min_output = min(output_lengths)
                    max_output = max(output_lengths)
                    self._log(f"  📝 Output длины - средн: {avg_output:.0f}, мин: {min_output}, макс: {max_output}")
            else:
                self._log(f"  ❌ Неверная структура: примеры не являются словарями")
                raise ValueError("Примеры должны быть словарями")
        else:
            self._log(f"  ❌ Неверная структура: данные не являются списком")
            raise ValueError("Данные должны быть списком примеров")
        
        # Определяем семейство модели и формат
        model_family = getattr(self.plugin, 'model_family', 'unknown')
        model_name = getattr(self.plugin, 'model_name', 'unknown')
        self._log(f"\n🤖 ИНФОРМАЦИЯ О МОДЕЛИ:")
        self._log(f"  🔧 Семейство модели: {model_family}")
        self._log(f"  📛 Название модели: {model_name}")
        
        # Преобразуем в формат для обучения
        self._log(f"\n🔄 ФОРМАТИРОВАНИЕ ДАННЫХ...")
        formatted_data = []
        skipped_count = 0
        
        for i, example in enumerate(data):
            try:
                if not isinstance(example, dict):
                    self._log(f"  ⚠️ Пропускаем пример {i}: неверный тип данных")
                    skipped_count += 1
                    continue
                
                if 'input' not in example or 'output' not in example:
                    self._log(f"  ⚠️ Пропускаем пример {i}: отсутствуют обязательные поля")
                    skipped_count += 1
                    continue
                
                # Создаем полный текст для обучения
                if model_family == "llama":
                    # Используем chat template для Llama
                    text = self._format_llama_training_text(example["input"], example["output"])
                    self._log(f"  🦙 Использован формат Llama для примера {i+1}")
                else:
                    # Общий формат
                    text = f"{example['input']}\n\n{example['output']}<|endoftext|>"
                    self._log(f"  🔧 Использован общий формат для примера {i+1}")
                
                formatted_data.append({"text": text})
                
                # Логируем каждый 100-й пример
                if (i + 1) % 100 == 0:
                    self._log(f"  📈 Обработано {i+1}/{len(data)} примеров")
                    
            except Exception as e:
                self._log(f"  ❌ Ошибка обработки примера {i}: {e}")
                skipped_count += 1
                continue
        
        self._log(f"✅ Форматирование завершено")
        self._log(f"  📊 Обработано примеров: {len(formatted_data)}")
        self._log(f"  ⚠️ Пропущено примеров: {skipped_count}")
        
        if len(formatted_data) == 0:
            self._log(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Нет валидных примеров после форматирования")
            raise ValueError("Нет валидных примеров после форматирования")
        
        # Разделяем на train/validation (90/10)
        self._log(f"\n📊 РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION:")
        split_ratio = 0.9
        split_idx = int(len(formatted_data) * split_ratio)
        
        # Перемешиваем данные перед разделением
        import random
        random.seed(42)  # Для воспроизводимости
        random.shuffle(formatted_data)
        self._log(f"  🔀 Данные перемешаны (seed=42)")
        
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        self._log(f"  📚 Training set: {len(train_data)} примеров ({len(train_data)/len(formatted_data)*100:.1f}%)")
        self._log(f"  🧪 Validation set: {len(val_data)} примеров ({len(val_data)/len(formatted_data)*100:.1f}%)")
        
        # Проверяем минимальные требования
        if len(train_data) < 10:
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень мало данных для обучения ({len(train_data)} примеров)")
        elif len(train_data) < 100:
            self._log(f"  🟡 Небольшой набор данных для обучения ({len(train_data)} примеров)")
        else:
            self._log(f"  ✅ Достаточно данных для обучения")
        
        if len(val_data) < 2:
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Очень мало данных для валидации ({len(val_data)} примеров)")
        
        # Создаем DatasetDict
        self._log(f"\n🏗️ СОЗДАНИЕ DATASETS...")
        try:
            dataset_dict = DatasetDict({
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data)
            })
            self._log(f"✅ DatasetDict создан успешно")
            
            # Дополнительная информация о датасете
            train_dataset = dataset_dict["train"]
            val_dataset = dataset_dict["validation"]
            
            self._log(f"  📋 Train dataset features: {list(train_dataset.features.keys())}")
            self._log(f"  📋 Validation dataset features: {list(val_dataset.features.keys())}")
            
            # Анализируем длины текстов в финальном датасете
            train_text_lengths = [len(ex['text']) for ex in train_dataset]
            if train_text_lengths:
                avg_length = sum(train_text_lengths) / len(train_text_lengths)
                min_length = min(train_text_lengths)
                max_length = max(train_text_lengths)
                
                self._log(f"  📏 Длины текстов - средн: {avg_length:.0f}, мин: {min_length}, макс: {max_length}")
                
                if avg_length > 4000:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Длинные тексты могут требовать много памяти")
                elif avg_length < 100:
                    self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Короткие тексты могут плохо обучаться")
                else:
                    self._log(f"  ✅ Длины текстов оптимальны")
                    
        except Exception as e:
            self._log(f"❌ ОШИБКА создания DatasetDict: {e}")
            raise
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        self._log(f"\n🎉 ПОДГОТОВКА ДАТАСЕТА ЗАВЕРШЕНА!")
        self._log(f"  ⏱️ Время выполнения: {duration}")
        self._log(f"  📊 Итоговая статистика:")
        self._log(f"    📚 Training: {len(train_data)} примеров")
        self._log(f"    🧪 Validation: {len(val_data)} примеров")
        self._log(f"    📈 Качество: {(len(formatted_data)/(len(data)) * 100):.1f}% валидных примеров")
        self._log("=" * 60)
        
        return dataset_dict
    
    def _format_llama_training_text(self, input_text: str, output_text: str) -> str:
        """Форматирует текст для обучения Llama модели"""
        if "llama-3" in self.plugin.model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        else:
            # Llama 2 format
            return f"<s>[INST] {input_text} [/INST] {output_text} </s>"
    
    def setup_lora_config(self, training_config: Dict[str, Any]):
        """
        Настраивает конфигурацию LoRA
        
        Args:
            training_config: Конфигурация обучения
            
        Returns:
            LoraConfig: Конфигурация LoRA
        """
        self._log("🔧 НАСТРОЙКА LORA КОНФИГУРАЦИИ:")
        
        # Получаем параметры с значениями по умолчанию
        rank = training_config.get("default_lora_rank", 16)
        alpha = training_config.get("default_lora_alpha", 32)
        dropout = training_config.get("lora_dropout", 0.1)
        target_modules = training_config.get("target_modules", ["q_proj", "v_proj"])
        
        self._log(f"  📊 LoRA ранг (r): {rank}")
        self._log(f"  📊 LoRA альфа: {alpha}")
        self._log(f"  📊 Dropout: {dropout}")
        self._log(f"  🎯 Целевые модули: {target_modules}")
        
        # Валидация параметров
        if rank <= 0:
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Некорректный ранг ({rank}), используем 16")
            rank = 16
        elif rank > 128:
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Высокий ранг ({rank}), может потребовать много памяти")
        
        if alpha <= 0:
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Некорректная альфа ({alpha}), используем 32")
            alpha = 32
            
        if not (0 <= dropout <= 1):
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Некорректный dropout ({dropout}), используем 0.1")
            dropout = 0.1
        
        if not target_modules or not isinstance(target_modules, list):
            self._log(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Некорректные целевые модули, используем по умолчанию")
            target_modules = ["q_proj", "v_proj"]
        
        # Создаем конфигурацию
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self._log(f"  ✅ LoRA конфигурация создана успешно")
            
            # Расчет количества обучаемых параметров
            param_efficiency = (rank * 2) / (4096 * 4096)  # Примерная оценка
            self._log(f"  📈 Примерная эффективность параметров: {param_efficiency:.4f}")
            
            if rank < 8:
                self._log(f"  💡 Рекомендация: Низкий ранг может ограничить обучаемость")
            elif rank > 64:
                self._log(f"  💡 Рекомендация: Высокий ранг увеличивает качество, но требует больше ресурсов")
            else:
                self._log(f"  ✅ Оптимальный ранг для большинства задач")
                
        except Exception as e:
            self._log(f"  ❌ ОШИБКА создания LoRA конфигурации: {e}")
            raise
        
        return lora_config
    
    def train_model(self, 
                   dataset_path: str, 
                   output_dir: str,
                   training_config: Dict[str, Any]) -> bool:
        """
        Основной метод обучения модели
        
        Args:
            dataset_path: Путь к датасету
            output_dir: Директория для сохранения модели
            training_config: Параметры обучения
            
        Returns:
            bool: True если обучение успешно
        """
        try:
            import traceback
            import sys
            from datetime import datetime
            
            self.training_active = True
            self.should_stop = False
            start_time = datetime.now()
            
            self._log("=" * 80)
            self._log("🚀 НАЧАЛО ОБУЧЕНИЯ LLM МОДЕЛИ")
            self._log("=" * 80)
            self._log(f"📅 Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"📁 Датасет: {dataset_path}")
            self._log(f"💾 Выходная директория: {output_dir}")
            self._log(f"🤖 Плагин: {self.plugin.__class__.__name__}")
            
            # 🖥️ СИСТЕМНАЯ ДИАГНОСТИКА
            self._log("\n🖥️ СИСТЕМНАЯ ДИАГНОСТИКА:")
            self._log(f"  🐍 Python версия: {sys.version.split()[0]}")
            if hasattr(torch, '__version__'):
                self._log(f"  🔥 PyTorch версия: {torch.__version__}")
            self._log(f"  🔥 CUDA доступна: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                self._log(f"  🎮 CUDA устройств: {torch.cuda.device_count()}")
                self._log(f"  📍 Текущее CUDA устройство: {torch.cuda.current_device()}")
                try:
                    device_props = torch.cuda.get_device_properties(0)
                    self._log(f"  🎮 GPU: {device_props.name}")
                    self._log(f"  💾 Память GPU: {device_props.total_memory / 1024**3:.1f} GB")
                    
                    # Проверяем текущее использование памяти
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    memory_free = (device_props.total_memory / 1024**3) - memory_cached
                    
                    self._log(f"  💾 Память выделена: {memory_allocated:.2f} GB")
                    self._log(f"  💾 Память зарезервирована: {memory_cached:.2f} GB")
                    self._log(f"  💾 Память свободна: {memory_free:.2f} GB")
                    
                    # Предупреждение о недостатке памяти
                    if memory_free < 2.0:
                        self._log("  ⚠️ ПРЕДУПРЕЖДЕНИЕ: Низкое количество свободной GPU памяти!")
                        
                except Exception as gpu_error:
                    self._log(f"  ⚠️ Ошибка получения информации о GPU: {gpu_error}")
            else:
                self._log("  ⚠️ ВНИМАНИЕ: CUDA недоступна, будет использован CPU (медленно)")
            
            # 📋 ДЕТАЛЬНЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ
            self._log("\n📋 ДЕТАЛЬНЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ:")
            for section, params in training_config.items():
                self._log(f"  📂 {section}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        self._log(f"    {key}: {value}")
                else:
                    self._log(f"    {params}")
            
            # Проверяем важные зависимости
            self._log("\n🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ:")
            try:
                import transformers
                self._log(f"  ✅ transformers: {transformers.__version__}")
            except ImportError:
                self._log("  ❌ transformers: НЕ УСТАНОВЛЕНО")
                
            try:
                import datasets
                self._log(f"  ✅ datasets: {datasets.__version__}")
            except ImportError:
                self._log("  ❌ datasets: НЕ УСТАНОВЛЕНО")
                
            try:
                import peft
                self._log(f"  ✅ peft: {peft.__version__}")
            except ImportError:
                self._log("  ❌ peft: НЕ УСТАНОВЛЕНО")
                
            try:
                import accelerate
                self._log(f"  ✅ accelerate: {accelerate.__version__}")
            except ImportError:
                self._log("  ❌ accelerate: НЕ УСТАНОВЛЕНО")
            
            # 1. Загружаем базовую модель если не загружена
            self._log("\n🤖 ===== ЭТАП 1: ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ =====")
            if not self.plugin.is_loaded:
                self._log("📥 Загрузка базовой модели...")
                model_name = getattr(self.plugin, 'model_name', 'неизвестная модель')
                self._log(f"   Модель: {model_name}")
                
                if not self.plugin.load_model():
                    raise ValueError("Не удалось загрузить базовую модель")
                self._log("✅ Базовая модель загружена успешно")
            else:
                self._log("✅ Базовая модель уже загружена")
            
            self.model = self.plugin.model
            self.tokenizer = self.plugin.tokenizer
            
            # Информация о модели
            if hasattr(self.model, 'num_parameters'):
                total_params = self.model.num_parameters()
                self._log(f"📊 Параметры модели: {total_params:,}")
            
            # 2. Подготавливаем датасет
            self._log("\n📊 ===== ЭТАП 2: ПОДГОТОВКА ДАТАСЕТА =====")
            self._log(f"📂 Загрузка датасета из: {dataset_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
                
            dataset = self.prepare_training_dataset(dataset_path)
            
            # Подробная статистика датасета
            self._log("📈 Статистика датасета:")
            for split_name, split_data in dataset.items():
                self._log(f"   {split_name}: {len(split_data)} примеров")
                if len(split_data) > 0:
                    # Показываем пример данных
                    example = split_data[0]
                    if 'text' in example:
                        text_len = len(example['text'])
                        text_preview = example['text'][:100] + "..." if text_len > 100 else example['text']
                        self._log(f"     Пример длина: {text_len} символов")
                        self._log(f"     Пример текст: {text_preview}")
            
            # 3. Токенизируем датасет
            self._log("\n🔤 ===== ЭТАП 3: ТОКЕНИЗАЦИЯ ДАТАСЕТА =====")
            max_length = training_config.get("max_sequence_length", 2048)
            self._log(f"📏 Максимальная длина последовательности: {max_length}")
            self._log(f"🔤 Размер словаря токенизатора: {len(self.tokenizer)}")
            
            tokenized_dataset = self._tokenize_dataset(dataset, training_config)
            
            # Статистика токенизации
            self._log("📊 Статистика токенизации:")
            for split_name, split_data in tokenized_dataset.items():
                self._log(f"   {split_name}: {len(split_data)} примеров")
                if len(split_data) > 0:
                    example_tokens = split_data[0]['input_ids']
                    self._log(f"     Пример длина токенов: {len(example_tokens)}")
            
            # 4. Настраиваем LoRA
            self._log("\n🔧 ===== ЭТАП 4: НАСТРОЙКА LORA =====")
            lora_config = self.setup_lora_config(training_config)
            self._log(f"🎯 Создание PEFT модели...")
            
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # Информация о LoRA параметрах
            trainable_params = 0
            all_params = 0
            for _, param in self.peft_model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
            self._log(f"📊 Параметры LoRA:")
            self._log(f"   Всего параметров: {all_params:,}")
            self._log(f"   Обучаемых параметров: {trainable_params:,}")
            self._log(f"   Процент обучаемых: {100 * trainable_params / all_params:.2f}%")
            
            # 5. Настраиваем параметры обучения
            self._log("\n📋 ===== ЭТАП 5: НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ =====")
            self._log(f"📁 Создание выходной директории: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            training_args = self._setup_training_args(output_dir, training_config)
            
            # Логируем ключевые параметры
            self._log("📊 Ключевые параметры обучения:")
            self._log(f"   Эпох: {training_args.num_train_epochs}")
            self._log(f"   Batch size (train): {training_args.per_device_train_batch_size}")
            self._log(f"   Batch size (eval): {training_args.per_device_eval_batch_size}")
            self._log(f"   Learning rate: {training_args.learning_rate}")
            self._log(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
            self._log(f"   FP16: {getattr(training_args, 'fp16', False)}")
            self._log(f"   Gradient checkpointing: {getattr(training_args, 'gradient_checkpointing', False)}")
            
            # Расчет эффективного размера батча и времени
            train_dataset_size = len(tokenized_dataset["train"])
            effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            steps_per_epoch = train_dataset_size // effective_batch_size
            total_steps = steps_per_epoch * training_args.num_train_epochs
            
            self._log(f"📊 Детальная информация об обучении:")
            self._log(f"   📄 Размер тренировочного датасета: {train_dataset_size}")
            self._log(f"   📈 Эффективный размер батча: {effective_batch_size}")
            self._log(f"   🔢 Шагов на эпоху: {steps_per_epoch}")
            self._log(f"   📊 Всего шагов обучения: {total_steps}")
            
            # 6. Создаем тренера
            self._log("\n🏃 ===== ЭТАП 6: СОЗДАНИЕ TRAINER =====")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            self._log("✅ Data collator создан")
            
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                data_collator=data_collator,
                callbacks=[TrainingProgressCallback(self)]
            )
            self._log("✅ Trainer создан успешно")
            
            # 7. Запускаем обучение
            self._log("\n🚀 ===== ЭТАП 7: ЗАПУСК ОБУЧЕНИЯ =====")
            self._log("🎯 Начинаем тренировку модели...")
            self._log(f"⏰ Время начала обучения: {datetime.now().strftime('%H:%M:%S')}")
            
            train_result = self.trainer.train()
            
            if self.should_stop:
                self._log("⏹️ Обучение остановлено пользователем")
                return False
            
            # Логируем результаты обучения
            end_time = datetime.now()
            training_duration = end_time - start_time
            self._log(f"⏰ Время окончания обучения: {end_time.strftime('%H:%M:%S')}")
            self._log(f"⏱️ Общее время обучения: {training_duration}")
            
            # Детальная статистика обучения
            if hasattr(train_result, 'training_loss'):
                final_loss = train_result.training_loss
                self._log(f"📉 Финальный training loss: {final_loss:.4f}")
                
                # Оценка качества loss
                if final_loss < 0.5:
                    loss_quality = "🟢 Отличный"
                elif final_loss < 1.0:
                    loss_quality = "🟡 Хороший"
                elif final_loss < 2.0:
                    loss_quality = "🟠 Удовлетворительный"
                else:
                    loss_quality = "🔴 Требует улучшения"
                    
                self._log(f"📊 Качество обучения: {loss_quality}")
            
            # Статистика производительности
            if hasattr(train_result, 'global_step') and train_result.global_step > 0:
                total_seconds = training_duration.total_seconds()
                steps_per_second = train_result.global_step / total_seconds
                seconds_per_step = total_seconds / train_result.global_step
                
                self._log(f"⚡ Производительность:")
                self._log(f"   🏃 Шагов в секунду: {steps_per_second:.2f}")
                self._log(f"   ⏱️ Секунд на шаг: {seconds_per_step:.2f}")
                
                if seconds_per_step < 2:
                    perf_rating = "🚀 Очень быстро"
                elif seconds_per_step < 10:
                    perf_rating = "⚡ Быстро"
                elif seconds_per_step < 30:
                    perf_rating = "🐎 Нормально"
                else:
                    perf_rating = "🐌 Медленно"
                    
                self._log(f"   📈 Оценка: {perf_rating}")
            
            # 8. Сохраняем модель
            self._log("\n💾 ===== ЭТАП 8: СОХРАНЕНИЕ МОДЕЛИ =====")
            self._log(f"📁 Сохранение LoRA адаптера в: {output_dir}")
            
            self.trainer.save_model()
            self._log("✅ LoRA адаптер сохранен")
            
            self.tokenizer.save_pretrained(output_dir)
            self._log("✅ Токенизатор сохранен")
            
            # 9. Сохраняем метрики и метаданные
            self._log("\n📊 ===== ЭТАП 9: СОХРАНЕНИЕ МЕТРИК =====")
            
            # Метрики обучения
            metrics_file = os.path.join(output_dir, "training_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            self._log(f"✅ Метрики сохранены: {metrics_file}")
            
            # Метаданные обучения
            metadata = {
                'base_model': getattr(self.plugin, 'model_name', 'unknown'),
                'plugin_class': self.plugin.__class__.__name__,
                'training_config': training_config,
                'created_at': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'training_duration': str(training_duration),
                'final_loss': getattr(train_result, 'training_loss', None),
                'total_steps': getattr(train_result, 'global_step', None),
                'dataset_stats': {
                    'train_size': len(tokenized_dataset["train"]),
                    'validation_size': len(tokenized_dataset["validation"]) if "validation" in tokenized_dataset else 0
                },
                'model_stats': {
                    'total_params': all_params,
                    'trainable_params': trainable_params,
                    'trainable_percent': 100 * trainable_params / all_params
                }
            }
            
            metadata_path = os.path.join(output_dir, 'training_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self._log(f"✅ Метаданные сохранены: {metadata_path}")
            
            self._log(f"\n🎉 ========== ОБУЧЕНИЕ LLM МОДЕЛИ ЗАВЕРШЕНО ==========")
            self._log(f"📁 LoRA адаптер сохранен в: {output_dir}")
            self._log(f"⏱️ Время обучения: {training_duration}")
            self._log(f"📊 Всего шагов: {getattr(train_result, 'global_step', 'неизвестно')}")
            
            return True
            
        except Exception as e:
            self._log(f"\n💥 ========== ОШИБКА ОБУЧЕНИЯ LLM МОДЕЛИ ==========")
            error_msg = f"❌ Критическая ошибка: {str(e)}"
            self._log(error_msg)
            
            # Подробная диагностика ошибки
            self._log("🔍 Диагностическая информация:")
            self._log(f"   Python версия: {sys.version}")
            if hasattr(torch, '__version__'):
                self._log(f"   PyTorch версия: {torch.__version__}")
            self._log(f"   CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self._log(f"   CUDA устройств: {torch.cuda.device_count()}")
                try:
                    self._log(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                except (RuntimeError, AttributeError, Exception) as e:
                    # Ошибка получения информации о GPU - не критично
                    pass
            
            self._log(f"   Рабочая директория: {os.getcwd()}")
            self._log(f"   Датасет существует: {os.path.exists(dataset_path) if 'dataset_path' in locals() else 'неизвестно'}")
            self._log(f"   Выходная директория: {output_dir}")
            
            # Полная трассировка ошибки
            self._log("\n🔍 Полная трассировка ошибки:")
            full_traceback = traceback.format_exc()
            for line in full_traceback.split('\n'):
                if line.strip():
                    self._log(f"   {line}")
            
            return False
        finally:
            self.training_active = False
    
    def _tokenize_dataset(self, dataset, training_config: Dict):
        """Токенизирует датасет"""
        def tokenize_function(examples):
            # Токенизируем с максимальной длиной
            max_length = training_config.get("max_sequence_length", 2048)
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # Устанавливаем labels равными input_ids для causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Применяем токенизацию
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Токенизация"
        )
        
        return tokenized_dataset
    
    def _setup_training_args(self, output_dir: str, training_config: Dict):
        """Настраивает аргументы обучения"""
        args_dict = training_config.get("training_args", {})
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args_dict.get("num_epochs", 3),
            per_device_train_batch_size=args_dict.get("batch_size", 4),
            per_device_eval_batch_size=args_dict.get("batch_size", 4),
            gradient_accumulation_steps=args_dict.get("gradient_accumulation_steps", 4),
            learning_rate=args_dict.get("learning_rate", 2e-4),
            warmup_steps=args_dict.get("warmup_steps", 100),
            logging_steps=args_dict.get("logging_steps", 10),
            eval_steps=args_dict.get("eval_steps", 500),
            save_steps=args_dict.get("save_steps", 500),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Отключаем wandb/tensorboard
            remove_unused_columns=False,
            dataloader_drop_last=args_dict.get("dataloader_drop_last", True),
            fp16=args_dict.get("fp16", True) and torch.cuda.is_available(),
            gradient_checkpointing=args_dict.get("gradient_checkpointing", True),
            logging_dir=os.path.join(output_dir, "logs"),
        )
    
    def stop_training(self):
        """Останавливает обучение"""
        self.should_stop = True
        if self.trainer:
            self.trainer.control.should_training_stop = True
        self._log("🛑 Запрос на остановку обучения...")
    
    def get_training_metrics(self) -> TrainingMetrics:
        """Возвращает текущие метрики обучения"""
        return self.current_metrics
    
    def get_training_history(self) -> List[Dict]:
        """Возвращает историю обучения"""
        return self.training_history
    
    def estimate_training_time(self, dataset_size: int, training_config: Dict) -> Dict[str, Any]:
        """Оценивает время обучения"""
        epochs = training_config.get("training_args", {}).get("num_epochs", 3)
        batch_size = training_config.get("training_args", {}).get("batch_size", 4)
        
        # Примерная оценка на основе размера модели
        if "70b" in self.plugin.model_name.lower():
            time_per_sample = 2.0  # секунды
        elif "13b" in self.plugin.model_name.lower():
            time_per_sample = 0.5
        else:
            time_per_sample = 0.2
        
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * epochs
        estimated_time = total_steps * time_per_sample
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_time_hours": estimated_time / 3600,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch
        }
    
    def _update_progress(self, current: int, total: int, message: str = ""):
        """Обновляет прогресс"""
        if self.progress_callback:
            progress = int((current / total) * 100) if total > 0 else 0
            self.progress_callback(progress, message)
    
    def _log(self, message: str):
        """Логирование"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Отправляем в callback если есть
        if hasattr(self.progress_callback, '__call__'):
            try:
                self.progress_callback(-1, log_message)  # -1 означает лог сообщение
            except (TypeError, AttributeError, Exception) as e:
                # Ошибка вызова callback - продолжаем без логирования
                pass


class TrainingProgressCallback:
    """Callback для отслеживания прогресса обучения"""
    
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        self.start_time = time.time()
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Начало обучения"""
        self.start_time = time.time()
        self.trainer_instance._log("🚀 Обучение начато")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Начало эпохи"""
        epoch = state.epoch
        self.trainer_instance.current_metrics.epoch = int(epoch)
        self.trainer_instance._log(f"📚 Эпоха {int(epoch) + 1}/{args.num_train_epochs}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Конец шага"""
        self.trainer_instance.current_metrics.step = state.global_step
        
        # Обновляем прогресс
        progress = (state.global_step / state.max_steps) * 100
        message = f"Шаг {state.global_step}/{state.max_steps} (Эпоха {int(state.epoch) + 1})"
        
        if self.trainer_instance.progress_callback:
            self.trainer_instance.progress_callback(int(progress), message)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Логирование метрик"""
        if logs:
            metrics = TrainingMetrics(
                epoch=int(state.epoch),
                step=state.global_step,
                loss=logs.get("train_loss", 0.0),
                eval_loss=logs.get("eval_loss", 0.0),
                learning_rate=logs.get("learning_rate", 0.0),
                samples_per_second=logs.get("train_samples_per_second", 0.0)
            )
            
            # Обновляем текущие метрики
            self.trainer_instance.current_metrics = metrics
            
            # Добавляем в историю
            metrics_dict = {
                "epoch": metrics.epoch,
                "step": metrics.step,
                "loss": metrics.loss,
                "eval_loss": metrics.eval_loss,
                "learning_rate": metrics.learning_rate,
                "timestamp": datetime.now().isoformat()
            }
            self.trainer_instance.training_history.append(metrics_dict)
            
            # Логируем
            if "train_loss" in logs:
                self.trainer_instance._log(f"📊 Шаг {metrics.step}: loss={metrics.loss:.4f}, lr={metrics.learning_rate:.2e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Конец обучения"""
        total_time = time.time() - self.start_time
        self.trainer_instance._log(f"[OK] Обучение завершено за {total_time/3600:.1f} часов") 
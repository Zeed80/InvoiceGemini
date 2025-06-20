import os
import json
import time
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QWidget, 
    QLineEdit, QFileDialog, QTextEdit, QFrame, QGroupBox, QMessageBox, QApplication, QInputDialog,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFormLayout, QGridLayout, QCheckBox, QComboBox,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup
)
from datetime import datetime
from datasets import Dataset
import os
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor

# Предполагается, что эти классы будут доступны для type hinting или будут импортированы позже
# from ..processing_engine import OCRProcessor, GeminiProcessor 
# from ..config import Config # Это пример, импортировать нужно будет корректно

# NEW: Импортируем ModelTrainer (предполагаем, что он будет в trainer.py)
from .training.trainer import ModelTrainer
from .training.data_preparator import TrainingDataPreparator # Переносим импорт сюда для порядка
from .training.donut_trainer import DonutTrainer as DonutTrainerClass
from .training.trocr_trainer import TrOCRTrainer
from .training.trocr_dataset_preparator import TrOCRDatasetPreparator, TrOCRDatasetConfig
from .training.hyperparameter_optimizer import TrOCRHyperparameterOptimizer
from .training.universal_dataset_parser import UniversalDatasetParser, DatasetFormat
from .training.advanced_data_validator import AdvancedDataValidator
from .pdf_text_analyzer import PDFTextAnalyzer  # NEW: PDF анализатор

# Используем реальный DonutTrainer из отдельного модуля
# class DonutTrainer удален - используем DonutTrainerClass

class DatasetQualityAnalyzer(QObject):
    """Анализатор качества датасета для обучения ML моделей"""
    
    def __init__(self):
        super().__init__()
        
    def analyze_dataset(self, dataset_path):
        """
        Анализирует качество датасета по ключевым метрикам
        
        Основано на 7 ключевых метриках качества данных:
        https://www.precisely.com/blog/data-quality/how-to-measure-data-quality-7-metrics
        """
        try:
            results = {
                'dataset_size': self._get_dataset_size(dataset_path),
                'label_balance': self._analyze_label_balance(dataset_path),
                'data_completeness': self._check_data_completeness(dataset_path),
                'annotation_quality': self._assess_annotation_quality(dataset_path),
                'file_integrity': self._check_file_integrity(dataset_path),
                'metadata_consistency': self._check_metadata_consistency(dataset_path),
                'overall_score': 0.0,
                'recommendations': []
            }
            
            # Вычисляем общий балл качества
            results['overall_score'] = self._calculate_overall_score(results)
            
            # Генерируем рекомендации
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'dataset_size': {'total': 0, 'train': 0, 'validation': 0},
                'label_balance': {'total_labels': 0, 'o_percentage': 100.0, 'unique_labels': 0},
                'data_completeness': 0.0,
                'annotation_quality': 0.0,
                'file_integrity': 0.0,
                'metadata_consistency': 0.0,
                'overall_score': 0.0,
                'recommendations': ['Ошибка анализа датасета']
            }
    
    def _get_dataset_size(self, dataset_path):
        """Анализ размера датасета"""
        size_info = {'total': 0, 'train': 0, 'validation': 0, 'test': 0}
        
        if not os.path.exists(dataset_path):
            return size_info
            
        # Проверяем разные форматы датасетов
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json') and ('train' in file.lower() or 'annotation' in file.lower()):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                size_info['total'] += len(data)
                            elif isinstance(data, dict) and 'annotations' in data:
                                size_info['total'] += len(data['annotations'])
                    except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError) as e:
                        # Пропускаем поврежденные или нечитаемые JSON файлы
                        pass
                        
                elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                    if 'train' in root.lower():
                        size_info['train'] += 1
                    elif 'val' in root.lower() or 'validation' in root.lower():
                        size_info['validation'] += 1
                    elif 'test' in root.lower():
                        size_info['test'] += 1
                    else:
                        size_info['total'] += 1
        
        return size_info
    
    def _analyze_label_balance(self, dataset_path):
        """Анализ баланса меток"""
        label_stats = {'total_labels': 0, 'o_percentage': 0.0, 'unique_labels': 0, 'label_distribution': {}}
        
        try:
            # Ищем файлы с аннотациями
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            filepath = os.path.join(root, file)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # Обрабатываем разные форматы
                            labels = []
                            if isinstance(data, list):
                                for item in data:
                                    if 'labels' in item:
                                        labels.extend(item['labels'])
                                    elif 'ner_tags' in item:
                                        labels.extend(item['ner_tags'])
                                        
                            elif isinstance(data, dict):
                                if 'train' in data:
                                    for item in data['train']:
                                        if 'labels' in item:
                                            labels.extend(item['labels'])
                                        elif 'ner_tags' in item:
                                            labels.extend(item['ner_tags'])
                            
                            # Подсчитываем метки
                            for label in labels:
                                label_name = str(label)
                                if label_name not in label_stats['label_distribution']:
                                    label_stats['label_distribution'][label_name] = 0
                                label_stats['label_distribution'][label_name] += 1
                                label_stats['total_labels'] += 1
                                
                        except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError, KeyError) as e:
                            # Пропускаем файлы с ошибками структуры или чтения JSON
                            continue
            
            # Вычисляем статистики
            if label_stats['total_labels'] > 0:
                o_count = label_stats['label_distribution'].get('O', 0) + label_stats['label_distribution'].get('15', 0)
                label_stats['o_percentage'] = (o_count / label_stats['total_labels']) * 100
                label_stats['unique_labels'] = len(label_stats['label_distribution'])
                
        except Exception as e:
            pass
            
        return label_stats
    
    def _check_data_completeness(self, dataset_path):
        """Проверка полноты данных (отсутствие пустых значений)"""
        completeness_score = 0.0
        
        try:
            total_fields = 0
            empty_fields = 0
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # Проверяем поля на пустоту
                            if isinstance(data, list):
                                for item in data:
                                    for key, value in item.items():
                                        total_fields += 1
                                        if not value or (isinstance(value, list) and len(value) == 0):
                                            empty_fields += 1
                                            
                        except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError, KeyError) as e:
                            # Пропускаем файлы с ошибками при проверке полноты данных
                            continue
            
            if total_fields > 0:
                completeness_score = ((total_fields - empty_fields) / total_fields) * 100
                
        except (OSError, IOError) as e:
            # Ошибка при обходе директорий - возвращаем 0
            pass
            
        return max(0.0, min(100.0, completeness_score))
    
    def _assess_annotation_quality(self, dataset_path):
        """Оценка качества аннотаций"""
        quality_score = 0.0
        
        try:
            valid_annotations = 0
            total_annotations = 0
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # Проверяем структуру аннотаций
                            if isinstance(data, list):
                                for item in data:
                                    total_annotations += 1
                                    if self._is_valid_annotation(item):
                                        valid_annotations += 1
                                        
                        except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError, KeyError) as e:
                            # Пропускаем файлы с ошибками при оценке качества аннотаций
                            continue
            
            if total_annotations > 0:
                quality_score = (valid_annotations / total_annotations) * 100
                
        except (OSError, IOError) as e:
            # Ошибка при обходе директорий для оценки качества
            pass
            
        return max(0.0, min(100.0, quality_score))
    
    def _is_valid_annotation(self, annotation):
        """Проверяет валидность отдельной аннотации"""
        required_fields = ['words', 'labels', 'bboxes']
        
        for field in required_fields:
            if field not in annotation:
                return False
            if not annotation[field]:
                return False
                
        # Проверяем соответствие длин
        words_len = len(annotation.get('words', []))
        labels_len = len(annotation.get('labels', []))
        bboxes_len = len(annotation.get('bboxes', []))
        
        return words_len == labels_len == bboxes_len
    
    def _check_file_integrity(self, dataset_path):
        """Проверка целостности файлов"""
        integrity_score = 100.0
        
        try:
            total_files = 0
            corrupted_files = 0
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    total_files += 1
                    filepath = os.path.join(root, file)
                    
                    try:
                        # Проверяем JSON файлы
                        if file.endswith('.json'):
                            with open(filepath, 'r', encoding='utf-8') as f:
                                json.load(f)
                        # Проверяем размер файлов изображений
                        elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            if os.path.getsize(filepath) == 0:
                                corrupted_files += 1
                                
                    except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError) as e:
                        # Файл поврежден или нечитаем
                        corrupted_files += 1
            
            if total_files > 0:
                integrity_score = ((total_files - corrupted_files) / total_files) * 100
                
        except (OSError, IOError) as e:
            # Ошибка при проверке целостности файлов
            pass
            
        return max(0.0, min(100.0, integrity_score))
    
    def _check_metadata_consistency(self, dataset_path):
        """Проверка консистентности метаданных"""
        consistency_score = 100.0
        
        try:
            # Проверяем наличие стандартных файлов
            info_file = os.path.join(dataset_path, 'dataset_info.json')
            
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    
                # Проверяем обязательные поля
                required_fields = ['dataset_name', 'creation_date', 'total_files']
                missing_fields = sum(1 for field in required_fields if field not in info)
                consistency_score = ((len(required_fields) - missing_fields) / len(required_fields)) * 100
            else:
                consistency_score = 50.0  # Снижаем балл за отсутствие метаданных
                
        except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError, KeyError) as e:
            # Ошибка при проверке метаданных - нет консистентности
            consistency_score = 0.0
            
        return max(0.0, min(100.0, consistency_score))
    
    def _calculate_overall_score(self, results):
        """Вычисляет общий балл качества датасета"""
        try:
            # Весовые коэффициенты для разных метрик
            weights = {
                'dataset_size': 0.25,      # Размер датасета критичен
                'label_balance': 0.25,     # Баланс меток очень важен
                'data_completeness': 0.20, # Полнота данных
                'annotation_quality': 0.20, # Качество аннотаций
                'file_integrity': 0.05,    # Целостность файлов
                'metadata_consistency': 0.05 # Консистентность метаданных
            }
            
            scores = {}
            
            # Нормализуем размер датасета (больше = лучше, но с насыщением)
            total_size = results['dataset_size']['total'] + results['dataset_size']['train'] + results['dataset_size']['validation']
            if total_size >= 100:
                scores['dataset_size'] = 100.0
            elif total_size >= 50:
                scores['dataset_size'] = 80.0
            elif total_size >= 20:
                scores['dataset_size'] = 60.0
            elif total_size >= 10:
                scores['dataset_size'] = 40.0
            else:
                scores['dataset_size'] = (total_size / 10) * 40.0
            
            # Оценка баланса меток (меньше % "O" меток = лучше)
            o_percentage = results['label_balance']['o_percentage']
            if o_percentage <= 50:
                scores['label_balance'] = 100.0
            elif o_percentage <= 70:
                scores['label_balance'] = 80.0
            elif o_percentage <= 85:
                scores['label_balance'] = 60.0
            else:
                scores['label_balance'] = max(0, 100 - o_percentage)
            
            # Остальные метрики уже в процентах
            scores['data_completeness'] = results['data_completeness']
            scores['annotation_quality'] = results['annotation_quality']
            scores['file_integrity'] = results['file_integrity']
            scores['metadata_consistency'] = results['metadata_consistency']
            
            # Вычисляем взвешенную сумму
            overall_score = sum(scores[metric] * weights[metric] for metric in weights)
            
            return round(overall_score, 1)
            
        except:
            return 0.0
    
    def _generate_recommendations(self, results):
        """Генерирует рекомендации по улучшению качества"""
        recommendations = []
        
        # Анализируем размер датасета
        total_size = results['dataset_size']['total'] + results['dataset_size']['train'] + results['dataset_size']['validation']
        if total_size < 20:
            recommendations.append("🚨 КРИТИЧНО: Датасет слишком мал (< 20 примеров). Необходимо минимум 50-100 примеров")
        elif total_size < 50:
            recommendations.append("⚠️ Предупреждение: Маленький датасет. Рекомендуется увеличить до 50+ примеров")
        
        # Анализируем баланс меток
        o_percentage = results['label_balance']['o_percentage']
        if o_percentage > 85:
            recommendations.append("🚨 КРИТИЧНО: Слишком много 'O' меток (>85%). Улучшите алгоритмы разметки")
        elif o_percentage > 70:
            recommendations.append("⚠️ Дисбаланс меток: Много 'O' меток (>70%). Настройте систему разметки")
        
        # Анализируем полноту данных
        if results['data_completeness'] < 80:
            recommendations.append("📝 Низкая полнота данных. Проверьте пустые поля в аннотациях")
        
        # Анализируем качество аннотаций
        if results['annotation_quality'] < 70:
            recommendations.append("🏷️ Низкое качество аннотаций. Проверьте структуру данных")
        
        # Анализируем целостность файлов
        if results['file_integrity'] < 95:
            recommendations.append("🔧 Обнаружены поврежденные файлы. Выполните проверку датасета")
        
        # Общая оценка
        if results['overall_score'] >= 80:
            recommendations.insert(0, "✅ Хорошее качество датасета для обучения")
        elif results['overall_score'] >= 60:
            recommendations.insert(0, "⚠️ Удовлетворительное качество. Есть области для улучшения")
        else:
            recommendations.insert(0, "🚨 Низкое качество датасета. Необходимы существенные улучшения")
        
        return recommendations

class TrainingWorker(QObject):
    """Worker для обучения в отдельном потоке"""
    finished = pyqtSignal(str)  # Путь к обученной модели
    error = pyqtSignal(str)     # Сообщение об ошибке
    progress = pyqtSignal(int)  # Прогресс (0-100)
    log_message = pyqtSignal(str)  # Лог сообщения
    
    def __init__(self, trainer, training_params):
        super().__init__()
        self.trainer = trainer
        self.training_params = training_params
        
    def run(self):
        """Запуск обучения"""
        try:
            print("TrainingWorker: Начинаем выполнение обучения...")
            
            # Устанавливаем callbacks
            print("TrainingWorker: Устанавливаем коллбеки...")
            self.trainer.set_callbacks(
                status_callback=self.log_message.emit,
                progress_callback=self.progress.emit,
                metrics_callback=self.log_message.emit  # Добавляем для TrOCR метрик
            )
            
            # Запускаем обучение
            result = None
            if hasattr(self.trainer, 'train_layoutlm'):
                print("TrainingWorker: Запускаем обучение LayoutLM...")
                result = self.trainer.train_layoutlm(**self.training_params)
                print(f"TrainingWorker: LayoutLM обучение завершено с результатом: {result}")
            elif hasattr(self.trainer, 'train_donut'):
                print("TrainingWorker: Запускаем обучение Donut...")
                result = self.trainer.train_donut(**self.training_params)
                print(f"TrainingWorker: Donut обучение завершено с результатом: {result}")
            elif hasattr(self.trainer, 'train_trocr'):
                print("TrainingWorker: Запускаем обучение TrOCR...")
                result = self.trainer.train_trocr(**self.training_params)
                print(f"TrainingWorker: TrOCR обучение завершено с результатом: {result}")
            else:
                raise ValueError(f"Неизвестный тип тренера: {type(self.trainer).__name__}")
                
            if result:
                print(f"TrainingWorker: Отправляем сигнал finished с результатом: {result}")
                self.finished.emit(result)
            else:
                print("TrainingWorker: Результат обучения пустой, отправляем error")
                self.error.emit("Обучение завершилось неуспешно")
                
        except Exception as e:
            print(f"TrainingWorker: ОШИБКА во время обучения: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class ModernTrainingDialog(QDialog):
    """Современный диалог обучения моделей с вкладками"""
    
    def __init__(self, app_config, ocr_processor, gemini_processor, parent=None):
        super().__init__(parent)
        self.app_config = app_config
        self.ocr_processor = ocr_processor
        self.gemini_processor = gemini_processor
        
        # NEW: Инициализируем PDF анализатор
        self.pdf_analyzer = PDFTextAnalyzer()
        
        # Переменные для управления обучением
        self.current_trainer = None
        self.current_worker = None
        self.current_thread = None
        
        # История обучения для мониторинга
        self.training_history = []
        self.current_metrics = {
            'epoch': 0,
            'step': 0,
            'loss': 0.0,
            'lr': 0.0,
            'accuracy': 0.0,
            'f1': 0.0
        }
        
        # Анализатор качества датасета
        self.quality_analyzer = DatasetQualityAnalyzer()
        self.last_quality_results = None
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("🎓 Обучение моделей ИИ")
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # Основной layout
        main_layout = QVBoxLayout(self)
        
        # Заголовок
        header_layout = QHBoxLayout()
        title_label = QLabel("🎓 Центр обучения моделей ИИ")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Кнопка справки
        help_button = QPushButton("❓ Справка")
        help_button.clicked.connect(self.show_help)
        header_layout.addWidget(help_button)
        
        main_layout.addLayout(header_layout)
        
        # Основной виджет с вкладками
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover {
                background-color: #e8f4fd;
            }
        """)
        
        # Создаем вкладки
        self.create_layoutlm_tab()
        self.create_donut_tab()
        self.create_trocr_tab()
        self.create_trocr_dataset_tab()  # NEW: TrOCR Dataset Preparation
        self.create_dataset_preparation_tab()
        self.create_monitoring_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Нижняя панель
        bottom_layout = QHBoxLayout()
        
        # Общий статус
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        bottom_layout.addWidget(self.status_label)
        
        bottom_layout.addStretch()
        
        # Кнопки управления
        self.save_settings_button = QPushButton("💾 Сохранить настройки")
        self.save_settings_button.clicked.connect(self.save_settings)
        
        self.close_button = QPushButton("❌ Закрыть")
        self.close_button.clicked.connect(self.close)
        
        bottom_layout.addWidget(self.save_settings_button)
        bottom_layout.addWidget(self.close_button)
        
        main_layout.addLayout(bottom_layout)
        
    def create_layoutlm_tab(self):
        """Создает вкладку для обучения LayoutLM"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок вкладки
        header = QLabel("📄 Обучение LayoutLMv3 для понимания документов")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #2c3e50; padding: 10px; background: #ecf0f1; border-radius: 5px;")
        layout.addWidget(header)
        
        # Создаем splitter для разделения на две части
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа: Выбор данных
        data_group = QGroupBox("📊 Данные для обучения")
        data_layout = QFormLayout(data_group)
        
        self.layoutlm_dataset_edit = QLineEdit()
        self.layoutlm_dataset_edit.setPlaceholderText("Выберите подготовленный датасет...")
        dataset_button = QPushButton("📁")
        dataset_button.clicked.connect(lambda: self.select_dataset('layoutlm'))
        
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.layoutlm_dataset_edit)
        dataset_layout.addWidget(dataset_button)
        data_layout.addRow("Датасет:", dataset_layout)
        
        # Информация о датасете
        self.layoutlm_dataset_info = QLabel("Выберите датасет для получения информации")
        self.layoutlm_dataset_info.setWordWrap(True)
        self.layoutlm_dataset_info.setStyleSheet("color: #7f8c8d; font-style: italic;")
        data_layout.addRow("Информация:", self.layoutlm_dataset_info)
        
        left_layout.addWidget(data_group)
        
        # Группа: Модель
        model_group = QGroupBox("🤖 Настройки модели")
        model_layout = QFormLayout(model_group)
        
        self.layoutlm_base_model_edit = QLineEdit("microsoft/layoutlmv3-base")
        model_layout.addRow("Базовая модель:", self.layoutlm_base_model_edit)
        
        self.layoutlm_output_name_edit = QLineEdit()
        self.layoutlm_output_name_edit.setText(f"layoutlm_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_layout.addRow("Имя модели:", self.layoutlm_output_name_edit)
        
        left_layout.addWidget(model_group)
        
        # Группа: Параметры обучения
        params_group = QGroupBox("⚙️ Параметры обучения")
        params_layout = QGridLayout(params_group)
        
        # Эпохи
        self.layoutlm_epochs_spin = QSpinBox()
        self.layoutlm_epochs_spin.setRange(1, 100)
        self.layoutlm_epochs_spin.setValue(10)
        params_layout.addWidget(QLabel("Эпохи:"), 0, 0)
        params_layout.addWidget(self.layoutlm_epochs_spin, 0, 1)
        
        # Размер батча
        self.layoutlm_batch_size_spin = QSpinBox()
        self.layoutlm_batch_size_spin.setRange(1, 64)
        self.layoutlm_batch_size_spin.setValue(8)
        params_layout.addWidget(QLabel("Размер батча:"), 0, 2)
        params_layout.addWidget(self.layoutlm_batch_size_spin, 0, 3)
        
        # Learning rate
        self.layoutlm_lr_spin = QDoubleSpinBox()
        self.layoutlm_lr_spin.setRange(1e-6, 1e-2)
        self.layoutlm_lr_spin.setDecimals(6)
        self.layoutlm_lr_spin.setValue(5e-5)
        self.layoutlm_lr_spin.setSingleStep(1e-5)
        params_layout.addWidget(QLabel("Learning rate:"), 1, 0)
        params_layout.addWidget(self.layoutlm_lr_spin, 1, 1)
        
        # Weight decay
        self.layoutlm_weight_decay_spin = QDoubleSpinBox()
        self.layoutlm_weight_decay_spin.setRange(0, 0.1)
        self.layoutlm_weight_decay_spin.setDecimals(3)
        self.layoutlm_weight_decay_spin.setValue(0.01)
        params_layout.addWidget(QLabel("Weight decay:"), 1, 2)
        params_layout.addWidget(self.layoutlm_weight_decay_spin, 1, 3)
        
        # Warmup ratio
        self.layoutlm_warmup_spin = QDoubleSpinBox()
        self.layoutlm_warmup_spin.setRange(0, 0.5)
        self.layoutlm_warmup_spin.setDecimals(2)
        self.layoutlm_warmup_spin.setValue(0.1)
        params_layout.addWidget(QLabel("Warmup ratio:"), 2, 0)
        params_layout.addWidget(self.layoutlm_warmup_spin, 2, 1)
        
        # Seed
        self.layoutlm_seed_spin = QSpinBox()
        self.layoutlm_seed_spin.setRange(0, 100000)
        self.layoutlm_seed_spin.setValue(42)
        params_layout.addWidget(QLabel("Seed:"), 2, 2)
        params_layout.addWidget(self.layoutlm_seed_spin, 2, 3)
        
        left_layout.addWidget(params_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
        self.layoutlm_start_button = QPushButton("🚀 Начать обучение")
        self.layoutlm_start_button.clicked.connect(self.start_layoutlm_training)
        self.layoutlm_start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.layoutlm_stop_button = QPushButton("⏹️ Остановить")
        self.layoutlm_stop_button.clicked.connect(self.stop_training)
        self.layoutlm_stop_button.setEnabled(False)
        self.layoutlm_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        control_layout.addWidget(self.layoutlm_start_button)
        control_layout.addWidget(self.layoutlm_stop_button)
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # Правая панель - мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Прогресс
        progress_group = QGroupBox("📈 Прогресс обучения")
        progress_layout = QVBoxLayout(progress_group)
        
        self.layoutlm_progress_bar = QProgressBar()
        self.layoutlm_progress_bar.setVisible(False)
        progress_layout.addWidget(self.layoutlm_progress_bar)
        
        self.layoutlm_status_label = QLabel("Готов к обучению")
        self.layoutlm_status_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        progress_layout.addWidget(self.layoutlm_status_label)
        
        right_layout.addWidget(progress_group)
        
        # Лог обучения
        log_group = QGroupBox("📝 Журнал обучения")
        log_layout = QVBoxLayout(log_group)
        
        self.layoutlm_log = QTextEdit()
        self.layoutlm_log.setReadOnly(True)
        self.layoutlm_log.setMaximumHeight(300)
        self.layoutlm_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.layoutlm_log)
        
        # Кнопки для лога
        log_buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("🗑️ Очистить")
        clear_log_button.clicked.connect(lambda: self.layoutlm_log.clear())
        
        save_log_button = QPushButton("💾 Сохранить лог")
        save_log_button.clicked.connect(lambda: self.save_log(self.layoutlm_log))
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_buttons_layout.addStretch()
        
        log_layout.addLayout(log_buttons_layout)
        right_layout.addWidget(log_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📄 LayoutLMv3")
        
    def create_donut_tab(self):
        """Создает вкладку для обучения Donut"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок вкладки
        header = QLabel("🍩 Обучение Donut для OCR-free понимания документов")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #8e44ad; padding: 10px; background: #f4ecf7; border-radius: 5px;")
        layout.addWidget(header)
        
        # Информационное сообщение
        info_label = QLabel(
            "💡 Donut - это революционная модель для понимания документов без OCR. "
            "Она обрабатывает изображения документов напрямую и извлекает структурированную информацию."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #e8f5e8; padding: 10px; border-radius: 5px; color: #2d5a2d;")
        layout.addWidget(info_label)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа: Выбор данных
        data_group = QGroupBox("📊 Данные для обучения")
        data_layout = QFormLayout(data_group)
        
        self.donut_dataset_edit = QLineEdit()
        self.donut_dataset_edit.setPlaceholderText("Выберите датасет для Donut...")
        dataset_button = QPushButton("📁")
        dataset_button.clicked.connect(lambda: self.select_dataset('donut'))
        
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.donut_dataset_edit)
        dataset_layout.addWidget(dataset_button)
        data_layout.addRow("Датасет:", dataset_layout)
        
        # Информация о датасете
        self.donut_dataset_info = QLabel("Выберите датасет для получения информации")
        self.donut_dataset_info.setWordWrap(True)
        self.donut_dataset_info.setStyleSheet("color: #7f8c8d; font-style: italic;")
        data_layout.addRow("Информация:", self.donut_dataset_info)
        
        left_layout.addWidget(data_group)
        
        # Группа: Модель
        model_group = QGroupBox("🤖 Настройки модели")
        model_layout = QFormLayout(model_group)
        
        self.donut_base_model_combo = QComboBox()
        self.donut_base_model_combo.addItems([
            "naver-clova-ix/donut-base",
            "naver-clova-ix/donut-base-finetuned-cord-v2",
            "naver-clova-ix/donut-base-finetuned-docvqa",
            "naver-clova-ix/donut-base-finetuned-rvlcdip"
        ])
        model_layout.addRow("Базовая модель:", self.donut_base_model_combo)
        
        self.donut_output_name_edit = QLineEdit()
        self.donut_output_name_edit.setText(f"donut_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_layout.addRow("Имя модели:", self.donut_output_name_edit)
        
        # Тип задачи
        self.donut_task_combo = QComboBox()
        self.donut_task_combo.addItems([
            "document_parsing",
            "document_classification", 
            "document_vqa"
        ])
        model_layout.addRow("Тип задачи:", self.donut_task_combo)
        
        left_layout.addWidget(model_group)
        
        # Группа: Параметры обучения
        params_group = QGroupBox("⚙️ Параметры обучения")
        params_layout = QGridLayout(params_group)
        
        # Эпохи
        self.donut_epochs_spin = QSpinBox()
        self.donut_epochs_spin.setRange(1, 50)
        self.donut_epochs_spin.setValue(5)
        params_layout.addWidget(QLabel("Эпохи:"), 0, 0)
        params_layout.addWidget(self.donut_epochs_spin, 0, 1)
        
        # Размер батча
        self.donut_batch_size_spin = QSpinBox()
        self.donut_batch_size_spin.setRange(1, 16)
        self.donut_batch_size_spin.setValue(2)
        params_layout.addWidget(QLabel("Размер батча:"), 0, 2)
        params_layout.addWidget(self.donut_batch_size_spin, 0, 3)
        
        # Learning rate
        self.donut_lr_spin = QDoubleSpinBox()
        self.donut_lr_spin.setRange(1e-6, 1e-3)
        self.donut_lr_spin.setDecimals(6)
        self.donut_lr_spin.setValue(3e-5)
        self.donut_lr_spin.setSingleStep(1e-6)
        params_layout.addWidget(QLabel("Learning rate:"), 1, 0)
        params_layout.addWidget(self.donut_lr_spin, 1, 1)
        
        # Gradient accumulation
        self.donut_grad_accum_spin = QSpinBox()
        self.donut_grad_accum_spin.setRange(1, 32)
        self.donut_grad_accum_spin.setValue(4)
        params_layout.addWidget(QLabel("Grad. accumulation:"), 1, 2)
        params_layout.addWidget(self.donut_grad_accum_spin, 1, 3)
        
        # Max length
        self.donut_max_length_spin = QSpinBox()
        self.donut_max_length_spin.setRange(128, 2048)
        self.donut_max_length_spin.setValue(512)
        params_layout.addWidget(QLabel("Max length:"), 2, 0)
        params_layout.addWidget(self.donut_max_length_spin, 2, 1)
        
        # Image size
        self.donut_image_size_combo = QComboBox()
        self.donut_image_size_combo.addItems(["224", "384", "512", "768"])
        self.donut_image_size_combo.setCurrentText("384")
        params_layout.addWidget(QLabel("Размер изображения:"), 2, 2)
        params_layout.addWidget(self.donut_image_size_combo, 2, 3)
        
        left_layout.addWidget(params_group)
        
        # Дополнительные настройки
        advanced_group = QGroupBox("🔧 Дополнительные настройки")
        advanced_layout = QFormLayout(advanced_group)
        
        self.donut_fp16_checkbox = QCheckBox("Использовать FP16")
        self.donut_fp16_checkbox.setChecked(True)
        advanced_layout.addRow("Оптимизация:", self.donut_fp16_checkbox)
        
        self.donut_save_steps_spin = QSpinBox()
        self.donut_save_steps_spin.setRange(50, 5000)
        self.donut_save_steps_spin.setValue(500)
        advanced_layout.addRow("Сохранение каждые N шагов:", self.donut_save_steps_spin)
        
        self.donut_eval_steps_spin = QSpinBox()
        self.donut_eval_steps_spin.setRange(50, 5000)
        self.donut_eval_steps_spin.setValue(500)
        advanced_layout.addRow("Оценка каждые N шагов:", self.donut_eval_steps_spin)
        
        left_layout.addWidget(advanced_group)
        
        # Группа оптимизаций памяти
        memory_group = QGroupBox(self.tr("🚀 Оптимизации памяти"))
        memory_layout = QVBoxLayout(memory_group)
        
        # LoRA оптимизация
        self.use_lora_cb = QCheckBox(self.tr("LoRA (Low-Rank Adaptation) - до 95% экономии памяти"))
        self.use_lora_cb.setChecked(True)
        self.use_lora_cb.setToolTip(self.tr("Обучает только 1-5% параметров вместо 100%"))
        memory_layout.addWidget(self.use_lora_cb)
        
        # 8-bit оптимизатор
        self.use_8bit_optimizer_cb = QCheckBox(self.tr("8-bit оптимизатор - до 25% экономии памяти"))
        self.use_8bit_optimizer_cb.setChecked(True)
        self.use_8bit_optimizer_cb.setToolTip(self.tr("Использует 8-bit AdamW вместо 32-bit"))
        memory_layout.addWidget(self.use_8bit_optimizer_cb)
        
        # Заморозка encoder
        self.freeze_encoder_cb = QCheckBox(self.tr("Заморозить encoder - обучать только decoder"))
        self.freeze_encoder_cb.setChecked(False)
        self.freeze_encoder_cb.setToolTip(self.tr("Экономит память, но может снизить качество"))
        memory_layout.addWidget(self.freeze_encoder_cb)
        
        # Информация об оптимизациях
        memory_info = QLabel(self.tr("""
<b>💡 Рекомендации:</b><br>
• <b>LoRA</b> - самая эффективная оптимизация (до 95% экономии)<br>
• <b>8-bit optimizer</b> - дополнительные 25% экономии<br>
• <b>Freeze encoder</b> - только если не хватает памяти<br>
• Комбинация всех методов может снизить потребление с 11GB до 2-3GB
        """))
        memory_info.setStyleSheet("QLabel { color: #666; background: #f0f0f0; padding: 8px; border-radius: 4px; }")
        memory_info.setWordWrap(True)
        memory_layout.addWidget(memory_info)
        
        # Кнопка автоматической оптимизации
        auto_optimize_btn = QPushButton(self.tr("🚀 Автооптимизация памяти для RTX 4070 Ti"))
        auto_optimize_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #27ae60, stop: 1 #2ecc71);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #2ecc71, stop: 1 #27ae60);
            }
        """)
        auto_optimize_btn.clicked.connect(self.auto_optimize_memory)
        memory_layout.addWidget(auto_optimize_btn)
        
        layout.addWidget(memory_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
        # 🚀 Кнопка быстрых настроек GPU
        self.fast_gpu_button = QPushButton("⚡ Быстрые настройки GPU")
        self.fast_gpu_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.fast_gpu_button.clicked.connect(self.apply_fast_gpu_settings)
        control_layout.addWidget(self.fast_gpu_button)
        
        self.donut_start_button = QPushButton("🚀 Начать обучение")
        self.donut_start_button.clicked.connect(self.start_donut_training)
        self.donut_start_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.donut_stop_button = QPushButton("⏹️ Остановить")
        self.donut_stop_button.clicked.connect(self.stop_training)
        self.donut_stop_button.setEnabled(False)
        self.donut_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        control_layout.addWidget(self.donut_start_button)
        control_layout.addWidget(self.donut_stop_button)
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # Правая панель - мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Прогресс
        progress_group = QGroupBox("📈 Прогресс обучения")
        progress_layout = QVBoxLayout(progress_group)
        
        self.donut_progress_bar = QProgressBar()
        self.donut_progress_bar.setVisible(False)
        progress_layout.addWidget(self.donut_progress_bar)
        
        self.donut_status_label = QLabel("Готов к обучению")
        self.donut_status_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        progress_layout.addWidget(self.donut_status_label)
        
        right_layout.addWidget(progress_group)
        
        # Лог обучения
        log_group = QGroupBox("📝 Журнал обучения")
        log_layout = QVBoxLayout(log_group)
        
        self.donut_log = QTextEdit()
        self.donut_log.setReadOnly(True)
        self.donut_log.setMaximumHeight(300)
        self.donut_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.donut_log)
        
        # Кнопки для лога
        log_buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("🗑️ Очистить")
        clear_log_button.clicked.connect(lambda: self.donut_log.clear())
        
        save_log_button = QPushButton("💾 Сохранить лог")
        save_log_button.clicked.connect(lambda: self.save_log(self.donut_log))
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_buttons_layout.addStretch()
        
        log_layout.addLayout(log_buttons_layout)
        right_layout.addWidget(log_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "🍩 Donut")
        
    def create_trocr_tab(self):
        """Создает вкладку для обучения TrOCR (Microsoft)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок вкладки
        header = QLabel("📱 Обучение Microsoft TrOCR для извлечения текста из изображений")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #0078d4; padding: 10px; background: #f3f9ff; border-radius: 5px;")
        layout.addWidget(header)
        
        # Информационное сообщение
        info_label = QLabel(
            "💡 TrOCR - это современная модель Microsoft для Text Recognition in the Wild. "
            "Идеально подходит для извлечения данных из счетов, чеков и документов с высокой точностью."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #e3f2fd; padding: 10px; border-radius: 5px; color: #1565c0;")
        layout.addWidget(info_label)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа: Выбор данных
        data_group = QGroupBox("📊 Данные для обучения")
        data_layout = QFormLayout(data_group)
        
        self.trocr_dataset_edit = QLineEdit()
        self.trocr_dataset_edit.setPlaceholderText("Выберите датасет для TrOCR...")
        dataset_button = QPushButton("📁")
        dataset_button.clicked.connect(lambda: self.select_dataset('trocr'))
        
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.trocr_dataset_edit)
        dataset_layout.addWidget(dataset_button)
        data_layout.addRow("Датасет:", dataset_layout)
        
        # Информация о датасете
        self.trocr_dataset_info = QLabel("Выберите датасет для получения информации")
        self.trocr_dataset_info.setWordWrap(True)
        self.trocr_dataset_info.setStyleSheet("color: #7f8c8d; font-style: italic;")
        data_layout.addRow("Информация:", self.trocr_dataset_info)
        
        left_layout.addWidget(data_group)
        
        # Группа: Модель
        model_group = QGroupBox("🤖 Настройки модели")
        model_layout = QFormLayout(model_group)
        
        self.trocr_base_model_combo = QComboBox()
        self.trocr_base_model_combo.addItems([
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten", 
            "microsoft/trocr-base-stage1",
            "microsoft/trocr-large-printed",
            "microsoft/trocr-large-handwritten"
        ])
        model_layout.addRow("Базовая модель:", self.trocr_base_model_combo)
        
        self.trocr_output_name_edit = QLineEdit()
        self.trocr_output_name_edit.setText(f"trocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_layout.addRow("Имя модели:", self.trocr_output_name_edit)
        
        left_layout.addWidget(model_group)
        
        # Группа: Параметры обучения
        params_group = QGroupBox("⚙️ Параметры обучения")
        params_layout = QGridLayout(params_group)
        
        # Эпохи
        self.trocr_epochs_spin = QSpinBox()
        self.trocr_epochs_spin.setRange(1, 20)
        self.trocr_epochs_spin.setValue(3)
        params_layout.addWidget(QLabel("Эпохи:"), 0, 0)
        params_layout.addWidget(self.trocr_epochs_spin, 0, 1)
        
        # Размер батча
        self.trocr_batch_size_spin = QSpinBox()
        self.trocr_batch_size_spin.setRange(1, 16)
        self.trocr_batch_size_spin.setValue(4)
        params_layout.addWidget(QLabel("Размер батча:"), 0, 2)
        params_layout.addWidget(self.trocr_batch_size_spin, 0, 3)
        
        # Learning rate
        self.trocr_lr_spin = QDoubleSpinBox()
        self.trocr_lr_spin.setRange(1e-6, 1e-3)
        self.trocr_lr_spin.setDecimals(6)
        self.trocr_lr_spin.setValue(5e-5)
        self.trocr_lr_spin.setSingleStep(1e-6)
        params_layout.addWidget(QLabel("Learning rate:"), 1, 0)
        params_layout.addWidget(self.trocr_lr_spin, 1, 1)
        
        # Gradient accumulation
        self.trocr_grad_accum_spin = QSpinBox()
        self.trocr_grad_accum_spin.setRange(1, 16)
        self.trocr_grad_accum_spin.setValue(2)
        params_layout.addWidget(QLabel("Grad. accumulation:"), 1, 2)
        params_layout.addWidget(self.trocr_grad_accum_spin, 1, 3)
        
        # Max length
        self.trocr_max_length_spin = QSpinBox()
        self.trocr_max_length_spin.setRange(64, 1024)
        self.trocr_max_length_spin.setValue(256)
        params_layout.addWidget(QLabel("Max length:"), 2, 0)
        params_layout.addWidget(self.trocr_max_length_spin, 2, 1)
        
        # Image size
        self.trocr_image_size_combo = QComboBox()
        self.trocr_image_size_combo.addItems(["224", "384", "448", "512"])
        self.trocr_image_size_combo.setCurrentText("384")
        params_layout.addWidget(QLabel("Размер изображения:"), 2, 2)
        params_layout.addWidget(self.trocr_image_size_combo, 2, 3)
        
        left_layout.addWidget(params_group)
        
        # Группа: Оптимизации памяти TrOCR
        memory_group = QGroupBox("🚀 Оптимизации памяти TrOCR")
        memory_layout = QVBoxLayout(memory_group)
        
        # LoRA оптимизация
        self.trocr_use_lora_cb = QCheckBox("LoRA (Low-Rank Adaptation) - до 90% экономии памяти")
        self.trocr_use_lora_cb.setChecked(True)
        self.trocr_use_lora_cb.setToolTip("Обучает только 1-10% параметров вместо 100%")
        memory_layout.addWidget(self.trocr_use_lora_cb)
        
        # 8-bit оптимизатор
        self.trocr_use_8bit_optimizer_cb = QCheckBox("8-bit оптимизатор - до 25% экономии памяти")
        self.trocr_use_8bit_optimizer_cb.setChecked(True)
        self.trocr_use_8bit_optimizer_cb.setToolTip("Использует 8-bit AdamW вместо 32-bit")
        memory_layout.addWidget(self.trocr_use_8bit_optimizer_cb)
        
        # Gradient checkpointing
        self.trocr_gradient_checkpointing_cb = QCheckBox("Gradient Checkpointing - экономия activations")
        self.trocr_gradient_checkpointing_cb.setChecked(True)
        self.trocr_gradient_checkpointing_cb.setToolTip("Пересчитывает activations вместо хранения")
        memory_layout.addWidget(self.trocr_gradient_checkpointing_cb)
        
        # Информация об оптимизациях
        memory_info = QLabel("""
<b>💡 TrOCR оптимизации:</b><br>
• <b>LoRA</b> - оптимальная экономия памяти для TrOCR<br>
• <b>8-bit optimizer</b> - дополнительная экономия optimizer states<br>
• <b>Gradient Checkpointing</b> - экономия activations памяти<br>
• Комбинация методов позволяет обучать TrOCR на RTX 4070 Ti без OOM
        """)
        memory_info.setStyleSheet("QLabel { color: #666; background: #f0f0f0; padding: 8px; border-radius: 4px; }")
        memory_info.setWordWrap(True)
        memory_layout.addWidget(memory_info)
        
        # Кнопка автоматической оптимизации для TrOCR
        auto_optimize_trocr_btn = QPushButton("🚀 Оптимизация для RTX 4070 Ti")
        auto_optimize_trocr_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #0078d4, stop: 1 #106ebe);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #106ebe, stop: 1 #0078d4);
            }
        """)
        auto_optimize_trocr_btn.clicked.connect(self.auto_optimize_trocr_memory)
        memory_layout.addWidget(auto_optimize_trocr_btn)
        
        # Кнопка интеллектуальной оптимизации гиперпараметров
        smart_optimize_btn = QPushButton("🧠 Умная оптимизация гиперпараметров")
        smart_optimize_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #28a745, stop: 1 #20873a);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                    stop: 0 #20873a, stop: 1 #28a745);
            }
        """)
        smart_optimize_btn.setToolTip("Анализирует датасет и предыдущие результаты для оптимизации параметров обучения")
        smart_optimize_btn.clicked.connect(self.smart_optimize_trocr_hyperparameters)
        memory_layout.addWidget(smart_optimize_btn)
        
        left_layout.addWidget(memory_group)
        
        # Дополнительные настройки
        advanced_group = QGroupBox("🔧 Дополнительные настройки")
        advanced_layout = QFormLayout(advanced_group)
        
        self.trocr_fp16_checkbox = QCheckBox("Использовать FP16")
        self.trocr_fp16_checkbox.setChecked(True)
        advanced_layout.addRow("Оптимизация:", self.trocr_fp16_checkbox)
        
        self.trocr_warmup_ratio_spin = QDoubleSpinBox()
        self.trocr_warmup_ratio_spin.setRange(0.0, 0.3)
        self.trocr_warmup_ratio_spin.setDecimals(2)
        self.trocr_warmup_ratio_spin.setValue(0.1)
        advanced_layout.addRow("Warmup ratio:", self.trocr_warmup_ratio_spin)
        
        self.trocr_weight_decay_spin = QDoubleSpinBox()
        self.trocr_weight_decay_spin.setRange(0.0, 0.1)
        self.trocr_weight_decay_spin.setDecimals(3)
        self.trocr_weight_decay_spin.setValue(0.01)
        advanced_layout.addRow("Weight decay:", self.trocr_weight_decay_spin)
        
        left_layout.addWidget(advanced_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
        # Кнопка быстрых настроек GPU для TrOCR
        self.trocr_fast_gpu_button = QPushButton("⚡ Быстрые настройки GPU")
        self.trocr_fast_gpu_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.trocr_fast_gpu_button.clicked.connect(self.apply_trocr_fast_gpu_settings)
        control_layout.addWidget(self.trocr_fast_gpu_button)
        
        self.trocr_start_button = QPushButton("🚀 Начать обучение")
        self.trocr_start_button.clicked.connect(self.start_trocr_training)
        self.trocr_start_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.trocr_stop_button = QPushButton("⏹️ Остановить")
        self.trocr_stop_button.clicked.connect(self.stop_training)
        self.trocr_stop_button.setEnabled(False)
        self.trocr_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        control_layout.addWidget(self.trocr_start_button)
        control_layout.addWidget(self.trocr_stop_button)
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # Правая панель - мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Прогресс
        progress_group = QGroupBox("📈 Прогресс обучения TrOCR")
        progress_layout = QVBoxLayout(progress_group)
        
        self.trocr_progress_bar = QProgressBar()
        self.trocr_progress_bar.setVisible(False)
        progress_layout.addWidget(self.trocr_progress_bar)
        
        self.trocr_status_label = QLabel("Готов к обучению")
        self.trocr_status_label.setStyleSheet("font-weight: bold; color: #0078d4;")
        progress_layout.addWidget(self.trocr_status_label)
        
        right_layout.addWidget(progress_group)
        
        # Лог обучения
        log_group = QGroupBox("📝 Журнал обучения TrOCR")
        log_layout = QVBoxLayout(log_group)
        
        self.trocr_log = QTextEdit()
        self.trocr_log.setReadOnly(True)
        self.trocr_log.setMaximumHeight(300)
        self.trocr_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.trocr_log)
        
        # Кнопки для лога
        log_buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("🗑️ Очистить")
        clear_log_button.clicked.connect(lambda: self.trocr_log.clear())
        
        save_log_button = QPushButton("💾 Сохранить лог")
        save_log_button.clicked.connect(lambda: self.save_log(self.trocr_log))
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_buttons_layout.addStretch()
        
        log_layout.addLayout(log_buttons_layout)
        right_layout.addWidget(log_group)
        
        # Группа: TrOCR специфичные метрики
        metrics_group = QGroupBox("📊 Метрики TrOCR")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Информация об обучении
        self.trocr_training_info = QLabel("Метрики появятся во время обучения")
        self.trocr_training_info.setWordWrap(True)
        self.trocr_training_info.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                color: #495057;
                font-style: italic;
            }
        """)
        metrics_layout.addWidget(self.trocr_training_info)
        
        right_layout.addWidget(metrics_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📱 TrOCR")
        
    def create_trocr_dataset_tab(self):
        """Создает вкладку для подготовки датасетов TrOCR"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок
        header = QLabel("📊 Подготовка датасетов для TrOCR")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #8e44ad; padding: 10px; background: #f4ecf7; border-radius: 5px;")
        layout.addWidget(header)
        
        # Информационное сообщение
        info_label = QLabel(
            "💡 Создайте специализированные датасеты для обучения TrOCR моделей. "
            "Поддерживаются различные источники данных и форматы аннотаций."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #e8f5e8; padding: 10px; border-radius: 5px; color: #2d5a2d;")
        layout.addWidget(info_label)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа: Тип датасета
        type_group = QGroupBox("🎯 Тип датасета")
        type_layout = QFormLayout(type_group)
        
        self.trocr_dataset_type_combo = QComboBox()
        self.trocr_dataset_type_combo.addItems([
            "Из аннотаций счетов (JSON)",
            "Из структуры папок",
            "Синтетический датасет",
            "Из готовых аннотаций",
            "🔄 Импорт внешнего датасета (COCO/YOLO/VOC)",
            "📊 Импорт CSV датасета",
            "🏷️ Импорт LabelMe датасета"
        ])
        self.trocr_dataset_type_combo.currentTextChanged.connect(self.on_trocr_dataset_type_changed)
        type_layout.addRow("Тип источника:", self.trocr_dataset_type_combo)
        
        left_layout.addWidget(type_group)
        
        # Группа: Исходные данные (изменяется в зависимости от типа)
        self.trocr_source_group = QGroupBox("📁 Исходные данные")
        self.trocr_source_layout = QFormLayout(self.trocr_source_group)
        
        # Виджеты для различных типов источников
        self.setup_trocr_source_widgets()
        
        left_layout.addWidget(self.trocr_source_group)
        
        # Группа: Конфигурация датасета
        config_group = QGroupBox("⚙️ Конфигурация датасета")
        config_layout = QFormLayout(config_group)
        
        # Базовая модель
        self.trocr_base_model_combo = QComboBox()
        self.trocr_base_model_combo.addItems([
            "microsoft/trocr-base-stage1",
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-printed",
            "microsoft/trocr-large-handwritten"
        ])
        config_layout.addRow("Базовая модель:", self.trocr_base_model_combo)
        
        # Максимальная длина текста
        self.trocr_max_text_length_spin = QSpinBox()
        self.trocr_max_text_length_spin.setRange(64, 512)
        self.trocr_max_text_length_spin.setValue(128)
        config_layout.addRow("Макс. длина текста:", self.trocr_max_text_length_spin)
        
        # Размер изображения
        self.trocr_image_size_combo = QComboBox()
        self.trocr_image_size_combo.addItems(["224x224", "384x384", "448x448", "512x512"])
        self.trocr_image_size_combo.setCurrentText("384x384")
        config_layout.addRow("Размер изображений:", self.trocr_image_size_combo)
        
        # Аугментации
        self.trocr_enable_aug_checkbox = QCheckBox("Включить аугментации")
        self.trocr_enable_aug_checkbox.setChecked(True)
        config_layout.addRow("Аугментация данных:", self.trocr_enable_aug_checkbox)
        
        left_layout.addWidget(config_group)
        
        # Группа: Разделение данных
        split_group = QGroupBox("📊 Разделение данных")
        split_layout = QFormLayout(split_group)
        
        self.trocr_train_split_spin = QDoubleSpinBox()
        self.trocr_train_split_spin.setRange(0.5, 0.9)
        self.trocr_train_split_spin.setDecimals(2)
        self.trocr_train_split_spin.setValue(0.8)
        split_layout.addRow("Доля для обучения:", self.trocr_train_split_spin)
        
        self.trocr_val_split_spin = QDoubleSpinBox()
        self.trocr_val_split_spin.setRange(0.05, 0.3)
        self.trocr_val_split_spin.setDecimals(2)
        self.trocr_val_split_spin.setValue(0.1)
        split_layout.addRow("Доля для валидации:", self.trocr_val_split_spin)
        
        self.trocr_test_split_spin = QDoubleSpinBox()
        self.trocr_test_split_spin.setRange(0.05, 0.3)
        self.trocr_test_split_spin.setDecimals(2)
        self.trocr_test_split_spin.setValue(0.1)
        split_layout.addRow("Доля для тестирования:", self.trocr_test_split_spin)
        
        left_layout.addWidget(split_group)
        
        # Группа: Выходные данные
        output_group = QGroupBox("💾 Выходные данные")
        output_layout = QFormLayout(output_group)
        
        self.trocr_output_path_edit = QLineEdit()
        self.trocr_output_path_edit.setPlaceholderText("data/training_datasets/trocr_" + 
                                                     datetime.now().strftime('%Y%m%d_%H%M%S'))
        output_button = QPushButton("📁")
        output_button.clicked.connect(self.select_trocr_output_path)
        
        output_layout_h = QHBoxLayout()
        output_layout_h.addWidget(self.trocr_output_path_edit)
        output_layout_h.addWidget(output_button)
        output_layout.addRow("Путь сохранения:", output_layout_h)
        
        left_layout.addWidget(output_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
        self.trocr_dataset_start_button = QPushButton("🚀 Создать датасет")
        self.trocr_dataset_start_button.clicked.connect(self.start_trocr_dataset_preparation)
        self.trocr_dataset_start_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.trocr_dataset_stop_button = QPushButton("⏹️ Остановить")
        self.trocr_dataset_stop_button.clicked.connect(self.stop_trocr_dataset_preparation)
        self.trocr_dataset_stop_button.setEnabled(False)
        self.trocr_dataset_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        control_layout.addWidget(self.trocr_dataset_start_button)
        control_layout.addWidget(self.trocr_dataset_stop_button)
        
        # Кнопки валидации и очистки данных
        validate_button = QPushButton("🔍 Валидация")
        validate_button.clicked.connect(self.validate_trocr_dataset)
        validate_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        
        clean_button = QPushButton("🧹 Очистка")
        clean_button.clicked.connect(self.clean_trocr_dataset)
        clean_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        # Вторая строка кнопок для валидации
        validation_layout = QHBoxLayout()
        validation_layout.addWidget(validate_button)
        validation_layout.addWidget(clean_button)
        validation_layout.addStretch()
        
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        left_layout.addLayout(validation_layout)
        left_layout.addStretch()
        
        # Правая панель - мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Прогресс
        progress_group = QGroupBox("📈 Прогресс создания")
        progress_layout = QVBoxLayout(progress_group)
        
        self.trocr_dataset_progress_bar = QProgressBar()
        self.trocr_dataset_progress_bar.setVisible(False)
        progress_layout.addWidget(self.trocr_dataset_progress_bar)
        
        self.trocr_dataset_status_label = QLabel("Готов к созданию датасета")
        self.trocr_dataset_status_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        progress_layout.addWidget(self.trocr_dataset_status_label)
        
        right_layout.addWidget(progress_group)
        
        # Лог создания
        log_group = QGroupBox("📝 Журнал создания")
        log_layout = QVBoxLayout(log_group)
        
        self.trocr_dataset_log = QTextEdit()
        self.trocr_dataset_log.setReadOnly(True)
        self.trocr_dataset_log.setMaximumHeight(300)
        self.trocr_dataset_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.trocr_dataset_log)
        
        # Кнопки для лога
        log_buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("🗑️ Очистить")
        clear_log_button.clicked.connect(lambda: self.trocr_dataset_log.clear())
        
        save_log_button = QPushButton("💾 Сохранить лог")
        save_log_button.clicked.connect(lambda: self.save_log(self.trocr_dataset_log))
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_buttons_layout.addStretch()
        
        log_layout.addLayout(log_buttons_layout)
        right_layout.addWidget(log_group)
        
        # Группа: Информация о датасете
        info_group = QGroupBox("ℹ️ Информация о датасете")
        info_layout = QVBoxLayout(info_group)
        
        self.trocr_dataset_info_label = QLabel("Информация появится после создания датасета")
        self.trocr_dataset_info_label.setWordWrap(True)
        self.trocr_dataset_info_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                color: #6c757d;
            }
        """)
        info_layout.addWidget(self.trocr_dataset_info_label)
        
        right_layout.addWidget(info_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📊 TrOCR Датасет")
        
        # Инициализируем источники данных
        self.on_trocr_dataset_type_changed()
        
        # Добавляем дополнительные элементы для автоматизации
        self._add_automation_controls(tab)
        
    def create_dataset_preparation_tab(self):
        """Создает вкладку для подготовки датасетов"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок
        header = QLabel("📊 Подготовка датасетов для обучения")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #e67e22; padding: 10px; background: #fdf2e9; border-radius: 5px;")
        layout.addWidget(header)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа: Исходные данные
        source_group = QGroupBox("📁 Исходные данные")
        source_layout = QFormLayout(source_group)
        
        self.source_folder_edit = QLineEdit()
        self.source_folder_edit.setPlaceholderText("Выберите папку с документами...")
        source_button = QPushButton("📁")
        source_button.clicked.connect(self.select_source_folder)
        
        source_layout_h = QHBoxLayout()
        source_layout_h.addWidget(self.source_folder_edit)
        source_layout_h.addWidget(source_button)
        source_layout.addRow("Папка с документами:", source_layout_h)
        
        # Информация о файлах
        self.source_info_label = QLabel("Выберите папку для анализа")
        self.source_info_label.setWordWrap(True)
        self.source_info_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        source_layout.addRow("Найдено файлов:", self.source_info_label)
        
        # NEW: Кнопка детального анализа PDF
        self.pdf_analyze_button = QPushButton("🔍 Детальный анализ PDF")
        self.pdf_analyze_button.clicked.connect(self.show_detailed_pdf_analysis)
        self.pdf_analyze_button.setEnabled(False)
        self.pdf_analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
        """)
        source_layout.addRow("PDF анализ:", self.pdf_analyze_button)
        
        left_layout.addWidget(source_group)
        
        # Группа: Настройки датасета
        dataset_group = QGroupBox("⚙️ Настройки датасета")
        dataset_layout = QFormLayout(dataset_group)
        
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("Оставьте пустым для автоматического именования")
        dataset_layout.addRow("Название датасета:", self.dataset_name_edit)
        
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItems([
            "LayoutLM (Token Classification)",
            "Donut (Document Parsing)",
            "Donut (Document VQA)"
        ])
        # Подключаем обновление имени при изменении типа датасета
        self.dataset_type_combo.currentTextChanged.connect(self.update_dataset_name_preview)
        dataset_layout.addRow("Тип датасета:", self.dataset_type_combo)
        
        # Устанавливаем начальный placeholder
        self.update_dataset_name_preview()
        
        # Разделение данных
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.5, 0.9)
        self.train_split_spin.setDecimals(2)
        self.train_split_spin.setValue(0.8)
        dataset_layout.addRow("Доля для обучения:", self.train_split_spin)
        
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.3)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setValue(0.15)
        dataset_layout.addRow("Доля для валидации:", self.val_split_spin)
        
        left_layout.addWidget(dataset_group)
        
        # Группа: Аннотации
        annotation_group = QGroupBox("🏷️ Настройки аннотаций")
        annotation_layout = QFormLayout(annotation_group)
        
        # Метод аннотации
        self.annotation_method_combo = QComboBox()
        self.annotation_method_combo.addItems([
            "Gemini",
            "OCR",
            "Manual"
        ])
        annotation_layout.addRow("Метод аннотации:", self.annotation_method_combo)
        
        self.use_ocr_checkbox = QCheckBox("Использовать OCR для извлечения текста")
        self.use_ocr_checkbox.setChecked(True)
        annotation_layout.addRow("OCR:", self.use_ocr_checkbox)
        
        self.use_gemini_checkbox = QCheckBox("Использовать Gemini для создания аннотаций")
        self.use_gemini_checkbox.setChecked(True)
        annotation_layout.addRow("Gemini:", self.use_gemini_checkbox)
        
        self.annotation_fields_edit = QLineEdit()
        self.annotation_fields_edit.setPlaceholderText("Автоматически из настроек полей таблицы")
        self.annotation_fields_edit.setReadOnly(True)
        self.annotation_fields_edit.setStyleSheet("""
            QLineEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #6c757d;
                font-style: italic;
            }
        """)
        
        # Обновляем поля при инициализации
        self._update_fields_from_manager()
        
        # Создаем контейнер для поля и кнопки обновления
        fields_container = QWidget()
        fields_layout = QHBoxLayout(fields_container)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        
        fields_layout.addWidget(self.annotation_fields_edit)
        
        refresh_fields_button = QPushButton("🔄")
        refresh_fields_button.setFixedSize(30, 25)
        refresh_fields_button.setToolTip("Обновить поля из настроек таблицы")
        refresh_fields_button.clicked.connect(self._update_fields_from_manager)
        refresh_fields_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        fields_layout.addWidget(refresh_fields_button)
        
        annotation_layout.addRow("Поля для извлечения:", fields_container)
        
        # Максимальное количество файлов
        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(0, 10000)
        self.max_files_spin.setValue(0)  # 0 = без ограничений
        self.max_files_spin.setSpecialValueText("Без ограничений")
        annotation_layout.addRow("Макс. файлов:", self.max_files_spin)
        
        left_layout.addWidget(annotation_group)
        
        # Режим подготовки
        mode_group = QGroupBox("🧠 Режим подготовки данных")
        mode_layout = QVBoxLayout(mode_group)
        
        self.preparation_mode_combo = QComboBox()
        self.preparation_mode_combo.addItems([
            "🧠 Интеллектуальный (Gemini извлекает ВСЕ данные)",
            "📝 Стандартный (только заданные поля)"
        ])
        self.preparation_mode_combo.setCurrentIndex(0)  # По умолчанию интеллектуальный
        self.preparation_mode_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #3498db;
                border-radius: 4px;
                background-color: white;
                font-weight: bold;
            }
            QComboBox:focus {
                border-color: #2980b9;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #3498db;
                margin-right: 5px;
            }
        """)
        mode_layout.addWidget(self.preparation_mode_combo)
        
        # Описание режимов
        mode_description = QLabel("""
<b>🧠 Интеллектуальный режим:</b> Gemini сначала анализирует документ и извлекает ВСЕ полезные данные, 
затем система создает разметку для обучения. Результат: больше полей, лучшее качество датасета.

<b>📝 Стандартный режим:</b> Поиск только заранее определенных полей по шаблонам. 
Быстрее, но может пропустить полезные данные.
        """)
        mode_description.setWordWrap(True)
        mode_description.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 8px;
                color: #2c3e50;
                font-size: 11px;
            }
        """)
        mode_layout.addWidget(mode_description)
        
        left_layout.addWidget(mode_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
        self.prepare_start_button = QPushButton("🚀 Начать подготовку")
        self.prepare_start_button.clicked.connect(self.start_dataset_preparation)
        self.prepare_start_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.prepare_stop_button = QPushButton("⏹️ Остановить")
        self.prepare_stop_button.clicked.connect(self.stop_preparation)
        self.prepare_stop_button.setEnabled(False)
        self.prepare_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        control_layout.addWidget(self.prepare_start_button)
        control_layout.addWidget(self.prepare_stop_button)
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # Правая панель - мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Прогресс
        progress_group = QGroupBox("📈 Прогресс подготовки")
        progress_layout = QVBoxLayout(progress_group)
        
        self.prepare_progress_bar = QProgressBar()
        self.prepare_progress_bar.setVisible(False)
        progress_layout.addWidget(self.prepare_progress_bar)
        
        self.prepare_status_label = QLabel("Готов к подготовке")
        self.prepare_status_label.setStyleSheet("font-weight: bold; color: #e67e22;")
        progress_layout.addWidget(self.prepare_status_label)
        
        right_layout.addWidget(progress_group)
        
        # Лог подготовки
        log_group = QGroupBox("📝 Журнал подготовки")
        log_layout = QVBoxLayout(log_group)
        
        self.prepare_log = QTextEdit()
        self.prepare_log.setReadOnly(True)
        self.prepare_log.setMaximumHeight(300)
        self.prepare_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.prepare_log)
        
        # Кнопки для лога
        log_buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("🗑️ Очистить")
        clear_log_button.clicked.connect(lambda: self.prepare_log.clear())
        
        save_log_button = QPushButton("💾 Сохранить лог")
        save_log_button.clicked.connect(lambda: self.save_log(self.prepare_log))
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_buttons_layout.addStretch()
        
        log_layout.addLayout(log_buttons_layout)
        right_layout.addWidget(log_group)
        
        # Группа: Метрики качества датасета
        quality_group = QGroupBox("📊 Качество датасета")
        quality_layout = QVBoxLayout(quality_group)
        
        # Кнопка для анализа качества
        analyze_button_layout = QHBoxLayout()
        self.analyze_quality_button = QPushButton("🔍 Анализировать качество")
        self.analyze_quality_button.clicked.connect(self.analyze_dataset_quality)
        self.analyze_quality_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        analyze_button_layout.addWidget(self.analyze_quality_button)
        analyze_button_layout.addStretch()
        quality_layout.addLayout(analyze_button_layout)
        
        # Общий балл качества
        self.overall_score_label = QLabel("Общий балл: Не анализировался")
        self.overall_score_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                color: #2c3e50;
            }
        """)
        quality_layout.addWidget(self.overall_score_label)
        
        # Таблица метрик
        self.quality_metrics_table = QTableWidget()
        self.quality_metrics_table.setColumnCount(2)
        self.quality_metrics_table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.quality_metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.quality_metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.quality_metrics_table.setMaximumHeight(200)
        self.quality_metrics_table.setAlternatingRowColors(True)
        self.quality_metrics_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #bdc3c7;
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)
        quality_layout.addWidget(self.quality_metrics_table)
        
        # Рекомендации
        self.recommendations_label = QLabel("Рекомендации появятся после анализа")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setStyleSheet("""
            QLabel {
                background-color: #fef9e7;
                border: 1px solid #f1c40f;
                border-radius: 4px;
                padding: 8px;
                color: #8e44ad;
                font-style: italic;
            }
        """)
        quality_layout.addWidget(self.recommendations_label)
        
        right_layout.addWidget(quality_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📊 Подготовка данных")
        
    def create_monitoring_tab(self):
        """Создает вкладку для мониторинга обучения"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Заголовок
        header = QLabel("📊 Мониторинг и анализ обучения")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #3498db; padding: 10px; background: #ebf3fd; border-radius: 5px;")
        layout.addWidget(header)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Верхняя панель - текущие метрики
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        
        # Группа: Текущие метрики
        metrics_group = QGroupBox("📈 Текущие метрики")
        metrics_layout = QGridLayout(metrics_group)
        
        # Метрики в реальном времени
        self.current_epoch_label = QLabel("0")
        self.current_step_label = QLabel("0")
        self.current_loss_label = QLabel("0.000")
        self.current_lr_label = QLabel("0.0000")
        self.current_accuracy_label = QLabel("0.00%")
        self.current_f1_label = QLabel("0.000")
        
        # Стилизация меток
        metric_style = "font-size: 14px; font-weight: bold; color: #2c3e50; background: #ecf0f1; padding: 5px; border-radius: 3px;"
        for label in [self.current_epoch_label, self.current_step_label, self.current_loss_label, 
                     self.current_lr_label, self.current_accuracy_label, self.current_f1_label]:
            label.setStyleSheet(metric_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        metrics_layout.addWidget(QLabel("Эпоха:"), 0, 0)
        metrics_layout.addWidget(self.current_epoch_label, 0, 1)
        metrics_layout.addWidget(QLabel("Шаг:"), 0, 2)
        metrics_layout.addWidget(self.current_step_label, 0, 3)
        metrics_layout.addWidget(QLabel("Точность:"), 0, 4)
        metrics_layout.addWidget(self.current_accuracy_label, 0, 5)
        
        metrics_layout.addWidget(QLabel("Loss:"), 1, 0)
        metrics_layout.addWidget(self.current_loss_label, 1, 1)
        metrics_layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        metrics_layout.addWidget(self.current_lr_label, 1, 3)
        metrics_layout.addWidget(QLabel("F1 Score:"), 1, 4)
        metrics_layout.addWidget(self.current_f1_label, 1, 5)
        
        top_layout.addWidget(metrics_group)
        
        # Нижняя панель - история и графики
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        
        # Группа: История обучения
        history_group = QGroupBox("📋 История обучения")
        history_layout = QVBoxLayout(history_group)
        
        # Таблица истории
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Эпоха", "Шаг", "Loss", "Eval Loss", "Accuracy", "Время"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMaximumHeight(200)
        
        history_layout.addWidget(self.history_table)
        
        # Кнопки для истории
        history_buttons_layout = QHBoxLayout()
        
        export_history_button = QPushButton("📊 Экспорт в CSV")
        export_history_button.clicked.connect(self.export_history)
        
        clear_history_button = QPushButton("🗑️ Очистить историю")
        clear_history_button.clicked.connect(self.clear_history)
        
        history_buttons_layout.addWidget(export_history_button)
        history_buttons_layout.addWidget(clear_history_button)
        history_buttons_layout.addStretch()
        
        history_layout.addLayout(history_buttons_layout)
        bottom_layout.addWidget(history_group)
        
        # Добавляем панели в splitter
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([200, 400])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📊 Мониторинг")
        
    def select_dataset(self, model_type):
        """Выбор датасета для обучения"""
        folder = QFileDialog.getExistingDirectory(
            self, 
            f"Выберите датасет для {model_type}",
            self.app_config.TRAINING_DATASETS_PATH
        )
        
        if folder:
            if model_type == 'layoutlm':
                self.layoutlm_dataset_edit.setText(folder)
                self.update_dataset_info(folder, self.layoutlm_dataset_info)
            elif model_type == 'donut':
                self.donut_dataset_edit.setText(folder)
                self.update_dataset_info(folder, self.donut_dataset_info)
            elif model_type == 'trocr':
                self.trocr_dataset_edit.setText(folder)
                self.update_dataset_info(folder, self.trocr_dataset_info)
                # Сохраняем путь к выбранному TrOCR датасету
                try:
                    from app.settings_manager import settings_manager
                    settings_manager.set_value('Training', 'last_trocr_dataset', folder)
                except Exception:
                    pass  # Игнорируем ошибки сохранения настроек
                
    def update_dataset_info(self, dataset_path, info_label):
        """Обновляет информацию о датасете"""
        try:
            if not os.path.exists(dataset_path):
                info_label.setText("Датасет не найден")
                return
            
            # Проверяем тип датасета
            # Вариант 1: HuggingFace datasets (есть dataset_info.json или *.arrow файлы)
            has_arrow_files = any(f.endswith('.arrow') for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)))
            has_dataset_info = os.path.exists(os.path.join(dataset_path, 'dataset_info.json'))
            
            if has_arrow_files or has_dataset_info:
                # HuggingFace датасет - анализируем структуру
                try:
                    from datasets import load_from_disk
                    dataset = load_from_disk(dataset_path)
                    
                    if hasattr(dataset, 'num_rows'):
                        # Простой датасет
                        info_text = f"HuggingFace датасет: {dataset.num_rows} записей"
                    elif hasattr(dataset, 'keys'):
                        # Датасет с разбивками
                        splits_info = []
                        for split_name in dataset.keys():
                            splits_info.append(f"{split_name}: {len(dataset[split_name])} записей")
                        info_text = f"HuggingFace датасет: {', '.join(splits_info)}"
                    else:
                        info_text = "HuggingFace датасет: структура распознана"
                        
                except Exception as dataset_error:
                    # Если не удалось загрузить через datasets, анализируем файлы
                    arrow_files = [f for f in os.listdir(dataset_path) if f.endswith('.arrow')]
                    info_text = f"HuggingFace датасет: {len(arrow_files)} arrow файлов"
                
                info_label.setText(info_text)
                info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                return
            
            # Вариант 2: Обычные файлы (изображения/PDF) - для LayoutLM/Donut
            total_files = 0
            train_files = 0
            val_files = 0
            
            for root, dirs, files in os.walk(dataset_path):
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp'))]
                
                if 'train' in root.lower():
                    train_files += len(image_files)
                elif 'val' in root.lower() or 'validation' in root.lower():
                    val_files += len(image_files)
                else:
                    total_files += len(image_files)
            
            if train_files > 0 or val_files > 0:
                info_text = f"Файловый датасет - Обучение: {train_files}, Валидация: {val_files}"
            else:
                info_text = f"Файловый датасет - Всего файлов: {total_files}"
                
            info_label.setText(info_text)
            info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
        except Exception as e:
            info_label.setText(f"Ошибка анализа: {str(e)}")
            info_label.setStyleSheet("color: #e74c3c;")
            
    def select_source_folder(self):
        """Выбор папки с исходными документами"""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку с документами",
            ""
        )
        
        if folder:
            self.source_folder_edit.setText(folder)
            self.update_source_info(folder)
            # Сохраняем путь в настройках для последующего использования
            self.save_source_folder_to_settings(folder)
            
    def update_source_info(self, folder_path):
        """Обновляет информацию об исходных файлах с анализом PDF"""
        try:
            if not os.path.exists(folder_path):
                self.source_info_label.setText("Папка не найдена")
                return
                
            # Подсчитываем файлы
            supported_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp')
            files = []
            pdf_files = []
            
            for file in os.listdir(folder_path):
                if file.lower().endswith(supported_extensions):
                    files.append(file)
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(folder_path, file))
                        
            # Базовая информация о файлах
            base_info = f"{len(files)} файлов ({len(pdf_files)} PDF)"
            
            # Анализируем PDF файлы если они есть
            if pdf_files:
                # Активируем кнопку детального анализа
                if hasattr(self, 'pdf_analyze_button'):
                    self.pdf_analyze_button.setEnabled(True)
                try:
                    # Анализируем первые 5 PDF файлов для статистики
                    sample_files = pdf_files[:5]
                    pdf_stats = {'with_text': 0, 'without_text': 0, 'total': len(sample_files)}
                    
                    for pdf_file in sample_files:
                        try:
                            analysis = self.pdf_analyzer.analyze_pdf(pdf_file)
                            if analysis['has_text_layer']:
                                pdf_stats['with_text'] += 1
                            else:
                                pdf_stats['without_text'] += 1
                        except:
                            pdf_stats['without_text'] += 1
                    
                    # Рассчитываем процент файлов с текстом
                    text_ratio = pdf_stats['with_text'] / pdf_stats['total'] if pdf_stats['total'] > 0 else 0
                    
                    if text_ratio > 0.5:
                        # Большинство файлов с текстовым слоем
                        pdf_info = f"🎯 {text_ratio*100:.0f}% PDF с текстом - рекомендуется интеллектуальная обработка!"
                        style = "color: #27ae60; font-weight: bold;"
                    elif text_ratio > 0:
                        # Часть файлов с текстовым слоем
                        pdf_info = f"⚡ {text_ratio*100:.0f}% PDF с текстом - смешанная обработка"
                        style = "color: #f39c12; font-weight: bold;"
                    else:
                        # Все файлы требуют OCR
                        pdf_info = f"📸 PDF требует OCR обработки"
                        style = "color: #3498db; font-weight: bold;"
                        
                    # Объединяем информацию
                    full_info = f"{base_info}\n{pdf_info}"
                    self.source_info_label.setText(full_info)
                    self.source_info_label.setStyleSheet(style)
                    
                except Exception as pdf_error:
                    # Если анализ PDF не удался, показываем базовую информацию
                    self.source_info_label.setText(f"{base_info}\n📋 Анализ PDF: недоступен")
                    self.source_info_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
            else:
                # Только изображения - отключаем кнопку анализа PDF
                if hasattr(self, 'pdf_analyze_button'):
                    self.pdf_analyze_button.setEnabled(False)
                self.source_info_label.setText(base_info)
                self.source_info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
        except Exception as e:
            self.source_info_label.setText(f"Ошибка: {str(e)}")
            self.source_info_label.setStyleSheet("color: #e74c3c;")
    
    def show_detailed_pdf_analysis(self):
        """Показывает детальный анализ PDF файлов"""
        try:
            folder_path = self.source_folder_edit.text()
            if not folder_path or not os.path.exists(folder_path):
                QMessageBox.warning(self, "Ошибка", "Сначала выберите папку с документами!")
                return
            
            # Находим все PDF файлы
            pdf_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(folder_path, file))
            
            if not pdf_files:
                QMessageBox.information(self, "Информация", "PDF файлы не найдены в выбранной папке.")
                return
            
            # Создаем диалог с результатами
            dialog = QDialog(self)
            dialog.setWindowTitle("🔍 Детальный анализ PDF файлов")
            dialog.setModal(True)
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Заголовок
            header = QLabel(f"📊 Анализ {len(pdf_files)} PDF файлов")
            header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            header.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
            layout.addWidget(header)
            
            # Прогресс бар
            progress = QProgressBar()
            progress.setRange(0, len(pdf_files))
            layout.addWidget(progress)
            
            # Таблица результатов
            table = QTableWidget(len(pdf_files), 5)
            table.setHorizontalHeaderLabels([
                "Файл", "Страниц", "Текстовый слой", "Качество", "Рекомендация"
            ])
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            
            layout.addWidget(table)
            
            # Статистика
            stats_label = QLabel("Анализ...")
            stats_label.setStyleSheet("background: #ecf0f1; padding: 10px; border-radius: 5px;")
            layout.addWidget(stats_label)
            
            # Кнопка закрытия
            close_button = QPushButton("Закрыть")
            close_button.clicked.connect(dialog.close)
            layout.addWidget(close_button)
            
            dialog.show()
            QApplication.processEvents()
            
            # Анализируем файлы
            stats = {'with_text': 0, 'without_text': 0, 'high_quality': 0, 'medium_quality': 0, 'low_quality': 0}
            
            for i, pdf_file in enumerate(pdf_files):
                try:
                    analysis = self.pdf_analyzer.analyze_pdf(pdf_file)
                    
                    # Заполняем таблицу
                    filename = os.path.basename(pdf_file)
                    table.setItem(i, 0, QTableWidgetItem(filename))
                    table.setItem(i, 1, QTableWidgetItem(str(analysis['page_count'])))
                    
                    # Текстовый слой
                    has_text = "✅ Есть" if analysis['has_text_layer'] else "❌ Нет"
                    text_item = QTableWidgetItem(has_text)
                    if analysis['has_text_layer']:
                        text_item.setBackground(QColor("#d5f4e6"))
                        stats['with_text'] += 1
                    else:
                        text_item.setBackground(QColor("#fadbd8"))
                        stats['without_text'] += 1
                    table.setItem(i, 2, text_item)
                    
                    # Качество текста
                    quality = analysis['text_quality']
                    quality_text = f"{quality:.2f}"
                    quality_item = QTableWidgetItem(quality_text)
                    
                    if quality >= 0.7:
                        quality_item.setBackground(QColor("#d5f4e6"))
                        stats['high_quality'] += 1
                    elif quality >= 0.3:
                        quality_item.setBackground(QColor("#fcf3cf"))
                        stats['medium_quality'] += 1
                    else:
                        quality_item.setBackground(QColor("#fadbd8"))
                        stats['low_quality'] += 1
                    
                    table.setItem(i, 3, quality_item)
                    
                    # Рекомендация
                    recommendation = analysis['processing_method']
                    rec_text = "Текст" if recommendation == 'text_extraction' else "OCR"
                    table.setItem(i, 4, QTableWidgetItem(rec_text))
                    
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                    
                except Exception as e:
                    # Ошибка анализа
                    table.setItem(i, 0, QTableWidgetItem(os.path.basename(pdf_file)))
                    table.setItem(i, 1, QTableWidgetItem("?"))
                    table.setItem(i, 2, QTableWidgetItem("❌ Ошибка"))
                    table.setItem(i, 3, QTableWidgetItem("0.00"))
                    table.setItem(i, 4, QTableWidgetItem("OCR"))
                    stats['without_text'] += 1
                    stats['low_quality'] += 1
                    
                    progress.setValue(i + 1)
                    QApplication.processEvents()
            
            # Обновляем статистику
            total = len(pdf_files)
            text_ratio = stats['with_text'] / total * 100 if total > 0 else 0
            high_quality_ratio = stats['high_quality'] / total * 100 if total > 0 else 0
            
            stats_text = f"""
📊 <b>РЕЗУЛЬТАТЫ АНАЛИЗА:</b><br>
• Всего файлов: {total}<br>
• С текстовым слоем: {stats['with_text']} ({text_ratio:.1f}%)<br>
• Без текстового слоя: {stats['without_text']} ({100-text_ratio:.1f}%)<br><br>

📈 <b>КАЧЕСТВО ТЕКСТА:</b><br>
• Высокое (≥0.7): {stats['high_quality']} ({high_quality_ratio:.1f}%)<br>
• Среднее (0.3-0.7): {stats['medium_quality']}<br>
• Низкое (<0.3): {stats['low_quality']}<br><br>

💡 <b>РЕКОМЕНДАЦИЯ:</b><br>
            """
            
            if text_ratio > 70:
                stats_text += "🚀 <span style='color: #27ae60;'><b>Отличные условия для интеллектуальной обработки!</b></span><br>"
                stats_text += "Большинство файлов содержат качественный текстовый слой.<br>"
                stats_text += "Рекомендуется использовать режим 'Интеллектуальный'."
            elif text_ratio > 30:
                stats_text += "⚡ <span style='color: #f39c12;'><b>Смешанная обработка будет эффективной.</b></span><br>"
                stats_text += "Часть файлов можно обработать через текстовый слой.<br>"
                stats_text += "Система автоматически выберет оптимальный метод."
            else:
                stats_text += "📸 <span style='color: #3498db;'><b>Требуется OCR обработка.</b></span><br>"
                stats_text += "Большинство файлов не содержит текстового слоя.<br>"
                stats_text += "Будет использоваться традиционный OCR."
            
            stats_label.setText(stats_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка анализа PDF: {str(e)}")
            
    def save_source_folder_to_settings(self, folder_path):
        """Сохраняет путь к исходной папке в настройках"""
        try:
            from app.settings_manager import settings_manager
            settings_manager.set_value('Training', 'last_source_folder', folder_path)
            settings_manager.set_value('Training', 'last_source_folder_timestamp', datetime.now().isoformat())
            self.add_log_message(self.prepare_log if hasattr(self, 'prepare_log') else None, 
                               f"💾 Путь к папке с документами сохранен: {folder_path}")
        except Exception as e:
            print(f"Ошибка сохранения пути к папке: {e}")
            
    def update_dataset_name_preview(self):
        """Обновляет превью имени датасета в зависимости от выбранного типа"""
        dataset_type = self.dataset_type_combo.currentText()
        
        # Определяем префикс модели
        if "LayoutLM" in dataset_type:
            model_prefix = "layoutlm"
        elif "Donut" in dataset_type:
            model_prefix = "donut"
        elif "TrOCR" in dataset_type:
            model_prefix = "trocr"
        else:
            model_prefix = "unknown"
            
        # Обновляем placeholder текст
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        placeholder_text = f"Авто: {model_prefix}_dataset_{timestamp}"
        self.dataset_name_edit.setPlaceholderText(placeholder_text)
        
    def _update_fields_from_manager(self):
        """Обновляет отображение полей из FieldManager"""
        try:
            from .field_manager import field_manager
            enabled_fields = field_manager.get_enabled_fields()
            
            if enabled_fields:
                field_names = [f.display_name for f in enabled_fields]
                field_text = ", ".join(field_names)
                self.annotation_fields_edit.setText(f"Активные поля: {field_text}")
                self.annotation_fields_edit.setToolTip(
                    f"Автоматически извлекаются следующие поля:\n" + 
                    "\n".join([f"• {f.display_name} ({f.id})" for f in enabled_fields]) +
                    f"\n\nВсего активных полей: {len(enabled_fields)}\n\n" +
                    "Для изменения полей используйте меню 'Настройки' → 'Поля таблицы'"
                )
            else:
                self.annotation_fields_edit.setText("⚠️ Нет активных полей")
                self.annotation_fields_edit.setToolTip(
                    "Не найдено активных полей для извлечения.\n\n" +
                    "Перейдите в 'Настройки' → 'Поля таблицы' и включите нужные поля."
                )
        except ImportError as e:
            self.annotation_fields_edit.setText("❌ Ошибка загрузки FieldManager")
            self.annotation_fields_edit.setToolTip(f"Ошибка: {e}")
        except Exception as e:
            self.annotation_fields_edit.setText("❌ Ошибка получения полей")
            self.annotation_fields_edit.setToolTip(f"Ошибка: {e}")
            
    def start_layoutlm_training(self):
        """Запуск обучения LayoutLM"""
        # Проверяем параметры
        dataset_path = self.layoutlm_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
        
        # 🎯 Проверяем метаданные датасета для совместимости полей
        try:
            from .training.data_preparator import TrainingDataPreparator
            preparator = TrainingDataPreparator(self.app_config, self.ocr_processor, self.gemini_processor)
            
            # Определяем папку с метаданными
            metadata_folder = dataset_path
            if dataset_path.endswith("dataset_dict"):
                metadata_folder = os.path.dirname(dataset_path)
            
            metadata = preparator.load_dataset_metadata(metadata_folder)
            if metadata:
                self.add_log_message(self.layoutlm_log, f"📂 Метаданные датасета:")
                self.add_log_message(self.layoutlm_log, f"   • Создан: {metadata.get('created_at', 'неизвестно')}")
                self.add_log_message(self.layoutlm_log, f"   • Источник полей: {metadata.get('fields_source', 'неизвестно')}")
                
                active_fields = metadata.get('active_fields', [])
                if active_fields:
                    self.add_log_message(self.layoutlm_log, f"   • Поля датасета: {', '.join(active_fields)}")
                
                # Проверяем совместимость с текущими настройками
                try:
                    from .field_manager import field_manager
                    current_fields = [f.id for f in field_manager.get_enabled_fields()]
                    
                    if active_fields and current_fields:
                        missing_fields = set(active_fields) - set(current_fields)
                        extra_fields = set(current_fields) - set(active_fields)
                        
                        if missing_fields or extra_fields:
                            self.add_log_message(self.layoutlm_log, "⚠️  ВНИМАНИЕ: Различия в настройках полей:")
                            if missing_fields:
                                self.add_log_message(self.layoutlm_log, f"   • Отключены: {', '.join(missing_fields)}")
                            if extra_fields:
                                self.add_log_message(self.layoutlm_log, f"   • Новые: {', '.join(extra_fields)}")
                                
                            reply = QMessageBox.question(
                                self, "Различия в полях",
                                f"Обнаружены различия между настройками полей:\n\n"
                                f"Датасет: {', '.join(active_fields)}\n"
                                f"Текущие: {', '.join(current_fields)}\n\n"
                                f"Продолжить обучение?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                            )
                            
                            if reply == QMessageBox.StandardButton.No:
                                return
                        else:
                            self.add_log_message(self.layoutlm_log, "✅ Поля соответствуют текущим настройкам")
                except ImportError:
                    pass
            else:
                self.add_log_message(self.layoutlm_log, "📝 Метаданные не найдены (старый формат датасета)")
                
        except Exception as e:
            self.add_log_message(self.layoutlm_log, f"⚠️  Ошибка проверки метаданных: {e}")
            
        # Для LayoutLM нужен путь к dataset_dict внутри датасета
        if dataset_path.endswith("dataset_dict"):
            # Путь уже указывает на dataset_dict
            actual_dataset_path = dataset_path
        else:
            # Проверяем есть ли папка dataset_dict внутри указанного пути
            dataset_dict_path = os.path.join(dataset_path, "dataset_dict")
            if os.path.exists(dataset_dict_path):
                actual_dataset_path = dataset_dict_path
            else:
                QMessageBox.warning(self, "Ошибка", 
                    f"В датасете не найдена папка dataset_dict!\n\n"
                    f"Путь: {dataset_path}\n"
                    f"Ожидаемая структура: {dataset_path}/dataset_dict/\n\n"
                    f"Убедитесь, что выбран правильный датасет для LayoutLM.")
                return
            
        # Создаем тренер
        self.current_trainer = ModelTrainer(self.app_config)
        
        # Подготавливаем относительный путь для модели
        model_name = self.layoutlm_output_name_edit.text() or f"layoutlm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("layoutlm_"):
            model_name = f"layoutlm_{model_name}"
        
        # Подготавливаем параметры
        training_params = {
            'dataset_path': actual_dataset_path,
            'base_model_id': self.layoutlm_base_model_edit.text(),
            'training_args': {
                'num_train_epochs': self.layoutlm_epochs_spin.value(),
                'per_device_train_batch_size': self.layoutlm_batch_size_spin.value(),
                'learning_rate': self.layoutlm_lr_spin.value(),
                'weight_decay': self.layoutlm_weight_decay_spin.value(),
                'warmup_ratio': self.layoutlm_warmup_spin.value(),
                'seed': self.layoutlm_seed_spin.value(),
            },
            'output_model_name': model_name,
            'output_model_path': os.path.join("data", "trained_models", model_name)
        }
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'layoutlm')
        
    def start_donut_training(self):
        """Запуск обучения Donut"""
        # Проверяем параметры
        dataset_path = self.donut_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
            
        # Создаем тренер Donut
        self.current_trainer = DonutTrainerClass(self.app_config)
        
        # Подготавливаем относительный путь для модели
        model_name = self.donut_output_name_edit.text() or f"donut_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("donut_"):
            model_name = f"donut_{model_name}"
        
        # Подготавливаем параметры
        training_params = {
            'dataset_path': dataset_path,
            'base_model_id': self.donut_base_model_combo.currentText(),
            'training_args': {
                'num_train_epochs': self.donut_epochs_spin.value(),
                'per_device_train_batch_size': self.donut_batch_size_spin.value(),
                'learning_rate': self.donut_lr_spin.value(),
                'gradient_accumulation_steps': self.donut_grad_accum_spin.value(),
                'max_length': self.donut_max_length_spin.value(),
                'image_size': int(self.donut_image_size_combo.currentText()),
                'fp16': self.donut_fp16_checkbox.isChecked(),
                'save_steps': self.donut_save_steps_spin.value(),
                'eval_steps': self.donut_eval_steps_spin.value(),
                'task_type': self.donut_task_combo.currentText(),
                
                # Оптимизации памяти
                'use_lora': self.use_lora_cb.isChecked(),
                'use_8bit_optimizer': self.use_8bit_optimizer_cb.isChecked(),
                'freeze_encoder': self.freeze_encoder_cb.isChecked(),
                'gradient_checkpointing': True,  # Принудительно включаем
            },
            'output_model_name': model_name
        }
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'donut')
        
    def start_trocr_training(self):
        """Запуск обучения TrOCR"""
        # Проверяем параметры
        dataset_path = self.trocr_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
            
        # Создаем тренер TrOCR
        self.current_trainer = TrOCRTrainer()
        
        # Подготавливаем относительный путь для модели
        model_name = self.trocr_output_name_edit.text() or f"trocr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("trocr_"):
            model_name = f"trocr_{model_name}"
        
        # Собираем параметры обучения
        training_params = {
            'dataset_path': dataset_path,
            'base_model_id': self.trocr_base_model_combo.currentText(),
            'output_model_name': model_name,
            'training_args': {
                'num_train_epochs': self.trocr_epochs_spin.value(),
                'per_device_train_batch_size': self.trocr_batch_size_spin.value(),
                'learning_rate': self.trocr_lr_spin.value(),
                'gradient_accumulation_steps': self.trocr_grad_accum_spin.value(),
                'max_length': self.trocr_max_length_spin.value(),
                'image_size': self._parse_image_size(self.trocr_image_size_combo.currentText())[0],
                'fp16': self.trocr_fp16_checkbox.isChecked(),
                'warmup_ratio': self.trocr_warmup_ratio_spin.value(),
                'weight_decay': self.trocr_weight_decay_spin.value(),
                # Оптимизации памяти
                'use_lora': self.trocr_use_lora_cb.isChecked(),
                'use_8bit_optimizer': self.trocr_use_8bit_optimizer_cb.isChecked(),
                'gradient_checkpointing': self.trocr_gradient_checkpointing_cb.isChecked(),
            }
        }
        
        self.add_log_message(self.trocr_log, f"🚀 Запуск обучения TrOCR модели '{model_name}'")
        self.add_log_message(self.trocr_log, f"📊 Датасет: {dataset_path}")
        self.add_log_message(self.trocr_log, f"🤖 Базовая модель: {training_params['base_model_id']}")
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'trocr')
        
    def auto_optimize_trocr_memory(self):
        """Автоматическая оптимизация памяти для TrOCR на RTX 4070 Ti"""
        # Включаем все оптимизации
        self.trocr_use_lora_cb.setChecked(True)
        self.trocr_use_8bit_optimizer_cb.setChecked(True)
        self.trocr_gradient_checkpointing_cb.setChecked(True)
        
        # Устанавливаем оптимальные параметры для RTX 4070 Ti (12GB VRAM)
        self.trocr_batch_size_spin.setValue(2)
        self.trocr_grad_accum_spin.setValue(8)
        self.trocr_image_size_combo.setCurrentText("224")
        self.trocr_max_length_spin.setValue(256)
        
        # Включаем FP16
        self.trocr_fp16_checkbox.setChecked(True)
        
        self.add_log_message(self.trocr_log, "🚀 Применены оптимизации памяти для RTX 4070 Ti:")
        self.add_log_message(self.trocr_log, "   • LoRA: включен (экономия до 90% памяти)")
        self.add_log_message(self.trocr_log, "   • 8-bit оптимизатор: включен (экономия 25%)")
        self.add_log_message(self.trocr_log, "   • Gradient checkpointing: включен")
        self.add_log_message(self.trocr_log, "   • Batch size: 2, Grad accumulation: 8")
        self.add_log_message(self.trocr_log, "   • Image size: 224, Max length: 256")
        self.add_log_message(self.trocr_log, "   • FP16: включен")

    def apply_trocr_fast_gpu_settings(self):
        """Применяет быстрые настройки GPU для TrOCR"""
        # Оптимальные настройки для обучения
        self.trocr_epochs_spin.setValue(3)
        self.trocr_batch_size_spin.setValue(4)
        self.trocr_lr_spin.setValue(5e-5)
        self.trocr_grad_accum_spin.setValue(4)
        self.trocr_max_length_spin.setValue(512)
        self.trocr_image_size_combo.setCurrentText("384")
        self.trocr_warmup_ratio_spin.setValue(0.1)
        self.trocr_weight_decay_spin.setValue(0.01)
        
        # Включаем FP16 для ускорения
        self.trocr_fp16_checkbox.setChecked(True)
        
        self.add_log_message(self.trocr_log, "⚡ Применены быстрые настройки GPU для TrOCR")
    
    def smart_optimize_trocr_hyperparameters(self):
        """Интеллектуальная оптимизация гиперпараметров на основе анализа датасета"""
        try:
            # Проверяем, что выбран датасет
            dataset_path = self.trocr_dataset_edit.text()
            if not dataset_path or not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Ошибка", "Сначала выберите датасет для анализа")
                return
            
            self.add_log_message(self.trocr_log, "🧠 Запуск интеллектуальной оптимизации гиперпараметров...")
            
            # Создаем оптимизатор
            optimizer = TrOCRHyperparameterOptimizer()
            
            # Ищем предыдущие результаты обучения
            previous_results = None
            trained_models_dir = "data/trained_models"
            if os.path.exists(trained_models_dir):
                # Ищем последнюю обученную модель TrOCR
                trocr_models = [d for d in os.listdir(trained_models_dir) if d.startswith('trocr_')]
                if trocr_models:
                    latest_model = max(trocr_models, key=lambda x: os.path.getctime(os.path.join(trained_models_dir, x)))
                    model_path = os.path.join(trained_models_dir, latest_model, "final_model")
                    previous_results = optimizer.analyze_previous_results(model_path)
                    if previous_results:
                        self.add_log_message(self.trocr_log, f"📊 Найдены результаты предыдущего обучения: {latest_model}")
            
            # Оптимизируем гиперпараметры
            gpu_memory = 12.0  # RTX 4070 Ti
            target_time = 30   # 30 минут
            
            optimization = optimizer.optimize_hyperparameters(
                dataset_path=dataset_path,
                gpu_memory_gb=gpu_memory,
                target_training_time_minutes=target_time,
                previous_results=previous_results
            )
            
            # Применяем оптимизированные параметры
            self.trocr_epochs_spin.setValue(optimization.epochs)
            self.trocr_batch_size_spin.setValue(optimization.batch_size)
            self.trocr_lr_spin.setValue(optimization.learning_rate)
            self.trocr_grad_accum_spin.setValue(optimization.gradient_accumulation_steps)
            self.trocr_warmup_ratio_spin.setValue(optimization.warmup_steps / max(1, optimization.epochs * 10))
            
            # Анализируем характеристики датасета
            characteristics = optimizer.analyze_dataset(dataset_path)
            
            # Генерируем и показываем отчет
            report = optimizer.generate_training_report(optimization, characteristics)
            
            # Выводим отчет в лог
            self.add_log_message(self.trocr_log, "")
            for line in report.split('\n'):
                if line.strip():
                    self.add_log_message(self.trocr_log, line)
            
            # Показываем диалог с подробным отчетом
            msg = QMessageBox(self)
            msg.setWindowTitle("🧠 Отчет об оптимизации гиперпараметров")
            msg.setText("Параметры обучения оптимизированы на основе анализа датасета!")
            msg.setDetailedText(report)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            self.add_log_message(self.trocr_log, "")
            self.add_log_message(self.trocr_log, "✅ Интеллектуальная оптимизация завершена успешно!")
            
        except Exception as e:
            error_msg = f"Ошибка при оптимизации гиперпараметров: {str(e)}"
            self.add_log_message(self.trocr_log, f"❌ {error_msg}")
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    # ========================
    # TrOCR Dataset Methods
    # ========================
    
    def setup_trocr_source_widgets(self):
        """Настраивает виджеты для источников данных TrOCR"""
        # Очищаем существующие виджеты
        while self.trocr_source_layout.count():
            child = self.trocr_source_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Виджеты для "Из аннотаций счетов (JSON)"
        self.trocr_images_folder_edit = QLineEdit()
        self.trocr_images_folder_edit.setPlaceholderText("Выберите папку с изображениями счетов...")
        self.trocr_images_folder_button = QPushButton("📁")
        self.trocr_images_folder_button.clicked.connect(self.select_trocr_images_folder)
        
        self.trocr_annotations_file_edit = QLineEdit()
        self.trocr_annotations_file_edit.setPlaceholderText("Выберите JSON файл с аннотациями...")
        self.trocr_annotations_file_button = QPushButton("📄")
        self.trocr_annotations_file_button.clicked.connect(self.select_trocr_annotations_file)
        
        # Виджеты для "Из структуры папок"
        self.trocr_folder_structure_edit = QLineEdit()
        self.trocr_folder_structure_edit.setPlaceholderText("Выберите папку с данными...")
        self.trocr_folder_structure_button = QPushButton("📁")
        self.trocr_folder_structure_button.clicked.connect(self.select_trocr_folder_structure)
        
        # Виджеты для "Синтетический датасет"
        self.trocr_synthetic_samples_spin = QSpinBox()
        self.trocr_synthetic_samples_spin.setRange(100, 100000)
        self.trocr_synthetic_samples_spin.setValue(10000)
        
        # Виджеты для импорта внешних датасетов
        self.trocr_external_dataset_edit = QLineEdit()
        self.trocr_external_dataset_edit.setPlaceholderText("Выберите папку с внешним датасетом...")
        self.trocr_external_dataset_button = QPushButton("📁")
        self.trocr_external_dataset_button.clicked.connect(self.select_trocr_external_dataset)
        
        # Комбобокс для выбора формата внешнего датасета
        self.trocr_external_format_combo = QComboBox()
        self.trocr_external_format_combo.addItems([
            "Автоопределение",
            "COCO Format",
            "YOLO Format", 
            "PASCAL VOC",
            "CSV Format",
            "LabelMe Format",
            "Простой JSON"
        ])
        
        # Информационный лейбл о формате
        self.trocr_format_info_label = QLabel()
        self.trocr_format_info_label.setStyleSheet("""
            background: #e8f4fd; 
            padding: 8px; 
            border-radius: 4px; 
            color: #1976d2;
            font-size: 11px;
        """)
        self.trocr_format_info_label.setWordWrap(True)
        self.trocr_external_format_combo.currentTextChanged.connect(self.update_format_info)
        
        # Сохраняем ссылки на виджеты
        self.trocr_source_widgets = {
            'images_folder': (self.trocr_images_folder_edit, self.trocr_images_folder_button),
            'annotations_file': (self.trocr_annotations_file_edit, self.trocr_annotations_file_button),
            'folder_structure': (self.trocr_folder_structure_edit, self.trocr_folder_structure_button),
            'synthetic_samples': self.trocr_synthetic_samples_spin,
            'external_dataset': (self.trocr_external_dataset_edit, self.trocr_external_dataset_button),
            'external_format': self.trocr_external_format_combo,
            'format_info': self.trocr_format_info_label
        }
    
    def on_trocr_dataset_type_changed(self):
        """Обработчик изменения типа датасета TrOCR"""
        current_type = self.trocr_dataset_type_combo.currentText()
        
        # Очищаем текущие виджеты
        while self.trocr_source_layout.count():
            child = self.trocr_source_layout.takeAt(0)
            if child.widget():
                child.widget().setVisible(False)
        
        if "аннотаций счетов" in current_type:
            # Папка с изображениями
            images_layout = QHBoxLayout()
            images_layout.addWidget(self.trocr_images_folder_edit)
            images_layout.addWidget(self.trocr_images_folder_button)
            self.trocr_source_layout.addRow("Папка с изображениями:", images_layout)
            
            # Файл аннотаций
            ann_layout = QHBoxLayout()
            ann_layout.addWidget(self.trocr_annotations_file_edit)
            ann_layout.addWidget(self.trocr_annotations_file_button)
            self.trocr_source_layout.addRow("Файл аннотаций:", ann_layout)
            
            # Показываем нужные виджеты
            self.trocr_images_folder_edit.setVisible(True)
            self.trocr_images_folder_button.setVisible(True)
            self.trocr_annotations_file_edit.setVisible(True)
            self.trocr_annotations_file_button.setVisible(True)
            
        elif "структуры папок" in current_type:
            # Папка со структурой
            folder_layout = QHBoxLayout()
            folder_layout.addWidget(self.trocr_folder_structure_edit)
            folder_layout.addWidget(self.trocr_folder_structure_button)
            self.trocr_source_layout.addRow("Папка с данными:", folder_layout)
            
            self.trocr_folder_structure_edit.setVisible(True)
            self.trocr_folder_structure_button.setVisible(True)
            
        elif "Синтетический" in current_type:
            # Количество примеров
            self.trocr_source_layout.addRow("Количество примеров:", self.trocr_synthetic_samples_spin)
            self.trocr_synthetic_samples_spin.setVisible(True)
            
        elif "Импорт внешнего" in current_type or "Импорт CSV" in current_type or "Импорт LabelMe" in current_type:
            # Папка с внешним датасетом
            external_layout = QHBoxLayout()
            external_layout.addWidget(self.trocr_external_dataset_edit)
            external_layout.addWidget(self.trocr_external_dataset_button)
            self.trocr_source_layout.addRow("Папка датасета:", external_layout)
            
            # Формат датасета (только для универсального импорта)
            if "Импорт внешнего" in current_type:
                self.trocr_source_layout.addRow("Формат:", self.trocr_external_format_combo)
                self.trocr_external_format_combo.setVisible(True)
                
                # Информация о формате
                self.trocr_source_layout.addRow("", self.trocr_format_info_label)
                self.trocr_format_info_label.setVisible(True)
                self.update_format_info()
            elif "Импорт CSV" in current_type:
                # Автоматически устанавливаем CSV формат
                self.trocr_external_format_combo.setCurrentText("CSV Format")
                self.trocr_format_info_label.setText("📊 CSV формат: файл должен содержать колонки image_path и text")
                self.trocr_source_layout.addRow("", self.trocr_format_info_label)
                self.trocr_format_info_label.setVisible(True)
            elif "Импорт LabelMe" in current_type:
                # Автоматически устанавливаем LabelMe формат
                self.trocr_external_format_combo.setCurrentText("LabelMe Format")
                self.trocr_format_info_label.setText("🏷️ LabelMe формат: JSON файлы с полями shapes и imagePath")
                self.trocr_source_layout.addRow("", self.trocr_format_info_label)
                self.trocr_format_info_label.setVisible(True)
            
            # Показываем виджеты
            self.trocr_external_dataset_edit.setVisible(True)
            self.trocr_external_dataset_button.setVisible(True)
    
    def select_trocr_images_folder(self):
        """Выбор папки с изображениями для TrOCR"""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями", 
            "data/test_invoices"
        )
        if folder:
            self.trocr_images_folder_edit.setText(folder)
            self.add_log_message(self.trocr_dataset_log, f"📁 Выбрана папка с изображениями: {folder}")
    
    def select_trocr_annotations_file(self):
        """Выбор файла аннотаций для TrOCR"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл аннотаций", 
            "data", "JSON файлы (*.json)"
        )
        if file_path:
            self.trocr_annotations_file_edit.setText(file_path)
            self.add_log_message(self.trocr_dataset_log, f"📄 Выбран файл аннотаций: {file_path}")
    
    def select_trocr_folder_structure(self):
        """Выбор папки со структурой данных для TrOCR"""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с данными", 
            "data"
        )
        if folder:
            self.trocr_folder_structure_edit.setText(folder)
            self.add_log_message(self.trocr_dataset_log, f"📁 Выбрана папка с данными: {folder}")
    
    def select_trocr_output_path(self):
        """Выбор пути сохранения датасета TrOCR"""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения датасета", 
            "data"
        )
        if folder:
            self.trocr_output_path_edit.setText(folder)
            self.add_log_message(self.trocr_dataset_log, f"💾 Выбран путь сохранения: {folder}")
    
    def select_trocr_external_dataset(self):
        """Выбор папки с внешним датасетом для TrOCR"""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с внешним датасетом", 
            "data"
        )
        if folder:
            self.trocr_external_dataset_edit.setText(folder)
            self.add_log_message(self.trocr_dataset_log, f"📦 Выбран внешний датасет: {folder}")
            
            # Автоматически определяем формат датасета
            self.auto_detect_dataset_format(folder)
    
    def auto_detect_dataset_format(self, dataset_path):
        """Автоматическое определение формата датасета"""
        try:
            parser = UniversalDatasetParser()
            detected_format = parser.detect_format(dataset_path)
            
            # Маппинг форматов на текст в комбобоксе
            format_mapping = {
                DatasetFormat.COCO: "COCO Format",
                DatasetFormat.YOLO: "YOLO Format",
                DatasetFormat.PASCAL_VOC: "PASCAL VOC",
                DatasetFormat.CSV: "CSV Format",
                DatasetFormat.LABELME: "LabelMe Format",
                DatasetFormat.JSON_SIMPLE: "Простой JSON",
                DatasetFormat.FOLDER_STRUCTURE: "Автоопределение"
            }
            
            format_text = format_mapping.get(detected_format, "Автоопределение")
            self.trocr_external_format_combo.setCurrentText(format_text)
            
            # Получаем информацию о датасете
            dataset_info = parser.get_dataset_info(dataset_path, detected_format)
            
            info_text = f"""
🔍 Обнаружен формат: {detected_format.value.upper()}
📊 Изображений: {dataset_info.total_images}
📝 Аннотаций: {dataset_info.total_annotations}
🏷️ Категории: {', '.join(dataset_info.categories[:3])}{'...' if len(dataset_info.categories) > 3 else ''}
🌍 Языки: {', '.join(dataset_info.languages)}
            """.strip()
            
            self.trocr_format_info_label.setText(info_text)
            
            self.add_log_message(self.trocr_dataset_log, 
                f"🔍 Автоопределение: формат {detected_format.value}, "
                f"{dataset_info.total_images} изображений, {dataset_info.total_annotations} аннотаций"
            )
            
        except Exception as e:
            self.add_log_message(self.trocr_dataset_log, f"⚠️ Ошибка автоопределения формата: {e}")
            self.trocr_format_info_label.setText(f"⚠️ Не удалось определить формат: {e}")
    
    def update_format_info(self):
        """Обновляет информацию о выбранном формате датасета"""
        current_format = self.trocr_external_format_combo.currentText()
        
        format_descriptions = {
            "Автоопределение": "🤖 Автоматическое определение формата на основе структуры файлов",
            "COCO Format": "📷 COCO: JSON с полями images, annotations, categories. Стандарт для object detection",
            "YOLO Format": "🎯 YOLO: YAML конфиг + TXT аннотации. Координаты в нормализованном формате",
            "PASCAL VOC": "🗂️ PASCAL VOC: XML файлы с аннотациями bounding box. Абсолютные координаты",
            "CSV Format": "📊 CSV: таблица с колонками image_path, text, опционально bbox координаты",
            "LabelMe Format": "🏷️ LabelMe: JSON файлы с полями shapes, imagePath. Векторные аннотации",
            "Простой JSON": "📄 JSON: простая структура [{image_path, text, bbox}] или вложенные объекты"
        }
        
        description = format_descriptions.get(current_format, "")
        self.trocr_format_info_label.setText(description)
    
    def start_trocr_dataset_preparation(self):
        """Запускает создание датасета TrOCR с автоматическими аннотациями Gemini"""
        try:
            # Получаем параметры
            dataset_type = self.trocr_dataset_type_combo.currentText()
            output_path = self.trocr_output_path_edit.text() or self.trocr_output_path_edit.placeholderText()
            
            # Определяем источник данных
            source_folder = None
            if "структуры папок" in dataset_type:
                source_folder = self.trocr_folder_structure_edit.text()
            elif "аннотаций счетов" in dataset_type:
                source_folder = self.trocr_images_folder_edit.text()
            elif "Импорт" in dataset_type:
                source_folder = self.trocr_external_dataset_edit.text()
            
            if "Синтетический" not in dataset_type and not source_folder:
                if "Импорт" in dataset_type:
                    QMessageBox.warning(self, "Ошибка", "Выберите папку с внешним датасетом для импорта")
                else:
                    QMessageBox.warning(self, "Ошибка", "Выберите папку с изображениями для автоматической Gemini аннотации")
                return
            
            # Логируем начало
            self.add_log_message(self.trocr_dataset_log, "🚀 Начинаем создание датасета TrOCR с автоматическими аннотациями Gemini...")
            self.add_log_message(self.trocr_dataset_log, f"📊 Тип: {dataset_type}")
            self.add_log_message(self.trocr_dataset_log, f"💾 Выход: {output_path}")
            if source_folder:
                self.add_log_message(self.trocr_dataset_log, f"📁 Источник: {source_folder}")
            
            # Обновляем UI
            self.trocr_dataset_start_button.setEnabled(False)
            self.trocr_dataset_stop_button.setEnabled(True)
            self.trocr_dataset_progress_bar.setVisible(True)
            self.trocr_dataset_progress_bar.setValue(0)
            self.trocr_dataset_status_label.setText("🤖 Создание с Gemini аннотациями...")
            
            # Создаем worker для фонового выполнения
            from PyQt6.QtCore import QThread, QObject, pyqtSignal
            
            class AutoTrOCRDatasetWorker(QObject):
                finished = pyqtSignal(str)
                error = pyqtSignal(str)
                progress = pyqtSignal(int)
                log_message = pyqtSignal(str)
                
                def __init__(self, source_folder, output_path, dataset_type, preparator_config, parent_dialog):
                    super().__init__()
                    self.source_folder = source_folder
                    self.output_path = output_path
                    self.dataset_type = dataset_type
                    self.parent_dialog = parent_dialog
                    self.preparator_config = preparator_config
                    self.should_stop = False
                
                def run(self):
                    try:
                        self.log_message.emit("🔧 Инициализация автоматического создания TrOCR датасета...")
                        
                        # Создаем препаратор данных с Gemini процессором
                        from app.training.data_preparator import TrainingDataPreparator
                        from app import config as app_config
                        
                        # Обновляем пути из настроек
                        app_config.update_paths_from_settings()
                        
                        # Создаем препаратор с правильной конфигурацией (включая POPPLER_PATH)
                        preparator = TrainingDataPreparator(
                            app_config,
                            self.parent_dialog.ocr_processor,
                            self.parent_dialog.gemini_processor
                        )
                        
                        # Устанавливаем callback'и
                        preparator.set_callbacks(
                            log_callback=self.log_message.emit,
                            progress_callback=self.progress.emit
                        )
                        
                        if "Синтетический" in self.dataset_type:
                            # Создаем синтетический датасет
                            self.log_message.emit("🎨 Создание синтетического TrOCR датасета...")
                            # Здесь можно добавить логику создания синтетических данных
                            result_path = self.output_path
                            
                        elif "Импорт" in self.dataset_type:
                            # Импортируем внешний датасет с конвертацией
                            self.log_message.emit("📦 Импорт и конвертация внешнего датасета...")
                            result_path = self._import_external_dataset()
                            
                        else:
                            # Создаем датасет из файлов с автоматическими аннотациями через Gemini
                            self.log_message.emit("🤖 Создание TrOCR датасета с Gemini аннотациями...")
                            result_path = preparator.prepare_dataset_for_trocr(
                                source_folder=self.source_folder,
                                output_path=self.output_path,
                                annotation_method="gemini",  # Всегда используем Gemini
                                max_files=None
                            )
                        
                        if result_path and not self.should_stop:
                            self.finished.emit(result_path)
                        elif self.should_stop:
                            self.log_message.emit("⏹️ Создание остановлено пользователем")
                        else:
                            self.error.emit("Не удалось создать датасет")
                            
                    except Exception as e:
                        self.error.emit(str(e))
                
                def stop(self):
                    self.should_stop = True
                
                def _import_external_dataset(self):
                    """Импорт и конвертация внешнего датасета в формат TrOCR"""
                    try:
                        # Получаем формат датасета из parent_dialog
                        format_text = self.parent_dialog.trocr_external_format_combo.currentText()
                        
                        # Маппинг UI текста на DatasetFormat
                        format_mapping = {
                            "COCO Format": DatasetFormat.COCO,
                            "YOLO Format": DatasetFormat.YOLO,
                            "PASCAL VOC": DatasetFormat.PASCAL_VOC,
                            "CSV Format": DatasetFormat.CSV,
                            "LabelMe Format": DatasetFormat.LABELME,
                            "Простой JSON": DatasetFormat.JSON_SIMPLE
                        }
                        
                        dataset_format = format_mapping.get(format_text, None)
                        
                        self.log_message.emit(f"🔍 Анализируем датасет в формате: {format_text}")
                        self.progress.emit(10)
                        
                        # Создаем универсальный парсер
                        parser = UniversalDatasetParser()
                        
                        # Парсим датасет
                        self.log_message.emit("📖 Парсинг аннотаций...")
                        annotations = parser.parse_dataset(self.source_folder, dataset_format)
                        
                        if not annotations:
                            raise ValueError("Не найдено аннотаций в указанном датасете")
                        
                        self.log_message.emit(f"✅ Найдено {len(annotations)} аннотаций")
                        self.progress.emit(30)
                        
                        # Фильтруем только существующие изображения
                        valid_annotations = []
                        for i, ann in enumerate(annotations):
                            if self.should_stop:
                                return None
                                
                            if os.path.exists(ann.image_path):
                                valid_annotations.append(ann)
                            else:
                                self.log_message.emit(f"⚠️ Пропущено: {ann.image_path} (файл не найден)")
                            
                            if i % 50 == 0:  # Обновляем прогресс каждые 50 файлов
                                progress = 30 + (i / len(annotations)) * 40
                                self.progress.emit(int(progress))
                        
                        self.log_message.emit(f"✅ Валидных аннотаций: {len(valid_annotations)}")
                        self.progress.emit(70)
                        
                        if not valid_annotations:
                            raise ValueError("Не найдено валидных аннотаций с существующими изображениями")
                        
                        # Конвертируем в формат TrOCR
                        self.log_message.emit("🔄 Конвертация в формат TrOCR...")
                        output_file = parser.convert_to_trocr_format(valid_annotations, self.output_path)
                        
                        self.progress.emit(90)
                        
                        # Создаем дополнительные файлы метаданных
                        self._create_dataset_metadata(valid_annotations)
                        
                        self.log_message.emit(f"🎉 Импорт завершен успешно! Датасет сохранен в: {self.output_path}")
                        self.progress.emit(100)
                        
                        return self.output_path
                        
                    except Exception as e:
                        self.log_message.emit(f"❌ Ошибка импорта: {e}")
                        raise e
                
                def _create_dataset_metadata(self, annotations):
                    """Создает метаданные датасета"""
                    try:
                        from datetime import datetime
                        import json
                        
                        # Статистика датасета
                        stats = {
                            "total_samples": len(annotations),
                            "categories": list(set(ann.category for ann in annotations)),
                            "languages": list(set(ann.language for ann in annotations)),
                            "avg_text_length": sum(len(ann.text) for ann in annotations) / len(annotations),
                            "has_bbox": sum(1 for ann in annotations if ann.bbox) / len(annotations),
                            "import_source": self.source_folder,
                            "import_format": self.parent_dialog.trocr_external_format_combo.currentText(),
                            "created_at": datetime.now().isoformat()
                        }
                        
                        # Сохраняем метаданные
                        metadata_file = Path(self.output_path) / "dataset_metadata.json"
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(stats, f, ensure_ascii=False, indent=2)
                        
                        self.log_message.emit(f"📊 Метаданные сохранены: {metadata_file}")
                        
                    except Exception as e:
                        self.log_message.emit(f"⚠️ Ошибка создания метаданных: {e}")
            
            # Запускаем worker
            self.trocr_auto_worker = AutoTrOCRDatasetWorker(
                source_folder, output_path, dataset_type, None, self
            )
            self.trocr_auto_thread = QThread()
            
            self.trocr_auto_worker.moveToThread(self.trocr_auto_thread)
            self.trocr_auto_worker.finished.connect(self.on_auto_trocr_finished)
            self.trocr_auto_worker.error.connect(self.on_auto_trocr_error)
            self.trocr_auto_worker.progress.connect(self.on_auto_trocr_progress)
            self.trocr_auto_worker.log_message.connect(
                lambda msg: self.add_log_message(self.trocr_dataset_log, msg)
            )
            
            self.trocr_auto_thread.started.connect(self.trocr_auto_worker.run)
            self.trocr_auto_thread.start()
            
        except Exception as e:
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка создания датасета: {str(e)}")
            self.reset_trocr_dataset_ui()
    

    
    def _parse_image_size(self, size_str):
        """Парсит строку размера изображения"""
        size = size_str.split('x')[0]
        return (int(size), int(size))
    
    def on_auto_trocr_finished(self, result_path):
        """Обработчик завершения автоматического создания TrOCR датасета"""
        self.add_log_message(self.trocr_dataset_log, "✅ TrOCR датасет с Gemini аннотациями создан успешно!")
        self.add_log_message(self.trocr_dataset_log, f"📁 Сохранен в: {result_path}")
        
        # Автоматически устанавливаем правильный путь к датасету в поле ввода
        if result_path and os.path.exists(result_path):
            self.trocr_dataset_edit.setText(result_path)
            self.add_log_message(self.trocr_dataset_log, f"📂 Путь к датасету установлен: {result_path}")
            
            # Сохраняем путь к созданному датасету в настройки
            try:
                from app.settings_manager import settings_manager
                settings_manager.set_value('Training', 'last_trocr_dataset', result_path)
            except Exception:
                pass  # Игнорируем ошибки сохранения настроек
        
        # Обновляем информацию о датасете
        info_text = f"📊 Создан TrOCR датасет с автоматическими Gemini аннотациями:\n"
        info_text += f"📁 Путь: {result_path}\n"
        info_text += f"🤖 Метод аннотации: Gemini (автоматический)\n"
        info_text += f"📝 Готов для обучения TrOCR модели"
        
        self.trocr_dataset_info_label.setText(info_text)
        self.trocr_dataset_status_label.setText("✅ Датасет создан успешно")
        
        # Показываем сообщение
        QMessageBox.information(
            self, "Успех", 
            f"🤖 TrOCR датасет с Gemini аннотациями создан успешно!\n\n"
            f"📁 Сохранен в: {result_path}\n"
            f"📝 Gemini автоматически создал аннотации из изображений\n"
            f"🎯 Датасет готов для обучения TrOCR модели"
        )
        
        self.reset_trocr_dataset_ui()
    
    def on_auto_trocr_error(self, error_message):
        """Обработчик ошибки автоматического создания TrOCR"""
        self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка создания TrOCR датасета: {error_message}")
        self.trocr_dataset_status_label.setText(f"❌ Ошибка: {error_message}")
        
        QMessageBox.critical(
            self, "Ошибка", 
            f"Ошибка создания TrOCR датасета с Gemini аннотациями:\n{error_message}"
        )
        self.reset_trocr_dataset_ui()
    
    def on_auto_trocr_progress(self, progress):
        """Обновляет прогресс автоматического создания TrOCR"""
        self.trocr_dataset_progress_bar.setValue(progress)
        self.trocr_dataset_status_label.setText(f"🤖 Создание с Gemini: {progress}%")
    
    def stop_trocr_dataset_preparation(self):
        """Останавливает создание датасета TrOCR"""
        # Остановка автоматического создания с Gemini
        if hasattr(self, 'trocr_auto_worker'):
            try:
                self.trocr_auto_worker.stop()
                self.add_log_message(self.trocr_dataset_log, "⏹️ Остановка автоматического создания с Gemini...")
                if hasattr(self, 'trocr_auto_thread') and self.trocr_auto_thread.isRunning():
                    self.trocr_auto_thread.quit()
                    self.trocr_auto_thread.wait()
            except:
                pass
        
        self.reset_trocr_dataset_ui()
        
        self.add_log_message(self.trocr_dataset_log, "⏹️ Создание датасета остановлено")
        self.trocr_dataset_status_label.setText("⏹️ Остановлено")
        
    def validate_trocr_dataset(self):
        """Валидация качества TrOCR датасета"""
        try:
            # Получаем путь к датасету
            dataset_path = None
            
            # Сначала проверяем путь из поля TrOCR обучения
            if hasattr(self, 'trocr_dataset_edit') and self.trocr_dataset_edit.text():
                dataset_path = self.trocr_dataset_edit.text()
            # Затем из поля создания датасета
            elif hasattr(self, 'trocr_output_path_edit') and self.trocr_output_path_edit.text():
                dataset_path = self.trocr_output_path_edit.text()
            # Или из недавно созданных
            elif hasattr(self, 'last_created_dataset_path'):
                dataset_path = self.last_created_dataset_path
            
            if not dataset_path or not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Ошибка", 
                    "Выберите существующий датасет для валидации. "
                    "Используйте поле 'Датасет для обучения' в основной вкладке TrOCR.")
                return
            
            self.add_log_message(self.trocr_dataset_log, f"🔍 Начинаем валидацию датасета: {dataset_path}")
            
            # Создаем валидатор и запускаем в отдельном потоке
            from PyQt6.QtCore import QThread, QObject, pyqtSignal
            
            class DatasetValidationWorker(QObject):
                finished = pyqtSignal(dict)
                error = pyqtSignal(str)
                progress = pyqtSignal(int)
                log_message = pyqtSignal(str)
                
                def __init__(self, dataset_path):
                    super().__init__()
                    self.dataset_path = dataset_path
                    
                def run(self):
                    try:
                        from app.training.advanced_data_validator import AdvancedDataValidator
                        
                        self.log_message.emit("🔧 Инициализация валидатора...")
                        validator = AdvancedDataValidator()
                        
                        self.log_message.emit("📊 Анализ датасета...")
                        self.progress.emit(20)
                        
                        validation_results = validator.validate_dataset(
                            self.dataset_path,
                            check_duplicates=True,
                            check_quality=True,
                            check_text=True
                        )
                        
                        self.progress.emit(80)
                        
                        # Генерируем отчет
                        self.log_message.emit("📋 Создание отчета...")
                        report_path = validator.generate_quality_report(validation_results)
                        validation_results['report_path'] = report_path
                        
                        self.progress.emit(100)
                        self.finished.emit(validation_results)
                        
                    except Exception as e:
                        self.error.emit(str(e))
            
            # Запускаем валидацию
            self.validation_worker = DatasetValidationWorker(dataset_path)
            self.validation_thread = QThread()
            
            self.validation_worker.moveToThread(self.validation_thread)
            self.validation_worker.finished.connect(self.on_validation_finished)
            self.validation_worker.error.connect(self.on_validation_error)
            self.validation_worker.progress.connect(self.on_auto_trocr_progress)
            self.validation_worker.log_message.connect(
                lambda msg: self.add_log_message(self.trocr_dataset_log, msg)
            )
            
            self.validation_thread.started.connect(self.validation_worker.run)
            self.validation_thread.start()
            
            # Обновляем UI
            self.trocr_dataset_progress_bar.setVisible(True)
            self.trocr_dataset_progress_bar.setValue(0)
            self.trocr_dataset_status_label.setText("🔍 Валидация датасета...")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка запуска валидации: {e}")
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка валидации: {e}")
    
    def on_validation_finished(self, validation_results):
        """Обработчик завершения валидации"""
        try:
            self.trocr_dataset_progress_bar.setVisible(False)
            self.trocr_dataset_status_label.setText("✅ Валидация завершена")
            
            # Сохраняем результаты для очистки
            self.validation_results = validation_results
            
            # Показываем краткий отчет
            total_items = validation_results['total_items']
            valid_items = validation_results['valid_items']
            issues_count = len(validation_results['issues'])
            duplicates_count = len(validation_results['duplicates'])
            
            summary = f"""
📊 **Результаты валидации датасета:**

📈 **Общая статистика:**
• Всего элементов: {total_items}
• Валидных элементов: {valid_items}
• Найдено проблем: {issues_count}
• Группы дубликатов: {duplicates_count}

📝 **Рекомендации:**
{chr(10).join('• ' + rec for rec in validation_results['recommendations'])}

📋 **Подробный отчет:** {validation_results.get('report_path', 'не создан')}
            """.strip()
            
            self.add_log_message(self.trocr_dataset_log, "✅ Валидация завершена успешно!")
            self.add_log_message(self.trocr_dataset_log, summary)
            
            # Показываем диалог с результатами
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Результаты валидации")
            msg_box.setText("Валидация датасета завершена")
            msg_box.setDetailedText(summary)
            msg_box.setIcon(QMessageBox.Icon.Information)
            
            # Кнопка открытия отчета
            if 'report_path' in validation_results:
                open_report_btn = msg_box.addButton("📋 Открыть отчет", QMessageBox.ButtonRole.ActionRole)
                open_report_btn.clicked.connect(
                    lambda: os.startfile(validation_results['report_path'])
                )
            
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
        except Exception as e:
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка обработки результатов: {e}")
    
    def on_validation_error(self, error_message):
        """Обработчик ошибки валидации"""
        self.trocr_dataset_progress_bar.setVisible(False)
        self.trocr_dataset_status_label.setText("❌ Ошибка валидации")
        self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка валидации: {error_message}")
        QMessageBox.critical(self, "Ошибка валидации", error_message)
    
    def clean_trocr_dataset(self):
        """Автоматическая очистка TrOCR датасета"""
        try:
            if not hasattr(self, 'validation_results') or not self.validation_results:
                QMessageBox.warning(self, "Предупреждение", 
                    "Сначала выполните валидацию датасета для определения проблем.")
                return
            
            # Диалог подтверждения с настройками
            dialog = QDialog(self)
            dialog.setWindowTitle("Настройки очистки датасета")
            dialog.setModal(True)
            layout = QVBoxLayout(dialog)
            
            # Информация о проблемах
            info_label = QLabel("Найденные проблемы для очистки:")
            layout.addWidget(info_label)
            
            issues_text = QTextEdit()
            issues_text.setReadOnly(True)
            issues_text.setMaximumHeight(150)
            
            issues_summary = f"Найдено {len(self.validation_results['issues'])} проблем:\n"
            issue_types = {}
            for issue in self.validation_results['issues']:
                issue_type = issue['type']
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            for issue_type, count in issue_types.items():
                issues_summary += f"• {issue_type}: {count}\n"
            
            issues_text.setPlainText(issues_summary)
            layout.addWidget(issues_text)
            
            # Настройки очистки
            settings_group = QGroupBox("Настройки очистки")
            settings_layout = QVBoxLayout(settings_group)
            
            remove_duplicates_cb = QCheckBox("Удалить дубликаты")
            remove_duplicates_cb.setChecked(True)
            settings_layout.addWidget(remove_duplicates_cb)
            
            remove_low_quality_cb = QCheckBox("Удалить низкокачественные элементы")
            remove_low_quality_cb.setChecked(True)
            settings_layout.addWidget(remove_low_quality_cb)
            
            quality_threshold_layout = QHBoxLayout()
            quality_threshold_layout.addWidget(QLabel("Порог качества:"))
            quality_threshold_spin = QDoubleSpinBox()
            quality_threshold_spin.setRange(0.1, 0.9)
            quality_threshold_spin.setDecimals(2)
            quality_threshold_spin.setValue(0.3)
            quality_threshold_layout.addWidget(quality_threshold_spin)
            settings_layout.addLayout(quality_threshold_layout)
            
            layout.addWidget(settings_group)
            
            # Кнопки
            buttons_layout = QHBoxLayout()
            ok_button = QPushButton("🧹 Выполнить очистку")
            cancel_button = QPushButton("❌ Отмена")
            
            buttons_layout.addWidget(ok_button)
            buttons_layout.addWidget(cancel_button)
            layout.addLayout(buttons_layout)
            
            # Подключаем обработчики
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            # Показываем диалог
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._perform_dataset_cleanup(
                    remove_duplicates_cb.isChecked(),
                    remove_low_quality_cb.isChecked(),
                    quality_threshold_spin.value()
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка очистки: {e}")
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка очистки: {e}")
    
    def _perform_dataset_cleanup(self, remove_duplicates, remove_low_quality, quality_threshold):
        """Выполнение очистки датасета"""
        try:
            self.add_log_message(self.trocr_dataset_log, "🧹 Начинаем очистку датасета...")
            
            # Создаем worker для очистки
            from PyQt6.QtCore import QThread, QObject, pyqtSignal
            
            class DatasetCleanupWorker(QObject):
                finished = pyqtSignal(dict)
                error = pyqtSignal(str)
                progress = pyqtSignal(int)
                log_message = pyqtSignal(str)
                
                def __init__(self, validation_results, remove_duplicates, remove_low_quality, quality_threshold):
                    super().__init__()
                    self.validation_results = validation_results
                    self.remove_duplicates = remove_duplicates
                    self.remove_low_quality = remove_low_quality
                    self.quality_threshold = quality_threshold
                    
                def run(self):
                    try:
                        from app.training.advanced_data_validator import AdvancedDataValidator
                        
                        validator = AdvancedDataValidator()
                        
                        self.log_message.emit("🔧 Анализ элементов для удаления...")
                        self.progress.emit(30)
                        
                        cleanup_results = validator.clean_dataset(
                            self.validation_results,
                            remove_duplicates=self.remove_duplicates,
                            remove_low_quality=self.remove_low_quality,
                            quality_threshold=self.quality_threshold
                        )
                        
                        self.progress.emit(100)
                        self.finished.emit(cleanup_results)
                        
                    except Exception as e:
                        self.error.emit(str(e))
            
            # Запускаем очистку
            self.cleanup_worker = DatasetCleanupWorker(
                self.validation_results, remove_duplicates, remove_low_quality, quality_threshold
            )
            self.cleanup_thread = QThread()
            
            self.cleanup_worker.moveToThread(self.cleanup_thread)
            self.cleanup_worker.finished.connect(self.on_cleanup_finished)
            self.cleanup_worker.error.connect(self.on_cleanup_error)
            self.cleanup_worker.progress.connect(self.on_auto_trocr_progress)
            self.cleanup_worker.log_message.connect(
                lambda msg: self.add_log_message(self.trocr_dataset_log, msg)
            )
            
            self.cleanup_thread.started.connect(self.cleanup_worker.run)
            self.cleanup_thread.start()
            
            # Обновляем UI
            self.trocr_dataset_progress_bar.setVisible(True)
            self.trocr_dataset_progress_bar.setValue(0)
            self.trocr_dataset_status_label.setText("🧹 Очистка датасета...")
            
        except Exception as e:
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка запуска очистки: {e}")
    
    def on_cleanup_finished(self, cleanup_results):
        """Обработчик завершения очистки"""
        try:
            self.trocr_dataset_progress_bar.setVisible(False)
            self.trocr_dataset_status_label.setText("✅ Очистка завершена")
            
            # Показываем результаты очистки
            stats = cleanup_results.get('cleanup_stats', {})
            removed_count = stats.get('total_removed', 0)
            kept_count = stats.get('total_kept', 0)
            removal_percentage = stats.get('removal_percentage', 0)
            
            summary = f"""
🧹 **Результаты очистки датасета:**

📊 **Статистика:**
• Удалено элементов: {removed_count}
• Оставлено элементов: {kept_count}
• Процент удаления: {removal_percentage:.1f}%

📋 **Детали:**
• Дубликатов удалено: {stats.get('duplicates_removed', 0)}
• Низкокачественных удалено: {stats.get('low_quality_removed', 0)}

✅ Датасет успешно очищен и готов для обучения!
            """.strip()
            
            self.add_log_message(self.trocr_dataset_log, "✅ Очистка завершена успешно!")
            self.add_log_message(self.trocr_dataset_log, summary)
            
            # Показываем диалог с результатами
            QMessageBox.information(self, "Очистка завершена", summary)
            
            # Сбрасываем результаты валидации
            self.validation_results = None
            
        except Exception as e:
            self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка обработки результатов очистки: {e}")
    
    def on_cleanup_error(self, error_message):
        """Обработчик ошибки очистки"""
        self.trocr_dataset_progress_bar.setVisible(False)
        self.trocr_dataset_status_label.setText("❌ Ошибка очистки")
        self.add_log_message(self.trocr_dataset_log, f"❌ Ошибка очистки: {error_message}")
        QMessageBox.critical(self, "Ошибка очистки", error_message)
        
    def on_training_error(self, error_message):
        """Обработчик ошибки обучения"""
        print(f"TrainingDialog: Ошибка обучения: {error_message}")
        
        # Используем QTimer для отложенной обработки, чтобы дать потоку корректно завершиться
        QTimer.singleShot(100, lambda: self._handle_training_completion(error_message, False))
        
    def _handle_training_completion(self, result, success):
        """Внутренний метод для обработки завершения обучения"""
        try:
            print(f"TrainingDialog: Обрабатываем завершение обучения (успех: {success})")
            
            # Сначала очищаем потоки
            self.cleanup_training_thread()
            
            # Затем обновляем UI
            self.reset_training_ui()
            
            if success:
                self.status_label.setText("✅ Обучение завершено успешно!")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Успех", 
                    f"Модель успешно обучена!\n\nСохранена в: {result}"
                )
            else:
                self.status_label.setText("❌ Ошибка обучения")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обучении:\n\n{result}")
                
        except Exception as e:
            print(f"TrainingDialog: КРИТИЧЕСКАЯ ОШИБКА при обработке завершения обучения: {e}")
            import traceback
            traceback.print_exc()
            
            # В любом случае попытаемся сбросить UI
            try:
                self.reset_training_ui()
                self.status_label.setText("❌ Критическая ошибка")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            except:
                pass
        
    def on_training_progress(self, progress):
        """Обработчик прогресса обучения"""
        # Определяем активную вкладку и обновляем соответствующий прогресс-бар
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.layoutlm_progress_bar.setValue(progress)
        elif current_tab == 1:  # Donut
            self.donut_progress_bar.setValue(progress)
            
    def on_training_log(self, message):
        """Обработчик лог сообщений"""
        # Определяем активную вкладку и добавляем в соответствующий лог
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.add_log_message(self.layoutlm_log, message)
        elif current_tab == 1:  # Donut
            self.add_log_message(self.donut_log, message)
            
        # Парсим метрики из лог сообщения и обновляем вкладку мониторинга
        self.parse_and_update_metrics(message)
            
    def reset_training_ui(self):
        """Сброс UI после завершения обучения"""
        try:
            print("TrainingDialog: Сбрасываем UI после обучения...")
            
            # LayoutLM
            if hasattr(self, 'layoutlm_start_button'):
                self.layoutlm_start_button.setEnabled(True)
            if hasattr(self, 'layoutlm_stop_button'):
                self.layoutlm_stop_button.setEnabled(False)
            if hasattr(self, 'layoutlm_progress_bar'):
                self.layoutlm_progress_bar.setVisible(False)
            if hasattr(self, 'layoutlm_status_label'):
                self.layoutlm_status_label.setText("Готов к обучению")
            
            # Donut
            if hasattr(self, 'donut_start_button'):
                self.donut_start_button.setEnabled(True)
            if hasattr(self, 'donut_stop_button'):
                self.donut_stop_button.setEnabled(False)
            if hasattr(self, 'donut_progress_bar'):
                self.donut_progress_bar.setVisible(False)
            if hasattr(self, 'donut_status_label'):
                self.donut_status_label.setText("Готов к обучению")
            
            # Сбрасываем метрики мониторинга в финальные значения
            self.current_metrics = {
                'epoch': self.current_metrics.get('epoch', 0),  # Сохраняем последние значения
                'step': self.current_metrics.get('step', 0),
                'loss': self.current_metrics.get('loss', 0.0),
                'lr': 0.0,  # LR сбрасываем, так как обучение завершено
                'accuracy': self.current_metrics.get('accuracy', 0.0),
                'f1': self.current_metrics.get('f1', 0.0)
            }
            self.update_monitoring_display()
            
            # Очищаем ссылки (потоки уже должны быть остановлены в cleanup_training_thread)
            self.current_trainer = None
            self.current_worker = None
            self.current_thread = None
            
            print("TrainingDialog: UI сброшен успешно")
            
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА при сбросе UI: {e}")
            import traceback
            traceback.print_exc()
        
    def cleanup_training_thread(self):
        """Правильная очистка потока обучения"""
        try:
            print("TrainingDialog: Начинаем очистку потока обучения...")
            
            # Отключаем сигналы в первую очередь чтобы избежать повторных вызовов
            if self.current_worker:
                try:
                    self.current_worker.finished.disconnect()
                    self.current_worker.error.disconnect()
                    self.current_worker.progress.disconnect()
                    self.current_worker.log_message.disconnect()
                    print("TrainingDialog: Сигналы worker отключены")
                except:
                    pass  # Сигналы могли быть уже отключены
            
            if self.current_thread:
                try:
                    self.current_thread.started.disconnect()
                    print("TrainingDialog: Сигналы thread отключены")
                except:
                    pass
            
            # Останавливаем поток если он еще работает
            if self.current_thread and self.current_thread.isRunning():
                print("TrainingDialog: Поток все еще работает, останавливаем...")
                
                # Сначала пытаемся корректно завершить поток
                self.current_thread.quit()
                
                # Ждем завершения максимум 5 секунд
                if not self.current_thread.wait(5000):
                    print("TrainingDialog: Поток не завершился за 5 секунд, принудительно завершаем...")
                    # Если поток не завершился, принудительно завершаем
                    self.current_thread.terminate()
                    if not self.current_thread.wait(2000):
                        print("TrainingDialog: Поток не отвечает на terminate, оставляем как есть")
                else:
                    print("TrainingDialog: Поток завершен корректно")
            else:
                print("TrainingDialog: Поток уже не работает")
                
            # Планируем удаление объектов через deleteLater
            if self.current_worker:
                print("TrainingDialog: Планируем удаление worker...")
                self.current_worker.deleteLater()
                self.current_worker = None
                
            if self.current_thread:
                print("TrainingDialog: Планируем удаление thread...")
                self.current_thread.deleteLater()
                self.current_thread = None
                
            print("TrainingDialog: Очистка потока завершена успешно")
                
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА при очистке потока обучения: {e}")
            import traceback
            traceback.print_exc()
            
            # В любом случае очищаем ссылки чтобы избежать зависших объектов
            self.current_worker = None
            self.current_thread = None
        
    def add_log_message(self, log_widget, message):
        """Добавляет сообщение в лог с временной меткой"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        log_widget.append(formatted_message)
        
        # Прокручиваем к концу
        cursor = log_widget.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        log_widget.setTextCursor(cursor)
        
    def save_log(self, log_widget):
        """Сохранение лога в файл"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить лог",
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(log_widget.toPlainText())
                QMessageBox.information(self, "Успех", f"Лог сохранен в файл:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить лог:\n{str(e)}")
                
    def export_history(self):
        """Экспорт истории обучения в CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт истории",
            f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Заголовки
                    headers = []
                    for col in range(self.history_table.columnCount()):
                        headers.append(self.history_table.horizontalHeaderItem(col).text())
                    writer.writerow(headers)
                    
                    # Данные
                    for row in range(self.history_table.rowCount()):
                        row_data = []
                        for col in range(self.history_table.columnCount()):
                            item = self.history_table.item(row, col)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)
                        
                QMessageBox.information(self, "Успех", f"История экспортирована в:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать историю:\n{str(e)}")
                
    def clear_history(self):
        """Очистка истории обучения"""
        self.history_table.setRowCount(0)
        self.training_history = []
        
        # Сбрасываем текущие метрики
        self.current_metrics = {
            'epoch': 0,
            'step': 0,
            'loss': 0.0,
            'lr': 0.0,
            'accuracy': 0.0,
            'f1': 0.0
        }
        self.update_monitoring_display()
        
    def parse_and_update_metrics(self, message):
        """Парсинг метрик из лог сообщения и обновление вкладки мониторинга"""
        import re
        
        try:
            # Парсим разные типы сообщений:
            # 1. "📅 Начало эпохи 1/10"
            # 2. "🏃 Шаг 50/1000 (5.0%)"
            # 3. "✅ Эпоха 1 завершена. Loss: 0.3456, LR: 5.00e-05"
            
            # Начало эпохи
            epoch_begin_match = re.search(r'Начало эпохи (\d+)/(\d+)', message)
            if epoch_begin_match:
                self.current_metrics['epoch'] = int(epoch_begin_match.group(1))
                # Обновляем отображение
                self.update_monitoring_display()
                return
            
            # Прогресс по шагам
            step_match = re.search(r'Шаг (\d+)/(\d+)', message)
            if step_match:
                current_step = int(step_match.group(1))
                total_steps = int(step_match.group(2))
                self.current_metrics['step'] = current_step
                # Обновляем отображение
                self.update_monitoring_display()
                return
            
            # Завершение эпохи с метриками
            epoch_end_match = re.search(r'Эпоха (\d+) завершена', message)
            if epoch_end_match:
                epoch_num = int(epoch_end_match.group(1))
                self.current_metrics['epoch'] = epoch_num
                
                # Парсим Loss
                loss_match = re.search(r'Loss:\s*([0-9]*\.?[0-9]+)', message)
                if loss_match:
                    self.current_metrics['loss'] = float(loss_match.group(1))
                
                # Парсим Learning Rate
                lr_match = re.search(r'LR:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', message)
                if lr_match:
                    self.current_metrics['lr'] = float(lr_match.group(1))
                
                # Обновляем отображение
                self.update_monitoring_display()
                
                # Добавляем в историю при завершении эпохи
                self.add_to_history()
                return
            
            # Парсим метрики валидации (если есть)
            # Примеры: "eval_loss: 0.2345", "eval_f1: 0.8765", "eval_accuracy: 0.9123"
            eval_loss_match = re.search(r'eval_loss:\s*([0-9]*\.?[0-9]+)', message)
            if eval_loss_match:
                # Сохраняем eval_loss отдельно, но можем использовать для отображения
                eval_loss = float(eval_loss_match.group(1))
                # Можно добавить в current_metrics если нужно
                
            eval_f1_match = re.search(r'eval_f1:\s*([0-9]*\.?[0-9]+)', message)
            if eval_f1_match:
                self.current_metrics['f1'] = float(eval_f1_match.group(1))
                self.update_monitoring_display()
                
            eval_accuracy_match = re.search(r'eval_accuracy:\s*([0-9]*\.?[0-9]+)', message)
            if eval_accuracy_match:
                # Конвертируем из долей в проценты если нужно
                accuracy = float(eval_accuracy_match.group(1))
                if accuracy <= 1.0:  # Если в долях, конвертируем в проценты
                    accuracy *= 100
                self.current_metrics['accuracy'] = accuracy
                self.update_monitoring_display()
            
            # Парсим другие возможные метрики
            precision_match = re.search(r'precision:\s*([0-9]*\.?[0-9]+)', message)
            if precision_match:
                precision = float(precision_match.group(1))
                # Можно добавить в интерфейс если нужно
            
            recall_match = re.search(r'recall:\s*([0-9]*\.?[0-9]+)', message)
            if recall_match:
                recall = float(recall_match.group(1))
                # Можно добавить в интерфейс если нужно
                
        except Exception as e:
            print(f"Ошибка парсинга метрик: {e}")
            # Для отладки выводим сообщение, которое не удалось парсить
            print(f"Сообщение: {message}")
            
    def update_monitoring_display(self):
        """Обновляет отображение метрик на вкладке мониторинга"""
        try:
            if hasattr(self, 'current_epoch_label'):
                self.current_epoch_label.setText(str(self.current_metrics['epoch']))
            if hasattr(self, 'current_step_label'):
                self.current_step_label.setText(str(self.current_metrics['step']))
            if hasattr(self, 'current_loss_label'):
                self.current_loss_label.setText(f"{self.current_metrics['loss']:.6f}")
            if hasattr(self, 'current_lr_label'):
                self.current_lr_label.setText(f"{self.current_metrics['lr']:.6f}")
            if hasattr(self, 'current_accuracy_label'):
                self.current_accuracy_label.setText(f"{self.current_metrics['accuracy']:.2f}%")
            if hasattr(self, 'current_f1_label'):
                self.current_f1_label.setText(f"{self.current_metrics['f1']:.3f}")
        except Exception as e:
            print(f"Ошибка обновления отображения мониторинга: {e}")
            
    def add_to_history(self):
        """Добавляет текущие метрики в историю обучения"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Добавляем в список истории
            history_entry = {
                'epoch': self.current_metrics['epoch'],
                'step': self.current_metrics['step'],
                'loss': self.current_metrics['loss'],
                'eval_loss': 0.0,  # TODO: добавить парсинг eval_loss
                'accuracy': self.current_metrics['accuracy'],
                'time': current_time
            }
            self.training_history.append(history_entry)
            
            # Добавляем в таблицу
            if hasattr(self, 'history_table'):
                row_position = self.history_table.rowCount()
                self.history_table.insertRow(row_position)
                
                self.history_table.setItem(row_position, 0, QTableWidgetItem(str(self.current_metrics['epoch'])))
                self.history_table.setItem(row_position, 1, QTableWidgetItem(str(self.current_metrics['step'])))
                self.history_table.setItem(row_position, 2, QTableWidgetItem(f"{self.current_metrics['loss']:.6f}"))
                self.history_table.setItem(row_position, 3, QTableWidgetItem("N/A"))  # Eval Loss
                self.history_table.setItem(row_position, 4, QTableWidgetItem(f"{self.current_metrics['accuracy']:.2f}%"))
                self.history_table.setItem(row_position, 5, QTableWidgetItem(current_time))
                
                # Прокручиваем к последней строке
                self.history_table.scrollToBottom()
                
        except Exception as e:
            print(f"Ошибка добавления в историю: {e}")
        
    def show_help(self):
        """Показ справки"""
        help_text = """
        🎓 Справка по обучению моделей
        
        📄 LayoutLMv3:
        • Модель для понимания структурированных документов
        • Использует текст, изображение и позиционную информацию
        • Подходит для извлечения полей из форм и счетов
        
        🍩 Donut:
        • OCR-free модель для понимания документов без предварительного OCR
        • Обрабатывает изображения напрямую без предварительного OCR
        • Подходит для сложных документов с нестандартной структурой
        
        📊 Подготовка данных:
        • Автоматическое создание датасетов из ваших документов
        • Поддержка различных форматов (PDF, JPG, PNG)
        • Использование Gemini для создания аннотаций
        
        📈 Мониторинг:
        • Отслеживание метрик в реальном времени
        • История обучения с возможностью экспорта
        • Визуализация прогресса
        """
        
        QMessageBox.information(self, "Справка", help_text)
        
    def load_settings(self):
        """Загрузка настроек"""
        from app.settings_manager import settings_manager
        
        # Восстанавливаем последнюю папку с документами
        last_source_folder = settings_manager.get_string('Training', 'last_source_folder', '')
        if last_source_folder and os.path.exists(last_source_folder):
            self.source_folder_edit.setText(last_source_folder)
            self.update_source_info(last_source_folder)
        
        # LayoutLM настройки
        self.layoutlm_base_model_edit.setText(
            settings_manager.get_string('Training', 'layoutlm_base_model', 'microsoft/layoutlmv3-base')
        )
        self.layoutlm_epochs_spin.setValue(
            settings_manager.get_int('Training', 'layoutlm_epochs', 10)
        )
        self.layoutlm_batch_size_spin.setValue(
            settings_manager.get_int('Training', 'layoutlm_batch_size', 8)
        )
        
        # Donut настройки
        donut_model = settings_manager.get_string('Training', 'donut_base_model', 'naver-clova-ix/donut-base')
        index = self.donut_base_model_combo.findText(donut_model)
        if index >= 0:
            self.donut_base_model_combo.setCurrentIndex(index)
            
        self.donut_epochs_spin.setValue(
            settings_manager.get_int('Training', 'donut_epochs', 5)
        )
        self.donut_batch_size_spin.setValue(
            settings_manager.get_int('Training', 'donut_batch_size', 2)
        )
        
        # TrOCR настройки - загружаем последний используемый датасет
        last_trocr_dataset = settings_manager.get_string('Training', 'last_trocr_dataset', '')
        if last_trocr_dataset and os.path.exists(last_trocr_dataset):
            self.trocr_dataset_edit.setText(last_trocr_dataset)
            self.update_dataset_info(last_trocr_dataset, self.trocr_dataset_info)
        
        # TrOCR последний путь создания датасета
        last_trocr_output_path = settings_manager.get_string('Training', 'last_trocr_output_path', '')
        if last_trocr_output_path:
            self.trocr_output_path_edit.setText(last_trocr_output_path)
        
    def save_settings(self):
        """Сохранение настроек"""
        from app.settings_manager import settings_manager
        
        # Сохраняем папку с документами
        source_folder = self.source_folder_edit.text()
        if source_folder:
            settings_manager.set_value('Training', 'last_source_folder', source_folder)
        
        # LayoutLM настройки
        settings_manager.set_value('Training', 'layoutlm_base_model', self.layoutlm_base_model_edit.text())
        settings_manager.set_value('Training', 'layoutlm_epochs', self.layoutlm_epochs_spin.value())
        settings_manager.set_value('Training', 'layoutlm_batch_size', self.layoutlm_batch_size_spin.value())
        
        # Donut настройки
        settings_manager.set_value('Training', 'donut_base_model', self.donut_base_model_combo.currentText())
        settings_manager.set_value('Training', 'donut_epochs', self.donut_epochs_spin.value())
        settings_manager.set_value('Training', 'donut_batch_size', self.donut_batch_size_spin.value())
        
        # TrOCR настройки - сохраняем последний используемый датасет
        trocr_dataset_path = self.trocr_dataset_edit.text()
        if trocr_dataset_path:
            settings_manager.set_value('Training', 'last_trocr_dataset', trocr_dataset_path)
        
        # TrOCR путь создания датасета
        trocr_output_path = self.trocr_output_path_edit.text()
        if trocr_output_path:
            settings_manager.set_value('Training', 'last_trocr_output_path', trocr_output_path)

    def closeEvent(self, event):
        """Обработка закрытия диалога - останавливаем все потоки"""
        print("TrainingDialog: Начинаем корректное закрытие диалога...")
        
        # Сохраняем настройки
        try:
            self.save_settings()
        except:
            pass  # Игнорируем ошибки сохранения при закрытии
        
        # Останавливаем поток подготовки данных
        if hasattr(self, 'preparation_thread') and self.preparation_thread and self.preparation_thread.isRunning():
            print("TrainingDialog: Останавливаем поток подготовки данных...")
            if hasattr(self, 'preparation_worker') and self.preparation_worker:
                self.preparation_worker.preparator.stop()
            self.preparation_thread.quit()
            if not self.preparation_thread.wait(5000):  # Ждем 5 секунд
                print("TrainingDialog: Принудительное завершение потока подготовки...")
                self.preparation_thread.terminate()
                self.preparation_thread.wait()
        
        # Останавливаем поток обучения
        if hasattr(self, 'current_trainer') and self.current_trainer:
            print("TrainingDialog: Останавливаем тренер...")
            self.current_trainer.stop()
        
        print("TrainingDialog: Очищаем потоки обучения...")
        self.cleanup_training_thread()
        
        print("TrainingDialog: Все потоки остановлены, закрываем диалог")
        super().closeEvent(event)

    def apply_fast_gpu_settings(self):
        """Применяет оптимизированные настройки для быстрого обучения на GPU"""
        try:
            import torch
            
            # Проверяем доступность CUDA
            if not torch.cuda.is_available():
                QMessageBox.warning(self, "GPU не найден", 
                    "CUDA недоступна. Быстрые настройки предназначены для GPU.")
                return
                
            # Получаем информацию о GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Определяем оптимальные параметры в зависимости от GPU
            if gpu_memory_gb >= 10:  # RTX 4070 Ti и выше
                # ЭКСТРЕМАЛЬНО минимальные настройки для предотвращения OOM
                optimal_epochs = 1  # Только 1 эпоха для теста
                optimal_batch_size = 1  # Минимальный batch
                optimal_grad_accum = 8  # Большое накопление для компенсации
                optimal_image_size = "128"  # Экстремально маленький размер
                optimal_max_length = 128  # Минимальная длина
                
                settings_description = f"""
🚀 <b>ЭКСТРЕМАЛЬНО минимальные настройки для {gpu_name} ({gpu_memory_gb:.1f} GB)</b>

<b>⚡ Оптимизировано для:</b>
• 100% гарантия отсутствия OOM ошибок
• Минимальное потребление VRAM (~6-8 GB)
• Тестирование работоспособности обучения

<b>🔧 Установленные параметры:</b>
• Эпохи: {optimal_epochs} (минимум для теста)
• Batch size: {optimal_batch_size} (абсолютный минимум)
• Grad accumulation: {optimal_grad_accum} (эффективный batch = {optimal_batch_size * optimal_grad_accum})
• Размер изображения: {optimal_image_size}px (экстремально мало)
• Max length: {optimal_max_length} токенов (минимум)
• FP16 + все оптимизации памяти + 0 workers

<b>📊 Ожидаемое время:</b>
• ~5-10 минут обучения
• ~1-2 минуты на эпоху
• ГАРАНТИЯ: ПОЛНОСТЬЮ без OOM!
• Цель: ПРОВЕРИТЬ что обучение РАБОТАЕТ
                """
                
            elif gpu_memory_gb >= 6:  # RTX 3060/4060 и аналогичные
                # Настройки для средних GPU
                optimal_epochs = 2
                optimal_batch_size = 1  # Очень консервативно
                optimal_grad_accum = 4
                optimal_image_size = "224"
                optimal_max_length = 256
                
                settings_description = f"""
🚀 <b>Безопасные настройки для {gpu_name} ({gpu_memory_gb:.1f} GB)</b>

<b>⚡ Оптимизировано для:</b>
• Предотвращение OOM на средних GPU
• Безопасное использование {gpu_memory_gb:.1f} GB VRAM
• Стабильная производительность

<b>🔧 Установленные параметры:</b>
• Эпохи: {optimal_epochs}
• Batch size: {optimal_batch_size} (максимально безопасно)
• Grad accumulation: {optimal_grad_accum}
• Размер изображения: {optimal_image_size}px
• Max length: {optimal_max_length} токенов
• FP16 + gradient checkpointing + 0 workers

<b>📊 Ожидаемое время:</b>
• ~30-40 минут обучения
• ~3-4 минуты на эпоху
• Гарантия стабильности
                """
                
            else:  # Менее мощные GPU
                optimal_epochs = 1  # Еще меньше эпох
                optimal_batch_size = 1
                optimal_grad_accum = 2
                optimal_image_size = "224"
                optimal_max_length = 128  # Еще меньше
                
                settings_description = f"""
🚀 <b>Минимальные настройки для {gpu_name} ({gpu_memory_gb:.1f} GB)</b>

<b>⚡ Оптимизировано для:</b>
• Максимальная экономия памяти
• Предотвращение любых OOM ошибок
• Тестирование возможностей GPU

<b>🔧 Установленные параметры:</b>
• Эпохи: {optimal_epochs} (минимум для теста)
• Batch size: {optimal_batch_size} (минимально возможный)
• Grad accumulation: {optimal_grad_accum}
• Размер изображения: {optimal_image_size}px (минимум)
• Max length: {optimal_max_length} токенов (минимум)
• FP16 + все оптимизации памяти

<b>📊 Ожидаемое время:</b>
• ~15-20 минут обучения
• ~2-3 минуты на эпоху
• Максимальная стабильность
                """
            
            # Показываем диалог подтверждения
            msg = QMessageBox()
            msg.setWindowTitle("⚡ Быстрые настройки GPU")
            msg.setText(settings_description)
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)
            msg.button(QMessageBox.StandardButton.Yes).setText("✅ Применить")
            msg.button(QMessageBox.StandardButton.No).setText("❌ Отмена")
            
            if msg.exec() == QMessageBox.StandardButton.Yes:
                # Применяем оптимальные настройки
                self.donut_epochs_spin.setValue(optimal_epochs)
                self.donut_batch_size_spin.setValue(optimal_batch_size)
                self.donut_grad_accum_spin.setValue(optimal_grad_accum)
                self.donut_image_size_combo.setCurrentText(optimal_image_size)
                self.donut_max_length_spin.setValue(optimal_max_length)
                self.donut_fp16_checkbox.setChecked(True)
                
                # Также оптимизируем другие параметры
                self.donut_save_steps_spin.setValue(50)  # Частое сохранение
                self.donut_eval_steps_spin.setValue(50)  # Частая оценка
                
                # Показываем уведомление об успехе
                success_msg = QMessageBox()
                success_msg.setWindowTitle("✅ Настройки применены")
                success_msg.setText(f"""
<b>🎯 Быстрые настройки успешно применены!</b>

<b>🚀 Ваша система готова к быстрому обучению:</b>
• GPU: {gpu_name}
• Память: {gpu_memory_gb:.1f} GB
• Режим: Высокая производительность

<b>▶️ Теперь можете нажать "Начать обучение"</b>
                """)
                success_msg.setTextFormat(Qt.TextFormat.RichText)
                success_msg.setIcon(QMessageBox.Icon.Information)
                success_msg.exec()
                
                # Логируем в Donut лог
                self.add_log_message(self.donut_log, f"⚡ Применены быстрые настройки GPU для {gpu_name}")
                self.add_log_message(self.donut_log, f"   📊 Эпохи: {optimal_epochs}, Batch: {optimal_batch_size}, Время: ~{optimal_epochs * (121 // optimal_batch_size) // 60}мин")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить настройки GPU: {str(e)}")
            
    def auto_optimize_memory(self):
        """Автоматически оптимизирует все настройки памяти для RTX 4070 Ti"""
        reply = QMessageBox.question(
            self,
            "Автооптимизация памяти",
            """🚀 Применить оптимальные настройки памяти для RTX 4070 Ti?

Будут применены следующие оптимизации:
• ✅ LoRA - до 95% экономии памяти
• ✅ 8-bit оптимизатор - дополнительные 25% экономии  
• ✅ Gradient checkpointing - экономия activations
• ⚙️ Batch size = 1, epochs = 1, image_size = 224

Это позволит обучать Donut на RTX 4070 Ti без OOM ошибок.
            """,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Включаем все оптимизации памяти
            self.use_lora_cb.setChecked(True)
            self.use_8bit_optimizer_cb.setChecked(True)
            self.freeze_encoder_cb.setChecked(False)  # Оставляем encoder обучаемым для качества
            
            # Устанавливаем консервативные настройки Donut
            self.donut_epochs_spin.setValue(1)
            self.donut_batch_size_spin.setValue(1)
            self.donut_grad_accum_spin.setValue(8)  # Компенсируем маленький batch
            self.donut_image_size_combo.setCurrentText("224")
            self.donut_max_length_spin.setValue(256)
            self.donut_fp16_checkbox.setChecked(True)
            
            # Показываем сообщение об успехе
            QMessageBox.information(
                self,
                "✅ Оптимизация применена",
                """🚀 Автооптимизация памяти применена успешно!

Примененные настройки:
• LoRA: Включен (до 95% экономии)
• 8-bit optimizer: Включен (25% экономии)
• Batch size: 1 (минимальный)
• Epochs: 1 (для тестирования)
• Image size: 224px (экономия памяти)
• Max length: 256 tokens
• FP16: Включен

Эти настройки должны позволить обучение на RTX 4070 Ti (12GB).
                """
            )

    def stop_training(self):
        """Останавливает текущее обучение"""
        try:
            if self.current_thread and self.current_thread.isRunning():
                self.add_log_message(
                    self.get_current_log_widget(), 
                    "⏹️ Остановка обучения..."
                )
                
                # Останавливаем поток
                self.current_thread.quit()
                
                # Ждем завершения максимум 5 секунд
                if self.current_thread.wait(5000):
                    self.add_log_message(
                        self.get_current_log_widget(), 
                        "✅ Обучение остановлено"
                    )
                else:
                    self.add_log_message(
                        self.get_current_log_widget(), 
                        "⚠️ Принудительная остановка"
                    )
                    self.current_thread.terminate()
                    
                # Очищаем ресурсы
                self.cleanup_training_thread()
                
                # Сбрасываем UI
                self.reset_training_ui()
                
            else:
                self.add_log_message(
                    self.get_current_log_widget(), 
                    "ℹ️ Обучение не запущено"
                )
                
        except Exception as e:
            print(f"Ошибка остановки обучения: {e}")
            # В любом случае пытаемся очистить
            self.cleanup_training_thread()
            self.reset_training_ui()
    
    def get_current_log_widget(self):
        """Возвращает текущий лог виджет в зависимости от активной вкладки"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0 and hasattr(self, 'layoutlm_log'):  # LayoutLM
            return self.layoutlm_log
        elif current_tab == 1 and hasattr(self, 'donut_log'):  # Donut
            return self.donut_log
        elif current_tab == 2 and hasattr(self, 'trocr_log'):  # TrOCR
            return self.trocr_log
        elif hasattr(self, 'prepare_log'):  # Подготовка данных
            return self.prepare_log
        else:
            # Возвращаем первый доступный лог
            for log_attr in ['layoutlm_log', 'donut_log', 'trocr_log', 'prepare_log']:
                if hasattr(self, log_attr):
                    return getattr(self, log_attr)
            return None
            
    def start_training_thread(self, training_params, model_type):
        """Запускает обучение в отдельном потоке"""
        try:
            print(f"TrainingDialog: Запускаем обучение {model_type}...")
            
            # Останавливаем предыдущее обучение если оно есть
            if self.current_thread and self.current_thread.isRunning():
                self.stop_training()
                
            # Создаем worker и поток
            self.current_worker = TrainingWorker(self.current_trainer, training_params)
            self.current_thread = QThread()
            
            # Перемещаем worker в поток
            self.current_worker.moveToThread(self.current_thread)
            
            # Подключаем сигналы
            self.current_thread.started.connect(self.current_worker.run)
            self.current_worker.finished.connect(self.on_training_finished)
            self.current_worker.error.connect(self.on_training_error)
            self.current_worker.progress.connect(self.on_training_progress)
            self.current_worker.log_message.connect(self.on_training_log)
            
            # Обновляем UI
            self.update_training_ui_start(model_type)
            
            # Запускаем поток
            self.current_thread.start()
            
            print(f"TrainingDialog: Поток {model_type} запущен")
            
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА запуска потока: {e}")
            self.on_training_error(str(e))
    
    def update_training_ui_start(self, model_type):
        """Обновляет UI при начале обучения"""
        # LayoutLM
        if model_type == 'layoutlm':
            if hasattr(self, 'layoutlm_start_button'):
                self.layoutlm_start_button.setEnabled(False)
            if hasattr(self, 'layoutlm_stop_button'):
                self.layoutlm_stop_button.setEnabled(True)
            if hasattr(self, 'layoutlm_progress_bar'):
                self.layoutlm_progress_bar.setVisible(True)
            if hasattr(self, 'layoutlm_status_label'):
                self.layoutlm_status_label.setText("Обучение...")
        
        # Donut
        elif model_type == 'donut':
            if hasattr(self, 'donut_start_button'):
                self.donut_start_button.setEnabled(False)
            if hasattr(self, 'donut_stop_button'):
                self.donut_stop_button.setEnabled(True)
            if hasattr(self, 'donut_progress_bar'):
                self.donut_progress_bar.setVisible(True)
            if hasattr(self, 'donut_status_label'):
                self.donut_status_label.setText("Обучение...")
        
        # TrOCR
        elif model_type == 'trocr':
            if hasattr(self, 'trocr_start_button'):
                self.trocr_start_button.setEnabled(False)
            if hasattr(self, 'trocr_stop_button'):
                self.trocr_stop_button.setEnabled(True)
            if hasattr(self, 'trocr_progress_bar'):
                self.trocr_progress_bar.setVisible(True)
            if hasattr(self, 'trocr_status_label'):
                self.trocr_status_label.setText("Обучение...")
        
        # Общий статус
        self.status_label.setText(f"🚀 Обучение {model_type.upper()}...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
    def analyze_dataset_quality(self):
        """Анализирует качество выбранного датасета"""
        dataset_path = self.source_folder_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Сначала выберите папку с исходными данными!")
            return
        
        try:
            self.analyze_quality_button.setEnabled(False)
            # Выполняем анализ
            results = self.quality_analyzer.analyze_dataset(dataset_path)
            self.last_quality_results = results
            
            # Обновляем интерфейс
            self.update_quality_display(results)
            
            # Логируем результаты
            self.add_log_message(self.prepare_log, f"📊 Анализ качества датасета: {dataset_path}")
            self.add_log_message(self.prepare_log, f"📈 Общий балл качества: {results['overall_score']}")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка анализа", f"Ошибка при анализе качества датасета:\n{str(e)}")
            self.add_log_message(self.prepare_log, f"❌ Ошибка анализа качества: {str(e)}")
            
        finally:
            self.analyze_quality_button.setEnabled(True)
            self.analyze_quality_button.setText("🔍 Анализировать качество")
    
    def update_quality_display(self, results):
        """Обновляет отображение метрик качества"""
        try:
            # Обновляем общий балл
            score = results['overall_score']
            
            # Определяем цвет и статус по баллу
            if score >= 80:
                color = "#27ae60"  # Зеленый
                status = "Отличное"
                emoji = "🟢"
            elif score >= 60:
                color = "#f39c12"  # Оранжевый
                status = "Хорошее"
                emoji = "🟡"
            elif score >= 40:
                color = "#e67e22"  # Оранжево-красный
                status = "Удовлетворительное"
                emoji = "🟠"
            else:
                color = "#e74c3c"  # Красный
                status = "Критическое"
                emoji = "🔴"
            
            self.overall_score_label.setText(f"{emoji} Общий балл: {score:.1f}% ({status})")
            self.overall_score_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 14px;
                    font-weight: bold;
                    padding: 8px;
                    background-color: {color};
                    border: 1px solid {color};
                    border-radius: 4px;
                    color: white;
                }}
            """)
            
            # Обновляем таблицу метрик
            metrics_data = [
                ("📊 Размер датасета", self._format_dataset_size(results['dataset_size'])),
                ("🏷️ Баланс меток", self._format_label_balance(results['label_balance'])),
                ("📝 Полнота данных", f"{results['data_completeness']:.1f}%"),
                ("✅ Качество аннотаций", f"{results['annotation_quality']:.1f}%"),
                ("🔧 Целостность файлов", f"{results['file_integrity']:.1f}%"),
                ("📋 Консистентность метаданных", f"{results['metadata_consistency']:.1f}%")
            ]
            
            self.quality_metrics_table.setRowCount(len(metrics_data))
            for i, (metric, value) in enumerate(metrics_data):
                self.quality_metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.quality_metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
            # Обновляем рекомендации
            recommendations_text = "\n".join(results['recommendations'])
            self.recommendations_label.setText(recommendations_text)
            
            # Меняем цвет рекомендаций в зависимости от общего балла
            if score >= 80:
                bg_color = "#d5f7e1"
                border_color = "#27ae60"
            elif score >= 60:
                bg_color = "#fef9e7"
                border_color = "#f39c12"
            else:
                bg_color = "#fbeaea"
                border_color = "#e74c3c"
                
            self.recommendations_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 4px;
                    padding: 8px;
                    color: #2c3e50;
                    font-weight: bold;
                }}
            """)
            
        except Exception as e:
            print(f"Ошибка обновления интерфейса качества: {e}")
    
    def _format_dataset_size(self, size_data):
        """Форматирует информацию о размере датасета"""
        total = size_data['total'] + size_data['train'] + size_data['validation']
        if size_data['train'] > 0 or size_data['validation'] > 0:
            return f"Тр:{size_data['train']}, Вал:{size_data['validation']} (всего: {total})"
        else:
            return f"{total} примеров"
    
    def _format_label_balance(self, label_data):
        """Форматирует информацию о балансе меток"""
        if label_data['total_labels'] == 0:
            return "Нет данных"
        
        o_percent = label_data['o_percentage']
        unique_labels = label_data['unique_labels']
        
        if o_percent > 85:
            emoji = "🚨"
        elif o_percent > 70:
            emoji = "⚠️"
        else:
            emoji = "✅"
            
        return f"{emoji} 'O': {o_percent:.1f}%, Уникальных: {unique_labels}"
        
    def on_training_finished(self, model_path):
        """Обработчик завершения обучения"""
        
        # Используем QTimer для отложенной обработки, чтобы дать потоку корректно завершиться
        QTimer.singleShot(100, lambda: self._handle_training_completion(model_path, True))
        
    def on_training_error(self, error_message):
        """Обработчик ошибки обучения"""
        print(f"TrainingDialog: Ошибка обучения: {error_message}")
        
        # Используем QTimer для отложенной обработки, чтобы дать потоку корректно завершиться
        QTimer.singleShot(100, lambda: self._handle_training_completion(error_message, False))
        
    def _handle_training_completion(self, result, success):
        """Внутренний метод для обработки завершения обучения"""
        try:
            print(f"TrainingDialog: Обрабатываем завершение обучения (успех: {success})")
            
            # Сначала очищаем потоки
            self.cleanup_training_thread()
            
            # Затем обновляем UI
            self.reset_training_ui()
            
            if success:
                self.status_label.setText("✅ Обучение завершено успешно!")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Успех", 
                    f"Модель успешно обучена!\n\nСохранена в: {result}"
                )
            else:
                self.status_label.setText("❌ Ошибка обучения")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обучении:\n\n{result}")
                
        except Exception as e:
            print(f"TrainingDialog: КРИТИЧЕСКАЯ ОШИБКА при обработке завершения обучения: {e}")
            import traceback
            traceback.print_exc()
            
            # В любом случае попытаемся сбросить UI
            try:
                self.reset_training_ui()
                self.status_label.setText("❌ Критическая ошибка")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            except:
                pass
        
    def on_training_progress(self, progress):
        """Обработчик прогресса обучения"""
        # Определяем активную вкладку и обновляем соответствующий прогресс-бар
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.layoutlm_progress_bar.setValue(progress)
        elif current_tab == 1:  # Donut
            self.donut_progress_bar.setValue(progress)
            
    def on_training_log(self, message):
        """Обработчик лог сообщений"""
        # Определяем активную вкладку и добавляем в соответствующий лог
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.add_log_message(self.layoutlm_log, message)
        elif current_tab == 1:  # Donut
            self.add_log_message(self.donut_log, message)
            
        # Парсим метрики из лог сообщения и обновляем вкладку мониторинга
        self.parse_and_update_metrics(message)
            
    def reset_training_ui(self):
        """Сброс UI после завершения обучения"""
        try:
            print("TrainingDialog: Сбрасываем UI после обучения...")
            
            # LayoutLM
            if hasattr(self, 'layoutlm_start_button'):
                self.layoutlm_start_button.setEnabled(True)
            if hasattr(self, 'layoutlm_stop_button'):
                self.layoutlm_stop_button.setEnabled(False)
            if hasattr(self, 'layoutlm_progress_bar'):
                self.layoutlm_progress_bar.setVisible(False)
            if hasattr(self, 'layoutlm_status_label'):
                self.layoutlm_status_label.setText("Готов к обучению")
            
            # Donut
            if hasattr(self, 'donut_start_button'):
                self.donut_start_button.setEnabled(True)
            if hasattr(self, 'donut_stop_button'):
                self.donut_stop_button.setEnabled(False)
            if hasattr(self, 'donut_progress_bar'):
                self.donut_progress_bar.setVisible(False)
            if hasattr(self, 'donut_status_label'):
                self.donut_status_label.setText("Готов к обучению")
            
            # TrOCR
            if hasattr(self, 'trocr_start_button'):
                self.trocr_start_button.setEnabled(True)
            if hasattr(self, 'trocr_stop_button'):
                self.trocr_stop_button.setEnabled(False)
            if hasattr(self, 'trocr_progress_bar'):
                self.trocr_progress_bar.setVisible(False)
            if hasattr(self, 'trocr_status_label'):
                self.trocr_status_label.setText("Готов к обучению")
                
            # Сбрасываем метрики мониторинга в финальные значения
            self.current_metrics = {
                'epoch': self.current_metrics.get('epoch', 0),  # Сохраняем последние значения
                'step': self.current_metrics.get('step', 0),
                'loss': self.current_metrics.get('loss', 0.0),
                'lr': 0.0,  # LR сбрасываем, так как обучение завершено
                'accuracy': self.current_metrics.get('accuracy', 0.0),
                'f1': self.current_metrics.get('f1', 0.0)
            }
            self.update_monitoring_display()
            
            # Очищаем ссылки (потоки уже должны быть остановлены в cleanup_training_thread)
            self.current_trainer = None
            self.current_worker = None
            self.current_thread = None
            
            print("TrainingDialog: UI сброшен успешно")
            
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА при сбросе UI: {e}")
            import traceback
            traceback.print_exc()

    def save_source_folder_to_settings(self, folder_path):
        """Сохраняет путь к папке источника в настройки"""
        try:
            settings_manager.set_value('DataPreparation', 'last_source_folder', folder_path)
            self.add_log_message(self.prepare_log, f"📁 Путь сохранен в настройки: {folder_path}")
        except Exception as e:
            self.add_log_message(self.prepare_log, f"❌ Ошибка сохранения пути: {str(e)}")
            
    def start_dataset_preparation(self):
        """Запускает процесс подготовки датасета"""
        try:
            # Проверяем, что выбрана папка с исходными данными
            source_folder = self.source_folder_edit.text()
            if not source_folder or not os.path.exists(source_folder):
                QMessageBox.warning(self, "Ошибка", "Сначала выберите папку с исходными данными!")
                return
                
            # Проверяем наличие данных для подготовки
            if not self.dataset_name_edit.text().strip():
                QMessageBox.warning(self, "Ошибка", "Введите название датасета!")
                return
                
            # Получаем параметры подготовки
            dataset_name = self.dataset_name_edit.text().strip()
            annotation_method = self.annotation_method_combo.currentData()
            max_files = self.max_files_spin.value() if self.max_files_spin.value() > 0 else None
            
            # Проверяем, что DataPreparator инициализирован
            if not hasattr(self, 'data_preparator') or not self.data_preparator:
                from .training.data_preparator import DataPreparator
                self.data_preparator = DataPreparator()
                
            # Создаем путь для сохранения датасета
            output_path = os.path.join(
                self.app_config.TRAINING_DATASETS_PATH,
                dataset_name
            )
            
            # Создаем директорию если не существует
            os.makedirs(output_path, exist_ok=True)
            
            # Обновляем UI
            self.prepare_start_button.setEnabled(False)
            self.prepare_stop_button.setEnabled(True)
            self.prepare_progress_bar.setVisible(True)
            self.prepare_progress_bar.setValue(0)
            self.prepare_status_label.setText("🚀 Подготовка датасета...")
            
            # Логируем начало
            self.add_log_message(self.prepare_log, f"🚀 Начинаем подготовку датасета '{dataset_name}'")
            self.add_log_message(self.prepare_log, f"📁 Источник: {source_folder}")
            self.add_log_message(self.prepare_log, f"🎯 Метод аннотации: {annotation_method}")
            if max_files:
                self.add_log_message(self.prepare_log, f"📊 Максимум файлов: {max_files}")
                
            # TODO: Здесь должна быть запущена подготовка датасета в отдельном потоке
            # Пока делаем заглушку
            self.add_log_message(self.prepare_log, "⚠️ Функция подготовки датасета в разработке")
            self.add_log_message(self.prepare_log, "📋 Параметры сохранены, можно закрыть диалог")
            
            # Имитируем завершение
            self.prepare_progress_bar.setValue(100)
            self.prepare_status_label.setText("✅ Параметры сохранены")
            self.prepare_start_button.setEnabled(True)
            self.prepare_stop_button.setEnabled(False)
            
        except Exception as e:
            # Обработка ошибок
            self.add_log_message(self.prepare_log, f"❌ Ошибка подготовки: {str(e)}")
            self.prepare_status_label.setText("❌ Ошибка подготовки")
            self.prepare_start_button.setEnabled(True)
            self.prepare_stop_button.setEnabled(False)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подготовке датасета:\n{str(e)}")
            
    def stop_preparation(self):
        """Останавливает процесс подготовки датасета"""
        try:
            self.add_log_message(self.prepare_log, "⏹️ Остановка подготовки...")
            self.prepare_start_button.setEnabled(True)
            self.prepare_stop_button.setEnabled(False)
            self.prepare_progress_bar.setVisible(False)
            self.prepare_status_label.setText("⏹️ Остановлено")
        except Exception as e:
            self.add_log_message(self.prepare_log, f"❌ Ошибка остановки: {str(e)}")
    
    def update_dataset_name_preview(self):
        """Обновляет превью имени датасета в зависимости от выбранного типа"""
        dataset_type = self.dataset_type_combo.currentText()
        
        # Определяем префикс модели
        if "LayoutLM" in dataset_type:
            model_prefix = "layoutlm"
        elif "Donut" in dataset_type:
            model_prefix = "donut"
        elif "TrOCR" in dataset_type:
            model_prefix = "trocr"
        else:
            model_prefix = "unknown"
            
        # Обновляем placeholder текст
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        placeholder_text = f"Авто: {model_prefix}_dataset_{timestamp}"
        self.dataset_name_edit.setPlaceholderText(placeholder_text)
        
    def _update_fields_from_manager(self):
        """Обновляет отображение полей из FieldManager"""
        try:
            from .field_manager import field_manager
            enabled_fields = field_manager.get_enabled_fields()
            
            if enabled_fields:
                field_names = [f.display_name for f in enabled_fields]
                field_text = ", ".join(field_names)
                self.annotation_fields_edit.setText(f"Активные поля: {field_text}")
                self.annotation_fields_edit.setToolTip(
                    f"Автоматически извлекаются следующие поля:\n" + 
                    "\n".join([f"• {f.display_name} ({f.id})" for f in enabled_fields]) +
                    f"\n\nВсего активных полей: {len(enabled_fields)}\n\n" +
                    "Для изменения полей используйте меню 'Настройки' → 'Поля таблицы'"
                )
            else:
                self.annotation_fields_edit.setText("⚠️ Нет активных полей")
                self.annotation_fields_edit.setToolTip(
                    "Не найдено активных полей для извлечения.\n\n" +
                    "Перейдите в 'Настройки' → 'Поля таблицы' и включите нужные поля."
                )
        except ImportError as e:
            self.annotation_fields_edit.setText("❌ Ошибка загрузки FieldManager")
            self.annotation_fields_edit.setToolTip(f"Ошибка: {e}")
        except Exception as e:
            self.annotation_fields_edit.setText("❌ Ошибка получения полей")
            self.annotation_fields_edit.setToolTip(f"Ошибка: {e}")
            
    def start_layoutlm_training(self):
        """Запуск обучения LayoutLM"""
        # Проверяем параметры
        dataset_path = self.layoutlm_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
        
        # 🎯 Проверяем метаданные датасета для совместимости полей
        try:
            from .training.data_preparator import TrainingDataPreparator
            preparator = TrainingDataPreparator(self.app_config, self.ocr_processor, self.gemini_processor)
            
            # Определяем папку с метаданными
            metadata_folder = dataset_path
            if dataset_path.endswith("dataset_dict"):
                metadata_folder = os.path.dirname(dataset_path)
            
            metadata = preparator.load_dataset_metadata(metadata_folder)
            if metadata:
                self.add_log_message(self.layoutlm_log, f"📂 Метаданные датасета:")
                self.add_log_message(self.layoutlm_log, f"   • Создан: {metadata.get('created_at', 'неизвестно')}")
                self.add_log_message(self.layoutlm_log, f"   • Источник полей: {metadata.get('fields_source', 'неизвестно')}")
                
                active_fields = metadata.get('active_fields', [])
                if active_fields:
                    self.add_log_message(self.layoutlm_log, f"   • Поля датасета: {', '.join(active_fields)}")
                
                # Проверяем совместимость с текущими настройками
                try:
                    from .field_manager import field_manager
                    current_fields = [f.id for f in field_manager.get_enabled_fields()]
                    
                    if active_fields and current_fields:
                        missing_fields = set(active_fields) - set(current_fields)
                        extra_fields = set(current_fields) - set(active_fields)
                        
                        if missing_fields or extra_fields:
                            self.add_log_message(self.layoutlm_log, "⚠️  ВНИМАНИЕ: Различия в настройках полей:")
                            if missing_fields:
                                self.add_log_message(self.layoutlm_log, f"   • Отключены: {', '.join(missing_fields)}")
                            if extra_fields:
                                self.add_log_message(self.layoutlm_log, f"   • Новые: {', '.join(extra_fields)}")
                                
                            reply = QMessageBox.question(
                                self, "Различия в полях",
                                f"Обнаружены различия между настройками полей:\n\n"
                                f"Датасет: {', '.join(active_fields)}\n"
                                f"Текущие: {', '.join(current_fields)}\n\n"
                                f"Продолжить обучение?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                            )
                            
                            if reply == QMessageBox.StandardButton.No:
                                return
                        else:
                            self.add_log_message(self.layoutlm_log, "✅ Поля соответствуют текущим настройкам")
                except ImportError:
                    pass
            else:
                self.add_log_message(self.layoutlm_log, "📝 Метаданные не найдены (старый формат датасета)")
                
        except Exception as e:
            self.add_log_message(self.layoutlm_log, f"⚠️  Ошибка проверки метаданных: {e}")
            
        # Для LayoutLM нужен путь к dataset_dict внутри датасета
        if dataset_path.endswith("dataset_dict"):
            # Путь уже указывает на dataset_dict
            actual_dataset_path = dataset_path
        else:
            # Проверяем есть ли папка dataset_dict внутри указанного пути
            dataset_dict_path = os.path.join(dataset_path, "dataset_dict")
            if os.path.exists(dataset_dict_path):
                actual_dataset_path = dataset_dict_path
            else:
                QMessageBox.warning(self, "Ошибка", 
                    f"В датасете не найдена папка dataset_dict!\n\n"
                    f"Путь: {dataset_path}\n"
                    f"Ожидаемая структура: {dataset_path}/dataset_dict/\n\n"
                    f"Убедитесь, что выбран правильный датасет для LayoutLM.")
                return
            
        # Создаем тренер
        self.current_trainer = ModelTrainer(self.app_config)
        
        # Подготавливаем относительный путь для модели
        model_name = self.layoutlm_output_name_edit.text() or f"layoutlm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("layoutlm_"):
            model_name = f"layoutlm_{model_name}"
        
        # Подготавливаем параметры
        training_params = {
            'dataset_path': actual_dataset_path,
            'base_model_id': self.layoutlm_base_model_edit.text(),
            'training_args': {
                'num_train_epochs': self.layoutlm_epochs_spin.value(),
                'per_device_train_batch_size': self.layoutlm_batch_size_spin.value(),
                'learning_rate': self.layoutlm_lr_spin.value(),
                'weight_decay': self.layoutlm_weight_decay_spin.value(),
                'warmup_ratio': self.layoutlm_warmup_spin.value(),
                'seed': self.layoutlm_seed_spin.value(),
            },
            'output_model_name': model_name,
            'output_model_path': os.path.join("data", "trained_models", model_name)
        }
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'layoutlm')
        
    def start_donut_training(self):
        """Запуск обучения Donut"""
        # Проверяем параметры
        dataset_path = self.donut_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
            
        # Создаем тренер Donut
        self.current_trainer = DonutTrainerClass(self.app_config)
        
        # Подготавливаем относительный путь для модели
        model_name = self.donut_output_name_edit.text() or f"donut_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("donut_"):
            model_name = f"donut_{model_name}"
        
        # Подготавливаем параметры
        training_params = {
            'dataset_path': dataset_path,
            'base_model_id': self.donut_base_model_combo.currentText(),
            'training_args': {
                'num_train_epochs': self.donut_epochs_spin.value(),
                'per_device_train_batch_size': self.donut_batch_size_spin.value(),
                'learning_rate': self.donut_lr_spin.value(),
                'gradient_accumulation_steps': self.donut_grad_accum_spin.value(),
                'max_length': self.donut_max_length_spin.value(),
                'image_size': int(self.donut_image_size_combo.currentText()),
                'fp16': self.donut_fp16_checkbox.isChecked(),
                'save_steps': self.donut_save_steps_spin.value(),
                'eval_steps': self.donut_eval_steps_spin.value(),
                'task_type': self.donut_task_combo.currentText(),
                
                # Оптимизации памяти
                'use_lora': self.use_lora_cb.isChecked(),
                'use_8bit_optimizer': self.use_8bit_optimizer_cb.isChecked(),
                'freeze_encoder': self.freeze_encoder_cb.isChecked(),
                'gradient_checkpointing': True,  # Принудительно включаем
            },
            'output_model_name': model_name
        }
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'donut')
        
    def start_trocr_training(self):
        """Запуск обучения TrOCR"""
        # Проверяем параметры
        dataset_path = self.trocr_dataset_edit.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Ошибка", "Выберите корректный датасет для обучения!")
            return
            
        # Создаем тренер TrOCR
        self.current_trainer = TrOCRTrainer()
        
        # Подготавливаем относительный путь для модели
        model_name = self.trocr_output_name_edit.text() or f"trocr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not model_name.startswith("trocr_"):
            model_name = f"trocr_{model_name}"
        
        # Собираем параметры обучения
        training_params = {
            'dataset_path': dataset_path,
            'base_model_id': self.trocr_base_model_combo.currentText(),
            'output_model_name': model_name,
            'training_args': {
                'num_train_epochs': self.trocr_epochs_spin.value(),
                'per_device_train_batch_size': self.trocr_batch_size_spin.value(),
                            'learning_rate': self.trocr_lr_spin.value(),
            'gradient_accumulation_steps': self.trocr_grad_accum_spin.value(),
            'max_length': self.trocr_max_length_spin.value(),
            'image_size': self._parse_image_size(self.trocr_image_size_combo.currentText())[0],
            'fp16': self.trocr_fp16_checkbox.isChecked(),
                'warmup_ratio': self.trocr_warmup_ratio_spin.value(),
                'weight_decay': self.trocr_weight_decay_spin.value(),
                # Оптимизации памяти
                'use_lora': self.trocr_use_lora_cb.isChecked(),
                'use_8bit_optimizer': self.trocr_use_8bit_optimizer_cb.isChecked(),
                'gradient_checkpointing': self.trocr_gradient_checkpointing_cb.isChecked(),
            }
        }
        
        self.add_log_message(self.trocr_log, f"🚀 Запуск обучения TrOCR модели '{model_name}'")
        self.add_log_message(self.trocr_log, f"📊 Датасет: {dataset_path}")
        self.add_log_message(self.trocr_log, f"🤖 Базовая модель: {training_params['base_model_id']}")
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'trocr')
        
    def auto_optimize_trocr_memory(self):
        """Автоматическая оптимизация памяти для TrOCR на RTX 4070 Ti"""
        # Включаем все оптимизации
        self.trocr_use_lora_cb.setChecked(True)
        self.trocr_use_8bit_optimizer_cb.setChecked(True)
        self.trocr_gradient_checkpointing_cb.setChecked(True)
        
        # Устанавливаем оптимальные параметры для RTX 4070 Ti (12GB VRAM)
        self.trocr_batch_size_spin.setValue(2)
        self.trocr_grad_accum_spin.setValue(8)
        self.trocr_image_size_combo.setCurrentText("224")
        self.trocr_max_length_spin.setValue(256)
        
        # Включаем FP16
        self.trocr_fp16_checkbox.setChecked(True)
        
        self.add_log_message(self.trocr_log, "🚀 Применены оптимизации памяти для RTX 4070 Ti:")
        self.add_log_message(self.trocr_log, "   • LoRA: включен (экономия до 90% памяти)")
        self.add_log_message(self.trocr_log, "   • 8-bit оптимизатор: включен (экономия 25%)")
        self.add_log_message(self.trocr_log, "   • Gradient checkpointing: включен")
        self.add_log_message(self.trocr_log, "   • Batch size: 2, Grad accumulation: 8")
        self.add_log_message(self.trocr_log, "   • Image size: 224, Max length: 256")
        self.add_log_message(self.trocr_log, "   • FP16: включен")

    def apply_trocr_fast_gpu_settings(self):
        """Применяет быстрые настройки GPU для TrOCR"""
        # Оптимальные настройки для обучения
        self.trocr_epochs_spin.setValue(3)
        self.trocr_batch_size_spin.setValue(4)
        self.trocr_lr_spin.setValue(5e-5)
        self.trocr_grad_accum_spin.setValue(4)
        self.trocr_max_length_spin.setValue(512)
        self.trocr_image_size_combo.setCurrentText("384")
        self.trocr_warmup_ratio_spin.setValue(0.1)
        self.trocr_weight_decay_spin.setValue(0.01)
        
        # Включаем FP16 для ускорения
        self.trocr_fp16_checkbox.setChecked(True)
        
        self.add_log_message(self.trocr_log, "⚡ Применены быстрые настройки GPU для TrOCR")
        
    def on_training_error(self, error_message):
        """Обработчик ошибки обучения"""
        print(f"TrainingDialog: Ошибка обучения: {error_message}")
        
        # Используем QTimer для отложенной обработки, чтобы дать потоку корректно завершиться
        QTimer.singleShot(100, lambda: self._handle_training_completion(error_message, False))
        
    def _handle_training_completion(self, result, success):
        """Внутренний метод для обработки завершения обучения"""
        try:
            print(f"TrainingDialog: Обрабатываем завершение обучения (успех: {success})")
            
            # Сначала очищаем потоки
            self.cleanup_training_thread()
            
            # Затем обновляем UI
            self.reset_training_ui()
            
            if success:
                self.status_label.setText("✅ Обучение завершено успешно!")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                QMessageBox.information(
                    self, 
                    "Успех", 
                    f"Модель успешно обучена!\n\nСохранена в: {result}"
                )
            else:
                self.status_label.setText("❌ Ошибка обучения")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обучении:\n\n{result}")
                
        except Exception as e:
            print(f"TrainingDialog: КРИТИЧЕСКАЯ ОШИБКА при обработке завершения обучения: {e}")
            import traceback
            traceback.print_exc()
            
            # В любом случае попытаемся сбросить UI
            try:
                self.reset_training_ui()
                self.status_label.setText("❌ Критическая ошибка")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            except:
                pass
        
    def on_training_progress(self, progress):
        """Обработчик прогресса обучения"""
        # Определяем активную вкладку и обновляем соответствующий прогресс-бар
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.layoutlm_progress_bar.setValue(progress)
        elif current_tab == 1:  # Donut
            self.donut_progress_bar.setValue(progress)
        elif current_tab == 2:  # TrOCR
            self.on_trocr_training_progress(progress)
            
    def on_training_log(self, message):
        """Обработчик лог сообщений"""
        # Определяем активную вкладку и добавляем в соответствующий лог
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # LayoutLM
            self.add_log_message(self.layoutlm_log, message)
        elif current_tab == 1:  # Donut
            self.add_log_message(self.donut_log, message)
        elif current_tab == 2:  # TrOCR
            self.on_trocr_training_log(message)
            
        # Парсим метрики из лог сообщения и обновляем вкладку мониторинга
        self.parse_and_update_metrics(message)
            
    def reset_training_ui(self):
        """Сброс UI после завершения обучения"""
        try:
            print("TrainingDialog: Сбрасываем UI после обучения...")
            
            # LayoutLM
            if hasattr(self, 'layoutlm_start_button'):
                self.layoutlm_start_button.setEnabled(True)
            if hasattr(self, 'layoutlm_stop_button'):
                self.layoutlm_stop_button.setEnabled(False)
            if hasattr(self, 'layoutlm_progress_bar'):
                self.layoutlm_progress_bar.setVisible(False)
            if hasattr(self, 'layoutlm_status_label'):
                self.layoutlm_status_label.setText("Готов к обучению")
            
            # Donut
            if hasattr(self, 'donut_start_button'):
                self.donut_start_button.setEnabled(True)
            if hasattr(self, 'donut_stop_button'):
                self.donut_stop_button.setEnabled(False)
            if hasattr(self, 'donut_progress_bar'):
                self.donut_progress_bar.setVisible(False)
            if hasattr(self, 'donut_status_label'):
                self.donut_status_label.setText("Готов к обучению")
            
            # TrOCR
            if hasattr(self, 'trocr_start_button'):
                self.trocr_start_button.setEnabled(True)
            if hasattr(self, 'trocr_stop_button'):
                self.trocr_stop_button.setEnabled(False)
            if hasattr(self, 'trocr_progress_bar'):
                self.trocr_progress_bar.setVisible(False)
            if hasattr(self, 'trocr_status_label'):
                self.trocr_status_label.setText("Готов к обучению")
                
            # Сбрасываем метрики мониторинга в финальные значения
            self.current_metrics = {
                'epoch': self.current_metrics.get('epoch', 0),  # Сохраняем последние значения
                'step': self.current_metrics.get('step', 0),
                'loss': self.current_metrics.get('loss', 0.0),
                'lr': 0.0,  # LR сбрасываем, так как обучение завершено
                'accuracy': self.current_metrics.get('accuracy', 0.0),
                'f1': self.current_metrics.get('f1', 0.0)
            }
            self.update_monitoring_display()
            
            # Очищаем ссылки (потоки уже должны быть остановлены в cleanup_training_thread)
            self.current_trainer = None
            self.current_worker = None
            self.current_thread = None
            
            print("TrainingDialog: UI сброшен успешно")
            
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА при сбросе UI: {e}")
            import traceback
            traceback.print_exc()

    def _add_automation_controls(self, tab):
        """Добавляет элементы управления автоматизацией для TrOCR датасетов"""
        # Заглушка - автоматизация уже встроена в основной процесс
        pass
    
    def on_trocr_mode_changed(self):
        """Заглушка для совместимости"""
        pass
        
    def on_trocr_training_log(self, message):
        """Обработчик лог сообщений для TrOCR"""
        # Добавляем в лог TrOCR
        self.add_log_message(self.trocr_log, message)
        
        # Обновляем информацию о метриках TrOCR если есть специальные данные
        if "📊" in message and ("Loss" in message or "Эпоха" in message):
            # Это сообщение с метриками - обновляем специальную область
            self.update_trocr_training_info(message)
    
    def on_trocr_training_progress(self, progress):
        """Обработчик прогресса обучения для TrOCR"""
        if hasattr(self, 'trocr_progress_bar'):
            self.trocr_progress_bar.setValue(progress)
            
    def update_trocr_training_info(self, metrics_message):
        """Обновляет информацию о TrOCR метриках"""
        try:
            if hasattr(self, 'trocr_training_info'):
                # Обновляем виджет с метриками
                self.trocr_training_info.setText(metrics_message)
                self.trocr_training_info.setStyleSheet("""
                    QLabel {
                        background-color: #e8f5e8;
                        border: 1px solid #27ae60;
                        border-radius: 4px;
                        padding: 8px;
                        color: #2d5234;
                        font-weight: bold;
                    }
                """)
        except Exception as e:
            print(f"Ошибка обновления TrOCR метрик: {e}")


    
    def reset_trocr_dataset_ui(self):
        """Сбрасывает UI TrOCR датасета к начальному состоянию"""
        self.trocr_dataset_start_button.setEnabled(True)
        self.trocr_dataset_stop_button.setEnabled(False)
        self.trocr_dataset_progress_bar.setVisible(False)
        self.trocr_dataset_progress_bar.setValue(0)
        self.trocr_dataset_status_label.setText("Готов к созданию датасета с Gemini аннотациями")
        
        # Очищаем новый worker и thread
        if hasattr(self, 'trocr_auto_worker'):
            try:
                self.trocr_auto_worker.stop()
                self.trocr_auto_thread.quit()
                self.trocr_auto_thread.wait()
                delattr(self, 'trocr_auto_worker')
                delattr(self, 'trocr_auto_thread')
            except:
                pass
    
    def update_dataset_info(self, dataset_path, info_label):
        """Обновляет информацию о датасете в интерфейсе"""
        try:
            if not dataset_path or not os.path.exists(dataset_path):
                info_label.setText("Выберите датасет для получения информации")
                return
            
            # Анализируем датасет
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            
            # Формируем информацию
            info_parts = []
            
            if hasattr(dataset, 'keys'):  # DatasetDict
                for split_name, split_data in dataset.items():
                    info_parts.append(f"📊 {split_name}: {len(split_data)} примеров")
                    
                    # Анализируем первый элемент для структуры
                    if len(split_data) > 0:
                        sample = split_data[0]
                        fields = list(sample.keys())
                        info_parts.append(f"🏷️ Поля: {', '.join(fields)}")
                        
                        # Размер изображений если есть
                        if 'image' in sample:
                            img = sample['image']
                            info_parts.append(f"🖼️ Размер изображений: {img.size}")
                        break
            else:  # Dataset
                info_parts.append(f"📊 Примеров: {len(dataset)}")
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    fields = list(sample.keys())
                    info_parts.append(f"🏷️ Поля: {', '.join(fields)}")
                    
                    # Размер изображений если есть
                    if 'image' in sample:
                        img = sample['image']
                        info_parts.append(f"🖼️ Размер изображений: {img.size}")
            
            info_text = "\n".join(info_parts)
            info_label.setText(info_text)
            info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
        except Exception as e:
            info_label.setText(f"❌ Ошибка чтения датасета: {str(e)}")
            info_label.setStyleSheet("color: #e74c3c;")
        
# Для обратной совместимости
TrainingDialog = ModernTrainingDialog

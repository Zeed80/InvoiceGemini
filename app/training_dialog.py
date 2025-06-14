import os
import json
import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QWidget, 
    QLineEdit, QFileDialog, QTextEdit, QFrame, QGroupBox, QMessageBox, QApplication, QInputDialog,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFormLayout, QGridLayout, QCheckBox, QComboBox,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor

# Предполагается, что эти классы будут доступны для type hinting или будут импортированы позже
# from ..processing_engine import OCRProcessor, GeminiProcessor 
# from ..config import Config # Это пример, импортировать нужно будет корректно

# NEW: Импортируем ModelTrainer (предполагаем, что он будет в trainer.py)
from .training.trainer import ModelTrainer
from .training.data_preparator import TrainingDataPreparator # Переносим импорт сюда для порядка
from .training.donut_trainer import DonutTrainer as DonutTrainerClass
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
                    except:
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
                                
                        except:
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
                                            
                        except:
                            continue
            
            if total_fields > 0:
                completeness_score = ((total_fields - empty_fields) / total_fields) * 100
                
        except:
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
                                        
                        except:
                            continue
            
            if total_annotations > 0:
                quality_score = (valid_annotations / total_annotations) * 100
                
        except:
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
                                
                    except:
                        corrupted_files += 1
            
            if total_files > 0:
                integrity_score = ((total_files - corrupted_files) / total_files) * 100
                
        except:
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
                
        except:
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
                log_callback=self.log_message.emit,
                progress_callback=self.progress.emit
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
            else:
                raise ValueError("Неизвестный тип тренера")
                
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
        self.layoutlm_output_name_edit.setText(f"layoutlm_v3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
        self.donut_output_name_edit.setText(f"donut_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        
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
                
    def update_dataset_info(self, dataset_path, info_label):
        """Обновляет информацию о датасете"""
        try:
            if not os.path.exists(dataset_path):
                info_label.setText("Датасет не найден")
                return
                
            # Подсчитываем файлы
            total_files = 0
            train_files = 0
            val_files = 0
            
            for root, dirs, files in os.walk(dataset_path):
                if 'train' in root.lower():
                    train_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))])
                elif 'val' in root.lower() or 'validation' in root.lower():
                    val_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))])
                else:
                    total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))])
            
            if train_files > 0 or val_files > 0:
                info_text = f"Обучение: {train_files} файлов, Валидация: {val_files} файлов"
            else:
                info_text = f"Всего файлов: {total_files}"
                
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
            settings_manager.set_value('Training', 'last_source_folder_timestamp', datetime.datetime.now().isoformat())
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
        else:
            model_prefix = "unknown"
            
        # Обновляем placeholder текст
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
        model_name = self.layoutlm_output_name_edit.text() or f"layoutlm_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        model_name = self.donut_output_name_edit.text() or f"donut_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            },
            'output_model_name': model_name,
            'output_model_path': os.path.join("data", "trained_models", model_name)
        }
        
        # Запускаем обучение в отдельном потоке
        self.start_training_thread(training_params, 'donut')
        
    def start_training_thread(self, training_params, model_type):
        """Запуск обучения в отдельном потоке"""
        # Сбрасываем метрики в начале обучения
        self.current_metrics = {
            'epoch': 0,
            'step': 0,
            'loss': 0.0,
            'lr': 0.0,
            'accuracy': 0.0,
            'f1': 0.0
        }
        self.update_monitoring_display()
        
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
        
        # НЕ добавляем автоматическое завершение потока - будем делать это в cleanup_training_thread
        # Это предотвращает двойную очистку и потенциальные ошибки
        
        # Обновляем UI
        if model_type == 'layoutlm':
            self.layoutlm_start_button.setEnabled(False)
            self.layoutlm_stop_button.setEnabled(True)
            self.layoutlm_progress_bar.setVisible(True)
            self.layoutlm_progress_bar.setValue(0)
            self.layoutlm_status_label.setText("🚀 Инициализация обучения...")
        elif model_type == 'donut':
            self.donut_start_button.setEnabled(False)
            self.donut_stop_button.setEnabled(True)
            self.donut_progress_bar.setVisible(True)
            self.donut_progress_bar.setValue(0)
            self.donut_status_label.setText("🚀 Инициализация обучения...")
            
        self.status_label.setText("🔄 Выполняется обучение модели...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
        # Запускаем поток
        self.current_thread.start()
        
    def start_dataset_preparation(self):
        """Запуск подготовки датасета"""
        source_folder = self.source_folder_edit.text()
        if not source_folder or not os.path.exists(source_folder):
            QMessageBox.warning(self, "Ошибка", "Выберите корректную папку с документами!")
            return
            
        # Получаем параметры подготовки
        dataset_type = self.dataset_type_combo.currentText()
        
        # Определяем префикс модели на основе выбранного типа датасета
        if "LayoutLM" in dataset_type:
            model_prefix = "layoutlm"
        elif "Donut" in dataset_type:
            model_prefix = "donut"
        else:
            model_prefix = "unknown"
            
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Берем пользовательское имя или создаем автоматическое
        user_dataset_name = self.dataset_name_edit.text().strip()
        
        if user_dataset_name:
            # Если пользователь указал имя, проверяем есть ли уже правильный префикс
            if not user_dataset_name.startswith(f"{model_prefix}_"):
                dataset_name = f"{model_prefix}_{user_dataset_name}"
            else:
                dataset_name = user_dataset_name
        else:
            # Автоматическое имя с правильным префиксом
            dataset_name = f"{model_prefix}_dataset_{timestamp}"
            
        annotation_method = self.annotation_method_combo.currentText().lower()
        max_files = self.max_files_spin.value() if self.max_files_spin.value() > 0 else None
        
        # Создаем относительный путь для датасета (внутри проекта)
        output_path = os.path.join("data", "training_datasets", dataset_name)
        
        # Определяем режим подготовки
        is_intelligent_mode = self.preparation_mode_combo.currentIndex() == 0
        
        if is_intelligent_mode:
            self.add_log_message(self.prepare_log, "🧠 Запуск интеллектуального режима подготовки")
            self.add_log_message(self.prepare_log, "   • Gemini будет извлекать ВСЕ полезные данные")
            self.add_log_message(self.prepare_log, "   • Ожидается более высокое качество датасета")
        else:
            self.add_log_message(self.prepare_log, "📝 Запуск стандартного режима подготовки")
        
        # Создаем data preparator
        data_preparator = TrainingDataPreparator(
            self.app_config, 
            self.ocr_processor, 
            self.gemini_processor
        )
        
        # Устанавливаем режим подготовки
        data_preparator.intelligent_mode = is_intelligent_mode
        
        # Коллбеки будут установлены внутри Worker для правильной работы с сигналами
        # data_preparator.set_callbacks(...) - убираем, чтобы избежать дублирования
        
        # Создаем worker для подготовки данных
        class DataPreparationWorker(QObject):
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            progress_updated = pyqtSignal(int)  # Сигнал для обновления прогресса
            log_message = pyqtSignal(str)       # Сигнал для логирования
            
            def __init__(self, preparator, source_folder, output_path, dataset_type, annotation_method, max_files):
                super().__init__()
                self.preparator = preparator
                self.source_folder = source_folder
                self.output_path = output_path
                self.dataset_type = dataset_type
                self.annotation_method = annotation_method
                self.max_files = max_files
                
            def run(self):
                import sys
                import traceback
                import os
                
                try:
                    print(f"DataPreparationWorker: ===============================")
                    print(f"DataPreparationWorker: ЗАПУСК WORKER В ЗАЩИЩЕННОМ РЕЖИМЕ")
                    print(f"DataPreparationWorker: ===============================")
                    print(f"DataPreparationWorker: PID процесса: {os.getpid()}")
                    print(f"DataPreparationWorker: Версия Python: {sys.version}")
                    print(f"DataPreparationWorker: Источник: {self.source_folder}")
                    print(f"DataPreparationWorker: Выход: {self.output_path}")
                    print(f"DataPreparationWorker: Тип датасета: {self.dataset_type}")
                    print(f"DataPreparationWorker: Метод: {self.annotation_method}")
                    print(f"DataPreparationWorker: Макс. файлов: {self.max_files}")
                    
                    # Проверяем состояние системы
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        print(f"DataPreparationWorker: Память до начала: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except Exception as mem_e:
                        print(f"DataPreparationWorker: Не удалось получить информацию о памяти: {mem_e}")
                    
                    # Проверяем параметры
                    if not self.source_folder or not os.path.exists(self.source_folder):
                        raise ValueError(f"Исходная папка не существует или недоступна: {self.source_folder}")
                    
                    if not self.output_path:
                        raise ValueError("Не указан путь для сохранения датасета")
                        
                    if not self.preparator:
                        raise ValueError("Preparator не инициализирован")
                    
                    print(f"DataPreparationWorker: Входные параметры проверены успешно")
                    
                    # Устанавливаем коллбеки для получения логов и прогресса
                    def log_callback(message):
                        try:
                            print(f"DataPreparator: {message}")
                            # Отправляем сигнал в UI
                            self.log_message.emit(message)
                        except Exception as log_e:
                            # Даже логирование может вылететь при Unicode ошибках
                            try:
                                safe_message = str(message).encode('ascii', 'replace').decode('ascii')
                                print(f"DataPreparator: {safe_message}")
                                self.log_message.emit(safe_message)
                            except:
                                print(f"DataPreparator: [СООБЩЕНИЕ НЕ МОЖЕТ БЫТЬ ОТОБРАЖЕНО]")
                                self.log_message.emit("[СООБЩЕНИЕ НЕ МОЖЕТ БЫТЬ ОТОБРАЖЕНО]")
                    
                    def progress_callback(progress):
                        try:
                            print(f"DataPreparator: Прогресс: {progress}%")
                            # Отправляем сигнал в UI
                            self.progress_updated.emit(progress)
                        except Exception as prog_e:
                            print(f"DataPreparator: Ошибка обновления прогресса: {prog_e}")
                        
                    print(f"DataPreparationWorker: Установка коллбеков...")
                    self.preparator.set_callbacks(
                        log_callback=log_callback,
                        progress_callback=progress_callback
                    )
                    print(f"DataPreparationWorker: Коллбеки установлены")
                    
                    # Выбираем метод подготовки в зависимости от типа датасета
                    result = None
                    
                    try:
                        if "LayoutLM" in self.dataset_type:
                            print(f"DataPreparationWorker: ЗАПУСК prepare_dataset_for_layoutlm_modern...")
                            result = self.preparator.prepare_dataset_for_layoutlm_modern(
                                source_folder=self.source_folder,
                                output_path=self.output_path,
                                task_type="token_classification",
                                annotation_method=self.annotation_method,
                                max_files=self.max_files
                            )
                            print(f"DataPreparationWorker: prepare_dataset_for_layoutlm_modern завершен")
                        else:
                            # Donut датасет
                            print(f"DataPreparationWorker: ЗАПУСК prepare_dataset_for_donut_modern...")
                            
                            # Определяем тип задачи для Donut
                            if "VQA" in self.dataset_type:
                                task_type = "document_vqa"
                            else:
                                task_type = "document_parsing"
                            
                            result = self.preparator.prepare_dataset_for_donut_modern(
                                source_folder=self.source_folder,
                                output_path=self.output_path,
                                task_type=task_type,
                                annotation_method=self.annotation_method,
                                max_files=self.max_files
                            )
                            print(f"DataPreparationWorker: prepare_dataset_for_donut_modern завершен")
                        
                    except SystemExit as sys_exit:
                        print(f"DataPreparationWorker: КРИТИЧЕСКАЯ ОШИБКА SystemExit: {sys_exit}")
                        print(f"DataPreparationWorker: Код выхода: {sys_exit.code}")
                        result = None
                        raise sys_exit
                        
                    except KeyboardInterrupt as kb_int:
                        print(f"DataPreparationWorker: Прерывание пользователем: {kb_int}")
                        result = None
                        raise kb_int
                        
                    except MemoryError as mem_err:
                        print(f"DataPreparationWorker: КРИТИЧЕСКАЯ ОШИБКА памяти: {mem_err}")
                        result = None
                        raise mem_err
                        
                    except Exception as prep_error:
                        print(f"DataPreparationWorker: ОШИБКА в prepare_dataset_for_donut_modern: {str(prep_error)}")
                        print(f"DataPreparationWorker: Тип ошибки: {type(prep_error).__name__}")
                        print(f"DataPreparationWorker: Полная трассировка preparator:")
                        try:
                            traceback_lines = traceback.format_exc().split('\n')
                            for line in traceback_lines:
                                if line.strip():
                                    print(f"DataPreparationWorker:   {line}")
                        except:
                            print(f"DataPreparationWorker: Не удалось получить трассировку")
                        result = None
                        raise prep_error
                    
                    # Проверяем результат
                    if result:
                        print(f"DataPreparationWorker: УСПЕХ! Датасет подготовлен: {result}")
                        try:
                            self.finished.emit(result)
                            print(f"DataPreparationWorker: Сигнал finished отправлен успешно")
                        except Exception as emit_error:
                            print(f"DataPreparationWorker: Ошибка отправки сигнала finished: {emit_error}")
                    else:
                        print(f"DataPreparationWorker: НЕУДАЧА - результат пустой")
                        error_msg = "Не удалось подготовить датасет - получен пустой результат"
                        try:
                            self.error.emit(error_msg)
                            print(f"DataPreparationWorker: Сигнал error отправлен")
                        except Exception as emit_error:
                            print(f"DataPreparationWorker: Ошибка отправки сигнала error: {emit_error}")
                        
                except SystemExit as sys_exit:
                    print(f"DataPreparationWorker: СИСТЕМНЫЙ ВЫХОД: {sys_exit}")
                    error_msg = f"Критическая системная ошибка: {sys_exit}"
                    try:
                        self.error.emit(error_msg)
                    except:
                        print(f"DataPreparationWorker: Не удалось отправить сигнал ошибки SystemExit")
                    
                except KeyboardInterrupt:
                    print(f"DataPreparationWorker: ПРЕРЫВАНИЕ ПОЛЬЗОВАТЕЛЕМ")
                    error_msg = "Операция прервана пользователем"
                    try:
                        self.error.emit(error_msg)
                    except:
                        print(f"DataPreparationWorker: Не удалось отправить сигнал ошибки KeyboardInterrupt")
                    
                except MemoryError as mem_error:
                    print(f"DataPreparationWorker: КРИТИЧЕСКАЯ ОШИБКА ПАМЯТИ: {mem_error}")
                    error_msg = f"Недостаточно памяти для выполнения операции: {mem_error}"
                    try:
                        self.error.emit(error_msg)
                    except:
                        print(f"DataPreparationWorker: Не удалось отправить сигнал ошибки MemoryError")
                    
                except Exception as global_error:
                    print(f"DataPreparationWorker: ГЛОБАЛЬНАЯ ОШИБКА WORKER: {str(global_error)}")
                    print(f"DataPreparationWorker: Тип глобальной ошибки: {type(global_error).__name__}")
                    
                    # Максимально безопасная трассировка
                    try:
                        traceback_text = traceback.format_exc()
                        print(f"DataPreparationWorker: Полная трассировка Worker:")
                        for line in traceback_text.split('\n'):
                            if line.strip():
                                print(f"DataPreparationWorker:   {line}")
                    except Exception as trace_error:
                        print(f"DataPreparationWorker: Не удалось получить трассировку: {trace_error}")
                    
                    # Проверяем состояние памяти при ошибке
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        print(f"DataPreparationWorker: Память при ошибке: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        print(f"DataPreparationWorker: Не удалось получить информацию о памяти при ошибке")
                    
                    error_msg = f"Критическая ошибка подготовки датасета: {str(global_error)}"
                    try:
                        self.error.emit(error_msg)
                        print(f"DataPreparationWorker: Сигнал глобальной ошибки отправлен")
                    except Exception as emit_error:
                        print(f"DataPreparationWorker: КРИТИЧНО: Не удалось отправить сигнал ошибки: {emit_error}")
                
                finally:
                    print(f"DataPreparationWorker: Завершение Worker (finally блок)")
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        print(f"DataPreparationWorker: Финальное использование памяти: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except:
                        pass
                    print(f"DataPreparationWorker: Worker завершен")
        
        # Создаем worker и поток
        self.preparation_worker = DataPreparationWorker(
            data_preparator, source_folder, output_path, dataset_type, annotation_method, max_files
        )
        self.preparation_thread = QThread()
        
        # Перемещаем worker в поток
        self.preparation_worker.moveToThread(self.preparation_thread)
        
        # Подключаем сигналы
        self.preparation_thread.started.connect(self.preparation_worker.run)
        self.preparation_worker.finished.connect(self.on_preparation_finished)
        self.preparation_worker.error.connect(self.on_preparation_error)
        self.preparation_worker.progress_updated.connect(self.on_preparation_progress)
        self.preparation_worker.log_message.connect(self.on_preparation_log)
        
        # ВАЖНО: Подключаем завершение потока
        self.preparation_worker.finished.connect(self.preparation_thread.quit)
        self.preparation_worker.error.connect(self.preparation_thread.quit)
        self.preparation_thread.finished.connect(self.preparation_worker.deleteLater)
        self.preparation_thread.finished.connect(self.preparation_thread.deleteLater)
        
        # Обновляем UI
        self.prepare_start_button.setEnabled(False)
        self.prepare_stop_button.setEnabled(True)
        self.prepare_progress_bar.setVisible(True)
        self.prepare_progress_bar.setValue(0)
        self.prepare_status_label.setText("🚀 Начинаем подготовку датасета...")
        
        # Запускаем поток
        self.preparation_thread.start()
        
    def stop_training(self):
        """Остановка обучения"""
        try:
            print("TrainingDialog: Остановка обучения по запросу пользователя...")
            
            if self.current_trainer:
                print("TrainingDialog: Останавливаем trainer...")
                self.current_trainer.stop()
                
            # Правильно очищаем потоки
            self.cleanup_training_thread()
                
            self.reset_training_ui()
            self.status_label.setText("⏹️ Обучение остановлено")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            
            print("TrainingDialog: Обучение остановлено успешно")
            
        except Exception as e:
            print(f"TrainingDialog: ОШИБКА при остановке обучения: {e}")
            import traceback
            traceback.print_exc()
            
            # В любом случае попытаемся сбросить UI
            try:
                self.reset_training_ui()
                self.status_label.setText("⚠️ Обучение остановлено с ошибками")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            except:
                pass
        
    def stop_preparation(self):
        """Остановка подготовки датасета"""
        if hasattr(self, 'preparation_worker') and self.preparation_worker:
            self.preparation_worker.preparator.stop()
            
        if hasattr(self, 'preparation_thread') and self.preparation_thread and self.preparation_thread.isRunning():
            self.preparation_thread.quit()
            self.preparation_thread.wait()
            
        self.prepare_start_button.setEnabled(True)
        self.prepare_stop_button.setEnabled(False)
        self.prepare_progress_bar.setVisible(False)
        self.prepare_status_label.setText("⏹️ Подготовка остановлена")
        
    def on_preparation_finished(self, dataset_path):
        """Обработчик завершения подготовки датасета"""
        self.prepare_start_button.setEnabled(True)
        self.prepare_stop_button.setEnabled(False)
        self.prepare_progress_bar.setVisible(False)
        self.prepare_status_label.setText("✅ Датасет подготовлен успешно!")
        
        # Сохраняем путь к созданному датасету для анализа качества
        self.last_created_dataset = dataset_path
        
        # Определяем тип датасета и автоматически заполняем соответствующее поле
        dataset_type = self.dataset_type_combo.currentText()
        
        # DataPreparator сохраняет HuggingFace Dataset в подпапку dataset_dict (если не сам путь уже dataset_dict)
        if dataset_path.endswith("dataset_dict"):
            hf_dataset_path = dataset_path
        else:
            hf_dataset_path = os.path.join(dataset_path, "dataset_dict")
        
        success_message = f"Датасет успешно подготовлен!\n\nСохранен в: {dataset_path}"
        
        if "LayoutLM" in dataset_type:
            # Для LayoutLM используем путь к HuggingFace датасету
            if os.path.exists(hf_dataset_path):
                self.layoutlm_dataset_edit.setText(hf_dataset_path)
                self.add_log_message(self.prepare_log, f"✅ Путь к LayoutLM датасету автоматически установлен: {hf_dataset_path}")
                success_message += f"\n\nПуть к LayoutLM Dataset: {hf_dataset_path}"
            else:
                self.add_log_message(self.prepare_log, f"⚠️ Предупреждение: не найдена папка dataset_dict в {dataset_path}")
                success_message += f"\n\n⚠️ Предупреждение: dataset_dict не найден"
        else:
            # Для Donut используем основную папку датасета
            self.donut_dataset_edit.setText(dataset_path)
            self.add_log_message(self.prepare_log, f"✅ Путь к Donut датасету автоматически установлен: {dataset_path}")
            success_message += f"\n\nПуть к Donut Dataset: {dataset_path}"
        
        # Автоматически запускаем анализ качества датасета
        try:
            self.add_log_message(self.prepare_log, "🔍 Запуск автоматического анализа качества...")
            results = self.quality_analyzer.analyze_dataset(dataset_path)
            self.last_quality_results = results
            self.update_quality_display(results)
            self.add_log_message(self.prepare_log, f"📊 Анализ качества завершен. Общий балл: {results['overall_score']:.1f}%")
            
            # Добавляем результат анализа к сообщению об успехе
            score = results['overall_score']
            if score >= 80:
                quality_status = "🟢 Отличное качество"
            elif score >= 60:
                quality_status = "🟡 Хорошее качество"
            elif score >= 40:
                quality_status = "🟠 Удовлетворительное качество"
            else:
                quality_status = "🔴 Требуется улучшение"
                
            success_message += f"\n\n📊 Анализ качества: {quality_status} ({score:.1f}%)"
            
        except Exception as e:
            self.add_log_message(self.prepare_log, f"⚠️ Ошибка анализа качества: {str(e)}")
            print(f"Ошибка автоматического анализа качества: {e}")
        
        QMessageBox.information(
            self,
            "Успех",
            success_message
        )
        
        # Очищаем ссылки
        self.preparation_worker = None
        self.preparation_thread = None
        
    def on_preparation_error(self, error_message):
        """Обработчик ошибки подготовки датасета"""
        self.prepare_start_button.setEnabled(True)
        self.prepare_stop_button.setEnabled(False)
        self.prepare_progress_bar.setVisible(False)
        self.prepare_status_label.setText("❌ Ошибка подготовки")
        
        QMessageBox.critical(
            self,
            "Ошибка",
            f"Произошла ошибка при подготовке датасета:\n\n{error_message}"
        )
        
        # Очищаем ссылки
        self.preparation_worker = None
        self.preparation_thread = None
        
    def on_preparation_progress(self, progress):
        """Обработчик прогресса подготовки датасета"""
        self.prepare_progress_bar.setValue(progress)
        
    def on_preparation_log(self, message):
        """Обработчик лог сообщений подготовки датасета"""
        self.add_log_message(self.prepare_log, message)
        
    def analyze_dataset_quality(self):
        """Анализирует качество выбранного или созданного датасета"""
        try:
            # Определяем путь к датасету для анализа
            dataset_path = None
            
            # Проверяем активную вкладку и соответствующие поля
            current_tab = self.tab_widget.currentIndex()
            
            if current_tab == 0:  # LayoutLM
                dataset_path = self.layoutlm_dataset_edit.text().strip()
            elif current_tab == 1:  # Donut
                dataset_path = self.donut_dataset_edit.text().strip()
            elif current_tab == 2:  # Подготовка данных
                # Проверяем последний созданный датасет
                if hasattr(self, 'last_created_dataset'):
                    dataset_path = self.last_created_dataset
                else:
                    # Спрашиваем пользователя выбрать датасет
                    dataset_path = QFileDialog.getExistingDirectory(
                        self,
                        "Выберите датасет для анализа качества",
                        self.app_config.TRAINING_DATASETS_PATH if hasattr(self.app_config, 'TRAINING_DATASETS_PATH') else ""
                    )
            
            if not dataset_path or not os.path.exists(dataset_path):
                # Предлагаем выбрать датасет
                dataset_path = QFileDialog.getExistingDirectory(
                    self,
                    "Выберите датасет для анализа качества",
                    self.app_config.TRAINING_DATASETS_PATH if hasattr(self.app_config, 'TRAINING_DATASETS_PATH') else ""
                )
                
                if not dataset_path:
                    return
            
            # Запускаем анализ
            self.analyze_quality_button.setEnabled(False)
            self.analyze_quality_button.setText("⏳ Анализируем...")
            QApplication.processEvents()
            
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
        print(f"TrainingDialog: Обучение завершено успешно: {model_path}")
        
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
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
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
            f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
            f"training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
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

# Для обратной совместимости
TrainingDialog = ModernTrainingDialog

# Для удобства запуска диалога отдельно, если потребуется:
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    # Mock-классы для тестирования диалога, если запускается отдельно
    class MockConfig:
        TRAINING_DATASETS_PATH = "mock_datasets"
        TRAINED_MODELS_PATH = "mock_trained_models"
        GEMINI_ANNOTATION_PROMPT_DEFAULT = "Mock Gemini Prompt"
        LAYOUTLM_MODEL_ID_FOR_TRAINING = "mock/layoutlm-base"
        DEFAULT_TRAIN_EPOCHS = 1
        DEFAULT_TRAIN_BATCH_SIZE = 1
        DEFAULT_LEARNING_RATE = 1e-5
        
    class MockProcessor:
        def process_image(self, *args, **kwargs): return {}
        def get_full_prompt(self, *args, **kwargs): return "Mock Prompt"

    app = QApplication(sys.argv)
    # Убедимся, что папки существуют для мок-запуска
    os.makedirs(MockConfig.TRAINING_DATASETS_PATH, exist_ok=True)
    os.makedirs(MockConfig.TRAINED_MODELS_PATH, exist_ok=True)

    dialog = TrainingDialog(MockConfig(), MockProcessor(), MockProcessor())
    dialog.show()
    sys.exit(app.exec()) 
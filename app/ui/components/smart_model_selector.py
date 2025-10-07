"""
Smart Model Selector - умный селектор моделей с AI рекомендациями
Заменяет 5 радиокнопок на компактный dropdown с подсказками
"""
from typing import Optional, Dict, Any
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
    QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import logging
logger = logging.getLogger(__name__)


class SmartModelSelector(QWidget):
    """
    Умный селектор модели с автоматическими рекомендациями
    
    Возможности:
    - Компактный dropdown вместо радиокнопок
    - Режим "Авто" с AI рекомендациями
    - Описания и tooltips для каждой модели
    - Показ ожидаемого времени обработки
    """
    
    model_changed = pyqtSignal(str)  # Испускается при изменении модели
    recommendation_requested = pyqtSignal(str)  # Запрос рекомендации для файла
    
    # Информация о моделях
    MODELS_INFO = {
        'auto': {
            'name': '🤖 Авто (рекомендуется)',
            'description': 'Автоматический выбор лучшей модели на основе анализа документа',
            'speed': 'Зависит от выбранной модели',
            'accuracy': 'Оптимальная',
            'requires_gpu': False,
            'requires_api': False,
            'tooltip': (
                'Автоматически выбирает оптимальную модель:\n'
                '• PDF с текстом → Gemini (быстро)\n'
                '• Сложная структура → LayoutLM (точно)\n'
                '• Низкое качество → Donut (надежно)'
            )
        },
        'gemini': {
            'name': '💎 Быстрый (Gemini)',
            'description': 'Облачная модель Google Gemini 2.0 Flash',
            'speed': '3-5 сек',
            'accuracy': 'Высокая (85-90%)',
            'requires_gpu': False,
            'requires_api': True,
            'tooltip': (
                'Google Gemini 2.0 Flash:\n'
                '• Самый быстрый вариант\n'
                '• Отлично работает с PDF текстом\n'
                '• Требует API ключ\n'
                '• Облачная обработка'
            )
        },
        'layoutlm': {
            'name': '🎯 Точный (LayoutLM)',
            'description': 'LayoutLMv3 - специализированная модель для документов',
            'speed': '10-15 сек',
            'accuracy': 'Максимальная (90-95%)',
            'requires_gpu': True,
            'requires_api': False,
            'tooltip': (
                'LayoutLMv3:\n'
                '• Лучшее понимание структуры\n'
                '• Максимальная точность\n'
                '• Требует GPU (рекомендуется)\n'
                '• Локальная обработка'
            )
        },
        'donut': {
            'name': '💪 Надежный (Donut)',
            'description': 'Donut - robust модель для любых условий',
            'speed': '8-12 сек',
            'accuracy': 'Хорошая (80-85%)',
            'requires_gpu': True,
            'requires_api': False,
            'tooltip': (
                'Donut:\n'
                '• Работает с низким качеством\n'
                '• Надежен для сканов\n'
                '• Средняя скорость\n'
                '• Локальная обработка'
            )
        },
        'trocr': {
            'name': '📝 OCR (TrOCR)',
            'description': 'TrOCR - специализированный OCR',
            'speed': '5-8 сек',
            'accuracy': 'Средняя (75-80%)',
            'requires_gpu': True,
            'requires_api': False,
            'tooltip': (
                'TrOCR:\n'
                '• Специализированный OCR\n'
                '• Хорош для рукописного текста\n'
                '• Требует дополнительной обработки\n'
                '• Локальная обработка'
            )
        }
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.recommendation = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Создание интерфейса"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Заголовок
        header = QLabel("Выбор модели:")
        header.setStyleSheet("font-weight: 600; font-size: 12px;")
        layout.addWidget(header)
        
        # Основной combobox
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(36)
        
        # Добавляем модели
        for model_id, info in self.MODELS_INFO.items():
            self.model_combo.addItem(info['name'], model_id)
        
        # Стилизация
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                color: #2c3e50;
            }
            
            QComboBox:hover {
                border-color: #3498db;
            }
            
            QComboBox:focus {
                border-color: #3498db;
                background-color: #f8fbfd;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #7f8c8d;
                margin-right: 8px;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                border: 2px solid #d5dbdb;
                selection-background-color: #e8f4f8;
                selection-color: #2c3e50;
                padding: 4px;
            }
        """)
        
        # Подключаем обработчик
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        
        layout.addWidget(self.model_combo)
        
        # Информационная панель
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)
        
        # Описание модели
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("font-size: 11px; color: #5a6c7d;")
        info_layout.addWidget(self.description_label)
        
        # Характеристики
        char_layout = QHBoxLayout()
        
        self.speed_label = QLabel()
        self.speed_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        char_layout.addWidget(self.speed_label)
        
        char_layout.addStretch()
        
        self.accuracy_label = QLabel()
        self.accuracy_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        char_layout.addWidget(self.accuracy_label)
        
        info_layout.addLayout(char_layout)
        
        layout.addWidget(self.info_frame)
        
        # Рекомендация (скрыта по умолчанию)
        self.recommendation_frame = QFrame()
        self.recommendation_frame.setVisible(False)
        self.recommendation_frame.setStyleSheet("""
            QFrame {
                background-color: #ebf5fb;
                border-left: 4px solid #3498db;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        rec_layout = QVBoxLayout(self.recommendation_frame)
        rec_layout.setContentsMargins(8, 8, 8, 8)
        rec_layout.setSpacing(4)
        
        rec_header = QLabel("💡 Рекомендация:")
        rec_header.setStyleSheet("font-weight: 600; font-size: 11px; color: #21618c;")
        rec_layout.addWidget(rec_header)
        
        self.recommendation_label = QLabel()
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet("font-size: 11px; color: #21618c;")
        rec_layout.addWidget(self.recommendation_label)
        
        layout.addWidget(self.recommendation_frame)
        
        # Обновляем информацию для текущей модели
        self._update_info()
    
    def _on_model_changed(self, index):
        """Обработка изменения модели"""
        model_id = self.model_combo.itemData(index)
        self._update_info()
        self.model_changed.emit(model_id)
        logger.info(f"Model changed to: {model_id}")
    
    def _update_info(self):
        """Обновление информационной панели"""
        model_id = self.get_current_model()
        info = self.MODELS_INFO.get(model_id, {})
        
        # Обновляем описание
        self.description_label.setText(info.get('description', ''))
        
        # Обновляем характеристики
        speed = info.get('speed', '')
        accuracy = info.get('accuracy', '')
        
        self.speed_label.setText(f"⏱️ {speed}")
        self.accuracy_label.setText(f"🎯 {accuracy}")
        
        # Обновляем tooltip
        tooltip = info.get('tooltip', '')
        self.model_combo.setToolTip(tooltip)
    
    def get_current_model(self) -> str:
        """Получить ID текущей выбранной модели"""
        return self.model_combo.currentData()
    
    def set_current_model(self, model_id: str):
        """
        Установить текущую модель
        
        Args:
            model_id: ID модели для выбора
        """
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_id:
                self.model_combo.setCurrentIndex(i)
                break
    
    def set_file_for_analysis(self, file_path: str):
        """
        Установить файл для анализа и получения рекомендации
        
        Args:
            file_path: Путь к файлу для анализа
        """
        self.current_file = file_path
        
        # Если выбран режим "Авто", показываем рекомендацию
        if self.get_current_model() == 'auto':
            self._show_recommendation_for_file(file_path)
    
    def _show_recommendation_for_file(self, file_path: str):
        """
        Показать рекомендацию модели для файла
        
        Args:
            file_path: Путь к файлу
        """
        try:
            recommendation = self._analyze_file_and_recommend(file_path)
            
            if recommendation:
                model_info = self.MODELS_INFO.get(recommendation['model'], {})
                
                self.recommendation_label.setText(
                    f"Рекомендуется: {model_info.get('name', recommendation['model'])}\n"
                    f"Причина: {recommendation.get('reason', 'Оптимальный выбор')}\n"
                    f"Ожидаемое время: {recommendation.get('estimated_time', 'неизвестно')}"
                )
                
                self.recommendation_frame.setVisible(True)
                self.recommendation = recommendation
        except Exception as e:
            logger.error(f"Error analyzing file for recommendation: {e}")
            self.recommendation_frame.setVisible(False)
    
    def _analyze_file_and_recommend(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Анализ файла и рекомендация модели
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с рекомендацией или None
        """
        file_path = Path(file_path)
        
        # Базовый анализ по расширению
        ext = file_path.suffix.lower()
        
        # PDF - проверяем текстовый слой
        if ext == '.pdf':
            try:
                from ...pdf_text_analyzer import has_text_layer
                
                if has_text_layer(str(file_path)):
                    return {
                        'model': 'gemini',
                        'confidence': 0.95,
                        'reason': 'PDF с текстовым слоем - Gemini даст быстрый и точный результат',
                        'estimated_time': '3-5 сек'
                    }
                else:
                    return {
                        'model': 'layoutlm',
                        'confidence': 0.90,
                        'reason': 'PDF без текста (скан) - LayoutLM лучше понимает структуру',
                        'estimated_time': '10-15 сек'
                    }
            except Exception as e:
                logger.warning(f"Could not analyze PDF: {e}")
                return {
                    'model': 'gemini',
                    'confidence': 0.70,
                    'reason': 'PDF файл - Gemini универсальный вариант',
                    'estimated_time': '5-8 сек'
                }
        
        # Изображения
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            # Можно добавить проверку качества изображения
            return {
                'model': 'donut',
                'confidence': 0.85,
                'reason': 'Изображение документа - Donut надежен для сканов',
                'estimated_time': '8-12 сек'
            }
        
        # По умолчанию
        return {
            'model': 'gemini',
            'confidence': 0.70,
            'reason': 'Универсальный вариант',
            'estimated_time': '5-10 сек'
        }
    
    def hide_recommendation(self):
        """Скрыть панель рекомендации"""
        self.recommendation_frame.setVisible(False)
        self.recommendation = None
    
    def get_recommendation(self) -> Optional[Dict[str, Any]]:
        """Получить текущую рекомендацию"""
        return self.recommendation


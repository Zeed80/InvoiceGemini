from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QProgressBar, QWidget, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import os

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Создаем виджет с вкладками
        self.tab_widget = QTabWidget()
        
        # Вкладка с метриками
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        
        self.loss_label = QLabel("Loss: N/A")
        self.accuracy_label = QLabel("Accuracy: N/A")
        self.f1_label = QLabel("F1 Score: N/A")
        self.lr_label = QLabel("Learning Rate: N/A")
        
        metrics_layout.addWidget(self.loss_label)
        metrics_layout.addWidget(self.accuracy_label)
        metrics_layout.addWidget(self.f1_label)
        metrics_layout.addWidget(self.lr_label)
        metrics_layout.addStretch()
        
        metrics_tab.setLayout(metrics_layout)
        
        # Вкладка с графиками
        plots_tab = QWidget()
        plots_layout = QVBoxLayout()
        
        # Создаем фигуру matplotlib
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)
        
        plots_tab.setLayout(plots_layout)
        
        # Добавляем вкладки
        self.tab_widget.addTab(metrics_tab, "Metrics")
        self.tab_widget.addTab(plots_tab, "Plots")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def update_metrics(self, metrics_file):
        """Обновляет отображение метрик из файла"""
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                
            if metrics['training_loss']:
                self.loss_label.setText(f"Loss: {metrics['training_loss'][-1]:.4f}")
                
            if metrics['learning_rates']:
                self.lr_label.setText(f"Learning Rate: {metrics['learning_rates'][-1]:.6f}")
                
            # Обновляем графики
            self.ax1.clear()
            self.ax2.clear()
            
            epochs = metrics['epochs']
            
            if metrics['training_loss']:
                self.ax1.plot(epochs, metrics['training_loss'], label='Training Loss')
            if metrics['eval_loss']:
                self.ax1.plot(epochs, metrics['eval_loss'], label='Validation Loss')
            
            self.ax1.set_title('Training and Validation Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend()
            
            if metrics['learning_rates']:
                self.ax2.plot(epochs, metrics['learning_rates'], label='Learning Rate')
                self.ax2.set_title('Learning Rate Schedule')
                self.ax2.set_xlabel('Epoch')
                self.ax2.set_ylabel('Learning Rate')
                self.ax2.legend()
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")

class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Основная информация
        info_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.progress_bar = QProgressBar()
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.progress_bar)
        
        # Метрики
        self.metrics_widget = MetricsWidget()
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        
        layout.addLayout(info_layout)
        layout.addWidget(self.metrics_widget)
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        self.setWindowTitle("Model Training")
        
    def update_progress(self, current_epoch, total_epochs):
        """Обновляет прогресс-бар и статус"""
        progress = int((current_epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Training: Epoch {current_epoch}/{total_epochs}")
        
    def update_metrics(self, metrics_file):
        """Обновляет отображение метрик"""
        self.metrics_widget.update_metrics(metrics_file)
        
    def training_finished(self, success):
        """Обработка завершения обучения"""
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        
        if success:
            self.status_label.setText("Training completed successfully")
        else:
            self.status_label.setText("Training failed")
            
    def start_training(self):
        """Запуск обучения"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing for training...")
        
    def stop_training(self):
        """Остановка обучения"""
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping training...") 
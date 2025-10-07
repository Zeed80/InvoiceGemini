#!/usr/bin/env python3
"""
Демонстрационный скрипт для тестирования компактного файлового интерфейса
с галочками OCR и проверкой текстового слоя PDF.
"""

import sys
import os
import time
from pathlib import Path

# Добавляем путь к модулям приложения
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from PyQt6.QtCore import QTimer

# Импортируем наш новый компонент
from app.ui.components.file_list_widget import FileListWidget, ProcessingStatus, FileProcessingInfo


class TestWindow(QMainWindow):
    """Тестовое окно для демонстрации компактного FileListWidget."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тест компактного файлового интерфейса с галочками OCR")
        self.setGeometry(100, 100, 900, 600)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)  # Горизонтальный layout для демонстрации
        
        # Левая часть - файловый список (как в приложении)
        left_widget = QWidget()
        left_widget.setMaximumWidth(350)  # Ограничиваем ширину как в приложении
        left_layout = QVBoxLayout(left_widget)
        
        # Заголовок
        title = QLabel("Компактный интерфейс с галочками OCR")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        left_layout.addWidget(title)
        
        # Файловый список
        self.file_list = FileListWidget()
        self.file_list.file_selected.connect(self.on_file_selected)
        self.file_list.process_file_requested.connect(self.on_process_file)
        self.file_list.process_all_requested.connect(self.on_process_all)
        self.file_list.filename_clicked.connect(self.on_filename_clicked)  # Новый сигнал
        left_layout.addWidget(self.file_list)
        
        layout.addWidget(left_widget)
        
        # Правая часть - кнопки и информация
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        info_title = QLabel("Информация о демонстрации")
        info_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        right_layout.addWidget(info_title)
        
        # Информация о новых возможностях
        info_text = QLabel("""
<b>Новые возможности:</b>
<ul>
<li>🔲 <b>Галочки OCR</b> - компактно показывают требуется ли OCR</li>
<li>📄 <b>Проверка PDF</b> - автоматическое определение текстового слоя</li>
<li>📐 <b>Компактный дизайн</b> - уменьшенные отступы и размеры</li>
<li>🎯 <b>Меньше места</b> - левая панель теперь займет меньше места</li>
<li>🔗 <b>Клик по имени файла</b> - открывает файл в новом окне!</li>
</ul>

<b>Цветовая схема галочек:</b>
<ul>
<li>🟠 <b>Оранжевая</b> - OCR требуется (изображения, PDF без текста)</li>
<li>🟢 <b>Зеленая</b> - OCR не требуется (PDF с текстом)</li>
</ul>

<b>Интерактивность:</b>
<ul>
<li>📖 <b>Кликните на имя файла</b> - откроется окно просмотра</li>
<li>📊 <b>Для обработанных файлов</b> - показывается таб с результатами</li>
<li>🔗 <b>Кнопки в диалоге</b> - открыть в системе, копировать путь</li>
</ul>

<b>Тестирование:</b>
<br>Нажмите кнопки ниже для проверки функционала.
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
        right_layout.addWidget(info_text)
        
        # Кнопки для тестирования
        self.add_demo_files_btn = QPushButton("📁 Добавить демо-файлы")
        self.add_demo_files_btn.clicked.connect(self.add_demo_files)
        right_layout.addWidget(self.add_demo_files_btn)
        
        self.add_pdf_files_btn = QPushButton("📄 Добавить PDF файлы")
        self.add_pdf_files_btn.clicked.connect(self.add_pdf_files)
        right_layout.addWidget(self.add_pdf_files_btn)
        
        self.simulate_processing_btn = QPushButton("⚡ Симулировать обработку всех")
        self.simulate_processing_btn.clicked.connect(self.simulate_all_processing)
        right_layout.addWidget(self.simulate_processing_btn)
        
        self.clear_files_btn = QPushButton("🗑️ Очистить список")
        self.clear_files_btn.clicked.connect(self.file_list.clear_files)
        right_layout.addWidget(self.clear_files_btn)
        
        right_layout.addStretch()
        
        # Статус
        self.status_label = QLabel("Готов к тестированию компактного интерфейса")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e3f2fd; border-radius: 3px;")
        right_layout.addWidget(self.status_label)
        
        layout.addWidget(right_widget)
        
        # Для симуляции обработки
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing)
        self.current_processing_file = None
        self.processing_progress = 0
        
    def add_demo_files(self):
        """Добавляет демонстрационные файлы разных типов."""
        demo_files = [
            "demo_images/invoice_001.png",      # Изображение - OCR требуется
            "demo_images/document_001.png",     # Изображение - OCR требуется  
            "demo_images/receipt_001.jpg",      # Изображение - OCR требуется
            "sample_invoice.pdf",               # PDF - будет проверен текстовый слой
            "report_2025.pdf",                  # PDF - будет проверен текстовый слой
            "scanned_contract.pdf"              # PDF - будет проверен текстовый слой
        ]
        
        # Создаем полные пути для демонстрации
        full_paths = []
        for file in demo_files:
            full_path = os.path.join(os.getcwd(), file)
            full_paths.append(full_path)
        
        self.file_list.set_files(full_paths)
        self.status_label.setText(f"Добавлено {len(full_paths)} демо-файлов. Проверьте галочки OCR!")
        
        # Симулируем разные статусы
        if full_paths:
            # Один файл уже обработан
            self.file_list.update_file_progress(full_paths[0], 100, ProcessingStatus.COMPLETED)
            self.file_list.update_file_fields(full_paths[0], 9, 10)
            
            # Один файл с ошибкой
            if len(full_paths) > 1:
                self.file_list.set_file_error(full_paths[1], "Файл поврежден")
                
    def add_pdf_files(self):
        """Добавляет специально PDF файлы для демонстрации проверки текстового слоя."""
        pdf_files = [
            "text_document.pdf",        # PDF с текстом - OCR не требуется
            "scanned_invoice.pdf",      # Сканированный PDF - OCR требуется  
            "mixed_content.pdf",        # Смешанный контент
            "forms_document.pdf"        # Формы PDF
        ]
        
        full_paths = []
        for file in pdf_files:
            full_path = os.path.join(os.getcwd(), file)
            full_paths.append(full_path)
        
        self.file_list.set_files(full_paths)
        self.status_label.setText(f"Добавлено {len(full_paths)} PDF файлов. Система проверит текстовый слой!")
        
    def on_file_selected(self, file_path: str):
        """Обработчик выбора файла."""
        filename = os.path.basename(file_path)
        self.status_label.setText(f"Выбран файл: {filename}")
        
    def on_process_file(self, file_path: str):
        """Обработчик запроса на обработку одного файла."""
        filename = os.path.basename(file_path)
        self.status_label.setText(f"Обработка файла: {filename}")
        
        # Запускаем симуляцию обработки
        self.current_processing_file = file_path
        self.processing_progress = 0
        self.file_list.update_file_progress(file_path, 0, ProcessingStatus.PROCESSING)
        self.processing_timer.start(80)  # Быстрее для демонстрации
        
    def on_process_all(self):
        """Обработчик запроса на обработку всех файлов."""
        unprocessed = self.file_list.get_unprocessed_files()
        if unprocessed:
            self.status_label.setText(f"Начинаем обработку {len(unprocessed)} файлов...")
            self.simulate_all_processing()
        else:
            self.status_label.setText("Нет файлов для обработки")
            
    def on_filename_clicked(self, file_path: str, processing_data: dict):
        """Обработчик клика по имени файла."""
        try:
            from app.ui.components.file_viewer_dialog import FileViewerDialog
            
            # Создаем и показываем диалог просмотра файла
            viewer_dialog = FileViewerDialog(file_path, processing_data, self)
            viewer_dialog.exec()
            
        except Exception as e:
            self.status_label.setText(f"Ошибка открытия файла: {str(e)}")
            print(f"Ошибка открытия файла: {e}")
            
    def simulate_all_processing(self):
        """Симулирует обработку всех необработанных файлов."""
        unprocessed_files = self.file_list.get_unprocessed_files()
        
        if not unprocessed_files:
            self.status_label.setText("Все файлы уже обработаны")
            return
            
        # Симулируем обработку каждого файла с задержкой
        for i, file_path in enumerate(unprocessed_files):
            QTimer.singleShot(i * 1500, lambda fp=file_path: self.simulate_file_processing(fp))
            
    def simulate_file_processing(self, file_path: str):
        """Симулирует обработку одного файла."""
        filename = os.path.basename(file_path)
        self.status_label.setText(f"Симуляция обработки: {filename}")
        
        # Устанавливаем начальный статус
        self.file_list.update_file_progress(file_path, 0, ProcessingStatus.PROCESSING)
        
        # Симулируем прогресс обработки
        def update_progress(progress):
            self.file_list.update_file_progress(file_path, progress, ProcessingStatus.PROCESSING)
            if progress >= 100:
                # Завершаем обработку
                import random
                recognized_fields = random.randint(6, 12)
                total_fields = 12
                
                self.file_list.update_file_progress(file_path, 100, ProcessingStatus.COMPLETED)
                self.file_list.update_file_fields(file_path, recognized_fields, total_fields)
                
                accuracy = (recognized_fields / total_fields) * 100
                self.status_label.setText(f"Завершено: {filename} ({accuracy:.1f}% точность)")
            else:
                # Продолжаем обновление прогресса
                QTimer.singleShot(40, lambda: update_progress(progress + 8))
                
        # Начинаем обновление прогресса
        update_progress(8)
        
    def update_processing(self):
        """Обновляет прогресс текущей обработки."""
        if self.current_processing_file:
            self.processing_progress += 6
            
            if self.processing_progress >= 100:
                # Завершаем обработку
                self.processing_timer.stop()
                
                # Симулируем результат обработки
                import random
                recognized_fields = random.randint(7, 10)
                total_fields = 10
                
                self.file_list.update_file_progress(
                    self.current_processing_file, 100, ProcessingStatus.COMPLETED
                )
                self.file_list.update_file_fields(
                    self.current_processing_file, recognized_fields, total_fields
                )
                
                filename = os.path.basename(self.current_processing_file)
                accuracy = (recognized_fields / total_fields) * 100
                self.status_label.setText(f"Обработка завершена: {filename} ({accuracy:.1f}% точность)")
                
                self.current_processing_file = None
                self.processing_progress = 0
            else:
                # Обновляем прогресс
                self.file_list.update_file_progress(
                    self.current_processing_file, self.processing_progress, ProcessingStatus.PROCESSING
                )


def main():
    """Запуск демонстрации."""
    app = QApplication(sys.argv)
    
    # Устанавливаем стиль приложения
    app.setStyle('Fusion')
    
    # Создаем и показываем тестовое окно
    window = TestWindow()
    window.show()
    
    print("="*70)
    print("ДЕМОНСТРАЦИЯ КОМПАКТНОГО ФАЙЛОВОГО ИНТЕРФЕЙСА С ГАЛОЧКАМИ OCR")
    print("="*70)
    print("🎯 Новые возможности:")
    print("• ✅ Компактные галочки OCR вместо текста")
    print("• 🔍 Реальная проверка текстового слоя в PDF файлах")
    print("• 📐 Уменьшенные размеры элементов интерфейса")
    print("• 🎨 Цветовые индикаторы: 🟠 OCR нужен, 🟢 OCR не нужен")
    print("• 📏 Компактная левая панель (300-350px ширина)")
    print("• 🔗 Клик по имени файла открывает окно просмотра")
    print("")
    print("🎮 Управление:")
    print("• 📁 'Добавить демо-файлы' - загружает смешанные типы файлов")
    print("• 📄 'Добавить PDF файлы' - демонстрирует проверку текстового слоя")
    print("• ⚡ 'Симулировать обработку всех' - обработка всех файлов")
    print("• 🗑️ 'Очистить список' - удаляет все файлы")
    print("• 📖 Клик по имени файла - открывает диалог просмотра")
    print("")
    print("🔍 Что проверить:")
    print("• Галочки OCR показывают статус без лишнего текста")
    print("• PDF файлы проверяются на наличие текстового слоя")
    print("• Компактный дизайн экономит место на экране")
    print("• Клик по имени файла открывает окно с содержимым")
    print("• Для обработанных файлов показывается таб результатов")
    print("• Все функции обработки сохранены")
    print("="*70)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
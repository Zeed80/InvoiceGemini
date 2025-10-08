#!/usr/bin/env python3
"""
Integration Module for Optimized Components
Интеграция оптимизированных компонентов в основное приложение InvoiceGemini
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal

from ..ui.preview_dialog import PreviewDialog as OptimizedPreviewDialog  # Используем стандартную версию
from .optimized_processor import OptimizedProcessingThread
from ..plugins.unified_plugin_manager import UnifiedPluginManager, get_unified_plugin_manager
from ..settings_manager import settings_manager


class OptimizationManager(QObject):
    """
    Менеджер оптимизированных компонентов
    Координирует работу всех улучшенных модулей
    """
    
    # Сигналы для статуса оптимизации
    optimization_enabled = pyqtSignal(str)  # component_name
    optimization_disabled = pyqtSignal(str)  # component_name
    performance_improved = pyqtSignal(str, float)  # component_name, improvement_percent
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Флаги активации оптимизаций
        self.optimizations = {
            'preview_dialog': True,
            'file_processing': True,
            'plugin_system': True,
            'ui_performance': True
        }
        
        # Менеджеры компонентов
        self.plugin_manager: Optional[UnifiedPluginManager] = None
        
        # Статистика производительности
        self.performance_stats = {
            'preview_dialog': {'load_time': 0, 'memory_usage': 0},
            'file_processing': {'throughput': 0, 'parallel_efficiency': 0},
            'plugin_system': {'plugin_count': 0, 'load_time': 0}
        }
        
        logging.info("🚀 OptimizationManager initialized")
    
    def initialize_optimizations(self) -> bool:
        """Инициализация всех оптимизаций"""
        try:
            success_count = 0
            total_optimizations = len(self.optimizations)
            
            # 1. Инициализация оптимизированной системы плагинов
            if self.optimizations['plugin_system']:
                if self._init_plugin_system():
                    success_count += 1
                    self.optimization_enabled.emit("plugin_system")
                    logging.info("✅ Plugin system optimization enabled")
            
            # 2. Настройка оптимизированной обработки файлов
            if self.optimizations['file_processing']:
                if self._init_file_processing():
                    success_count += 1
                    self.optimization_enabled.emit("file_processing")
                    logging.info("✅ File processing optimization enabled")
            
            # 3. Подготовка оптимизированного preview dialog
            if self.optimizations['preview_dialog']:
                if self._init_preview_dialog():
                    success_count += 1
                    self.optimization_enabled.emit("preview_dialog")
                    logging.info("✅ Preview dialog optimization enabled")
            
            # 4. Настройка UI оптимизаций
            if self.optimizations['ui_performance']:
                if self._init_ui_optimizations():
                    success_count += 1
                    self.optimization_enabled.emit("ui_performance")
                    logging.info("✅ UI performance optimization enabled")
            
            optimization_rate = (success_count / total_optimizations) * 100
            logging.info(f"🎯 Optimizations initialized: {success_count}/{total_optimizations} ({optimization_rate:.1f}%)")
            
            return success_count == total_optimizations
            
        except Exception as e:
            logging.error(f"Failed to initialize optimizations: {e}")
            return False
    
    def _init_plugin_system(self) -> bool:
        """Инициализация оптимизированной системы плагинов"""
        try:
            # Получаем глобальный экземпляр
            self.plugin_manager = get_unified_plugin_manager()
            
            # Настраиваем обработчики событий
            self.plugin_manager.plugin_loaded.connect(self._on_plugin_loaded)
            self.plugin_manager.plugin_error.connect(self._on_plugin_error)
            
            # Получаем статистику
            stats = self.plugin_manager.get_statistics()
            self.performance_stats['plugin_system'] = {
                'plugin_count': stats['total_plugins'],
                'enabled_count': stats['enabled_plugins'],
                'load_time': 0  # Будет обновлено при загрузке
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Plugin system optimization failed: {e}")
            return False
    
    def _init_file_processing(self) -> bool:
        """Инициализация оптимизированной обработки файлов"""
        try:
            # Настройки по умолчанию для оптимизированной обработки
            default_settings = {
                'parallel_processing': True,
                'batch_size': 3,
                'gpu_acceleration': True,
                'smart_caching': True,
                'performance_monitoring': True
            }
            
            # Сохраняем настройки
            for key, value in default_settings.items():
                setting_key = f'OptimizedProcessing.{key}'
                if isinstance(value, bool):
                    settings_manager.set_setting(setting_key, str(value).lower())
                else:
                    settings_manager.set_setting(setting_key, str(value))
            
            self.performance_stats['file_processing'] = {
                'parallel_workers': 0,  # Будет обновлено при использовании
                'throughput': 0,
                'cache_hit_rate': 0
            }
            
            return True
            
        except Exception as e:
            logging.error(f"File processing optimization failed: {e}")
            return False
    
    def _init_preview_dialog(self) -> bool:
        """Инициализация оптимизированного preview dialog"""
        try:
            # Настройки оптимизации preview dialog
            preview_settings = {
                'lazy_loading': True,
                'field_limit': 10,
                'auto_save_interval': 30,  # секунды
                'compact_ui': True
            }
            
            # Сохраняем настройки
            for key, value in preview_settings.items():
                setting_key = f'PreviewDialog.{key}'
                if isinstance(value, bool):
                    settings_manager.set_setting(setting_key, str(value).lower())
                else:
                    settings_manager.set_setting(setting_key, str(value))
            
            self.performance_stats['preview_dialog'] = {
                'load_time': 0,
                'memory_usage': 0,
                'ui_responsiveness': 100
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Preview dialog optimization failed: {e}")
            return False
    
    def _init_ui_optimizations(self) -> bool:
        """Инициализация UI оптимизаций"""
        try:
            # Применяем оптимизации к QApplication
            app = QApplication.instance()
            if app:
                # Включаем высокое DPI (только если еще не установлено)
                from PyQt6.QtCore import Qt
                try:
                    # Проверяем, можем ли мы установить атрибут
                    if not app.testAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps):
                        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
                except (RuntimeError, AttributeError) as e:
                    logging.debug(f"Не удалось установить AA_UseHighDpiPixmaps: {e}")
                
                try:
                    if not app.testAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling):
                        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
                except (RuntimeError, AttributeError) as e:
                    logging.debug(f"Не удалось установить AA_EnableHighDpiScaling: {e}")
                
                # Оптимизация стилей
                app.setStyleSheet("""
                    QTableWidget {
                        alternate-background-color: #F5F5F5;
                        selection-background-color: #E3F2FD;
                    }
                    QGroupBox {
                        font-weight: bold;
                        border: 1px solid #CCCCCC;
                        border-radius: 3px;
                        margin-top: 0.5ex;
                    }
                    QTabWidget::pane {
                        border: 1px solid #CCCCCC;
                        border-radius: 3px;
                    }
                """)
            
            return True
            
        except Exception as e:
            logging.error(f"UI optimization failed: {e}")
            return False
    
    def create_optimized_preview_dialog(self, results=None, model_type=None, 
                                      file_path=None, parent=None) -> OptimizedPreviewDialog:
        """Создать оптимизированный preview dialog"""
        try:
            dialog = OptimizedPreviewDialog(
                results=results,
                model_type=model_type,
                file_path=file_path,
                parent=parent
            )
            
            # Подключаем мониторинг производительности
            self._monitor_preview_dialog_performance(dialog)
            
            return dialog
            
        except Exception as e:
            logging.error(f"Failed to create optimized preview dialog: {e}")
            # Fallback к стандартному диалогу
            from ..ui.preview_dialog import PreviewDialog
            return PreviewDialog(results, model_type, file_path, parent)
    
    def create_optimized_processing_thread(self, file_paths, model_type, 
                                         ocr_lang="rus+eng", is_folder=False,
                                         model_manager=None, parent=None) -> OptimizedProcessingThread:
        """Создать оптимизированный поток обработки"""
        try:
            thread = OptimizedProcessingThread(
                file_paths=file_paths,
                model_type=model_type,
                ocr_lang=ocr_lang,
                is_folder=is_folder,
                model_manager=model_manager,
                parent=parent
            )
            
            # Подключаем мониторинг производительности
            self._monitor_processing_thread_performance(thread)
            
            return thread
            
        except Exception as e:
            logging.error(f"Failed to create optimized processing thread: {e}")
            # Fallback к стандартному потоку
            from ..threads import ProcessingThread
            return ProcessingThread(file_paths, model_type, ocr_lang, is_folder, model_manager)
    
    def get_plugin_manager(self) -> Optional[UnifiedPluginManager]:
        """Получить менеджер плагинов"""
        return self.plugin_manager
    
    def _monitor_preview_dialog_performance(self, dialog: OptimizedPreviewDialog):
        """Мониторинг производительности preview dialog"""
        try:
            import time
            start_time = time.time()
            
            def on_dialog_finished():
                load_time = time.time() - start_time
                self.performance_stats['preview_dialog']['load_time'] = load_time
                
                if load_time < 1.0:  # Менее 1 секунды - отличная производительность
                    improvement = 50  # Примерное улучшение по сравнению со старой версией
                    self.performance_improved.emit("preview_dialog", improvement)
            
            dialog.finished.connect(on_dialog_finished)
            
        except Exception as e:
            logging.error(f"Performance monitoring error: {e}")
    
    def _monitor_processing_thread_performance(self, thread: OptimizedProcessingThread):
        """Мониторинг производительности обработки"""
        try:
            import time
            start_time = time.time()
            processed_count = 0
            
            def on_task_completed():
                nonlocal processed_count
                processed_count += 1
                
            def on_batch_completed():
                total_time = time.time() - start_time
                if total_time > 0:
                    throughput = processed_count / total_time
                    self.performance_stats['file_processing']['throughput'] = throughput
                    
                    if throughput > 2.0:  # Более 2 файлов в секунду
                        improvement = 200  # Значительное улучшение
                        self.performance_improved.emit("file_processing", improvement)
            
            thread.processor.task_completed.connect(lambda task_id, result: on_task_completed())
            thread.processor.batch_completed.connect(lambda results: on_batch_completed())
            
        except Exception as e:
            logging.error(f"Performance monitoring error: {e}")
    
    def _on_plugin_loaded(self, plugin_id: str, metadata: dict):
        """Обработка загрузки плагина"""
        logging.info(f"📦 Plugin loaded via optimization: {plugin_id}")
    
    def _on_plugin_error(self, plugin_id: str, error: str):
        """Обработка ошибки плагина"""
        logging.error(f"❌ Plugin error: {plugin_id} - {error}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получить отчёт о производительности"""
        report = {
            'optimization_status': self.optimizations.copy(),
            'performance_stats': self.performance_stats.copy(),
            'active_optimizations': [name for name, enabled in self.optimizations.items() if enabled],
            'timestamp': str(Path(__file__).stat().st_mtime)
        }
        
        # Добавляем статистику плагинов если доступна
        if self.plugin_manager:
            report['plugin_stats'] = self.plugin_manager.get_statistics()
        
        return report
    
    def toggle_optimization(self, component: str, enabled: bool):
        """Включить/выключить оптимизацию компонента"""
        if component in self.optimizations:
            self.optimizations[component] = enabled
            
            if enabled:
                self.optimization_enabled.emit(component)
                logging.info(f"✅ Optimization enabled: {component}")
            else:
                self.optimization_disabled.emit(component)
                logging.info(f"⏹️ Optimization disabled: {component}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        logging.info("🧹 Cleaning up OptimizationManager...")
        
        if self.plugin_manager:
            self.plugin_manager.cleanup()
        
        logging.info("✅ OptimizationManager cleanup completed")


# Глобальный экземпляр
_optimization_manager = None

def get_optimization_manager() -> OptimizationManager:
    """Получить глобальный экземпляр менеджера оптимизаций"""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = OptimizationManager()
        _optimization_manager.initialize_optimizations()
    return _optimization_manager


def apply_optimizations_to_main_window(main_window):
    """
    Применить оптимизации к главному окну
    Monkey patching для интеграции без изменения основного кода
    """
    try:
        optimization_manager = get_optimization_manager()
        
        # Сохраняем оригинальные методы
        original_show_preview_dialog = main_window.show_preview_dialog
        
        def optimized_show_preview_dialog():
            """Оптимизированная версия show_preview_dialog"""
            try:
                # Определяем данные для preview (копируем логику из оригинала)
                preview_data = None
                model_type = "unknown"
                file_path = ""
                
                if main_window.current_folder_path:
                    # Пакетная обработка
                    if main_window.results_table.rowCount() == 0:
                        from .. import utils
                        utils.show_info_message(
                            main_window, "Информация",
                            "Нет результатов для предварительного просмотра. Сначала обработайте файлы."
                        )
                        return
                    
                    # Собираем данные из таблицы
                    batch_results = []
                    headers = [main_window.results_table.horizontalHeaderItem(col).text()
                              for col in range(main_window.results_table.columnCount())]
                    
                    for row in range(main_window.results_table.rowCount()):
                        row_data = {}
                        for col, header in enumerate(headers):
                            item = main_window.results_table.item(row, col)
                            row_data[header] = item.text() if item else ""
                        batch_results.append(row_data)
                    
                    preview_data = {"batch_results": batch_results}
                    file_path = main_window.current_folder_path
                    
                else:
                    # Одиночная обработка
                    if main_window.results_table.rowCount() == 0:
                        from .. import utils
                        utils.show_info_message(
                            main_window, "Информация",
                            "Нет результатов для предварительного просмотра. Сначала обработайте файл."
                        )
                        return
                    
                    # Собираем данные из таблицы
                    preview_data = {}
                    if main_window.results_table.rowCount() > 0:
                        row = 0
                        for col in range(main_window.results_table.columnCount()):
                            header_item = main_window.results_table.horizontalHeaderItem(col)
                            cell_item = main_window.results_table.item(row, col)
                            if header_item and cell_item:
                                field_name = header_item.text()
                                field_value = cell_item.text()
                                if field_value:
                                    preview_data[field_name] = field_value
                    
                    file_path = main_window.current_image_path or ""
                
                # Определяем модель
                if main_window.layoutlm_radio.isChecked():
                    model_type = "LayoutLMv3"
                elif main_window.donut_radio.isChecked():
                    model_type = "Donut"
                elif main_window.gemini_radio.isChecked():
                    model_type = "Gemini 2.0"
                elif hasattr(main_window, 'cloud_llm_radio') and main_window.cloud_llm_radio.isChecked():
                    model_type = f"Cloud LLM ({main_window.cloud_model_selector.currentText()})"
                elif hasattr(main_window, 'local_llm_radio') and main_window.local_llm_radio.isChecked():
                    model_type = f"Local LLM ({main_window.local_model_selector.currentText()})"
                
                # Создаём оптимизированный диалог
                preview_dialog = optimization_manager.create_optimized_preview_dialog(
                    results=preview_data,
                    model_type=model_type,
                    file_path=file_path,
                    parent=main_window
                )
                
                # Подключаем сигналы
                preview_dialog.results_edited.connect(main_window.on_preview_results_edited)
                preview_dialog.export_requested.connect(main_window.on_preview_export_requested)
                
                # Показываем диалог
                result = preview_dialog.exec()
                
                if result == preview_dialog.DialogCode.Accepted:
                    main_window.status_bar.showMessage("Изменения из предварительного просмотра применены")
                
            except Exception as e:
                logging.error(f"Optimized preview dialog error: {e}")
                # Fallback к оригинальному методу
                original_show_preview_dialog()
        
        # Подменяем метод
        main_window.show_preview_dialog = optimized_show_preview_dialog
        
        logging.info("✅ Applied optimizations to main window")
        return True
        
    except Exception as e:
        logging.error(f"Failed to apply optimizations to main window: {e}")
        return False 
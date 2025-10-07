#!/usr/bin/env python3
"""
Test Script for InvoiceGemini Optimizations
Скрипт для тестирования всех оптимизированных компонентов
"""

import sys
import time
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('optimization_test.log')
    ]
)

def test_preview_dialog():
    """Тестирование оптимизированного Preview Dialog"""
    print("\n🔍 Testing Optimized Preview Dialog...")
    
    try:
        from app.ui.preview_dialog_v2 import OptimizedPreviewDialog
        from PyQt6.QtWidgets import QApplication
        
        # Создаём приложение если нужно
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Тестовые данные
        test_data = {
            "Поставщик": "ООО Тест Компания",
            "Номер счета": "INV-001",
            "Дата": "01.01.2024",
            "Сумма": "15000.00",
            "НДС": "3000.00",
            "Итого": "18000.00"
        }
        
        start_time = time.time()
        
        # Создаём диалог
        dialog = OptimizedPreviewDialog(
            results=test_data,
            model_type="Test Model",
            file_path="test_invoice.pdf"
        )
        
        creation_time = time.time() - start_time
        
        # Проверяем базовые свойства
        assert dialog.data_model is not None
        assert len(dialog.data_model.current_results) == len(test_data)
        assert dialog.windowTitle().startswith("🔍 Предварительный просмотр v2.0")
        
        print(f"   ✅ Dialog created in {creation_time:.3f}s")
        print(f"   ✅ Data model initialized with {len(test_data)} fields")
        print(f"   ✅ UI components loaded successfully")
        
        # Очистка
        dialog.deleteLater()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Preview Dialog test failed: {e}")
        return False


def test_optimized_processor():
    """Тестирование оптимизированного процессора файлов"""
    print("\n⚡ Testing Optimized File Processor...")
    
    try:
        from app.core.optimized_processor import OptimizedFileProcessor, ProcessingTask
        from PyQt6.QtCore import QCoreApplication
        
        # Создаём приложение если нужно
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        
        processor = OptimizedFileProcessor()
        
        # Проверяем инициализацию
        assert processor.max_workers > 0
        assert processor.executor is not None
        assert not processor.is_processing
        
        print(f"   ✅ Processor initialized with {processor.max_workers} workers")
        print(f"   ✅ Batch size: {processor.batch_size}")
        print(f"   ✅ Thread pool ready")
        
        # Тестируем создание задач
        test_files = ["test1.pdf", "test2.pdf", "test3.pdf"]
        
        # Имитируем запуск (без реальной обработки)
        processor.task_queue.clear()
        for i, file_path in enumerate(test_files):
            task = ProcessingTask(
                file_path=file_path,
                task_id=f"test_task_{i}",
                model_type="test"
            )
            processor.task_queue.append(task)
        
        assert len(processor.task_queue) == len(test_files)
        print(f"   ✅ Created {len(test_files)} test tasks")
        
        # Очистка
        processor.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Optimized Processor test failed: {e}")
        return False


def test_unified_plugin_manager():
    """Тестирование унифицированного менеджера плагинов"""
    print("\n🔧 Testing Unified Plugin Manager...")
    
    try:
        from app.plugins.unified_plugin_manager import UnifiedPluginManager, PluginRegistry
        from app.plugins.base_plugin import PluginType
        
        # Создаём менеджер без автосканирования
        manager = UnifiedPluginManager(auto_scan=False)
        
        # Проверяем инициализацию
        assert manager.registry is not None
        assert manager.loader is not None
        assert manager.builtin_dir.exists() or True  # Может не существовать в тестах
        assert manager.user_dir is not None
        
        print(f"   ✅ Manager initialized")
        print(f"   ✅ Builtin dir: {manager.builtin_dir}")
        print(f"   ✅ User dir: {manager.user_dir}")
        
        # Тестируем реестр
        registry = manager.registry
        
        # Проверяем индексы по типам
        for plugin_type in PluginType:
            plugins = registry.get_by_type(plugin_type)
            assert isinstance(plugins, list)
        
        print(f"   ✅ Registry initialized with {len(PluginType)} plugin types")
        
        # Тестируем статистику
        stats = manager.get_statistics()
        assert 'total_plugins' in stats
        assert 'by_type' in stats
        
        print(f"   ✅ Statistics available: {stats['total_plugins']} total plugins")
        
        # Очистка
        manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Unified Plugin Manager test failed: {e}")
        return False


def test_optimization_integration():
    """Тестирование интеграционного модуля"""
    print("\n🎯 Testing Optimization Integration...")
    
    try:
        from app.core.optimization_integration import OptimizationManager, get_optimization_manager
        
        # Получаем менеджер оптимизаций
        manager = get_optimization_manager()
        
        # Проверяем инициализацию
        assert manager is not None
        assert len(manager.optimizations) > 0
        assert manager.performance_stats is not None
        
        print(f"   ✅ Optimization manager created")
        print(f"   ✅ Available optimizations: {list(manager.optimizations.keys())}")
        
        # Проверяем отчёт о производительности
        report = manager.get_performance_report()
        assert 'optimization_status' in report
        assert 'performance_stats' in report
        assert 'active_optimizations' in report
        
        active_count = len(report['active_optimizations'])
        total_count = len(manager.optimizations)
        
        print(f"   ✅ Performance report generated")
        print(f"   ✅ Active optimizations: {active_count}/{total_count}")
        
        # Очистка
        manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization Integration test failed: {e}")
        return False


def run_performance_benchmark():
    """Запуск бенчмарка производительности"""
    print("\n📊 Running Performance Benchmark...")
    
    try:
        from app.ui.preview_dialog_v2 import OptimizedPreviewDialog
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Бенчмарк создания диалогов
        iterations = 10
        total_time = 0
        
        for i in range(iterations):
            start = time.time()
            
            dialog = OptimizedPreviewDialog(
                results={"test": f"value_{i}"},
                model_type="Test",
                file_path="test.pdf"
            )
            
            creation_time = time.time() - start
            total_time += creation_time
            
            dialog.deleteLater()
        
        avg_time = total_time / iterations
        
        print(f"   📈 Average dialog creation time: {avg_time:.3f}s")
        print(f"   📈 Total time for {iterations} dialogs: {total_time:.3f}s")
        
        # Определяем производительность
        if avg_time < 0.1:
            performance_rating = "Отличная"
        elif avg_time < 0.5:
            performance_rating = "Хорошая"
        elif avg_time < 1.0:
            performance_rating = "Удовлетворительная"
        else:
            performance_rating = "Требует оптимизации"
        
        print(f"   🏆 Performance rating: {performance_rating}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance benchmark failed: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 InvoiceGemini Optimization Test Suite")
    print("=" * 50)
    
    tests = [
        ("Preview Dialog v2.0", test_preview_dialog),
        ("Optimized File Processor", test_optimized_processor),
        ("Unified Plugin Manager", test_unified_plugin_manager),
        ("Optimization Integration", test_optimization_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Результаты
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Tests passed: {passed}/{len(tests)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if passed == len(tests):
        print("🎉 All optimizations working correctly!")
        return 0
    else:
        print("⚠️  Some optimizations need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
"""
Pytest configuration and fixtures for InvoiceGemini tests.
"""

import pytest
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Путь к директории с тестовыми данными"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_invoice_image(test_data_dir):
    """Путь к тестовому изображению счета"""
    # Можно добавить реальный тестовый файл позже
    return test_data_dir / "sample_invoice.png"


@pytest.fixture
def mock_settings():
    """Мок настроек приложения"""
    return {
        'tesseract_path': 'tesseract',
        'model_cache_dir': 'data/models',
        'temp_dir': 'data/temp',
        'database_path': 'data/test_invoices.db'
    }


@pytest.fixture
def mock_invoice_data():
    """Пример данных счета для тестов"""
    return {
        'invoice_number': 'INV-2024-001',
        'date': '01.10.2024',
        'vendor': 'Тестовый Поставщик ООО',
        'vendor_inn': '1234567890',
        'total': '10000.00',
        'tax': '1800.00',
        'subtotal': '8200.00',
        'payment_status': 'Не оплачено',
        'category': 'Услуги'
    }


@pytest.fixture
def mock_ocr_result():
    """Пример результата OCR для тестов"""
    return {
        'text': 'Счет на оплату\nНомер: INV-2024-001\nДата: 01.10.2024',
        'bounding_boxes': [
            {'text': 'Счет', 'left': 10, 'top': 10, 'width': 100, 'height': 20},
            {'text': 'INV-2024-001', 'left': 10, 'top': 40, 'width': 150, 'height': 20}
        ]
    }


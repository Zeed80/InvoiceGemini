"""
Unit tests for BackupManager
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from app.core.backup_manager import BackupManager


class TestBackupManager:
    """Тесты для BackupManager"""
    
    @pytest.fixture
    def temp_app_dir(self):
        """Создаем временную директорию для тестов"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Очистка после теста
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def backup_manager(self, temp_app_dir):
        """Создаем экземпляр BackupManager для тестов"""
        return BackupManager(app_data_dir=str(temp_app_dir))
    
    def test_initialization(self, backup_manager, temp_app_dir):
        """Тест инициализации BackupManager"""
        assert backup_manager.app_data_dir == temp_app_dir
        assert backup_manager.backup_dir.exists()
        assert backup_manager.backup_dir == temp_app_dir / "backups"
    
    def test_backup_settings_method_exists(self, backup_manager):
        """Тест наличия метода backup_settings"""
        assert hasattr(backup_manager, 'backup_settings')
        assert callable(backup_manager.backup_settings)
    
    def test_backup_settings_returns_bool(self, backup_manager):
        """Тест что backup_settings возвращает boolean"""
        result = backup_manager.backup_settings()
        assert isinstance(result, bool)
    
    def test_create_backup_without_models(self, backup_manager, temp_app_dir):
        """Тест создания бекапа без моделей"""
        # Создаем тестовый файл настроек
        settings_file = temp_app_dir / "settings.ini"
        settings_file.write_text("[DEFAULT]\ntest_key=test_value")
        
        success, result = backup_manager.create_backup(
            backup_name="test_backup",
            include_models=False,
            include_cache=False
        )
        
        assert isinstance(success, bool)
        assert isinstance(result, str)
    
    def test_get_app_version(self, backup_manager, temp_app_dir):
        """Тест получения версии приложения"""
        version = backup_manager._get_app_version()
        assert isinstance(version, str)
        assert version == "1.0.0"  # По умолчанию
        
        # Создаем файл версии
        version_file = temp_app_dir / "version.txt"
        version_file.write_text("2.0.0")
        
        version = backup_manager._get_app_version()
        assert version == "2.0.0"


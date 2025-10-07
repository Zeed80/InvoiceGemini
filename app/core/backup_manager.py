"""
Менеджер резервного копирования настроек и данных.
"""

import os
import json
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import configparser

logger = logging.getLogger(__name__)


class BackupManager:
    """Менеджер для создания и восстановления резервных копий"""
    
    def __init__(self, app_data_dir: str = None):
        """
        Инициализация менеджера резервного копирования.
        
        Args:
            app_data_dir: Директория с данными приложения
        """
        self.app_data_dir = Path(app_data_dir or ".")
        self.backup_dir = self.app_data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Определяем важные директории и файлы
        self.important_paths = {
            'settings': 'settings.ini',
            'fields_config': 'data/table_fields.json',
            'models_cache': 'data/models/',
            'templates': 'data/templates/',
            'plugins': 'plugins/user/',
            'prompts': 'data/prompts/',
            'secrets': 'data/secrets/',
            'cache': 'data/cache/'
        }
        
    def create_backup(self, backup_name: Optional[str] = None, 
                     include_models: bool = False,
                     include_cache: bool = False) -> Tuple[bool, str]:
        """
        Создание резервной копии.
        
        Args:
            backup_name: Имя резервной копии (если не указано, генерируется автоматически)
            include_models: Включать ли модели ML (могут занимать много места)
            include_cache: Включать ли кэш
            
        Returns:
            Tuple[bool, str]: (Успех операции, Путь к файлу или сообщение об ошибке)
        """
        try:
            # Генерируем имя файла
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
                
            backup_file = self.backup_dir / f"{backup_name}.zip"
            
            # Создаем метаданные резервной копии
            metadata = {
                'created_at': datetime.now().isoformat(),
                'version': self._get_app_version(),
                'include_models': include_models,
                'include_cache': include_cache,
                'files_count': 0,
                'total_size': 0
            }
            
            # Создаем ZIP архив
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                files_added = 0
                total_size = 0
                
                # Добавляем файлы и директории
                for key, path in self.important_paths.items():
                    # Пропускаем модели и кэш если не требуется
                    if key == 'models_cache' and not include_models:
                        continue
                    if key == 'cache' and not include_cache:
                        continue
                        
                    full_path = self.app_data_dir / path
                    
                    if full_path.exists():
                        if full_path.is_file():
                            # Добавляем файл
                            arcname = f"backup/{path}"
                            zf.write(full_path, arcname)
                            files_added += 1
                            total_size += full_path.stat().st_size
                            logger.info(f"Добавлен файл: {path}")
                            
                        elif full_path.is_dir():
                            # Добавляем директорию рекурсивно
                            for file_path in full_path.rglob('*'):
                                if file_path.is_file():
                                    # Вычисляем относительный путь
                                    rel_path = file_path.relative_to(self.app_data_dir)
                                    arcname = f"backup/{rel_path}"
                                    zf.write(file_path, arcname)
                                    files_added += 1
                                    total_size += file_path.stat().st_size
                                    
                            logger.info(f"Добавлена директория: {path}")
                            
                # Обновляем метаданные
                metadata['files_count'] = files_added
                metadata['total_size'] = total_size
                
                # Сохраняем метаданные
                metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                zf.writestr('backup_metadata.json', metadata_json)
                
            # Проверяем размер созданного архива
            backup_size = backup_file.stat().st_size
            size_mb = backup_size / (1024 * 1024)
            
            logger.info(f"Резервная копия создана: {backup_file} ({size_mb:.2f} MB)")
            return True, str(backup_file)
            
        except Exception as e:
            error_msg = f"Ошибка создания резервной копии: {e}"
            logger.error(error_msg)
            return False, error_msg
            
    def restore_backup(self, backup_path: str, restore_models: bool = True,
                      restore_cache: bool = True) -> Tuple[bool, str]:
        """
        Восстановление из резервной копии.
        
        Args:
            backup_path: Путь к файлу резервной копии
            restore_models: Восстанавливать ли модели ML
            restore_cache: Восстанавливать ли кэш
            
        Returns:
            Tuple[bool, str]: (Успех операции, Сообщение)
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return False, f"Файл резервной копии не найден: {backup_path}"
                
            # Создаем временную директорию для распаковки
            temp_dir = self.backup_dir / f"temp_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Распаковываем архив
                with zipfile.ZipFile(backup_file, 'r') as zf:
                    # Проверяем наличие метаданных
                    if 'backup_metadata.json' in zf.namelist():
                        metadata_content = zf.read('backup_metadata.json')
                        metadata = json.loads(metadata_content)
                        logger.info(f"Метаданные резервной копии: {metadata}")
                        
                    # Распаковываем все файлы
                    zf.extractall(temp_dir)
                    
                # Перемещаем файлы на место
                backup_content_dir = temp_dir / 'backup'
                
                if not backup_content_dir.exists():
                    return False, "Неверная структура резервной копии"
                    
                # Создаем резервную копию текущих настроек перед восстановлением
                auto_backup_name = f"auto_backup_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_backup(auto_backup_name, include_models=False, include_cache=False)
                
                # Восстанавливаем файлы
                restored_count = 0
                
                for item in backup_content_dir.rglob('*'):
                    if item.is_file():
                        # Определяем целевой путь
                        rel_path = item.relative_to(backup_content_dir)
                        target_path = self.app_data_dir / rel_path
                        
                        # Пропускаем модели и кэш если не требуется
                        if 'models/' in str(rel_path) and not restore_models:
                            continue
                        if 'cache/' in str(rel_path) and not restore_cache:
                            continue
                            
                        # Создаем директории если нужно
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Копируем файл
                        shutil.copy2(item, target_path)
                        restored_count += 1
                        
                logger.info(f"Восстановлено файлов: {restored_count}")
                
                # Очищаем временную директорию
                shutil.rmtree(temp_dir)
                
                return True, f"Успешно восстановлено {restored_count} файлов"
                
            except Exception as e:
                # Очищаем временную директорию в случае ошибки
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise e
                
        except Exception as e:
            error_msg = f"Ошибка восстановления резервной копии: {e}"
            logger.error(error_msg)
            return False, error_msg
            
    def list_backups(self) -> List[Dict]:
        """
        Получение списка доступных резервных копий.
        
        Returns:
            List[Dict]: Список информации о резервных копиях
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                # Базовая информация
                backup_info = {
                    'name': backup_file.stem,
                    'file_path': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'size_mb': backup_file.stat().st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(backup_file.stat().st_mtime),
                    'metadata': None
                }
                
                # Пытаемся прочитать метаданные
                try:
                    with zipfile.ZipFile(backup_file, 'r') as zf:
                        if 'backup_metadata.json' in zf.namelist():
                            metadata_content = zf.read('backup_metadata.json')
                            backup_info['metadata'] = json.loads(metadata_content)
                except:
                    pass
                    
                backups.append(backup_info)
                
            except Exception as e:
                logger.warning(f"Не удалось прочитать информацию о резервной копии {backup_file}: {e}")
                
        # Сортируем по дате создания (новые первые)
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return backups
        
    def delete_backup(self, backup_path: str) -> Tuple[bool, str]:
        """
        Удаление резервной копии.
        
        Args:
            backup_path: Путь к файлу резервной копии
            
        Returns:
            Tuple[bool, str]: (Успех операции, Сообщение)
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return False, "Файл резервной копии не найден"
                
            # Проверяем, что файл в директории резервных копий
            if backup_file.parent != self.backup_dir:
                return False, "Файл не является резервной копией"
                
            # Удаляем файл
            backup_file.unlink()
            
            logger.info(f"Резервная копия удалена: {backup_file}")
            return True, "Резервная копия успешно удалена"
            
        except Exception as e:
            error_msg = f"Ошибка удаления резервной копии: {e}"
            logger.error(error_msg)
            return False, error_msg
            
    def auto_cleanup(self, max_backups: int = 10, max_age_days: int = 30):
        """
        Автоматическая очистка старых резервных копий.
        
        Args:
            max_backups: Максимальное количество резервных копий
            max_age_days: Максимальный возраст резервных копий в днях
        """
        try:
            backups = self.list_backups()
            
            # Фильтруем автоматические резервные копии
            auto_backups = [b for b in backups if 'auto_backup' in b['name']]
            
            # Удаляем старые резервные копии
            now = datetime.now()
            deleted_count = 0
            
            for backup in auto_backups:
                age_days = (now - backup['created']).days
                
                # Удаляем если слишком старая
                if age_days > max_age_days:
                    success, _ = self.delete_backup(backup['file_path'])
                    if success:
                        deleted_count += 1
                        logger.info(f"Удалена старая резервная копия: {backup['name']} (возраст: {age_days} дней)")
                        
            # Удаляем лишние резервные копии если их слишком много
            if len(auto_backups) > max_backups:
                # Оставляем только последние max_backups
                for backup in auto_backups[max_backups:]:
                    success, _ = self.delete_backup(backup['file_path'])
                    if success:
                        deleted_count += 1
                        logger.info(f"Удалена лишняя резервная копия: {backup['name']}")
                        
            if deleted_count > 0:
                logger.info(f"Автоочистка завершена. Удалено резервных копий: {deleted_count}")
                
        except Exception as e:
            logger.error(f"Ошибка автоочистки резервных копий: {e}")
            
    def export_settings(self, export_path: str) -> Tuple[bool, str]:
        """
        Экспорт только настроек (без данных).
        
        Args:
            export_path: Путь для сохранения файла настроек
            
        Returns:
            Tuple[bool, str]: (Успех операции, Сообщение)
        """
        try:
            settings_file = self.app_data_dir / self.important_paths['settings']
            
            if not settings_file.exists():
                return False, "Файл настроек не найден"
                
            # Копируем файл настроек
            shutil.copy2(settings_file, export_path)
            
            logger.info(f"Настройки экспортированы: {export_path}")
            return True, "Настройки успешно экспортированы"
            
        except Exception as e:
            error_msg = f"Ошибка экспорта настроек: {e}"
            logger.error(error_msg)
            return False, error_msg
            
    def import_settings(self, import_path: str) -> Tuple[bool, str]:
        """
        Импорт настроек.
        
        Args:
            import_path: Путь к файлу настроек для импорта
            
        Returns:
            Tuple[bool, str]: (Успех операции, Сообщение)
        """
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                return False, "Файл настроек не найден"
                
            # Проверяем, что это файл настроек
            try:
                config = configparser.ConfigParser()
                config.read(import_file, encoding='utf-8')
            except:
                return False, "Неверный формат файла настроек"
                
            # Создаем резервную копию текущих настроек
            auto_backup_name = f"auto_backup_before_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.create_backup(auto_backup_name, include_models=False, include_cache=False)
            
            # Копируем новые настройки
            settings_file = self.app_data_dir / self.important_paths['settings']
            shutil.copy2(import_file, settings_file)
            
            logger.info(f"Настройки импортированы из: {import_path}")
            return True, "Настройки успешно импортированы"
            
        except Exception as e:
            error_msg = f"Ошибка импорта настроек: {e}"
            logger.error(error_msg)
            return False, error_msg
            
    def _get_app_version(self) -> str:
        """Получение версии приложения"""
        try:
            # Пытаемся прочитать версию из файла или настроек
            version_file = self.app_data_dir / "version.txt"
            if version_file.exists():
                return version_file.read_text().strip()
        except (IOError, OSError) as e:
            logger.debug(f"Не удалось прочитать файл версии: {e}")
            
        return "1.0.0"  # Версия по умолчанию
    
    def backup_settings(self) -> bool:
        """
        Быстрое создание резервной копии только настроек.
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            success, result = self.create_backup(
                backup_name=f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                include_models=False,
                include_cache=False
            )
            if success:
                logger.info(f"Резервная копия настроек создана: {result}")
            else:
                logger.warning(f"Не удалось создать резервную копию настроек: {result}")
            return success
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии настроек: {e}", exc_info=True)
            return False 